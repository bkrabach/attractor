//! Test utilities for the attractor crate.
//!
//! [`MockCodergenBackend`] provides a configurable backend for testing the
//! pipeline execution engine without any real LLM calls.

use crate::error::EngineError;
use crate::graph::Node;
use crate::handler::codergen::{CodergenBackend, CodergenResult};
use crate::state::context::{Context, Outcome};
use async_trait::async_trait;
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// MockCodergenBackend
// ---------------------------------------------------------------------------

/// A configurable mock backend for testing pipeline execution.
///
/// Responses are queued per node ID and consumed in FIFO order.  When a
/// node's queue is exhausted the `default` outcome is returned.
pub struct MockCodergenBackend {
    /// Per-node response queues: node_id → ordered responses.
    responses: Mutex<HashMap<String, VecDeque<CodergenResult>>>,
    /// Returned when a node has no queued responses.
    default: Outcome,
    /// Call history: list of `(node_id, prompt)` pairs.
    calls: Mutex<Vec<(String, String)>>,
}

impl MockCodergenBackend {
    /// Create a new mock with a default SUCCESS outcome.
    pub fn new() -> Self {
        MockCodergenBackend {
            responses: Mutex::new(HashMap::new()),
            default: Outcome::success(),
            calls: Mutex::new(Vec::new()),
        }
    }

    /// Override the default outcome for nodes with no queued response.
    pub fn with_default(mut self, outcome: Outcome) -> Self {
        self.default = outcome;
        self
    }

    /// Queue a `CodergenResult` for `node_id` (appended to the back).
    pub fn add_response(&self, node_id: &str, result: CodergenResult) {
        self.responses
            .lock()
            .expect("mock backend lock poisoned")
            .entry(node_id.to_string())
            .or_default()
            .push_back(result);
    }

    /// Queue a SUCCESS outcome for `node_id`.
    pub fn add_success(&self, node_id: &str) {
        self.add_response(node_id, CodergenResult::Outcome(Outcome::success()));
    }

    /// Queue a FAIL outcome for `node_id`.
    pub fn add_fail(&self, node_id: &str, reason: &str) {
        self.add_response(
            node_id,
            CodergenResult::Outcome(Outcome::fail(reason.to_string())),
        );
    }

    /// Queue a RETRY outcome for `node_id`.
    pub fn add_retry(&self, node_id: &str, reason: &str) {
        self.add_response(
            node_id,
            CodergenResult::Outcome(Outcome::retry(reason.to_string())),
        );
    }

    /// Return a snapshot of all calls as `(node_id, prompt)` pairs.
    pub fn calls(&self) -> Vec<(String, String)> {
        self.calls
            .lock()
            .expect("mock backend calls lock poisoned")
            .clone()
    }

    /// Return the number of times `node_id` was called.
    pub fn call_count(&self, node_id: &str) -> usize {
        self.calls
            .lock()
            .expect("mock backend calls lock poisoned")
            .iter()
            .filter(|(id, _)| id == node_id)
            .count()
    }
}

impl Default for MockCodergenBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CodergenBackend for MockCodergenBackend {
    async fn run(
        &self,
        node: &Node,
        prompt: &str,
        _context: &Context,
    ) -> Result<CodergenResult, EngineError> {
        // Record the call.
        self.calls
            .lock()
            .expect("mock backend calls lock poisoned")
            .push((node.id.clone(), prompt.to_string()));

        // Dequeue or use default.
        let result = {
            let mut responses = self
                .responses
                .lock()
                .expect("mock backend responses lock poisoned");
            responses.get_mut(&node.id).and_then(|q| q.pop_front())
        };

        Ok(result.unwrap_or_else(|| CodergenResult::Outcome(self.default.clone())))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Node;
    use crate::state::context::StageStatus;

    fn make_node(id: &str) -> Node {
        Node {
            id: id.to_string(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn default_success() {
        let backend = MockCodergenBackend::new();
        let node = make_node("A");
        let ctx = Context::new();
        let result = backend.run(&node, "prompt", &ctx).await.unwrap();
        match result {
            CodergenResult::Outcome(o) => assert_eq!(o.status, StageStatus::Success),
            _ => panic!("expected Outcome"),
        }
    }

    #[tokio::test]
    async fn queued_responses_in_fifo_order() {
        let backend = MockCodergenBackend::new();
        backend.add_fail("A", "first");
        backend.add_success("A");
        let ctx = Context::new();
        let node = make_node("A");

        let r1 = backend.run(&node, "p", &ctx).await.unwrap();
        match r1 {
            CodergenResult::Outcome(o) => assert_eq!(o.status, StageStatus::Fail),
            _ => panic!(),
        }
        let r2 = backend.run(&node, "p", &ctx).await.unwrap();
        match r2 {
            CodergenResult::Outcome(o) => assert_eq!(o.status, StageStatus::Success),
            _ => panic!(),
        }
    }

    #[tokio::test]
    async fn falls_back_to_default_when_queue_empty() {
        let backend = MockCodergenBackend::new().with_default(Outcome::fail("default-fail"));
        let node = make_node("A");
        let ctx = Context::new();
        let r = backend.run(&node, "p", &ctx).await.unwrap();
        match r {
            CodergenResult::Outcome(o) => assert_eq!(o.status, StageStatus::Fail),
            _ => panic!(),
        }
    }

    #[tokio::test]
    async fn calls_recorded() {
        let backend = MockCodergenBackend::new();
        let node_a = make_node("A");
        let node_b = make_node("B");
        let ctx = Context::new();

        backend.run(&node_a, "prompt A", &ctx).await.unwrap();
        backend.run(&node_b, "prompt B", &ctx).await.unwrap();
        backend.run(&node_a, "prompt A2", &ctx).await.unwrap();

        let calls = backend.calls();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0], ("A".to_string(), "prompt A".to_string()));
        assert_eq!(calls[1], ("B".to_string(), "prompt B".to_string()));

        assert_eq!(backend.call_count("A"), 2);
        assert_eq!(backend.call_count("B"), 1);
        assert_eq!(backend.call_count("C"), 0);
    }

    #[tokio::test]
    async fn per_node_queues_independent() {
        let backend = MockCodergenBackend::new();
        backend.add_fail("A", "a-fail");
        backend.add_success("B");

        let ctx = Context::new();
        let na = make_node("A");
        let nb = make_node("B");

        let ra = backend.run(&na, "p", &ctx).await.unwrap();
        let rb = backend.run(&nb, "p", &ctx).await.unwrap();

        match ra {
            CodergenResult::Outcome(o) => assert_eq!(o.status, StageStatus::Fail),
            _ => panic!(),
        }
        match rb {
            CodergenResult::Outcome(o) => assert_eq!(o.status, StageStatus::Success),
            _ => panic!(),
        }
    }
}
