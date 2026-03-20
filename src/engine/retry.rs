//! Retry policy and backoff configuration (NLSpec §3.5–3.6).
//!
//! [`RetryPolicy`] wraps handler execution with exponential-backoff retry,
//! tracking attempt counts, emitting events, and honouring `allow_partial`.

use crate::events::PipelineEvent;
use crate::graph::{Graph, Node};
use crate::handler::Handler;
use crate::state::context::{Context, Outcome, StageStatus};
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;
use tokio::sync::broadcast;

// ---------------------------------------------------------------------------
// BackoffConfig
// ---------------------------------------------------------------------------

/// Delay calculation configuration for retry backoff.
#[derive(Debug, Clone)]
pub struct BackoffConfig {
    /// Delay before the first retry in milliseconds. Default 200.
    pub initial_delay_ms: u64,
    /// Multiply the previous delay by this factor. Default 2.0 (exponential).
    pub backoff_factor: f64,
    /// Hard cap on any single delay in milliseconds. Default 60 000.
    pub max_delay_ms: u64,
    /// Add ±50 % uniform random jitter to prevent thundering herd. Default true.
    pub jitter: bool,
}

impl Default for BackoffConfig {
    fn default() -> Self {
        BackoffConfig {
            initial_delay_ms: 200,
            backoff_factor: 2.0,
            max_delay_ms: 60_000,
            jitter: true,
        }
    }
}

impl BackoffConfig {
    /// Compute the delay for a given `attempt` (1-indexed; first retry = 1).
    ///
    /// ```text
    /// delay = initial_delay_ms * backoff_factor^(attempt-1)
    /// delay = min(delay, max_delay_ms)
    /// if jitter: delay *= random_uniform(0.5, 1.5)
    /// ```
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let raw = self.initial_delay_ms as f64
            * self.backoff_factor.powi(attempt.saturating_sub(1) as i32);
        let capped = raw.min(self.max_delay_ms as f64);
        let final_ms = if self.jitter {
            // Uniform in [0.5, 1.5]
            let jitter: f64 = 0.5 + rand_f64();
            capped * jitter
        } else {
            capped
        };
        Duration::from_millis(final_ms as u64)
    }
}

// Simple platform-agnostic random float in [0.0, 1.0).
fn rand_f64() -> f64 {
    // We use a simple LCG seeded from current time nanos to avoid pulling in
    // the full `rand` crate for a single float.  For test determinism, callers
    // can disable jitter via `BackoffConfig { jitter: false, .. }`.
    use std::time::SystemTime;
    let seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(12345) as u64;
    // Simple xorshift
    let mut x = seed ^ (seed << 13);
    x ^= x >> 7;
    x ^= x << 17;
    (x & 0xFFFF_FFFF) as f64 / 0x1_0000_0000u64 as f64
}

// ---------------------------------------------------------------------------
// RetryPolicy
// ---------------------------------------------------------------------------

/// Retry policy for a node execution.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Total executions allowed (1 = no retries).
    pub max_attempts: u32,
    /// Backoff configuration.
    pub backoff: BackoffConfig,
}

impl RetryPolicy {
    /// Build from node attrs with graph-level fallback.
    ///
    /// `max_attempts = node.max_retries + 1`
    ///
    /// Falls back to `graph.graph_attrs.default_max_retry + 1` when
    /// `node.max_retries == 0`.
    pub fn from_node(node: &Node, graph: &Graph) -> Self {
        let max_retries = if node.max_retries > 0 {
            node.max_retries
        } else {
            graph.graph_attrs.default_max_retry
        };
        RetryPolicy {
            max_attempts: max_retries.saturating_add(1).max(1),
            backoff: BackoffConfig::default(),
        }
    }

    /// No-retry policy (execute exactly once).
    pub fn none() -> Self {
        RetryPolicy {
            max_attempts: 1,
            backoff: BackoffConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// execute_with_retry
// ---------------------------------------------------------------------------

/// Execute `handler` with the given `policy`, sleeping between retries.
///
/// Emits `PipelineEvent::StageRetrying` before each sleep and
/// `PipelineEvent::StageFailed` on permanent failure.  Event send failures
/// are silently ignored.
#[allow(clippy::too_many_arguments)]
pub async fn execute_with_retry(
    handler: &dyn Handler,
    node: &Node,
    context: &Context,
    graph: &Graph,
    logs_root: &Path,
    policy: &RetryPolicy,
    node_retries: &mut HashMap<String, u32>,
    event_tx: &broadcast::Sender<PipelineEvent>,
) -> Outcome {
    let max = policy.max_attempts.max(1);

    for attempt in 1..=max {
        // Execute the handler; map Err → fail outcome.
        let outcome = match handler.execute(node, context, graph, logs_root).await {
            Ok(o) => o,
            Err(e) => {
                let reason = e.to_string();
                if attempt < max {
                    let delay = policy.backoff.delay_for_attempt(attempt);
                    let _ = event_tx.send(PipelineEvent::StageRetrying {
                        name: node.id.clone(),
                        index: attempt as usize,
                        attempt,
                        delay,
                    });
                    tokio::time::sleep(delay).await;
                    continue;
                }
                let out = Outcome::fail(reason.clone());
                let _ = event_tx.send(PipelineEvent::StageFailed {
                    name: node.id.clone(),
                    index: attempt as usize,
                    error: reason,
                    will_retry: false,
                });
                return out;
            }
        };

        match outcome.status {
            StageStatus::Success | StageStatus::PartialSuccess | StageStatus::Skipped => {
                // Reset per-node retry counter on success.
                node_retries.remove(&node.id);
                return outcome;
            }
            StageStatus::Retry => {
                if attempt < max {
                    // Increment retry counter.
                    let count = node_retries.entry(node.id.clone()).or_insert(0);
                    *count += 1;
                    let delay = policy.backoff.delay_for_attempt(attempt);
                    let _ = event_tx.send(PipelineEvent::StageRetrying {
                        name: node.id.clone(),
                        index: attempt as usize,
                        attempt,
                        delay,
                    });
                    tokio::time::sleep(delay).await;
                    continue;
                }
                // Retries exhausted.
                if node.allow_partial {
                    return Outcome {
                        status: StageStatus::PartialSuccess,
                        notes: "retries exhausted, partial accepted".to_string(),
                        failure_reason: outcome.failure_reason,
                        ..Default::default()
                    };
                }
                let out = Outcome::fail("max retries exceeded");
                let _ = event_tx.send(PipelineEvent::StageFailed {
                    name: node.id.clone(),
                    index: attempt as usize,
                    error: "max retries exceeded".into(),
                    will_retry: false,
                });
                return out;
            }
            StageStatus::Fail => {
                // V2-ATR-002: FAIL must be retried just like RETRY (NLSpec §3.4).
                if attempt < max {
                    let count = node_retries.entry(node.id.clone()).or_insert(0);
                    *count += 1;
                    let delay = policy.backoff.delay_for_attempt(attempt);
                    let _ = event_tx.send(PipelineEvent::StageRetrying {
                        name: node.id.clone(),
                        index: attempt as usize,
                        attempt,
                        delay,
                    });
                    tokio::time::sleep(delay).await;
                    continue;
                }
                // Retries exhausted.
                if node.allow_partial {
                    return Outcome {
                        status: StageStatus::PartialSuccess,
                        notes: "retries exhausted, partial accepted".to_string(),
                        failure_reason: outcome.failure_reason,
                        ..Default::default()
                    };
                }
                let out = Outcome::fail(outcome.failure_reason.clone());
                let _ = event_tx.send(PipelineEvent::StageFailed {
                    name: node.id.clone(),
                    index: attempt as usize,
                    error: outcome.failure_reason,
                    will_retry: false,
                });
                return out;
            }
        }
    }

    // Safety fallback (should not be reached).
    Outcome::fail("max retries exceeded")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Graph, GraphAttrs, Node};
    use async_trait::async_trait;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn make_graph(default_max_retry: u32) -> Graph {
        let mut g = Graph::new("test".into());
        g.graph_attrs = GraphAttrs {
            default_max_retry,
            ..Default::default()
        };
        g
    }

    fn make_node(max_retries: u32, allow_partial: bool) -> Node {
        Node {
            id: "test_node".into(),
            max_retries,
            allow_partial,
            ..Default::default()
        }
    }

    // --- BackoffConfig ---

    #[test]
    fn delay_no_jitter_exponential() {
        let cfg = BackoffConfig {
            initial_delay_ms: 200,
            backoff_factor: 2.0,
            max_delay_ms: 60_000,
            jitter: false,
        };
        assert_eq!(cfg.delay_for_attempt(1).as_millis(), 200);
        assert_eq!(cfg.delay_for_attempt(2).as_millis(), 400);
        assert_eq!(cfg.delay_for_attempt(3).as_millis(), 800);
    }

    #[test]
    fn delay_capped_at_max() {
        let cfg = BackoffConfig {
            initial_delay_ms: 1_000,
            backoff_factor: 10.0,
            max_delay_ms: 5_000,
            jitter: false,
        };
        // 1000 * 10^5 = 100_000_000 >> 5000
        assert_eq!(cfg.delay_for_attempt(6).as_millis(), 5_000);
    }

    #[test]
    fn delay_linear_factor_one() {
        let cfg = BackoffConfig {
            initial_delay_ms: 500,
            backoff_factor: 1.0,
            max_delay_ms: 60_000,
            jitter: false,
        };
        assert_eq!(cfg.delay_for_attempt(1).as_millis(), 500);
        assert_eq!(cfg.delay_for_attempt(3).as_millis(), 500);
    }

    // --- GAP-ATR-008: jitter produces varying delays ---

    #[test]
    fn delay_with_jitter_varies_across_calls() {
        // GAP-ATR-008: NLSpec §11.5 — "Jitter is applied to backoff delays
        // when configured".  With jitter=true, the random component should cause
        // delay_for_attempt to return different values across multiple calls.
        // Statistical test: run 20 times; at least two results must differ.
        let cfg = BackoffConfig {
            initial_delay_ms: 1_000,
            backoff_factor: 1.0,
            max_delay_ms: 60_000,
            jitter: true,
        };

        let delays: Vec<u128> = (0..20)
            .map(|_| cfg.delay_for_attempt(1).as_millis())
            .collect();

        let first = delays[0];
        assert!(
            delays.iter().any(|&d| d != first),
            "jitter=true should produce varying delays across 20 calls; all were {first} ms. \
             delays: {delays:?}"
        );
    }

    // --- RetryPolicy::from_node ---

    #[test]
    fn policy_from_node_max_retries() {
        let g = make_graph(50);
        let n = make_node(3, false);
        let p = RetryPolicy::from_node(&n, &g);
        assert_eq!(p.max_attempts, 4); // 3 retries + 1 initial
    }

    #[test]
    fn policy_from_node_falls_back_to_graph_default() {
        let g = make_graph(10);
        let n = make_node(0, false); // 0 = unset, use graph default
        let p = RetryPolicy::from_node(&n, &g);
        assert_eq!(p.max_attempts, 11);
    }

    #[test]
    fn policy_none_is_one_attempt() {
        let p = RetryPolicy::none();
        assert_eq!(p.max_attempts, 1);
    }

    // --- execute_with_retry ---

    // We need a Mutex-wrapped handler to allow mutating the response list.
    struct SequenceHandler {
        calls: Arc<AtomicU32>,
        responses: std::sync::Mutex<std::collections::VecDeque<Outcome>>,
        default: Outcome,
    }

    impl SequenceHandler {
        fn new(responses: Vec<Outcome>) -> Self {
            SequenceHandler {
                calls: Arc::new(AtomicU32::new(0)),
                responses: std::sync::Mutex::new(responses.into()),
                default: Outcome::success(),
            }
        }
        fn call_count(&self) -> u32 {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl Handler for SequenceHandler {
        async fn execute(
            &self,
            _node: &Node,
            _context: &Context,
            _graph: &Graph,
            _logs_root: &Path,
        ) -> Result<Outcome, crate::error::EngineError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let mut q = self.responses.lock().unwrap();
            Ok(q.pop_front().unwrap_or(self.default.clone()))
        }
    }

    fn no_jitter_policy(max_attempts: u32) -> RetryPolicy {
        RetryPolicy {
            max_attempts,
            backoff: BackoffConfig {
                initial_delay_ms: 0,
                backoff_factor: 1.0,
                max_delay_ms: 0,
                jitter: false,
            },
        }
    }

    fn make_node_simple() -> Node {
        make_node(0, false)
    }

    #[tokio::test]
    async fn success_on_first_attempt() {
        let (tx, _rx) = tokio::sync::broadcast::channel(16);
        let handler = SequenceHandler::new(vec![Outcome::success()]);
        let ctx = Context::new();
        let g = make_graph(0);
        let node = make_node_simple();
        let policy = no_jitter_policy(3);
        let mut retries = HashMap::new();
        let out = execute_with_retry(
            &handler,
            &node,
            &ctx,
            &g,
            std::path::Path::new("/tmp"),
            &policy,
            &mut retries,
            &tx,
        )
        .await;
        assert_eq!(out.status, StageStatus::Success);
        assert_eq!(handler.call_count(), 1);
    }

    #[tokio::test]
    async fn retry_twice_then_success() {
        let (tx, _rx) = tokio::sync::broadcast::channel(16);
        let handler = SequenceHandler::new(vec![
            Outcome::retry("not ready"),
            Outcome::retry("still not ready"),
            Outcome::success(),
        ]);
        let ctx = Context::new();
        let g = make_graph(0);
        let node = make_node_simple();
        let policy = no_jitter_policy(3);
        let mut retries = HashMap::new();
        let out = execute_with_retry(
            &handler,
            &node,
            &ctx,
            &g,
            std::path::Path::new("/tmp"),
            &policy,
            &mut retries,
            &tx,
        )
        .await;
        assert_eq!(out.status, StageStatus::Success);
        assert_eq!(handler.call_count(), 3);
    }

    #[tokio::test]
    async fn fail_exhausts_retries_like_retry() {
        // V2-ATR-002: StageStatus::Fail must be retried like Retry.
        // With policy max_attempts=3 and handler always returning Fail,
        // the handler must be called 3 times before returning a final Fail.
        let (tx, _rx) = tokio::sync::broadcast::channel(16);
        let handler = SequenceHandler::new(vec![
            Outcome::fail("bad"),
            Outcome::fail("still bad"),
            Outcome::fail("really bad"),
        ]);
        let ctx = Context::new();
        let g = make_graph(0);
        let node = make_node_simple();
        let policy = no_jitter_policy(3);
        let mut retries = HashMap::new();
        let out = execute_with_retry(
            &handler,
            &node,
            &ctx,
            &g,
            std::path::Path::new("/tmp"),
            &policy,
            &mut retries,
            &tx,
        )
        .await;
        assert_eq!(out.status, StageStatus::Fail);
        assert_eq!(
            handler.call_count(),
            3,
            "Fail must be retried; expected 3 calls"
        );
    }

    #[tokio::test]
    async fn fail_triggers_retry_then_success() {
        // V2-ATR-002: Fail on first call → retry → Success on second call.
        // With max_attempts=2, pipeline succeeds (not fails immediately).
        let (tx, _rx) = tokio::sync::broadcast::channel(16);
        let handler = SequenceHandler::new(vec![
            Outcome::fail("first attempt failed"),
            Outcome::success(),
        ]);
        let ctx = Context::new();
        let g = make_graph(0);
        let node = make_node_simple();
        let policy = no_jitter_policy(2); // 1 retry allowed
        let mut retries = HashMap::new();
        let out = execute_with_retry(
            &handler,
            &node,
            &ctx,
            &g,
            std::path::Path::new("/tmp"),
            &policy,
            &mut retries,
            &tx,
        )
        .await;
        assert_eq!(
            out.status,
            StageStatus::Success,
            "Fail then Success with max_attempts=2 must succeed"
        );
        assert_eq!(handler.call_count(), 2, "Handler must be called twice");
    }

    #[tokio::test]
    async fn retry_exhausted_returns_fail() {
        let (tx, _rx) = tokio::sync::broadcast::channel(16);
        // All retries return Retry; max_attempts=2 means 1 initial + 1 retry
        let handler = SequenceHandler::new(vec![
            Outcome::retry("x"),
            Outcome::retry("x"),
            Outcome::retry("x"),
        ]);
        let ctx = Context::new();
        let g = make_graph(0);
        let node = make_node_simple();
        let policy = no_jitter_policy(2);
        let mut retries = HashMap::new();
        let out = execute_with_retry(
            &handler,
            &node,
            &ctx,
            &g,
            std::path::Path::new("/tmp"),
            &policy,
            &mut retries,
            &tx,
        )
        .await;
        assert_eq!(out.status, StageStatus::Fail);
        assert_eq!(handler.call_count(), 2);
    }

    #[tokio::test]
    async fn allow_partial_on_exhaustion() {
        let (tx, _rx) = tokio::sync::broadcast::channel(16);
        let handler = SequenceHandler::new(vec![Outcome::retry("x"), Outcome::retry("x")]);
        let ctx = Context::new();
        let g = make_graph(0);
        let mut node = make_node_simple();
        node.allow_partial = true;
        let policy = no_jitter_policy(2);
        let mut retries = HashMap::new();
        let out = execute_with_retry(
            &handler,
            &node,
            &ctx,
            &g,
            std::path::Path::new("/tmp"),
            &policy,
            &mut retries,
            &tx,
        )
        .await;
        assert_eq!(out.status, StageStatus::PartialSuccess);
    }

    #[tokio::test]
    async fn single_attempt_policy_no_retry() {
        let (tx, _rx) = tokio::sync::broadcast::channel(16);
        let handler = SequenceHandler::new(vec![Outcome::retry("x")]);
        let ctx = Context::new();
        let g = make_graph(0);
        let node = make_node_simple();
        let policy = no_jitter_policy(1);
        let mut retries = HashMap::new();
        let out = execute_with_retry(
            &handler,
            &node,
            &ctx,
            &g,
            std::path::Path::new("/tmp"),
            &policy,
            &mut retries,
            &tx,
        )
        .await;
        assert_eq!(out.status, StageStatus::Fail);
        assert_eq!(handler.call_count(), 1);
    }
}
