//! Parallel fan-out handler for `shape=component` / `type="parallel"` nodes.
//!
//! Fans out to all outgoing edges concurrently using [`tokio::task::JoinSet`].
//! Each branch receives an isolated context clone; results are collected and
//! stored in `context["parallel.results"]` for a downstream fan-in node.

use crate::error::EngineError;
use crate::events::PipelineEvent;
use crate::graph::{Graph, Node, Value};
use crate::handler::Handler;
use crate::handler::fan_in::BranchResult;
use crate::state::context::{Context, Outcome, StageStatus};
use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::broadcast;

// ---------------------------------------------------------------------------
// Join and error policies
// ---------------------------------------------------------------------------

/// Policy for determining when fan-out is considered complete.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JoinPolicy {
    /// Wait for ALL branches. Default.
    WaitAll,
    /// Satisfied after the first successful branch.
    FirstSuccess,
    /// At least K of N branches must succeed.
    KOfN(usize),
}

impl JoinPolicy {
    pub fn parse(s: &str) -> Self {
        if let Some(rest) = s.strip_prefix("k_of_n:") {
            if let Ok(k) = rest.parse::<usize>() {
                return JoinPolicy::KOfN(k);
            }
        }
        match s {
            "first_success" => JoinPolicy::FirstSuccess,
            _ => JoinPolicy::WaitAll,
        }
    }
}

/// Policy for handling branch failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorPolicy {
    /// Continue remaining branches on failure. Default.
    Continue,
    /// Cancel all remaining branches on first failure.
    FailFast,
    /// Ignore failures; use only successful results.
    Ignore,
}

impl ErrorPolicy {
    pub fn parse(s: &str) -> Self {
        match s {
            "fail_fast" => ErrorPolicy::FailFast,
            "ignore" => ErrorPolicy::Ignore,
            _ => ErrorPolicy::Continue,
        }
    }
}

// ---------------------------------------------------------------------------
// ParallelHandler
// ---------------------------------------------------------------------------

/// Handler for `shape=component` (`type="parallel"`) nodes.
///
/// Executes the DIRECT target node of each outgoing branch edge concurrently.
/// Results are stored in `context["parallel.results"]` as a JSON string.
pub struct ParallelHandler {
    registry: Arc<crate::handler::HandlerRegistry>,
    event_tx: broadcast::Sender<PipelineEvent>,
}

impl ParallelHandler {
    pub fn new(
        registry: Arc<crate::handler::HandlerRegistry>,
        event_tx: broadcast::Sender<PipelineEvent>,
    ) -> Self {
        ParallelHandler { registry, event_tx }
    }
}

#[async_trait]
impl Handler for ParallelHandler {
    async fn execute(
        &self,
        node: &Node,
        context: &Context,
        graph: &Graph,
        logs_root: &Path,
    ) -> Result<Outcome, EngineError> {
        let branches = graph.outgoing_edges(&node.id);
        if branches.is_empty() {
            return Ok(Outcome::fail("No branches to execute"));
        }

        // Read configuration from node extra attrs.
        let join_policy = node
            .extra
            .get("join_policy")
            .and_then(|v| v.as_str())
            .map(JoinPolicy::parse)
            .unwrap_or(JoinPolicy::WaitAll);
        let error_policy = node
            .extra
            .get("error_policy")
            .and_then(|v| v.as_str())
            .map(ErrorPolicy::parse)
            .unwrap_or(ErrorPolicy::Continue);
        let max_parallel: usize = node
            .extra
            .get("max_parallel")
            .and_then(|v| {
                if let Value::Int(n) = v {
                    Some(*n as usize)
                } else {
                    None
                }
            })
            .unwrap_or(4)
            .max(1);

        let branch_count = branches.len();
        let _ = self
            .event_tx
            .send(PipelineEvent::ParallelStarted { branch_count });

        let parallel_start = Instant::now();

        // Build a list of (branch_id, branch_node) to spawn.
        let mut branch_nodes: Vec<(String, Node)> = Vec::new();
        for (i, edge) in branches.iter().enumerate() {
            let branch_id = edge.to.clone();
            let branch_node = match graph.node(&branch_id) {
                Some(n) => n.clone(),
                None => {
                    return Ok(Outcome::fail(format!(
                        "Branch target node '{branch_id}' not found in graph"
                    )));
                }
            };
            let _ = self.event_tx.send(PipelineEvent::ParallelBranchStarted {
                branch: branch_id.clone(),
                index: i,
            });
            branch_nodes.push((branch_id, branch_node));
        }

        // Execute branches using a semaphore-bounded JoinSet.
        let semaphore = Arc::new(tokio::sync::Semaphore::new(max_parallel));
        let mut join_set: tokio::task::JoinSet<BranchResult> = tokio::task::JoinSet::new();

        let registry = Arc::clone(&self.registry);
        let graph_clone = graph.clone();
        let logs_root_buf = logs_root.to_path_buf();

        for (branch_id, branch_node) in branch_nodes {
            let sem = Arc::clone(&semaphore);
            let reg = Arc::clone(&registry);
            let g = graph_clone.clone();
            let lr = logs_root_buf.clone();
            let branch_ctx = context.clone_isolated();
            let bid = branch_id.clone();

            join_set.spawn(async move {
                let _permit = sem.acquire_owned().await.expect("semaphore closed");
                let branch_start = Instant::now();
                let handler = reg.resolve(&branch_node);
                let result = handler.execute(&branch_node, &branch_ctx, &g, &lr).await;
                let _duration = branch_start.elapsed();
                match result {
                    Ok(out) => {
                        // Capture failure_reason for any non-success status
                        let error = if !out.status.is_success() {
                            let reason = &out.failure_reason;
                            if reason.is_empty() {
                                None
                            } else {
                                Some(reason.clone())
                            }
                        } else {
                            None
                        };
                        BranchResult {
                            branch_id: bid,
                            status: out.status,
                            notes: out.notes,
                            error,
                        }
                    }
                    Err(e) => {
                        let error_msg = e.to_string();
                        BranchResult::fail(bid, format!("Handler error: {error_msg}"), error_msg)
                    }
                }
            });
        }

        // Collect results.
        let mut results: Vec<BranchResult> = Vec::with_capacity(branch_count);
        let mut should_abort = false;

        while let Some(res) = join_set.join_next().await {
            match res {
                Ok(branch_result) => {
                    // Emit branch completed event with error detail.
                    let _ = self.event_tx.send(PipelineEvent::ParallelBranchCompleted {
                        branch: branch_result.branch_id.clone(),
                        index: results.len(),
                        duration: std::time::Duration::ZERO,
                        success: branch_result.status.is_success(),
                        error: branch_result.error.clone(),
                    });

                    if error_policy == ErrorPolicy::FailFast
                        && branch_result.status == StageStatus::Fail
                        && !should_abort
                    {
                        should_abort = true;
                        join_set.abort_all();
                    }
                    results.push(branch_result);
                }
                Err(join_err) => {
                    // Task aborted or panicked — record as fail with details.
                    let error_msg = if join_err.is_cancelled() {
                        "branch task cancelled".to_string()
                    } else if join_err.is_panic() {
                        "branch task panicked".to_string()
                    } else {
                        format!("branch task failed: {join_err}")
                    };
                    // branch_id is irrecoverable from JoinSet on panic/cancel
                    let branch = BranchResult::fail(
                        "unknown",
                        "Branch task did not complete normally",
                        &error_msg,
                    );
                    let _ = self.event_tx.send(PipelineEvent::ParallelBranchCompleted {
                        branch: "unknown".to_string(),
                        index: results.len(),
                        duration: std::time::Duration::ZERO,
                        success: false,
                        error: Some(error_msg),
                    });
                    results.push(branch);
                }
            }
        }

        let duration = parallel_start.elapsed();
        let success_count = results.iter().filter(|r| r.status.is_success()).count();
        let failure_count = results
            .iter()
            .filter(|r| r.status == StageStatus::Fail)
            .count();

        let _ = self.event_tx.send(PipelineEvent::ParallelCompleted {
            duration,
            success_count,
            failure_count,
        });

        // Store results in context for fan-in.
        let results_json = serde_json::to_string(&results).unwrap_or_else(|_| "[]".to_string());
        context.set("parallel.results", Value::Str(results_json));

        // Determine overall outcome.
        let final_status = match join_policy {
            JoinPolicy::WaitAll => {
                if failure_count == 0 {
                    StageStatus::Success
                } else {
                    StageStatus::PartialSuccess
                }
            }
            JoinPolicy::FirstSuccess => {
                if success_count > 0 {
                    StageStatus::Success
                } else {
                    StageStatus::Fail
                }
            }
            JoinPolicy::KOfN(k) => {
                if success_count >= k {
                    StageStatus::Success
                } else {
                    StageStatus::Fail
                }
            }
        };

        Ok(Outcome {
            status: final_status,
            notes: format!(
                "Parallel: {success_count} succeeded, {failure_count} failed of {branch_count}"
            ),
            ..Default::default()
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Edge, Graph, GraphAttrs, Node};
    use crate::handler::{CodergenHandler, HandlerRegistry, StartHandler};
    use crate::testing::MockCodergenBackend;
    use std::sync::Arc;

    fn make_parallel_graph(branch_ids: Vec<&str>) -> (Graph, String) {
        let mut g = Graph::new("test".into());
        g.graph_attrs = GraphAttrs {
            default_max_retry: 0,
            ..Default::default()
        };

        let parallel_id = "parallel_node".to_string();
        let par = Node {
            id: parallel_id.clone(),
            shape: "component".to_string(),
            ..Default::default()
        };
        g.nodes.insert(parallel_id.clone(), par);

        for bid in &branch_ids {
            let n = Node {
                id: bid.to_string(),
                shape: "box".to_string(),
                ..Default::default()
            };
            g.nodes.insert(bid.to_string(), n);

            g.edges.push(Edge {
                from: parallel_id.clone(),
                to: bid.to_string(),
                ..Default::default()
            });
        }

        (g, parallel_id)
    }

    fn make_registry_with_mock(mock: Arc<MockCodergenBackend>) -> Arc<HandlerRegistry> {
        let codergen = CodergenHandler::new(Some(Box::new(MockProxyBackend(mock))));
        let default = Arc::new(codergen);
        let mut reg = HandlerRegistry::new(Arc::new(StartHandler));
        reg.register("codergen", default.clone());
        Arc::new(reg)
    }

    // Proxy to allow Arc<MockCodergenBackend>
    struct MockProxyBackend(Arc<MockCodergenBackend>);

    #[async_trait::async_trait]
    impl crate::handler::CodergenBackend for MockProxyBackend {
        async fn run(
            &self,
            node: &Node,
            prompt: &str,
            ctx: &Context,
        ) -> Result<crate::handler::CodergenResult, EngineError> {
            self.0.run(node, prompt, ctx).await
        }
    }

    #[tokio::test]
    async fn all_success_returns_success() {
        let (tx, _rx) = broadcast::channel(64);
        let (graph, node_id) = make_parallel_graph(vec!["A", "B"]);
        let mock = Arc::new(MockCodergenBackend::new());
        let registry = make_registry_with_mock(mock);
        let handler = ParallelHandler::new(registry, tx);

        let dir = tempfile::tempdir().unwrap();
        let ctx = Context::new();
        let node = graph.node(&node_id).unwrap().clone();
        let out = handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();
        assert_eq!(out.status, StageStatus::Success);
    }

    #[tokio::test]
    async fn no_branches_returns_fail() {
        let (tx, _rx) = broadcast::channel(64);
        let mut g = Graph::new("test".into());
        let n = Node {
            id: "p".to_string(),
            ..Default::default()
        };
        g.nodes.insert("p".to_string(), n.clone());

        let default_handler = Arc::new(StartHandler);
        let reg = Arc::new(HandlerRegistry::new(default_handler));
        let handler = ParallelHandler::new(reg, tx);
        let dir = tempfile::tempdir().unwrap();
        let ctx = Context::new();
        let out = handler.execute(&n, &ctx, &g, dir.path()).await.unwrap();
        assert_eq!(out.status, StageStatus::Fail);
    }

    #[tokio::test]
    async fn results_stored_in_context() {
        let (tx, _rx) = broadcast::channel(64);
        let (graph, node_id) = make_parallel_graph(vec!["A", "B"]);
        let mock = Arc::new(MockCodergenBackend::new());
        let registry = make_registry_with_mock(mock);
        let handler = ParallelHandler::new(registry, tx);

        let dir = tempfile::tempdir().unwrap();
        let ctx = Context::new();
        let node = graph.node(&node_id).unwrap().clone();
        handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();

        let results_str = ctx.get_string("parallel.results");
        assert!(!results_str.is_empty());
        let results: Vec<BranchResult> = serde_json::from_str(&results_str).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn first_success_policy() {
        let (tx, _rx) = broadcast::channel(64);
        let (mut graph, node_id) = make_parallel_graph(vec!["A", "B"]);
        // Add join_policy extra attr
        graph.nodes.get_mut(&node_id).unwrap().extra.insert(
            "join_policy".to_string(),
            Value::Str("first_success".to_string()),
        );
        let mock = Arc::new(MockCodergenBackend::new());
        let registry = make_registry_with_mock(mock);
        let handler = ParallelHandler::new(registry, tx);
        let dir = tempfile::tempdir().unwrap();
        let ctx = Context::new();
        let node = graph.node(&node_id).unwrap().clone();
        let out = handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();
        assert_eq!(out.status, StageStatus::Success);
    }

    #[tokio::test]
    async fn failed_branch_has_error_in_result() {
        let (tx, _rx) = broadcast::channel(64);
        let (graph, node_id) = make_parallel_graph(vec!["A", "B"]);
        let mock = Arc::new(MockCodergenBackend::new());
        // A succeeds, B fails with a specific reason
        mock.add_success("A");
        mock.add_fail("B", "authentication error: invalid api key");
        let registry = make_registry_with_mock(mock);
        let handler = ParallelHandler::new(registry, tx);

        let dir = tempfile::tempdir().unwrap();
        let ctx = Context::new();
        let node = graph.node(&node_id).unwrap().clone();
        let out = handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();
        // WaitAll: one success + one fail = PartialSuccess
        assert_eq!(out.status, StageStatus::PartialSuccess);

        // Verify results in context contain the error
        let results_str = ctx.get_string("parallel.results");
        let results: Vec<BranchResult> = serde_json::from_str(&results_str).unwrap();
        assert_eq!(results.len(), 2);

        // Find the failed branch
        let failed = results.iter().find(|r| r.status == StageStatus::Fail);
        assert!(failed.is_some(), "should have a failed branch");
        let failed = failed.unwrap();
        assert_eq!(failed.branch_id, "B");
        assert!(
            failed.error.is_some(),
            "failed branch should have error message"
        );
        assert_eq!(
            failed.error.as_deref(),
            Some("authentication error: invalid api key")
        );
    }

    #[tokio::test]
    async fn successful_branch_has_no_error_in_result() {
        let (tx, _rx) = broadcast::channel(64);
        let (graph, node_id) = make_parallel_graph(vec!["A"]);
        let mock = Arc::new(MockCodergenBackend::new());
        let registry = make_registry_with_mock(mock);
        let handler = ParallelHandler::new(registry, tx);

        let dir = tempfile::tempdir().unwrap();
        let ctx = Context::new();
        let node = graph.node(&node_id).unwrap().clone();
        handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();

        let results_str = ctx.get_string("parallel.results");
        let results: Vec<BranchResult> = serde_json::from_str(&results_str).unwrap();
        assert_eq!(results.len(), 1);
        assert!(
            results[0].error.is_none(),
            "successful branch should not have error"
        );
    }

    #[tokio::test]
    async fn branch_failure_emits_event_with_error() {
        let (tx, mut rx) = broadcast::channel(64);
        let (graph, node_id) = make_parallel_graph(vec!["A"]);
        let mock = Arc::new(MockCodergenBackend::new());
        mock.add_fail("A", "model not found");
        let registry = make_registry_with_mock(mock);
        let handler = ParallelHandler::new(registry, tx);

        let dir = tempfile::tempdir().unwrap();
        let ctx = Context::new();
        let node = graph.node(&node_id).unwrap().clone();
        handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();

        // Drain events and find ParallelBranchCompleted
        let mut found_error_event = false;
        while let Ok(ev) = rx.try_recv() {
            if let PipelineEvent::ParallelBranchCompleted { success, error, .. } = ev {
                assert!(!success);
                assert!(error.is_some(), "event should carry error message");
                assert_eq!(error.as_deref(), Some("model not found"));
                found_error_event = true;
            }
        }
        assert!(
            found_error_event,
            "should have received ParallelBranchCompleted with error"
        );
    }

    /// A handler that always returns Err(EngineError) to test the error path.
    struct ErrorHandler;

    #[async_trait::async_trait]
    impl Handler for ErrorHandler {
        async fn execute(
            &self,
            _node: &Node,
            _context: &Context,
            _graph: &Graph,
            _logs_root: &Path,
        ) -> Result<crate::state::context::Outcome, EngineError> {
            Err(EngineError::Backend(
                "LLM provider returned 401 Unauthorized".into(),
            ))
        }
    }

    #[tokio::test]
    async fn handler_error_captured_in_branch_result() {
        let (tx, _rx) = broadcast::channel(64);
        let (graph, node_id) = make_parallel_graph(vec!["A"]);

        // Register ErrorHandler as the codergen handler
        let default_handler = Arc::new(StartHandler);
        let mut reg = HandlerRegistry::new(default_handler);
        reg.register("codergen", Arc::new(ErrorHandler));
        let reg = Arc::new(reg);

        let handler = ParallelHandler::new(reg, tx);
        let dir = tempfile::tempdir().unwrap();
        let ctx = Context::new();
        let node = graph.node(&node_id).unwrap().clone();
        let out = handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();
        // WaitAll with 1 failed branch → PartialSuccess (not Fail)
        assert_eq!(out.status, StageStatus::PartialSuccess);

        let results_str = ctx.get_string("parallel.results");
        let results: Vec<BranchResult> = serde_json::from_str(&results_str).unwrap();
        assert_eq!(results.len(), 1);
        let r = &results[0];
        assert_eq!(r.status, StageStatus::Fail);
        assert!(r.error.is_some());
        assert!(
            r.error.as_deref().unwrap().contains("401 Unauthorized"),
            "error should contain the original error message, got: {:?}",
            r.error
        );
    }
}
