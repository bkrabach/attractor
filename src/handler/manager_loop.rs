//! Manager loop handler for `shape=house` / `type="stack.manager_loop"` nodes.
//!
//! Supervises a child pipeline by running it in a background task and polling
//! checkpoints in an observe → (steer) → wait cycle until the child completes
//! or max cycles are exceeded (NLSpec §4.11).

use crate::engine::{PipelineRunner, RunConfig};
use crate::error::EngineError;
use crate::graph::{Graph, Node, Value};
use crate::handler::Handler;
use crate::state::checkpoint::Checkpoint;
use crate::state::context::{Context, Outcome, StageStatus};
use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

// ---------------------------------------------------------------------------
// ManagerLoopHandler
// ---------------------------------------------------------------------------

/// Handler for `shape=house` (`type="stack.manager_loop"`) nodes.
///
/// Runs a child pipeline in a background task and observes/steers it through
/// polling cycles.  Requires a factory function that produces a configured
/// [`PipelineRunner`].
pub struct ManagerLoopHandler {
    runner_factory: Arc<dyn Fn() -> PipelineRunner + Send + Sync>,
}

impl ManagerLoopHandler {
    /// Create with a factory that produces a configured [`PipelineRunner`].
    ///
    /// The factory is called once per `execute()` invocation.
    pub fn new(runner_factory: Arc<dyn Fn() -> PipelineRunner + Send + Sync>) -> Self {
        ManagerLoopHandler { runner_factory }
    }
}

#[async_trait]
impl Handler for ManagerLoopHandler {
    async fn execute(
        &self,
        node: &Node,
        context: &Context,
        graph: &Graph,
        logs_root: &Path,
    ) -> Result<Outcome, EngineError> {
        // Read child dotfile path.
        let child_dotfile = node
            .extra
            .get("stack.child_dotfile")
            .or_else(|| graph.graph_attrs.extra.get("stack.child_dotfile"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_default();

        if child_dotfile.is_empty() {
            return Ok(Outcome::fail("No child dotfile configured"));
        }

        // Read child DOT source.
        let child_dot = match tokio::fs::read_to_string(&child_dotfile).await {
            Ok(s) => s,
            Err(e) => return Ok(Outcome::fail(format!("Failed to read child dotfile: {e}"))),
        };

        // Read configuration.
        let poll_ms: u64 = node
            .extra
            .get("manager.poll_interval_ms")
            .and_then(|v| {
                if let Value::Int(n) = v {
                    Some(*n as u64)
                } else {
                    None
                }
            })
            .unwrap_or(45_000);

        let max_cycles: u32 = node
            .extra
            .get("manager.max_cycles")
            .and_then(|v| {
                if let Value::Int(n) = v {
                    Some(*n as u32)
                } else {
                    None
                }
            })
            .unwrap_or(1_000);

        let actions: Vec<String> = node
            .extra
            .get("manager.actions")
            .and_then(|v| v.as_str())
            .unwrap_or("observe,wait")
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();

        // Build child logs root.
        let child_logs_root = logs_root.join("child");
        tokio::fs::create_dir_all(&child_logs_root).await?;

        // Spawn child pipeline.
        let runner = (self.runner_factory)();
        let child_config = RunConfig::new(child_logs_root.clone());
        let child_handle: tokio::task::JoinHandle<Result<crate::engine::RunResult, EngineError>> =
            tokio::task::spawn(async move { runner.run(&child_dot, child_config).await });

        let poll_interval = Duration::from_millis(poll_ms);

        // Observe-steer-wait loop.
        for _cycle in 1..=max_cycles {
            // Observe: read child checkpoint.
            if actions.iter().any(|a| a == "observe") {
                if let Ok(cp) = Checkpoint::load(&Checkpoint::default_path(&child_logs_root)) {
                    context.set(
                        "stack.child.current_node",
                        Value::Str(cp.current_node.clone()),
                    );
                    context.set(
                        "stack.child.completed_count",
                        Value::Int(cp.completed_nodes.len() as i64),
                    );
                    for (k, v) in &cp.context_values {
                        context.set(&format!("stack.child.context.{k}"), v.clone());
                    }
                }
            }

            // Check if child task finished.
            if child_handle.is_finished() {
                let result = child_handle.await.unwrap_or_else(|_| {
                    Err(EngineError::Handler {
                        node_id: node.id.clone(),
                        message: "child task panicked".to_string(),
                    })
                });

                return match result {
                    Ok(run_result) => {
                        let status_str = run_result.status.as_str().to_string();
                        context.set("stack.child.status", Value::Str("completed".to_string()));
                        context.set("stack.child.outcome", Value::Str(status_str));
                        if run_result.status.is_success() {
                            Ok(Outcome {
                                status: StageStatus::Success,
                                notes: "Child pipeline completed successfully".to_string(),
                                ..Default::default()
                            })
                        } else {
                            Ok(Outcome::fail("Child pipeline failed"))
                        }
                    }
                    Err(e) => Ok(Outcome::fail(format!("Child pipeline error: {e}"))),
                };
            }

            // Wait.
            if actions.iter().any(|a| a == "wait") {
                tokio::time::sleep(poll_interval).await;
            }
        }

        // Max cycles exceeded — abort child.
        child_handle.abort();
        Ok(Outcome::fail("Manager loop: max cycles exceeded"))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::PipelineRunnerBuilder;
    use crate::graph::{Graph, Node};
    use std::sync::Arc;

    fn make_child_dot(goal: &str) -> String {
        format!(
            r#"digraph child {{
    graph [goal="{goal}"]
    start [shape=Mdiamond]
    exit  [shape=Msquare]
    work  [prompt="do work"]
    start -> work -> exit
}}"#
        )
    }

    #[tokio::test]
    async fn no_child_dotfile_returns_fail() {
        let dir = tempfile::tempdir().unwrap();
        let factory: Arc<dyn Fn() -> PipelineRunner + Send + Sync> =
            Arc::new(|| PipelineRunnerBuilder::new().build().0);
        let handler = ManagerLoopHandler::new(factory);
        let g = Graph::new("test".into());
        let ctx = Context::new();
        let node = Node::default();
        let out = handler.execute(&node, &ctx, &g, dir.path()).await.unwrap();
        assert_eq!(out.status, StageStatus::Fail);
        assert!(out.failure_reason.contains("No child dotfile"));
    }

    #[tokio::test]
    async fn child_pipeline_success() {
        let dir = tempfile::tempdir().unwrap();
        let child_dir = tempfile::tempdir().unwrap();

        // Write child DOT file.
        let child_dot_content = make_child_dot("child goal");
        let child_dot_path = child_dir.path().join("child.dot");
        std::fs::write(&child_dot_path, &child_dot_content).unwrap();

        let factory: Arc<dyn Fn() -> PipelineRunner + Send + Sync> =
            Arc::new(|| PipelineRunnerBuilder::new().build().0);
        let handler = ManagerLoopHandler::new(factory);

        let mut g = Graph::new("test".into());
        let mut n = Node {
            id: "manager".to_string(),
            ..Default::default()
        };
        n.extra.insert(
            "stack.child_dotfile".to_string(),
            Value::Str(child_dot_path.to_str().unwrap().to_string()),
        );
        // Fast polling for tests.
        n.extra
            .insert("manager.poll_interval_ms".to_string(), Value::Int(10));
        n.extra
            .insert("manager.max_cycles".to_string(), Value::Int(1000));
        g.nodes.insert("manager".to_string(), n.clone());

        let ctx = Context::new();
        let out = handler.execute(&n, &ctx, &g, dir.path()).await.unwrap();
        assert_eq!(out.status, StageStatus::Success);
    }
}
