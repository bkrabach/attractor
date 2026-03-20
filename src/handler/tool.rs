//! Tool handler — executes a configured shell command for `shape=parallelogram` nodes.

use async_trait::async_trait;
use std::path::Path;
use tokio::process::Command;

use crate::error::EngineError;
use crate::graph::{Graph, Node, Value};
use crate::handler::Handler;
use crate::state::context::{Context, Outcome};

/// Handler for `shape=parallelogram` (`type="tool"`) nodes.
///
/// Reads the shell command from `node.extra["tool_command"]`, executes it via
/// `sh -c`, captures stdout/stderr, and returns an [`Outcome`] with the
/// output stored in `context_updates["tool.output"]`.
pub struct ToolHandler;

#[async_trait]
impl Handler for ToolHandler {
    async fn execute(
        &self,
        node: &Node,
        context: &Context,
        _graph: &Graph,
        logs_root: &Path,
    ) -> Result<Outcome, EngineError> {
        // 1. Read tool_command from node extra attributes.
        let command = match node.extra.get("tool_command").and_then(|v| v.as_str()) {
            Some(cmd) if !cmd.is_empty() => cmd.to_owned(),
            _ => {
                return Ok(Outcome::fail("No tool_command specified"));
            }
        };

        // 2. Determine the working directory for the command.
        //    ATR-BUG-001: Use _working_dir from context (set by RunConfig) so
        //    tool_command scripts run in the pipeline's project directory, not
        //    the temporary logs directory.  Fall back to logs_root for backwards
        //    compatibility when no working_dir is configured.
        let cwd_string = context.get_string("_working_dir");
        let effective_cwd: &Path = if cwd_string.is_empty() {
            logs_root
        } else {
            Path::new(cwd_string.as_str())
        };

        // 3. Build the async command.
        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg(&command);
        cmd.current_dir(effective_cwd);

        // 4. Execute with optional timeout.
        let output_result = if let Some(timeout) = node.timeout {
            match tokio::time::timeout(timeout, cmd.output()).await {
                Ok(result) => result,
                Err(_) => {
                    return Ok(Outcome::fail(format!(
                        "Tool timed out after {}ms",
                        timeout.as_millis()
                    )));
                }
            }
        } else {
            cmd.output().await
        };

        // 5. Handle spawn / IO error.
        let output = match output_result {
            Ok(o) => o,
            Err(e) => {
                return Ok(Outcome::fail(format!("Failed to spawn command: {e}")));
            }
        };

        // 6. Build outcome based on exit code.
        let exit_code = output.status.code().unwrap_or(-1) as i64;
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

        let mut context_updates = std::collections::HashMap::new();
        context_updates.insert("tool.exit_code".to_string(), Value::Int(exit_code));

        if output.status.success() {
            context_updates.insert("tool.output".to_string(), Value::Str(stdout.clone()));
            // SRV-BUG-005: DOT edge conditions use `context.tool_stdout` (e.g.
            // `condition="context.tool_stdout=done"`) but the handler only stored
            // stdout as `tool.output`.  Store it under `tool_stdout` as well so
            // all existing dotpowers.dot conditions resolve correctly.
            context_updates.insert("tool_stdout".to_string(), Value::Str(stdout.clone()));
            Ok(Outcome {
                notes: format!("Tool completed: {command}"),
                context_updates,
                ..Outcome::success()
            })
        } else {
            // Non-zero exit: still expose stdout/stderr for condition checks.
            context_updates.insert("tool.output".to_string(), Value::Str(stdout.clone()));
            context_updates.insert("tool_stdout".to_string(), Value::Str(stdout.clone()));
            Ok(Outcome {
                failure_reason: format!("Tool exited with code {exit_code}: {stderr}"),
                context_updates,
                ..Outcome::fail(String::new())
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::state::context::StageStatus;
    use std::time::Duration;

    fn make_node_with_command(cmd: &str) -> Node {
        let mut n = Node {
            id: "tool_node".to_string(),
            ..Default::default()
        };
        n.extra
            .insert("tool_command".to_string(), Value::Str(cmd.to_string()));
        n
    }

    #[tokio::test]
    async fn echo_command_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let handler = ToolHandler;
        let node = make_node_with_command("echo hello");
        let ctx = Context::new();
        let graph = Graph::new("t".into());
        let outcome = handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();
        assert_eq!(outcome.status, StageStatus::Success);
        assert_eq!(
            outcome.context_updates.get("tool.output"),
            Some(&Value::Str("hello\n".to_string()))
        );
        assert_eq!(
            outcome.context_updates.get("tool.exit_code"),
            Some(&Value::Int(0))
        );
    }

    /// SRV-BUG-005: DOT edge conditions use `context.tool_stdout` but
    /// ToolHandler only stored stdout as `tool.output`.  The handler must
    /// ALSO store stdout under the `tool_stdout` key so conditions like
    /// `context.tool_stdout=done` can resolve correctly.
    #[tokio::test]
    async fn stdout_stored_as_tool_stdout_key_for_conditions() {
        let dir = tempfile::tempdir().unwrap();
        let handler = ToolHandler;
        let node = make_node_with_command("printf 'done'");
        let ctx = Context::new();
        let graph = Graph::new("t".into());
        let outcome = handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();
        assert_eq!(outcome.status, StageStatus::Success);
        // Must be stored under tool_stdout for DOT conditions like
        // condition="context.tool_stdout=done"
        assert_eq!(
            outcome.context_updates.get("tool_stdout"),
            Some(&Value::Str("done".to_string())),
            "tool_stdout key required for DOT edge conditions"
        );
    }

    #[tokio::test]
    async fn non_zero_exit_fails() {
        let dir = tempfile::tempdir().unwrap();
        let handler = ToolHandler;
        let node = make_node_with_command("exit 1");
        let ctx = Context::new();
        let graph = Graph::new("t".into());
        let outcome = handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();
        assert_eq!(outcome.status, StageStatus::Fail);
        assert!(outcome.context_updates.contains_key("tool.exit_code"));
    }

    #[tokio::test]
    async fn missing_command_fails() {
        let dir = tempfile::tempdir().unwrap();
        let handler = ToolHandler;
        let node = Node::default(); // no tool_command extra attr
        let ctx = Context::new();
        let graph = Graph::new("t".into());
        let outcome = handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();
        assert_eq!(outcome.status, StageStatus::Fail);
        assert!(outcome.failure_reason.contains("tool_command"));
    }

    /// ATR-BUG-001: When `_working_dir` is set in context, the tool command
    /// must run in that directory, NOT in `logs_root`.
    #[tokio::test]
    async fn uses_working_dir_from_context_when_set() {
        let logs_dir = tempfile::tempdir().unwrap();
        let work_dir = tempfile::tempdir().unwrap();

        // Create a marker file ONLY in the working directory.
        std::fs::write(work_dir.path().join("marker.txt"), "found-it").unwrap();

        let handler = ToolHandler;
        let node = make_node_with_command("cat marker.txt");
        let ctx = Context::new();
        // Set _working_dir in context (as the engine does via RunConfig).
        ctx.set(
            "_working_dir",
            Value::Str(work_dir.path().to_string_lossy().to_string()),
        );
        let graph = Graph::new("t".into());
        let outcome = handler
            .execute(&node, &ctx, &graph, logs_dir.path())
            .await
            .unwrap();

        assert_eq!(
            outcome.status,
            StageStatus::Success,
            "tool must succeed when running in working_dir; got: {}",
            outcome.failure_reason
        );
        assert_eq!(
            outcome.context_updates.get("tool.output"),
            Some(&Value::Str("found-it".to_string())),
            "tool must read file from working_dir, not logs_root"
        );
    }

    /// ATR-BUG-001: When `_working_dir` is NOT set, tool falls back to
    /// `logs_root` (backwards compatibility).
    #[tokio::test]
    async fn falls_back_to_logs_root_when_no_working_dir() {
        let logs_dir = tempfile::tempdir().unwrap();

        // Create a marker file in logs_root.
        std::fs::write(logs_dir.path().join("marker.txt"), "in-logs").unwrap();

        let handler = ToolHandler;
        let node = make_node_with_command("cat marker.txt");
        let ctx = Context::new();
        // No _working_dir set.
        let graph = Graph::new("t".into());
        let outcome = handler
            .execute(&node, &ctx, &graph, logs_dir.path())
            .await
            .unwrap();

        assert_eq!(outcome.status, StageStatus::Success);
        assert_eq!(
            outcome.context_updates.get("tool.output"),
            Some(&Value::Str("in-logs".to_string())),
            "tool must fall back to logs_root when no _working_dir"
        );
    }

    #[tokio::test]
    async fn timeout_fails() {
        let dir = tempfile::tempdir().unwrap();
        let handler = ToolHandler;
        let mut node = make_node_with_command("sleep 10");
        node.timeout = Some(Duration::from_millis(50));
        let ctx = Context::new();
        let graph = Graph::new("t".into());
        let outcome = handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();
        assert_eq!(outcome.status, StageStatus::Fail);
        assert!(
            outcome.failure_reason.contains("timed out"),
            "expected timeout message, got: {}",
            outcome.failure_reason
        );
    }
}
