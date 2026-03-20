//! Codergen handler — the primary LLM task handler for `shape=box` nodes.
//!
//! Builds a prompt (expanding `$goal`), calls a pluggable [`CodergenBackend`],
//! writes `prompt.md` / `response.md` / `status.json` to the stage log
//! directory, and returns an [`Outcome`].

use async_trait::async_trait;
use serde_json;
use std::path::Path;
use tokio::fs;

use crate::error::EngineError;
use crate::graph::{Graph, Node, Value};
use crate::handler::Handler;
use crate::state::context::{Context, Outcome};

// ---------------------------------------------------------------------------
// CodergenResult
// ---------------------------------------------------------------------------

/// The value returned by a [`CodergenBackend`].
#[derive(Debug)]
pub enum CodergenResult {
    /// Plain text response — the handler wraps it in a Success [`Outcome`].
    Text(String),
    /// Full outcome returned directly by the backend (e.g. an agent that
    /// already decided success/fail/retry).
    Outcome(Outcome),
}

// ---------------------------------------------------------------------------
// CodergenBackend trait
// ---------------------------------------------------------------------------

/// The pluggable LLM execution backend for codergen nodes.
///
/// Implementations may call an LLM API directly, wrap a `coding-agent-loop`
/// Session, spawn a CLI subprocess, or do anything else.  The pipeline engine
/// only cares about the returned [`CodergenResult`].
#[async_trait]
pub trait CodergenBackend: Send + Sync {
    /// Execute the LLM task described by `prompt` and return a result.
    async fn run(
        &self,
        node: &Node,
        prompt: &str,
        context: &Context,
    ) -> Result<CodergenResult, EngineError>;
}

// ---------------------------------------------------------------------------
// CodergenHandler
// ---------------------------------------------------------------------------

/// Handler for `shape=box` (codergen / LLM task) nodes.
///
/// If `backend` is `None`, the handler runs in **simulation mode** and returns
/// a synthetic response without calling any external service.
pub struct CodergenHandler {
    backend: Option<Box<dyn CodergenBackend>>,
}

impl CodergenHandler {
    /// Create a handler with the given backend.  Pass `None` for simulation mode.
    pub fn new(backend: Option<Box<dyn CodergenBackend>>) -> Self {
        CodergenHandler { backend }
    }
}

#[async_trait]
impl Handler for CodergenHandler {
    async fn execute(
        &self,
        node: &Node,
        context: &Context,
        graph: &Graph,
        logs_root: &Path,
    ) -> Result<Outcome, EngineError> {
        // 1. Build prompt (fall back to label if prompt is empty).
        let raw_prompt = if !node.prompt.is_empty() {
            node.prompt.clone()
        } else {
            node.label.clone()
        };
        let prompt = expand_variables(&raw_prompt, graph, context);

        // 1b. Prepend preamble if present.
        // The *original* prompt (without preamble) is written to prompt.md so that
        // `build_full()` does not re-read prior preambles, avoiding O(n²) transcript
        // inflation.  The augmented prompt (with preamble) is sent to the backend only.
        let augmented_prompt = {
            let preamble = context.get_string("_preamble");
            if preamble.is_empty() {
                prompt.clone()
            } else {
                format!("{preamble}\n---\n{prompt}")
            }
        };

        // 2. Create stage directory.
        let stage_dir = logs_root.join(&node.id);
        fs::create_dir_all(&stage_dir).await?;

        // 3. Write prompt.md (original prompt, without preamble).
        fs::write(stage_dir.join("prompt.md"), &prompt).await?;

        // 4. Call backend (or simulate) with augmented prompt.
        let response_text: String;
        if let Some(backend) = &self.backend {
            match backend.run(node, &augmented_prompt, context).await {
                Ok(CodergenResult::Outcome(mut outcome)) => {
                    // Backend returned a full outcome.
                    // NLSpec §11.6: response.md must be written unconditionally.
                    let response_content = if !outcome.notes.is_empty() {
                        outcome.notes.clone()
                    } else {
                        format!("[Outcome] status={}", outcome.status)
                    };
                    fs::write(stage_dir.join("response.md"), &response_content).await?;
                    write_status(&stage_dir, &outcome).await?;
                    // ATR-BUG-005: Ensure last_stage and last_response are set
                    // even when the backend returns a full Outcome, so downstream
                    // human gates can find the correct LLM response.
                    outcome
                        .context_updates
                        .entry("last_stage".to_string())
                        .or_insert_with(|| Value::Str(node.id.clone()));
                    let truncated: String = response_content.chars().take(200).collect();
                    outcome
                        .context_updates
                        .entry("last_response".to_string())
                        .or_insert_with(|| Value::Str(truncated));
                    return Ok(outcome);
                }
                Ok(CodergenResult::Text(text)) => {
                    response_text = text;
                }
                Err(e) => {
                    let outcome = Outcome::fail(e.to_string());
                    write_status(&stage_dir, &outcome).await?;
                    return Ok(outcome);
                }
            }
        } else {
            response_text = format!("[Simulated] Response for stage: {}", node.id);
        }

        // 5. Write response.md.
        fs::write(stage_dir.join("response.md"), &response_text).await?;

        // 6. Build outcome.
        let truncated_response: String = response_text.chars().take(200).collect();
        let mut context_updates = std::collections::HashMap::new();
        context_updates.insert("last_stage".to_string(), Value::Str(node.id.clone()));
        context_updates.insert("last_response".to_string(), Value::Str(truncated_response));

        let outcome = Outcome {
            notes: format!("Stage completed: {}", node.id),
            context_updates,
            ..Outcome::success()
        };

        // 7. Write status.json.
        write_status(&stage_dir, &outcome).await?;

        Ok(outcome)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Expand `$goal` in `prompt`.
///
/// `$goal` is the only built-in template variable.  Context values reach LLMs
/// via the Preamble Transform, not template substitution.
pub fn expand_variables(prompt: &str, graph: &Graph, _context: &Context) -> String {
    prompt.replace("$goal", &graph.graph_attrs.goal)
}

/// Serialise `outcome` to pretty JSON and write to `{stage_dir}/status.json`.
async fn write_status(stage_dir: &Path, outcome: &Outcome) -> Result<(), EngineError> {
    let json = serde_json::to_string_pretty(outcome).map_err(|e| EngineError::Handler {
        node_id: stage_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string(),
        message: format!("failed to serialise outcome: {e}"),
    })?;
    fs::write(stage_dir.join("status.json"), json).await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Graph, GraphAttrs};
    use crate::state::context::StageStatus;
    use std::sync::Arc;

    struct MockTextBackend {
        response: String,
    }

    #[async_trait]
    impl CodergenBackend for MockTextBackend {
        async fn run(
            &self,
            _node: &Node,
            _prompt: &str,
            _context: &Context,
        ) -> Result<CodergenResult, EngineError> {
            Ok(CodergenResult::Text(self.response.clone()))
        }
    }

    struct MockOutcomeBackend {
        outcome: Outcome,
    }

    #[async_trait]
    impl CodergenBackend for MockOutcomeBackend {
        async fn run(
            &self,
            _node: &Node,
            _prompt: &str,
            _context: &Context,
        ) -> Result<CodergenResult, EngineError> {
            Ok(CodergenResult::Outcome(self.outcome.clone()))
        }
    }

    struct ErrorBackend;

    #[async_trait]
    impl CodergenBackend for ErrorBackend {
        async fn run(
            &self,
            _node: &Node,
            _prompt: &str,
            _context: &Context,
        ) -> Result<CodergenResult, EngineError> {
            Err(EngineError::Backend("backend exploded".to_string()))
        }
    }

    fn make_graph(goal: &str) -> Graph {
        let mut g = Graph::new("test".to_string());
        g.graph_attrs = GraphAttrs {
            goal: goal.to_string(),
            default_max_retry: 50,
            ..Default::default()
        };
        g
    }

    fn make_node(id: &str, prompt: &str) -> Node {
        let mut n = Node::default();
        n.id = id.to_string();
        n.prompt = prompt.to_string();
        n.label = id.to_string();
        n
    }

    #[tokio::test]
    async fn simulation_mode_returns_success() {
        let dir = tempfile::tempdir().unwrap();
        let handler = CodergenHandler::new(None);
        let node = make_node("plan", "Plan something");
        let ctx = Context::new();
        let graph = make_graph("test goal");
        let outcome = handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();
        assert_eq!(outcome.status, StageStatus::Success);
    }

    #[tokio::test]
    async fn goal_variable_expanded() {
        let dir = tempfile::tempdir().unwrap();
        let backend = MockTextBackend {
            response: "done".to_string(),
        };
        let handler = CodergenHandler::new(Some(Box::new(backend)));
        let node = make_node("plan", "Work on: $goal");
        let ctx = Context::new();
        let graph = make_graph("build a rocket");
        handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();

        let prompt_text =
            std::fs::read_to_string(dir.path().join("plan").join("prompt.md")).unwrap();
        assert_eq!(prompt_text, "Work on: build a rocket");
    }

    #[tokio::test]
    async fn empty_prompt_uses_label() {
        let dir = tempfile::tempdir().unwrap();
        let backend = MockTextBackend {
            response: "ok".to_string(),
        };
        let handler = CodergenHandler::new(Some(Box::new(backend)));
        let mut node = make_node("impl", "");
        node.label = "My Label".to_string();
        let ctx = Context::new();
        let graph = make_graph("goal");
        handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();

        let prompt_text =
            std::fs::read_to_string(dir.path().join("impl").join("prompt.md")).unwrap();
        assert_eq!(prompt_text, "My Label");
    }

    #[tokio::test]
    async fn prompt_md_written() {
        let dir = tempfile::tempdir().unwrap();
        let handler = CodergenHandler::new(None);
        let node = make_node("step", "Do the thing");
        let ctx = Context::new();
        let graph = make_graph("g");
        handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();
        let path = dir.path().join("step").join("prompt.md");
        assert!(path.exists());
        let content = std::fs::read_to_string(path).unwrap();
        assert_eq!(content, "Do the thing");
    }

    #[tokio::test]
    async fn response_md_written() {
        let dir = tempfile::tempdir().unwrap();
        let backend = MockTextBackend {
            response: "my response".to_string(),
        };
        let handler = CodergenHandler::new(Some(Box::new(backend)));
        let node = make_node("step", "prompt");
        let ctx = Context::new();
        let graph = make_graph("g");
        handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();
        let content = std::fs::read_to_string(dir.path().join("step").join("response.md")).unwrap();
        assert_eq!(content, "my response");
    }

    #[tokio::test]
    async fn status_json_written_and_valid() {
        let dir = tempfile::tempdir().unwrap();
        let handler = CodergenHandler::new(None);
        let node = make_node("step", "prompt");
        let ctx = Context::new();
        let graph = make_graph("g");
        handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();
        let text = std::fs::read_to_string(dir.path().join("step").join("status.json")).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
        // NLSpec Appendix C: field is named "outcome", not "status"
        assert_eq!(parsed["outcome"], "success");
    }

    #[tokio::test]
    async fn backend_text_response_in_context_updates() {
        let dir = tempfile::tempdir().unwrap();
        let backend = MockTextBackend {
            response: "text response".to_string(),
        };
        let handler = CodergenHandler::new(Some(Box::new(backend)));
        let node = make_node("n", "p");
        let ctx = Context::new();
        let graph = make_graph("g");
        let outcome = handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();
        assert_eq!(
            outcome.context_updates.get("last_response"),
            Some(&Value::Str("text response".to_string()))
        );
    }

    #[tokio::test]
    async fn response_md_written_for_outcome_variant() {
        // NLSpec §11.6: response.md must be written unconditionally, including
        // when the backend returns CodergenResult::Outcome directly.
        let dir = tempfile::tempdir().unwrap();
        let expected = Outcome {
            notes: "outcome notes text".to_string(),
            ..Outcome::success()
        };
        let backend = MockOutcomeBackend {
            outcome: expected.clone(),
        };
        let handler = CodergenHandler::new(Some(Box::new(backend)));
        let node = make_node("n", "p");
        let ctx = Context::new();
        let graph = make_graph("g");
        handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();
        let path = dir.path().join("n").join("response.md");
        assert!(
            path.exists(),
            "response.md must exist when backend returns Outcome"
        );
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(
            !content.is_empty(),
            "response.md must not be empty when backend returns Outcome"
        );
    }

    #[tokio::test]
    async fn backend_outcome_returned_directly() {
        let dir = tempfile::tempdir().unwrap();
        let expected = Outcome {
            notes: "from backend".to_string(),
            ..Outcome::fail("intentional")
        };
        let backend = MockOutcomeBackend {
            outcome: expected.clone(),
        };
        let handler = CodergenHandler::new(Some(Box::new(backend)));
        let node = make_node("n", "p");
        let ctx = Context::new();
        let graph = make_graph("g");
        let outcome = handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();
        assert_eq!(outcome.status, StageStatus::Fail);
        assert_eq!(outcome.notes, "from backend");
    }

    #[tokio::test]
    async fn backend_error_returns_fail_outcome() {
        let dir = tempfile::tempdir().unwrap();
        let handler = CodergenHandler::new(Some(Box::new(ErrorBackend)));
        let node = make_node("n", "p");
        let ctx = Context::new();
        let graph = make_graph("g");
        let outcome = handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();
        assert_eq!(outcome.status, StageStatus::Fail);
        assert!(outcome.failure_reason.contains("backend exploded"));
    }

    #[tokio::test]
    async fn last_response_stored_up_to_200_chars() {
        let dir = tempfile::tempdir().unwrap();
        let long_response = "A".repeat(300);
        let backend = MockTextBackend {
            response: long_response,
        };
        let handler = CodergenHandler::new(Some(Box::new(backend)));
        let node = make_node("n", "p");
        let ctx = Context::new();
        let graph = make_graph("g");
        let outcome = handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();
        if let Some(Value::Str(s)) = outcome.context_updates.get("last_response") {
            assert_eq!(s.len(), 200, "last_response must truncate at 200 chars");
        } else {
            panic!("last_response not found or wrong type");
        }
    }

    #[tokio::test]
    async fn last_response_not_truncated_when_short() {
        let dir = tempfile::tempdir().unwrap();
        let short_response = "Hello world".to_string();
        let backend = MockTextBackend {
            response: short_response.clone(),
        };
        let handler = CodergenHandler::new(Some(Box::new(backend)));
        let node = make_node("n", "p");
        let ctx = Context::new();
        let graph = make_graph("g");
        let outcome = handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();
        assert_eq!(
            outcome.context_updates.get("last_response"),
            Some(&Value::Str(short_response)),
            "short responses must be stored in full"
        );
    }

    #[test]
    fn expand_variables_replaces_goal() {
        let ctx = Context::new();
        let graph = make_graph("build a rocket");
        let result = expand_variables("Goal is: $goal", &graph, &ctx);
        assert_eq!(result, "Goal is: build a rocket");
    }

    #[test]
    fn expand_variables_multiple_occurrences() {
        let ctx = Context::new();
        let graph = make_graph("X");
        let result = expand_variables("$goal and $goal", &graph, &ctx);
        assert_eq!(result, "X and X");
    }

    #[test]
    fn expand_variables_no_goal_in_prompt() {
        let ctx = Context::new();
        let graph = make_graph("X");
        let result = expand_variables("no variable here", &graph, &ctx);
        assert_eq!(result, "no variable here");
    }

    #[test]
    fn expand_variables_does_not_expand_context_keys() {
        let ctx = Context::new();
        ctx.set(
            "human.gate.response",
            Value::Str("user said hello".to_string()),
        );
        let graph = make_graph("my goal");
        // Context keys must NOT be expanded — they reach LLMs via Preamble Transform only
        let result = expand_variables("Feedback: $human.gate.response", &graph, &ctx);
        assert_eq!(result, "Feedback: $human.gate.response");
    }

    // -----------------------------------------------------------------------
    // Preamble prepending tests (PREAMBLE-014)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn preamble_prepended_to_backend_call_not_prompt_md() {
        // prompt.md must contain the ORIGINAL prompt (no preamble) to avoid
        // O(n²) transcript inflation when build_full() reads it back.
        // The augmented prompt (with preamble) is sent to the backend only.
        use std::sync::Mutex;

        struct PromptCapturingBackend {
            captured: Arc<Mutex<Option<String>>>,
        }

        #[async_trait]
        impl CodergenBackend for PromptCapturingBackend {
            async fn run(
                &self,
                _node: &Node,
                prompt: &str,
                _context: &Context,
            ) -> Result<CodergenResult, EngineError> {
                *self.captured.lock().unwrap() = Some(prompt.to_string());
                Ok(CodergenResult::Text("done".to_string()))
            }
        }

        let dir = tempfile::tempdir().unwrap();
        let captured = Arc::new(Mutex::new(None::<String>));
        let backend = PromptCapturingBackend {
            captured: captured.clone(),
        };
        let handler = CodergenHandler::new(Some(Box::new(backend)));
        let node = make_node("step", "Do the work for: $goal");
        let ctx = Context::new();
        ctx.set(
            "_preamble",
            Value::Str("## Pipeline Context\nGoal: build a rocket".to_string()),
        );
        let graph = make_graph("build a rocket");
        handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();

        // prompt.md must contain the ORIGINAL prompt (no preamble).
        let prompt_text =
            std::fs::read_to_string(dir.path().join("step").join("prompt.md")).unwrap();
        assert_eq!(
            prompt_text, "Do the work for: build a rocket",
            "prompt.md must contain original prompt without preamble; got: {prompt_text}"
        );

        // The backend must receive the AUGMENTED prompt (with preamble).
        let backend_prompt = captured.lock().unwrap().clone().unwrap();
        assert!(
            backend_prompt.starts_with("## Pipeline Context"),
            "backend must receive preamble-augmented prompt; got: {backend_prompt}"
        );
        assert!(
            backend_prompt.contains("---"),
            "preamble and prompt must be separated by ---; got: {backend_prompt}"
        );
        assert!(
            backend_prompt.contains("Do the work for: build a rocket"),
            "original prompt must appear after separator; got: {backend_prompt}"
        );
    }

    #[tokio::test]
    async fn no_preamble_prompt_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let backend = MockTextBackend {
            response: "done".to_string(),
        };
        let handler = CodergenHandler::new(Some(Box::new(backend)));
        let node = make_node("step", "Do the work");
        let ctx = Context::new();
        // No _preamble set
        let graph = make_graph("goal");
        handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();

        let prompt_text =
            std::fs::read_to_string(dir.path().join("step").join("prompt.md")).unwrap();
        assert_eq!(
            prompt_text, "Do the work",
            "prompt must be unchanged when no _preamble is set"
        );
    }

    #[tokio::test]
    async fn empty_preamble_prompt_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let backend = MockTextBackend {
            response: "done".to_string(),
        };
        let handler = CodergenHandler::new(Some(Box::new(backend)));
        let node = make_node("step", "Do the work");
        let ctx = Context::new();
        ctx.set("_preamble", Value::Str(String::new()));
        let graph = make_graph("goal");
        handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();

        let prompt_text =
            std::fs::read_to_string(dir.path().join("step").join("prompt.md")).unwrap();
        assert_eq!(
            prompt_text, "Do the work",
            "prompt must be unchanged when _preamble is empty"
        );
    }
}
