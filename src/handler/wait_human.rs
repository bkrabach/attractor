//! Wait-for-human handler for `shape=hexagon` / `type="wait.human"` nodes.
//!
//! Derives choices from the node's outgoing edge labels, presents them to the
//! configured [`Interviewer`], and returns the selected label as
//! `preferred_label` in the [`Outcome`].

use async_trait::async_trait;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::fs;

use crate::error::EngineError;
use crate::graph::{Graph, Node, Value};
use crate::handler::Handler;
use crate::interviewer::{AnswerValue, Interviewer, Question, QuestionOption, QuestionType};
use crate::state::context::{Context, Outcome};

// ---------------------------------------------------------------------------
// WaitForHumanHandler
// ---------------------------------------------------------------------------

/// Handler for `shape=hexagon` (`type="wait.human"`) nodes.
///
/// Blocks pipeline execution until the [`Interviewer`] returns an answer,
/// then routes by setting `preferred_label` in the returned [`Outcome`].
pub struct WaitForHumanHandler {
    interviewer: Arc<dyn Interviewer>,
}

impl WaitForHumanHandler {
    /// Create a new handler with the given interviewer.
    pub fn new(interviewer: Arc<dyn Interviewer>) -> Self {
        WaitForHumanHandler { interviewer }
    }
}

#[async_trait]
impl Handler for WaitForHumanHandler {
    async fn execute(
        &self,
        node: &Node,
        context: &Context,
        graph: &Graph,
        logs_root: &Path,
    ) -> Result<Outcome, EngineError> {
        // 1. Derive choices from outgoing edges.
        let edges = graph.outgoing_edges(&node.id);
        if edges.is_empty() {
            return Ok(Outcome::fail("No outgoing edges for human gate"));
        }

        // Build question metadata from context so the interviewer can display
        // relevant LLM output before prompting the human.
        let metadata = {
            let mut m = HashMap::new();
            let llm_context = context
                .get("last_response")
                .map(|v| v.to_string_repr())
                .unwrap_or_default();
            if !llm_context.is_empty() {
                m.insert("last_response".to_string(), llm_context);
            }
            if let Some(prev) = context.get("human.gate.response") {
                m.insert("previous_response".to_string(), prev.to_string_repr());
            }
            // ATR-BUG-005: Prefer _last_codergen_stage (set by the engine for the
            // most recent codergen/LLM node) over last_stage so that tool/conditional
            // nodes between the LLM and the human gate don't hide the LLM response.
            let codergen_stage = context
                .get("_last_codergen_stage")
                .or_else(|| context.get("last_stage"));
            if let Some(stage_val) = codergen_stage {
                let stage_name = stage_val.to_string_repr();
                // Expose the resolved codergen node ID so the UI can fetch its
                // response directly (instead of using the last completed node).
                m.insert("last_codergen_node".to_string(), stage_name.clone());
                let response_md = logs_root.join(&stage_name).join("response.md");
                if response_md.exists() {
                    m.insert(
                        "response_md_path".to_string(),
                        response_md.to_string_lossy().to_string(),
                    );
                }
            }
            m
        };

        // 2. Check for freeform mode (SRV-BUG-004).
        //
        // Hexagon nodes with `mode="freeform"` present a free-text textarea
        // instead of a checkbox list.  The user's response is stored in
        // context as `human.gate.response` and the pipeline routes via the
        // first outgoing edge.
        let is_freeform = matches!(
            node.extra.get("mode"),
            Some(Value::Str(s)) if s == "freeform"
        );

        if is_freeform {
            let question_text = if !node.label.is_empty() {
                node.label.clone()
            } else {
                "Enter your response:".to_string()
            };

            let question = Question {
                text: question_text,
                question_type: QuestionType::FreeText,
                options: vec![],
                default: None,
                timeout: node.timeout,
                stage: node.id.clone(),
                metadata: metadata.clone(),
            };

            let answer = self.interviewer.ask(question).await;

            match &answer.value {
                AnswerValue::Timeout => {
                    if let Some(Value::Str(default_choice)) = node.extra.get("human.default_choice")
                    {
                        let preferred = default_choice.clone();
                        let outcome = build_freeform_outcome(&preferred, &edges, "");
                        write_response_md(logs_root, &node.id, "[timeout — default choice]")
                            .await?;
                        write_status(logs_root, &node.id, &outcome).await?;
                        return Ok(outcome);
                    }
                    return Ok(Outcome::retry("human gate timeout, no default"));
                }
                AnswerValue::Skipped => {
                    return Ok(Outcome::fail("human skipped interaction"));
                }
                _ => {}
            }

            let response_text = answer.text.clone();

            // Route via the first outgoing edge label.
            let route_label = edges
                .first()
                .map(|e| {
                    if !e.label.is_empty() {
                        e.label.as_str()
                    } else {
                        e.to.as_str()
                    }
                })
                .unwrap_or("")
                .to_string();

            let outcome = build_freeform_outcome(&route_label, &edges, &response_text);
            write_response_md(logs_root, &node.id, &response_text).await?;
            write_status(logs_root, &node.id, &outcome).await?;
            return Ok(outcome);
        }

        // 3. Standard mode: derive choices from outgoing edges.
        let mut options: Vec<QuestionOption> = Vec::new();
        for edge in &edges {
            let display_label = if !edge.label.is_empty() {
                edge.label.clone()
            } else {
                edge.to.clone()
            };
            let (key, _stripped) = parse_accelerator(&display_label);
            options.push(QuestionOption {
                key,
                label: display_label,
            });
        }

        // 4. Build question.
        let question_text = if !node.label.is_empty() {
            node.label.clone()
        } else {
            "Select an option:".to_string()
        };

        let question = Question {
            text: question_text,
            question_type: QuestionType::MultiSelect,
            options: options.clone(),
            default: None,
            timeout: node.timeout,
            stage: node.id.clone(),
            metadata,
        };

        // 3. Present to interviewer.
        let answer = self.interviewer.ask(question).await;

        // 4. Handle timeout and skip.
        match &answer.value {
            AnswerValue::Timeout => {
                // Check for a configured default choice.
                if let Some(Value::Str(default_choice)) = node.extra.get("human.default_choice") {
                    let preferred = default_choice.clone();
                    let outcome = build_outcome(&preferred, &options, &edges);
                    write_response_md(
                        logs_root,
                        &node.id,
                        &format!("[timeout — default: {preferred}]"),
                    )
                    .await?;
                    write_status(logs_root, &node.id, &outcome).await?;
                    return Ok(outcome);
                }
                return Ok(Outcome::retry("human gate timeout, no default"));
            }
            AnswerValue::Skipped => {
                return Ok(Outcome::fail("human skipped interaction"));
            }
            _ => {}
        }

        // 5. Determine selected label.
        let selected_label = if let Some(opt) = &answer.selected_option {
            opt.label.clone()
        } else if let AnswerValue::Selected(key) = &answer.value {
            // Find the option whose key matches.
            options
                .iter()
                .find(|o| o.key.eq_ignore_ascii_case(key))
                .map(|o| o.label.clone())
                .unwrap_or_else(|| answer.text.clone())
        } else {
            // Fallback to text or first option.
            if !answer.text.is_empty() {
                answer.text.clone()
            } else {
                options.first().map(|o| o.label.clone()).unwrap_or_default()
            }
        };

        // 6. Build outcome.
        let outcome = build_outcome(&selected_label, &options, &edges);
        write_response_md(logs_root, &node.id, &selected_label).await?;
        write_status(logs_root, &node.id, &outcome).await?;
        Ok(outcome)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build the [`Outcome`] for a freeform gate node.
///
/// Routes via the first outgoing edge whose label matches `route_label`, and
/// stores the human's free-text response in `human.gate.response`.
fn build_freeform_outcome(
    route_label: &str,
    edges: &[&crate::graph::Edge],
    response_text: &str,
) -> Outcome {
    let suggested: Vec<String> = edges
        .iter()
        .filter(|e| {
            let display = if !e.label.is_empty() {
                e.label.as_str()
            } else {
                e.to.as_str()
            };
            display == route_label
        })
        .map(|e| e.to.clone())
        .collect();

    let mut context_updates = HashMap::new();
    context_updates.insert(
        "human.gate.response".to_string(),
        Value::Str(response_text.to_string()),
    );
    context_updates.insert(
        "human.gate.label".to_string(),
        Value::Str(route_label.to_string()),
    );

    Outcome {
        preferred_label: route_label.to_string(),
        suggested_next_ids: suggested,
        context_updates,
        ..Outcome::success()
    }
}

/// Build the [`Outcome`] for the selected label.
fn build_outcome(
    selected_label: &str,
    options: &[QuestionOption],
    edges: &[&crate::graph::Edge],
) -> Outcome {
    // Find the key for the selected label.
    let selected_key = options
        .iter()
        .find(|o| o.label == selected_label)
        .map(|o| o.key.clone())
        .unwrap_or_default();

    // Find any matching edge targets.
    let suggested: Vec<String> = edges
        .iter()
        .filter(|e| {
            let display = if !e.label.is_empty() {
                e.label.as_str()
            } else {
                e.to.as_str()
            };
            display == selected_label
        })
        .map(|e| e.to.clone())
        .collect();

    let mut context_updates = HashMap::new();
    context_updates.insert(
        "human.gate.selected".to_string(),
        Value::Str(selected_key.clone()),
    );
    context_updates.insert(
        "human.gate.label".to_string(),
        Value::Str(selected_label.to_string()),
    );

    Outcome {
        preferred_label: selected_label.to_string(),
        suggested_next_ids: suggested,
        context_updates,
        ..Outcome::success()
    }
}

/// Write `response.md` to `{logs_root}/{node_id}/response.md`.
async fn write_response_md(
    logs_root: &Path,
    node_id: &str,
    response_text: &str,
) -> Result<(), EngineError> {
    let stage_dir = logs_root.join(node_id);
    fs::create_dir_all(&stage_dir).await?;
    fs::write(stage_dir.join("response.md"), response_text).await?;
    Ok(())
}

/// Write `status.json` to `{logs_root}/{node_id}/status.json`.
async fn write_status(
    logs_root: &Path,
    node_id: &str,
    outcome: &Outcome,
) -> Result<(), EngineError> {
    let stage_dir = logs_root.join(node_id);
    fs::create_dir_all(&stage_dir).await?;
    let json = serde_json::to_string_pretty(outcome).map_err(|e| EngineError::Handler {
        node_id: node_id.to_string(),
        message: format!("failed to serialise outcome: {e}"),
    })?;
    fs::write(stage_dir.join("status.json"), json).await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Accelerator key parsing
// ---------------------------------------------------------------------------

/// Parse an accelerator key from an edge label.
///
/// Priority order:
/// 1. `[K] Label` → key = `K` (uppercased)
/// 2. `K) Label`  → key = `K` (uppercased; K is a single non-whitespace char)
/// 3. `K - Label` → key = `K` (uppercased; K is a single non-whitespace char)
/// 4. First non-whitespace character (uppercased)
///
/// Returns `(key, stripped_label)` where `stripped_label` is the label with
/// the accelerator prefix removed and leading whitespace trimmed.
pub fn parse_accelerator(label: &str) -> (String, String) {
    let label = label.trim();

    if label.is_empty() {
        return (String::new(), String::new());
    }

    // Pattern 1: [K] rest
    if label.starts_with('[') {
        let mut chars = label.chars().skip(1);
        if let Some(k) = chars.next() {
            let rest: String = chars.collect();
            if let Some(after_bracket) = rest.strip_prefix(']') {
                let stripped = after_bracket.trim_start().to_string();
                return (k.to_ascii_uppercase().to_string(), stripped);
            }
        }
    }

    let mut chars = label.chars();
    let first = match chars.next() {
        Some(c) => c,
        None => return (String::new(), String::new()),
    };

    // Pattern 2: K) rest   (single char followed by ')')
    {
        let rest: String = chars.clone().collect();
        if let Some(after_paren) = rest.strip_prefix(')') {
            let stripped = after_paren.trim_start().to_string();
            return (first.to_ascii_uppercase().to_string(), stripped);
        }
    }

    // Pattern 3: K - rest  (single char, optional spaces, '-', rest)
    {
        let rest: String = chars.clone().collect();
        let trimmed = rest.trim_start();
        if let Some(after_dash) = trimmed.strip_prefix('-') {
            let stripped = after_dash.trim_start().to_string();
            return (first.to_ascii_uppercase().to_string(), stripped);
        }
    }

    // Fallback: first non-whitespace char is the key; label unchanged.
    let key = first.to_ascii_uppercase().to_string();
    (key, label.to_string())
}

/// Normalise a label for comparison: lowercase + trim whitespace +
/// strip leading accelerator prefix.
pub fn normalize_label(label: &str) -> String {
    let (_key, stripped) = parse_accelerator(label);
    let candidate = if stripped.is_empty() {
        label.trim().to_string()
    } else {
        stripped
    };
    candidate.to_lowercase()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Edge, Graph, Node};
    use crate::interviewer::{
        Answer, AnswerValue, AutoApproveInterviewer, QuestionType, QueueInterviewer,
        RecordingInterviewer,
    };
    use crate::state::context::StageStatus;
    use std::sync::Arc;

    fn make_graph_with_edges(node_id: &str, labels: &[&str]) -> Graph {
        let mut g = Graph::new("test".into());
        let mut n = Node::default();
        n.id = node_id.to_string();
        n.label = "Choose:".to_string();
        n.shape = "hexagon".to_string();
        g.nodes.insert(node_id.to_string(), n);

        for (i, lbl) in labels.iter().enumerate() {
            let target = format!("target_{i}");
            let mut tn = Node::default();
            tn.id = target.clone();
            g.nodes.insert(target.clone(), tn);
            g.edges.push(Edge {
                from: node_id.to_string(),
                to: target,
                label: lbl.to_string(),
                ..Default::default()
            });
        }
        g
    }

    // -- parse_accelerator --

    #[test]
    fn parse_bracket_pattern() {
        let (key, stripped) = parse_accelerator("[Y] Yes");
        assert_eq!(key, "Y");
        assert_eq!(stripped, "Yes");
    }

    #[test]
    fn parse_paren_pattern() {
        let (key, stripped) = parse_accelerator("Y) Yes");
        assert_eq!(key, "Y");
        assert_eq!(stripped, "Yes");
    }

    #[test]
    fn parse_dash_pattern() {
        let (key, stripped) = parse_accelerator("Y - Yes");
        assert_eq!(key, "Y");
        assert_eq!(stripped, "Yes");
    }

    #[test]
    fn parse_fallback() {
        let (key, stripped) = parse_accelerator("Yes, deploy");
        assert_eq!(key, "Y");
        assert_eq!(stripped, "Yes, deploy");
    }

    #[test]
    fn parse_empty_label() {
        let (key, stripped) = parse_accelerator("");
        assert!(key.is_empty());
        assert!(stripped.is_empty());
    }

    #[test]
    fn parse_lowercase_key_uppercased() {
        let (key, _) = parse_accelerator("[a] approve");
        assert_eq!(key, "A");
    }

    // -- normalize_label --

    #[test]
    fn normalize_bracket_prefix() {
        assert_eq!(normalize_label("[A] Approve"), "approve");
    }

    #[test]
    fn normalize_plain_label() {
        assert_eq!(normalize_label("Yes"), "yes");
    }

    // -- WaitForHumanHandler --

    #[tokio::test]
    async fn auto_approve_selects_first_option() {
        let dir = tempfile::tempdir().unwrap();
        let iv = Arc::new(AutoApproveInterviewer);
        let handler = WaitForHumanHandler::new(iv);
        let graph = make_graph_with_edges("gate", &["[A] Approve", "[R] Reject"]);
        let node = graph.node("gate").unwrap().clone();
        let ctx = Context::new();
        let outcome = handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();
        assert_eq!(outcome.status, StageStatus::Success);
        assert_eq!(outcome.preferred_label, "[A] Approve");
    }

    #[tokio::test]
    async fn queue_interviewer_routes_correctly() {
        let dir = tempfile::tempdir().unwrap();
        let opt = QuestionOption {
            key: "R".to_string(),
            label: "[R] Reject".to_string(),
        };
        let answer = Answer::selected(opt);
        let iv = Arc::new(QueueInterviewer::new(vec![answer]));
        let handler = WaitForHumanHandler::new(iv);
        let graph = make_graph_with_edges("gate", &["[A] Approve", "[R] Reject"]);
        let node = graph.node("gate").unwrap().clone();
        let ctx = Context::new();
        let outcome = handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();
        assert_eq!(outcome.status, StageStatus::Success);
        assert_eq!(outcome.preferred_label, "[R] Reject");
    }

    #[tokio::test]
    async fn no_edges_returns_fail() {
        let dir = tempfile::tempdir().unwrap();
        let iv = Arc::new(AutoApproveInterviewer);
        let handler = WaitForHumanHandler::new(iv);
        let graph = make_graph_with_edges("gate", &[]);
        let node = graph.node("gate").unwrap().clone();
        let ctx = Context::new();
        let outcome = handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();
        assert_eq!(outcome.status, StageStatus::Fail);
    }

    #[tokio::test]
    async fn skipped_answer_returns_fail() {
        let dir = tempfile::tempdir().unwrap();
        let iv = Arc::new(QueueInterviewer::new(vec![Answer::skipped()]));
        let handler = WaitForHumanHandler::new(iv);
        let graph = make_graph_with_edges("gate", &["Approve"]);
        let node = graph.node("gate").unwrap().clone();
        let ctx = Context::new();
        let outcome = handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();
        assert_eq!(outcome.status, StageStatus::Fail);
    }

    #[tokio::test]
    async fn timeout_without_default_returns_retry() {
        let dir = tempfile::tempdir().unwrap();
        let iv = Arc::new(QueueInterviewer::new(vec![Answer::timeout()]));
        let handler = WaitForHumanHandler::new(iv);
        let graph = make_graph_with_edges("gate", &["Approve"]);
        let node = graph.node("gate").unwrap().clone();
        let ctx = Context::new();
        let outcome = handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();
        assert_eq!(outcome.status, StageStatus::Retry);
    }

    #[tokio::test]
    async fn status_json_written() {
        let dir = tempfile::tempdir().unwrap();
        let iv = Arc::new(AutoApproveInterviewer);
        let handler = WaitForHumanHandler::new(iv);
        let graph = make_graph_with_edges("gate", &["Approve"]);
        let node = graph.node("gate").unwrap().clone();
        let ctx = Context::new();
        handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();
        let path = dir.path().join("gate").join("status.json");
        assert!(path.exists());
    }

    // -- Freeform mode tests (SRV-BUG-004) --

    fn make_freeform_graph(node_id: &str) -> (Graph, Node) {
        let mut g = Graph::new("test".into());
        let mut node = Node::default();
        node.id = node_id.to_string();
        node.label = "Brainstorm with Human".to_string();
        node.shape = "hexagon".to_string();
        node.extra
            .insert("mode".to_string(), Value::Str("freeform".to_string()));
        g.nodes.insert(node_id.to_string(), node.clone());

        let mut target = Node::default();
        target.id = "next_node".to_string();
        g.nodes.insert("next_node".to_string(), target);

        g.edges.push(Edge {
            from: node_id.to_string(),
            to: "next_node".to_string(),
            label: "RefineUnderstanding".to_string(),
            ..Default::default()
        });
        (g, node)
    }

    #[tokio::test]
    async fn freeform_mode_asks_free_text_question() {
        // RED: current code always creates MultiSelect; this must fail before the fix.
        let dir = tempfile::tempdir().unwrap();
        let (g, node) = make_freeform_graph("brainstorm");

        let answer = Answer {
            value: AnswerValue::Selected("I want to build a task manager".to_string()),
            selected_option: None,
            text: "I want to build a task manager".to_string(),
        };
        let queue_iv = QueueInterviewer::new(vec![answer]);
        let recording_iv = Arc::new(RecordingInterviewer::new(queue_iv));
        let handler = WaitForHumanHandler::new(recording_iv.clone());

        let ctx = Context::new();
        let _ = handler.execute(&node, &ctx, &g, dir.path()).await.unwrap();

        let recordings = recording_iv.recordings();
        assert_eq!(recordings.len(), 1);
        assert_eq!(
            recordings[0].question.question_type,
            QuestionType::FreeText,
            "freeform mode must ask a FreeText question, not {:?}",
            recordings[0].question.question_type
        );
    }

    #[tokio::test]
    async fn freeform_mode_stores_response_in_context() {
        // RED: current code does not set human.gate.response; this must fail before fix.
        let dir = tempfile::tempdir().unwrap();
        let (g, node) = make_freeform_graph("brainstorm");

        let answer = Answer {
            value: AnswerValue::Selected("I want to build a task manager".to_string()),
            selected_option: None,
            text: "I want to build a task manager".to_string(),
        };
        let iv = Arc::new(QueueInterviewer::new(vec![answer]));
        let handler = WaitForHumanHandler::new(iv);

        let ctx = Context::new();
        let outcome = handler.execute(&node, &ctx, &g, dir.path()).await.unwrap();

        assert_eq!(outcome.status, StageStatus::Success);

        assert!(
            outcome.context_updates.contains_key("human.gate.response"),
            "freeform mode must store response in context as human.gate.response"
        );
        let response = outcome.context_updates.get("human.gate.response").unwrap();
        assert_eq!(
            response,
            &Value::Str("I want to build a task manager".to_string()),
            "human.gate.response must contain the user's free text"
        );

        assert!(
            outcome
                .suggested_next_ids
                .contains(&"next_node".to_string()),
            "freeform mode must route to the outgoing edge target"
        );
    }

    #[tokio::test]
    async fn wait_human_includes_last_response_in_metadata() {
        // RED: context parameter is currently ignored; metadata will be empty.
        let dir = tempfile::tempdir().unwrap();
        let (g, node) = make_freeform_graph("brainstorm");

        let answer = Answer {
            value: AnswerValue::Selected("user response".to_string()),
            selected_option: None,
            text: "user response".to_string(),
        };
        let queue_iv = QueueInterviewer::new(vec![answer]);
        let recording_iv = Arc::new(RecordingInterviewer::new(queue_iv));
        let handler = WaitForHumanHandler::new(recording_iv.clone());

        let ctx = Context::new();
        ctx.set("last_response", Value::Str("LLM output text".to_string()));

        let _ = handler.execute(&node, &ctx, &g, dir.path()).await.unwrap();

        let recordings = recording_iv.recordings();
        assert_eq!(recordings.len(), 1);
        assert!(
            recordings[0]
                .question
                .metadata
                .contains_key("last_response"),
            "question metadata must contain last_response key"
        );
        assert_eq!(
            recordings[0]
                .question
                .metadata
                .get("last_response")
                .unwrap(),
            "LLM output text",
            "question metadata last_response must match context value"
        );
    }

    #[tokio::test]
    async fn wait_human_includes_previous_response_in_metadata() {
        // RED: previous human.gate.response must also appear in metadata.
        let dir = tempfile::tempdir().unwrap();
        let (g, node) = make_freeform_graph("brainstorm");

        let answer = Answer {
            value: AnswerValue::Selected("user response".to_string()),
            selected_option: None,
            text: "user response".to_string(),
        };
        let queue_iv = QueueInterviewer::new(vec![answer]);
        let recording_iv = Arc::new(RecordingInterviewer::new(queue_iv));
        let handler = WaitForHumanHandler::new(recording_iv.clone());

        let ctx = Context::new();
        ctx.set(
            "human.gate.response",
            Value::Str("Previous human reply".to_string()),
        );

        let _ = handler.execute(&node, &ctx, &g, dir.path()).await.unwrap();

        let recordings = recording_iv.recordings();
        assert_eq!(recordings.len(), 1);
        assert!(
            recordings[0]
                .question
                .metadata
                .contains_key("previous_response"),
            "question metadata must contain previous_response key when human.gate.response is set"
        );
        assert_eq!(
            recordings[0]
                .question
                .metadata
                .get("previous_response")
                .unwrap(),
            "Previous human reply",
        );
    }

    #[tokio::test]
    async fn context_updates_set_correctly() {
        let dir = tempfile::tempdir().unwrap();
        let iv = Arc::new(AutoApproveInterviewer);
        let handler = WaitForHumanHandler::new(iv);
        let graph = make_graph_with_edges("gate", &["[A] Approve"]);
        let node = graph.node("gate").unwrap().clone();
        let ctx = Context::new();
        let outcome = handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();
        assert!(outcome.context_updates.contains_key("human.gate.selected"));
        assert!(outcome.context_updates.contains_key("human.gate.label"));
    }

    #[tokio::test]
    async fn freeform_mode_writes_response_md() {
        let dir = tempfile::tempdir().unwrap();
        let (g, node) = make_freeform_graph("brainstorm");

        let answer = Answer {
            value: AnswerValue::Selected("My freeform response text".to_string()),
            selected_option: None,
            text: "My freeform response text".to_string(),
        };
        let iv = Arc::new(QueueInterviewer::new(vec![answer]));
        let handler = WaitForHumanHandler::new(iv);

        let ctx = Context::new();
        handler.execute(&node, &ctx, &g, dir.path()).await.unwrap();

        let path = dir.path().join("brainstorm").join("response.md");
        assert!(
            path.exists(),
            "response.md should be written for freeform mode"
        );
        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, "My freeform response text");
    }

    #[tokio::test]
    async fn standard_mode_writes_response_md() {
        let dir = tempfile::tempdir().unwrap();
        let iv = Arc::new(AutoApproveInterviewer);
        let handler = WaitForHumanHandler::new(iv);
        let graph = make_graph_with_edges("gate", &["[A] Approve", "[R] Reject"]);
        let node = graph.node("gate").unwrap().clone();
        let ctx = Context::new();
        handler
            .execute(&node, &ctx, &graph, dir.path())
            .await
            .unwrap();

        let path = dir.path().join("gate").join("response.md");
        assert!(
            path.exists(),
            "response.md should be written for standard mode"
        );
        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, "[A] Approve");
    }

    // -- response_md_path in metadata (task_6) --

    #[tokio::test]
    async fn wait_human_includes_response_md_path_when_last_stage_has_response_md() {
        // RED: metadata does not yet include response_md_path.
        let dir = tempfile::tempdir().unwrap();
        let (g, node) = make_freeform_graph("brainstorm");

        // Pre-create the response.md file for the last_stage node.
        let last_stage_dir = dir.path().join("prev_node");
        std::fs::create_dir_all(&last_stage_dir).unwrap();
        std::fs::write(
            last_stage_dir.join("response.md"),
            "Full LLM output from disk",
        )
        .unwrap();

        let answer = Answer {
            value: AnswerValue::Selected("user response".to_string()),
            selected_option: None,
            text: "user response".to_string(),
        };
        let queue_iv = QueueInterviewer::new(vec![answer]);
        let recording_iv = Arc::new(RecordingInterviewer::new(queue_iv));
        let handler = WaitForHumanHandler::new(recording_iv.clone());

        let ctx = Context::new();
        ctx.set("last_stage", Value::Str("prev_node".to_string()));

        let _ = handler.execute(&node, &ctx, &g, dir.path()).await.unwrap();

        let recordings = recording_iv.recordings();
        assert_eq!(recordings.len(), 1);
        assert!(
            recordings[0]
                .question
                .metadata
                .contains_key("response_md_path"),
            "question metadata must contain response_md_path when last_stage has a response.md"
        );
        let path_val = recordings[0]
            .question
            .metadata
            .get("response_md_path")
            .unwrap();
        assert!(
            path_val.ends_with("prev_node/response.md"),
            "response_md_path must point to logs_root/last_stage/response.md, got: {path_val}"
        );
    }

    #[tokio::test]
    async fn wait_human_omits_response_md_path_when_file_missing() {
        // When last_stage is set but response.md doesn't exist, path must NOT be added.
        let dir = tempfile::tempdir().unwrap();
        let (g, node) = make_freeform_graph("brainstorm");

        let answer = Answer {
            value: AnswerValue::Selected("user response".to_string()),
            selected_option: None,
            text: "user response".to_string(),
        };
        let queue_iv = QueueInterviewer::new(vec![answer]);
        let recording_iv = Arc::new(RecordingInterviewer::new(queue_iv));
        let handler = WaitForHumanHandler::new(recording_iv.clone());

        let ctx = Context::new();
        ctx.set("last_stage", Value::Str("nonexistent_node".to_string()));

        let _ = handler.execute(&node, &ctx, &g, dir.path()).await.unwrap();

        let recordings = recording_iv.recordings();
        assert_eq!(recordings.len(), 1);
        assert!(
            !recordings[0]
                .question
                .metadata
                .contains_key("response_md_path"),
            "question metadata must NOT contain response_md_path when file does not exist"
        );
    }

    // -- ATR-BUG-005: _last_codergen_stage preference --

    #[tokio::test]
    async fn wait_human_uses_last_codergen_stage_for_response_md_path() {
        // ATR-BUG-005: When _last_codergen_stage is set and points to a node
        // with response.md, the metadata should use that path even when
        // last_stage points to a different node (e.g. a tool node) that has
        // no response.md.
        let dir = tempfile::tempdir().unwrap();
        let (g, node) = make_freeform_graph("brainstorm");

        // Create response.md for the LLM node but NOT for the tool node.
        let llm_dir = dir.path().join("llm_node");
        std::fs::create_dir_all(&llm_dir).unwrap();
        std::fs::write(llm_dir.join("response.md"), "LLM generated output").unwrap();
        // tool_node directory intentionally does NOT have response.md.

        let answer = Answer {
            value: AnswerValue::Selected("user response".to_string()),
            selected_option: None,
            text: "user response".to_string(),
        };
        let queue_iv = QueueInterviewer::new(vec![answer]);
        let recording_iv = Arc::new(RecordingInterviewer::new(queue_iv));
        let handler = WaitForHumanHandler::new(recording_iv.clone());

        let ctx = Context::new();
        // _last_codergen_stage points to the LLM node (has response.md).
        ctx.set("_last_codergen_stage", Value::Str("llm_node".to_string()));
        // last_stage points to the tool node (no response.md).
        ctx.set("last_stage", Value::Str("tool_node".to_string()));

        let _ = handler.execute(&node, &ctx, &g, dir.path()).await.unwrap();

        let recordings = recording_iv.recordings();
        assert_eq!(recordings.len(), 1);

        // response_md_path must point to llm_node's response.md, NOT tool_node's.
        assert!(
            recordings[0]
                .question
                .metadata
                .contains_key("response_md_path"),
            "question metadata must contain response_md_path from _last_codergen_stage"
        );
        let path_val = recordings[0]
            .question
            .metadata
            .get("response_md_path")
            .unwrap();
        assert!(
            path_val.ends_with("llm_node/response.md"),
            "response_md_path must point to llm_node/response.md, got: {path_val}"
        );

        // last_codergen_node must be exposed in metadata.
        assert_eq!(
            recordings[0]
                .question
                .metadata
                .get("last_codergen_node")
                .unwrap(),
            "llm_node",
            "last_codergen_node metadata must equal _last_codergen_stage"
        );
    }
}
