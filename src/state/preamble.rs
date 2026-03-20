//! Preamble builder: synthesises context carryover text based on fidelity mode.

use crate::state::{Context, FidelityMode};
use std::path::Path;

/// Find the smallest byte index >= `idx` that is a valid UTF-8 char boundary.
///
/// Equivalent to the unstable `str::ceil_char_boundary` (rust-lang/rust#93743).
fn ceil_char_boundary(s: &str, idx: usize) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    let mut i = idx;
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// Maximum total bytes for `full` mode before front-truncation.
///
/// We use byte length (not char count) because `String::len()` returns bytes.
/// For typical UTF-8 LLM output, 100KB of bytes is close to 100K characters.
const FULL_MODE_MAX_BYTES: usize = 100_000;

/// Build the preamble string for the next LLM session.
///
/// Matches on `mode` and routes to the appropriate builder function.
pub fn build_preamble(
    mode: &FidelityMode,
    _thread_key: &str,
    context: &Context,
    completed_nodes: &[String],
    logs_root: &Path,
) -> String {
    match mode {
        FidelityMode::Truncate => build_truncate(context),
        FidelityMode::Compact => build_compact(context, completed_nodes),
        FidelityMode::Full => build_full(context, completed_nodes, logs_root),
        FidelityMode::SummaryLow => build_summary(context, completed_nodes, SummaryDetail::Low),
        FidelityMode::SummaryMedium => {
            build_summary(context, completed_nodes, SummaryDetail::Medium)
        }
        FidelityMode::SummaryHigh => build_summary(context, completed_nodes, SummaryDetail::High),
    }
}

/// Build the compact-mode preamble.
///
/// Produces a structured bullet-point summary of pipeline state:
/// - `## Pipeline Context` header
/// - `Goal: {goal}` if set
/// - `Completed stages: {comma-separated list}` if any
/// - `Human interactions:` section with ALL human gate responses from the
///   entire run (ATR-BUG-002: prevents context loss over many iterations)
/// - `Most recent human input: {value}` if `human.gate.response` is set
/// - `Previous output summary: {value}` if `last_response` is set
///
/// Sections are joined with newlines.
pub(crate) fn build_compact(context: &Context, completed_nodes: &[String]) -> String {
    let mut sections = vec!["## Pipeline Context".to_string()];

    let goal = context.get_string("graph.goal");
    if !goal.is_empty() {
        sections.push(format!("Goal: {goal}"));
    }

    if !completed_nodes.is_empty() {
        sections.push(format!("Completed stages: {}", completed_nodes.join(", ")));
    }

    // ATR-BUG-002: Include the FULL human interaction history, not just the
    // most recent `human.gate.response`.  The engine accumulates all human
    // gate interactions in `_human_interactions` (newline-separated).
    let human_history = context.get_string("_human_interactions");
    if !human_history.is_empty() {
        sections.push(format!("Human interactions:\n{human_history}"));
    } else {
        // Fallback: if no accumulated history, show the single most recent
        // response (backwards compatibility for pre-ATR-BUG-002 checkpoints).
        let human_input = context.get_string("human.gate.response");
        if !human_input.is_empty() {
            sections.push(format!("Human input: {human_input}"));
        }
    }

    let last_response = context.get_string("last_response");
    if !last_response.is_empty() {
        sections.push(format!("Previous output summary: {last_response}"));
    }

    sections.join("\n")
}

// ---------------------------------------------------------------------------
// Summary modes (PREAMBLE-009)
// ---------------------------------------------------------------------------

/// Detail level for summary modes.
enum SummaryDetail {
    Low,
    Medium,
    High,
}

/// `summary:*` modes — narrative summary with variable detail.
///
/// - **Low** (~600 tokens): goal, stage count, stage flow.
/// - **Medium** (~1500 tokens): adds recent outcomes, context values, human input.
/// - **High** (~3000 tokens): adds full context snapshot dump.
fn build_summary(context: &Context, completed_nodes: &[String], detail: SummaryDetail) -> String {
    let mut sections = Vec::new();
    let goal = context.get_string("graph.goal");

    // All levels include the goal and completed stages.
    if !goal.is_empty() {
        sections.push(format!("# Pipeline Summary\n\nGoal: {goal}"));
    }

    if !completed_nodes.is_empty() {
        sections.push(format!(
            "Completed {} stages: {}",
            completed_nodes.len(),
            completed_nodes.join(" → ")
        ));
    }

    // Medium and High include context values.
    if matches!(detail, SummaryDetail::Medium | SummaryDetail::High) {
        let last_response = context.get_string("last_response");
        if !last_response.is_empty() {
            sections.push(format!("Most recent output: {last_response}"));
        }

        // Include full human interaction history (ATR-BUG-002).
        let human_history = context.get_string("_human_interactions");
        if !human_history.is_empty() {
            sections.push(format!("Human interactions:\n{human_history}"));
        } else {
            // Fallback: show single most recent response when no history.
            let human_input = context.get_string("human.gate.response");
            if !human_input.is_empty() {
                sections.push(format!("Human input: {human_input}"));
            }
        }

        let outcome = context.get_string("outcome");
        if !outcome.is_empty() {
            sections.push(format!("Last outcome: {outcome}"));
        }

        let preferred = context.get_string("preferred_label");
        if !preferred.is_empty() {
            sections.push(format!("Routing decision: {preferred}"));
        }
    }

    // High includes full context snapshot dump.
    if matches!(detail, SummaryDetail::High) {
        let snapshot = context.snapshot();
        let mut context_pairs: Vec<String> = snapshot
            .iter()
            .filter(|(k, _)| {
                // Skip keys already covered above and internal keys.
                !matches!(
                    k.as_str(),
                    "graph.goal"
                        | "graph.name"
                        | "last_response"
                        | "human.gate.response"
                        | "outcome"
                        | "preferred_label"
                        | "current_node"
                        | "_preamble"
                        | "_working_dir"
                        | "_human_interactions"
                )
            })
            .map(|(k, v)| format!("  {k}: {}", v.to_string_repr()))
            .collect();
        context_pairs.sort();
        if !context_pairs.is_empty() {
            sections.push(format!("Context state:\n{}", context_pairs.join("\n")));
        }
    }

    sections.join("\n\n")
}

/// Build the truncate-mode preamble.
///
/// Returns `"Pipeline goal: {goal}"` when a goal is present in context,
/// or an empty string when no goal is set.
pub(crate) fn build_truncate(context: &Context) -> String {
    let goal = context.get_string("graph.goal");
    if goal.is_empty() {
        String::new()
    } else {
        format!("Pipeline goal: {goal}")
    }
}

/// `full` mode — filesystem replay of the complete conversation.
///
/// Reads `prompt.md` and `response.md` from each completed node's stage
/// directory and concatenates them as a conversation transcript.
///
/// If the total exceeds [`FULL_MODE_MAX_BYTES`], content is truncated from
/// the front, keeping the most recent exchanges.
fn build_full(context: &Context, completed_nodes: &[String], logs_root: &Path) -> String {
    let mut transcript = Vec::new();

    // Prepend the goal for orientation.
    let goal = context.get_string("graph.goal");
    if !goal.is_empty() {
        transcript.push(format!("Pipeline goal: {goal}\n"));
    }

    for node_id in completed_nodes {
        let node_dir = logs_root.join(node_id);

        let prompt = std::fs::read_to_string(node_dir.join("prompt.md")).ok();
        let response = std::fs::read_to_string(node_dir.join("response.md")).ok();

        // Skip nodes with no artifacts on disk.
        if prompt.is_none() && response.is_none() {
            continue;
        }

        let mut section = format!("## {node_id}\n");
        if let Some(p) = prompt {
            section.push_str(&format!("\n### Prompt\n{p}\n"));
        }
        if let Some(r) = response {
            section.push_str(&format!("\n### Response\n{r}\n"));
        }

        transcript.push(section);
    }

    let full_text = transcript.join("\n");

    // Truncate from front if too long.
    if full_text.len() > FULL_MODE_MAX_BYTES {
        let target_start = full_text.len() - FULL_MODE_MAX_BYTES;
        // Walk forward to a char boundary (safety for multi-byte UTF-8).
        let safe_start = ceil_char_boundary(&full_text, target_start);
        // Then find a clean line break point.
        let break_point = full_text[safe_start..]
            .find('\n')
            .map(|i| safe_start + i + 1)
            .unwrap_or(safe_start);
        format!(
            "[... earlier context truncated ...]\n{}",
            &full_text[break_point..]
        )
    } else {
        full_text
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Value;
    use std::path::Path;

    #[test]
    fn truncate_mode_returns_goal() {
        let ctx = Context::new();
        ctx.set("graph.goal", Value::Str("Build the feature".to_string()));
        let result = build_truncate(&ctx);
        assert_eq!(result, "Pipeline goal: Build the feature");
    }

    #[test]
    fn truncate_mode_empty_goal_returns_empty() {
        let ctx = Context::new();
        // No goal set — expect empty string
        let result = build_truncate(&ctx);
        assert_eq!(result, "");
    }

    #[test]
    fn build_preamble_truncate_routes_correctly() {
        let ctx = Context::new();
        ctx.set("graph.goal", Value::Str("End-to-end routing".to_string()));
        let result = build_preamble(&FidelityMode::Truncate, "", &ctx, &[], Path::new(""));
        assert_eq!(result, "Pipeline goal: End-to-end routing");
    }

    #[test]
    fn compact_mode_includes_completed_stages() {
        let ctx = Context::new();
        ctx.set("graph.goal", Value::Str("Fix the bug".to_string()));
        ctx.set("human.gate.response", Value::Str("approved".to_string()));
        ctx.set(
            "last_response",
            Value::Str("The fix was applied.".to_string()),
        );
        let completed = vec!["stage-a".to_string(), "stage-b".to_string()];
        let result = build_preamble(&FidelityMode::Compact, "", &ctx, &completed, Path::new(""));
        assert!(
            result.contains("## Pipeline Context"),
            "missing header; got: {result}"
        );
        assert!(
            result.contains("stage-a") && result.contains("stage-b"),
            "missing completed stages; got: {result}"
        );
        assert!(
            result.contains("approved"),
            "missing human input; got: {result}"
        );
        assert!(
            result.contains("The fix was applied."),
            "missing previous output summary; got: {result}"
        );
    }

    /// ATR-BUG-002: compact preamble must include the FULL human interaction
    /// history, not just the most recent `human.gate.response`.
    #[test]
    fn compact_mode_includes_full_human_interaction_history() {
        let ctx = Context::new();
        ctx.set("graph.goal", Value::Str("Design feature".to_string()));

        // Simulate the engine's accumulation of human interactions across
        // multiple iterations (gate1, gate2, gate3).
        ctx.set(
            "_human_interactions",
            Value::Str(
                "[gate1] chose \"approve\": sounds good\n\
                 [gate2] responded: needs more detail\n\
                 [gate3] chose \"merge\""
                    .to_string(),
            ),
        );
        // Most recent (overwrites previous):
        ctx.set(
            "human.gate.response",
            Value::Str("needs more detail".to_string()),
        );

        let completed = vec![
            "gate1".to_string(),
            "gate2".to_string(),
            "gate3".to_string(),
        ];
        let result = build_preamble(&FidelityMode::Compact, "", &ctx, &completed, Path::new(""));

        // Must include ALL historical interactions, not just the latest.
        assert!(
            result.contains("[gate1]"),
            "must include gate1 interaction; got: {result}"
        );
        assert!(
            result.contains("sounds good"),
            "must include gate1 response text; got: {result}"
        );
        assert!(
            result.contains("[gate2]"),
            "must include gate2 interaction; got: {result}"
        );
        assert!(
            result.contains("[gate3]"),
            "must include gate3 interaction; got: {result}"
        );
        assert!(
            result.contains("Human interactions:"),
            "must have human interactions header; got: {result}"
        );
    }

    /// ATR-BUG-002: When no human interactions exist, compact mode must NOT
    /// emit an empty "Human interactions:" section.
    #[test]
    fn compact_mode_no_human_interactions_section_when_empty() {
        let ctx = Context::new();
        ctx.set("graph.goal", Value::Str("Automated run".to_string()));

        let completed = vec!["step1".to_string()];
        let result = build_preamble(&FidelityMode::Compact, "", &ctx, &completed, Path::new(""));

        assert!(
            !result.contains("Human interactions:"),
            "must not emit empty human interactions section; got: {result}"
        );
    }

    // -----------------------------------------------------------------------
    // Summary mode tests (PREAMBLE-009)
    // -----------------------------------------------------------------------

    #[test]
    fn summary_low_is_brief() {
        let ctx = Context::new();
        ctx.set("graph.goal", Value::Str("build a rocket".to_string()));
        ctx.set("last_response", Value::Str("step done".to_string()));
        let completed = vec!["Start".to_string(), "Plan".to_string()];
        let result = build_preamble(
            &FidelityMode::SummaryLow,
            "",
            &ctx,
            &completed,
            Path::new("/tmp"),
        );
        assert!(result.contains("build a rocket"), "must include goal");
        assert!(
            result.contains("2 stages"),
            "must include stage count; got: {result}"
        );
        assert!(
            result.len() < 3000,
            "summary:low should be brief, got {} chars",
            result.len()
        );
        // Low should NOT include context values like last_response.
        assert!(
            !result.contains("step done"),
            "summary:low must NOT include last_response; got: {result}"
        );
    }

    #[test]
    fn summary_medium_includes_context_values() {
        let ctx = Context::new();
        ctx.set("graph.goal", Value::Str("build a rocket".to_string()));
        ctx.set("last_response", Value::Str("step done".to_string()));
        ctx.set("outcome", Value::Str("success".to_string()));
        ctx.set("human.gate.response", Value::Str("looks good".to_string()));
        let completed = vec!["Start".to_string(), "Plan".to_string()];
        let result = build_preamble(
            &FidelityMode::SummaryMedium,
            "",
            &ctx,
            &completed,
            Path::new("/tmp"),
        );
        assert!(result.contains("build a rocket"), "must include goal");
        assert!(
            result.contains("step done"),
            "summary:medium must include last_response; got: {result}"
        );
        assert!(
            result.contains("success"),
            "summary:medium must include outcome; got: {result}"
        );
        assert!(
            result.contains("looks good"),
            "summary:medium must include human input; got: {result}"
        );
    }

    #[test]
    fn summary_high_includes_more_detail() {
        let ctx = Context::new();
        ctx.set("graph.goal", Value::Str("build a rocket".to_string()));
        ctx.set("last_response", Value::Str("step done".to_string()));
        ctx.set("outcome", Value::Str("success".to_string()));
        ctx.set("preferred_label", Value::Str("continue".to_string()));
        ctx.set("tool.output", Value::Str("grep result".to_string()));
        let completed = vec!["Start".to_string(), "Plan".to_string()];
        let result_low = build_preamble(
            &FidelityMode::SummaryLow,
            "",
            &ctx,
            &completed,
            Path::new("/tmp"),
        );
        let result_high = build_preamble(
            &FidelityMode::SummaryHigh,
            "",
            &ctx,
            &completed,
            Path::new("/tmp"),
        );
        assert!(
            result_high.len() >= result_low.len(),
            "summary:high ({} chars) should be >= summary:low ({} chars)",
            result_high.len(),
            result_low.len()
        );
        // High includes context state dump with additional keys.
        assert!(
            result_high.contains("Context state:"),
            "summary:high must include context state dump; got: {result_high}"
        );
        assert!(
            result_high.contains("tool.output"),
            "summary:high must include tool.output in context dump; got: {result_high}"
        );
    }

    #[test]
    fn summary_high_excludes_internal_keys() {
        let ctx = Context::new();
        ctx.set("graph.goal", Value::Str("test".to_string()));
        ctx.set("_preamble", Value::Str("should be hidden".to_string()));
        ctx.set("_working_dir", Value::Str("/tmp/should-hide".to_string()));
        ctx.set("visible_key", Value::Str("visible".to_string()));
        let completed = vec!["Step".to_string()];
        let result = build_preamble(
            &FidelityMode::SummaryHigh,
            "",
            &ctx,
            &completed,
            Path::new("/tmp"),
        );
        assert!(
            !result.contains("should be hidden"),
            "_preamble must be filtered; got: {result}"
        );
        assert!(
            !result.contains("/tmp/should-hide"),
            "_working_dir must be filtered; got: {result}"
        );
        assert!(
            result.contains("visible"),
            "non-internal keys must be visible; got: {result}"
        );
    }

    // -----------------------------------------------------------------------
    // Full mode tests (PREAMBLE-010)
    // -----------------------------------------------------------------------

    #[test]
    fn full_mode_replays_from_filesystem() {
        let dir = tempfile::tempdir().unwrap();

        // Create two node directories with prompt/response files.
        let node_a_dir = dir.path().join("NodeA");
        std::fs::create_dir_all(&node_a_dir).unwrap();
        std::fs::write(node_a_dir.join("prompt.md"), "What should we build?").unwrap();
        std::fs::write(node_a_dir.join("response.md"), "Let's build a rocket.").unwrap();

        let node_b_dir = dir.path().join("NodeB");
        std::fs::create_dir_all(&node_b_dir).unwrap();
        std::fs::write(node_b_dir.join("prompt.md"), "How should we build it?").unwrap();
        std::fs::write(node_b_dir.join("response.md"), "Start with the engine.").unwrap();

        let ctx = Context::new();
        ctx.set("graph.goal", Value::Str("Build a rocket".to_string()));
        let completed = vec!["NodeA".to_string(), "NodeB".to_string()];

        let result = build_preamble(&FidelityMode::Full, "", &ctx, &completed, dir.path());

        assert!(
            result.contains("What should we build?"),
            "must include NodeA prompt"
        );
        assert!(
            result.contains("Let's build a rocket."),
            "must include NodeA response"
        );
        assert!(
            result.contains("How should we build it?"),
            "must include NodeB prompt"
        );
        assert!(
            result.contains("Start with the engine."),
            "must include NodeB response"
        );
    }

    #[test]
    fn full_mode_skips_missing_files() {
        let dir = tempfile::tempdir().unwrap();

        // Only NodeA has files; NodeB directory doesn't exist.
        let node_a_dir = dir.path().join("NodeA");
        std::fs::create_dir_all(&node_a_dir).unwrap();
        std::fs::write(node_a_dir.join("prompt.md"), "Prompt A").unwrap();
        std::fs::write(node_a_dir.join("response.md"), "Response A").unwrap();

        let ctx = Context::new();
        let completed = vec!["NodeA".to_string(), "NodeB".to_string()];

        let result = build_preamble(&FidelityMode::Full, "", &ctx, &completed, dir.path());

        assert!(result.contains("Prompt A"), "must include existing node");
        // Must not panic or error on missing NodeB.
    }

    #[test]
    fn full_mode_truncates_from_front_when_too_long() {
        let dir = tempfile::tempdir().unwrap();

        // Create a node with a very long response.
        let node_dir = dir.path().join("BigNode");
        std::fs::create_dir_all(&node_dir).unwrap();
        std::fs::write(node_dir.join("prompt.md"), "short prompt").unwrap();
        // 120KB response — exceeds FULL_MODE_MAX_BYTES (100K).
        std::fs::write(node_dir.join("response.md"), "X".repeat(120_000)).unwrap();

        let ctx = Context::new();
        let completed = vec!["BigNode".to_string()];

        let result = build_preamble(&FidelityMode::Full, "", &ctx, &completed, dir.path());

        assert!(
            result.len() <= 110_000,
            "full mode must truncate excessive content, got {} chars",
            result.len()
        );
    }

    #[test]
    fn full_mode_includes_goal() {
        let dir = tempfile::tempdir().unwrap();

        let ctx = Context::new();
        ctx.set("graph.goal", Value::Str("test goal".to_string()));

        let result = build_preamble(&FidelityMode::Full, "", &ctx, &[], dir.path());

        assert!(
            result.contains("Pipeline goal: test goal"),
            "full mode must include goal; got: {result}"
        );
    }

    #[test]
    fn full_mode_empty_when_no_nodes_no_goal() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = Context::new();

        let result = build_preamble(&FidelityMode::Full, "", &ctx, &[], dir.path());

        assert!(
            result.is_empty(),
            "full mode with no nodes and no goal must be empty; got: {result}"
        );
    }
}
