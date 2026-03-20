//! Edge selection algorithm (NLSpec §3.3).
//!
//! Five-step deterministic priority:
//! 1. Condition-matching edges (evaluates edge `condition` expressions)
//! 2. Preferred label match (normalised: lowercase, trim, strip accelerator prefix)
//! 3. Suggested next IDs from the outcome
//! 4. Highest weight among unconditional edges
//! 5. Lexicographic tiebreak on target node ID

use crate::condition::evaluate_condition;
use crate::graph::{Edge, Graph, Value};
use crate::state::context::{Context, Outcome};
use serde_json::Value as JsonValue;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Select the best outgoing edge from `node_id` given `outcome` and `context`.
///
/// Returns `None` when there are no outgoing edges from the node.
pub fn select_edge(
    node_id: &str,
    outcome: &Outcome,
    context: &Context,
    graph: &Graph,
) -> Option<Edge> {
    let edges = graph.outgoing_edges(node_id);
    if edges.is_empty() {
        return None;
    }

    // Build a JSON context map for condition evaluation.
    let ctx_map = build_json_context(context);
    let outcome_status = outcome.status.as_str();
    let preferred_label = outcome.preferred_label.as_str();

    // -----------------------------------------------------------------------
    // Step 1: Condition-matching edges
    // -----------------------------------------------------------------------
    let condition_matched: Vec<&Edge> = edges
        .iter()
        .filter(|e| {
            if e.condition.is_empty() {
                return false;
            }
            evaluate_condition(&e.condition, outcome_status, preferred_label, &ctx_map)
                .unwrap_or(false)
        })
        .copied()
        .collect();

    if !condition_matched.is_empty() {
        return Some(best_by_weight_then_lexical(condition_matched).clone());
    }

    // -----------------------------------------------------------------------
    // Step 2: Preferred label match
    // -----------------------------------------------------------------------
    if !outcome.preferred_label.is_empty() {
        let norm_preferred = normalize_label(&outcome.preferred_label);
        for edge in &edges {
            if normalize_label(&edge.label) == norm_preferred {
                return Some((*edge).clone());
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 3: Suggested next IDs
    // -----------------------------------------------------------------------
    if !outcome.suggested_next_ids.is_empty() {
        for suggested_id in &outcome.suggested_next_ids {
            for edge in &edges {
                if &edge.to == suggested_id {
                    return Some((*edge).clone());
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 4 & 5: Highest weight + lexical tiebreak (unconditional edges)
    // -----------------------------------------------------------------------
    let unconditional: Vec<&Edge> = edges
        .iter()
        .filter(|e| e.condition.is_empty())
        .copied()
        .collect();

    if !unconditional.is_empty() {
        return Some(best_by_weight_then_lexical(unconditional).clone());
    }

    // Fallback: best among all edges (all have conditions, none matched — pick by weight/lexical)
    Some(best_by_weight_then_lexical(edges).clone())
}

/// Normalise a label for preferred-label matching.
///
/// Steps: trim → strip accelerator prefix → lowercase.
pub fn normalize_label(label: &str) -> String {
    let trimmed = label.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let stripped = strip_accelerator_prefix(trimmed);
    stripped.to_lowercase()
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Select the edge with the highest weight; break ties lexicographically by `to`.
fn best_by_weight_then_lexical(mut edges: Vec<&Edge>) -> &Edge {
    edges.sort_by(|a, b| {
        b.weight
            .cmp(&a.weight) // descending weight
            .then_with(|| a.to.cmp(&b.to)) // ascending target id
    });
    edges[0]
}

/// Strip leading accelerator prefix from `label` (already trimmed).
///
/// Patterns:
/// - `[K] rest` → `rest`
/// - `K) rest`  → `rest`
/// - `K - rest` → `rest` (K is a single alphanum char)
fn strip_accelerator_prefix(label: &str) -> &str {
    let chars: Vec<char> = label.chars().collect();
    let n = chars.len();

    // Pattern 1: [K] rest
    if n >= 4 && chars[0] == '[' {
        // Find closing ']'
        if let Some(close) = chars[1..].iter().position(|&c| c == ']') {
            let after = close + 2; // skip '[', chars[1..=close+1]
            // Skip one optional space after ']'
            let start = if after < n && chars[after] == ' ' {
                after + 1
            } else {
                after
            };
            if start <= n {
                return &label[chars[..start].iter().collect::<String>().len()..];
            }
        }
    }

    // Pattern 2: K) rest (single alphanum, then ')')
    if n >= 3 && chars[0].is_alphanumeric() && chars[1] == ')' {
        let rest = &label[2..];
        return rest.trim_start();
    }

    // Pattern 3: K - rest (single alphanum, space(s), '-', space(s))
    if n >= 4 && chars[0].is_alphanumeric() {
        let rest_raw = &label[chars[0].len_utf8()..];
        let rest_trimmed = rest_raw.trim_start();
        if let Some(after_dash) = rest_trimmed.strip_prefix('-') {
            return after_dash.trim_start();
        }
    }

    label
}

/// Build a `HashMap<String, JsonValue>` from the context for condition evaluation.
fn build_json_context(context: &Context) -> HashMap<String, JsonValue> {
    context
        .snapshot()
        .into_iter()
        .map(|(k, v)| {
            let jv = match v {
                Value::Str(s) => JsonValue::String(s),
                Value::Int(n) => JsonValue::Number(n.into()),
                Value::Float(f) => serde_json::Number::from_f64(f)
                    .map(JsonValue::Number)
                    .unwrap_or(JsonValue::Null),
                Value::Bool(b) => JsonValue::Bool(b),
                Value::Duration(d) => JsonValue::String(format!("{}ms", d.as_millis())),
            };
            (k, jv)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Graph, Node};
    use crate::state::context::StageStatus;

    fn make_graph_with_edges(from: &str, targets: Vec<(&str, &str, &str, i32)>) -> Graph {
        // targets: (to, label, condition, weight)
        let mut g = Graph::new("test".into());
        let src = Node {
            id: from.to_string(),
            ..Default::default()
        };
        g.nodes.insert(from.to_string(), src);
        for (to, lbl, cond, wt) in &targets {
            let n = Node {
                id: to.to_string(),
                ..Default::default()
            };
            g.nodes.insert(to.to_string(), n);
            g.edges.push(Edge {
                from: from.to_string(),
                to: to.to_string(),
                label: lbl.to_string(),
                condition: cond.to_string(),
                weight: *wt,
                ..Default::default()
            });
        }
        g
    }

    fn outcome_with(status: StageStatus, preferred: &str, suggested: Vec<&str>) -> Outcome {
        Outcome {
            status,
            preferred_label: preferred.to_string(),
            suggested_next_ids: suggested.into_iter().map(String::from).collect(),
            ..Default::default()
        }
    }

    // --- normalize_label ---

    #[test]
    fn normalize_plain() {
        assert_eq!(normalize_label("Yes"), "yes");
    }

    #[test]
    fn normalize_bracket_prefix() {
        assert_eq!(normalize_label("[Y] Yes"), "yes");
    }

    #[test]
    fn normalize_paren_prefix() {
        assert_eq!(normalize_label("Y) Yes"), "yes");
    }

    #[test]
    fn normalize_dash_prefix() {
        assert_eq!(normalize_label("Y - Yes"), "yes");
    }

    #[test]
    fn normalize_empty() {
        assert_eq!(normalize_label(""), "");
    }

    #[test]
    fn normalize_trim_whitespace() {
        assert_eq!(normalize_label("  Hello  "), "hello");
    }

    // --- select_edge: no edges ---

    #[test]
    fn no_edges_returns_none() {
        let graph = make_graph_with_edges("A", vec![]);
        let ctx = Context::new();
        let outcome = Outcome::success();
        assert!(select_edge("A", &outcome, &ctx, &graph).is_none());
    }

    // --- Step 1: Condition matching ---

    #[test]
    fn condition_match_wins_over_preferred_label() {
        let graph = make_graph_with_edges(
            "A",
            vec![
                ("B", "maybe", "outcome=success", 0),
                ("C", "success", "", 0), // preferred label would match "success"
            ],
        );
        let ctx = Context::new();
        // preferred_label="success" but condition on B matches → B wins
        let outcome = outcome_with(StageStatus::Success, "success", vec![]);
        let edge = select_edge("A", &outcome, &ctx, &graph).unwrap();
        assert_eq!(edge.to, "B");
    }

    #[test]
    fn condition_no_match_falls_to_step2() {
        let graph = make_graph_with_edges(
            "A",
            vec![
                ("B", "", "outcome=fail", 0), // condition won't match
                ("C", "yes", "", 0),          // preferred label match
            ],
        );
        let ctx = Context::new();
        let outcome = outcome_with(StageStatus::Success, "yes", vec![]);
        let edge = select_edge("A", &outcome, &ctx, &graph).unwrap();
        assert_eq!(edge.to, "C");
    }

    // --- Step 2: Preferred label ---

    #[test]
    fn preferred_label_exact() {
        let graph = make_graph_with_edges("A", vec![("B", "Yes", "", 0), ("C", "No", "", 0)]);
        let ctx = Context::new();
        let outcome = outcome_with(StageStatus::Success, "Yes", vec![]);
        let edge = select_edge("A", &outcome, &ctx, &graph).unwrap();
        assert_eq!(edge.to, "B");
    }

    #[test]
    fn preferred_label_normalized_match() {
        let graph = make_graph_with_edges("A", vec![("B", "[Y] Yes", "", 0), ("C", "No", "", 0)]);
        let ctx = Context::new();
        // preferred_label without accelerator
        let outcome = outcome_with(StageStatus::Success, "Yes", vec![]);
        let edge = select_edge("A", &outcome, &ctx, &graph).unwrap();
        assert_eq!(edge.to, "B");
    }

    // --- Step 3: Suggested next IDs ---

    #[test]
    fn suggested_ids_used_when_no_label() {
        let graph = make_graph_with_edges("A", vec![("B", "", "", 0), ("C", "", "", 0)]);
        let ctx = Context::new();
        let outcome = outcome_with(StageStatus::Success, "", vec!["C"]);
        let edge = select_edge("A", &outcome, &ctx, &graph).unwrap();
        assert_eq!(edge.to, "C");
    }

    // --- Step 4 & 5: Weight + lexical ---

    #[test]
    fn highest_weight_wins() {
        let graph = make_graph_with_edges(
            "A",
            vec![("B", "", "", 2), ("C", "", "", 5), ("D", "", "", 1)],
        );
        let ctx = Context::new();
        let outcome = Outcome::success();
        let edge = select_edge("A", &outcome, &ctx, &graph).unwrap();
        assert_eq!(edge.to, "C");
    }

    #[test]
    fn equal_weight_lexical_tiebreak() {
        let graph = make_graph_with_edges("A", vec![("Z", "", "", 0), ("A_target", "", "", 0)]);
        let ctx = Context::new();
        let outcome = Outcome::success();
        let edge = select_edge("A", &outcome, &ctx, &graph).unwrap();
        assert_eq!(edge.to, "A_target"); // "A_target" < "Z" lexicographically
    }

    // --- Context condition evaluation ---

    #[test]
    fn condition_uses_context() {
        let graph = make_graph_with_edges(
            "A",
            vec![("B", "", "context.flag=true", 0), ("C", "", "", 0)],
        );
        let ctx = Context::new();
        ctx.set("flag", Value::Str("true".to_string()));
        let outcome = Outcome::success();
        let edge = select_edge("A", &outcome, &ctx, &graph).unwrap();
        assert_eq!(edge.to, "B");
    }

    #[test]
    fn condition_unmatched_falls_to_unconditional() {
        let graph = make_graph_with_edges(
            "A",
            vec![("B", "", "context.flag=true", 5), ("C", "", "", 0)],
        );
        let ctx = Context::new();
        ctx.set("flag", Value::Str("false".to_string()));
        let outcome = Outcome::success();
        let edge = select_edge("A", &outcome, &ctx, &graph).unwrap();
        assert_eq!(edge.to, "C");
    }

    // --- Multiple condition matches: best_by_weight_then_lexical ---

    #[test]
    fn multiple_condition_matches_use_weight() {
        let graph = make_graph_with_edges(
            "A",
            vec![
                ("B", "", "outcome=success", 1),
                ("C", "", "outcome=success", 5),
            ],
        );
        let ctx = Context::new();
        let outcome = Outcome::success();
        let edge = select_edge("A", &outcome, &ctx, &graph).unwrap();
        assert_eq!(edge.to, "C");
    }
}
