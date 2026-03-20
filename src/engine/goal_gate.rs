//! Goal gate enforcement (NLSpec §3.4).
//!
//! Before the pipeline exits through a terminal node, all nodes with
//! `goal_gate=true` must have a SUCCESS or PARTIAL_SUCCESS outcome.
//! If any gate is unsatisfied the engine jumps to a retry target instead.

use crate::graph::Graph;
use crate::state::context::Outcome;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Check whether all `goal_gate=true` nodes have a satisfactory outcome.
///
/// Only nodes that appear in `node_outcomes` (i.e. nodes that were actually
/// executed) are checked.
///
/// Returns:
/// - `Ok(())` when all gate nodes passed (or no gate nodes were executed).
/// - `Err(gate_node_id)` for the first unsatisfied gate found.
pub fn check_goal_gates(
    graph: &Graph,
    node_outcomes: &HashMap<String, Outcome>,
) -> Result<(), String> {
    for (node_id, outcome) in node_outcomes {
        if let Some(node) = graph.node(node_id) {
            if node.goal_gate && !outcome.status.is_success() {
                return Err(node_id.clone());
            }
        }
    }
    Ok(())
}

/// Resolve the retry target for an unsatisfied goal gate node.
///
/// Resolution order (highest priority first):
/// 1. `node.retry_target` — if non-empty and the node exists in the graph.
/// 2. `node.fallback_retry_target`
/// 3. `graph.graph_attrs.retry_target`
/// 4. `graph.graph_attrs.fallback_retry_target`
/// 5. `None` — caller must fail the pipeline.
pub fn resolve_gate_retry_target<'a>(gate_node_id: &str, graph: &'a Graph) -> Option<&'a str> {
    let node = graph.node(gate_node_id)?;

    if !node.retry_target.is_empty() && graph.node(&node.retry_target).is_some() {
        return Some(&node.retry_target);
    }
    if !node.fallback_retry_target.is_empty() && graph.node(&node.fallback_retry_target).is_some() {
        return Some(&node.fallback_retry_target);
    }
    if !graph.graph_attrs.retry_target.is_empty()
        && graph.node(&graph.graph_attrs.retry_target).is_some()
    {
        return Some(&graph.graph_attrs.retry_target);
    }
    if !graph.graph_attrs.fallback_retry_target.is_empty()
        && graph
            .node(&graph.graph_attrs.fallback_retry_target)
            .is_some()
    {
        return Some(&graph.graph_attrs.fallback_retry_target);
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Graph, GraphAttrs, Node};
    use crate::state::context::StageStatus;

    fn make_gate_graph() -> Graph {
        let mut g = Graph::new("test".into());
        g.graph_attrs = GraphAttrs {
            retry_target: "fallback_node".to_string(),
            default_max_retry: 50,
            ..Default::default()
        };

        let gate = Node {
            id: "critical".to_string(),
            goal_gate: true,
            retry_target: "retry_node".to_string(),
            ..Default::default()
        };
        g.nodes.insert("critical".to_string(), gate);

        let retry = Node {
            id: "retry_node".to_string(),
            ..Default::default()
        };
        g.nodes.insert("retry_node".to_string(), retry);

        let fallback = Node {
            id: "fallback_node".to_string(),
            ..Default::default()
        };
        g.nodes.insert("fallback_node".to_string(), fallback);

        g
    }

    fn outcome(status: StageStatus) -> Outcome {
        Outcome {
            status,
            ..Default::default()
        }
    }

    // --- check_goal_gates ---

    #[test]
    fn no_gate_nodes_passes() {
        let mut g = Graph::new("test".into());
        let n = Node {
            id: "A".to_string(),
            goal_gate: false,
            ..Default::default()
        };
        g.nodes.insert("A".to_string(), n);

        let mut outcomes = HashMap::new();
        outcomes.insert("A".to_string(), outcome(StageStatus::Fail));

        assert!(check_goal_gates(&g, &outcomes).is_ok());
    }

    #[test]
    fn gate_success_passes() {
        let g = make_gate_graph();
        let mut outcomes = HashMap::new();
        outcomes.insert("critical".to_string(), outcome(StageStatus::Success));
        assert!(check_goal_gates(&g, &outcomes).is_ok());
    }

    #[test]
    fn gate_partial_success_passes() {
        let g = make_gate_graph();
        let mut outcomes = HashMap::new();
        outcomes.insert("critical".to_string(), outcome(StageStatus::PartialSuccess));
        assert!(check_goal_gates(&g, &outcomes).is_ok());
    }

    #[test]
    fn gate_fail_returns_err() {
        let g = make_gate_graph();
        let mut outcomes = HashMap::new();
        outcomes.insert("critical".to_string(), outcome(StageStatus::Fail));
        let result = check_goal_gates(&g, &outcomes);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "critical");
    }

    #[test]
    fn gate_retry_returns_err() {
        let g = make_gate_graph();
        let mut outcomes = HashMap::new();
        outcomes.insert("critical".to_string(), outcome(StageStatus::Retry));
        assert!(check_goal_gates(&g, &outcomes).is_err());
    }

    #[test]
    fn non_gate_fail_ignored() {
        let mut g = make_gate_graph();
        // Add a non-gate node
        let other = Node {
            id: "other".to_string(),
            goal_gate: false,
            ..Default::default()
        };
        g.nodes.insert("other".to_string(), other);

        let mut outcomes = HashMap::new();
        outcomes.insert("critical".to_string(), outcome(StageStatus::Success));
        outcomes.insert("other".to_string(), outcome(StageStatus::Fail));

        assert!(check_goal_gates(&g, &outcomes).is_ok());
    }

    #[test]
    fn node_not_in_outcomes_not_checked() {
        let g = make_gate_graph();
        // "critical" is a gate node but has no outcome recorded → not checked
        let outcomes: HashMap<String, Outcome> = HashMap::new();
        assert!(check_goal_gates(&g, &outcomes).is_ok());
    }

    // --- resolve_gate_retry_target ---

    #[test]
    fn node_retry_target_first() {
        let g = make_gate_graph();
        let target = resolve_gate_retry_target("critical", &g);
        assert_eq!(target, Some("retry_node"));
    }

    #[test]
    fn fallback_to_graph_retry_target() {
        let mut g = make_gate_graph();
        // Remove node-level retry_target
        g.nodes.get_mut("critical").unwrap().retry_target = String::new();
        g.nodes.get_mut("critical").unwrap().fallback_retry_target = String::new();

        let target = resolve_gate_retry_target("critical", &g);
        assert_eq!(target, Some("fallback_node"));
    }

    #[test]
    fn no_valid_target_returns_none() {
        let mut g = Graph::new("test".into());
        let gate = Node {
            id: "gate".to_string(),
            goal_gate: true,
            ..Default::default()
        };
        g.nodes.insert("gate".to_string(), gate);

        let target = resolve_gate_retry_target("gate", &g);
        assert!(target.is_none());
    }

    #[test]
    fn invalid_retry_target_skipped() {
        let mut g = make_gate_graph();
        // Point node retry_target to a non-existent node
        g.nodes.get_mut("critical").unwrap().retry_target = "nonexistent".to_string();

        // Should fall through to graph retry_target
        let target = resolve_gate_retry_target("critical", &g);
        assert_eq!(target, Some("fallback_node"));
    }
}
