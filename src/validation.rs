//! Graph validation and linting (NLSpec §7).
//!
//! ## Public API
//! - [`validate`] — run all built-in rules plus optional extra rules
//! - [`validate_or_raise`] — like `validate` but returns `Err` on any Error-severity diagnostic
//!
//! ## Built-In Rules
//! | Rule ID               | Severity | Check |
//! |-----------------------|----------|-------|
//! | `start_node`          | ERROR    | Exactly one Mdiamond node |
//! | `terminal_node`       | ERROR    | At least one Msquare node |
//! | `start_no_incoming`   | ERROR    | Start node has no incoming edges |
//! | `exit_no_outgoing`    | ERROR    | Exit node(s) have no outgoing edges |
//! | `reachability`        | WARNING  | All nodes reachable from start via BFS |
//! | `edge_target_exists`  | ERROR    | Edge from/to reference existing nodes |
//! | `condition_syntax`    | ERROR    | Edge condition strings parse correctly |
//! | `stylesheet_syntax`   | ERROR    | model_stylesheet parses correctly |
//! | `type_known`          | WARNING  | node_type is a recognised handler type |
//! | `fidelity_valid`      | WARNING  | fidelity values are in the valid set |
//! | `retry_target_exists` | WARNING  | retry targets reference existing nodes |
//! | `goal_gate_has_retry` | WARNING  | goal_gate nodes have a retry_target |
//! | `prompt_on_llm_nodes` | WARNING  | codergen nodes have a prompt or label |

use crate::condition::parse_condition;
use crate::error::ValidationError;
use crate::graph::Graph;
use crate::stylesheet::parse_stylesheet;
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Diagnostic model
// ---------------------------------------------------------------------------

/// A single validation diagnostic.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Diagnostic {
    /// Rule identifier (e.g., `"start_node"`).
    pub rule: String,
    /// Severity level.
    pub severity: Severity,
    /// Human-readable description of the issue.
    pub message: String,
    /// Relevant node ID, if applicable.
    pub node_id: Option<String>,
    /// Relevant edge as `(from, to)`, if applicable.
    pub edge: Option<(String, String)>,
    /// Suggested fix, if available.
    pub fix: Option<String>,
}

impl Diagnostic {
    fn error(rule: &str, message: impl Into<String>) -> Self {
        Diagnostic {
            rule: rule.to_string(),
            severity: Severity::Error,
            message: message.into(),
            node_id: None,
            edge: None,
            fix: None,
        }
    }

    fn warning(rule: &str, message: impl Into<String>) -> Self {
        Diagnostic {
            rule: rule.to_string(),
            severity: Severity::Warning,
            message: message.into(),
            node_id: None,
            edge: None,
            fix: None,
        }
    }

    fn with_node(mut self, id: impl Into<String>) -> Self {
        self.node_id = Some(id.into());
        self
    }

    fn with_edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.edge = Some((from.into(), to.into()));
        self
    }

    fn with_fix(mut self, fix: impl Into<String>) -> Self {
        self.fix = Some(fix.into());
        self
    }
}

/// Diagnostic severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    Info,
    Warning,
    Error,
}

// ---------------------------------------------------------------------------
// LintRule trait
// ---------------------------------------------------------------------------

/// Trait implemented by all lint rules (built-in and custom).
pub trait LintRule: Send + Sync {
    /// The rule's unique identifier (e.g., `"start_node"`).
    fn name(&self) -> &str;
    /// Apply the rule to `graph` and return any diagnostics.
    fn apply(&self, graph: &Graph) -> Vec<Diagnostic>;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run all built-in lint rules plus any `extra_rules` against `graph`.
///
/// Returns all diagnostics (both errors and warnings) in rule order.
pub fn validate(graph: &Graph, extra_rules: &[Box<dyn LintRule>]) -> Vec<Diagnostic> {
    let built_in: Vec<Box<dyn LintRule>> = vec![
        Box::new(StartNodeRule),
        Box::new(TerminalNodeRule),
        Box::new(StartNoIncomingRule),
        Box::new(ExitNoOutgoingRule),
        Box::new(ReachabilityRule),
        Box::new(EdgeTargetExistsRule),
        Box::new(ConditionSyntaxRule),
        Box::new(StylesheetSyntaxRule),
        Box::new(TypeKnownRule),
        Box::new(FidelityValidRule),
        Box::new(RetryTargetExistsRule),
        Box::new(GoalGateHasRetryRule),
        Box::new(PromptOnLlmNodesRule),
    ];

    let mut diagnostics = Vec::new();
    for rule in &built_in {
        diagnostics.extend(rule.apply(graph));
    }
    for rule in extra_rules {
        diagnostics.extend(rule.apply(graph));
    }
    diagnostics
}

/// Run validation and return `Err(ValidationError::Failed)` if any Error-severity
/// diagnostic exists.  Otherwise return `Ok` with the full diagnostic list.
pub fn validate_or_raise(
    graph: &Graph,
    extra_rules: &[Box<dyn LintRule>],
) -> Result<Vec<Diagnostic>, ValidationError> {
    let diagnostics = validate(graph, extra_rules);
    let error_count = diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .count();
    if error_count > 0 {
        Err(ValidationError::Failed { count: error_count })
    } else {
        Ok(diagnostics)
    }
}

// ---------------------------------------------------------------------------
// Built-in rule: start_node
// ---------------------------------------------------------------------------

struct StartNodeRule;
impl LintRule for StartNodeRule {
    fn name(&self) -> &str {
        "start_node"
    }
    fn apply(&self, graph: &Graph) -> Vec<Diagnostic> {
        let starts: Vec<&str> = graph
            .nodes
            .values()
            .filter(|n| n.shape == "Mdiamond")
            .map(|n| n.id.as_str())
            .collect();
        match starts.len() {
            0 => vec![
                Diagnostic::error("start_node", "pipeline has no start node (shape=Mdiamond)")
                    .with_fix("add a node with shape=Mdiamond"),
            ],
            1 => vec![],
            _ => vec![Diagnostic::error(
                "start_node",
                format!(
                    "pipeline has {} start nodes (shape=Mdiamond): {}",
                    starts.len(),
                    starts.join(", ")
                ),
            )],
        }
    }
}

// ---------------------------------------------------------------------------
// Built-in rule: terminal_node
// ---------------------------------------------------------------------------

struct TerminalNodeRule;
impl LintRule for TerminalNodeRule {
    fn name(&self) -> &str {
        "terminal_node"
    }
    fn apply(&self, graph: &Graph) -> Vec<Diagnostic> {
        let exits: Vec<&str> = graph
            .nodes
            .values()
            .filter(|n| n.shape == "Msquare")
            .map(|n| n.id.as_str())
            .collect();
        match exits.len() {
            0 => vec![
                Diagnostic::error("terminal_node", "pipeline has no exit node (shape=Msquare)")
                    .with_fix("add a node with shape=Msquare"),
            ],
            1 => vec![],
            _ => vec![Diagnostic::error(
                "terminal_node",
                format!(
                    "pipeline has {} exit nodes (shape=Msquare): {}; exactly one is required",
                    exits.len(),
                    exits.join(", ")
                ),
            )],
        }
    }
}

// ---------------------------------------------------------------------------
// Built-in rule: start_no_incoming
// ---------------------------------------------------------------------------

struct StartNoIncomingRule;
impl LintRule for StartNoIncomingRule {
    fn name(&self) -> &str {
        "start_no_incoming"
    }
    fn apply(&self, graph: &Graph) -> Vec<Diagnostic> {
        let mut diags = Vec::new();
        for node in graph.nodes.values().filter(|n| n.shape == "Mdiamond") {
            let incoming = graph.incoming_edges(&node.id);
            if !incoming.is_empty() {
                diags.push(
                    Diagnostic::error(
                        "start_no_incoming",
                        format!(
                            "start node '{}' has {} incoming edge(s); start nodes must have no incoming edges",
                            node.id,
                            incoming.len()
                        ),
                    )
                    .with_node(&node.id),
                );
            }
        }
        diags
    }
}

// ---------------------------------------------------------------------------
// Built-in rule: exit_no_outgoing
// ---------------------------------------------------------------------------

struct ExitNoOutgoingRule;
impl LintRule for ExitNoOutgoingRule {
    fn name(&self) -> &str {
        "exit_no_outgoing"
    }
    fn apply(&self, graph: &Graph) -> Vec<Diagnostic> {
        let mut diags = Vec::new();
        for node in graph.nodes.values().filter(|n| n.shape == "Msquare") {
            let outgoing = graph.outgoing_edges(&node.id);
            if !outgoing.is_empty() {
                diags.push(
                    Diagnostic::error(
                        "exit_no_outgoing",
                        format!(
                            "exit node '{}' has {} outgoing edge(s); exit nodes must have no outgoing edges",
                            node.id,
                            outgoing.len()
                        ),
                    )
                    .with_node(&node.id),
                );
            }
        }
        diags
    }
}

// ---------------------------------------------------------------------------
// Built-in rule: reachability
// ---------------------------------------------------------------------------

struct ReachabilityRule;
impl LintRule for ReachabilityRule {
    fn name(&self) -> &str {
        "reachability"
    }
    fn apply(&self, graph: &Graph) -> Vec<Diagnostic> {
        let start = match graph.start_node() {
            Some(n) => n.id.clone(),
            None => return vec![], // start_node rule covers this
        };

        // BFS from start
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        queue.push_back(start.clone());
        visited.insert(start);

        while let Some(current) = queue.pop_front() {
            for edge in graph.outgoing_edges(&current) {
                if !visited.contains(&edge.to) {
                    visited.insert(edge.to.clone());
                    queue.push_back(edge.to.clone());
                }
            }
        }

        graph
            .nodes
            .keys()
            .filter(|id| !visited.contains(*id))
            .map(|id| {
                Diagnostic::warning(
                    "reachability",
                    format!("node '{id}' is not reachable from the start node"),
                )
                .with_node(id)
                .with_fix(format!(
                    "add an edge leading to '{id}' from a reachable node"
                ))
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Built-in rule: edge_target_exists
// ---------------------------------------------------------------------------

struct EdgeTargetExistsRule;
impl LintRule for EdgeTargetExistsRule {
    fn name(&self) -> &str {
        "edge_target_exists"
    }
    fn apply(&self, graph: &Graph) -> Vec<Diagnostic> {
        let mut diags = Vec::new();
        for edge in &graph.edges {
            if !graph.nodes.contains_key(&edge.from) {
                diags.push(
                    Diagnostic::error(
                        "edge_target_exists",
                        format!("edge references unknown source node '{}'", edge.from),
                    )
                    .with_edge(&edge.from, &edge.to),
                );
            }
            if !graph.nodes.contains_key(&edge.to) {
                diags.push(
                    Diagnostic::error(
                        "edge_target_exists",
                        format!("edge references unknown target node '{}'", edge.to),
                    )
                    .with_edge(&edge.from, &edge.to),
                );
            }
        }
        diags
    }
}

// ---------------------------------------------------------------------------
// Built-in rule: condition_syntax
// ---------------------------------------------------------------------------

struct ConditionSyntaxRule;
impl LintRule for ConditionSyntaxRule {
    fn name(&self) -> &str {
        "condition_syntax"
    }
    fn apply(&self, graph: &Graph) -> Vec<Diagnostic> {
        let mut diags = Vec::new();
        for edge in &graph.edges {
            if edge.condition.is_empty() {
                continue;
            }
            if let Err(e) = parse_condition(&edge.condition) {
                diags.push(
                    Diagnostic::error(
                        "condition_syntax",
                        format!(
                            "edge '{}'->'{}: invalid condition {:?}: {e}",
                            edge.from, edge.to, edge.condition
                        ),
                    )
                    .with_edge(&edge.from, &edge.to),
                );
            }
        }
        diags
    }
}

// ---------------------------------------------------------------------------
// Built-in rule: stylesheet_syntax
// ---------------------------------------------------------------------------

struct StylesheetSyntaxRule;
impl LintRule for StylesheetSyntaxRule {
    fn name(&self) -> &str {
        "stylesheet_syntax"
    }
    fn apply(&self, graph: &Graph) -> Vec<Diagnostic> {
        let ss = &graph.graph_attrs.model_stylesheet;
        if ss.is_empty() {
            return vec![];
        }
        if let Err(e) = parse_stylesheet(ss) {
            vec![Diagnostic::error(
                "stylesheet_syntax",
                format!("model_stylesheet is invalid: {e}"),
            )]
        } else {
            vec![]
        }
    }
}

// ---------------------------------------------------------------------------
// Built-in rule: type_known
// ---------------------------------------------------------------------------

const KNOWN_HANDLER_TYPES: &[&str] = &[
    "start",
    "exit",
    "codergen",
    "wait.human",
    "conditional",
    "parallel",
    "parallel.fan_in",
    "tool",
    "stack.manager_loop",
];

struct TypeKnownRule;
impl LintRule for TypeKnownRule {
    fn name(&self) -> &str {
        "type_known"
    }
    fn apply(&self, graph: &Graph) -> Vec<Diagnostic> {
        let mut diags = Vec::new();
        for node in graph.nodes.values() {
            if node.node_type.is_empty() {
                continue;
            }
            if !KNOWN_HANDLER_TYPES.contains(&node.node_type.as_str()) {
                diags.push(
                    Diagnostic::warning(
                        "type_known",
                        format!(
                            "node '{}' has unrecognised type '{}'",
                            node.id, node.node_type
                        ),
                    )
                    .with_node(&node.id),
                );
            }
        }
        diags
    }
}

// ---------------------------------------------------------------------------
// Built-in rule: fidelity_valid
// ---------------------------------------------------------------------------

const VALID_FIDELITIES: &[&str] = &[
    "full",
    "truncate",
    "compact",
    "summary:low",
    "summary:medium",
    "summary:high",
];

struct FidelityValidRule;
impl LintRule for FidelityValidRule {
    fn name(&self) -> &str {
        "fidelity_valid"
    }
    fn apply(&self, graph: &Graph) -> Vec<Diagnostic> {
        let mut diags = Vec::new();
        for node in graph.nodes.values() {
            if !node.fidelity.is_empty() && !VALID_FIDELITIES.contains(&node.fidelity.as_str()) {
                diags.push(
                    Diagnostic::warning(
                        "fidelity_valid",
                        format!(
                            "node '{}' has invalid fidelity '{}'; valid values: {}",
                            node.id,
                            node.fidelity,
                            VALID_FIDELITIES.join(", ")
                        ),
                    )
                    .with_node(&node.id),
                );
            }
        }
        for edge in &graph.edges {
            if !edge.fidelity.is_empty() && !VALID_FIDELITIES.contains(&edge.fidelity.as_str()) {
                diags.push(
                    Diagnostic::warning(
                        "fidelity_valid",
                        format!(
                            "edge '{}'->'{}' has invalid fidelity '{}'",
                            edge.from, edge.to, edge.fidelity
                        ),
                    )
                    .with_edge(&edge.from, &edge.to),
                );
            }
        }
        diags
    }
}

// ---------------------------------------------------------------------------
// Built-in rule: retry_target_exists
// ---------------------------------------------------------------------------

struct RetryTargetExistsRule;
impl LintRule for RetryTargetExistsRule {
    fn name(&self) -> &str {
        "retry_target_exists"
    }
    fn apply(&self, graph: &Graph) -> Vec<Diagnostic> {
        let mut diags = Vec::new();
        let check = |target: &str, context: &str| -> Option<Diagnostic> {
            if !target.is_empty() && !graph.nodes.contains_key(target) {
                Some(Diagnostic::warning(
                    "retry_target_exists",
                    format!("{context} references unknown node '{target}'"),
                ))
            } else {
                None
            }
        };

        // Graph-level targets
        if let Some(d) = check(&graph.graph_attrs.retry_target, "graph.retry_target") {
            diags.push(d);
        }
        if let Some(d) = check(
            &graph.graph_attrs.fallback_retry_target,
            "graph.fallback_retry_target",
        ) {
            diags.push(d);
        }

        // Node-level targets
        for node in graph.nodes.values() {
            if let Some(d) = check(
                &node.retry_target,
                &format!("node '{}' retry_target", node.id),
            ) {
                diags.push(d.with_node(&node.id));
            }
            if let Some(d) = check(
                &node.fallback_retry_target,
                &format!("node '{}' fallback_retry_target", node.id),
            ) {
                diags.push(d.with_node(&node.id));
            }
        }
        diags
    }
}

// ---------------------------------------------------------------------------
// Built-in rule: goal_gate_has_retry
// ---------------------------------------------------------------------------

struct GoalGateHasRetryRule;
impl LintRule for GoalGateHasRetryRule {
    fn name(&self) -> &str {
        "goal_gate_has_retry"
    }
    fn apply(&self, graph: &Graph) -> Vec<Diagnostic> {
        let mut diags = Vec::new();
        let graph_has_retry = !graph.graph_attrs.retry_target.is_empty()
            || !graph.graph_attrs.fallback_retry_target.is_empty();

        for node in graph.nodes.values().filter(|n| n.goal_gate) {
            let node_has_retry =
                !node.retry_target.is_empty() || !node.fallback_retry_target.is_empty();
            if !node_has_retry && !graph_has_retry {
                diags.push(
                    Diagnostic::warning(
                        "goal_gate_has_retry",
                        format!(
                            "node '{}' has goal_gate=true but no retry_target is configured \
                             (neither on the node nor at graph level); \
                             if the goal gate fails the pipeline will error",
                            node.id
                        ),
                    )
                    .with_node(&node.id),
                );
            }
        }
        diags
    }
}

// ---------------------------------------------------------------------------
// Built-in rule: prompt_on_llm_nodes
// ---------------------------------------------------------------------------

struct PromptOnLlmNodesRule;
impl LintRule for PromptOnLlmNodesRule {
    fn name(&self) -> &str {
        "prompt_on_llm_nodes"
    }
    fn apply(&self, graph: &Graph) -> Vec<Diagnostic> {
        let mut diags = Vec::new();
        for node in graph.nodes.values() {
            let is_codergen = node.shape == "box"
                || node.node_type == "codergen"
                || (node.shape.is_empty() && node.node_type.is_empty());
            if is_codergen && node.prompt.is_empty() && node.label.is_empty() {
                diags.push(
                    Diagnostic::warning(
                        "prompt_on_llm_nodes",
                        format!(
                            "node '{}' resolves to the codergen handler but has no 'prompt' or 'label' attribute",
                            node.id
                        ),
                    )
                    .with_node(&node.id),
                );
            }
        }
        diags
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Edge, Graph, Node};

    /// Build a minimal valid 2-node graph: start → exit.
    fn minimal_valid_graph() -> Graph {
        let mut g = Graph::new("test".to_string());
        let start = Node {
            id: "start".to_string(),
            label: "Start".to_string(),
            shape: "Mdiamond".to_string(),
            ..Default::default()
        };
        g.nodes.insert("start".to_string(), start);

        let exit = Node {
            id: "exit".to_string(),
            label: "Exit".to_string(),
            shape: "Msquare".to_string(),
            ..Default::default()
        };
        g.nodes.insert("exit".to_string(), exit);

        g.edges.push(Edge {
            from: "start".to_string(),
            to: "exit".to_string(),
            ..Default::default()
        });
        g
    }

    fn linear_3node_graph() -> Graph {
        let mut g = Graph::new("test".to_string());
        for (id, shape) in &[
            ("start", "Mdiamond"),
            ("middle", "box"),
            ("exit", "Msquare"),
        ] {
            let n = Node {
                id: id.to_string(),
                label: id.to_string(),
                shape: shape.to_string(),
                ..Default::default()
            };
            g.nodes.insert(id.to_string(), n);
        }
        g.edges.push(Edge {
            from: "start".into(),
            to: "middle".into(),
            ..Default::default()
        });
        g.edges.push(Edge {
            from: "middle".into(),
            to: "exit".into(),
            ..Default::default()
        });
        g
    }

    // --- validate on valid graph ---

    #[test]
    fn valid_graph_no_errors() {
        let g = minimal_valid_graph();
        let diags = validate(&g, &[]);
        let errors: Vec<_> = diags
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
    }

    #[test]
    fn valid_3node_graph_no_errors() {
        let g = linear_3node_graph();
        let diags = validate(&g, &[]);
        let errors: Vec<_> = diags
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
    }

    // --- start_node rule ---

    #[test]
    fn missing_start_node_error() {
        let mut g = minimal_valid_graph();
        g.nodes.get_mut("start").unwrap().shape = "box".to_string();
        let diags = validate(&g, &[]);
        assert!(
            diags
                .iter()
                .any(|d| d.rule == "start_node" && d.severity == Severity::Error)
        );
    }

    #[test]
    fn multiple_start_nodes_error() {
        let mut g = minimal_valid_graph();
        let extra = Node {
            id: "start2".to_string(),
            shape: "Mdiamond".to_string(),
            ..Default::default()
        };
        g.nodes.insert("start2".to_string(), extra);
        let diags = validate(&g, &[]);
        assert!(
            diags
                .iter()
                .any(|d| d.rule == "start_node" && d.severity == Severity::Error)
        );
    }

    // --- terminal_node rule ---

    #[test]
    fn missing_exit_node_error() {
        let mut g = minimal_valid_graph();
        g.nodes.get_mut("exit").unwrap().shape = "box".to_string();
        let diags = validate(&g, &[]);
        assert!(
            diags
                .iter()
                .any(|d| d.rule == "terminal_node" && d.severity == Severity::Error)
        );
    }

    #[test]
    fn multiple_exit_nodes_error() {
        // V2-ATR-003: NLSpec §11.2 "Exactly one exit node (shape=Msquare) is required".
        // Two Msquare nodes must produce an ERROR diagnostic.
        let mut g = minimal_valid_graph();
        let exit2 = Node {
            id: "exit2".to_string(),
            shape: "Msquare".to_string(),
            label: "Exit2".to_string(),
            ..Default::default()
        };
        g.nodes.insert("exit2".to_string(), exit2);
        let diags = validate(&g, &[]);
        assert!(
            diags
                .iter()
                .any(|d| d.rule == "terminal_node" && d.severity == Severity::Error),
            "Two Msquare nodes must produce a terminal_node ERROR; got: {diags:?}"
        );
    }

    // --- start_no_incoming ---

    #[test]
    fn start_node_with_incoming_edge_error() {
        let mut g = linear_3node_graph();
        g.edges.push(Edge {
            from: "middle".into(),
            to: "start".into(),
            ..Default::default()
        });
        let diags = validate(&g, &[]);
        assert!(
            diags
                .iter()
                .any(|d| d.rule == "start_no_incoming" && d.severity == Severity::Error)
        );
    }

    // --- exit_no_outgoing ---

    #[test]
    fn exit_node_with_outgoing_edge_error() {
        let mut g = linear_3node_graph();
        g.edges.push(Edge {
            from: "exit".into(),
            to: "middle".into(),
            ..Default::default()
        });
        let diags = validate(&g, &[]);
        assert!(
            diags
                .iter()
                .any(|d| d.rule == "exit_no_outgoing" && d.severity == Severity::Error)
        );
    }

    // --- reachability ---

    #[test]
    fn orphan_node_error() {
        let mut g = linear_3node_graph();
        let orphan = Node {
            id: "orphan".to_string(),
            shape: "box".to_string(),
            ..Default::default()
        };
        g.nodes.insert("orphan".to_string(), orphan);
        let diags = validate(&g, &[]);
        assert!(
            diags
                .iter()
                .any(|d| d.rule == "reachability" && d.node_id.as_deref() == Some("orphan"))
        );
    }

    #[test]
    fn orphan_node_is_warning_not_error() {
        // NLSpec §11.12 parity matrix: "orphan node → warning"
        let mut g = linear_3node_graph();
        let orphan = Node {
            id: "orphan".to_string(),
            shape: "box".to_string(),
            ..Default::default()
        };
        g.nodes.insert("orphan".to_string(), orphan);
        let diags = validate(&g, &[]);
        let reachability_diag = diags
            .iter()
            .find(|d| d.rule == "reachability" && d.node_id.as_deref() == Some("orphan"))
            .expect("expected a reachability diagnostic for orphan node");
        assert_eq!(
            reachability_diag.severity,
            Severity::Warning,
            "orphan node should produce a Warning, not an Error"
        );
    }

    // --- edge_target_exists ---

    #[test]
    fn edge_to_nonexistent_node_error() {
        let mut g = linear_3node_graph();
        g.edges.push(Edge {
            from: "middle".into(),
            to: "ghost".into(),
            ..Default::default()
        });
        let diags = validate(&g, &[]);
        assert!(
            diags
                .iter()
                .any(|d| d.rule == "edge_target_exists" && d.severity == Severity::Error)
        );
    }

    // --- condition_syntax ---

    #[test]
    fn invalid_condition_error() {
        let mut g = minimal_valid_graph();
        g.edges[0].condition = "outcome>>bad".to_string(); // no valid operator
        let diags = validate(&g, &[]);
        assert!(
            diags
                .iter()
                .any(|d| d.rule == "condition_syntax" && d.severity == Severity::Error)
        );
    }

    #[test]
    fn valid_condition_no_error() {
        let mut g = minimal_valid_graph();
        g.edges[0].condition = "outcome=success".to_string();
        let diags = validate(&g, &[]);
        assert!(
            !diags
                .iter()
                .any(|d| d.rule == "condition_syntax" && d.severity == Severity::Error)
        );
    }

    // --- stylesheet_syntax ---

    #[test]
    fn invalid_stylesheet_error() {
        let mut g = minimal_valid_graph();
        g.graph_attrs.model_stylesheet = "* { not_a_prop: val; }".to_string();
        let diags = validate(&g, &[]);
        assert!(
            diags
                .iter()
                .any(|d| d.rule == "stylesheet_syntax" && d.severity == Severity::Error)
        );
    }

    #[test]
    fn empty_stylesheet_no_error() {
        let g = minimal_valid_graph();
        let diags = validate(&g, &[]);
        assert!(!diags.iter().any(|d| d.rule == "stylesheet_syntax"));
    }

    // --- goal_gate_has_retry ---

    #[test]
    fn goal_gate_without_retry_warning() {
        let mut g = linear_3node_graph();
        g.nodes.get_mut("middle").unwrap().goal_gate = true;
        let diags = validate(&g, &[]);
        assert!(
            diags
                .iter()
                .any(|d| d.rule == "goal_gate_has_retry" && d.severity == Severity::Warning)
        );
    }

    #[test]
    fn goal_gate_with_graph_retry_no_warning() {
        let mut g = linear_3node_graph();
        g.nodes.get_mut("middle").unwrap().goal_gate = true;
        g.graph_attrs.retry_target = "middle".to_string();
        let diags = validate(&g, &[]);
        assert!(!diags.iter().any(|d| d.rule == "goal_gate_has_retry"));
    }

    // --- validate_or_raise ---

    #[test]
    fn validate_or_raise_ok_on_valid_graph() {
        let g = linear_3node_graph();
        assert!(validate_or_raise(&g, &[]).is_ok());
    }

    #[test]
    fn validate_or_raise_err_on_invalid_graph() {
        let g = Graph::new("empty".to_string());
        // No nodes at all → start_node + terminal_node errors
        let result = validate_or_raise(&g, &[]);
        assert!(result.is_err());
        if let Err(ValidationError::Failed { count }) = result {
            assert!(count >= 2);
        }
    }

    #[test]
    fn validate_or_raise_ok_with_only_warnings() {
        let mut g = linear_3node_graph();
        // Add a node with unknown type → warning only
        g.nodes.get_mut("middle").unwrap().node_type = "my_custom_handler".to_string();
        let result = validate_or_raise(&g, &[]);
        assert!(result.is_ok());
        let diags = result.unwrap();
        assert!(
            diags
                .iter()
                .any(|d| d.rule == "type_known" && d.severity == Severity::Warning)
        );
    }

    // --- GAP-ATR-004: prompt_on_llm_nodes boundary ---

    #[test]
    fn prompt_warning_suppressed_by_label() {
        // GAP-ATR-004: a node with empty `prompt` but non-empty `label`
        // must NOT trigger the prompt_on_llm_nodes warning.
        let mut g = linear_3node_graph();
        // middle has shape=box (codergen), no prompt, but has a label set from the helper
        // Confirm middle already has a label:
        assert!(!g.nodes["middle"].label.is_empty());
        // Explicitly clear prompt to be sure
        g.nodes.get_mut("middle").unwrap().prompt = String::new();
        let diags = validate(&g, &[]);
        assert!(
            !diags.iter().any(|d| d.rule == "prompt_on_llm_nodes"),
            "node with empty prompt but non-empty label must not trigger prompt warning; got: {diags:?}"
        );
    }

    // --- GAP-ATR-005: all Diagnostic fields populated ---

    #[test]
    fn all_diagnostic_fields_populated_simultaneously() {
        // GAP-ATR-005: NLSpec §11.2 says lint results include rule name,
        // severity, node/edge ID, and message — verify all four are present.
        let mut g = linear_3node_graph();
        // Add an orphan to trigger reachability warning (has node_id)
        let orphan = Node {
            id: "orphan".to_string(),
            shape: "box".to_string(),
            label: "Orphan".to_string(),
            ..Default::default()
        };
        g.nodes.insert("orphan".to_string(), orphan);

        let diags = validate(&g, &[]);
        let d = diags
            .iter()
            .find(|d| d.rule == "reachability" && d.node_id.as_deref() == Some("orphan"))
            .expect("expected reachability diagnostic for orphan node");

        // Verify all four fields are non-empty / present simultaneously
        assert!(!d.rule.is_empty(), "rule must be non-empty");
        assert!(!d.message.is_empty(), "message must be non-empty");
        assert!(
            d.node_id.is_some(),
            "node_id must be Some(_) for a node-level diagnostic"
        );
        // severity is always populated (it's an enum, not Option)
        let _severity = d.severity; // just touching it proves it's accessible
    }

    // --- custom rule ---

    #[test]
    fn custom_extra_rule_runs() {
        struct AlwaysWarn;
        impl LintRule for AlwaysWarn {
            fn name(&self) -> &str {
                "always_warn"
            }
            fn apply(&self, _graph: &Graph) -> Vec<Diagnostic> {
                vec![Diagnostic::warning("always_warn", "custom warning")]
            }
        }

        let g = linear_3node_graph();
        let extra: Vec<Box<dyn LintRule>> = vec![Box::new(AlwaysWarn)];
        let diags = validate(&g, &extra);
        assert!(diags.iter().any(|d| d.rule == "always_warn"));
    }
}
