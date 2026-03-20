//! Context fidelity mode resolution (NLSpec §5.4).
//!
//! Fidelity controls how much prior conversation and state is carried into
//! the next LLM session.  The resolution cascade is:
//!
//! 1. Edge `fidelity` attribute (highest precedence)
//! 2. Target node `fidelity` attribute
//! 3. Graph `default_fidelity` attribute
//! 4. Built-in default: [`FidelityMode::Compact`]

use crate::graph::{Edge, Graph, Node};

// ---------------------------------------------------------------------------
// FidelityMode
// ---------------------------------------------------------------------------

/// The fidelity mode controlling how much prior state is carried forward.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FidelityMode {
    /// Reuse the same LLM session (full conversation history preserved).
    Full,
    /// Fresh session with only graph goal + run ID (minimal context).
    Truncate,
    /// Fresh session with structured bullet-point summary.
    Compact,
    /// Fresh session with brief textual summary (~600 tokens).
    SummaryLow,
    /// Fresh session with moderate detail (~1500 tokens).
    SummaryMedium,
    /// Fresh session with detailed summary (~3000 tokens).
    SummaryHigh,
}

impl FidelityMode {
    /// Parse a fidelity string.  Returns `None` for unknown values.
    ///
    /// Unlike `std::str::FromStr`, this returns `Option` rather than `Result`,
    /// allowing callers to silently fall through to lower-priority sources.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "full" => Some(FidelityMode::Full),
            "truncate" => Some(FidelityMode::Truncate),
            "compact" => Some(FidelityMode::Compact),
            "summary:low" => Some(FidelityMode::SummaryLow),
            "summary:medium" => Some(FidelityMode::SummaryMedium),
            "summary:high" => Some(FidelityMode::SummaryHigh),
            _ => None,
        }
    }

    /// Return the canonical string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            FidelityMode::Full => "full",
            FidelityMode::Truncate => "truncate",
            FidelityMode::Compact => "compact",
            FidelityMode::SummaryLow => "summary:low",
            FidelityMode::SummaryMedium => "summary:medium",
            FidelityMode::SummaryHigh => "summary:high",
        }
    }
}

impl std::fmt::Display for FidelityMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// resolve_fidelity
// ---------------------------------------------------------------------------

/// Resolve the fidelity mode for a target node given an optional incoming edge.
///
/// Precedence (highest first):
/// 1. `edge.fidelity` — if non-empty and a recognised mode
/// 2. `node.fidelity` — if non-empty and a recognised mode
/// 3. `graph.graph_attrs.default_fidelity` — if non-empty and a recognised mode
/// 4. [`FidelityMode::Compact`] (built-in default)
pub fn resolve_fidelity(edge: Option<&Edge>, node: &Node, graph: &Graph) -> FidelityMode {
    // Step 1: edge fidelity
    if let Some(e) = edge {
        if !e.fidelity.is_empty() {
            if let Some(m) = FidelityMode::parse(&e.fidelity) {
                return m;
            }
        }
    }

    // Step 2: node fidelity
    if !node.fidelity.is_empty() {
        if let Some(m) = FidelityMode::parse(&node.fidelity) {
            return m;
        }
    }

    // Step 3: graph default
    let default = &graph.graph_attrs.default_fidelity;
    if !default.is_empty() {
        if let Some(m) = FidelityMode::parse(default) {
            return m;
        }
    }

    // Step 4: built-in default
    FidelityMode::Compact
}

// ---------------------------------------------------------------------------
// resolve_thread_id
// ---------------------------------------------------------------------------

/// Resolve the thread key for `full`-fidelity session reuse.
///
/// Precedence:
/// 1. `node.thread_id` (if non-empty)
/// 2. `edge.thread_id` (if edge present and non-empty)
/// 3. `prev_node_id` (fallback)
pub fn resolve_thread_id<'a>(
    edge: Option<&'a Edge>,
    node: &'a Node,
    prev_node_id: &'a str,
) -> &'a str {
    if !node.thread_id.is_empty() {
        return &node.thread_id;
    }
    if let Some(e) = edge {
        if !e.thread_id.is_empty() {
            return &e.thread_id;
        }
    }
    prev_node_id
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Edge, Graph, GraphAttrs, Node};

    fn make_graph(default_fidelity: &str) -> Graph {
        let mut g = Graph::new("test".into());
        g.graph_attrs = GraphAttrs {
            default_fidelity: default_fidelity.to_string(),
            default_max_retry: 50,
            ..Default::default()
        };
        g
    }

    fn make_edge(fidelity: &str, thread_id: &str) -> Edge {
        Edge {
            from: "A".into(),
            to: "B".into(),
            fidelity: fidelity.to_string(),
            thread_id: thread_id.to_string(),
            ..Default::default()
        }
    }

    fn make_node(fidelity: &str, thread_id: &str) -> Node {
        Node {
            fidelity: fidelity.to_string(),
            thread_id: thread_id.to_string(),
            ..Default::default()
        }
    }

    // --- FidelityMode::from_str ---

    #[test]
    fn parse_all_modes() {
        assert_eq!(FidelityMode::parse("full"), Some(FidelityMode::Full));
        assert_eq!(
            FidelityMode::parse("truncate"),
            Some(FidelityMode::Truncate)
        );
        assert_eq!(FidelityMode::parse("compact"), Some(FidelityMode::Compact));
        assert_eq!(
            FidelityMode::parse("summary:low"),
            Some(FidelityMode::SummaryLow)
        );
        assert_eq!(
            FidelityMode::parse("summary:medium"),
            Some(FidelityMode::SummaryMedium)
        );
        assert_eq!(
            FidelityMode::parse("summary:high"),
            Some(FidelityMode::SummaryHigh)
        );
    }

    #[test]
    fn parse_unknown_returns_none() {
        assert_eq!(FidelityMode::parse("unknown"), None);
        assert_eq!(FidelityMode::parse(""), None);
        assert_eq!(FidelityMode::parse("FULL"), None); // case-sensitive
    }

    #[test]
    fn as_str_roundtrips() {
        for mode in [
            FidelityMode::Full,
            FidelityMode::Truncate,
            FidelityMode::Compact,
            FidelityMode::SummaryLow,
            FidelityMode::SummaryMedium,
            FidelityMode::SummaryHigh,
        ] {
            assert_eq!(FidelityMode::parse(mode.as_str()), Some(mode));
        }
    }

    // --- resolve_fidelity ---

    #[test]
    fn edge_fidelity_highest_precedence() {
        let g = make_graph("truncate");
        let edge = make_edge("full", "");
        let node = make_node("summary:low", "");
        assert_eq!(resolve_fidelity(Some(&edge), &node, &g), FidelityMode::Full);
    }

    #[test]
    fn node_fidelity_overrides_graph() {
        let g = make_graph("truncate");
        let node = make_node("summary:high", "");
        assert_eq!(resolve_fidelity(None, &node, &g), FidelityMode::SummaryHigh);
    }

    #[test]
    fn graph_default_fidelity_used() {
        let g = make_graph("summary:medium");
        let node = make_node("", "");
        assert_eq!(
            resolve_fidelity(None, &node, &g),
            FidelityMode::SummaryMedium
        );
    }

    #[test]
    fn default_is_compact_when_nothing_set() {
        let g = make_graph("");
        let node = make_node("", "");
        assert_eq!(resolve_fidelity(None, &node, &g), FidelityMode::Compact);
    }

    #[test]
    fn invalid_edge_fidelity_falls_through() {
        let g = make_graph("compact");
        let edge = make_edge("notvalid", "");
        let node = make_node("full", "");
        // Edge fidelity invalid → try node → "full"
        assert_eq!(resolve_fidelity(Some(&edge), &node, &g), FidelityMode::Full);
    }

    #[test]
    fn invalid_node_fidelity_falls_through_to_graph() {
        let g = make_graph("truncate");
        let node = make_node("bogus", "");
        assert_eq!(resolve_fidelity(None, &node, &g), FidelityMode::Truncate);
    }

    // --- resolve_thread_id ---

    #[test]
    fn node_thread_id_wins() {
        let edge = make_edge("", "edge-thread");
        let node = make_node("", "node-thread");
        assert_eq!(resolve_thread_id(Some(&edge), &node, "prev"), "node-thread");
    }

    #[test]
    fn edge_thread_id_used_when_no_node_thread() {
        let edge = make_edge("", "edge-thread");
        let node = make_node("", "");
        assert_eq!(resolve_thread_id(Some(&edge), &node, "prev"), "edge-thread");
    }

    #[test]
    fn falls_back_to_prev_node_id() {
        let node = make_node("", "");
        assert_eq!(resolve_thread_id(None, &node, "prev-node"), "prev-node");
    }
}
