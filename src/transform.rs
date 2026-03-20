//! AST transforms applied after parsing and before validation (NLSpec §9.1–9.2).
//!
//! Transforms receive and return a [`Graph`], allowing structural mutations
//! without modifying the original DOT source.
//!
//! Built-in transforms:
//! - [`VariableExpansionTransform`] — replaces `$goal` in node prompts
//! - [`StylesheetApplicationTransform`] — applies `model_stylesheet` to nodes

use crate::graph::Graph;
use crate::stylesheet::{apply_stylesheet, parse_stylesheet};

// ---------------------------------------------------------------------------
// Transform trait
// ---------------------------------------------------------------------------

/// A graph AST transform applied after parsing and before validation.
///
/// The transform receives ownership of the graph and must return a (possibly
/// modified) graph.  Transforms are applied in registration order.
pub trait Transform: Send + Sync {
    fn apply(&self, graph: Graph) -> Graph;
}

// ---------------------------------------------------------------------------
// VariableExpansionTransform
// ---------------------------------------------------------------------------

/// Expands `$goal` in every node's `prompt` attribute.
///
/// Replaces all occurrences of the literal string `$goal` with
/// `graph.graph_attrs.goal`.
pub struct VariableExpansionTransform;

impl Transform for VariableExpansionTransform {
    fn apply(&self, mut graph: Graph) -> Graph {
        let goal = graph.graph_attrs.goal.clone();
        for node in graph.nodes.values_mut() {
            if node.prompt.contains("$goal") {
                node.prompt = node.prompt.replace("$goal", &goal);
            }
        }
        graph
    }
}

// ---------------------------------------------------------------------------
// StylesheetApplicationTransform
// ---------------------------------------------------------------------------

/// Parses `graph.graph_attrs.model_stylesheet` and applies it to all nodes.
///
/// Empty stylesheet → graph returned unchanged.
/// Parse error → log a warning and return graph unchanged.
pub struct StylesheetApplicationTransform;

impl Transform for StylesheetApplicationTransform {
    fn apply(&self, mut graph: Graph) -> Graph {
        let css = graph.graph_attrs.model_stylesheet.clone();
        if css.is_empty() {
            return graph;
        }
        match parse_stylesheet(&css) {
            Ok(stylesheet) => {
                apply_stylesheet(&stylesheet, &mut graph);
            }
            Err(e) => {
                tracing::warn!("stylesheet parse error (transform skipped): {e}");
            }
        }
        graph
    }
}

// ---------------------------------------------------------------------------
// apply_transforms
// ---------------------------------------------------------------------------

/// Apply a sequence of transforms to a graph in order.
///
/// Each transform receives the graph returned by the previous one.
pub fn apply_transforms(graph: Graph, transforms: &[Box<dyn Transform>]) -> Graph {
    transforms.iter().fold(graph, |g, t| t.apply(g))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Graph, GraphAttrs, Node};

    fn make_graph(goal: &str, stylesheet: &str) -> Graph {
        let mut g = Graph::new("test".into());
        g.graph_attrs = GraphAttrs {
            goal: goal.to_string(),
            model_stylesheet: stylesheet.to_string(),
            default_max_retry: 50,
            ..Default::default()
        };
        g
    }

    fn add_node(graph: &mut Graph, id: &str, prompt: &str) {
        let n = Node {
            id: id.to_string(),
            prompt: prompt.to_string(),
            ..Default::default()
        };
        graph.nodes.insert(id.to_string(), n);
    }

    // --- VariableExpansionTransform ---

    #[test]
    fn expands_goal_in_prompt() {
        let mut g = make_graph("build a rocket", "");
        add_node(&mut g, "A", "Work on: $goal");
        let g = VariableExpansionTransform.apply(g);
        assert_eq!(g.nodes["A"].prompt, "Work on: build a rocket");
    }

    #[test]
    fn expands_multiple_occurrences() {
        let mut g = make_graph("X", "");
        add_node(&mut g, "A", "$goal and $goal");
        let g = VariableExpansionTransform.apply(g);
        assert_eq!(g.nodes["A"].prompt, "X and X");
    }

    #[test]
    fn no_goal_var_leaves_prompt_unchanged() {
        let mut g = make_graph("X", "");
        add_node(&mut g, "A", "no variable here");
        let g = VariableExpansionTransform.apply(g);
        assert_eq!(g.nodes["A"].prompt, "no variable here");
    }

    #[test]
    fn empty_goal_replaces_with_empty_string() {
        let mut g = make_graph("", "");
        add_node(&mut g, "A", "Goal: $goal");
        let g = VariableExpansionTransform.apply(g);
        assert_eq!(g.nodes["A"].prompt, "Goal: ");
    }

    // --- StylesheetApplicationTransform ---

    #[test]
    fn empty_stylesheet_noop() {
        let mut g = make_graph("", "");
        add_node(&mut g, "A", "");
        let g = StylesheetApplicationTransform.apply(g);
        assert_eq!(g.nodes["A"].llm_model, "");
    }

    #[test]
    fn stylesheet_sets_llm_model() {
        let css = "* { llm_model: claude-opus; }";
        let mut g = make_graph("", css);
        add_node(&mut g, "A", "");
        let g = StylesheetApplicationTransform.apply(g);
        assert_eq!(g.nodes["A"].llm_model, "claude-opus");
    }

    #[test]
    fn invalid_stylesheet_returns_graph_unchanged() {
        let css = "this is not valid css!!!";
        let mut g = make_graph("", css);
        add_node(&mut g, "A", "");
        let original_model = g.nodes["A"].llm_model.clone();
        let g = StylesheetApplicationTransform.apply(g);
        assert_eq!(g.nodes["A"].llm_model, original_model);
    }

    // --- apply_transforms ---

    #[test]
    fn apply_transforms_in_order() {
        let css = "* { llm_model: base; }";
        let mut g = make_graph("the goal", css);
        add_node(&mut g, "A", "do $goal with model");

        let transforms: Vec<Box<dyn Transform>> = vec![
            Box::new(VariableExpansionTransform),
            Box::new(StylesheetApplicationTransform),
        ];

        let g = apply_transforms(g, &transforms);
        assert_eq!(g.nodes["A"].prompt, "do the goal with model");
        assert_eq!(g.nodes["A"].llm_model, "base");
    }

    #[test]
    fn apply_transforms_empty_vec_noop() {
        let mut g = make_graph("", "");
        add_node(&mut g, "A", "$goal");
        let g = apply_transforms(g, &[]);
        assert_eq!(g.nodes["A"].prompt, "$goal"); // not expanded
    }

    // --- GAP-ATR-002: class attribute + stylesheet merge pipeline ---

    #[test]
    fn class_attr_plus_stylesheet_merge_sets_llm_model() {
        // GAP-ATR-002: parse DOT with class="fast" on a node AND
        // model_stylesheet at graph level, apply StylesheetApplicationTransform,
        // verify node's llm_model is set from the class rule.
        use crate::parser::parse_dot;

        let dot = r#"
digraph {
    graph [model_stylesheet=".fast { llm_model: turbo-model; }"]
    start [shape=Mdiamond]
    exit  [shape=Msquare]
    task  [class="fast", prompt="do the work"]
    start -> task -> exit
}
"#;
        let graph = parse_dot(dot).expect("DOT should parse");
        // Verify class was parsed correctly
        assert_eq!(graph.nodes["task"].class, "fast");
        assert_eq!(
            graph.graph_attrs.model_stylesheet,
            ".fast { llm_model: turbo-model; }"
        );

        // Apply the stylesheet transform
        let transformed = StylesheetApplicationTransform.apply(graph);

        // Verify the llm_model was set by the class rule
        assert_eq!(
            transformed.nodes["task"].llm_model, "turbo-model",
            "stylesheet class rule should set llm_model on matching node"
        );
        // Non-matching nodes should not have llm_model set
        assert_eq!(transformed.nodes["start"].llm_model, "");
    }
}
