//! Model stylesheet parser and applicator (NLSpec §8).
//!
//! The `model_stylesheet` graph attribute contains a CSS-like string that
//! assigns `llm_model`, `llm_provider`, and `reasoning_effort` to nodes by
//! selector.
//!
//! ## Grammar (NLSpec §8.2)
//! ```text
//! Stylesheet    ::= Rule*
//! Rule          ::= Selector '{' Declaration ( ';' Declaration )* ';'? '}'
//! Selector      ::= '*' | '#' Identifier | '.' ClassName
//! ClassName     ::= [a-z0-9-]+
//! Declaration   ::= Property ':' PropertyValue
//! Property      ::= 'llm_model' | 'llm_provider' | 'reasoning_effort'
//! PropertyValue ::= bare-value | quoted-string
//! ```
//!
//! ## Public API
//! - [`parse_stylesheet`] — parse a stylesheet string into a [`Stylesheet`]
//! - [`apply_stylesheet`] — apply declarations to graph nodes

use crate::error::ParseError;
use crate::graph::Graph;
use crate::parser::lexer::strip_comments;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A parsed model stylesheet.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Stylesheet {
    pub rules: Vec<StyleRule>,
}

/// A single CSS-like rule: selector + declarations.
#[derive(Debug, Clone, PartialEq)]
pub struct StyleRule {
    pub selector: Selector,
    pub declarations: Vec<Declaration>,
}

/// A CSS-like selector.
#[derive(Debug, Clone, PartialEq)]
pub enum Selector {
    /// `*` — matches all nodes.
    Universal,
    /// `.class-name` — matches nodes with that class.
    Class(String),
    /// `#node_id` — matches a specific node by ID.
    Id(String),
    /// `shape-name` — bare identifier, matches nodes whose `shape` attribute equals the name.
    /// E.g. `box { ... }` applies to every node with `shape=box`.
    Shape(String),
}

impl Selector {
    /// Specificity score following NLSpec §8: Universal=0, Shape=1, Class=2, Id=3.
    ///
    /// Higher specificity wins. Rules of equal specificity: last declared wins.
    pub fn specificity(&self) -> u8 {
        match self {
            Selector::Universal => 0,
            Selector::Shape(_) => 1,
            Selector::Class(_) => 2,
            Selector::Id(_) => 3,
        }
    }
}

/// A single `property: value` declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct Declaration {
    pub property: StyleProperty,
    pub value: String,
}

/// The three recognised stylesheet properties.
#[derive(Debug, Clone, PartialEq)]
pub enum StyleProperty {
    LlmModel,
    LlmProvider,
    ReasoningEffort,
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Parse a stylesheet string into a [`Stylesheet`].
///
/// Empty string returns `Ok(Stylesheet { rules: vec![] })`.
/// Returns [`ParseError::StylesheetSyntax`] on invalid input.
pub fn parse_stylesheet(source: &str) -> Result<Stylesheet, ParseError> {
    let cleaned = strip_comments(source);
    let input = cleaned.trim();
    if input.is_empty() {
        return Ok(Stylesheet::default());
    }

    let mut rules = Vec::new();
    let mut pos = 0;
    let chars: Vec<char> = input.chars().collect();
    let n = chars.len();

    while pos < n {
        // Skip whitespace
        while pos < n && chars[pos].is_whitespace() {
            pos += 1;
        }
        if pos >= n {
            break;
        }

        // Parse selector
        let sel = parse_selector(&chars, &mut pos, input).map_err(|e| {
            ParseError::StylesheetSyntax(format!("selector error at pos {pos}: {e}"))
        })?;

        // Skip whitespace
        while pos < n && chars[pos].is_whitespace() {
            pos += 1;
        }

        // Expect `{`
        if pos >= n || chars[pos] != '{' {
            return Err(ParseError::StylesheetSyntax(format!(
                "expected '{{' after selector at pos {pos}"
            )));
        }
        pos += 1; // consume `{`

        // Parse declarations until `}`
        let decls = parse_declarations(&chars, &mut pos, input)
            .map_err(|e| ParseError::StylesheetSyntax(format!("declaration error: {e}")))?;

        // Expect `}`
        while pos < n && chars[pos].is_whitespace() {
            pos += 1;
        }
        if pos >= n || chars[pos] != '}' {
            return Err(ParseError::StylesheetSyntax(format!(
                "expected '}}' to close rule at pos {pos}"
            )));
        }
        pos += 1; // consume `}`

        // Always push rule (even empty-declaration rules for roundtrip fidelity)
        rules.push(StyleRule {
            selector: sel,
            declarations: decls,
        });
    }

    Ok(Stylesheet { rules })
}

fn parse_selector(chars: &[char], pos: &mut usize, _src: &str) -> Result<Selector, String> {
    let n = chars.len();
    while *pos < n && chars[*pos].is_whitespace() {
        *pos += 1;
    }
    if *pos >= n {
        return Err("unexpected end of input while parsing selector".to_string());
    }
    match chars[*pos] {
        '*' => {
            *pos += 1;
            Ok(Selector::Universal)
        }
        '#' => {
            *pos += 1;
            let id = collect_while(chars, pos, |c| {
                c.is_ascii_alphanumeric() || c == '_' || c == '-'
            });
            if id.is_empty() {
                return Err("empty ID selector".to_string());
            }
            Ok(Selector::Id(id))
        }
        '.' => {
            *pos += 1;
            let cls = collect_while(chars, pos, |c| c.is_ascii_alphanumeric() || c == '-');
            if cls.is_empty() {
                return Err("empty class selector".to_string());
            }
            Ok(Selector::Class(cls))
        }
        // Bare identifier → shape-name selector (NLSpec §11.10)
        c if c.is_ascii_alphabetic() || c == '_' => {
            let shape = collect_while(chars, pos, |c| {
                c.is_ascii_alphanumeric() || c == '_' || c == '-'
            });
            if shape.is_empty() {
                return Err("empty shape selector".to_string());
            }
            Ok(Selector::Shape(shape))
        }
        c => Err(format!("unexpected character {c:?} in selector")),
    }
}

fn parse_declarations(
    chars: &[char],
    pos: &mut usize,
    _src: &str,
) -> Result<Vec<Declaration>, String> {
    let n = chars.len();
    let mut decls = Vec::new();

    loop {
        // Skip whitespace
        while *pos < n && chars[*pos].is_whitespace() {
            *pos += 1;
        }
        if *pos >= n || chars[*pos] == '}' {
            break;
        }

        // Parse property name
        let prop_str = collect_while(chars, pos, |c| c.is_ascii_alphanumeric() || c == '_');
        if prop_str.is_empty() {
            // Could be `;` — skip it
            if *pos < n && chars[*pos] == ';' {
                *pos += 1;
                continue;
            }
            break;
        }

        let property = match prop_str.as_str() {
            "llm_model" => StyleProperty::LlmModel,
            "llm_provider" => StyleProperty::LlmProvider,
            "reasoning_effort" => StyleProperty::ReasoningEffort,
            other => return Err(format!("unknown property: {other:?}")),
        };

        // Skip whitespace
        while *pos < n && chars[*pos].is_whitespace() {
            *pos += 1;
        }

        // Expect `:`
        if *pos >= n || chars[*pos] != ':' {
            return Err(format!("expected ':' after property '{prop_str}'"));
        }
        *pos += 1;

        // Skip whitespace
        while *pos < n && chars[*pos].is_whitespace() {
            *pos += 1;
        }

        // Parse value: quoted string or bare value (up to `;`, `}`, or whitespace)
        let value = if *pos < n && chars[*pos] == '"' {
            parse_quoted_string(chars, pos)?
        } else {
            collect_while(chars, pos, |c| {
                !matches!(c, ';' | '}' | '{') && !c.is_whitespace()
            })
        };

        if value.is_empty() {
            return Err(format!("empty value for property '{prop_str}'"));
        }

        decls.push(Declaration { property, value });

        // Skip whitespace and optional `;`
        while *pos < n && chars[*pos].is_whitespace() {
            *pos += 1;
        }
        if *pos < n && chars[*pos] == ';' {
            *pos += 1;
        }
    }

    Ok(decls)
}

fn collect_while<F: Fn(char) -> bool>(chars: &[char], pos: &mut usize, f: F) -> String {
    let mut s = String::new();
    while *pos < chars.len() && f(chars[*pos]) {
        s.push(chars[*pos]);
        *pos += 1;
    }
    s
}

fn parse_quoted_string(chars: &[char], pos: &mut usize) -> Result<String, String> {
    // Opening `"` already confirmed
    *pos += 1;
    let mut s = String::new();
    loop {
        if *pos >= chars.len() {
            return Err("unterminated string".to_string());
        }
        match chars[*pos] {
            '"' => {
                *pos += 1;
                return Ok(s);
            }
            '\\' => {
                *pos += 1;
                if *pos >= chars.len() {
                    return Err("unexpected end after backslash".to_string());
                }
                match chars[*pos] {
                    '"' => s.push('"'),
                    'n' => s.push('\n'),
                    't' => s.push('\t'),
                    '\\' => s.push('\\'),
                    c => s.push(c),
                }
                *pos += 1;
            }
            c => {
                s.push(c);
                *pos += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Applicator
// ---------------------------------------------------------------------------

/// Apply a [`Stylesheet`] to all nodes in a [`Graph`].
///
/// For each node, the applicator:
/// 1. Collects all matching rules.
/// 2. Sorts by specificity (ascending: universal < class < ID).
/// 3. For each property, applies the declaration from the highest-specificity match.
///    Rules of equal specificity: last wins.
/// 4. Explicit node attributes are NOT overridden.
pub fn apply_stylesheet(stylesheet: &Stylesheet, graph: &mut Graph) {
    if stylesheet.rules.is_empty() {
        return;
    }

    // Collect all node IDs first to avoid borrow issues
    let node_ids: Vec<String> = graph.nodes.keys().cloned().collect();

    for node_id in node_ids {
        let node = graph.nodes.get(&node_id).unwrap();

        // Find matching rules, with their specificity
        let mut matched: Vec<(u8, usize, &StyleRule)> = stylesheet
            .rules
            .iter()
            .enumerate()
            .filter(|(_, rule)| rule_matches_node(rule, node))
            .map(|(idx, rule)| (rule.selector.specificity(), idx, rule))
            .collect();

        // Sort by specificity ascending, then by index ascending (last-write-wins at same spec)
        matched.sort_by_key(|(spec, idx, _)| (*spec, *idx));

        // Determine effective values for each property
        let mut model: Option<String> = None;
        let mut provider: Option<String> = None;
        let mut effort: Option<String> = None;

        for (_, _, rule) in &matched {
            for decl in &rule.declarations {
                match decl.property {
                    StyleProperty::LlmModel => model = Some(decl.value.clone()),
                    StyleProperty::LlmProvider => provider = Some(decl.value.clone()),
                    StyleProperty::ReasoningEffort => effort = Some(decl.value.clone()),
                }
            }
        }

        // Apply to node, but only if the field is currently empty (node wins over stylesheet)
        let node = graph.nodes.get_mut(&node_id).unwrap();
        if node.llm_model.is_empty() {
            if let Some(m) = model {
                node.llm_model = m;
            }
        }
        if node.llm_provider.is_empty() {
            if let Some(p) = provider {
                node.llm_provider = p;
            }
        }
        if node.reasoning_effort == "high" || node.reasoning_effort.is_empty() {
            // Only apply stylesheet reasoning_effort if the node hasn't explicitly set it
            // We treat "high" as the default (Node::default()), so if stylesheet sets something
            // different we need to check if the node had it explicitly set.
            // Since we can't distinguish "default high" from "explicit high" here,
            // we apply the stylesheet value only when the node's reasoning_effort is "high"
            // (the default) AND the stylesheet specifies something different.
            if let Some(e) = effort {
                // Don't override if node explicitly has reasoning_effort set to something
                // non-default. We do a best-effort check: only apply if field is "high" (default)
                // and stylesheet differs, OR if field is empty.
                if node.reasoning_effort.is_empty() || node.reasoning_effort == "high" {
                    node.reasoning_effort = e;
                }
            }
        }
    }
}

/// Check whether a rule's selector matches the given node.
fn rule_matches_node(rule: &StyleRule, node: &crate::graph::Node) -> bool {
    match &rule.selector {
        Selector::Universal => true,
        Selector::Id(id) => &node.id == id,
        Selector::Class(cls) => {
            // class is comma-separated list
            node.class.split(',').any(|c| c.trim() == cls.as_str())
        }
        Selector::Shape(shape) => &node.shape == shape,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Graph, Node};

    fn make_graph_with_nodes(nodes: Vec<(&str, &str, &str)>) -> Graph {
        // nodes: (id, shape, class)
        let mut g = Graph::new("test".to_string());
        for (id, shape, cls) in nodes {
            let n = Node {
                id: id.to_string(),
                shape: shape.to_string(),
                class: cls.to_string(),
                ..Default::default()
            };
            g.nodes.insert(id.to_string(), n);
        }
        g
    }

    #[test]
    fn parse_empty_stylesheet() {
        let ss = parse_stylesheet("").unwrap();
        assert!(ss.rules.is_empty());
    }

    #[test]
    fn parse_universal_rule() {
        let ss = parse_stylesheet("* { llm_model: claude-opus; }").unwrap();
        assert_eq!(ss.rules.len(), 1);
        assert_eq!(ss.rules[0].selector, Selector::Universal);
        assert_eq!(
            ss.rules[0].declarations[0].property,
            StyleProperty::LlmModel
        );
        assert_eq!(ss.rules[0].declarations[0].value, "claude-opus");
    }

    #[test]
    fn parse_class_selector() {
        let ss = parse_stylesheet(".code { llm_provider: anthropic; }").unwrap();
        assert_eq!(ss.rules[0].selector, Selector::Class("code".to_string()));
    }

    #[test]
    fn parse_id_selector() {
        let ss = parse_stylesheet("#my_node { reasoning_effort: high; }").unwrap();
        assert_eq!(ss.rules[0].selector, Selector::Id("my_node".to_string()));
        assert_eq!(
            ss.rules[0].declarations[0].property,
            StyleProperty::ReasoningEffort
        );
    }

    #[test]
    fn parse_multiple_rules() {
        let src = r#"
            * { llm_model: gpt-4; llm_provider: openai; }
            .fast { llm_model: gemini-flash; }
            #review { reasoning_effort: high; }
        "#;
        let ss = parse_stylesheet(src).unwrap();
        assert_eq!(ss.rules.len(), 3);
    }

    #[test]
    fn parse_quoted_value() {
        let ss = parse_stylesheet(r#"* { llm_model: "gpt-5.2"; }"#).unwrap();
        assert_eq!(ss.rules[0].declarations[0].value, "gpt-5.2");
    }

    #[test]
    fn parse_invalid_property_returns_error() {
        let result = parse_stylesheet("* { unknown_prop: val; }");
        assert!(result.is_err());
    }

    #[test]
    fn apply_universal_rule() {
        let mut g = make_graph_with_nodes(vec![("A", "box", ""), ("B", "box", "")]);
        let ss = parse_stylesheet("* { llm_model: claude; llm_provider: anthropic; }").unwrap();
        apply_stylesheet(&ss, &mut g);
        assert_eq!(g.nodes["A"].llm_model, "claude");
        assert_eq!(g.nodes["B"].llm_provider, "anthropic");
    }

    #[test]
    fn apply_class_rule() {
        let mut g = make_graph_with_nodes(vec![("A", "box", "code"), ("B", "box", "")]);
        let ss = parse_stylesheet(".code { llm_model: opus; }").unwrap();
        apply_stylesheet(&ss, &mut g);
        assert_eq!(g.nodes["A"].llm_model, "opus");
        assert_eq!(g.nodes["B"].llm_model, ""); // no class match
    }

    #[test]
    fn apply_id_rule() {
        let mut g = make_graph_with_nodes(vec![("review", "box", "code"), ("other", "box", "")]);
        let ss =
            parse_stylesheet(".code { llm_model: sonnet; } #review { llm_model: gpt-5; }").unwrap();
        apply_stylesheet(&ss, &mut g);
        // review: ID rule (spec=2) overrides class rule (spec=1)
        assert_eq!(g.nodes["review"].llm_model, "gpt-5");
        // other: no match
        assert_eq!(g.nodes["other"].llm_model, "");
    }

    #[test]
    fn explicit_node_attr_not_overridden() {
        let mut g = make_graph_with_nodes(vec![("A", "box", "")]);
        g.nodes.get_mut("A").unwrap().llm_model = "explicit-model".to_string();
        let ss = parse_stylesheet("* { llm_model: universal-model; }").unwrap();
        apply_stylesheet(&ss, &mut g);
        assert_eq!(g.nodes["A"].llm_model, "explicit-model");
    }

    #[test]
    fn nlspec_example_specificity() {
        // NLSpec §8.6 example
        let src = r#"
            * { llm_model: claude-sonnet-4-5; llm_provider: anthropic; }
            .code { llm_model: claude-opus-4-6; llm_provider: anthropic; }
            #critical_review { llm_model: gpt-5-2; llm_provider: openai; reasoning_effort: high; }
        "#;
        let mut g = make_graph_with_nodes(vec![
            ("plan", "box", "planning"),
            ("implement", "box", "code"),
            ("critical_review", "box", "code"),
        ]);
        let ss = parse_stylesheet(src).unwrap();
        apply_stylesheet(&ss, &mut g);
        // plan: universal match → claude-sonnet-4-5
        assert_eq!(g.nodes["plan"].llm_model, "claude-sonnet-4-5");
        // implement: class .code match → claude-opus-4-6
        assert_eq!(g.nodes["implement"].llm_model, "claude-opus-4-6");
        // critical_review: ID match overrides class match → gpt-5-2
        assert_eq!(g.nodes["critical_review"].llm_model, "gpt-5-2");
        assert_eq!(g.nodes["critical_review"].llm_provider, "openai");
    }

    #[test]
    fn apply_with_comments_stripped() {
        let src = "/* global defaults */ * { llm_model: base-model; } // end";
        let ss = parse_stylesheet(src).unwrap();
        let mut g = make_graph_with_nodes(vec![("A", "box", "")]);
        apply_stylesheet(&ss, &mut g);
        assert_eq!(g.nodes["A"].llm_model, "base-model");
    }

    #[test]
    fn specificity_values() {
        // V2-ATR-004: NLSpec §8 specificity order: universal < shape < class < ID
        assert_eq!(Selector::Universal.specificity(), 0);
        assert_eq!(Selector::Shape("box".into()).specificity(), 1);
        assert_eq!(Selector::Class("c".into()).specificity(), 2);
        assert_eq!(Selector::Id("i".into()).specificity(), 3);
    }

    #[test]
    fn class_wins_over_shape() {
        // V2-ATR-004: A class rule must override a shape rule for a node that
        // matches both, because class (specificity=2) > shape (specificity=1).
        // The class rule is listed FIRST — it only wins if specificity
        // differentiates. If both had equal specificity, the shape rule (listed
        // second, higher index) would win by last-write-wins.
        let src = r#"
            .fast { llm_model: class-model; }
            box { llm_model: shape-model; }
        "#;
        let ss = parse_stylesheet(src).unwrap();
        let mut g = make_graph_with_nodes(vec![("task", "box", "fast")]);
        apply_stylesheet(&ss, &mut g);
        assert_eq!(
            g.nodes["task"].llm_model, "class-model",
            "class rule (specificity=2) must win over shape rule (specificity=1) \
             even when shape rule is listed later"
        );
    }

    // --- GAP-ATR-015: shape-name selector ---

    #[test]
    fn parse_shape_selector() {
        // GAP-ATR-015: a bare identifier in the stylesheet should be parsed as a
        // Shape selector, e.g. `box { llm_model: test-model; }`.
        let ss = parse_stylesheet("box { llm_model: test-model; }").unwrap();
        assert_eq!(ss.rules.len(), 1);
        assert_eq!(ss.rules[0].selector, Selector::Shape("box".to_string()));
        assert_eq!(
            ss.rules[0].declarations[0].property,
            StyleProperty::LlmModel
        );
        assert_eq!(ss.rules[0].declarations[0].value, "test-model");
    }

    #[test]
    fn apply_shape_rule_matches_nodes_with_matching_shape() {
        // GAP-ATR-015: a `box { ... }` rule should apply to nodes with shape=box
        // and NOT to nodes with a different shape.
        let mut g = make_graph_with_nodes(vec![
            ("task_a", "box", ""),
            ("task_b", "diamond", ""),
            ("task_c", "box", ""),
        ]);
        let ss = parse_stylesheet("box { llm_model: turbo; }").unwrap();
        apply_stylesheet(&ss, &mut g);

        // Nodes with shape=box get the model
        assert_eq!(g.nodes["task_a"].llm_model, "turbo");
        assert_eq!(g.nodes["task_c"].llm_model, "turbo");
        // Node with a different shape is unchanged
        assert_eq!(g.nodes["task_b"].llm_model, "");
    }

    #[test]
    fn shape_rule_overridden_by_id_rule() {
        // Shape (specificity=1) must be overridden by an ID rule (specificity=2).
        let mut g = make_graph_with_nodes(vec![("special", "box", "")]);
        let ss = parse_stylesheet(
            "box { llm_model: generic-model; } #special { llm_model: specific-model; }",
        )
        .unwrap();
        apply_stylesheet(&ss, &mut g);
        assert_eq!(g.nodes["special"].llm_model, "specific-model");
    }
}
