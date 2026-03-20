//! Core graph model types produced by the DOT parser.
//!
//! [`Graph`] is the in-memory representation of a parsed `.dot` file.
//! [`Node`] and [`Edge`] hold per-statement data.
//! [`Value`] is the typed attribute value enum.
//! [`GraphAttrs`] holds graph-level configuration attributes.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Bool coercion helper
// ---------------------------------------------------------------------------

/// Coerce a `Value` to `bool`.
///
/// Accepts:
/// - `Value::Bool(b)` directly.
/// - `Value::Str(s)` where `s` case-insensitively equals `"true"` → `true`,
///   or `"false"` → `false`.
///
/// Returns `None` for any other variant or unrecognised string.
fn value_as_bool(v: &Value) -> Option<bool> {
    match v {
        Value::Bool(b) => Some(*b),
        Value::Str(s) => match s.to_ascii_lowercase().as_str() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Value — typed attribute value
// ---------------------------------------------------------------------------

/// A typed attribute value parsed from a DOT attribute block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    /// A double-quoted string (escape sequences resolved).
    Str(String),
    /// A signed integer.
    Int(i64),
    /// A floating-point number.
    Float(f64),
    /// A boolean literal (`true` / `false`).
    Bool(bool),
    /// A duration with unit suffix (e.g., `900s`, `15m`).
    #[serde(with = "duration_millis_serde")]
    Duration(Duration),
}

impl Value {
    /// Return the contained string, or `None` if a different variant.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Coerce any variant to a string representation.
    pub fn to_string_repr(&self) -> String {
        match self {
            Value::Str(s) => s.clone(),
            Value::Int(n) => n.to_string(),
            Value::Float(f) => f.to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Duration(d) => format!("{}ms", d.as_millis()),
        }
    }
}

/// Custom serde module: serialize `Duration` as `{"__duration_ms": N}`.
///
/// Using an object shape means it won't be mistaken for `Value::Int` when
/// deserializing an untagged enum.
mod duration_millis_serde {
    use serde::{Deserializer, Serialize, Serializer};
    use std::time::Duration;

    #[derive(serde::Serialize, serde::Deserialize)]
    struct DurationMs {
        #[serde(rename = "__duration_ms")]
        ms: u64,
    }

    pub fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        DurationMs {
            ms: d.as_millis() as u64,
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        // Fully-qualified call avoids needing `serde::Deserialize` in scope
        // while the function itself is also named `deserialize`.
        let wrapper: DurationMs = serde::Deserialize::deserialize(d)?;
        Ok(Duration::from_millis(wrapper.ms))
    }
}

// ---------------------------------------------------------------------------
// GraphAttrs — graph-level configuration
// ---------------------------------------------------------------------------

/// Graph-level configuration attributes (`graph [ ... ]` or bare `key = value` at graph scope).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphAttrs {
    /// Human-readable pipeline goal. Exposed as `$goal` in prompt templates.
    pub goal: String,
    /// Display name for the graph (used in visualisation).
    pub label: String,
    /// CSS-like model stylesheet string. Parsed by `stylesheet::parse_stylesheet`.
    pub model_stylesheet: String,
    /// Global retry ceiling for nodes that omit `max_retries`. Default 50.
    pub default_max_retry: u32,
    /// Node ID to jump to if exit is reached with unsatisfied goal gates.
    pub retry_target: String,
    /// Secondary jump target if `retry_target` is invalid.
    pub fallback_retry_target: String,
    /// Default context fidelity mode (e.g., `"compact"`).
    pub default_fidelity: String,
    /// Default thread ID for `full` fidelity sessions.
    pub default_thread_id: String,
    /// Unknown / extra graph-level attributes.
    #[serde(default)]
    pub extra: HashMap<String, Value>,
}

impl GraphAttrs {
    /// Build a `GraphAttrs` from the parsed raw attribute map.
    pub fn from_attrs(attrs: HashMap<String, Value>) -> Self {
        let mut ga = GraphAttrs {
            default_max_retry: 50,
            ..Default::default()
        };
        let mut extra = HashMap::new();
        for (key, val) in attrs {
            match key.as_str() {
                "goal" => ga.goal = val.to_string_repr(),
                "label" => ga.label = val.to_string_repr(),
                "model_stylesheet" => ga.model_stylesheet = val.to_string_repr(),
                "default_max_retry" => {
                    if let Value::Int(n) = &val {
                        ga.default_max_retry = (*n).max(0) as u32;
                    }
                }
                "retry_target" => ga.retry_target = val.to_string_repr(),
                "fallback_retry_target" => ga.fallback_retry_target = val.to_string_repr(),
                "default_fidelity" | "context_fidelity_default" => {
                    ga.default_fidelity = val.to_string_repr()
                }
                "default_thread_id" | "context_thread_default" => {
                    ga.default_thread_id = val.to_string_repr()
                }
                _ => {
                    extra.insert(key, val);
                }
            }
        }
        ga.extra = extra;
        ga
    }
}

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

/// A parsed node statement from the DOT source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Bare DOT identifier (used as the key in `Graph::nodes`).
    pub id: String,
    /// Human-readable display name. Defaults to `id` when omitted.
    pub label: String,
    /// Graphviz shape (determines handler type). Default `"box"`.
    pub shape: String,
    /// Explicit handler type override (`type` attribute). Empty → shape-based resolution.
    pub node_type: String,
    /// LLM prompt text. Supports `$goal` variable expansion.
    pub prompt: String,
    /// Additional retry attempts beyond the initial execution. Default 0.
    pub max_retries: u32,
    /// If `true`, this node must reach SUCCESS before the pipeline can exit.
    pub goal_gate: bool,
    /// Node ID to jump to on failure/exhaustion.
    pub retry_target: String,
    /// Secondary failure jump target.
    pub fallback_retry_target: String,
    /// Context fidelity mode for this node's LLM session.
    pub fidelity: String,
    /// Explicit thread key for LLM session reuse under `full` fidelity.
    pub thread_id: String,
    /// Comma-separated CSS-like class names for model stylesheet targeting.
    pub class: String,
    /// Maximum execution time for this node.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub timeout: Option<Duration>,
    /// LLM model identifier (overridable by stylesheet).
    pub llm_model: String,
    /// LLM provider key (auto-detected from model if empty).
    pub llm_provider: String,
    /// LLM reasoning effort: `"low"`, `"medium"`, `"high"`. Default `"high"`.
    pub reasoning_effort: String,
    /// If `true` and no status was written, the engine auto-generates SUCCESS.
    pub auto_status: bool,
    /// Accept PARTIAL_SUCCESS when retries are exhausted.
    pub allow_partial: bool,
    /// Unknown / extra node attributes.
    #[serde(default)]
    pub extra: HashMap<String, Value>,
}

impl Default for Node {
    fn default() -> Self {
        Node {
            id: String::new(),
            label: String::new(),
            shape: "box".to_string(),
            node_type: String::new(),
            prompt: String::new(),
            max_retries: 0,
            goal_gate: false,
            retry_target: String::new(),
            fallback_retry_target: String::new(),
            fidelity: String::new(),
            thread_id: String::new(),
            class: String::new(),
            timeout: None,
            llm_model: String::new(),
            llm_provider: String::new(),
            reasoning_effort: "high".to_string(),
            auto_status: false,
            allow_partial: false,
            extra: HashMap::new(),
        }
    }
}

impl Node {
    /// Apply a raw attribute map on top of this node's fields.
    /// Known keys set struct fields; unknown keys go to `extra`.
    pub fn apply_attrs(&mut self, attrs: HashMap<String, Value>) {
        for (key, val) in attrs {
            match key.as_str() {
                "label" => self.label = val.to_string_repr(),
                "shape" => self.shape = val.to_string_repr(),
                "type" => self.node_type = val.to_string_repr(),
                "prompt" => self.prompt = val.to_string_repr(),
                "max_retries" => {
                    if let Value::Int(n) = &val {
                        self.max_retries = (*n).max(0) as u32;
                    }
                }
                "goal_gate" => {
                    if let Some(b) = value_as_bool(&val) {
                        self.goal_gate = b;
                    }
                }
                "retry_target" => self.retry_target = val.to_string_repr(),
                "fallback_retry_target" => self.fallback_retry_target = val.to_string_repr(),
                "fidelity" => self.fidelity = val.to_string_repr(),
                "thread_id" => self.thread_id = val.to_string_repr(),
                "class" => self.class = val.to_string_repr(),
                "timeout" => {
                    if let Value::Duration(d) = &val {
                        self.timeout = Some(*d);
                    }
                }
                "llm_model" => self.llm_model = val.to_string_repr(),
                "llm_provider" => self.llm_provider = val.to_string_repr(),
                "reasoning_effort" => self.reasoning_effort = val.to_string_repr(),
                "auto_status" => {
                    if let Some(b) = value_as_bool(&val) {
                        self.auto_status = b;
                    }
                }
                "allow_partial" => {
                    if let Some(b) = value_as_bool(&val) {
                        self.allow_partial = b;
                    }
                }
                _ => {
                    self.extra.insert(key, val);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Edge
// ---------------------------------------------------------------------------

/// A parsed edge statement from the DOT source.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Edge {
    /// Source node ID.
    pub from: String,
    /// Target node ID.
    pub to: String,
    /// Human-facing caption and routing key. Used for preferred-label matching.
    pub label: String,
    /// Boolean guard expression (evaluated by condition parser at runtime).
    pub condition: String,
    /// Numeric priority for edge selection. Higher weight wins. Default 0.
    pub weight: i32,
    /// Override fidelity mode for the target node.
    pub fidelity: String,
    /// Override thread ID for session reuse at the target node.
    pub thread_id: String,
    /// When `true`, terminates the current run and re-launches with a fresh log directory.
    pub loop_restart: bool,
}

impl Edge {
    /// Apply a raw attribute map to this edge's fields.
    pub fn apply_attrs(&mut self, attrs: &HashMap<String, Value>) {
        for (key, val) in attrs {
            match key.as_str() {
                "label" => self.label = val.to_string_repr(),
                "condition" => self.condition = val.to_string_repr(),
                "weight" => {
                    if let Value::Int(n) = val {
                        self.weight = *n as i32;
                    }
                }
                "fidelity" => self.fidelity = val.to_string_repr(),
                "thread_id" => self.thread_id = val.to_string_repr(),
                "loop_restart" => {
                    if let Some(b) = value_as_bool(val) {
                        self.loop_restart = b;
                    }
                }
                _ => {} // unknown edge attrs are silently ignored
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

/// In-memory representation of a parsed DOT pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    /// Name from `digraph Name { ... }` (may be empty for anonymous graphs).
    pub name: String,
    /// Nodes keyed by their DOT identifier, in declaration order.
    pub nodes: IndexMap<String, Node>,
    /// Directed edges in declaration order.
    pub edges: Vec<Edge>,
    /// Graph-level configuration attributes.
    pub graph_attrs: GraphAttrs,
    /// Accumulated node defaults (`node [...]` blocks).
    #[serde(default)]
    pub node_defaults: HashMap<String, Value>,
    /// Accumulated edge defaults (`edge [...]` blocks).
    #[serde(default)]
    pub edge_defaults: HashMap<String, Value>,
}

impl Graph {
    /// Create an empty graph with the given name.
    pub fn new(name: String) -> Self {
        Graph {
            name,
            nodes: IndexMap::new(),
            edges: Vec::new(),
            graph_attrs: GraphAttrs {
                default_max_retry: 50,
                ..Default::default()
            },
            node_defaults: HashMap::new(),
            edge_defaults: HashMap::new(),
        }
    }

    /// Return all outgoing edges from `node_id`.
    pub fn outgoing_edges(&self, node_id: &str) -> Vec<&Edge> {
        self.edges.iter().filter(|e| e.from == node_id).collect()
    }

    /// Return all incoming edges to `node_id`.
    pub fn incoming_edges(&self, node_id: &str) -> Vec<&Edge> {
        self.edges.iter().filter(|e| e.to == node_id).collect()
    }

    /// Return the start node (first node with `shape == "Mdiamond"`), or `None`.
    pub fn start_node(&self) -> Option<&Node> {
        self.nodes.values().find(|n| n.shape == "Mdiamond")
    }

    /// Return the first exit node (first node with `shape == "Msquare"`), or `None`.
    pub fn exit_node(&self) -> Option<&Node> {
        self.nodes.values().find(|n| n.shape == "Msquare")
    }

    /// Return a reference to the node with the given ID, or `None`.
    pub fn node(&self, id: &str) -> Option<&Node> {
        self.nodes.get(id)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph() -> Graph {
        let mut g = Graph::new("test".to_string());
        let start = Node {
            id: "start".to_string(),
            shape: "Mdiamond".to_string(),
            ..Default::default()
        };
        g.nodes.insert("start".to_string(), start);

        let middle = Node {
            id: "middle".to_string(),
            ..Default::default()
        };
        g.nodes.insert("middle".to_string(), middle);

        let exit = Node {
            id: "exit".to_string(),
            shape: "Msquare".to_string(),
            ..Default::default()
        };
        g.nodes.insert("exit".to_string(), exit);

        g.edges.push(Edge {
            from: "start".to_string(),
            to: "middle".to_string(),
            ..Default::default()
        });
        g.edges.push(Edge {
            from: "middle".to_string(),
            to: "exit".to_string(),
            ..Default::default()
        });
        g
    }

    #[test]
    fn node_default_shape() {
        let n = Node::default();
        assert_eq!(n.shape, "box");
        assert_eq!(n.reasoning_effort, "high");
        assert_eq!(n.max_retries, 0);
        assert!(!n.goal_gate);
        assert!(!n.auto_status);
    }

    #[test]
    fn graph_attrs_default_max_retry() {
        let ga = GraphAttrs::default();
        // GraphAttrs::default() uses derive Default so default_max_retry=0
        // but Graph::new sets it to 50
        let g = Graph::new("x".to_string());
        assert_eq!(g.graph_attrs.default_max_retry, 50);
        let _ = ga; // suppress unused
    }

    #[test]
    fn graph_start_exit_node() {
        let g = make_graph();
        assert_eq!(g.start_node().unwrap().id, "start");
        assert_eq!(g.exit_node().unwrap().id, "exit");
    }

    #[test]
    fn graph_outgoing_incoming_edges() {
        let g = make_graph();
        let out = g.outgoing_edges("start");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].to, "middle");

        let inc = g.incoming_edges("exit");
        assert_eq!(inc.len(), 1);
        assert_eq!(inc[0].from, "middle");
    }

    #[test]
    fn graph_node_lookup() {
        let g = make_graph();
        assert!(g.node("start").is_some());
        assert!(g.node("nonexistent").is_none());
    }

    #[test]
    fn value_duration_serde_roundtrip() {
        let v = Value::Duration(Duration::from_millis(900_000));
        let json = serde_json::to_string(&v).unwrap();
        let back: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn value_to_string_repr() {
        assert_eq!(Value::Str("hello".into()).to_string_repr(), "hello");
        assert_eq!(Value::Int(42).to_string_repr(), "42");
        assert_eq!(Value::Bool(true).to_string_repr(), "true");
        assert_eq!(
            Value::Duration(Duration::from_secs(1)).to_string_repr(),
            "1000ms"
        );
    }

    #[test]
    fn graph_attrs_from_attrs() {
        let mut attrs = HashMap::new();
        attrs.insert("goal".to_string(), Value::Str("run tests".to_string()));
        attrs.insert("default_max_retry".to_string(), Value::Int(10));
        attrs.insert("rankdir".to_string(), Value::Str("LR".to_string()));
        let ga = GraphAttrs::from_attrs(attrs);
        assert_eq!(ga.goal, "run tests");
        assert_eq!(ga.default_max_retry, 10);
        assert!(ga.extra.contains_key("rankdir"));
    }

    #[test]
    fn node_apply_attrs_sets_fields() {
        let mut n = Node {
            id: "my_node".to_string(),
            ..Default::default()
        };
        let mut attrs = HashMap::new();
        attrs.insert("shape".to_string(), Value::Str("Mdiamond".to_string()));
        attrs.insert("max_retries".to_string(), Value::Int(3));
        attrs.insert("goal_gate".to_string(), Value::Bool(true));
        attrs.insert(
            "timeout".to_string(),
            Value::Duration(Duration::from_secs(900)),
        );
        n.apply_attrs(attrs);
        assert_eq!(n.shape, "Mdiamond");
        assert_eq!(n.max_retries, 3);
        assert!(n.goal_gate);
        assert_eq!(n.timeout, Some(Duration::from_secs(900)));
    }

    #[test]
    fn edge_apply_attrs() {
        let mut e = Edge::default();
        let mut attrs = HashMap::new();
        attrs.insert("label".to_string(), Value::Str("Yes".to_string()));
        attrs.insert("weight".to_string(), Value::Int(5));
        attrs.insert(
            "condition".to_string(),
            Value::Str("outcome=success".to_string()),
        );
        e.apply_attrs(&attrs);
        assert_eq!(e.label, "Yes");
        assert_eq!(e.weight, 5);
        assert_eq!(e.condition, "outcome=success");
    }

    // --- V2-ATR-001: Bool coercion ---

    #[test]
    fn edge_apply_attrs_loop_restart_string_true() {
        // V2-ATR-001: loop_restart="true" (a quoted string → Value::Str) must be
        // coerced to bool true, not silently ignored.
        let mut e = Edge::default();
        assert!(!e.loop_restart, "loop_restart starts false");
        let mut attrs = HashMap::new();
        attrs.insert("loop_restart".to_string(), Value::Str("true".to_string()));
        e.apply_attrs(&attrs);
        assert!(
            e.loop_restart,
            "loop_restart=\"true\" must set loop_restart=true"
        );
    }

    #[test]
    fn edge_apply_attrs_loop_restart_string_false() {
        // V2-ATR-001: loop_restart="false" must set loop_restart=false.
        let mut e = Edge {
            loop_restart: true,
            ..Default::default()
        };
        let mut attrs = HashMap::new();
        attrs.insert("loop_restart".to_string(), Value::Str("false".to_string()));
        e.apply_attrs(&attrs);
        assert!(
            !e.loop_restart,
            "loop_restart=\"false\" must set loop_restart=false"
        );
    }

    #[test]
    fn node_apply_attrs_goal_gate_string_true() {
        // V2-ATR-001: goal_gate="true" (quoted string) must be coerced to bool true.
        let mut n = Node::default();
        assert!(!n.goal_gate, "goal_gate starts false");
        let mut attrs = HashMap::new();
        attrs.insert("goal_gate".to_string(), Value::Str("true".to_string()));
        n.apply_attrs(attrs);
        assert!(n.goal_gate, "goal_gate=\"true\" must set goal_gate=true");
    }

    #[test]
    fn node_apply_attrs_auto_status_string_true() {
        // V2-ATR-001: auto_status="true" (quoted string) must be coerced to bool true.
        let mut n = Node::default();
        let mut attrs = HashMap::new();
        attrs.insert("auto_status".to_string(), Value::Str("true".to_string()));
        n.apply_attrs(attrs);
        assert!(
            n.auto_status,
            "auto_status=\"true\" must set auto_status=true"
        );
    }

    #[test]
    fn node_apply_attrs_allow_partial_string_true() {
        // V2-ATR-001: allow_partial="true" (quoted string) must be coerced.
        let mut n = Node::default();
        let mut attrs = HashMap::new();
        attrs.insert("allow_partial".to_string(), Value::Str("true".to_string()));
        n.apply_attrs(attrs);
        assert!(
            n.allow_partial,
            "allow_partial=\"true\" must set allow_partial=true"
        );
    }

    #[test]
    fn parse_context_fidelity_default_alias() {
        let mut attrs = HashMap::new();
        attrs.insert(
            "context_fidelity_default".to_string(),
            Value::Str("truncate".to_string()),
        );
        let ga = GraphAttrs::from_attrs(attrs);
        assert_eq!(
            ga.default_fidelity, "truncate",
            "context_fidelity_default must map to default_fidelity field"
        );
    }

    #[test]
    fn parse_context_thread_default() {
        let mut attrs = HashMap::new();
        attrs.insert(
            "context_thread_default".to_string(),
            Value::Str("my-thread".to_string()),
        );
        let ga = GraphAttrs::from_attrs(attrs);
        assert_eq!(
            ga.default_thread_id, "my-thread",
            "context_thread_default must map to default_thread_id field"
        );
    }

    #[test]
    fn both_fidelity_alias_and_canonical_work() {
        let mut attrs = HashMap::new();
        attrs.insert(
            "default_fidelity".to_string(),
            Value::Str("compact".to_string()),
        );
        let ga = GraphAttrs::from_attrs(attrs);
        assert_eq!(ga.default_fidelity, "compact");
    }
}
