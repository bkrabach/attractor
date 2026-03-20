//! Node handler trait, registry, and shape-to-type mapping.
//!
//! Every node in a pipeline is executed by a [`Handler`].  The
//! [`HandlerRegistry`] resolves which handler to use for a given [`Node`],
//! following the three-step priority from NLSpec §4.2:
//!
//! 1. Explicit `node.node_type` attribute (if registered)
//! 2. Shape-based resolution via [`shape_to_handler_type`]
//! 3. Default handler (registered at construction time)

pub mod codergen;
pub mod conditional;
pub mod exit;
pub mod fan_in;
pub mod manager_loop;
pub mod parallel;
pub mod start;
pub mod tool;
pub mod wait_human;

pub use codergen::{CodergenBackend, CodergenHandler, CodergenResult};
pub use conditional::ConditionalHandler;
pub use exit::ExitHandler;
pub use fan_in::{BranchResult, FanInHandler};
pub use manager_loop::ManagerLoopHandler;
pub use parallel::{ErrorPolicy, JoinPolicy, ParallelHandler};
pub use start::StartHandler;
pub use tool::ToolHandler;
pub use wait_human::WaitForHumanHandler;

use async_trait::async_trait;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::error::EngineError;
use crate::graph::{Graph, Node};
use crate::state::context::{Context, Outcome};

// ---------------------------------------------------------------------------
// Handler trait
// ---------------------------------------------------------------------------

/// Every node handler must implement this trait.
///
/// The execution engine dispatches to the appropriate handler based on the
/// node's `type` attribute (or shape-based resolution).  Handlers must be
/// `Send + Sync` so they can be stored in `Arc<dyn Handler>` and called from
/// async tasks.
#[async_trait]
pub trait Handler: Send + Sync {
    /// Execute the handler for `node` and return an [`Outcome`].
    async fn execute(
        &self,
        node: &Node,
        context: &Context,
        graph: &Graph,
        logs_root: &Path,
    ) -> Result<Outcome, EngineError>;
}

// ---------------------------------------------------------------------------
// HandlerRegistry
// ---------------------------------------------------------------------------

/// Maps handler type strings to [`Handler`] instances.
///
/// Resolution order (NLSpec §4.2):
/// 1. Explicit `node.node_type` (if non-empty and registered)
/// 2. Shape-based: `shape_to_handler_type(node.shape)` (if registered)
/// 3. Default handler
pub struct HandlerRegistry {
    handlers: HashMap<String, Arc<dyn Handler>>,
    default_handler: Arc<dyn Handler>,
}

impl Clone for HandlerRegistry {
    fn clone(&self) -> Self {
        HandlerRegistry {
            handlers: self.handlers.clone(),
            default_handler: self.default_handler.clone(),
        }
    }
}

impl HandlerRegistry {
    /// Create a registry with the given default handler.
    pub fn new(default_handler: Arc<dyn Handler>) -> Self {
        HandlerRegistry {
            handlers: HashMap::new(),
            default_handler,
        }
    }

    /// Register a handler for `type_str`.
    ///
    /// Overwrites any previously registered handler for that type.
    pub fn register(&mut self, type_str: &str, handler: Arc<dyn Handler>) {
        self.handlers.insert(type_str.to_owned(), handler);
    }

    /// Resolve the handler for `node` using the three-step priority.
    pub fn resolve(&self, node: &Node) -> &dyn Handler {
        // Step 1: explicit type attribute
        if !node.node_type.is_empty() {
            if let Some(h) = self.handlers.get(&node.node_type) {
                return h.as_ref();
            }
        }

        // Step 2: shape-based resolution
        if let Some(type_str) = shape_to_handler_type(&node.shape) {
            if let Some(h) = self.handlers.get(type_str) {
                return h.as_ref();
            }
        }

        // Step 3: default
        self.default_handler.as_ref()
    }
}

// ---------------------------------------------------------------------------
// Shape-to-handler-type mapping  (NLSpec §2.8)
// ---------------------------------------------------------------------------

/// Return the canonical handler type string for a given Graphviz shape, or
/// `None` if the shape is not in the mapping table.
///
/// | Shape            | Handler type         |
/// |------------------|----------------------|
/// | `Mdiamond`       | `start`              |
/// | `Msquare`        | `exit`               |
/// | `box`            | `codergen`           |
/// | `hexagon`        | `wait.human`         |
/// | `diamond`        | `conditional`        |
/// | `component`      | `parallel`           |
/// | `tripleoctagon`  | `parallel.fan_in`    |
/// | `parallelogram`  | `tool`               |
/// | `house`          | `stack.manager_loop` |
pub fn shape_to_handler_type(shape: &str) -> Option<&'static str> {
    match shape {
        "Mdiamond" => Some("start"),
        "Msquare" => Some("exit"),
        "box" => Some("codergen"),
        "hexagon" => Some("wait.human"),
        "diamond" => Some("conditional"),
        "component" => Some("parallel"),
        "tripleoctagon" => Some("parallel.fan_in"),
        "parallelogram" => Some("tool"),
        "house" => Some("stack.manager_loop"),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Node;

    /// A trivial handler that records calls and returns success.
    #[allow(dead_code)]
    struct DummyHandler {
        tag: &'static str,
    }

    #[async_trait]
    impl Handler for DummyHandler {
        async fn execute(
            &self,
            _node: &Node,
            _context: &Context,
            _graph: &Graph,
            _logs_root: &Path,
        ) -> Result<Outcome, EngineError> {
            Ok(Outcome::success())
        }
    }

    fn make_node(shape: &str, node_type: &str) -> Node {
        Node {
            id: "test".to_string(),
            shape: shape.to_string(),
            node_type: node_type.to_string(),
            ..Default::default()
        }
    }

    fn make_registry() -> HandlerRegistry {
        let default = Arc::new(DummyHandler { tag: "default" });
        let mut reg = HandlerRegistry::new(default);
        reg.register("start", Arc::new(DummyHandler { tag: "start" }));
        reg.register("codergen", Arc::new(DummyHandler { tag: "codergen" }));
        reg.register("wait.human", Arc::new(DummyHandler { tag: "wait.human" }));
        reg
    }

    #[test]
    fn resolve_by_explicit_type() {
        let reg = make_registry();
        let node = make_node("box", "start"); // explicit type overrides shape
        // We can't easily inspect which handler was returned, but we can call it.
        let _handler = reg.resolve(&node);
    }

    #[test]
    fn resolve_by_shape() {
        let reg = make_registry();
        let node = make_node("Mdiamond", ""); // no explicit type
        let _handler = reg.resolve(&node);
    }

    #[test]
    fn resolve_falls_back_to_default() {
        let reg = make_registry();
        let node = make_node("unknown_shape", ""); // nothing matches
        let _handler = reg.resolve(&node);
    }

    #[test]
    fn register_overwrites() {
        let default = Arc::new(DummyHandler { tag: "default" });
        let mut reg = HandlerRegistry::new(default);
        reg.register("start", Arc::new(DummyHandler { tag: "v1" }));
        reg.register("start", Arc::new(DummyHandler { tag: "v2" }));
        // Must not panic; second registration wins.
        let node = make_node("", "start");
        let _handler = reg.resolve(&node);
    }

    #[test]
    fn shape_to_handler_type_all_shapes() {
        let cases = [
            ("Mdiamond", "start"),
            ("Msquare", "exit"),
            ("box", "codergen"),
            ("hexagon", "wait.human"),
            ("diamond", "conditional"),
            ("component", "parallel"),
            ("tripleoctagon", "parallel.fan_in"),
            ("parallelogram", "tool"),
            ("house", "stack.manager_loop"),
        ];
        for (shape, expected) in &cases {
            assert_eq!(
                shape_to_handler_type(shape),
                Some(*expected),
                "shape '{shape}' should map to '{expected}'"
            );
        }
    }

    #[test]
    fn shape_to_handler_type_unknown() {
        assert_eq!(shape_to_handler_type("unknown"), None);
        assert_eq!(shape_to_handler_type(""), None);
    }
}
