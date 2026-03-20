//! Conditional handler — no-op for `shape=diamond` routing nodes.

use async_trait::async_trait;
use std::path::Path;

use crate::error::EngineError;
use crate::graph::{Graph, Node};
use crate::handler::Handler;
use crate::state::context::{Context, Outcome};

/// No-op handler for `shape=diamond` (conditional) nodes.
///
/// Returns [`Outcome::success`] with a descriptive `notes` field.  All actual
/// routing logic lives in the engine's edge-selection algorithm (which
/// evaluates `condition` attributes on outgoing edges).
pub struct ConditionalHandler;

#[async_trait]
impl Handler for ConditionalHandler {
    async fn execute(
        &self,
        node: &Node,
        _context: &Context,
        _graph: &Graph,
        _logs_root: &Path,
    ) -> Result<Outcome, EngineError> {
        Ok(Outcome {
            notes: format!("Conditional node evaluated: {}", node.id),
            ..Outcome::success()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::state::context::StageStatus;

    #[tokio::test]
    async fn returns_success_with_notes() {
        let handler = ConditionalHandler;
        let node = Node {
            id: "gate".to_string(),
            ..Default::default()
        };
        let ctx = Context::new();
        let graph = Graph::new("test".into());
        let result = handler
            .execute(&node, &ctx, &graph, Path::new("/tmp"))
            .await
            .unwrap();
        assert_eq!(result.status, StageStatus::Success);
        assert!(
            result.notes.contains("gate"),
            "notes should contain node id"
        );
    }
}
