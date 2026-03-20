//! Pipeline event types emitted during execution.
//!
//! Events are distributed via a [`tokio::sync::broadcast`] channel. Consumers
//! subscribe before calling `PipelineRunner::run` and process events
//! independently (TUI, logging, metrics, etc.).

use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::sync::broadcast;

// ---------------------------------------------------------------------------
// Duration serde helper — same pattern as graph.rs
// ---------------------------------------------------------------------------

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
        let wrapper: DurationMs = serde::Deserialize::deserialize(d)?;
        Ok(Duration::from_millis(wrapper.ms))
    }
}

// ---------------------------------------------------------------------------
// PipelineEvent
// ---------------------------------------------------------------------------

/// Typed events emitted by the execution engine during a pipeline run.
///
/// Serialized with an `"event"` discriminant tag for JSON readability:
/// `{"event": "stage_started", "name": "...", "index": 0}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum PipelineEvent {
    // -- Pipeline lifecycle --
    /// Pipeline execution has begun.
    PipelineStarted { name: String, id: String },
    /// Pipeline completed successfully.
    PipelineCompleted {
        #[serde(with = "duration_millis_serde")]
        duration: Duration,
        artifact_count: usize,
    },
    /// Pipeline failed with an error.
    PipelineFailed {
        error: String,
        #[serde(with = "duration_millis_serde")]
        duration: Duration,
    },

    // -- Stage lifecycle --
    /// A node handler is about to execute.
    StageStarted { name: String, index: usize },
    /// A node handler completed successfully.
    StageCompleted {
        name: String,
        index: usize,
        #[serde(with = "duration_millis_serde")]
        duration: Duration,
    },
    /// A node handler failed.
    StageFailed {
        name: String,
        index: usize,
        error: String,
        will_retry: bool,
    },
    /// A node handler is being retried.
    StageRetrying {
        name: String,
        index: usize,
        attempt: u32,
        #[serde(with = "duration_millis_serde")]
        delay: Duration,
    },

    // -- Parallel execution --
    /// A parallel fan-out node started.
    ParallelStarted { branch_count: usize },
    /// A single parallel branch started.
    ParallelBranchStarted { branch: String, index: usize },
    /// A single parallel branch completed.
    ParallelBranchCompleted {
        branch: String,
        index: usize,
        #[serde(with = "duration_millis_serde")]
        duration: Duration,
        success: bool,
        /// Error message when the branch failed.  `None` for successful branches.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// All parallel branches have completed.
    ParallelCompleted {
        #[serde(with = "duration_millis_serde")]
        duration: Duration,
        success_count: usize,
        failure_count: usize,
    },

    // -- Human interaction --
    /// An interviewer question was presented.
    InterviewStarted { question: String, stage: String },
    /// An interviewer question was answered.
    InterviewCompleted {
        question: String,
        answer: String,
        #[serde(with = "duration_millis_serde")]
        duration: Duration,
    },
    /// An interviewer question timed out.
    InterviewTimeout {
        question: String,
        stage: String,
        #[serde(with = "duration_millis_serde")]
        duration: Duration,
    },

    // -- Checkpoint --
    /// A checkpoint was written to disk.
    CheckpointSaved { node_id: String },
}

// ---------------------------------------------------------------------------
// Channel helpers
// ---------------------------------------------------------------------------

/// Default capacity for the pipeline event broadcast channel.
pub const EVENT_CHANNEL_CAPACITY: usize = 256;

/// Create a new broadcast channel pair for pipeline events.
pub fn event_channel() -> (
    broadcast::Sender<PipelineEvent>,
    broadcast::Receiver<PipelineEvent>,
) {
    broadcast::channel(EVENT_CHANNEL_CAPACITY)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_started_serializes_with_tag() {
        let ev = PipelineEvent::PipelineStarted {
            name: "test".into(),
            id: "abc".into(),
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains("\"event\":\"pipeline_started\""));
        assert!(json.contains("\"name\":\"test\""));
    }

    #[test]
    fn stage_completed_duration_roundtrip() {
        let ev = PipelineEvent::StageCompleted {
            name: "plan".into(),
            index: 1,
            duration: Duration::from_millis(1500),
        };
        let json = serde_json::to_string(&ev).unwrap();
        let back: PipelineEvent = serde_json::from_str(&json).unwrap();
        match back {
            PipelineEvent::StageCompleted { duration, .. } => {
                assert_eq!(duration, Duration::from_millis(1500));
            }
            _ => panic!("unexpected variant"),
        }
    }

    #[test]
    fn event_channel_send_recv() {
        let (tx, mut rx) = event_channel();
        let ev = PipelineEvent::CheckpointSaved {
            node_id: "plan".into(),
        };
        tx.send(ev.clone()).unwrap();
        let received = rx.try_recv().unwrap();
        matches!(received, PipelineEvent::CheckpointSaved { .. });
    }

    #[test]
    fn multiple_receivers_each_get_event() {
        let (tx, mut rx1) = event_channel();
        let mut rx2 = tx.subscribe();
        let ev = PipelineEvent::PipelineStarted {
            name: "p".into(),
            id: "1".into(),
        };
        tx.send(ev).unwrap();
        rx1.try_recv().unwrap();
        rx2.try_recv().unwrap();
    }

    #[test]
    fn pipeline_event_clone() {
        let ev = PipelineEvent::StageStarted {
            name: "x".into(),
            index: 0,
        };
        let _cloned = ev.clone();
    }

    #[test]
    fn parallel_branch_completed_with_error_serializes() {
        let ev = PipelineEvent::ParallelBranchCompleted {
            branch: "ReviewGemini".into(),
            index: 2,
            duration: Duration::from_millis(500),
            success: false,
            error: Some("authentication error (gemini): invalid api key".into()),
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains("\"error\""));
        assert!(json.contains("invalid api key"));
        let back: PipelineEvent = serde_json::from_str(&json).unwrap();
        match back {
            PipelineEvent::ParallelBranchCompleted { error, success, .. } => {
                assert!(!success);
                assert_eq!(
                    error.as_deref(),
                    Some("authentication error (gemini): invalid api key")
                );
            }
            _ => panic!("unexpected variant"),
        }
    }

    #[test]
    fn parallel_branch_completed_without_error_omits_field() {
        let ev = PipelineEvent::ParallelBranchCompleted {
            branch: "ReviewOpus".into(),
            index: 0,
            duration: Duration::from_millis(200),
            success: true,
            error: None,
        };
        let json = serde_json::to_string(&ev).unwrap();
        // error field should be absent when None (skip_serializing_if)
        assert!(!json.contains("\"error\""));
    }

    #[test]
    fn parallel_branch_completed_backward_compat_deserialize() {
        // Old JSON without error field should deserialize fine
        let json = r#"{"event":"parallel_branch_completed","branch":"A","index":0,"duration":{"__duration_ms":100},"success":true}"#;
        let ev: PipelineEvent = serde_json::from_str(json).unwrap();
        match ev {
            PipelineEvent::ParallelBranchCompleted { error, success, .. } => {
                assert!(success);
                assert!(error.is_none());
            }
            _ => panic!("unexpected variant"),
        }
    }
}
