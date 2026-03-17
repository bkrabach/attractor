//! attractor CLI binary entry point.
//!
//! Runs DOT-based AI pipelines from the command line.
//!
//! # Usage
//! ```bash
//! attractor run pipeline.dot
//! attractor run pipeline.dot --goal "Build feature X" --model gpt-4o
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use clap::{Parser, Subcommand};
use tracing_subscriber::{EnvFilter, fmt};

use attractor::engine::{PipelineRunner, RunConfig};
use attractor::error::EngineError;
use attractor::events::PipelineEvent;
use attractor::graph::Node;
use attractor::handler::{CodergenBackend, CodergenHandler, CodergenResult};
use attractor::state::context::Context;
use unified_llm::{Client, GenerateParams, generate};

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

/// DOT-based AI pipeline runner.
#[derive(Debug, Parser)]
#[command(
    name = "attractor",
    about = "Run DOT-based AI pipeline graphs from the command line",
    version
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

/// Available CLI sub-commands.
#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Parse and execute a DOT pipeline file.
    Run {
        /// Path to the DOT pipeline file.
        file: PathBuf,

        /// Override the pipeline goal defined in the DOT file.
        ///
        /// Replaces any `goal="..."` attribute in the graph header.
        #[arg(long)]
        goal: Option<String>,

        /// Default LLM model for codergen nodes that don't specify their own.
        ///
        /// Overrides the model selected by the stylesheet or node attribute.
        #[arg(long, default_value = "gpt-4o")]
        model: String,

        /// Directory for stage logs, checkpoints, and artifacts.
        ///
        /// Defaults to a temporary directory under `/tmp/attractor-<uuid>`.
        #[arg(long)]
        logs_dir: Option<PathBuf>,
    },
}

// ---------------------------------------------------------------------------
// LLM CodergenBackend
// ---------------------------------------------------------------------------

/// A [`CodergenBackend`] that calls the LLM API via [`unified_llm`].
///
/// Uses [`Client::from_env()`] so it picks up API keys from environment
/// variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`).
///
/// Model priority (highest to lowest):
/// 1. Node-level `llm_model` attribute (set by stylesheet or DOT directly)
/// 2. `default_model` supplied at construction time
pub struct LlmCodergenBackend {
    client: Client,
    default_model: String,
}

impl LlmCodergenBackend {
    pub fn new(client: Client, default_model: impl Into<String>) -> Self {
        LlmCodergenBackend {
            client,
            default_model: default_model.into(),
        }
    }
}

#[async_trait]
impl CodergenBackend for LlmCodergenBackend {
    async fn run(
        &self,
        node: &Node,
        prompt: &str,
        _ctx: &Context,
    ) -> Result<CodergenResult, EngineError> {
        // Honour the per-node model if the stylesheet/DOT set one.
        let model = if !node.llm_model.is_empty() {
            node.llm_model.clone()
        } else {
            self.default_model.clone()
        };

        let mut params = GenerateParams::new(model, prompt);
        params.client = Some(self.client.clone());

        // Route to the node's explicit provider when specified.
        // Without this, the Client defaults to the first registered provider,
        // which rejects model names belonging to other providers.
        if !node.llm_provider.is_empty() {
            params.provider = Some(node.llm_provider.clone());
        }

        let result = generate(params).await.map_err(|e| EngineError::Handler {
            node_id: node.id.clone(),
            message: format!("LLM call failed: {e}"),
        })?;

        Ok(CodergenResult::Text(result.text))
    }
}

// ---------------------------------------------------------------------------
// Goal injection helper
// ---------------------------------------------------------------------------

/// Patch the DOT source to set (or replace) the `goal` attribute.
///
/// If a `goal=` attribute already exists in the graph header it is replaced in-place.
/// Otherwise the function inserts `goal="<value>"` after the opening `[` of the
/// first `graph [...]` block.  If no such block exists the source is returned
/// unchanged (the engine will use an empty goal).
pub fn inject_goal(dot: &str, goal: &str) -> String {
    use regex::Regex;

    // Replace an existing goal value.
    let re_existing = Regex::new(r#"(?i)(goal\s*=\s*")([^"]*)(")"#).expect("valid regex");
    if re_existing.is_match(dot) {
        return re_existing
            .replace(dot, format!(r#"${{1}}{goal}${{3}}"#))
            .into_owned();
    }

    // No goal attribute — insert one into the first `graph [...]` block.
    let re_graph_bracket = Regex::new(r#"(?i)(graph\s*\[)"#).expect("valid regex");
    if re_graph_bracket.is_match(dot) {
        return re_graph_bracket
            .replace(dot, format!(r#"${{1}}goal="{goal}", "#))
            .into_owned();
    }

    // Fallback — leave unchanged.
    dot.to_string()
}

// ---------------------------------------------------------------------------
// Event printer
// ---------------------------------------------------------------------------

/// Print a [`PipelineEvent`] to stdout in a human-readable format.
pub fn print_event(event: &PipelineEvent) {
    match event {
        PipelineEvent::PipelineStarted { name, id } => {
            println!("[pipeline] started  name={name:?}  id={id}");
        }
        PipelineEvent::PipelineCompleted {
            duration,
            artifact_count,
        } => {
            println!(
                "[pipeline] completed  duration={:.2?}  artifacts={artifact_count}",
                duration
            );
        }
        PipelineEvent::PipelineFailed { error, duration } => {
            println!(
                "[pipeline] FAILED  duration={:.2?}  error={error}",
                duration
            );
        }
        PipelineEvent::StageStarted { name, index } => {
            println!("[stage #{index}] started  {name}");
        }
        PipelineEvent::StageCompleted {
            name,
            index,
            duration,
        } => {
            println!("[stage #{index}] completed  {name}  ({:.2?})", duration);
        }
        PipelineEvent::StageFailed {
            name,
            index,
            error,
            will_retry,
        } => {
            println!("[stage #{index}] FAILED  {name}  will_retry={will_retry}  error={error}");
        }
        PipelineEvent::StageRetrying {
            name,
            index,
            attempt,
            delay,
        } => {
            println!(
                "[stage #{index}] retrying  {name}  attempt={attempt}  delay={:.2?}",
                delay
            );
        }
        PipelineEvent::ParallelStarted { branch_count } => {
            println!("[parallel] started  branches={branch_count}");
        }
        PipelineEvent::ParallelBranchStarted { branch, index } => {
            println!("[parallel #{index}] branch started  {branch}");
        }
        PipelineEvent::ParallelBranchCompleted {
            branch,
            index,
            duration,
            success,
        } => {
            println!(
                "[parallel #{index}] branch completed  {branch}  success={success}  ({:.2?})",
                duration
            );
        }
        PipelineEvent::ParallelCompleted {
            duration,
            success_count,
            failure_count,
        } => {
            println!(
                "[parallel] completed  success={success_count}  failures={failure_count}  ({:.2?})",
                duration
            );
        }
        PipelineEvent::InterviewStarted { question, stage } => {
            println!("[interview] question at {stage}: {question}");
        }
        PipelineEvent::InterviewCompleted {
            question: _,
            answer,
            duration,
        } => {
            println!(
                "[interview] answered  answer={answer:?}  ({:.2?})",
                duration
            );
        }
        PipelineEvent::InterviewTimeout {
            question, stage, ..
        } => {
            println!("[interview] TIMEOUT at {stage}: {question}");
        }
        PipelineEvent::CheckpointSaved { node_id } => {
            println!("[checkpoint] saved  node={node_id}");
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline execution helper
// ---------------------------------------------------------------------------

/// Parse, validate, and run a pipeline from a DOT file.
///
/// Returns `Ok(())` on success, `Err(String)` with a human-readable message
/// on failure (so `main` can print it and exit with a non-zero code).
pub async fn run_pipeline(
    file: PathBuf,
    goal: Option<String>,
    model: String,
    logs_dir: Option<PathBuf>,
) -> Result<(), String> {
    // Read DOT source.
    let dot = tokio::fs::read_to_string(&file)
        .await
        .map_err(|e| format!("failed to read {}: {e}", file.display()))?;

    // Optionally inject goal override.
    let dot = match goal {
        Some(ref g) => inject_goal(&dot, g),
        None => dot,
    };

    // Build LLM client from environment variables.
    let client = Client::from_env().map_err(|e| {
        format!(
            "failed to create LLM client: {e}\n\
             hint: set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY"
        )
    })?;

    // Wire up CodergenHandler with the real LLM backend.
    let backend = Box::new(LlmCodergenBackend::new(client, model));
    let codergen = Arc::new(CodergenHandler::new(Some(backend)));
    let (runner, mut events) = PipelineRunner::builder()
        .with_handler("codergen", codergen)
        .build();

    // Resolve logs directory.
    let logs_root = logs_dir.unwrap_or_else(|| {
        std::env::temp_dir().join(format!("attractor-{}", uuid::Uuid::new_v4()))
    });

    // Spawn background task to print events as they arrive.
    let print_task = tokio::spawn(async move {
        loop {
            match events.recv().await {
                Ok(ev) => print_event(&ev),
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    eprintln!("[warn] event stream lagged, {n} events dropped");
                }
            }
        }
    });

    println!("attractor: running {}", file.display());
    println!("attractor: logs → {}", logs_root.display());

    let config = RunConfig::new(&logs_root);
    let result = runner.run(&dot, config).await;

    // Wait for the event printer to drain.
    let _ = print_task.await;

    match result {
        Ok(run_result) => {
            println!(
                "\nattractor: pipeline finished — status={}",
                run_result.status
            );
            println!(
                "attractor: completed nodes: {}",
                run_result.completed_nodes.join(", ")
            );
            Ok(())
        }
        Err(e) => Err(format!("pipeline execution failed: {e}")),
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    // Initialise tracing — respects RUST_LOG, defaults to "warn" so normal
    // pipeline stdout output isn't mixed with tracing noise.
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            file,
            goal,
            model,
            logs_dir,
        } => {
            if let Err(e) = run_pipeline(file, goal, model, logs_dir).await {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    // ── Cli defaults ──────────────────────────────────────────────────────

    #[test]
    fn cli_run_minimal_args() {
        let cli = Cli::try_parse_from(["attractor", "run", "pipeline.dot"]).unwrap();
        match cli.command {
            Commands::Run {
                file,
                goal,
                model,
                logs_dir,
            } => {
                assert_eq!(file, PathBuf::from("pipeline.dot"));
                assert!(goal.is_none());
                assert_eq!(model, "gpt-4o");
                assert!(logs_dir.is_none());
            }
        }
    }

    #[test]
    fn cli_run_with_goal_and_model() {
        let cli = Cli::try_parse_from([
            "attractor",
            "run",
            "pipeline.dot",
            "--goal",
            "Build feature X",
            "--model",
            "claude-haiku-4-5-20251001",
        ])
        .unwrap();
        match cli.command {
            Commands::Run {
                file,
                goal,
                model,
                logs_dir,
            } => {
                assert_eq!(file, PathBuf::from("pipeline.dot"));
                assert_eq!(goal.as_deref(), Some("Build feature X"));
                assert_eq!(model, "claude-haiku-4-5-20251001");
                assert!(logs_dir.is_none());
            }
        }
    }

    #[test]
    fn cli_run_with_logs_dir() {
        let cli = Cli::try_parse_from([
            "attractor",
            "run",
            "pipeline.dot",
            "--logs-dir",
            "/tmp/my-logs",
        ])
        .unwrap();
        match cli.command {
            Commands::Run { logs_dir, .. } => {
                assert_eq!(logs_dir, Some(PathBuf::from("/tmp/my-logs")));
            }
        }
    }

    // ── inject_goal ────────────────────────────────────────────────────────

    #[test]
    fn inject_goal_replaces_existing_goal() {
        let dot = r#"digraph { graph [goal="old goal"] start [shape=Mdiamond] }"#;
        let patched = inject_goal(dot, "new goal");
        assert!(
            patched.contains(r#"goal="new goal""#),
            "existing goal must be replaced; got: {patched}"
        );
        assert!(
            !patched.contains("old goal"),
            "old goal must be removed; got: {patched}"
        );
    }

    #[test]
    fn inject_goal_inserts_into_graph_block() {
        let dot = r#"digraph { graph [label="pipeline"] start [shape=Mdiamond] }"#;
        let patched = inject_goal(dot, "inserted goal");
        assert!(
            patched.contains(r#"goal="inserted goal""#),
            "goal must be inserted when not present; got: {patched}"
        );
    }

    #[test]
    fn inject_goal_no_graph_block_returns_unchanged() {
        let dot = "digraph { start [shape=Mdiamond] }";
        let patched = inject_goal(dot, "my goal");
        // No graph [] block — source returned unchanged.
        assert_eq!(patched, dot);
    }

    // ── print_event ────────────────────────────────────────────────────────

    #[test]
    fn print_event_does_not_panic_on_all_variants() {
        use std::time::Duration;

        let events: Vec<PipelineEvent> = vec![
            PipelineEvent::PipelineStarted {
                name: "test".into(),
                id: "abc".into(),
            },
            PipelineEvent::PipelineCompleted {
                duration: Duration::from_millis(100),
                artifact_count: 3,
            },
            PipelineEvent::PipelineFailed {
                error: "boom".into(),
                duration: Duration::from_millis(50),
            },
            PipelineEvent::StageStarted {
                name: "plan".into(),
                index: 0,
            },
            PipelineEvent::StageCompleted {
                name: "plan".into(),
                index: 0,
                duration: Duration::from_millis(200),
            },
            PipelineEvent::StageFailed {
                name: "plan".into(),
                index: 0,
                error: "oops".into(),
                will_retry: true,
            },
            PipelineEvent::StageRetrying {
                name: "plan".into(),
                index: 0,
                attempt: 1,
                delay: Duration::from_millis(500),
            },
            PipelineEvent::ParallelStarted { branch_count: 3 },
            PipelineEvent::ParallelBranchStarted {
                branch: "a".into(),
                index: 0,
            },
            PipelineEvent::ParallelBranchCompleted {
                branch: "a".into(),
                index: 0,
                duration: Duration::from_millis(100),
                success: true,
            },
            PipelineEvent::ParallelCompleted {
                duration: Duration::from_millis(300),
                success_count: 2,
                failure_count: 1,
            },
            PipelineEvent::InterviewStarted {
                question: "approve?".into(),
                stage: "gate".into(),
            },
            PipelineEvent::InterviewCompleted {
                question: "approve?".into(),
                answer: "yes".into(),
                duration: Duration::from_millis(10),
            },
            PipelineEvent::InterviewTimeout {
                question: "approve?".into(),
                stage: "gate".into(),
                duration: Duration::from_millis(30_000),
            },
            PipelineEvent::CheckpointSaved {
                node_id: "plan".into(),
            },
        ];

        // Just verify no variant causes a panic.
        for ev in &events {
            print_event(ev); // stdout output is fine in tests
        }
    }
}
