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
use coding_agent_loop::profile::{anthropic_profile, gemini_profile, openai_profile};
use coding_agent_loop::turns::Turn;
use coding_agent_loop::{LocalExecutionEnvironment, Session, SessionConfig};
use regex::Regex;
use unified_llm::Client;

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

/// A [`CodergenBackend`] that runs a full agentic tool loop via
/// [`coding_agent_loop::Session`].
///
/// Each `run()` call creates a fresh `Session` with the appropriate
/// [`ProviderProfile`](coding_agent_loop::profile::ProviderProfile), submits
/// the prompt, and collects the final assistant text after all tool calls
/// have been executed.
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

/// Infer the LLM provider from a model name string.
///
/// Returns `"anthropic"`, `"gemini"`, or `"openai"` (the default fallback).
fn infer_provider(model: &str) -> &'static str {
    let m = model.trim().to_lowercase();
    if m.starts_with("claude") {
        "anthropic"
    } else if m.starts_with("gemini") {
        "gemini"
    } else {
        // Covers gpt-*, o1-*, o3-*, codex-*, and anything else.
        "openai"
    }
}

#[async_trait]
impl CodergenBackend for LlmCodergenBackend {
    async fn run(
        &self,
        node: &Node,
        prompt: &str,
        ctx: &Context,
    ) -> Result<CodergenResult, EngineError> {
        // 1. Resolve model: node-level override > default.
        let model = if !node.llm_model.is_empty() {
            node.llm_model.clone()
        } else {
            self.default_model.clone()
        };

        // 2. Resolve provider: node-level override > infer from model name.
        let provider_id = if !node.llm_provider.is_empty() {
            node.llm_provider.as_str()
        } else {
            infer_provider(&model)
        };

        // 3. Select the appropriate provider profile (tool set + system prompt).
        let profile = match provider_id {
            "anthropic" => anthropic_profile(&model),
            "gemini" => gemini_profile(&model),
            _ => openai_profile(&model),
        };

        // 4. Resolve working directory from pipeline context.
        let working_dir_str = ctx.get_string("_working_dir");
        let working_dir = if working_dir_str.is_empty() {
            std::env::current_dir().unwrap_or_else(|_| std::env::temp_dir())
        } else {
            PathBuf::from(&working_dir_str)
        };
        let env = Box::new(LocalExecutionEnvironment::new(&working_dir));

        // 5. Configure the session with reasonable limits for a pipeline node.
        //
        // Only override reasoning_effort when the DOT author explicitly set a
        // non-default value.  The parser defaults every node to "high", so
        // forwarding it unconditionally would inject reasoning_effort into
        // requests for models that don't support it (e.g. gpt-4o, gemini).
        // SessionConfig::default() leaves it as None, which is correct for
        // all models.  Only explicit "low"/"medium" overrides take effect.
        let config = SessionConfig {
            max_tool_rounds_per_input: 50, // prevent runaway tool loops
            reasoning_effort: if !node.reasoning_effort.is_empty()
                && node.reasoning_effort != "high"
            {
                Some(node.reasoning_effort.clone())
            } else {
                None
            },
            ..Default::default()
        };

        // 6. Create a fresh Session and run the agentic loop.
        let mut session = Session::new(config, profile, env, self.client.clone());

        session
            .submit(prompt)
            .await
            .map_err(|e| EngineError::Handler {
                node_id: node.id.clone(),
                message: format!("agent session failed: {e}"),
            })?;

        // 7. Extract the last non-empty assistant text from the session history.
        let text = session
            .history()
            .turns()
            .iter()
            .rev()
            .find_map(|t| match t {
                Turn::Assistant(a) if !a.content.is_empty() => Some(a.content.clone()),
                _ => None,
            })
            .unwrap_or_default();

        // 8. Graceful shutdown (flush events, clean up subagents).
        session.shutdown().await;

        Ok(CodergenResult::Text(text))
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
    // Escape `$` so it is never treated as a capture-group backreference by
    // the regex engine's replacement-string parser (`$$` → literal `$`).
    let safe_goal = goal.replace('$', "$$");

    // Replace an existing goal value.
    let re_existing = Regex::new(r#"(?i)(goal\s*=\s*")([^"]*)(")"#).expect("valid regex");
    if re_existing.is_match(dot) {
        return re_existing
            .replace(dot, format!(r#"${{1}}{safe_goal}${{3}}"#))
            .into_owned();
    }

    // No goal attribute — insert one into the first `graph [...]` block.
    let re_graph_bracket = Regex::new(r#"(?i)(graph\s*\[)"#).expect("valid regex");
    if re_graph_bracket.is_match(dot) {
        return re_graph_bracket
            .replace(dot, format!(r#"${{1}}goal="{safe_goal}", "#))
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
            error,
        } => {
            if !success {
                if let Some(err) = &error {
                    println!(
                        "[parallel #{index}] branch FAILED  {branch}  error={err}  ({:.2?})",
                        duration
                    );
                } else {
                    println!(
                        "[parallel #{index}] branch FAILED  {branch}  (no error detail)  ({:.2?})",
                        duration
                    );
                }
            } else {
                println!(
                    "[parallel #{index}] branch completed  {branch}  ({:.2?})",
                    duration
                );
            }
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
        .with_interviewer(Arc::new(attractor::interviewer::ConsoleInterviewer))
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

    let config = RunConfig::new(&logs_root)
        .with_working_dir(std::env::current_dir().unwrap_or_else(|_| std::env::temp_dir()));
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

    #[test]
    fn inject_goal_dollar_sign_in_goal_is_literal() {
        // Regression: `$` in a regex replacement string is interpreted as a
        // capture-group backreference.  "fix $1 issue" must appear verbatim.
        let dot = r#"digraph { graph [goal="old"] start [shape=Mdiamond] }"#;
        let patched = inject_goal(dot, "fix $1 issue");
        assert!(
            patched.contains(r#"goal="fix $1 issue""#),
            "$ in goal must be treated as literal; got: {patched}"
        );
    }

    #[test]
    fn inject_goal_dollar_sign_in_goal_insert_branch() {
        // Regression guard for the INSERT path: no existing goal= attribute,
        // so inject_goal uses the re_graph_bracket branch.  `$` in the
        // replacement must still be treated as a literal character.
        let dot = r#"digraph { graph [label="pipeline"] start [shape=Mdiamond] }"#;
        let patched = inject_goal(dot, "cost $5 per run");
        assert!(
            patched.contains(r#"goal="cost $5 per run""#),
            "$ in goal must be literal on the insert path; got: {patched}"
        );
    }

    // ── infer_provider ─────────────────────────────────────────────────────────

    #[test]
    fn infer_provider_openai_models() {
        assert_eq!(infer_provider("gpt-4o"), "openai");
        assert_eq!(infer_provider("gpt-4o-mini"), "openai");
        assert_eq!(infer_provider("o1-preview"), "openai");
        assert_eq!(infer_provider("o3-mini"), "openai");
        assert_eq!(infer_provider("codex-4o"), "openai");
    }

    #[test]
    fn infer_provider_anthropic_models() {
        assert_eq!(infer_provider("claude-opus-4-5"), "anthropic");
        assert_eq!(infer_provider("claude-haiku-4-5-20251001"), "anthropic");
        assert_eq!(infer_provider("claude-sonnet-4-20250514"), "anthropic");
        // Case-insensitive: model strings from DOT may vary.
        assert_eq!(infer_provider("Claude-3-Opus"), "anthropic");
    }

    #[test]
    fn infer_provider_gemini_models() {
        assert_eq!(infer_provider("gemini-2.5-pro"), "gemini");
        assert_eq!(infer_provider("gemini-2.0-flash"), "gemini");
        assert_eq!(infer_provider("Gemini-Pro"), "gemini");
    }

    #[test]
    fn infer_provider_unknown_defaults_to_openai() {
        assert_eq!(infer_provider("some-custom-model"), "openai");
        assert_eq!(infer_provider("deepseek-r1"), "openai");
        assert_eq!(infer_provider(""), "openai");
    }

    #[test]
    fn infer_provider_trims_whitespace() {
        // DOT attribute values may have leading/trailing whitespace.
        assert_eq!(infer_provider(" claude-opus-4-5"), "anthropic");
        assert_eq!(infer_provider("  gemini-2.5-pro  "), "gemini");
        assert_eq!(infer_provider(" gpt-4o "), "openai");
    }

    // ── ConsoleInterviewer trait-bound ─────────────────────────────────────────────

    #[test]
    fn console_interviewer_implements_interviewer_trait() {
        use attractor::interviewer::{ConsoleInterviewer, Interviewer};
        fn assert_is_interviewer<T: Interviewer>() {}
        assert_is_interviewer::<ConsoleInterviewer>();
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
                error: None,
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
