//! End-to-end pipeline tests.
//!
//! Step 1b: Run consensus_task.dot through PipelineRunner with MockCodergenBackend.
//! Step 1c: LIVE_TEST=1 — 3-node pipeline using real unified_llm::Client.

use std::fs;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn make_runner_with_mock(
    mock: Arc<attractor::testing::MockCodergenBackend>,
) -> (
    attractor::PipelineRunner,
    tokio::sync::broadcast::Receiver<attractor::PipelineEvent>,
) {
    struct MockProxy(Arc<attractor::testing::MockCodergenBackend>);

    #[async_trait::async_trait]
    impl attractor::CodergenBackend for MockProxy {
        async fn run(
            &self,
            node: &attractor::graph::Node,
            prompt: &str,
            ctx: &attractor::state::context::Context,
        ) -> Result<attractor::handler::CodergenResult, attractor::EngineError> {
            self.0.run(node, prompt, ctx).await
        }
    }

    let codergen = Arc::new(attractor::handler::CodergenHandler::new(Some(Box::new(
        MockProxy(mock),
    ))));
    attractor::PipelineRunner::builder()
        .with_handler("codergen", codergen)
        .build()
}

// ---------------------------------------------------------------------------
// Step 1b: consensus_task.dot — parse, validate, attempt run
// ---------------------------------------------------------------------------

#[tokio::test]
async fn e2e_consensus_task_mock() {
    let dir = tempfile::tempdir().unwrap();
    let content = fs::read_to_string("../consensus_task.dot")
        .expect("consensus_task.dot not found at project root");

    // --- Parse ---
    println!("=== Step 1b: consensus_task.dot ===");
    let graph = match attractor::parser::parse_dot(&content) {
        Ok(g) => {
            println!("PARSE OK: {} nodes, {} edges", g.nodes.len(), g.edges.len());
            println!("  graph.goal = {:?}", g.graph_attrs.goal);
            g
        }
        Err(e) => {
            println!("PARSE ERROR: {e}");
            panic!("Parse failed: {e}");
        }
    };

    // --- Validate ---
    let diags = attractor::validation::validate(&graph, &[]);
    println!("VALIDATION DIAGNOSTICS ({}):", diags.len());
    for d in &diags {
        println!("  [{:?}] {:?} — {}", d.severity, d.node_id, d.message);
    }

    // --- Attempt PipelineRunner.run ---
    let mock = Arc::new(attractor::testing::MockCodergenBackend::new());
    let (runner, _rx) = make_runner_with_mock(mock);
    let config = attractor::RunConfig::new(dir.path());

    println!("ATTEMPTING PIPELINE RUN (consensus_task.dot)...");
    match runner.run(&content, config).await {
        Ok(result) => {
            println!(
                "RUN RESULT: status={:?}, completed_nodes={}",
                result.status,
                result.completed_nodes.len()
            );
            println!("  completed_nodes: {:?}", result.completed_nodes);
        }
        Err(e) => {
            println!("RUN ERROR (expected if validation fails): {e:?}");
        }
    }
}

#[tokio::test]
async fn e2e_semport_mock() {
    let dir = tempfile::tempdir().unwrap();
    let content =
        fs::read_to_string("../semport.dot").expect("semport.dot not found at project root");

    println!("=== Step 1b: semport.dot ===");
    let graph = match attractor::parser::parse_dot(&content) {
        Ok(g) => {
            println!("PARSE OK: {} nodes, {} edges", g.nodes.len(), g.edges.len());
            g
        }
        Err(e) => {
            println!("PARSE ERROR: {e}");
            panic!("Parse failed: {e}");
        }
    };

    let diags = attractor::validation::validate(&graph, &[]);
    println!("VALIDATION DIAGNOSTICS ({}):", diags.len());
    for d in &diags {
        println!("  [{:?}] {:?} — {}", d.severity, d.node_id, d.message);
    }

    let mock = Arc::new(attractor::testing::MockCodergenBackend::new());
    let (runner, _rx) = make_runner_with_mock(mock);
    let config = attractor::RunConfig::new(dir.path());

    println!("ATTEMPTING PIPELINE RUN (semport.dot)...");
    match runner.run(&content, config).await {
        Ok(result) => {
            println!(
                "RUN RESULT: status={:?}, completed_nodes={}",
                result.status,
                result.completed_nodes.len()
            );
            println!("  completed_nodes: {:?}", result.completed_nodes);
        }
        Err(e) => {
            println!("RUN ERROR (expected if validation fails): {e:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// Step 1c: LIVE_TEST=1 — real 3-node pipeline with unified_llm::Client
// ---------------------------------------------------------------------------

// V2-ATR-007: Updated to match §11.13 — 5-node pipeline with goal_gate=true
// and conditional branching.  Kept gated behind LIVE_TEST=1.
const SPEC_PIPELINE_DOT: &str = r#"digraph Test {
    graph [goal="Say hello world in exactly three words"]
    start     [shape=Mdiamond]
    plan      [shape=box, prompt="Plan how to say hello world in exactly three words."]
    implement [shape=box, prompt="Say hello world. Just the three-word phrase, nothing else.", goal_gate=true]
    review    [shape=box, prompt="Review: does the output say exactly 'hello world' or 'Hello world'? Answer YES or NO."]
    done      [shape=Msquare]
    start -> plan
    plan  -> implement
    implement -> review  [label="goal_met"]
    implement -> plan    [label="revise"]
    review -> done
}"#;

#[cfg(test)]
mod live_tests {
    use super::*;
    use attractor::handler::CodergenResult;
    use attractor::state::context::{Context, Outcome};
    use attractor::{CodergenBackend, EngineError};
    use unified_llm::{Client, Message, Request};

    /// A thin CodergenBackend wrapper around unified_llm::Client.
    struct UnifiedLlmBackend {
        client: Client,
        model: String,
    }

    impl UnifiedLlmBackend {
        fn new(client: Client, model: impl Into<String>) -> Self {
            UnifiedLlmBackend {
                client,
                model: model.into(),
            }
        }
    }

    #[async_trait::async_trait]
    impl CodergenBackend for UnifiedLlmBackend {
        async fn run(
            &self,
            node: &attractor::graph::Node,
            prompt: &str,
            _ctx: &Context,
        ) -> Result<CodergenResult, EngineError> {
            println!(
                "  [UnifiedLlmBackend] calling model={} for node={}",
                self.model, node.id
            );
            let messages = vec![Message::user(prompt)];
            let request = Request::new(self.model.clone(), messages);
            match self.client.complete(request).await {
                Ok(response) => {
                    let text = response.text();
                    println!(
                        "  [UnifiedLlmBackend] response ({} chars): {}",
                        text.len(),
                        text.trim()
                    );
                    let outcome = Outcome {
                        notes: format!("LLM response for {}: {}", node.id, text.trim()),
                        ..Outcome::success()
                    };
                    Ok(CodergenResult::Outcome(outcome))
                }
                Err(e) => {
                    println!("  [UnifiedLlmBackend] LLM error: {e}");
                    Err(EngineError::Backend(format!("unified_llm error: {e}")))
                }
            }
        }
    }

    #[tokio::test]
    async fn live_5_node_pipeline_with_goal_gate() {
        if std::env::var("LIVE_TEST").unwrap_or_default() != "1" {
            println!("SKIPPED — set LIVE_TEST=1 to run");
            return;
        }

        // V2-ATR-007: Updated to match §11.13 — 5-node pipeline with goal_gate.
        println!("=== LIVE_TEST 5-node pipeline (§11.13) ===");

        // Build client from environment.
        let client = match Client::from_env() {
            Ok(c) => {
                println!(
                    "unified_llm::Client OK — default provider: {}",
                    c.default_provider()
                );
                c
            }
            Err(e) => {
                println!("LIVE_TEST SKIP: could not build Client from env: {e}");
                println!("  Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY");
                return;
            }
        };

        // Pick a cheap, fast model for the test.
        let model = match client.default_provider() {
            "anthropic" => "claude-haiku-3-5",
            "openai" => "gpt-4o-mini",
            "gemini" => "gemini-2.0-flash",
            other => {
                println!("Unknown default provider '{other}', trying gpt-4o-mini");
                "gpt-4o-mini"
            }
        };
        println!("Using model: {model}");

        let dir = tempfile::tempdir().unwrap();
        let backend = UnifiedLlmBackend::new(client, model);
        let codergen = Arc::new(attractor::handler::CodergenHandler::new(Some(Box::new(
            backend,
        ))));
        let (runner, _rx) = attractor::PipelineRunner::builder()
            .with_handler("codergen", codergen)
            .build();

        println!("DOT pipeline:\n{}", SPEC_PIPELINE_DOT);
        println!("Attempting run...");

        match runner
            .run(SPEC_PIPELINE_DOT, attractor::RunConfig::new(dir.path()))
            .await
        {
            Ok(result) => {
                println!("RUN RESULT: status={:?}", result.status);
                println!("  completed_nodes: {:?}", result.completed_nodes);
                // Print artifact files if written.
                let task_dir = dir.path().join("task");
                if task_dir.exists() {
                    for fname in ["prompt.md", "response.md", "status.json"] {
                        let p = task_dir.join(fname);
                        if p.exists() {
                            let text = fs::read_to_string(&p).unwrap_or_default();
                            println!("  {fname}: {}", text.trim());
                        }
                    }
                }
                assert_eq!(
                    result.status,
                    attractor::state::context::StageStatus::Success,
                    "Expected 5-node pipeline to succeed"
                );
                // Verify all 5 key nodes completed.
                for node in &["plan", "implement", "review"] {
                    assert!(
                        result.completed_nodes.contains(&node.to_string()),
                        "Expected node '{}' to be completed; completed: {:?}",
                        node,
                        result.completed_nodes
                    );
                }
            }
            Err(e) => {
                println!("RUN ERROR: {e:?}");
                panic!("Live pipeline run failed: {e}");
            }
        }
    }
}
