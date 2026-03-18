//! Human-in-the-loop interviewer interface.
//!
//! All human interaction in Attractor goes through the [`Interviewer`] trait.
//! This abstraction allows the pipeline to present questions through any
//! frontend — CLI, web UI, Slack, or a programmatic queue for testing.
//!
//! Built-in implementations:
//! - [`AutoApproveInterviewer`] — always selects YES / first option (CI/CD)
//! - [`QueueInterviewer`] — reads from a pre-filled queue (deterministic testing)

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Question model
// ---------------------------------------------------------------------------

/// The kind of question being asked.
///
/// Variant names match NLSpec §11.8: SINGLE_SELECT, MULTI_SELECT, FREE_TEXT, CONFIRM.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuestionType {
    /// Select a single option (binary yes/no or from a list).
    SingleSelect,
    /// Select one or more options from a list.
    MultiSelect,
    /// Free text input.
    FreeText,
    /// Yes/no confirmation prompt.
    Confirmation,
}

/// One selectable option in a [`QuestionType::MultiSelect`] question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionOption {
    /// Accelerator key (e.g. `"Y"`, `"A"`, `"1"`).
    pub key: String,
    /// Display label (e.g. `"Yes, deploy to production"`).
    pub label: String,
}

/// A question presented to a human via the [`Interviewer`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Question {
    /// The question text to display.
    pub text: String,
    /// The kind of input expected.
    pub question_type: QuestionType,
    /// Options for [`QuestionType::MultiSelect`] questions.
    pub options: Vec<QuestionOption>,
    /// Default answer if no response is received within `timeout`.
    pub default: Option<Answer>,
    /// Maximum wait time.  `None` means wait indefinitely.
    pub timeout: Option<Duration>,
    /// ID of the stage that generated this question (for display / logging).
    pub stage: String,
    /// Arbitrary metadata key-value pairs.
    pub metadata: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// Answer model
// ---------------------------------------------------------------------------

/// The selected answer value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerValue {
    /// Affirmative answer.
    Yes,
    /// Negative answer.
    No,
    /// Human explicitly skipped the question.
    Skipped,
    /// No response received within the timeout.
    Timeout,
    /// A specific option was selected (holds the option key).
    Selected(String),
}

/// Answer returned by an [`Interviewer`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Answer {
    /// The abstract answer value.
    pub value: AnswerValue,
    /// The full selected option (populated for [`QuestionType::MultiSelect`]).
    pub selected_option: Option<QuestionOption>,
    /// Free-text representation of the answer.
    pub text: String,
}

impl Answer {
    /// Affirmative answer.
    pub fn yes() -> Self {
        Answer {
            value: AnswerValue::Yes,
            selected_option: None,
            text: "yes".to_string(),
        }
    }

    /// Negative answer.
    pub fn no() -> Self {
        Answer {
            value: AnswerValue::No,
            selected_option: None,
            text: "no".to_string(),
        }
    }

    /// Skipped — human chose not to answer.
    pub fn skipped() -> Self {
        Answer {
            value: AnswerValue::Skipped,
            selected_option: None,
            text: String::new(),
        }
    }

    /// Timeout — no response within the allowed time.
    pub fn timeout() -> Self {
        Answer {
            value: AnswerValue::Timeout,
            selected_option: None,
            text: String::new(),
        }
    }

    /// A specific option was selected.
    pub fn selected(option: QuestionOption) -> Self {
        Answer {
            value: AnswerValue::Selected(option.key.clone()),
            text: option.label.clone(),
            selected_option: Some(option),
        }
    }
}

// ---------------------------------------------------------------------------
// Interviewer trait
// ---------------------------------------------------------------------------

/// Human-in-the-loop interface.
///
/// Implementations handle the presentation layer — CLI prompts, web UI,
/// Slack messages, or programmatic queues for testing.
#[async_trait]
pub trait Interviewer: Send + Sync {
    /// Present a question and wait for an answer.
    async fn ask(&self, question: Question) -> Answer;

    /// Present multiple questions in sequence and collect answers.
    async fn ask_multiple(&self, questions: Vec<Question>) -> Vec<Answer> {
        let mut answers = Vec::with_capacity(questions.len());
        for q in questions {
            answers.push(self.ask(q).await);
        }
        answers
    }

    /// Inform the human of a message without requiring an answer.
    async fn inform(&self, _message: &str, _stage: &str) {}
}

// ---------------------------------------------------------------------------
// AutoApproveInterviewer
// ---------------------------------------------------------------------------

/// Always selects YES / the first option.
///
/// Used for CI/CD pipelines and automated tests where no human is present.
pub struct AutoApproveInterviewer;

#[async_trait]
impl Interviewer for AutoApproveInterviewer {
    async fn ask(&self, question: Question) -> Answer {
        match question.question_type {
            // V2-ATR-006: For SingleSelect with options, return the first option
            // (not a generic "yes"). This matches what an automated approver
            // should do — pick the first (typically the affirmative) option.
            QuestionType::SingleSelect => {
                if let Some(first) = question.options.into_iter().next() {
                    Answer::selected(first)
                } else {
                    // No options → binary yes/no prompt.
                    Answer::yes()
                }
            }
            QuestionType::Confirmation => Answer::yes(),
            QuestionType::MultiSelect => {
                if let Some(first) = question.options.into_iter().next() {
                    Answer::selected(first)
                } else {
                    Answer {
                        value: AnswerValue::Selected("auto-approved".to_string()),
                        selected_option: None,
                        text: "auto-approved".to_string(),
                    }
                }
            }
            QuestionType::FreeText => Answer {
                value: AnswerValue::Selected("auto-approved".to_string()),
                selected_option: None,
                text: "auto-approved".to_string(),
            },
        }
    }

    async fn inform(&self, _message: &str, _stage: &str) {}
}

// ---------------------------------------------------------------------------
// QueueInterviewer
// ---------------------------------------------------------------------------

/// Reads answers from a pre-filled queue.
///
/// Returns [`Answer::skipped`] when the queue is empty.  Used for
/// deterministic testing and scenario replay.
pub struct QueueInterviewer {
    answers: Mutex<VecDeque<Answer>>,
}

impl QueueInterviewer {
    /// Create a new queue interviewer with the provided answers.
    pub fn new(answers: impl IntoIterator<Item = Answer>) -> Self {
        QueueInterviewer {
            answers: Mutex::new(answers.into_iter().collect()),
        }
    }

    /// Push an additional answer to the back of the queue.
    pub fn push(&self, answer: Answer) {
        self.answers
            .lock()
            .expect("queue interviewer lock poisoned")
            .push_back(answer);
    }

    /// Return the number of answers remaining in the queue.
    pub fn remaining(&self) -> usize {
        self.answers
            .lock()
            .expect("queue interviewer lock poisoned")
            .len()
    }
}

#[async_trait]
impl Interviewer for QueueInterviewer {
    async fn ask(&self, _question: Question) -> Answer {
        self.answers
            .lock()
            .expect("queue interviewer lock poisoned")
            .pop_front()
            .unwrap_or_else(Answer::skipped)
    }

    async fn inform(&self, _message: &str, _stage: &str) {}
}

// ---------------------------------------------------------------------------
// CallbackInterviewer
// ---------------------------------------------------------------------------

/// Delegates question answering to an async callback function.
///
/// Useful for integrating with external systems (Slack, web UI, API) where
/// the answering logic is provided at construction time.
pub struct CallbackInterviewer<F>
where
    F: Fn(Question) -> std::pin::Pin<Box<dyn std::future::Future<Output = Answer> + Send>>
        + Send
        + Sync,
{
    callback: F,
}

impl<F> CallbackInterviewer<F>
where
    F: Fn(Question) -> std::pin::Pin<Box<dyn std::future::Future<Output = Answer> + Send>>
        + Send
        + Sync,
{
    /// Create a new callback interviewer with the provided async function.
    pub fn new(callback: F) -> Self {
        CallbackInterviewer { callback }
    }
}

#[async_trait]
impl<F> Interviewer for CallbackInterviewer<F>
where
    F: Fn(Question) -> std::pin::Pin<Box<dyn std::future::Future<Output = Answer> + Send>>
        + Send
        + Sync,
{
    async fn ask(&self, question: Question) -> Answer {
        (self.callback)(question).await
    }
}

// ---------------------------------------------------------------------------
// RecordingInterviewer
// ---------------------------------------------------------------------------

/// A recorded question-answer pair.
#[derive(Debug, Clone)]
pub struct Recording {
    pub question: Question,
    pub answer: Answer,
}

/// Wraps another [`Interviewer`] and records all question-answer interactions.
///
/// Useful for replay, debugging, and audit trails.
pub struct RecordingInterviewer<I: Interviewer> {
    inner: I,
    recordings: Mutex<Vec<Recording>>,
}

impl<I: Interviewer> RecordingInterviewer<I> {
    /// Create a new recording interviewer wrapping `inner`.
    pub fn new(inner: I) -> Self {
        RecordingInterviewer {
            inner,
            recordings: Mutex::new(Vec::new()),
        }
    }

    /// Return a snapshot of all recorded interactions.
    pub fn recordings(&self) -> Vec<Recording> {
        self.recordings
            .lock()
            .expect("recording interviewer lock poisoned")
            .clone()
    }

    /// Return the number of recorded interactions.
    pub fn recording_count(&self) -> usize {
        self.recordings
            .lock()
            .expect("recording interviewer lock poisoned")
            .len()
    }
}

#[async_trait]
impl<I: Interviewer + Send + Sync> Interviewer for RecordingInterviewer<I> {
    async fn ask(&self, question: Question) -> Answer {
        let answer = self.inner.ask(question.clone()).await;
        self.recordings
            .lock()
            .expect("recording interviewer lock poisoned")
            .push(Recording {
                question,
                answer: answer.clone(),
            });
        answer
    }

    async fn inform(&self, message: &str, stage: &str) {
        self.inner.inform(message, stage).await;
    }
}

// ---------------------------------------------------------------------------
// Context display helper
// ---------------------------------------------------------------------------

/// Format the LLM output block to display before a human-gate prompt.
///
/// Returns `Some(block)` when the question's `metadata` contains a non-empty
/// `"last_response"` key; `None` otherwise.  The caller is responsible for
/// printing the returned string before showing the question text.
pub fn format_llm_context_block(question: &Question) -> Option<String> {
    match question.metadata.get("last_response") {
        Some(llm_output) if !llm_output.is_empty() => Some(format!(
            "--- LLM Output ---\n{}\n--- End Output ---\n",
            llm_output
        )),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// ConsoleInterviewer
// ---------------------------------------------------------------------------

/// Reads questions from and writes answers to the terminal (stdin/stdout).
///
/// Suitable for interactive CLI use.  For CI/CD environments prefer
/// [`AutoApproveInterviewer`]; for deterministic testing prefer
/// [`QueueInterviewer`].
///
/// ## Behaviour
///
/// | Question type | Output | Input parsing |
/// |---|---|---|
/// | `SingleSelect` / `MultiSelect` | Prints numbered list | Parses `1` … `N`, then key match, then label match |
/// | `FreeText` | Prints question text | Returns raw trimmed line |
/// | `Confirmation` | Prints `[y/n]` prompt | `y`/`yes` → Yes, anything else → No |
pub struct ConsoleInterviewer;

#[async_trait]
impl Interviewer for ConsoleInterviewer {
    async fn ask(&self, question: Question) -> Answer {
        use tokio::io::AsyncBufReadExt as _;

        // Show LLM context before prompting (if available).
        if let Some(block) = format_llm_context_block(&question) {
            println!("{}", block);
        }

        // Print the question text.
        println!("{}", question.text);

        // Print options for select questions.
        match question.question_type {
            QuestionType::SingleSelect | QuestionType::MultiSelect => {
                for (i, opt) in question.options.iter().enumerate() {
                    println!("  {}. {}", i + 1, opt.label);
                }
            }
            QuestionType::Confirmation => {
                print!("[y/n]: ");
            }
            QuestionType::FreeText => {}
        }

        // Read a line from stdin.
        let stdin = tokio::io::stdin();
        let mut reader = tokio::io::BufReader::new(stdin);
        let mut line = String::new();
        let _ = reader.read_line(&mut line).await;
        let input = line.trim().to_string();

        // Parse response based on question type.
        match question.question_type {
            QuestionType::SingleSelect | QuestionType::MultiSelect => {
                // Try numeric index first (1-based).
                if let Ok(n) = input.parse::<usize>() {
                    if n >= 1 && n <= question.options.len() {
                        return Answer::selected(question.options[n - 1].clone());
                    }
                }
                // Try key match (case-insensitive).
                if let Some(opt) = question
                    .options
                    .iter()
                    .find(|o| o.key.eq_ignore_ascii_case(&input))
                {
                    return Answer::selected(opt.clone());
                }
                // Try label match (case-insensitive).
                if let Some(opt) = question
                    .options
                    .iter()
                    .find(|o| o.label.eq_ignore_ascii_case(&input))
                {
                    return Answer::selected(opt.clone());
                }
                // Fall back to first option, or yes if no options.
                if let Some(first) = question.options.first() {
                    Answer::selected(first.clone())
                } else {
                    Answer::yes()
                }
            }
            QuestionType::FreeText => Answer {
                value: AnswerValue::Selected(input.clone()),
                selected_option: None,
                text: input,
            },
            QuestionType::Confirmation => {
                if input.to_ascii_lowercase().starts_with('y') {
                    Answer::yes()
                } else {
                    Answer::no()
                }
            }
        }
    }

    async fn inform(&self, message: &str, _stage: &str) {
        println!("{message}");
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mc_question(options: Vec<(&str, &str)>) -> Question {
        Question {
            text: "Choose".to_string(),
            question_type: QuestionType::MultiSelect,
            options: options
                .into_iter()
                .map(|(k, l)| QuestionOption {
                    key: k.to_string(),
                    label: l.to_string(),
                })
                .collect(),
            default: None,
            timeout: None,
            stage: "test".to_string(),
            metadata: HashMap::new(),
        }
    }

    fn make_yesno_question() -> Question {
        Question {
            text: "Proceed?".to_string(),
            question_type: QuestionType::SingleSelect,
            options: vec![],
            default: None,
            timeout: None,
            stage: "test".to_string(),
            metadata: HashMap::new(),
        }
    }

    // -- Answer constructors --

    #[test]
    fn answer_yes() {
        let a = Answer::yes();
        assert_eq!(a.value, AnswerValue::Yes);
        assert_eq!(a.text, "yes");
    }

    #[test]
    fn answer_no() {
        let a = Answer::no();
        assert_eq!(a.value, AnswerValue::No);
    }

    #[test]
    fn answer_skipped() {
        let a = Answer::skipped();
        assert_eq!(a.value, AnswerValue::Skipped);
        assert!(a.text.is_empty());
    }

    #[test]
    fn answer_timeout() {
        let a = Answer::timeout();
        assert_eq!(a.value, AnswerValue::Timeout);
    }

    #[test]
    fn answer_selected() {
        let opt = QuestionOption {
            key: "Y".to_string(),
            label: "Yes, deploy".to_string(),
        };
        let a = Answer::selected(opt.clone());
        assert_eq!(a.value, AnswerValue::Selected("Y".to_string()));
        assert_eq!(a.text, "Yes, deploy");
        assert!(a.selected_option.is_some());
    }

    #[test]
    fn answer_serde_roundtrip() {
        let a = Answer::yes();
        let json = serde_json::to_string(&a).unwrap();
        let back: Answer = serde_json::from_str(&json).unwrap();
        assert_eq!(back.value, AnswerValue::Yes);
    }

    // -- AutoApproveInterviewer --

    #[tokio::test]
    async fn auto_approve_yesno() {
        let iv = AutoApproveInterviewer;
        let a = iv.ask(make_yesno_question()).await;
        assert_eq!(a.value, AnswerValue::Yes);
    }

    // V2-ATR-006: AutoApproveInterviewer returns first option text for
    // SingleSelect with options, NOT generic "yes".
    #[tokio::test]
    async fn auto_approve_single_select_with_options_returns_first_option() {
        let iv = AutoApproveInterviewer;
        let q = Question {
            text: "Choose action".to_string(),
            question_type: QuestionType::SingleSelect,
            options: vec![
                QuestionOption {
                    key: "A".to_string(),
                    label: "Approve".to_string(),
                },
                QuestionOption {
                    key: "R".to_string(),
                    label: "Reject".to_string(),
                },
            ],
            default: None,
            timeout: None,
            stage: "test".to_string(),
            metadata: HashMap::new(),
        };
        let a = iv.ask(q).await;
        // Must return the FIRST option (Approve), NOT a generic "yes".
        assert_eq!(
            a.value,
            AnswerValue::Selected("A".to_string()),
            "AutoApproveInterviewer must return first option key for SingleSelect with options"
        );
        assert_eq!(a.text, "Approve");
    }

    // SingleSelect with NO options still returns yes (backward-compatible).
    #[tokio::test]
    async fn auto_approve_single_select_no_options_returns_yes() {
        let iv = AutoApproveInterviewer;
        let a = iv.ask(make_yesno_question()).await;
        assert_eq!(a.value, AnswerValue::Yes);
    }

    #[tokio::test]
    async fn auto_approve_confirmation() {
        let iv = AutoApproveInterviewer;
        let q = Question {
            question_type: QuestionType::Confirmation,
            ..make_yesno_question()
        };
        let a = iv.ask(q).await;
        assert_eq!(a.value, AnswerValue::Yes);
    }

    #[tokio::test]
    async fn auto_approve_multiple_choice_returns_first() {
        let iv = AutoApproveInterviewer;
        let q = make_mc_question(vec![("A", "Approve"), ("R", "Reject")]);
        let a = iv.ask(q).await;
        assert_eq!(a.value, AnswerValue::Selected("A".to_string()));
        assert_eq!(a.text, "Approve");
    }

    #[tokio::test]
    async fn auto_approve_empty_mc() {
        let iv = AutoApproveInterviewer;
        let q = make_mc_question(vec![]);
        let a = iv.ask(q).await;
        assert_eq!(a.value, AnswerValue::Selected("auto-approved".to_string()));
    }

    #[tokio::test]
    async fn auto_approve_freeform() {
        let iv = AutoApproveInterviewer;
        let q = Question {
            question_type: QuestionType::FreeText,
            ..make_yesno_question()
        };
        let a = iv.ask(q).await;
        assert_eq!(a.value, AnswerValue::Selected("auto-approved".to_string()));
    }

    // -- GAP-ATR-013: QuestionType enum variants match NLSpec §11.8 ---

    #[test]
    fn question_type_variant_names_match_nlspec() {
        // GAP-ATR-013: NLSpec §11.8 specifies SINGLE_SELECT, MULTI_SELECT,
        // FREE_TEXT, CONFIRM.  Verify the enum has the correctly-named variants.
        let single = QuestionType::SingleSelect;
        let multi = QuestionType::MultiSelect;
        let free = QuestionType::FreeText;
        let confirm = QuestionType::Confirmation;

        // Serde roundtrip using snake_case serialisation
        let s = serde_json::to_string(&single).unwrap();
        assert_eq!(s, r#""single_select""#);

        let s = serde_json::to_string(&multi).unwrap();
        assert_eq!(s, r#""multi_select""#);

        let s = serde_json::to_string(&free).unwrap();
        assert_eq!(s, r#""free_text""#);

        let s = serde_json::to_string(&confirm).unwrap();
        assert_eq!(s, r#""confirmation""#);

        // All four variants must be distinct
        assert_ne!(QuestionType::SingleSelect, QuestionType::MultiSelect);
        assert_ne!(QuestionType::SingleSelect, QuestionType::FreeText);
        assert_ne!(QuestionType::SingleSelect, QuestionType::Confirmation);
    }

    #[tokio::test]
    async fn auto_approve_ask_multiple() {
        let iv = AutoApproveInterviewer;
        let questions = vec![make_yesno_question(), make_yesno_question()];
        let answers = iv.ask_multiple(questions).await;
        assert_eq!(answers.len(), 2);
        assert!(answers.iter().all(|a| a.value == AnswerValue::Yes));
    }

    // -- QueueInterviewer --

    #[tokio::test]
    async fn queue_pops_front() {
        let iv = QueueInterviewer::new(vec![Answer::yes(), Answer::no()]);
        assert_eq!(iv.remaining(), 2);
        let a1 = iv.ask(make_yesno_question()).await;
        assert_eq!(a1.value, AnswerValue::Yes);
        assert_eq!(iv.remaining(), 1);
        let a2 = iv.ask(make_yesno_question()).await;
        assert_eq!(a2.value, AnswerValue::No);
        assert_eq!(iv.remaining(), 0);
    }

    #[tokio::test]
    async fn queue_empty_returns_skipped() {
        let iv = QueueInterviewer::new(vec![]);
        let a = iv.ask(make_yesno_question()).await;
        assert_eq!(a.value, AnswerValue::Skipped);
    }

    #[tokio::test]
    async fn queue_push_appends() {
        let iv = QueueInterviewer::new(vec![Answer::yes()]);
        iv.push(Answer::no());
        assert_eq!(iv.remaining(), 2);
        iv.ask(make_yesno_question()).await; // yes
        let a = iv.ask(make_yesno_question()).await; // no
        assert_eq!(a.value, AnswerValue::No);
    }

    // -- ConsoleInterviewer context display --

    #[test]
    fn console_interviewer_displays_llm_context_before_prompt() {
        // RED: format_llm_context_block does not exist yet.
        let mut q = make_yesno_question();
        q.metadata.insert(
            "last_response".to_string(),
            "This is what the LLM produced.".to_string(),
        );

        let block = format_llm_context_block(&q);
        assert!(
            block.is_some(),
            "must produce context block when last_response is in metadata"
        );
        let block_str = block.unwrap();
        assert!(block_str.contains("--- LLM Output ---"), "must include header");
        assert!(
            block_str.contains("This is what the LLM produced."),
            "must include LLM content"
        );
        assert!(block_str.contains("--- End Output ---"), "must include footer");

        // Order: header → content → footer
        let header_pos = block_str.find("--- LLM Output ---").unwrap();
        let content_pos = block_str.find("This is what the LLM produced.").unwrap();
        let footer_pos = block_str.find("--- End Output ---").unwrap();
        assert!(header_pos < content_pos, "header must appear before content");
        assert!(content_pos < footer_pos, "content must appear before footer");
    }

    #[test]
    fn console_interviewer_no_context_block_when_metadata_empty() {
        // RED: format_llm_context_block does not exist yet.
        let q = make_yesno_question();
        let block = format_llm_context_block(&q);
        assert!(
            block.is_none(),
            "must not produce context block when no last_response in metadata"
        );
    }

    #[test]
    fn console_interviewer_no_context_block_when_last_response_empty() {
        // RED: format_llm_context_block does not exist yet.
        let mut q = make_yesno_question();
        q.metadata
            .insert("last_response".to_string(), String::new());
        let block = format_llm_context_block(&q);
        assert!(
            block.is_none(),
            "must not produce context block when last_response is empty string"
        );
    }

    #[tokio::test]
    async fn queue_ask_multiple_count() {
        let iv = QueueInterviewer::new(vec![Answer::yes(), Answer::yes(), Answer::yes()]);
        let answers = iv
            .ask_multiple(vec![
                make_yesno_question(),
                make_yesno_question(),
                make_yesno_question(),
            ])
            .await;
        assert_eq!(answers.len(), 3);
    }
}
