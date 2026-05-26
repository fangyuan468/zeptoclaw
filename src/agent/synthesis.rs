//! Final synthesis turn — phase 2 of the agent loop state-machine refactor.
//!
//! When the tool loop finishes but the model's content is empty / contains
//! only provider tool markup, or when the loop is cut off at
//! `max_tool_iterations` with tool calls still pending, the agent runs a
//! single **tools-disabled** synthesis turn so the user receives a
//! coherent answer (or an explicit failure) rather than a blank message.
//!
//! See `docs/plans/doing/agent-loop-state-machine-refactor.md` §5, §7, §9.

use std::sync::Arc;

use crate::error::Result;
use crate::providers::{ChatOptions, LLMProvider, LLMResponse};
use crate::session::Message;

/// Instruction appended to the messages list right before the synthesis
/// call. Kept generic so the agent's normal language / persona context
/// (provided by the system prompt and prior messages) determines the
/// output language and tone; this instruction only constrains behaviour.
///
/// Mirrors the prompt drafted in plan §7.
pub const SYNTHESIS_INSTRUCTION: &str = "You are producing the final user-facing answer. \
Do not call tools. Use only the existing tool results and conversation context. \
If evidence is incomplete or a source could not be fetched, say so explicitly. \
Return a concise answer that directly satisfies the user's request.";

/// Run a single tools-disabled synthesis turn.
///
/// Arguments:
/// * `provider` — the LLM provider to query. Shared via `Arc` so the
///   streaming forwarder task can invoke synthesis without holding `&self`.
/// * `messages` — base conversation context, normally the same messages
///   that fed the final tool-loop call. Caller passes ownership; this fn
///   appends the synthesis instruction internally.
/// * `model` — optional model override, mirroring the contract of
///   `LLMProvider::chat`.
/// * `options` — chat options reused from the originating call.
///
/// Returns the raw `LLMResponse`. The caller is responsible for
/// classifying the result (typically via
/// [`crate::agent::turn::classify_final_content`]) and persisting / not
/// persisting the assistant message accordingly. Synthesis must run at
/// most once per agent turn (the call sites enforce this; this fn is
/// purely a one-shot helper).
///
/// The tools catalog passed to the provider is **always empty** so the
/// model cannot re-enter the tool loop.
pub async fn run_final_synthesis(
    provider: Arc<dyn LLMProvider>,
    mut messages: Vec<Message>,
    model: Option<String>,
    options: ChatOptions,
) -> Result<LLMResponse> {
    messages.push(Message::system(SYNTHESIS_INSTRUCTION));
    provider
        .chat(messages, Vec::new(), model.as_deref(), options)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::{LLMResponse, ToolDefinition};
    use async_trait::async_trait;
    use std::sync::Mutex;

    /// Minimal LLMProvider mock that records the most recent chat call so
    /// we can assert the synthesis contract (empty tools, appended
    /// instruction, model / options passthrough) without depending on a
    /// real upstream service.
    #[derive(Debug, Default)]
    struct RecordingProvider {
        last_messages: Mutex<Vec<Message>>,
        last_tools_len: Mutex<usize>,
        last_model: Mutex<Option<String>>,
        last_max_tokens: Mutex<Option<u32>>,
        response_text: &'static str,
    }

    #[async_trait]
    impl LLMProvider for RecordingProvider {
        fn name(&self) -> &str {
            "recording"
        }

        fn default_model(&self) -> &str {
            "recording-model"
        }

        async fn chat(
            &self,
            messages: Vec<Message>,
            tools: Vec<ToolDefinition>,
            model: Option<&str>,
            options: ChatOptions,
        ) -> Result<LLMResponse> {
            *self.last_messages.lock().unwrap() = messages;
            *self.last_tools_len.lock().unwrap() = tools.len();
            *self.last_model.lock().unwrap() = model.map(str::to_string);
            *self.last_max_tokens.lock().unwrap() = options.max_tokens;
            Ok(LLMResponse::text(self.response_text))
        }
    }

    fn make_provider(response: &'static str) -> Arc<RecordingProvider> {
        Arc::new(RecordingProvider {
            response_text: response,
            ..Default::default()
        })
    }

    #[tokio::test]
    async fn synthesis_forwards_response_content() {
        let provider = make_provider("the final answer");
        let response = run_final_synthesis(
            provider.clone(),
            vec![Message::user("hello")],
            None,
            ChatOptions::new(),
        )
        .await
        .expect("synthesis chat should succeed");
        assert_eq!(response.content, "the final answer");
        assert!(response.tool_calls.is_empty());
    }

    #[tokio::test]
    async fn synthesis_passes_empty_tools_catalog() {
        let provider = make_provider("ok");
        let _ = run_final_synthesis(
            provider.clone(),
            vec![Message::user("anything")],
            None,
            ChatOptions::new(),
        )
        .await
        .unwrap();
        let recorded = *provider.last_tools_len.lock().unwrap();
        assert_eq!(recorded, 0, "synthesis must disable tools");
    }

    #[tokio::test]
    async fn synthesis_appends_instruction_to_messages() {
        let provider = make_provider("ok");
        let _ = run_final_synthesis(
            provider.clone(),
            vec![
                Message::user("first user message"),
                Message::assistant("assistant turn"),
            ],
            None,
            ChatOptions::new(),
        )
        .await
        .unwrap();
        let recorded = provider.last_messages.lock().unwrap().clone();
        assert_eq!(recorded.len(), 3, "instruction should be appended");
        let last = recorded.last().expect("last message present");
        assert!(
            last.content.starts_with("You are producing the final user-facing answer."),
            "last message should be the synthesis instruction, got: {}",
            last.content
        );
    }

    #[tokio::test]
    async fn synthesis_passes_model_and_options() {
        let provider = make_provider("ok");
        let _ = run_final_synthesis(
            provider.clone(),
            vec![Message::user("q")],
            Some("custom-model".to_string()),
            ChatOptions::new().with_max_tokens(1234),
        )
        .await
        .unwrap();
        assert_eq!(
            *provider.last_model.lock().unwrap(),
            Some("custom-model".to_string())
        );
        assert_eq!(*provider.last_max_tokens.lock().unwrap(), Some(1234));
    }
}
