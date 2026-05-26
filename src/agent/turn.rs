//! Agent turn outcome classification.
//!
//! Phase 1 of `docs/plans/doing/agent-loop-state-machine-refactor.md`:
//! normalize a raw `LLMResponse` (or a final streamed content string) into a
//! small set of explicit outcomes so the agent loop no longer treats
//! `!has_tool_calls()` as a synonym for "completed successfully".
//!
//! This module intentionally contains pure logic only. It does not depend on
//! session state, the message bus, or the loop runtime. Phases 2-4 will build
//! on top of these primitives (`FinalSynthesis`, `ToolObservation`, etc.).

use crate::providers::{LLMResponse, LLMToolCall};

/// Classified outcome of a single LLM turn.
///
/// Phase 1 covers four cases; later phases will extend with
/// `NeedClarification` and `ProviderError`.
///
/// `PartialEq`/`Eq` are intentionally not derived: `LLMToolCall` from
/// `providers::types` does not implement them, and forcing those derives
/// purely to support test assertions would be an out-of-scope public-API
/// change. Tests use `matches!` and destructuring instead.
#[derive(Debug, Clone)]
pub enum TurnOutcome {
    /// Model requested one or more tool calls. Tool loop should execute them.
    ToolCalls(Vec<LLMToolCall>),
    /// Model produced a non-empty, user-visible final answer.
    FinalAnswer(String),
    /// Model produced no tool calls and no meaningful text (blank / whitespace).
    EmptyAnswer,
    /// Model produced no structured tool calls but only provider-private tool
    /// markup (e.g. `<minimax:tool_call>`) leaked into `content`.
    ProviderMarkupOnly,
}

/// Classify a complete `LLMResponse` into a `TurnOutcome`.
///
/// Structured tool calls take precedence. When no tool calls are present, the
/// `content` field is examined for blank / provider-markup-only / final-answer
/// shapes via [`classify_final_content`].
pub fn classify_turn_outcome(response: &LLMResponse) -> TurnOutcome {
    if !response.tool_calls.is_empty() {
        return TurnOutcome::ToolCalls(response.tool_calls.clone());
    }
    classify_final_content(&response.content)
}

/// Classify a final content string (no tool-calls context) into a
/// `TurnOutcome`.
///
/// Used by the streaming path on `StreamEvent::Done { content, .. }` and by
/// the non-streaming path after the tool loop exits, when only the textual
/// payload determines whether the turn produced a user-visible answer.
pub fn classify_final_content(content: &str) -> TurnOutcome {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return TurnOutcome::EmptyAnswer;
    }
    if looks_like_provider_tool_markup(trimmed) {
        return TurnOutcome::ProviderMarkupOnly;
    }
    TurnOutcome::FinalAnswer(content.to_string())
}

/// Heuristic check: does `text` look like provider-private tool-call markup
/// leaked into the assistant content (e.g. MiniMax `<minimax:tool_call>`)?
///
/// The check is intentionally conservative:
///
/// 1. The trimmed text must start with `<`. Natural-language answers — even
///    when they mention `<tool_call>` for explanatory purposes — almost never
///    start that way.
/// 2. The trimmed text must contain at least one tag from a small allow-list
///    of known provider markup tags. We do not flag arbitrary XML/HTML.
///
/// This avoids the unbounded "anything that looks like XML" false-positive
/// space while still catching the cases that have actually broken users
/// (MiniMax XML markup, OpenAI-compatible tool_call leaks).
pub fn looks_like_provider_tool_markup(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() || !trimmed.starts_with('<') {
        return false;
    }
    let lower = trimmed.to_ascii_lowercase();
    const TAGS: &[&str] = &[
        "<minimax:tool_call",
        "</minimax:tool_call",
        "<tool_call",
        "</tool_call",
        "<tool_use",
        "</tool_use",
        "<function_call",
        "</function_call",
    ];
    TAGS.iter().any(|tag| lower.contains(tag))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::{LLMResponse, LLMToolCall};

    fn response_text(content: &str) -> LLMResponse {
        LLMResponse::text(content)
    }

    fn response_with_tools(content: &str, tool_calls: Vec<LLMToolCall>) -> LLMResponse {
        LLMResponse::with_tools(content, tool_calls)
    }

    // ----- classify_turn_outcome -----

    #[test]
    fn classify_tool_calls_takes_priority_over_content() {
        let tc = LLMToolCall::new("c1", "web_search", r#"{"query":"x"}"#);
        let r = response_with_tools("ignored", vec![tc.clone()]);
        match classify_turn_outcome(&r) {
            TurnOutcome::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].id, "c1");
                assert_eq!(calls[0].name, "web_search");
            }
            other => panic!("expected ToolCalls, got {other:?}"),
        }
    }

    #[test]
    fn classify_empty_response_returns_empty_answer() {
        assert!(matches!(
            classify_turn_outcome(&response_text("\n\n\n")),
            TurnOutcome::EmptyAnswer
        ));
    }

    #[test]
    fn classify_blank_content_returns_empty_answer() {
        assert!(matches!(
            classify_turn_outcome(&response_text("   \t  ")),
            TurnOutcome::EmptyAnswer
        ));
    }

    #[test]
    fn classify_provider_markup_only_response() {
        let r = response_text("<minimax:tool_call>{\"name\":\"x\"}</minimax:tool_call>");
        assert!(matches!(
            classify_turn_outcome(&r),
            TurnOutcome::ProviderMarkupOnly
        ));
    }

    #[test]
    fn classify_plain_text_returns_final_answer() {
        let r = response_text("Hello, here is the answer.");
        match classify_turn_outcome(&r) {
            TurnOutcome::FinalAnswer(text) => {
                assert_eq!(text, "Hello, here is the answer.");
            }
            other => panic!("expected FinalAnswer, got {other:?}"),
        }
    }

    #[test]
    fn classify_preserves_original_content_with_surrounding_whitespace() {
        let r = response_text("\n\n  real answer  \n");
        match classify_turn_outcome(&r) {
            TurnOutcome::FinalAnswer(text) => {
                assert_eq!(text, "\n\n  real answer  \n");
            }
            other => panic!("expected FinalAnswer, got {other:?}"),
        }
    }

    // ----- classify_final_content -----

    #[test]
    fn classify_final_content_empty() {
        assert!(matches!(
            classify_final_content(""),
            TurnOutcome::EmptyAnswer
        ));
        assert!(matches!(
            classify_final_content("   \n"),
            TurnOutcome::EmptyAnswer
        ));
    }

    #[test]
    fn classify_final_content_provider_markup() {
        assert!(matches!(
            classify_final_content("<tool_call>{\"name\":\"x\"}</tool_call>"),
            TurnOutcome::ProviderMarkupOnly
        ));
    }

    #[test]
    fn classify_final_content_plain_text() {
        match classify_final_content("plain answer") {
            TurnOutcome::FinalAnswer(text) => assert_eq!(text, "plain answer"),
            other => panic!("expected FinalAnswer, got {other:?}"),
        }
    }

    // ----- looks_like_provider_tool_markup -----

    #[test]
    fn markup_detector_rejects_empty_and_whitespace() {
        assert!(!looks_like_provider_tool_markup(""));
        assert!(!looks_like_provider_tool_markup("   "));
        assert!(!looks_like_provider_tool_markup("\n\t  \n"));
    }

    #[test]
    fn markup_detector_rejects_plain_text() {
        assert!(!looks_like_provider_tool_markup("Hello world"));
        assert!(!looks_like_provider_tool_markup("Answer: 42."));
    }

    #[test]
    fn markup_detector_rejects_text_with_embedded_tag() {
        assert!(!looks_like_provider_tool_markup(
            "Here is what <tool_call> means: ..."
        ));
        assert!(!looks_like_provider_tool_markup(
            "See the `<minimax:tool_call>` element."
        ));
    }

    #[test]
    fn markup_detector_rejects_unknown_xml_tag() {
        assert!(!looks_like_provider_tool_markup("<note>hello</note>"));
        assert!(!looks_like_provider_tool_markup("<p>paragraph</p>"));
    }

    #[test]
    fn markup_detector_accepts_minimax_tag() {
        assert!(looks_like_provider_tool_markup(
            "<minimax:tool_call>{\"name\":\"web_search\"}</minimax:tool_call>"
        ));
    }

    #[test]
    fn markup_detector_case_insensitive() {
        assert!(looks_like_provider_tool_markup(
            "<MINIMAX:TOOL_CALL>{}</MINIMAX:TOOL_CALL>"
        ));
        assert!(looks_like_provider_tool_markup("<Tool_Call>{}</Tool_Call>"));
    }

    #[test]
    fn markup_detector_accepts_known_alternates() {
        assert!(looks_like_provider_tool_markup("<tool_call>{}</tool_call>"));
        assert!(looks_like_provider_tool_markup("<tool_use>{}</tool_use>"));
        assert!(looks_like_provider_tool_markup(
            "<function_call>{}</function_call>"
        ));
    }

    #[test]
    fn markup_detector_tolerates_leading_whitespace() {
        assert!(looks_like_provider_tool_markup(
            "  \n<minimax:tool_call>{}</minimax:tool_call>\n"
        ));
    }
}
