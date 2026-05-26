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

/// Streaming-mode guard that prevents provider tool markup from "appearing
/// on screen" before the terminating `Done` event has been classified.
///
/// Motivation: the non-streaming end-of-turn classification on its own is
/// not enough. The forwarder task in `process_message_streaming` receives a
/// sequence of `StreamEvent::Delta` chunks followed by `StreamEvent::Done`.
/// If we only classify the `Done` payload, any `Delta` chunks that contained
/// e.g. `<minimax:tool_call>` have already been forwarded to the channel
/// layer and rendered to the user.
///
/// The guard implements a minimal "look at the first non-whitespace
/// character" heuristic:
///
/// * Before any non-whitespace character is observed, content is buffered
///   silently.
/// * When the first non-whitespace character arrives, the guard decides
///   between two modes:
///   - `<`: switch to **buffering** mode. Subsequent deltas accumulate in
///     `buffer` and are not forwarded. The caller drains the buffer with
///     [`Self::take_buffered`] only after `Done` has been classified as a
///     legitimate `FinalAnswer`, so suspicious-looking markup is held back
///     until the terminating classification verdict says it is safe.
///   - anything else: switch to **streaming** mode. The accumulated leading
///     whitespace plus this chunk is returned for immediate forwarding, and
///     all subsequent deltas pass through verbatim.
///
/// This intentionally errs on the side of caution: any answer starting with
/// `<` (extremely rare in practice for natural-language final answers) loses
/// streaming responsiveness but is forwarded as a single Delta on success.
/// Provider markup never reaches the user.
#[derive(Debug, Default)]
pub struct StreamingMarkupGuard {
    decided: bool,
    is_buffering: bool,
    buffer: String,
}

impl StreamingMarkupGuard {
    /// Create a fresh guard in the "not yet decided" state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one `StreamEvent::Delta` text chunk to the guard.
    ///
    /// Returns `Some(text_to_forward)` if the chunk (or a previously
    /// withheld prefix combined with it) should be forwarded downstream
    /// immediately, or `None` if it should be withheld pending later
    /// disposition by [`Self::take_buffered`].
    pub fn on_delta(&mut self, text: &str) -> Option<String> {
        if self.decided {
            if self.is_buffering {
                self.buffer.push_str(text);
                return None;
            }
            return Some(text.to_string());
        }

        self.buffer.push_str(text);
        let trimmed = self.buffer.trim_start();
        if trimmed.is_empty() {
            return None;
        }

        self.decided = true;
        if trimmed.starts_with('<') {
            self.is_buffering = true;
            None
        } else {
            self.is_buffering = false;
            Some(std::mem::take(&mut self.buffer))
        }
    }

    /// Drain any buffered text accumulated during buffering mode.
    ///
    /// The caller should invoke this **only after** the corresponding
    /// `StreamEvent::Done` payload has been classified as
    /// [`TurnOutcome::FinalAnswer`]. The buffered text is then forwarded as
    /// a single Delta so the channel layer can render the final answer
    /// while remaining consistent with `Done.content`.
    ///
    /// Returns `None` in streaming mode (the buffer is always empty there)
    /// or when the guard is in buffering mode but has accumulated nothing.
    pub fn take_buffered(&mut self) -> Option<String> {
        if !self.is_buffering || self.buffer.is_empty() {
            return None;
        }
        Some(std::mem::take(&mut self.buffer))
    }

    /// Whether the guard is currently withholding content (buffering mode
    /// after seeing a `<`-prefixed first non-whitespace character).
    pub fn is_buffering(&self) -> bool {
        self.is_buffering
    }
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

    // ----- StreamingMarkupGuard -----

    #[test]
    fn guard_forwards_normal_text_immediately() {
        let mut g = StreamingMarkupGuard::new();
        assert_eq!(g.on_delta("Hello"), Some("Hello".to_string()));
        assert_eq!(g.on_delta(", world").as_deref(), Some(", world"));
        assert!(!g.is_buffering());
        assert!(g.take_buffered().is_none());
    }

    #[test]
    fn guard_holds_leading_whitespace_until_first_visible_char() {
        let mut g = StreamingMarkupGuard::new();
        // Pure whitespace deltas are buffered until decision time.
        assert!(g.on_delta("   ").is_none());
        assert!(g.on_delta("\n").is_none());
        // First visible char `H` triggers streaming mode; the accumulated
        // leading whitespace is flushed together with this chunk.
        let flushed = g.on_delta("Hello").expect("should flush on decision");
        assert_eq!(flushed, "   \nHello");
        // Subsequent chunks pass through verbatim.
        assert_eq!(g.on_delta(" world"), Some(" world".to_string()));
        assert!(!g.is_buffering());
    }

    #[test]
    fn guard_buffers_when_first_visible_char_is_angle_bracket() {
        let mut g = StreamingMarkupGuard::new();
        // First Delta starts with `<` — guard enters buffering mode.
        assert!(g.on_delta("<minimax").is_none());
        assert!(g.is_buffering());
        // Subsequent chunks continue to accumulate, nothing is forwarded.
        assert!(g.on_delta(":tool_call>").is_none());
        assert!(g.on_delta("{\"q\":\"x\"}").is_none());
        assert!(g.on_delta("</minimax:tool_call>").is_none());
        let drained = g.take_buffered().expect("buffer should contain markup");
        assert_eq!(
            drained,
            "<minimax:tool_call>{\"q\":\"x\"}</minimax:tool_call>"
        );
        // Calling take_buffered a second time yields nothing.
        assert!(g.take_buffered().is_none());
    }

    #[test]
    fn guard_buffers_when_chunked_markup_spans_first_delta_with_whitespace() {
        let mut g = StreamingMarkupGuard::new();
        // Leading whitespace alone is held.
        assert!(g.on_delta("\n  ").is_none());
        // First visible char `<` => buffering mode; the whitespace stays in
        // the buffer rather than leaking out.
        assert!(g.on_delta("<tool_call>").is_none());
        assert!(g.is_buffering());
        let drained = g.take_buffered().unwrap();
        assert_eq!(drained, "\n  <tool_call>");
    }

    #[test]
    fn guard_streaming_mode_returns_none_for_take_buffered() {
        let mut g = StreamingMarkupGuard::new();
        let _ = g.on_delta("Plain answer");
        assert!(!g.is_buffering());
        // Streaming mode never withholds content, so nothing to drain.
        assert!(g.take_buffered().is_none());
    }

    #[test]
    fn guard_take_buffered_does_not_flush_when_only_whitespace_accumulated() {
        let mut g = StreamingMarkupGuard::new();
        // Stream is exclusively whitespace deltas; no decision is ever made.
        assert!(g.on_delta("   \n").is_none());
        assert!(g.on_delta("\t").is_none());
        // Caller may invoke take_buffered after Done; buffer is non-empty
        // but the guard is not in buffering mode, so nothing is flushed.
        assert!(!g.is_buffering());
        assert!(g.take_buffered().is_none());
    }
}
