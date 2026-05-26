//! Pure formatting helpers extracted from `agent::loop`.
//!
//! Phase 4 alternative helper extraction: mechanical move only, no logic
//! changes. Visibility is `pub(super)` so only `agent::loop` can use these.

use crate::providers::{LLMResponse, LLMToolCall};
use crate::session::{Message, ToolCall};

use super::loop_guard::truncate_utf8;

pub(super) fn build_tool_result_preview(result: &str) -> Option<String> {
    let trimmed = result.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut preview = truncate_utf8(trimmed, 2000).to_string();
    if preview.len() < trimmed.len() {
        preview.push_str("\n...");
    }
    Some(preview)
}

pub(super) fn prettify_tool_arguments(raw_args: &str) -> String {
    serde_json::from_str::<serde_json::Value>(raw_args)
        .ok()
        .and_then(|value| serde_json::to_string_pretty(&value).ok())
        .unwrap_or_else(|| raw_args.to_string())
}

pub(super) fn resolve_streamed_response_text(delta_accum: &str, done_content: &str) -> String {
    let done_trimmed = done_content.trim();
    if done_trimmed.is_empty() {
        delta_accum.to_string()
    } else {
        done_content.to_string()
    }
}

pub(super) fn assistant_message_with_tool_calls(tool_calls: &[LLMToolCall]) -> Message {
    let mut assistant_msg = Message::assistant("");
    assistant_msg.tool_calls = Some(
        tool_calls
            .iter()
            .map(|tc| ToolCall {
                id: tc.id.clone(),
                name: tc.name.clone(),
                arguments: tc.arguments.clone(),
            })
            .collect(),
    );
    assistant_msg
}

pub(super) fn build_thinking_detail(response: &LLMResponse) -> Option<String> {
    if !response.has_tool_calls() {
        return None;
    }
    let content = response.content.trim();
    if content.is_empty() {
        return None;
    }

    let merged = format!("Model draft:\n{}", content);
    let mut detail = truncate_utf8(&merged, 8000).to_string();
    if detail.len() < merged.len() {
        detail.push_str("\n...");
    }
    Some(detail)
}

pub(super) fn build_tool_result_payload(result: &str, budget: usize) -> (String, Option<String>) {
    let sanitized_result = crate::utils::sanitize::sanitize_tool_result(result, budget);
    let result_preview = build_tool_result_preview(&sanitized_result);
    (sanitized_result, result_preview)
}
