//! A2UI message extraction and emission helpers extracted from `agent::loop`.
//!
//! Phase 4 alternative helper extraction: mechanical move only.

use std::sync::Arc;

use crate::agent::agui_events;
use crate::bus::MessageBus;

use super::loop_events::publish_custom_ui_event;
use super::mermaid::{
    build_a2ui_messages_from_mermaid_xychart, parse_mermaid_xychart_spec,
    strip_mermaid_xychart_block,
};

pub(super) fn is_a2ui_message_object(obj: &serde_json::Map<String, serde_json::Value>) -> bool {
    obj.contains_key("createSurface")
        || obj.contains_key("updateComponents")
        || obj.contains_key("updateDataModel")
        || obj.contains_key("deleteSurface")
        || obj.contains_key("beginRendering")
        || obj.contains_key("surfaceUpdate")
        || obj.contains_key("dataModelUpdate")
}

pub(super) fn normalize_a2ui_payload(value: serde_json::Value) -> Option<Vec<serde_json::Value>> {
    match value {
        serde_json::Value::Array(items) => {
            let messages: Vec<serde_json::Value> =
                items.into_iter().filter(|item| item.is_object()).collect();
            if messages.is_empty() {
                None
            } else {
                Some(messages)
            }
        }
        serde_json::Value::Object(obj) => {
            if let Some(serde_json::Value::Array(items)) = obj.get("messages") {
                let messages: Vec<serde_json::Value> = items
                    .iter()
                    .filter(|item| item.is_object())
                    .cloned()
                    .collect();
                if messages.is_empty() {
                    None
                } else {
                    Some(messages)
                }
            } else if let Some(serde_json::Value::Object(msg)) = obj.get("message") {
                Some(vec![serde_json::Value::Object(msg.clone())])
            } else if is_a2ui_message_object(&obj) {
                Some(vec![serde_json::Value::Object(obj)])
            } else {
                None
            }
        }
        _ => None,
    }
}

pub(super) fn parse_a2ui_messages_block(raw_block: &str) -> Option<Vec<serde_json::Value>> {
    let trimmed = raw_block.trim();
    if trimmed.is_empty() {
        return None;
    }
    let parsed: serde_json::Value = serde_json::from_str(trimmed).ok()?;
    normalize_a2ui_payload(parsed)
}

pub(super) fn extract_a2ui_messages_from_response(
    content: &str,
) -> (String, Vec<serde_json::Value>) {
    let mut cleaned = String::with_capacity(content.len());
    let mut messages = Vec::new();
    let mut cursor = 0usize;

    while let Some(start_rel) = content[cursor..].find("```a2ui") {
        let start = cursor + start_rel;
        cleaned.push_str(&content[cursor..start]);

        let header_end = match content[start..].find('\n') {
            Some(offset) => start + offset + 1,
            None => {
                cleaned.push_str(&content[start..]);
                cursor = content.len();
                break;
            }
        };

        let body_end = match content[header_end..].find("```") {
            Some(offset) => header_end + offset,
            None => {
                cleaned.push_str(&content[start..]);
                cursor = content.len();
                break;
            }
        };

        let block = &content[header_end..body_end];
        if let Some(mut parsed) = parse_a2ui_messages_block(block) {
            messages.append(&mut parsed);
        } else {
            cleaned.push_str(&content[start..(body_end + 3)]);
        }

        cursor = body_end + 3;
    }

    if cursor < content.len() {
        cleaned.push_str(&content[cursor..]);
    }

    let mut cleaned = cleaned.trim().to_string();
    if messages.is_empty() {
        if let Some(spec) = parse_mermaid_xychart_spec(&cleaned) {
            messages = build_a2ui_messages_from_mermaid_xychart(&spec);
            cleaned = strip_mermaid_xychart_block(&cleaned);
        }
    }
    (cleaned, messages)
}

pub(super) async fn emit_a2ui_messages(
    bus: &Arc<MessageBus>,
    channel: &str,
    chat_id: &str,
    messages: &[serde_json::Value],
) {
    for message in messages {
        publish_custom_ui_event(
            bus,
            Some(channel),
            Some(chat_id),
            agui_events::A2UI,
            message,
            Some("[a2ui] update"),
        )
        .await;
    }
}
