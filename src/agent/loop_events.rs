//! Event publishing helpers (thinking status, tool call status, custom UI)
//! extracted from `agent::loop`.
//!
//! Phase 4 alternative helper extraction: mechanical move only.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::Serialize;

use crate::agent::agui_events;
use crate::bus::message::{
    OutboundMessageKind, OUTBOUND_CUSTOM_NAME_KEY, OUTBOUND_CUSTOM_PAYLOAD_KEY,
    OUTBOUND_CUSTOM_SUMMARY_KEY,
};
use crate::bus::{InboundMessage, MessageBus, OutboundMessage};
use crate::tools::ToolContext;

/// Inbound metadata flag set by streaming-capable transports (e.g. ACP-HTTP)
/// to opt the response path into token-level streaming via `OutboundMessage`
/// `Chunk`/`ChunkEnd` fragments. Channels that do not set this flag receive
/// the legacy single `Full` reply.
pub(super) const STREAMING_CAPABLE_METADATA_KEY: &str = "streaming_capable";
pub(super) const ACP_HTTP_CHANNEL: &str = "acp_http";
pub(super) const ACP_STDIO_CHANNEL: &str = "acp";

static THINKING_EVENT_SEQ: AtomicU64 = AtomicU64::new(1);

pub(super) fn supports_custom_ui_channel(channel: &str) -> bool {
    channel == ACP_HTTP_CHANNEL || channel == ACP_STDIO_CHANNEL
}

pub(super) fn is_streaming_capable(msg: &InboundMessage) -> bool {
    msg.metadata
        .get(STREAMING_CAPABLE_METADATA_KEY)
        .is_some_and(|v| v == "true")
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ThinkingStatusPayload {
    thought_id: String,
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    elapsed_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ToolCallStatusPayload {
    tool_call_id: String,
    tool_name: String,
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    elapsed_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result_preview: Option<String>,
}

#[derive(Debug, Clone)]
pub(super) struct ThinkingEventToken {
    thought_id: String,
    started_at: std::time::Instant,
}

pub(super) struct ThinkingScope {
    bus: Arc<MessageBus>,
    channel: String,
    chat_id: String,
    token: ThinkingEventToken,
}

pub(super) fn start_thinking_event_token() -> ThinkingEventToken {
    let id = THINKING_EVENT_SEQ.fetch_add(1, Ordering::Relaxed);
    ThinkingEventToken {
        thought_id: format!("thought_{}", id),
        started_at: std::time::Instant::now(),
    }
}

impl ThinkingScope {
    pub(super) async fn start(bus: Arc<MessageBus>, channel: &str, chat_id: &str) -> Self {
        let token = start_thinking_event_token();
        publish_thinking_status_event(
            &bus,
            Some(channel),
            Some(chat_id),
            &token.thought_id,
            "thinking",
            None,
            None,
        )
        .await;
        Self {
            bus,
            channel: channel.to_string(),
            chat_id: chat_id.to_string(),
            token,
        }
    }

    pub(super) async fn finish(self, detail: Option<&str>) {
        publish_thinking_status_event(
            &self.bus,
            Some(self.channel.as_str()),
            Some(self.chat_id.as_str()),
            &self.token.thought_id,
            "done",
            Some(self.token.started_at.elapsed().as_millis() as u64),
            detail,
        )
        .await;
    }
}

pub(super) async fn publish_custom_ui_event<T: Serialize>(
    bus: &Arc<MessageBus>,
    channel: Option<&str>,
    chat_id: Option<&str>,
    event_name: &str,
    payload: &T,
    summary: Option<&str>,
) {
    let (Some(channel), Some(chat_id)) = (channel, chat_id) else {
        return;
    };
    if !supports_custom_ui_channel(channel) {
        return;
    }
    let Ok(payload_json) = serde_json::to_string(payload) else {
        return;
    };
    let mut msg = OutboundMessage::new(channel, chat_id, "")
        .with_kind(OutboundMessageKind::Custom)
        .with_metadata(OUTBOUND_CUSTOM_NAME_KEY, event_name)
        .with_metadata(OUTBOUND_CUSTOM_PAYLOAD_KEY, &payload_json);
    if let Some(summary) = summary.filter(|v| !v.is_empty()) {
        msg = msg.with_metadata(OUTBOUND_CUSTOM_SUMMARY_KEY, summary);
    }
    let _ = bus.publish_outbound(msg).await;
}

pub(super) async fn publish_thinking_status_event(
    bus: &Arc<MessageBus>,
    channel: Option<&str>,
    chat_id: Option<&str>,
    thought_id: &str,
    status: &'static str,
    elapsed_ms: Option<u64>,
    detail: Option<&str>,
) {
    let payload = ThinkingStatusPayload {
        thought_id: thought_id.to_string(),
        status,
        elapsed_ms,
        detail: detail.map(ToString::to_string),
    };
    let summary: Option<String> = match status {
        "thinking" => Some("[thinking] started".to_string()),
        "done" => Some(
            elapsed_ms
                .map(|ms| format!("[thinking] done ({}ms)", ms))
                .unwrap_or_else(|| "[thinking] done".to_string()),
        ),
        _ => None,
    };
    publish_custom_ui_event(
        bus,
        channel,
        chat_id,
        agui_events::THINKING_STATUS,
        &payload,
        summary.as_deref(),
    )
    .await;
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn publish_tool_call_status_event(
    bus: &Arc<MessageBus>,
    ctx: &ToolContext,
    tool_call_id: &str,
    tool_name: &str,
    status: &'static str,
    elapsed_ms: Option<u64>,
    arguments: Option<&str>,
    result: Option<&str>,
    error: Option<&str>,
    result_preview: Option<&str>,
) {
    let payload = ToolCallStatusPayload {
        tool_call_id: tool_call_id.to_string(),
        tool_name: tool_name.to_string(),
        status,
        elapsed_ms,
        arguments: arguments.map(ToString::to_string),
        result: result.map(ToString::to_string),
        error: error.map(ToString::to_string),
        result_preview: result_preview.map(ToString::to_string),
    };
    let summary = match status {
        "started" => Some(format!("[tool] {} started", tool_name)),
        "done" => Some(format!(
            "[tool] {} done{}",
            tool_name,
            elapsed_ms
                .map(|ms| format!(" ({}ms)", ms))
                .unwrap_or_default()
        )),
        "failed" => Some(format!("[tool] {} failed", tool_name)),
        _ => None,
    };
    publish_custom_ui_event(
        bus,
        ctx.channel.as_deref(),
        ctx.chat_id.as_deref(),
        agui_events::TOOL_CALL,
        &payload,
        summary.as_deref(),
    )
    .await;
}

pub(super) enum ToolCallOutcome<'a> {
    Done,
    Failed { raw_error: &'a str },
}

pub(super) async fn publish_tool_call_started_event(
    bus: &Arc<MessageBus>,
    ctx: &ToolContext,
    tool_call_id: &str,
    tool_name: &str,
    pretty_args: &str,
) {
    publish_tool_call_status_event(
        bus,
        ctx,
        tool_call_id,
        tool_name,
        "started",
        None,
        Some(pretty_args),
        None,
        None,
        None,
    )
    .await;
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn publish_tool_call_finished_event(
    bus: &Arc<MessageBus>,
    ctx: &ToolContext,
    tool_call_id: &str,
    tool_name: &str,
    outcome: ToolCallOutcome<'_>,
    elapsed_ms: u64,
    pretty_args: &str,
    sanitized_result: &str,
    result_preview: Option<&str>,
) {
    let (status, error) = match outcome {
        ToolCallOutcome::Done => ("done", None),
        ToolCallOutcome::Failed { raw_error } => ("failed", Some(raw_error)),
    };
    publish_tool_call_status_event(
        bus,
        ctx,
        tool_call_id,
        tool_name,
        status,
        Some(elapsed_ms),
        Some(pretty_args),
        Some(sanitized_result),
        error,
        result_preview,
    )
    .await;
}
