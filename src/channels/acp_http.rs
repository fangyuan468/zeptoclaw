//! ACP (Agent Client Protocol) streamable HTTP channel.
//!
//! Listens on a TCP port and accepts `POST /acp` requests carrying JSON-RPC 2.0
//! messages. For `session/prompt` requests the connection is kept open and the
//! agent reply is streamed back as Server-Sent Events:
//!
//! ```text
//! POST /acp  →  HTTP/1.1 200 OK
//!               Content-Type: text/event-stream
//!
//!               data: {"jsonrpc":"2.0","method":"session/update","params":{...}}\n\n
//!               data: {"jsonrpc":"2.0","id":1,"result":{"stopReason":"end_turn"}}\n\n
//! ```
//!
//! `initialize`, `session/new`, and `session/cancel` return synchronous JSON
//! responses on the same connection.
//!
//! The channel registers under the name `"acp_http"` so that its sessions are
//! independent from the stdio `"acp"` channel and the bus routes outbound
//! messages to the correct transport.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinSet;
use tracing::{debug, error, info, warn};

use crate::bus::message::{
    OutboundMessageKind, OUTBOUND_CUSTOM_NAME_KEY, OUTBOUND_CUSTOM_PAYLOAD_KEY,
};
use crate::bus::{InboundMessage, MessageBus, OutboundMessage};
use crate::config::{AcpChannelConfig, AcpHttpConfig};
use crate::error::{Result, ZeptoError};

use super::acp_protocol::{
    strip_agent_error_prefix, AgentCapabilities, AgentInfo, ContentBlock, InitializeResult,
    JsonRpcRequest, SessionInfo, SessionListParams, SessionListResult, SessionNewResult,
    SessionPromptResult, SessionUpdateParams, SessionUpdatePayload, JSONRPC_SERVER_ERROR,
};
use super::{BaseChannelConfig, Channel};

/// Channel name used for bus routing. Must differ from the stdio channel ("acp").
pub const ACP_HTTP_CHANNEL_NAME: &str = "acp_http";
const ACP_HTTP_SENDER_ID: &str = "acp_client";

/// Maximum size of a complete HTTP request (headers + body).
const MAX_REQUEST_BYTES: usize = 118_784; // 8 KB headers + ~110 KB body
/// Maximum prompt content after text extraction.
const MAX_PROMPT_BYTES: usize = 102_400;
/// Maximum concurrent ACP sessions.
const MAX_ACP_SESSIONS: usize = 1_000;
/// Maximum number of in-flight TCP connections accepted before new ones are
/// dropped.  Each accepted connection allocates a MAX_REQUEST_BYTES buffer
/// before auth is checked, so an unlimited accept loop is a memory DoS vector.
const MAX_CONCURRENT_CONNECTIONS: usize = 128;
/// How long (seconds) to wait for the agent to reply to session/prompt.
const PROMPT_TIMEOUT_SECS: u64 = 300;
/// Interval between SSE comment heartbeats while a prompt is in flight.
/// Must be short enough that no upstream (reqwest, docker-proxy, OS TCP
/// keepalive defaults) decides the stream is dead. 5s is well under the
/// observed ~7s "SSE ended without a response frame" failure window.
const SSE_HEARTBEAT_SECS: u64 = 5;

// --- HTTP response helpers ---

/// Returns the CORS header line when open, or an empty string when restricted.
fn cors_line(open_cors: bool) -> &'static str {
    if open_cors {
        "Access-Control-Allow-Origin: *\r\n"
    } else {
        ""
    }
}

/// Build a CORS preflight response (OPTIONS). When `open_cors` is false the
/// `Access-Control-Allow-Origin` header is omitted so browsers enforce
/// same-origin policy.
fn build_cors_preflight(open_cors: bool) -> String {
    format!(
        "HTTP/1.1 204 No Content\r\n{}Access-Control-Allow-Methods: POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\nContent-Length: 0\r\n\r\n",
        cors_line(open_cors),
    )
}

/// Build a 204 No Content response for JSON-RPC notifications (no body).
fn build_204_notification(open_cors: bool) -> String {
    format!(
        "HTTP/1.1 204 No Content\r\n{}Content-Length: 0\r\n\r\n",
        cors_line(open_cors),
    )
}

/// Build a self-contained HTTP error response with a correct Content-Length.
fn build_http_error(status_line: &str, body: &str, open_cors: bool) -> String {
    format!(
        "HTTP/1.1 {}\r\nContent-Type: application/json\r\n{}Content-Length: {}\r\nConnection: close\r\n\r\n{}",
        status_line,
        cors_line(open_cors),
        body.len(),
        body
    )
}

// --- Internal state ---

/// Per-session pending prompt: cancellation state set by session/cancel.
///
/// The original JSON-RPC request id is retained in the connection handler's
/// stack frame and passed directly to `stream_prompt_response`, so it does
/// not need to be stored here.
struct PendingPrompt {
    cancelled: bool,
}

/// One frame delivered from the bus-facing `send()` path to the SSE writer
/// (`stream_prompt_response`). Multiple frames may flow per in-flight
/// `session/prompt` request to support B4-zc token-level streaming.
///
/// Terminal variants (`Full`, `End`, `Error`) cause the writer to close the
/// connection and clean up both `state.pending` and `pending_http`. `Chunk`
/// is non-terminal: the writer emits an `agent_message_chunk` notification
/// and keeps waiting for more frames.
#[derive(Debug)]
enum PromptFragment {
    /// A streaming delta token. Non-terminal.
    Chunk(String),
    /// Complete non-streaming reply. Terminal. Writer emits a single
    /// `agent_message` `session/update` followed by the `session/prompt`
    /// JSON-RPC response.
    Full { content: String, cancelled: bool },
    /// End-of-stream marker for a reply that was already delivered via
    /// preceding `Chunk`s. Terminal. Writer emits only the `session/prompt`
    /// JSON-RPC response.
    End { cancelled: bool },
    /// Provider / runtime failure surfaced to the in-flight request as a
    /// JSON-RPC error. Terminal.
    Error(String),
    /// Structured custom event emitted as `session/update(agent_custom)`.
    /// Non-terminal.
    Custom {
        name: String,
        payload: serde_json::Value,
    },
}

/// Buffer depth for the per-prompt fragment channel. Sized generously so a
/// bursty LLM never backpressures the bus dispatcher (token-level streams
/// typically emit <100 deltas/s; 64 frames is ~half a second of head-room).
const PROMPT_FRAGMENT_BUFFER: usize = 64;

type PromptMap = Arc<Mutex<HashMap<String, mpsc::Sender<PromptFragment>>>>;

/// A live ACP session: working directory + last-active timestamp.
struct SessionEntry {
    cwd: String,
    last_active: std::time::Instant,
}

/// Mutable per-channel ACP state shared between the accept loop and `send()`.
struct AcpHttpState {
    /// Session IDs → session entry (cwd + last-active timestamp).
    sessions: HashMap<String, SessionEntry>,
    /// Tracks in-flight session/prompt requests so `send()` can retrieve the
    /// original request id and cancelled flag when the agent replies.
    pending: HashMap<String, PendingPrompt>,
}

impl AcpHttpState {
    fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            pending: HashMap::new(),
        }
    }

    /// Remove sessions whose `last_active` is older than `ttl`.  Also removes
    /// their pending entries so no orphan prompts remain.
    fn sweep_expired(&mut self, ttl: Duration) {
        let now = std::time::Instant::now();
        let expired: Vec<String> = self
            .sessions
            .iter()
            .filter(|(_, entry)| now.duration_since(entry.last_active) >= ttl)
            .map(|(id, _)| id.clone())
            .collect();
        for id in &expired {
            self.sessions.remove(id);
            self.pending.remove(id);
        }
    }
}

/// Parsed representation of an inbound HTTP request.
struct ParsedRequest {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InjectInboundParams {
    session_id: String,
    content: String,
    kind: String,
    request_id: String,
}

// --- Channel struct ---

/// ACP streamable HTTP channel.
///
/// Registers as `"acp_http"` in the channel manager. When `send()` is called
/// with an `OutboundMessage` for this channel, it delivers the agent reply to
/// the waiting HTTP connection handler via an in-process oneshot channel,
/// which then writes the SSE events and closes the connection.
pub struct AcpHttpChannel {
    config: AcpChannelConfig,
    http_config: AcpHttpConfig,
    base_config: BaseChannelConfig,
    bus: Arc<MessageBus>,
    running: Arc<AtomicBool>,
    state: Arc<Mutex<AcpHttpState>>,
    /// Session ID → sender half of the oneshot that bridges `send()` to the
    /// HTTP connection handler waiting for the agent reply.
    pending_http: PromptMap,
    /// Handle to the spawned accept-loop task; held so `stop()` can abort and
    /// await it, ensuring the TcpListener is released before returning.
    accept_handle: Option<tokio::task::JoinHandle<()>>,
    /// Tracks in-flight connection handler tasks so `stop()` can abort and
    /// await them all, preventing handlers from outliving the channel.
    conn_tasks: Arc<Mutex<JoinSet<()>>>,
}

impl AcpHttpChannel {
    pub fn new(
        config: AcpChannelConfig,
        http_config: AcpHttpConfig,
        base_config: BaseChannelConfig,
        bus: Arc<MessageBus>,
    ) -> Self {
        Self {
            config,
            http_config,
            base_config,
            bus,
            running: Arc::new(AtomicBool::new(false)),
            state: Arc::new(Mutex::new(AcpHttpState::new())),
            pending_http: Arc::new(Mutex::new(HashMap::new())),
            accept_handle: None,
            conn_tasks: Arc::new(Mutex::new(JoinSet::new())),
        }
    }

    // -------------------------------------------------------------------------
    // HTTP parsing helpers
    // -------------------------------------------------------------------------

    fn find_header_end(data: &[u8]) -> Option<usize> {
        data.windows(4).position(|w| w == b"\r\n\r\n")
    }

    fn parse_request(raw: &[u8]) -> Option<ParsedRequest> {
        let s = std::str::from_utf8(raw).ok()?;
        let pos = s.find("\r\n\r\n")?;
        let header_section = &s[..pos];
        let mut lines = header_section.lines();
        let request_line = lines.next()?;
        let mut parts = request_line.split_whitespace();
        let method = parts.next()?.to_uppercase();
        let path = parts.next()?.to_string();
        let mut headers = Vec::new();
        for line in lines {
            if let Some(colon) = line.find(':') {
                headers.push((
                    line[..colon].trim().to_string(),
                    line[colon + 1..].trim().to_string(),
                ));
            }
        }
        // Bound the body by Content-Length to prevent pipelined/malformed
        // trailing bytes from being folded into the JSON payload.
        let content_len = Self::content_length(&headers);
        let body_start = pos + 4;
        let body_end = (body_start + content_len).min(s.len());
        let body = s[body_start..body_end].to_string();
        Some(ParsedRequest {
            method,
            path,
            headers,
            body,
        })
    }

    fn content_length(headers: &[(String, String)]) -> usize {
        headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case("content-length"))
            .and_then(|(_, v)| v.trim().parse::<usize>().ok())
            .unwrap_or(0)
    }

    fn validate_auth(headers: &[(String, String)], token: &Option<String>) -> bool {
        let required = match token {
            Some(t) => t,
            None => return true,
        };
        let expected = format!("Bearer {}", required);
        headers.iter().any(|(n, v)| {
            n.eq_ignore_ascii_case("authorization") && constant_time_eq(v.trim(), &expected)
        })
    }

    // -------------------------------------------------------------------------
    // JSON-RPC / HTTP response builders
    // -------------------------------------------------------------------------

    fn json_rpc_result(id: Option<serde_json::Value>, result: serde_json::Value) -> String {
        serde_json::to_string(&serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result
        }))
        .unwrap_or_else(|_| {
            r#"{"jsonrpc":"2.0","error":{"code":-32603,"message":"serialize error"}}"#.to_string()
        })
    }

    fn json_rpc_error(id: Option<serde_json::Value>, code: i64, message: &str) -> String {
        serde_json::to_string(&serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": { "code": code, "message": message }
        }))
        .unwrap_or_else(|_| {
            r#"{"jsonrpc":"2.0","error":{"code":-32603,"message":"serialize error"}}"#.to_string()
        })
    }

    /// Wrap a JSON-RPC body in an HTTP 200 response. When `open_cors` is true
    /// the `Access-Control-Allow-Origin: *` header is included; otherwise it is
    /// omitted so browsers enforce same-origin policy.
    fn http_200(body: &str, open_cors: bool) -> String {
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n{}Content-Length: {}\r\nConnection: close\r\n\r\n{}",
            cors_line(open_cors),
            body.len(),
            body
        )
    }

    /// Wrap a JSON-RPC body in an HTTP 400 response. CORS header inclusion is
    /// controlled by `open_cors`.
    fn http_400(body: &str, open_cors: bool) -> String {
        format!(
            "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n{}Content-Length: {}\r\nConnection: close\r\n\r\n{}",
            cors_line(open_cors),
            body.len(),
            body
        )
    }

    /// Format a single SSE data event: `data: <payload>\n\n`.
    fn sse_event(data: &str) -> String {
        format!("data: {}\n\n", data)
    }

    /// Write a `session/update` JSON-RPC notification as one SSE frame.
    /// `session_update_kind` is the discriminator inside the update payload
    /// (`agent_message_chunk` for streaming deltas, `agent_message` for a
    /// non-streaming full reply). Returns `false` if the underlying TCP
    /// write failed (caller should `break` out of its emit loop).
    async fn write_session_update_sse(
        stream: &mut tokio::net::TcpStream,
        session_id: &str,
        content: &str,
        session_update_kind: &str,
    ) -> bool {
        let update = SessionUpdateParams {
            session_id: session_id.to_string(),
            update: SessionUpdatePayload {
                session_update: session_update_kind.to_string(),
                content: Some(ContentBlock::text(content)),
                tool_call_id: None,
                title: None,
                kind: None,
                status: None,
                custom_name: None,
                custom_payload: None,
            },
        };
        let Ok(update_json) = serde_json::to_string(&serde_json::json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": serde_json::to_value(&update).unwrap_or(serde_json::Value::Null)
        })) else {
            // Serialization can only fail for non-serializable values, which
            // is impossible here — but skip the frame rather than abort.
            return true;
        };
        let ev = Self::sse_event(&update_json);
        if stream.write_all(ev.as_bytes()).await.is_err() {
            return false;
        }
        let _ = stream.flush().await;
        true
    }

    /// Write a custom structured `session/update` notification as one SSE
    /// frame. Returns `false` on write failure.
    async fn write_custom_session_update_sse(
        stream: &mut tokio::net::TcpStream,
        session_id: &str,
        name: &str,
        payload: &serde_json::Value,
    ) -> bool {
        let update = SessionUpdateParams {
            session_id: session_id.to_string(),
            update: SessionUpdatePayload {
                session_update: "agent_custom".to_string(),
                content: None,
                tool_call_id: None,
                title: None,
                kind: None,
                status: None,
                custom_name: Some(name.to_string()),
                custom_payload: Some(payload.clone()),
            },
        };
        let Ok(update_json) = serde_json::to_string(&serde_json::json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": serde_json::to_value(&update).unwrap_or(serde_json::Value::Null)
        })) else {
            return true;
        };
        let ev = Self::sse_event(&update_json);
        if stream.write_all(ev.as_bytes()).await.is_err() {
            return false;
        }
        let _ = stream.flush().await;
        true
    }

    // -------------------------------------------------------------------------
    // Protocol handler helpers (synchronous methods)
    // -------------------------------------------------------------------------

    async fn do_initialize(
        _config: &AcpChannelConfig,
        id: Option<serde_json::Value>,
        params: Option<serde_json::Value>,
    ) -> String {
        if let Some(init_params) = params
            .and_then(|p| serde_json::from_value::<super::acp_protocol::InitializeParams>(p).ok())
        {
            if let Some(ref ci) = init_params.client_info {
                info!(
                    client_name = ?ci.name,
                    client_version = ?ci.version,
                    "ACP-HTTP: client initialized"
                );
            }
        }
        let result = InitializeResult {
            protocol_version: serde_json::json!("1"),
            agent_capabilities: AgentCapabilities {
                load_session: Some(false),
                prompt_capabilities: Some(serde_json::json!({
                    "image": false, "audio": false, "embeddedContext": false
                })),
                mcp_capabilities: Some(serde_json::json!({ "http": false, "sse": false })),
                session_capabilities: Some({
                    let mut m = HashMap::new();
                    m.insert("list".to_string(), serde_json::json!({}));
                    m
                }),
            },
            agent_info: Some(AgentInfo {
                name: "zeptoclaw".to_string(),
                title: Some("ZeptoClaw".to_string()),
                version: env!("CARGO_PKG_VERSION").to_string(),
            }),
            auth_methods: vec![],
        };
        match serde_json::to_value(result) {
            Ok(v) => Self::json_rpc_result(id, v),
            Err(e) => Self::json_rpc_error(id, -32603, &format!("serialize error: {}", e)),
        }
    }

    async fn do_session_new(
        state: &Arc<Mutex<AcpHttpState>>,
        base_config: &BaseChannelConfig,
        config: &AcpChannelConfig,
        id: Option<serde_json::Value>,
        params: Option<serde_json::Value>,
    ) -> String {
        if !base_config.is_allowed(ACP_HTTP_SENDER_ID) {
            return Self::json_rpc_error(id, -32000, "Unauthorized");
        }
        // ACP spec: cwd is required and MUST be an absolute path.
        let cwd = params
            .and_then(|p| serde_json::from_value::<super::acp_protocol::SessionNewParams>(p).ok())
            .and_then(|p| p.cwd)
            .filter(|c| !c.is_empty());
        let cwd = match cwd {
            Some(c) => c,
            None => return Self::json_rpc_error(id, -32602, "session/new: cwd is required"),
        };
        if !cwd.starts_with('/') {
            return Self::json_rpc_error(id, -32602, "session/new: cwd must be an absolute path");
        }
        if cwd.len() > 4096 {
            return Self::json_rpc_error(id, -32602, "session/new: cwd exceeds 4096 bytes");
        }
        if cwd_contains_traversal(&cwd) {
            return Self::json_rpc_error(
                id,
                -32602,
                "session/new: cwd must not contain '..' path segments",
            );
        }
        let session_id = format!("acph_{}", super::acp_protocol::new_id());
        {
            let mut st = state.lock().await;
            // Sweep expired sessions before the cap check when a TTL is configured.
            if let Some(ttl) = config.session_ttl_secs {
                st.sweep_expired(Duration::from_secs(ttl));
            }
            if st.sessions.len() >= MAX_ACP_SESSIONS {
                return Self::json_rpc_error(
                    id,
                    -32000,
                    &format!("too many sessions (limit: {})", MAX_ACP_SESSIONS),
                );
            }
            st.sessions.insert(
                session_id.clone(),
                SessionEntry {
                    cwd,
                    last_active: std::time::Instant::now(),
                },
            );
        }
        let result = SessionNewResult { session_id };
        match serde_json::to_value(result) {
            Ok(v) => Self::json_rpc_result(id, v),
            Err(e) => Self::json_rpc_error(id, -32603, &format!("serialize error: {}", e)),
        }
    }

    /// Handle a `session/cancel` request (id always present; notifications are
    /// handled earlier in `handle_connection` and never reach this function).
    async fn do_session_cancel(
        state: &Arc<Mutex<AcpHttpState>>,
        base_config: &BaseChannelConfig,
        id: Option<serde_json::Value>,
        params: Option<serde_json::Value>,
    ) -> String {
        if !base_config.is_allowed(ACP_HTTP_SENDER_ID) {
            return Self::json_rpc_error(id, -32000, "Unauthorized");
        }
        if let Some(p) = params.and_then(|p| {
            serde_json::from_value::<super::acp_protocol::SessionCancelParams>(p).ok()
        }) {
            let mut st = state.lock().await;
            if let Some(pending) = st.pending.get_mut(&p.session_id) {
                pending.cancelled = true;
                debug!(session_id = %p.session_id, "ACP-HTTP: marked prompt as cancelled");
            }
        }
        Self::json_rpc_result(id, serde_json::json!({}))
    }

    /// Handle x.openshell/inject_inbound: publish a synthetic inbound message
    /// into an existing ACP session without opening a `session/prompt`.
    async fn do_inject_inbound(
        state: &Arc<Mutex<AcpHttpState>>,
        base_config: &BaseChannelConfig,
        bus: &Arc<MessageBus>,
        id: Option<serde_json::Value>,
        params: Option<serde_json::Value>,
    ) -> String {
        if !base_config.is_allowed(ACP_HTTP_SENDER_ID) {
            return Self::json_rpc_error(id, -32000, "Unauthorized");
        }
        let params =
            match params.and_then(|p| serde_json::from_value::<InjectInboundParams>(p).ok()) {
                Some(p) => p,
                None => {
                    return Self::json_rpc_error(
                        id,
                        -32602,
                        "x.openshell/inject_inbound: missing or invalid params",
                    )
                }
            };
        if params.session_id.is_empty() || params.kind.is_empty() || params.request_id.is_empty() {
            return Self::json_rpc_error(
                id,
                -32602,
                "x.openshell/inject_inbound: sessionId/kind/requestId are required",
            );
        }
        {
            let mut st = state.lock().await;
            let Some(entry) = st.sessions.get_mut(&params.session_id) else {
                return Self::json_rpc_error(
                    id,
                    JSONRPC_SERVER_ERROR,
                    &format!("ACP: unknown session {}", params.session_id),
                );
            };
            entry.last_active = std::time::Instant::now();
        }
        let inbound = InboundMessage::new(
            ACP_HTTP_CHANNEL_NAME,
            ACP_HTTP_SENDER_ID,
            &params.session_id,
            &params.content,
        )
        .with_metadata("inject_kind", &params.kind)
        .with_metadata("request_id", &params.request_id);
        let delivered = match bus.publish_inbound_with_status(inbound).await {
            Ok(intercepted) => intercepted,
            Err(e) => {
                return Self::json_rpc_error(
                    id,
                    -32603,
                    &format!("x.openshell/inject_inbound: publish failed: {}", e),
                )
            }
        };
        Self::json_rpc_result(id, serde_json::json!({ "delivered": delivered }))
    }

    /// Handle session/list: return all live sessions with per-session metadata.
    async fn do_session_list(
        state: &Arc<Mutex<AcpHttpState>>,
        base_config: &BaseChannelConfig,
        id: Option<serde_json::Value>,
        params: Option<serde_json::Value>,
    ) -> String {
        if !base_config.is_allowed(ACP_HTTP_SENDER_ID) {
            return Self::json_rpc_error(id, -32000, "Unauthorized");
        }
        let st = state.lock().await;
        // Parse params; apply cwd filter when present (cursor/pagination not yet implemented).
        let list_params: Option<SessionListParams> = match params {
            None => None,
            Some(p) => match serde_json::from_value::<SessionListParams>(p) {
                Ok(lp) => Some(lp),
                Err(e) => {
                    return Self::json_rpc_error(id, -32602, &format!("Invalid params: {}", e));
                }
            },
        };
        let cwd_filter = list_params.and_then(|p| p.cwd);
        let sessions: Vec<SessionInfo> = st
            .sessions
            .iter()
            .filter(|(_, entry)| {
                if let Some(ref filter) = cwd_filter {
                    entry.cwd.as_str() == filter.as_str()
                } else {
                    true
                }
            })
            .map(|(sid, entry)| SessionInfo {
                session_id: sid.clone(),
                cwd: entry.cwd.clone(),
                title: None,
                updated_at: None,
                meta: Some(serde_json::json!({ "pending": st.pending.contains_key(sid) })),
            })
            .collect();
        let result = SessionListResult {
            sessions,
            next_cursor: None,
        };
        match serde_json::to_value(result) {
            Ok(v) => Self::json_rpc_result(id, v),
            Err(e) => Self::json_rpc_error(id, -32603, &format!("serialize error: {}", e)),
        }
    }

    // -------------------------------------------------------------------------
    // session/prompt: validation + bus publish, returns oneshot receiver
    // -------------------------------------------------------------------------

    /// Validate a session/prompt request and register it for async delivery.
    ///
    /// On success returns `Ok((session_id, rx))`.
    /// On validation failure returns `Ok(Err(error_body_string))`.
    /// On internal (bus) failure returns `Err(...)`.
    async fn register_prompt(
        state: &Arc<Mutex<AcpHttpState>>,
        pending_http: &PromptMap,
        base_config: &BaseChannelConfig,
        bus: &Arc<MessageBus>,
        id: Option<serde_json::Value>,
        params: Option<serde_json::Value>,
    ) -> Result<std::result::Result<(String, mpsc::Receiver<PromptFragment>), String>> {
        if !base_config.is_allowed(ACP_HTTP_SENDER_ID) {
            return Ok(Err(Self::json_rpc_error(id, -32000, "Unauthorized")));
        }
        let params: super::acp_protocol::SessionPromptParams =
            match params.and_then(|p| serde_json::from_value(p).ok()) {
                Some(p) => p,
                None => {
                    return Ok(Err(Self::json_rpc_error(
                        id,
                        -32602,
                        "session/prompt: missing or invalid params",
                    )));
                }
            };
        let session_id = params.session_id.clone();
        let content = super::acp_protocol::prompt_blocks_to_text(&params.prompt);
        if content.is_empty() {
            return Ok(Err(Self::json_rpc_error(
                id,
                -32602,
                "session/prompt: prompt content is empty",
            )));
        }
        if content.len() > MAX_PROMPT_BYTES {
            return Ok(Err(Self::json_rpc_error(
                id,
                -32602,
                &format!(
                    "session/prompt: content too large ({} bytes, limit {})",
                    content.len(),
                    MAX_PROMPT_BYTES
                ),
            )));
        }
        {
            let mut st = state.lock().await;
            match st.sessions.get_mut(&session_id) {
                None => {
                    return Ok(Err(Self::json_rpc_error(
                        id,
                        -32000,
                        &format!("ACP: unknown session {}", session_id),
                    )));
                }
                Some(entry) => {
                    entry.last_active = std::time::Instant::now();
                }
            }
            if st.pending.contains_key(&session_id) {
                return Ok(Err(Self::json_rpc_error(
                    id,
                    -32602,
                    "session/prompt: a prompt is already in flight for this session",
                )));
            }
            st.pending
                .insert(session_id.clone(), PendingPrompt { cancelled: false });
        }
        let (tx, rx) = mpsc::channel::<PromptFragment>(PROMPT_FRAGMENT_BUFFER);
        {
            pending_http.lock().await.insert(session_id.clone(), tx);
        }
        // Tell the agent loop this transport can consume token-level
        // streaming via OutboundMessage Chunk/ChunkEnd fragments.
        let inbound = InboundMessage::new(
            ACP_HTTP_CHANNEL_NAME,
            ACP_HTTP_SENDER_ID,
            &session_id,
            &content,
        )
        .with_metadata("streaming_capable", "true");
        if let Err(e) = bus.publish_inbound(inbound).await {
            // Roll back state so the session can accept a future prompt.
            state.lock().await.pending.remove(&session_id);
            pending_http.lock().await.remove(&session_id);
            return Err(ZeptoError::Channel(format!(
                "ACP-HTTP: failed to publish inbound: {}",
                e
            )));
        }
        debug!(session_id = %session_id, "ACP-HTTP: published session/prompt to bus");
        Ok(Ok((session_id, rx)))
    }

    // -------------------------------------------------------------------------
    // SSE streaming for session/prompt
    // -------------------------------------------------------------------------

    async fn stream_prompt_response(
        stream: &mut tokio::net::TcpStream,
        session_id: &str,
        id: Option<serde_json::Value>,
        mut rx: mpsc::Receiver<PromptFragment>,
        state: &Arc<Mutex<AcpHttpState>>,
        pending_http: &PromptMap,
        open_cors: bool,
    ) {
        let sse_headers = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n{}X-Accel-Buffering: no\r\n\r\n",
            cors_line(open_cors),
        );
        if stream.write_all(sse_headers.as_bytes()).await.is_err() {
            // Client disconnected before we could start.
            state.lock().await.pending.remove(session_id);
            pending_http.lock().await.remove(session_id);
            return;
        }
        let _ = stream.flush().await;

        // Consume fragments until a terminal variant is seen, the senders
        // are dropped, or the absolute prompt deadline expires. We
        // intentionally interleave an SSE comment "heartbeat" every
        // SSE_HEARTBEAT_SECS so a long stall (e.g. agent blocked waiting
        // on a user approval delivered through a different channel)
        // doesn't leave the HTTP/1.1 stream idle. Without it, the
        // scheduler's reqwest client / docker-proxy can decide the
        // stream has ended without a final response frame and bail
        // ~5–10s into the wait, even though the agent is healthily
        // blocked on `oneshot::recv()`. SSE comment lines (`: …\n\n`)
        // are silently skipped by any conforming SSE consumer.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(PROMPT_TIMEOUT_SECS);
        let heartbeat = Duration::from_secs(SSE_HEARTBEAT_SECS);
        loop {
            let frag = tokio::select! {
                biased;
                maybe = rx.recv() => match maybe {
                    Some(frag) => frag,
                    None => {
                        // All senders dropped without a terminal frame,
                        // e.g. channel stop() or abrupt shutdown.
                        let ev = Self::sse_event(&Self::json_rpc_error(
                            id.clone(),
                            -32603,
                            "agent session closed",
                        ));
                        let _ = stream.write_all(ev.as_bytes()).await;
                        let _ = stream.flush().await;
                        break;
                    }
                },
                _ = tokio::time::sleep_until(deadline) => {
                    let ev = Self::sse_event(&Self::json_rpc_error(
                        id.clone(),
                        -32603,
                        "session/prompt timed out",
                    ));
                    let _ = stream.write_all(ev.as_bytes()).await;
                    let _ = stream.flush().await;
                    break;
                }
                _ = tokio::time::sleep(heartbeat) => {
                    // SSE comment: line starting with ':' is ignored by
                    // any conforming consumer (browsers, reqwest's
                    // bytes_stream, etc.). All it does is keep the
                    // underlying TCP stream alive for intermediaries
                    // that close idle HTTP/1.1 connections.
                    if stream.write_all(b":ka\n\n").await.is_err() {
                        break;
                    }
                    let _ = stream.flush().await;
                    continue;
                }
            };

            match frag {
                PromptFragment::Chunk(content) => {
                    // Non-terminal — emit an agent_message_chunk
                    // notification and keep the connection open.
                    if !Self::write_session_update_sse(
                        stream,
                        session_id,
                        &content,
                        "agent_message_chunk",
                    )
                    .await
                    {
                        break;
                    }
                }
                PromptFragment::Custom { name, payload } => {
                    if !Self::write_custom_session_update_sse(stream, session_id, &name, &payload)
                        .await
                    {
                        break;
                    }
                }
                PromptFragment::Full { content, cancelled } => {
                    // Single-shot reply (non-streaming agents). Emit a
                    // full `agent_message` then the prompt response.
                    if !Self::write_session_update_sse(
                        stream,
                        session_id,
                        &content,
                        "agent_message",
                    )
                    .await
                    {
                        break;
                    }
                    let stop_reason = if cancelled { "cancelled" } else { "end_turn" };
                    let prompt_result = SessionPromptResult {
                        stop_reason: stop_reason.to_string(),
                    };
                    if let Ok(result_val) = serde_json::to_value(&prompt_result) {
                        let body = Self::json_rpc_result(id.clone(), result_val);
                        let ev = Self::sse_event(&body);
                        let _ = stream.write_all(ev.as_bytes()).await;
                    }
                    let _ = stream.flush().await;
                    break;
                }
                PromptFragment::End { cancelled } => {
                    // Streamed reply already delivered via preceding
                    // Chunks — only the prompt response is left to write.
                    let stop_reason = if cancelled { "cancelled" } else { "end_turn" };
                    let prompt_result = SessionPromptResult {
                        stop_reason: stop_reason.to_string(),
                    };
                    if let Ok(result_val) = serde_json::to_value(&prompt_result) {
                        let body = Self::json_rpc_result(id.clone(), result_val);
                        let ev = Self::sse_event(&body);
                        let _ = stream.write_all(ev.as_bytes()).await;
                    }
                    let _ = stream.flush().await;
                    break;
                }
                PromptFragment::Error(content) => {
                    debug!(
                        session_id = %session_id,
                        "ACP-HTTP: surfacing agent error as JSON-RPC error response"
                    );
                    let ev = Self::sse_event(&Self::json_rpc_error(
                        id.clone(),
                        JSONRPC_SERVER_ERROR,
                        strip_agent_error_prefix(&content),
                    ));
                    let _ = stream.write_all(ev.as_bytes()).await;
                    let _ = stream.flush().await;
                    break;
                }
            }
        }

        // Unified cleanup after loop exit (terminal frame, error, or
        // timeout). `send()` does not touch these maps itself — exactly
        // one cleanup path avoids double-remove races.
        state.lock().await.pending.remove(session_id);
        pending_http.lock().await.remove(session_id);
    }

    // -------------------------------------------------------------------------
    // TCP connection handler
    // -------------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    async fn handle_connection(
        mut stream: tokio::net::TcpStream,
        config: AcpChannelConfig,
        http_config: AcpHttpConfig,
        base_config: BaseChannelConfig,
        bus: Arc<MessageBus>,
        state: Arc<Mutex<AcpHttpState>>,
        pending_http: PromptMap,
    ) {
        // When auth is configured, omit CORS headers so browsers enforce
        // same-origin policy. Without auth, keep open CORS for local dev.
        let open_cors = http_config.auth_token.is_none();

        // Read the full request (headers + body) with a per-request size cap.
        // The outer 30s deadline prevents slow-loris attacks where a client
        // drips one byte at a time, resetting a per-read timeout indefinitely.
        let mut buf = vec![0u8; MAX_REQUEST_BYTES];
        let mut total = 0usize;
        let read_result = tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                if total >= buf.len() {
                    return Err("payload too large");
                }
                match tokio::time::timeout(Duration::from_secs(10), stream.read(&mut buf[total..]))
                    .await
                {
                    Ok(Ok(0)) => break,
                    Ok(Ok(n)) => {
                        total += n;
                        if let Some(hend) = Self::find_header_end(&buf[..total]) {
                            // Only parse the header section here. `parse_request`
                            // does `from_utf8` on whatever slice it gets, and the
                            // body is still being read — if a UTF-8 multi-byte
                            // character (e.g. CJK) lands across a TCP boundary,
                            // `from_utf8(&buf[..total])` will fail and the old
                            // `else break` would abort the read with body still
                            // truncated, then the second `parse_request` at the
                            // bottom of `handle_connection` would 400 with body
                            // `{}`. Headers are ASCII (HTTP/1.1 RFC 7230 §3.2.4)
                            // so a header-only slice is always valid UTF-8 once
                            // `find_header_end` has matched.
                            if let Some(req) = Self::parse_request(&buf[..hend + 4]) {
                                let body_received = total - hend - 4;
                                if body_received >= Self::content_length(&req.headers) {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }
                    }
                    Ok(Err(e)) => {
                        debug!("ACP-HTTP: read error: {}", e);
                        return Err("read error");
                    }
                    Err(_) => {
                        debug!("ACP-HTTP: read timeout");
                        return Err("read timeout");
                    }
                }
            }
            Ok(())
        })
        .await;
        match read_result {
            Ok(Ok(())) => {}
            Ok(Err("payload too large")) => {
                let resp = build_http_error(
                    "413 Payload Too Large",
                    r#"{"error":"payload too large"}"#,
                    open_cors,
                );
                let _ = stream.write_all(resp.as_bytes()).await;
                return;
            }
            Ok(Err(_)) => return,
            Err(_) => {
                debug!("ACP-HTTP: total request deadline exceeded (slow-loris?)");
                return;
            }
        }
        if total == 0 {
            return;
        }

        let req = match Self::parse_request(&buf[..total]) {
            Some(r) => r,
            None => {
                let _ = stream
                    .write_all(Self::http_400("{}", open_cors).as_bytes())
                    .await;
                return;
            }
        };

        // CORS preflight.
        if req.method == "OPTIONS" {
            let resp = build_cors_preflight(open_cors);
            let _ = stream.write_all(resp.as_bytes()).await;
            return;
        }

        // Only POST /acp or POST / is accepted.
        if req.path != "/acp" && req.path != "/" {
            let resp = build_http_error("404 Not Found", r#"{"error":"not found"}"#, open_cors);
            let _ = stream.write_all(resp.as_bytes()).await;
            return;
        }
        if req.method != "POST" {
            let resp = build_http_error(
                "405 Method Not Allowed",
                r#"{"error":"method not allowed"}"#,
                open_cors,
            );
            let _ = stream.write_all(resp.as_bytes()).await;
            return;
        }

        // Bearer token auth.
        if !Self::validate_auth(&req.headers, &http_config.auth_token) {
            let resp =
                build_http_error("401 Unauthorized", r#"{"error":"unauthorized"}"#, open_cors);
            let _ = stream.write_all(resp.as_bytes()).await;
            return;
        }

        // Parse JSON-RPC envelope.
        let rpc: JsonRpcRequest = match serde_json::from_str(&req.body) {
            Ok(r) => r,
            Err(e) => {
                let body = Self::json_rpc_error(None, -32700, &format!("parse error: {}", e));
                let resp = Self::http_400(&body, open_cors);
                let _ = stream.write_all(resp.as_bytes()).await;
                return;
            }
        };
        if rpc.jsonrpc != "2.0" {
            let body =
                Self::json_rpc_error(rpc.id, -32600, "Invalid Request: jsonrpc must be \"2.0\"");
            let resp = Self::http_200(&body, open_cors);
            let _ = stream.write_all(resp.as_bytes()).await;
            return;
        }

        let id = rpc.id.clone();
        let params = rpc.params.clone();

        // JSON-RPC notifications (id absent) must not receive a response body
        // (JSON-RPC 2.0 §4.1).  Apply any state-mutating side-effects for known
        // notification methods, then return 204 No Content with no body.
        // Unknown/unsupported notification methods are silently ignored.
        if id.is_none() {
            if rpc.method.as_str() == "session/cancel" {
                if let Some(p) = params.and_then(|p| {
                    serde_json::from_value::<super::acp_protocol::SessionCancelParams>(p).ok()
                }) {
                    let mut st = state.lock().await;
                    if let Some(pending) = st.pending.get_mut(&p.session_id) {
                        pending.cancelled = true;
                        debug!(session_id = %p.session_id, "ACP-HTTP: marked prompt as cancelled (notification)");
                    }
                }
            }
            let resp = build_204_notification(open_cors);
            let _ = stream.write_all(resp.as_bytes()).await;
            return;
        }

        match rpc.method.as_str() {
            "initialize" => {
                let body = Self::do_initialize(&config, id, params).await;
                let resp = Self::http_200(&body, open_cors);
                let _ = stream.write_all(resp.as_bytes()).await;
            }
            "session/new" => {
                let body = Self::do_session_new(&state, &base_config, &config, id, params).await;
                let resp = Self::http_200(&body, open_cors);
                let _ = stream.write_all(resp.as_bytes()).await;
            }
            "session/cancel" => {
                let body = Self::do_session_cancel(&state, &base_config, id, params).await;
                let resp = Self::http_200(&body, open_cors);
                let _ = stream.write_all(resp.as_bytes()).await;
            }
            "session/list" => {
                let body = Self::do_session_list(&state, &base_config, id, params).await;
                let resp = Self::http_200(&body, open_cors);
                let _ = stream.write_all(resp.as_bytes()).await;
            }
            "x.openshell/inject_inbound" => {
                let body = Self::do_inject_inbound(&state, &base_config, &bus, id, params).await;
                let resp = Self::http_200(&body, open_cors);
                let _ = stream.write_all(resp.as_bytes()).await;
            }
            "session/prompt" => {
                match Self::register_prompt(
                    &state,
                    &pending_http,
                    &base_config,
                    &bus,
                    id.clone(),
                    params,
                )
                .await
                {
                    Err(e) => {
                        error!("ACP-HTTP: session/prompt internal error: {}", e);
                        let body =
                            Self::json_rpc_error(id, -32603, &format!("internal error: {}", e));
                        let resp = Self::http_200(&body, open_cors);
                        let _ = stream.write_all(resp.as_bytes()).await;
                    }
                    Ok(Err(err_body)) => {
                        // Validation failure — plain JSON response, no SSE.
                        let resp = Self::http_200(&err_body, open_cors);
                        let _ = stream.write_all(resp.as_bytes()).await;
                    }
                    Ok(Ok((session_id, rx))) => {
                        // Validation passed — stream SSE response.
                        Self::stream_prompt_response(
                            &mut stream,
                            &session_id,
                            id,
                            rx,
                            &state,
                            &pending_http,
                            open_cors,
                        )
                        .await;
                    }
                }
            }
            _ => {
                let body =
                    Self::json_rpc_error(id, -32601, &format!("method not found: {}", rpc.method));
                let resp = Self::http_200(&body, open_cors);
                let _ = stream.write_all(resp.as_bytes()).await;
            }
        }
    }

    // -------------------------------------------------------------------------
    // TCP accept loop
    // -------------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    async fn run_accept_loop(
        listener: TcpListener,
        config: AcpChannelConfig,
        http_config: AcpHttpConfig,
        base_config: BaseChannelConfig,
        bus: Arc<MessageBus>,
        state: Arc<Mutex<AcpHttpState>>,
        pending_http: PromptMap,
        running: Arc<AtomicBool>,
        conn_tasks: Arc<Mutex<JoinSet<()>>>,
    ) {
        info!(
            "ACP-HTTP: listening on {}:{}",
            http_config.bind, http_config.port
        );
        let conn_sem = Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_CONNECTIONS));
        while running.load(Ordering::SeqCst) {
            match tokio::time::timeout(Duration::from_secs(1), listener.accept()).await {
                Ok(Ok((stream, addr))) => {
                    debug!("ACP-HTTP: accepted connection from {}", addr);
                    let permit = match Arc::clone(&conn_sem).try_acquire_owned() {
                        Ok(p) => p,
                        Err(_) => {
                            debug!(
                                "ACP-HTTP: connection limit ({}) reached, dropping {}",
                                MAX_CONCURRENT_CONNECTIONS, addr
                            );
                            continue;
                        }
                    };
                    let config = config.clone();
                    let http_config = http_config.clone();
                    let base_config = base_config.clone();
                    let bus = Arc::clone(&bus);
                    let state = Arc::clone(&state);
                    let pending_http = Arc::clone(&pending_http);
                    {
                        let mut tasks = conn_tasks.lock().await;
                        // Reap any already-finished handlers before registering
                        // the new one, preventing unbounded JoinSet growth.
                        while tasks.try_join_next().is_some() {}
                        tasks.spawn(async move {
                            let _permit = permit; // released when handler completes
                            Self::handle_connection(
                                stream,
                                config,
                                http_config,
                                base_config,
                                bus,
                                state,
                                pending_http,
                            )
                            .await;
                        });
                    }
                }
                Ok(Err(e)) => {
                    error!("ACP-HTTP: accept error: {}", e);
                }
                Err(_) => {
                    // accept timeout — loop back and recheck `running`
                }
            }
        }
        running.store(false, Ordering::SeqCst);
        info!("ACP-HTTP: accept loop exited");
    }
}

// -------------------------------------------------------------------------
// Channel trait implementation
// -------------------------------------------------------------------------

#[async_trait]
impl Channel for AcpHttpChannel {
    fn name(&self) -> &str {
        ACP_HTTP_CHANNEL_NAME
    }

    async fn start(&mut self) -> Result<()> {
        if self.running.swap(true, Ordering::SeqCst) {
            info!("ACP-HTTP channel already running");
            return Ok(());
        }
        let addr = format!("{}:{}", self.http_config.bind, self.http_config.port);
        let listener = TcpListener::bind(&addr).await.map_err(|e| {
            // Reset the flag so is_running() doesn't report a stale true state.
            self.running.store(false, Ordering::SeqCst);
            ZeptoError::Channel(format!("ACP-HTTP: failed to bind {}: {}", addr, e))
        })?;

        let config = self.config.clone();
        let http_config = self.http_config.clone();
        let base_config = self.base_config.clone();
        let bus = Arc::clone(&self.bus);
        let state = Arc::clone(&self.state);
        let pending_http = Arc::clone(&self.pending_http);
        let running = Arc::clone(&self.running);
        let conn_tasks = Arc::clone(&self.conn_tasks);

        let handle = tokio::spawn(async move {
            Self::run_accept_loop(
                listener,
                config,
                http_config,
                base_config,
                bus,
                state,
                pending_http,
                running,
                conn_tasks,
            )
            .await;
        });
        self.accept_handle = Some(handle);
        if self.http_config.auth_token.is_none() {
            warn!(
                "ACP-HTTP channel started without an auth_token on {}:{}. \
                 Combined with wildcard CORS, any webpage can reach this endpoint \
                 (DNS rebinding risk). Set acp.http.auth_token in your config.",
                self.http_config.bind, self.http_config.port
            );
        } else {
            info!(
                "ACP-HTTP channel started on {}:{}",
                self.http_config.bind, self.http_config.port
            );
        }
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        self.running.store(false, Ordering::SeqCst);
        // Drain pending_http: dropping the oneshot senders causes any connection
        // handlers currently awaiting a prompt response to receive RecvError and
        // return, so no HTTP connection is left hanging after stop().
        self.pending_http.lock().await.clear();
        // Clear state.pending so sessions are not permanently marked in-flight
        // across a stop/restart cycle (supervisor may restart the channel).
        self.state.lock().await.pending.clear();
        // Abort and await the accept-loop task so the TcpListener is released
        // before this method returns (mirrors the stdio AcpChannel pattern).
        if let Some(handle) = self.accept_handle.take() {
            handle.abort();
            let _ = handle.await;
        }
        // Abort and drain all in-flight connection handler tasks so they do not
        // outlive the channel after stop() returns.
        {
            let mut tasks = self.conn_tasks.lock().await;
            tasks.abort_all();
            while tasks.join_next().await.is_some() {}
        }
        Ok(())
    }

    /// Called by the bus dispatcher when the agent produces a reply for a
    /// session that originated from this channel.
    ///
    /// Looks up the waiting HTTP connection handler via `pending_http` and
    /// delivers one [`PromptFragment`] per call. Multiple calls may land on
    /// the same in-flight prompt (B4-zc token streaming); the SSE writer
    /// side is responsible for cleaning up `state.pending` and
    /// `pending_http` after a terminal fragment.
    async fn send(&self, msg: OutboundMessage) -> Result<()> {
        if msg.channel != ACP_HTTP_CHANNEL_NAME {
            return Ok(());
        }
        let session_id = msg.chat_id.clone();

        // Check that the session is known and peek the cancellation flag
        // without consuming the pending record — the SSE writer consumes
        // it on its way out (after emitting the terminal frame).
        let (session_exists, cancelled) = {
            let st = self.state.lock().await;
            let exists = st.sessions.contains_key(&session_id);
            let cancelled = st
                .pending
                .get(&session_id)
                .map(|p| p.cancelled)
                .unwrap_or(false);
            (exists, cancelled)
        };

        if !session_exists {
            debug!(
                session_id = %session_id,
                "ACP-HTTP: outbound for unknown session, skipping"
            );
            return Ok(());
        }

        // `pending_http` holds a clone of the per-prompt mpsc sender. We
        // clone once per call and release the lock immediately so the
        // mutex never spans the `tx.send().await` below.
        let sender = self.pending_http.lock().await.get(&session_id).cloned();
        let Some(tx) = sender else {
            debug!(
                session_id = %session_id,
                kind = ?msg.kind,
                "ACP-HTTP: outbound with no waiting handler, dropping"
            );
            return Ok(());
        };

        let fragment = if msg.is_error() {
            PromptFragment::Error(msg.content)
        } else {
            // Tools that emit a `for_user` keep-typing status (e.g.
            // `searxng_search` returning `ToolOutput::split(real_results,
            // "Searching (SearXNG)...")`) reach us as kind=Full with
            // metadata `keep_typing=true`. They are intermediate progress
            // strings, not the turn's final reply — route them as Chunk so
            // the SSE writer emits an `agent_message_chunk` notification
            // without closing the pending prompt. Without this remap the
            // first such status string ends the turn early and the LLM's
            // real reply lands on a closed pending_http and gets dropped.
            let keep_typing = msg.metadata.get("keep_typing").is_some_and(|v| v == "true");
            match msg.kind {
                OutboundMessageKind::Chunk => PromptFragment::Chunk(msg.content),
                OutboundMessageKind::ChunkEnd => PromptFragment::End { cancelled },
                OutboundMessageKind::Custom => {
                    let Some(name) = msg.metadata.get(OUTBOUND_CUSTOM_NAME_KEY).cloned() else {
                        debug!("ACP-HTTP: custom outbound missing custom.name, dropping");
                        return Ok(());
                    };
                    let Some(raw_payload) = msg.metadata.get(OUTBOUND_CUSTOM_PAYLOAD_KEY) else {
                        debug!("ACP-HTTP: custom outbound missing custom.payload, dropping");
                        return Ok(());
                    };
                    let payload: serde_json::Value = match serde_json::from_str(raw_payload) {
                        Ok(v) => v,
                        Err(e) => {
                            debug!(error = %e, "ACP-HTTP: invalid custom.payload JSON, dropping");
                            return Ok(());
                        }
                    };
                    PromptFragment::Custom { name, payload }
                }
                OutboundMessageKind::Full if keep_typing => PromptFragment::Chunk(msg.content),
                OutboundMessageKind::Full => PromptFragment::Full {
                    content: msg.content,
                    cancelled,
                },
            }
        };

        // Intentional `await` (not `try_send`): dropping a Chunk would
        // corrupt the visible reply on the client. Backpressure here is
        // bounded by the per-prompt mpsc capacity (small, so producer
        // pacing tracks the SSE writer's drain rate). If the receiver
        // was dropped (client disconnected) this is a no-op; the SSE
        // writer's cleanup path will handle state.
        let _ = tx.send(fragment).await;
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    fn is_allowed(&self, user_id: &str) -> bool {
        self.base_config.is_allowed(user_id)
    }
}

// -------------------------------------------------------------------------
// Free helpers
// -------------------------------------------------------------------------

/// Returns `true` if `cwd` contains `..` path segments, which could be used
/// for path traversal attacks.
pub(super) fn cwd_contains_traversal(cwd: &str) -> bool {
    std::path::Path::new(cwd)
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
}

/// Constant-time string comparison (prevents timing side-channels on auth tokens).
///
/// Does NOT short-circuit on length mismatch — XORs up to `max(a.len(), b.len())`
/// bytes so that token length cannot be inferred via timing.
fn constant_time_eq(a: &str, b: &str) -> bool {
    let a = a.as_bytes();
    let b = b.as_bytes();
    let max_len = a.len().max(b.len());
    // Fold the length difference into the accumulator so a length mismatch
    // produces a non-zero result without revealing which side is longer.
    let mut result = (a.len() ^ b.len()) as u8;
    for i in 0..max_len {
        result |= a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0);
    }
    result == 0
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::agui_events;
    use crate::bus::MessageBus;
    use crate::config::{AcpChannelConfig, AcpHttpConfig};

    fn make_channel() -> AcpHttpChannel {
        let bus = Arc::new(MessageBus::new());
        AcpHttpChannel::new(
            AcpChannelConfig::default(),
            AcpHttpConfig::default(),
            BaseChannelConfig::new(ACP_HTTP_CHANNEL_NAME),
            bus,
        )
    }

    #[test]
    fn test_channel_name() {
        let ch = make_channel();
        assert_eq!(ch.name(), ACP_HTTP_CHANNEL_NAME);
    }

    #[test]
    fn test_is_not_running_initially() {
        assert!(!make_channel().is_running());
    }

    #[test]
    fn test_prompt_blocks_to_text_text_only() {
        use super::super::acp_protocol::PromptContentBlock;
        let blocks = vec![
            PromptContentBlock::Text {
                text: "Hello".to_string(),
            },
            PromptContentBlock::Text {
                text: "World".to_string(),
            },
        ];
        assert_eq!(
            crate::channels::acp_protocol::prompt_blocks_to_text(&blocks),
            "Hello\nWorld"
        );
    }

    #[test]
    fn test_prompt_blocks_to_text_skips_non_text() {
        use super::super::acp_protocol::PromptContentBlock;
        let blocks = vec![
            PromptContentBlock::Text {
                text: "only this".to_string(),
            },
            PromptContentBlock::Other,
        ];
        assert_eq!(
            crate::channels::acp_protocol::prompt_blocks_to_text(&blocks),
            "only this"
        );
    }

    #[tokio::test]
    async fn test_send_ignores_wrong_channel() {
        let ch = make_channel();
        let session_id = "acph_test".to_string();
        {
            let mut st = ch.state.lock().await;
            st.sessions.insert(
                session_id.clone(),
                SessionEntry {
                    cwd: "/test".to_string(),
                    last_active: std::time::Instant::now(),
                },
            );
            st.pending
                .insert(session_id.clone(), PendingPrompt { cancelled: false });
        }
        let msg = OutboundMessage {
            channel: "telegram".to_string(),
            chat_id: session_id.clone(),
            content: "hello".to_string(),
            reply_to: None,
            metadata: Default::default(),
            kind: Default::default(),
        };
        assert!(ch.send(msg).await.is_ok());
        // pending entry must be untouched
        let st = ch.state.lock().await;
        assert!(
            st.pending.contains_key(&session_id),
            "wrong-channel send must not consume the pending entry"
        );
    }

    #[tokio::test]
    async fn test_send_skips_unknown_session() {
        let ch = make_channel();
        let msg = OutboundMessage {
            channel: ACP_HTTP_CHANNEL_NAME.to_string(),
            chat_id: "acph_ghost".to_string(),
            content: "hello".to_string(),
            reply_to: None,
            metadata: Default::default(),
            kind: Default::default(),
        };
        assert!(ch.send(msg).await.is_ok());
        assert!(ch.state.lock().await.sessions.is_empty());
    }

    #[tokio::test]
    async fn test_send_delivers_full_fragment() {
        let ch = make_channel();
        let session_id = "acph_deliver".to_string();
        let (tx, mut rx) = mpsc::channel::<PromptFragment>(PROMPT_FRAGMENT_BUFFER);
        {
            let mut st = ch.state.lock().await;
            st.sessions.insert(
                session_id.clone(),
                SessionEntry {
                    cwd: "/test".to_string(),
                    last_active: std::time::Instant::now(),
                },
            );
            st.pending
                .insert(session_id.clone(), PendingPrompt { cancelled: false });
            ch.pending_http.lock().await.insert(session_id.clone(), tx);
        }
        let msg = OutboundMessage {
            channel: ACP_HTTP_CHANNEL_NAME.to_string(),
            chat_id: session_id.clone(),
            content: "agent reply".to_string(),
            reply_to: None,
            metadata: Default::default(),
            kind: Default::default(),
        };
        assert!(ch.send(msg).await.is_ok());
        match rx.recv().await.expect("must receive fragment") {
            PromptFragment::Full { content, cancelled } => {
                assert_eq!(content, "agent reply");
                assert!(!cancelled);
            }
            other => panic!("expected Full, got {other:?}"),
        }
        // send() must NOT consume pending; cleanup is the writer's job.
        assert!(ch.state.lock().await.pending.contains_key(&session_id));
        assert!(ch.pending_http.lock().await.contains_key(&session_id));
    }

    #[tokio::test]
    async fn test_send_marks_cancelled() {
        let ch = make_channel();
        let session_id = "acph_cancel".to_string();
        let (tx, mut rx) = mpsc::channel::<PromptFragment>(PROMPT_FRAGMENT_BUFFER);
        {
            let mut st = ch.state.lock().await;
            st.sessions.insert(
                session_id.clone(),
                SessionEntry {
                    cwd: "/test".to_string(),
                    last_active: std::time::Instant::now(),
                },
            );
            st.pending
                .insert(session_id.clone(), PendingPrompt { cancelled: true });
            ch.pending_http.lock().await.insert(session_id.clone(), tx);
        }
        let msg = OutboundMessage {
            channel: ACP_HTTP_CHANNEL_NAME.to_string(),
            chat_id: session_id.clone(),
            content: "reply after cancel".to_string(),
            reply_to: None,
            metadata: Default::default(),
            kind: Default::default(),
        };
        assert!(ch.send(msg).await.is_ok());
        match rx.recv().await.expect("must receive fragment") {
            PromptFragment::Full { cancelled, .. } => {
                assert!(
                    cancelled,
                    "cancelled flag must be forwarded to HTTP handler"
                );
            }
            other => panic!("expected Full, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_send_forwards_error_flag() {
        // When agent/loop.rs marks an outbound as an error via
        // OutboundMessage::mark_error(), send() must surface it as a
        // PromptFragment::Error so the SSE writer emits a JSON-RPC error
        // response instead of a normal session/update.
        let ch = make_channel();
        let session_id = "acph_err".to_string();
        let (tx, mut rx) = mpsc::channel::<PromptFragment>(PROMPT_FRAGMENT_BUFFER);
        {
            let mut st = ch.state.lock().await;
            st.sessions.insert(
                session_id.clone(),
                SessionEntry {
                    cwd: "/test".to_string(),
                    last_active: std::time::Instant::now(),
                },
            );
            st.pending
                .insert(session_id.clone(), PendingPrompt { cancelled: false });
            ch.pending_http.lock().await.insert(session_id.clone(), tx);
        }
        let mut msg = OutboundMessage::new(
            ACP_HTTP_CHANNEL_NAME,
            &session_id,
            "Error: Provider error: upstream connect error",
        );
        msg.mark_error();
        assert!(ch.send(msg).await.is_ok());
        match rx.recv().await.expect("must receive fragment") {
            PromptFragment::Error(content) => {
                assert!(content.contains("upstream connect error"));
            }
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_send_chunk_emits_chunk_fragment() {
        // Streaming token: kind=Chunk should map to PromptFragment::Chunk
        // and must NOT touch pending (the End fragment closes the prompt).
        let ch = make_channel();
        let session_id = "acph_stream".to_string();
        let (tx, mut rx) = mpsc::channel::<PromptFragment>(PROMPT_FRAGMENT_BUFFER);
        {
            let mut st = ch.state.lock().await;
            st.sessions.insert(
                session_id.clone(),
                SessionEntry {
                    cwd: "/test".to_string(),
                    last_active: std::time::Instant::now(),
                },
            );
            st.pending
                .insert(session_id.clone(), PendingPrompt { cancelled: false });
            ch.pending_http.lock().await.insert(session_id.clone(), tx);
        }
        let msg = OutboundMessage::new(ACP_HTTP_CHANNEL_NAME, &session_id, "tok ")
            .with_kind(OutboundMessageKind::Chunk);
        assert!(ch.send(msg).await.is_ok());
        match rx.recv().await.expect("must receive fragment") {
            PromptFragment::Chunk(content) => assert_eq!(content, "tok "),
            other => panic!("expected Chunk, got {other:?}"),
        }
        assert!(ch.state.lock().await.pending.contains_key(&session_id));
    }

    #[tokio::test]
    async fn test_send_chunk_end_emits_end_fragment() {
        // ChunkEnd is the stream terminator: PromptFragment::End must carry
        // the pending cancelled flag, send() still leaves cleanup to writer.
        let ch = make_channel();
        let session_id = "acph_stream_end".to_string();
        let (tx, mut rx) = mpsc::channel::<PromptFragment>(PROMPT_FRAGMENT_BUFFER);
        {
            let mut st = ch.state.lock().await;
            st.sessions.insert(
                session_id.clone(),
                SessionEntry {
                    cwd: "/test".to_string(),
                    last_active: std::time::Instant::now(),
                },
            );
            st.pending
                .insert(session_id.clone(), PendingPrompt { cancelled: true });
            ch.pending_http.lock().await.insert(session_id.clone(), tx);
        }
        let msg = OutboundMessage::new(ACP_HTTP_CHANNEL_NAME, &session_id, "")
            .with_kind(OutboundMessageKind::ChunkEnd);
        assert!(ch.send(msg).await.is_ok());
        match rx.recv().await.expect("must receive fragment") {
            PromptFragment::End { cancelled } => assert!(cancelled),
            other => panic!("expected End, got {other:?}"),
        }
        assert!(ch.state.lock().await.pending.contains_key(&session_id));
    }

    #[tokio::test]
    async fn test_send_custom_emits_custom_fragment() {
        let ch = make_channel();
        let session_id = "acph_custom".to_string();
        let (tx, mut rx) = mpsc::channel::<PromptFragment>(PROMPT_FRAGMENT_BUFFER);
        {
            let mut st = ch.state.lock().await;
            st.sessions.insert(
                session_id.clone(),
                SessionEntry {
                    cwd: "/test".to_string(),
                    last_active: std::time::Instant::now(),
                },
            );
            st.pending
                .insert(session_id.clone(), PendingPrompt { cancelled: false });
            ch.pending_http.lock().await.insert(session_id.clone(), tx);
        }
        let msg = OutboundMessage::new(ACP_HTTP_CHANNEL_NAME, &session_id, "")
            .with_kind(OutboundMessageKind::Custom)
            .with_metadata(OUTBOUND_CUSTOM_NAME_KEY, agui_events::APPROVAL_REQUEST)
            .with_metadata(
                OUTBOUND_CUSTOM_PAYLOAD_KEY,
                r#"{"requestId":"01JTEST","toolName":"write_file"}"#,
            );
        assert!(ch.send(msg).await.is_ok());
        match rx.recv().await.expect("must receive fragment") {
            PromptFragment::Custom { name, payload } => {
                assert_eq!(name, agui_events::APPROVAL_REQUEST);
                assert_eq!(payload["toolName"], "write_file");
            }
            other => panic!("expected Custom, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_deny_by_default_blocks_session_new() {
        let bus = Arc::new(MessageBus::new());
        let base = BaseChannelConfig {
            name: ACP_HTTP_CHANNEL_NAME.to_string(),
            allowlist: vec![],
            deny_by_default: true,
        };
        let ch = AcpHttpChannel::new(
            AcpChannelConfig {
                deny_by_default: true,
                ..AcpChannelConfig::default()
            },
            AcpHttpConfig::default(),
            base,
            bus,
        );
        let result = AcpHttpChannel::do_session_new(
            &ch.state,
            &ch.base_config,
            &ch.config,
            Some(serde_json::json!(1)),
            Some(serde_json::json!({ "cwd": "/workspace" })),
        )
        .await;
        assert!(
            result.contains("Unauthorized"),
            "deny_by_default must block session/new"
        );
        assert!(ch.state.lock().await.sessions.is_empty());
    }

    #[tokio::test]
    async fn test_inject_inbound_unknown_session_returns_server_error() {
        let ch = make_channel();
        let body = AcpHttpChannel::do_inject_inbound(
            &ch.state,
            &ch.base_config,
            &ch.bus,
            Some(serde_json::json!(1)),
            Some(serde_json::json!({
                "sessionId": "acph_missing",
                "content": "yes",
                "kind": "approval_response",
                "requestId": "req_1"
            })),
        )
        .await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["error"]["code"], JSONRPC_SERVER_ERROR);
    }

    #[tokio::test]
    async fn test_inject_inbound_reports_delivered_true_when_intercepted() {
        let ch = make_channel();
        let session_id = "acph_inject_true".to_string();
        {
            let mut st = ch.state.lock().await;
            st.sessions.insert(
                session_id.clone(),
                SessionEntry {
                    cwd: "/test".to_string(),
                    last_active: std::time::Instant::now(),
                },
            );
        }
        ch.bus
            .set_inbound_interceptor(Arc::new(|msg: &InboundMessage| {
                msg.metadata.get("inject_kind").map(String::as_str) == Some("approval_response")
                    && msg.metadata.get("request_id").map(String::as_str) == Some("req_1")
            }));
        let body = AcpHttpChannel::do_inject_inbound(
            &ch.state,
            &ch.base_config,
            &ch.bus,
            Some(serde_json::json!(1)),
            Some(serde_json::json!({
                "sessionId": session_id,
                "content": "yes",
                "kind": "approval_response",
                "requestId": "req_1"
            })),
        )
        .await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["result"]["delivered"], true);
    }

    #[tokio::test]
    async fn test_inject_inbound_reports_delivered_false_when_queued() {
        let ch = make_channel();
        let session_id = "acph_inject_false".to_string();
        {
            let mut st = ch.state.lock().await;
            st.sessions.insert(
                session_id.clone(),
                SessionEntry {
                    cwd: "/test".to_string(),
                    last_active: std::time::Instant::now(),
                },
            );
        }
        let body = AcpHttpChannel::do_inject_inbound(
            &ch.state,
            &ch.base_config,
            &ch.bus,
            Some(serde_json::json!(1)),
            Some(serde_json::json!({
                "sessionId": session_id,
                "content": "yes",
                "kind": "approval_response",
                "requestId": "req_1"
            })),
        )
        .await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["result"]["delivered"], false);
        let inbound = ch
            .bus
            .consume_inbound()
            .await
            .expect("inbound must be queued");
        assert_eq!(inbound.chat_id, "acph_inject_false");
        assert_eq!(
            inbound.metadata.get("request_id").map(String::as_str),
            Some("req_1")
        );
    }

    #[test]
    fn test_constant_time_eq_same() {
        assert!(constant_time_eq("hello", "hello"));
    }

    #[test]
    fn test_constant_time_eq_different() {
        assert!(!constant_time_eq("hello", "world"));
        assert!(!constant_time_eq("hello", "hello!"));
    }

    #[test]
    fn test_sse_event_format() {
        let ev = AcpHttpChannel::sse_event(r#"{"foo":"bar"}"#);
        assert_eq!(ev, "data: {\"foo\":\"bar\"}\n\n");
    }

    #[test]
    fn test_http_200_content_length() {
        let body = r#"{"result":"ok"}"#;
        let resp = AcpHttpChannel::http_200(body, true);
        assert!(resp.contains(&format!("Content-Length: {}", body.len())));
        assert!(resp.ends_with(body));
    }

    #[tokio::test]
    async fn test_session_list_no_sessions() {
        let ch = make_channel();
        let body = AcpHttpChannel::do_session_list(
            &ch.state,
            &ch.base_config,
            Some(serde_json::json!(1)),
            None,
        )
        .await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["result"]["sessions"], serde_json::json!([]));
    }

    #[tokio::test]
    async fn test_session_list_empty() {
        let ch = make_channel();
        let body = AcpHttpChannel::do_session_list(
            &ch.state,
            &ch.base_config,
            Some(serde_json::json!(1)),
            None,
        )
        .await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["result"]["sessions"], serde_json::json!([]));
    }

    #[tokio::test]
    async fn test_session_list_shows_sessions_with_pending_flag() {
        let ch = make_channel();
        let sid_a = "acph_list_a".to_string();
        let sid_b = "acph_list_b".to_string();
        {
            let mut st = ch.state.lock().await;
            st.sessions.insert(
                sid_a.clone(),
                SessionEntry {
                    cwd: "/test".to_string(),
                    last_active: std::time::Instant::now(),
                },
            );
            st.sessions.insert(
                sid_b.clone(),
                SessionEntry {
                    cwd: "/test".to_string(),
                    last_active: std::time::Instant::now(),
                },
            );
            st.pending
                .insert(sid_a.clone(), PendingPrompt { cancelled: false });
        }
        let body = AcpHttpChannel::do_session_list(
            &ch.state,
            &ch.base_config,
            Some(serde_json::json!(1)),
            None,
        )
        .await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        let sessions = v["result"]["sessions"].as_array().unwrap();
        assert_eq!(sessions.len(), 2);
        let find = |sid: &str| sessions.iter().find(|s| s["sessionId"] == sid).cloned();
        assert_eq!(find(&sid_a).unwrap()["_meta"]["pending"], true);
        assert_eq!(find(&sid_b).unwrap()["_meta"]["pending"], false);
    }

    // -------------------------------------------------------------------------
    // Regression: notification semantics (issue #388, second bug)
    // -------------------------------------------------------------------------

    /// A `session/cancel` sent as a notification (no id) must apply the
    /// cancellation flag and return exactly `HTTP_204_NOTIFICATION` — no body.
    #[tokio::test]
    async fn test_session_cancel_notification_returns_204_no_body() {
        let ch = make_channel();
        let session_id = "acph_notif_cancel".to_string();
        {
            let mut st = ch.state.lock().await;
            st.sessions.insert(
                session_id.clone(),
                SessionEntry {
                    cwd: "/test".to_string(),
                    last_active: std::time::Instant::now(),
                },
            );
            st.pending
                .insert(session_id.clone(), PendingPrompt { cancelled: false });
        }

        // Simulate `handle_connection` notification path: id is None.
        // The early-return block processes the cancel and writes HTTP_204_NOTIFICATION.
        // We test the state effect directly (the HTTP write is an I/O side-effect).
        {
            let params = serde_json::json!({ "sessionId": session_id });
            if let Ok(p) = serde_json::from_value::<
                crate::channels::acp_protocol::SessionCancelParams,
            >(params.clone())
            {
                let mut st = ch.state.lock().await;
                if let Some(pending) = st.pending.get_mut(&p.session_id) {
                    pending.cancelled = true;
                }
            }
        }

        assert!(
            ch.state
                .lock()
                .await
                .pending
                .get(&session_id)
                .unwrap()
                .cancelled,
            "cancel notification must set the cancelled flag"
        );
    }

    /// `session/cancel` sent as a request (id present) must return a JSON-RPC
    /// result body — NOT 204.
    #[tokio::test]
    async fn test_session_cancel_request_returns_json_result() {
        let ch = make_channel();
        let session_id = "acph_req_cancel".to_string();
        {
            let mut st = ch.state.lock().await;
            st.sessions.insert(
                session_id.clone(),
                SessionEntry {
                    cwd: "/test".to_string(),
                    last_active: std::time::Instant::now(),
                },
            );
            st.pending
                .insert(session_id.clone(), PendingPrompt { cancelled: false });
        }

        let body = AcpHttpChannel::do_session_cancel(
            &ch.state,
            &ch.base_config,
            Some(serde_json::json!(42)),
            Some(serde_json::json!({ "sessionId": session_id })),
        )
        .await;

        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["id"], 42, "response must echo the request id");
        assert!(
            v["result"].is_object(),
            "result must be present for a request"
        );
        assert!(v.get("error").is_none(), "no error field expected");
    }

    /// The `build_204_notification` helper must produce a well-formed 204 with
    /// no body and Content-Length: 0.
    #[test]
    fn test_http_204_notification_has_no_body() {
        let resp = build_204_notification(true);
        assert!(resp.contains("204 No Content"), "must be a 204 status");
        assert!(
            resp.contains("Content-Length: 0"),
            "Content-Length must be 0"
        );
        // The response must end immediately after the blank line — no body.
        assert!(
            resp.ends_with("\r\n\r\n"),
            "response must end with the blank line and no trailing body"
        );
    }

    // -------------------------------------------------------------------------
    // initialize → session/new → session/list round-trip
    // -------------------------------------------------------------------------

    /// `initialize` must return a well-formed response with the required fields
    /// and must not include a `clientId` field (not part of the ACP spec).
    #[tokio::test]
    async fn test_initialize_returns_spec_fields() {
        let ch = make_channel();
        let body =
            AcpHttpChannel::do_initialize(&ch.config, Some(serde_json::json!(1)), None).await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(
            v["result"]["protocolVersion"].is_string(),
            "protocolVersion must be a string"
        );
        assert!(
            v["result"]["agentCapabilities"].is_object(),
            "agentCapabilities must be present"
        );
        assert!(
            v["result"].get("clientId").is_none(),
            "clientId must not be in the response"
        );
    }

    /// Happy-path round-trip: `initialize` → `session/new` → valid `sessionId`.
    #[tokio::test]
    async fn test_initialize_to_session_new_round_trip() {
        let ch = make_channel();

        AcpHttpChannel::do_initialize(&ch.config, Some(serde_json::json!(1)), None).await;

        let new_body = AcpHttpChannel::do_session_new(
            &ch.state,
            &ch.base_config,
            &ch.config,
            Some(serde_json::json!(2)),
            Some(serde_json::json!({ "cwd": "/workspace" })),
        )
        .await;
        let new_v: serde_json::Value = serde_json::from_str(&new_body).unwrap();

        assert!(
            new_v.get("error").is_none(),
            "session/new must succeed: {new_body}"
        );
        assert!(
            new_v["result"]["sessionId"].as_str().is_some(),
            "session/new must return a sessionId: {new_body}"
        );
    }

    /// Multi-round positive path: `initialize` → `session/new` → `session/list`.
    #[tokio::test]
    async fn test_initialize_to_session_new_to_session_list_round_trip() {
        let ch = make_channel();

        AcpHttpChannel::do_initialize(&ch.config, Some(serde_json::json!(1)), None).await;

        let new_body = AcpHttpChannel::do_session_new(
            &ch.state,
            &ch.base_config,
            &ch.config,
            Some(serde_json::json!(2)),
            Some(serde_json::json!({ "cwd": "/workspace" })),
        )
        .await;
        let session_id = serde_json::from_str::<serde_json::Value>(&new_body).unwrap()["result"]
            ["sessionId"]
            .as_str()
            .expect("session/new must succeed")
            .to_string();

        let list_body = AcpHttpChannel::do_session_list(
            &ch.state,
            &ch.base_config,
            Some(serde_json::json!(3)),
            None,
        )
        .await;
        let list_v: serde_json::Value = serde_json::from_str(&list_body).unwrap();

        assert!(
            list_v.get("error").is_none(),
            "session/list must succeed: {list_body}"
        );
        let sessions = list_v["result"]["sessions"].as_array().unwrap();
        assert_eq!(sessions.len(), 1, "exactly one session must be listed");
        assert_eq!(
            sessions[0]["sessionId"].as_str().unwrap(),
            session_id,
            "listed sessionId must match the one returned by session/new"
        );
    }

    /// `session/new` must reject requests that omit `cwd`.
    #[tokio::test]
    async fn test_session_new_rejects_missing_cwd() {
        let ch = make_channel();
        let body = AcpHttpChannel::do_session_new(
            &ch.state,
            &ch.base_config,
            &ch.config,
            Some(serde_json::json!(1)),
            None,
        )
        .await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(
            v["error"]["code"], -32602,
            "missing cwd must give -32602: {body}"
        );
        assert!(ch.state.lock().await.sessions.is_empty());
    }

    /// `session/new` must reject a relative (non-absolute) `cwd`.
    #[tokio::test]
    async fn test_session_new_rejects_relative_cwd() {
        let ch = make_channel();
        let body = AcpHttpChannel::do_session_new(
            &ch.state,
            &ch.base_config,
            &ch.config,
            Some(serde_json::json!(1)),
            Some(serde_json::json!({ "cwd": "relative/path" })),
        )
        .await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(
            v["error"]["code"], -32602,
            "relative cwd must give -32602: {body}"
        );
        assert!(ch.state.lock().await.sessions.is_empty());
    }

    /// `session/new` must accept an absolute `cwd` and store it.
    #[tokio::test]
    async fn test_session_new_stores_absolute_cwd() {
        let ch = make_channel();
        let body = AcpHttpChannel::do_session_new(
            &ch.state,
            &ch.base_config,
            &ch.config,
            Some(serde_json::json!(1)),
            Some(serde_json::json!({ "cwd": "/home/user/project" })),
        )
        .await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(
            v.get("error").is_none(),
            "absolute cwd must succeed: {body}"
        );
        let sid = v["result"]["sessionId"].as_str().unwrap().to_string();
        let st = ch.state.lock().await;
        assert_eq!(
            st.sessions.get(&sid).map(|e| e.cwd.as_str()),
            Some("/home/user/project")
        );
    }

    /// `register_prompt` must reject calls with an unknown session ID.
    #[tokio::test]
    async fn test_register_prompt_marks_streaming_capable() {
        // Regression: register_prompt must tag the published inbound with
        // streaming_capable=true so the agent loop opts into the
        // Chunk/ChunkEnd streaming dispatch path.
        let ch = make_channel();
        let session_id = "acph_stream_meta".to_string();
        {
            let mut st = ch.state.lock().await;
            st.sessions.insert(
                session_id.clone(),
                SessionEntry {
                    cwd: "/test".to_string(),
                    last_active: std::time::Instant::now(),
                },
            );
        }

        let result = AcpHttpChannel::register_prompt(
            &ch.state,
            &ch.pending_http,
            &ch.base_config,
            &ch.bus,
            Some(serde_json::json!(1)),
            Some(serde_json::json!({
                "sessionId": session_id,
                "prompt": [{ "type": "text", "text": "hi" }],
            })),
        )
        .await
        .expect("register_prompt should succeed")
        .expect("session should be known");
        assert_eq!(result.0, session_id);

        let inbound = ch
            .bus
            .consume_inbound()
            .await
            .expect("inbound must be published");
        assert_eq!(
            inbound
                .metadata
                .get("streaming_capable")
                .map(String::as_str),
            Some("true"),
            "inbound must carry streaming_capable=true",
        );
    }

    #[tokio::test]
    async fn test_register_prompt_blocked_without_client_id() {
        let ch = make_channel();

        let result = AcpHttpChannel::register_prompt(
            &ch.state,
            &ch.pending_http,
            &ch.base_config,
            &ch.bus,
            Some(serde_json::json!(2)),
            Some(serde_json::json!({
                "sessionId": "acph_nonexistent",
                "prompt": [{ "type": "text", "text": "hello" }]
            })),
        )
        .await
        .expect("register_prompt must not return an Err variant here");

        assert!(
            result.is_err(),
            "register_prompt must reject calls with an unknown session"
        );
        let err_body = result.unwrap_err();
        assert!(
            err_body.contains("-32600") || err_body.contains("-32000"),
            "rejection must be an RPC error: {err_body}"
        );
    }

    // -------------------------------------------------------------------------
    // Read-loop UTF-8 boundary regression
    //
    // Repro for the HTTP-400 we hit when scope-isolated session/prompt requests
    // started shipping multi-KB CJK histories: TCP read returns a partial buffer
    // whose final byte falls inside a multi-byte UTF-8 char. The pre-fix code
    // called `parse_request(&buf[..total])`, whose `from_utf8` would fail on the
    // truncated tail; the read loop then `else break`d with the body still
    // incomplete and the second `parse_request` 400'd with body `{}`.
    //
    // The fix passes `&buf[..hend + 4]` (header section only, ASCII-safe) to
    // `parse_request` inside the read loop. These tests pin that contract.
    // -------------------------------------------------------------------------

    #[test]
    fn test_read_loop_parses_headers_with_partial_utf8_body() {
        let body = "\u{20AC}\u{20AC}\u{20AC}\u{20AC}".repeat(1024);
        let body_bytes = body.as_bytes();
        let headers = format!(
            "POST / HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body_bytes.len()
        );
        let mut buf = Vec::new();
        buf.extend_from_slice(headers.as_bytes());
        buf.extend_from_slice(body_bytes);

        let hend = AcpHttpChannel::find_header_end(&buf).expect("header terminator present");

        for body_chunk in [1usize, 5, 17, 31, 64, 257] {
            let total = hend + 4 + body_chunk;
            let raw = &buf[..total];

            // Old behavior: full-buffer parse fails the moment a multi-byte
            // char straddles `total` — exactly the bug.
            if std::str::from_utf8(raw).is_err() {
                assert!(
                    AcpHttpChannel::parse_request(raw).is_none(),
                    "sanity: pre-fix parse_request must fail on partial UTF-8"
                );
            }

            // Post-fix contract: header-only slice is always parseable, and
            // the read loop can compute body_received vs Content-Length.
            let header_slice = &buf[..hend + 4];
            let req = AcpHttpChannel::parse_request(header_slice)
                .expect("header-only slice must parse regardless of body bytes");
            let cl = AcpHttpChannel::content_length(&req.headers);
            assert_eq!(cl, body_bytes.len());
            let body_received = total - hend - 4;
            assert!(
                body_received < cl,
                "loop must keep reading; body_received={body_received} cl={cl}"
            );
        }
    }

    #[test]
    fn test_read_loop_break_when_full_body_received() {
        let body = "\u{20AC}\u{20AC}".to_string();
        let body_bytes = body.as_bytes();
        let headers = format!(
            "POST / HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
            body_bytes.len()
        );
        let mut buf = Vec::new();
        buf.extend_from_slice(headers.as_bytes());
        buf.extend_from_slice(body_bytes);

        let hend = AcpHttpChannel::find_header_end(&buf).unwrap();
        let req = AcpHttpChannel::parse_request(&buf[..hend + 4]).unwrap();
        let cl = AcpHttpChannel::content_length(&req.headers);
        let body_received = buf.len() - hend - 4;
        assert!(
            body_received >= cl,
            "loop must terminate; body_received={body_received} cl={cl}"
        );
    }

    // -------------------------------------------------------------------------
    // CORS hardening tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_http_200_no_cors_when_auth() {
        let body = r#"{"result":"ok"}"#;
        let resp = AcpHttpChannel::http_200(body, false);
        assert!(
            !resp.contains("Access-Control-Allow-Origin"),
            "CORS header must be absent when open_cors=false"
        );
        assert!(resp.contains(&format!("Content-Length: {}", body.len())));
    }

    #[test]
    fn test_cors_preflight_open() {
        let resp = build_cors_preflight(true);
        assert!(resp.contains("Access-Control-Allow-Origin: *"));
        assert!(resp.contains("Access-Control-Allow-Methods"));
        assert!(resp.contains("204 No Content"));
    }

    #[test]
    fn test_cors_preflight_restricted() {
        let resp = build_cors_preflight(false);
        assert!(
            !resp.contains("Access-Control-Allow-Origin"),
            "CORS origin header must be absent when restricted"
        );
        assert!(resp.contains("Access-Control-Allow-Methods"));
        assert!(resp.contains("204 No Content"));
    }

    #[test]
    fn test_build_http_error_cors() {
        let open = build_http_error("400 Bad Request", r#"{"error":"bad"}"#, true);
        assert!(open.contains("Access-Control-Allow-Origin: *"));

        let restricted = build_http_error("400 Bad Request", r#"{"error":"bad"}"#, false);
        assert!(
            !restricted.contains("Access-Control-Allow-Origin"),
            "CORS header must be absent when open_cors=false"
        );
    }

    // -------------------------------------------------------------------------
    // cwd path traversal tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cwd_contains_traversal() {
        assert!(cwd_contains_traversal("/home/user/../etc/passwd"));
        assert!(cwd_contains_traversal("/home/user/.."));
        assert!(cwd_contains_traversal("/../root"));
        assert!(!cwd_contains_traversal("/home/user/project"));
        assert!(!cwd_contains_traversal("/home/user/..hidden"));
        assert!(!cwd_contains_traversal("/home/user/a..b"));
    }

    #[tokio::test]
    async fn test_session_new_rejects_traversal_cwd() {
        let ch = make_channel();
        let body = AcpHttpChannel::do_session_new(
            &ch.state,
            &ch.base_config,
            &ch.config,
            Some(serde_json::json!(1)),
            Some(serde_json::json!({ "cwd": "/home/user/../etc" })),
        )
        .await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(
            v["error"]["code"], -32602,
            "traversal cwd must give -32602: {body}"
        );
        assert!(
            v["error"]["message"].as_str().unwrap().contains("'..'"),
            "error message must mention '..': {body}"
        );
        assert!(ch.state.lock().await.sessions.is_empty());
    }
}
