//! Tool-execution / approval / loop-guard / routing helpers extracted from
//! `agent::loop`.
//!
//! Phase 4 alternative helper extraction: mechanical move only.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use tokio::sync::RwLock;
use tracing::warn;

use crate::bus::{InboundMessage, OutboundMessage};
use crate::providers::LLMToolCall;
use crate::session::{Message, Role};
use crate::tools::approval::{ApprovalGate, ApprovalRequest, ApprovalResponse};
use crate::tools::{ToolCategory, ToolRegistry};

use super::loop_guard::{truncate_utf8, LoopGuard, LoopGuardAction, ToolCallSig};

pub(super) const INTERACTIVE_CLI_METADATA_KEY: &str = "interactive_cli";
pub(super) const TRUSTED_LOCAL_SESSION_METADATA_KEY: &str = "trusted_local_session";

pub(super) type ApprovalFuture = Pin<Box<dyn Future<Output = ApprovalResponse> + Send>>;
pub(super) type ApprovalHandler = Arc<dyn Fn(ApprovalRequest) -> ApprovalFuture + Send + Sync>;

pub(super) fn is_trusted_local_session(msg: &InboundMessage) -> bool {
    msg.channel == "cli"
        && msg
            .metadata
            .get(INTERACTIVE_CLI_METADATA_KEY)
            .is_some_and(|value| value == "true")
        && msg
            .metadata
            .get(TRUSTED_LOCAL_SESSION_METADATA_KEY)
            .is_some_and(|value| value == "true")
        && msg
            .metadata
            .get("is_batch")
            .is_none_or(|value| value != "true")
}

pub(super) async fn resolve_tool_approval(
    gate: &ApprovalGate,
    approval_handler: Option<&ApprovalHandler>,
    thread_identity: &crate::tools::thread_identity::ThreadIdentity,
    tool_name: &str,
    args: &serde_json::Value,
    channel: Option<&str>,
    chat_id: Option<&str>,
) -> Option<String> {
    // PR3: HardFloor takes precedence over the regular ApprovalGate.
    // A HardFloor hit forces interactive approval even when the gate
    // would normally let the call through (e.g. policy = AlwaysAllow,
    // or thread mode = AutoApprove — that branch is enforced in the
    // gateway handler, which sees `hard_floor_reason.is_some()` and
    // skips its AutoApprove bypass).
    //
    // The matcher is `Copy + cheap`; we instantiate one per call rather
    // than holding it on the agent because it's stateless and shaves
    // a constructor argument off `resolve_tool_approval`.
    let hard_floor = crate::tools::hard_floor::HardFloorMatcher::default().check(tool_name, args);

    if hard_floor.is_none() && !gate.requires_approval(tool_name) {
        return None;
    }

    if let Some(handler) = approval_handler {
        // Stamp the request with the process-level `(user_id, agent_id)`
        // pair so the broker can key its pending entry on something
        // stable across ACP session churn. `Unknown` leaves both fields
        // None and the handler falls back to `chat_id`-keyed routing —
        // identical to pre-PR1 behaviour.
        let (user_id, agent_id) = match thread_identity.as_known() {
            Some(k) => (Some(k.user.as_str()), Some(k.agent.as_str())),
            None => (None, None),
        };
        let mut request = gate
            .create_request(tool_name, args)
            .with_routing(channel, chat_id)
            .with_thread(user_id, agent_id);
        if let Some(rule) = hard_floor {
            request = request.with_hard_floor(rule.reason);
        }
        match handler(request).await {
            ApprovalResponse::Approved => None,
            ApprovalResponse::Denied(reason) => Some(format!(
                "Tool '{}' was denied by user approval. {}",
                tool_name, reason
            )),
            ApprovalResponse::TimedOut => {
                // PR3: HardFloor entries collapse TimedOut → Denied so
                // the LLM can't retry into a silent execution. The
                // regular path keeps surfacing TimedOut so it stays
                // distinguishable from an explicit user no.
                if let Some(rule) = hard_floor {
                    tracing::warn!(
                        target: "audit::approval",
                        tool = %tool_name,
                        rule = %rule.id,
                        decision = "denied_on_hard_floor_timeout",
                        "HardFloor approval timed out — forced Denied"
                    );
                    Some(format!(
                        "Tool '{}' was denied: HardFloor approval ({}) timed out.",
                        tool_name, rule.id
                    ))
                } else {
                    Some(format!(
                        "Tool '{}' approval timed out and was not executed.",
                        tool_name
                    ))
                }
            }
        }
    } else {
        let prompt = gate.format_approval_request(tool_name, args);
        Some(format!(
            "Tool '{}' requires user approval and was not executed. {}",
            tool_name, prompt
        ))
    }
}

/// Returns `true` if any tool in the batch may cause ordering-sensitive side effects
/// (filesystem writes, shell commands) and the batch should be executed sequentially
/// rather than in parallel.
///
/// Unknown tools (not found in the registry) default to `true` (fail-safe: serialize).
pub(super) async fn needs_sequential_execution(
    tools: &Arc<RwLock<ToolRegistry>>,
    tool_calls: &[LLMToolCall],
    lazy_tool_schema: bool,
) -> bool {
    let guard = tools.read().await;
    tool_calls.iter().any(|tc| {
        let name = resolve_tool_call_name(&guard, &tc.name, lazy_tool_schema).0;
        guard
            .get(&name)
            .map(|t| {
                matches!(
                    t.category(),
                    ToolCategory::FilesystemWrite | ToolCategory::Shell
                )
            })
            .unwrap_or(true) // unknown tool → serialize to be safe
    })
}

pub(super) fn resolve_tool_call_name(
    tools: &ToolRegistry,
    provider_name: &str,
    lazy_tool_schema: bool,
) -> (String, String) {
    if !lazy_tool_schema {
        let exposed = tools
            .exposed_name_for_tool(provider_name)
            .unwrap_or_else(|| provider_name.to_string());
        return (provider_name.to_string(), exposed);
    }

    if let Some(handle) = tools.resolve_exposed(provider_name) {
        return (handle.tool_name, handle.exposed_name);
    }

    (provider_name.to_string(), provider_name.to_string())
}

/// Check the loop guard for repeated tool-call patterns.
///
/// Returns `true` if the circuit breaker tripped and the caller should break.
pub(super) fn check_loop_guard(
    guard: &mut LoopGuard,
    tool_calls: &[LLMToolCall],
    session: &mut crate::session::Session,
) -> bool {
    let call_sigs: Vec<ToolCallSig<'_>> = tool_calls
        .iter()
        .map(|tc| ToolCallSig {
            name: tc.name.as_str(),
            arguments: tc.arguments.as_str(),
        })
        .collect();
    match guard.check(&call_sigs) {
        LoopGuardAction::Allow => false,
        LoopGuardAction::Warn {
            reason,
            suggested_delay_ms,
        } => {
            warn!(reason = %reason, "Loop guard warning");
            let delay_hint = suggested_delay_ms
                .map(|ms| format!(" (suggested delay: {}ms)", ms))
                .unwrap_or_default();
            session.add_message(Message::system(&format!(
                "[LoopGuard] {reason}{delay_hint}.",
            )));
            false
        }
        LoopGuardAction::Block { reason } => {
            warn!(reason = %reason, "Loop guard blocked tool call");
            session.add_message(Message::system(&format!("[LoopGuard] blocked: {reason}.",)));
            true
        }
        LoopGuardAction::CircuitBreak { total_repetitions } => {
            warn!(
                total_repetitions = total_repetitions,
                "Loop guard circuit breaker triggered"
            );
            session.add_message(Message::system(&format!(
                "[LoopGuard] circuit breaker tripped ({total_repetitions} total repetitions).",
            )));
            true
        }
    }
}

/// Record tool outcomes with the loop guard and check for repeated identical results.
///
/// Returns `true` if the circuit breaker tripped and the caller should break.
pub(super) fn check_loop_guard_outcomes(
    guard: &mut LoopGuard,
    tool_calls: &[LLMToolCall],
    results: &[(String, String)],
    session: &mut crate::session::Session,
) -> bool {
    // Build a lookup from tool call id -> (name, arguments).
    let call_map: std::collections::HashMap<&str, (&str, &str)> = tool_calls
        .iter()
        .map(|tc| (tc.id.as_str(), (tc.name.as_str(), tc.arguments.as_str())))
        .collect();

    for (id, result) in results {
        if let Some((name, args)) = call_map.get(id.as_str()) {
            let prefix = truncate_utf8(result, 1000);
            if let Some(action) = guard.record_outcome(name, args, prefix) {
                match action {
                    LoopGuardAction::Block { reason } => {
                        warn!(reason = %reason, "Loop guard blocked repeated outcome");
                        session.add_message(Message::system(&format!(
                            "[LoopGuard] blocked: {reason}.",
                        )));
                        return true;
                    }
                    LoopGuardAction::CircuitBreak { total_repetitions } => {
                        warn!(
                            total_repetitions = total_repetitions,
                            "Loop guard circuit breaker triggered via outcome"
                        );
                        session.add_message(Message::system(&format!(
                            "[LoopGuard] circuit breaker tripped ({total_repetitions} total repetitions).",
                        )));
                        return true;
                    }
                    LoopGuardAction::Warn {
                        reason,
                        suggested_delay_ms,
                    } => {
                        warn!(reason = %reason, "Loop guard outcome warning");
                        let delay_hint = suggested_delay_ms
                            .map(|ms| format!(" (suggested delay: {}ms)", ms))
                            .unwrap_or_default();
                        session.add_message(Message::system(&format!(
                            "[LoopGuard] {reason}{delay_hint}.",
                        )));
                    }
                    LoopGuardAction::Allow => {}
                }
            }
        }
    }
    false
}

/// Propagate channel-specific routing metadata (e.g. `telegram_thread_id`)
/// from an inbound message to an outbound message so that the response is
/// delivered to the correct forum topic / thread.
pub(super) fn propagate_routing_metadata(
    outbound: &mut OutboundMessage,
    inbound: &InboundMessage,
) {
    if let Some(tid) = inbound.metadata.get("telegram_thread_id") {
        outbound
            .metadata
            .insert("telegram_thread_id".to_string(), tid.clone());
    }
    if let Some(mid) = inbound.metadata.get("telegram_message_id") {
        outbound
            .metadata
            .insert("telegram_message_id".to_string(), mid.clone());
    }
    if let Some(mid) = inbound.metadata.get("discord_message_id") {
        outbound
            .metadata
            .insert("discord_message_id".to_string(), mid.clone());
    }
}

/// Sync trimmed tool-result messages from the resolved (preflight-mutated) buffer
/// back into the session so that the trims persist across iterations and saves.
/// Matches on `tool_call_id` since resolved messages include a system-prompt prefix.
pub(super) fn sync_trimmed_tool_results(
    session_messages: &mut [Message],
    resolved_messages: &[Message],
) {
    for resolved in resolved_messages {
        if resolved.role != Role::Tool {
            continue;
        }
        // Search in reverse so that when providers reuse sequential IDs
        // like `call_1` across turns, we match the most recent occurrence
        // rather than an earlier historical one.
        if let Some(session_msg) = session_messages
            .iter_mut()
            .rev()
            .find(|m| m.role == Role::Tool && m.tool_call_id == resolved.tool_call_id)
        {
            if session_msg.content != resolved.content {
                session_msg.content.clone_from(&resolved.content);
                session_msg
                    .content_parts
                    .clone_from(&resolved.content_parts);
            }
        }
    }
}
