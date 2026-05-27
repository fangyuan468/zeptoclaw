//! Agent state machine harness.
//!
//! Phase 4.1 of the state-machine refactor moves `process_message` out of the
//! `AgentLoop` orchestration shell into this dedicated runner. The body is
//! kept **verbatim**; only `self.<field>` was rewritten to `self.agent.<field>`
//! and `Self::<static_helper>` to `AgentLoop::<static_helper>`. No control
//! flow / error handling / parameter / ordering changes.
//!
//! Visibility is `pub(super)` — `AgentLoop` reaches `Harness` via a thin
//! delegator in `agent::loop`, and nothing outside `crate::agent::*` should
//! observe the indirection.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use futures::FutureExt;
use tracing::{debug, error, info, warn};

use crate::agent::context_monitor::{CompactionUrgency, ContextMonitor, PreflightAction};
use crate::agent::loop_guard::LoopGuard;
use crate::bus::InboundMessage;
use crate::cache::ResponseCache;
use crate::error::{Result, ZeptoError};
use crate::providers::ChatOptions;
use crate::session::{Message, Role};
use crate::tools::ToolContext;

use super::file_artifact::{
    build_file_artifact_payload, prepare_file_artifact_candidate, publish_file_artifact_event,
};
use super::format::{
    assistant_message_with_tool_calls, build_thinking_detail, build_tool_result_payload,
    prettify_tool_arguments,
};
use super::inbound::inbound_to_message;
use super::loop_events::{
    publish_tool_call_finished_event, publish_tool_call_started_event, ThinkingScope,
    ToolCallOutcome,
};
use super::observations::{ToolObservation, ToolObservationKind};
use super::r#loop::AgentLoop;
use super::tool_feedback::{ToolFeedback, ToolFeedbackPhase};
use super::tool_helpers::{
    check_loop_guard, check_loop_guard_outcomes, is_trusted_local_session,
    needs_sequential_execution, resolve_tool_approval, resolve_tool_call_name,
    sync_trimmed_tool_results,
};
use super::turn::{classify_final_content, classify_synthesis_trigger, TurnOutcome};

/// Per-turn runner that drives the agent state machine for a single inbound
/// message. Borrows `&AgentLoop` for the duration of one turn; does **not**
/// own any state of its own. Future phases (4.2 streaming, 5 anchored
/// summary) will add typed state fields here without touching `AgentLoop`.
pub(super) struct Harness<'a> {
    pub(super) agent: &'a AgentLoop,
}

impl<'a> Harness<'a> {
    pub(super) fn new(agent: &'a AgentLoop) -> Self {
        Self { agent }
    }

    pub(super) async fn process_message(&self, msg: &InboundMessage) -> Result<String> {
        // Acquire a per-session lock to serialize concurrent messages for the
        // same session key. Different sessions can still proceed concurrently.
        let session_lock = self.agent.session_lock_for(&msg.session_key).await;
        let _session_guard = session_lock.lock().await;

        // Reset per-run counters so limits apply to each process_message call
        // independently, not across the lifetime of the AgentLoop struct.
        self.agent.tool_call_limit.reset();
        self.agent.token_budget.reset();

        // Resolve the inbound message content first (inlines text attachments) so the
        // injection scanner sees the fully-expanded prompt, not just msg.content.
        let user_message = inbound_to_message(msg, None).await;
        let resolved_user_prompt = user_message.content.clone();

        // Tiered inbound injection scanning: block untrusted channels, warn others.
        // Runs before provider resolution so injected payloads are rejected immediately
        // without touching the session or LLM.
        // Scans the RESOLVED content (after text attachments are inlined) so injected
        // payloads in attachments never reach the model.
        if self.agent.config.safety.enabled && self.agent.config.safety.injection_check_enabled {
            let scan = crate::safety::sanitizer::check_injection(&resolved_user_prompt);
            if scan.was_modified {
                let channel = msg.channel.as_str();
                match channel {
                    "webhook" => {
                        warn!(
                            channel = channel,
                            sender = %msg.sender_id,
                            warnings = ?scan.warnings,
                            "Inbound injection BLOCKED from untrusted channel"
                        );
                        crate::audit::log_audit_event(
                            crate::audit::AuditCategory::InjectionAttempt,
                            crate::audit::AuditSeverity::Critical,
                            "inbound_injection_blocked",
                            &format!("Channel: {}, sender: {}", channel, msg.sender_id),
                            true,
                        );
                        return Err(ZeptoError::Tool(
                            "Message rejected: potential prompt injection detected".into(),
                        ));
                    }
                    _ => {
                        warn!(
                            channel = channel,
                            sender = %msg.sender_id,
                            warnings = ?scan.warnings,
                            "Inbound injection WARNING from allowlisted channel"
                        );
                        crate::audit::log_audit_event(
                            crate::audit::AuditCategory::InjectionAttempt,
                            crate::audit::AuditSeverity::Warning,
                            "inbound_injection_warned",
                            &format!("Channel: {}, sender: {}", channel, msg.sender_id),
                            false,
                        );
                    }
                }
            }
        }

        // Resolve the provider. Held until end of the function but not across
        // awaits that would block set_provider() writes for long.
        let provider = self
            .agent
            .resolve_provider_for_message(msg)
            .await
            .ok_or_else(|| ZeptoError::Provider("No provider configured".into()))?;
        let usage_metrics = {
            let metrics = self.agent.usage_metrics.read().await;
            metrics.clone()
        };
        let metrics_collector = Arc::clone(&self.agent.metrics_collector);

        // Get or create session
        let mut session = self.agent.session_manager.get_or_create(&msg.session_key).await?;

        // Add the user message BEFORE compaction so compaction sees the full context.
        session.add_message(user_message);

        // Apply three-tier context overflow recovery if needed
        if let Some(ref monitor) = self.agent.context_monitor {
            if let Some(urgency) = monitor.urgency(&session.messages) {
                if matches!(urgency, CompactionUrgency::Normal) {
                    // Skip memory flush in emergency/critical mode to recover faster.
                    self.agent.memory_flush(&session.messages).await;
                }

                let context_limit = self.agent.config.compaction.context_limit;
                let tool_result_cap = self.agent.config.agents.defaults.max_tool_result_bytes;
                let (recovered, tier) = crate::agent::compaction::try_recover_context_with_urgency(
                    session.messages,
                    context_limit,
                    urgency,
                    8,               // keep_recent for tier 1
                    tool_result_cap, // tool result budget for tier 2
                    self.agent.config.compaction.safety_margin,
                );
                if tier > 0 {
                    debug!(
                        tier = tier,
                        urgency = ?urgency,
                        "Context recovered via tier {} compaction", tier
                    );
                }
                session.messages = recovered;
            }
        }

        // Build messages with history and per-message memory override.
        // Pass an empty user_input string: the current user message is already
        // in session.messages above, so we must not add a duplicate plain-text
        // entry here.
        let memory_override = self.agent.build_memory_override(&resolved_user_prompt).await;
        let mut messages = self
            .agent
            .build_resolved_messages(msg, &session, memory_override.as_deref())
            .await;

        // Get tool definitions (short-lived read lock)
        let tool_definitions = {
            let tools = self.agent.tools.read().await;
            tools.definitions_for_mode(
                self.agent.config.agents.defaults.lazy_tool_schema,
                self.agent.config.agents.defaults.compact_tools,
            )
        };

        // Pre-flight context guard: trim oversized tool results and check budget
        if let Some(ref monitor) = self.agent.context_monitor {
            match monitor.preflight_check(&mut messages, &tool_definitions) {
                PreflightAction::Ok => {}
                PreflightAction::Trimmed => {
                    debug!("Pre-flight guard trimmed oversized tool results");
                    sync_trimmed_tool_results(&mut session.messages, &messages);
                }
                PreflightAction::NeedsCompaction => {
                    warn!("Pre-flight guard: context too large, triggering emergency compaction");
                    let context_limit = self.agent.config.compaction.context_limit;
                    let tool_result_cap = self.agent.config.agents.defaults.max_tool_result_bytes;
                    let (recovered, _tier) =
                        crate::agent::compaction::try_recover_context_with_urgency(
                            session.messages,
                            context_limit,
                            CompactionUrgency::Emergency,
                            5,
                            tool_result_cap,
                            self.agent.config.compaction.safety_margin,
                        );
                    session.messages = recovered;
                    messages = self
                        .agent
                        .build_resolved_messages(msg, &session, memory_override.as_deref())
                        .await;
                }
            }
        }

        // Build chat options
        let options = ChatOptions::new()
            .with_max_tokens(self.agent.config.agents.defaults.max_tokens)
            .with_temperature(self.agent.config.agents.defaults.temperature);

        let model_string = self.agent.resolve_model_for_message(msg);
        let model = Some(model_string.as_str());

        // Check token budget before first LLM call
        if self.agent.token_budget.is_exceeded() {
            return Err(ZeptoError::Provider(format!(
                "Token budget exceeded: {}",
                self.agent.token_budget.summary()
            )));
        }

        // Build cache key from (model, system_prompt, user_prompt) for the
        // initial LLM call only. Tool follow-up calls are never cached.
        let cache_key = self.agent.cache.as_ref().map(|_| {
            let system_prompt = messages
                .first()
                .filter(|m| m.role == Role::System)
                .map(|m| m.content.as_str())
                .unwrap_or("");
            ResponseCache::cache_key(
                self.agent.config.agents.defaults.model.as_str(),
                system_prompt,
                &resolved_user_prompt,
            )
        });

        // Check response cache before calling the provider.
        // The MutexGuard must be dropped before any .await to remain Send.
        let cached_hit = if let (Some(ref cache_mutex), Some(ref key)) = (&self.agent.cache, &cache_key) {
            cache_mutex.lock().ok().and_then(|mut c| c.get(key))
        } else {
            None
        };
        if let Some(cached_response) = cached_hit {
            debug!("Cache hit for initial prompt");
            // User message was already added to session before build_messages.
            session.add_message(Message::assistant(&cached_response));
            self.agent.session_manager.save(&session).await?;
            return Ok(cached_response);
        }

        // Send thinking feedback
        if let Some(tx) = self.agent.tool_feedback_tx.read().await.as_ref() {
            let _ = tx.send(ToolFeedback {
                tool_name: String::new(),
                phase: ToolFeedbackPhase::Thinking,
                args_json: None,
            });
        }
        let thinking_scope =
            ThinkingScope::start(Arc::clone(&self.agent.bus), &msg.channel, &msg.chat_id).await;

        // Call LLM with overflow retry -- provider lock is NOT held during this await
        let mut response = {
            let max_retries = self.agent.config.compaction.overflow_retries;
            let mut last_messages = messages;
            let mut last_tool_defs = tool_definitions;
            let mut result = provider
                .chat(
                    last_messages.clone(),
                    last_tool_defs.clone(),
                    model,
                    options.clone(),
                )
                .await;

            let mut attempt = 0u32;
            while let Err(ref e) = result {
                if !AgentLoop::is_context_overflow(e) || attempt >= max_retries {
                    break;
                }
                if self.agent.context_monitor.is_none() {
                    break; // compaction disabled — don't rewrite history
                }
                warn!(
                    attempt = attempt + 1,
                    max = max_retries,
                    "Context overflow, compacting and retrying"
                );
                let urgency = AgentLoop::overflow_retry_urgency(attempt);
                let ctx_limit = self.agent.config.compaction.context_limit;
                let cap = self.agent.config.agents.defaults.max_tool_result_bytes;
                let (recovered, _) = crate::agent::compaction::try_recover_context_with_urgency(
                    session.messages,
                    ctx_limit,
                    urgency,
                    8,
                    cap,
                    self.agent.config.compaction.safety_margin,
                );
                session.messages = recovered;
                last_messages = self
                    .agent
                    .build_resolved_messages(msg, &session, memory_override.as_deref())
                    .await;
                last_tool_defs = {
                    let tools = self.agent.tools.read().await;
                    tools.definitions_for_mode(
                        self.agent.config.agents.defaults.lazy_tool_schema,
                        self.agent.config.agents.defaults.compact_tools,
                    )
                };
                result = provider
                    .chat(
                        last_messages.clone(),
                        last_tool_defs.clone(),
                        model,
                        options.clone(),
                    )
                    .await;
                attempt += 1;
            }
            result?
        };

        // Send thinking done feedback
        if let Some(tx) = self.agent.tool_feedback_tx.read().await.as_ref() {
            let _ = tx.send(ToolFeedback {
                tool_name: String::new(),
                phase: ToolFeedbackPhase::ThinkingDone,
                args_json: None,
            });
        }
        let thinking_detail = build_thinking_detail(&response);
        thinking_scope.finish(thinking_detail.as_deref()).await;

        if let (Some(metrics), Some(usage)) = (usage_metrics.as_ref(), response.usage.as_ref()) {
            metrics.record_tokens_with_cache(
                usage.prompt_tokens as u64,
                usage.completion_tokens as u64,
                usage.cached_tokens as u64,
                usage.cache_creation_tokens as u64,
            );
        }
        if let Some(usage) = response.usage.as_ref() {
            metrics_collector.record_tokens_with_cache(
                usage.prompt_tokens as u64,
                usage.completion_tokens as u64,
                usage.cached_tokens as u64,
                usage.cache_creation_tokens as u64,
            );
            self.agent
                .token_budget
                .record(usage.prompt_tokens as u64, usage.completion_tokens as u64);
        }

        // Cache the response if it has no tool calls (pure text reply).
        // Responses with tool calls depend on tool execution and are not cacheable.
        if !response.has_tool_calls() {
            if let (Some(ref cache_mutex), Some(key)) = (&self.agent.cache, cache_key) {
                let token_count = response
                    .usage
                    .as_ref()
                    .map(|u| u.completion_tokens)
                    .unwrap_or(0);
                if let Ok(mut cache) = cache_mutex.lock() {
                    cache.put(key, response.content.clone(), token_count);
                    debug!("Cached initial LLM response");
                }
            }
        }

        // User message was already added to session before build_messages above.

        // Tool loop
        let max_iterations = self.agent.config.agents.defaults.max_tool_iterations;
        let mut iteration = 0;
        let mut chain_tracker = crate::safety::chain_alert::ChainTracker::new();
        let mut loop_guard = if self.agent.config.agents.defaults.loop_guard.enabled {
            Some(LoopGuard::new(
                self.agent.config.agents.defaults.loop_guard.clone(),
            ))
        } else {
            None
        };

        while response.has_tool_calls() && iteration < max_iterations {
            iteration += 1;
            debug!("Tool iteration {} of {}", iteration, max_iterations);

            // Enforce tool call limit BEFORE recording metrics or adding
            // the assistant message to the session. This ensures max_tool_calls=0
            // never writes an orphaned tool-call message, and partial truncation
            // keeps the transcript consistent (only executed calls are recorded).
            if self.agent.tool_call_limit.is_exceeded() {
                info!(
                    count = self.agent.tool_call_limit.count(),
                    limit = ?self.agent.tool_call_limit.limit(),
                    "Tool call limit already reached, skipping tool execution"
                );
                break;
            }
            // Truncate batch to remaining budget so we never overshoot.
            if let Some(remaining) = self.agent.tool_call_limit.remaining() {
                let allowed = remaining as usize;
                if allowed < response.tool_calls.len() {
                    info!(
                        batch_size = response.tool_calls.len(),
                        remaining = allowed,
                        "Truncating tool call batch to remaining budget"
                    );
                    response.tool_calls.truncate(allowed);
                }
            }

            // Record metrics AFTER truncation so counts reflect actual execution.
            if let Some(metrics) = usage_metrics.as_ref() {
                metrics.record_tool_calls(response.tool_calls.len() as u64);
            }

            // Add assistant message with tool calls (post-truncation).
            // Some OpenAI-compatible providers also echo provider-specific
            // tool-call markup in `content`; the structured tool_calls are
            // the source of truth, so keep that markup out of future prompts.
            session.add_message(assistant_message_with_tool_calls(&response.tool_calls));

            // Execute tool calls in parallel
            let workspace = self.agent.config.workspace_path();
            let workspace_str = workspace.to_string_lossy();
            let tool_ctx = ToolContext::new()
                .with_channel(&msg.channel, &msg.chat_id)
                .with_workspace(&workspace_str)
                .with_batch(msg.metadata.get("is_batch").is_some_and(|v| v == "true"));

            let approval_gate = Arc::clone(&self.agent.approval_gate);
            let approval_handler = self.agent.approval_handler.read().await.clone();
            let safety_layer = self.agent.safety_layer.clone();
            let taint_engine = self.agent.taint.clone();
            let hook_engine = Arc::new(
                crate::hooks::HookEngine::new(self.agent.config.hooks.clone())
                    .with_bus(Arc::clone(&self.agent.bus)),
            );

            // Compute dynamic tool result budget based on remaining context space
            let current_tokens = ContextMonitor::estimate_tokens_with_margin(
                &session.messages,
                self.agent.config.compaction.safety_margin,
            );
            let context_limit = self.agent.config.compaction.context_limit;
            let max_result_bytes = self.agent.config.agents.defaults.max_tool_result_bytes;
            let result_budget = crate::utils::sanitize::compute_tool_result_budget_with_share(
                context_limit,
                current_tokens,
                response.tool_calls.len(),
                max_result_bytes,
                self.agent.config.compaction.single_tool_result_share,
            );

            let tool_feedback_tx = self.agent.tool_feedback_tx.clone();
            #[cfg(feature = "panel")]
            let event_bus_clone = self.agent.event_bus.clone();
            let is_dry_run = self.agent.dry_run.load(Ordering::SeqCst);
            let current_agent_mode = self.agent.agent_mode;
            let trusted_local_session = is_trusted_local_session(msg);
            let lazy_tool_schema = self.agent.config.agents.defaults.lazy_tool_schema;
            let any_approval_gated_tool = if approval_handler.is_some() {
                let guard = self.agent.tools.read().await;
                response.tool_calls.iter().any(|tool_call| {
                    let name = resolve_tool_call_name(&guard, &tool_call.name, lazy_tool_schema).0;
                    approval_gate.requires_approval(&name)
                })
            } else {
                false
            };

            let run_sequential = (!trusted_local_session
                && approval_handler.is_some()
                && any_approval_gated_tool)
                || needs_sequential_execution(&self.agent.tools, &response.tool_calls, lazy_tool_schema)
                    .await;
            let tool_timeout_secs = if self.agent.config.agents.defaults.tool_timeout_secs > 0 {
                self.agent.config.agents.defaults.tool_timeout_secs
            } else {
                self.agent.config.agents.defaults.agent_timeout_secs
            };
            let tool_timeout = std::time::Duration::from_secs(tool_timeout_secs.max(1));

            // Clone inbound metadata for routing propagation in tool `for_user` messages.
            let inbound_metadata = msg.metadata.clone();

            let tool_futures: Vec<_> = response
                .tool_calls
                .iter()
                .map(|tool_call| {
                    let tools = Arc::clone(&self.agent.tools);
                    let ctx = tool_ctx.clone();
                    let name = tool_call.name.clone();
                    let id = tool_call.id.clone();
                    let raw_args = tool_call.arguments.clone();
                    let usage_metrics = usage_metrics.clone();
                    let metrics_collector = Arc::clone(&metrics_collector);
                    let gate = Arc::clone(&approval_gate);
                    let approval_handler = approval_handler.clone();
                    let thread_identity = self.agent.thread_identity();
                    let hooks = Arc::clone(&hook_engine);
                    let safety = safety_layer.clone();
                    let taint = taint_engine.clone();
                    let budget = result_budget;
                    let tool_feedback_tx = tool_feedback_tx.clone();
                    #[cfg(feature = "panel")]
                    let event_bus = event_bus_clone.clone();
                    let dry_run = is_dry_run;
                    let agent_mode = current_agent_mode;
                    let bus_for_tools = Arc::clone(&self.agent.bus);
                    let inbound_meta = inbound_metadata.clone();

                    async move {
                        let args: serde_json::Value = match serde_json::from_str(&raw_args) {
                            Ok(v) => v,
                            Err(e) => {
                                tracing::warn!(tool = %name, error = %e, "Invalid JSON in tool arguments");
                                serde_json::json!({"_parse_error": format!("Invalid arguments JSON: {}", e)})
                            }
                        };
                        let (name, exposed_name) = {
                            let tools_guard = tools.read().await;
                            resolve_tool_call_name(&tools_guard, &name, lazy_tool_schema)
                        };

                        if lazy_tool_schema {
                            let validation_error = {
                                let tools_guard = tools.read().await;
                                tools_guard.validate_tool_args_lazy(&name, &args, &exposed_name)
                            };
                            if let Some(output) = validation_error {
                                if let Some(metrics) = usage_metrics.as_ref() {
                                    metrics.record_tool_validation_failure();
                                }
                                // Lazy-schema validation rejected the args
                                // before the tool ran; treat as a soft
                                // error so the model can rewrite the call.
                                return ToolObservation::pre_execution(
                                    id,
                                    name,
                                    output.for_llm,
                                    ToolObservationKind::SoftError,
                                );
                            }
                            if name == "get_tool_schema" {
                                if let Some(metrics) = usage_metrics.as_ref() {
                                    metrics.record_tool_schema_retrieval();
                                }
                            }
                        }

                        // Check hooks before executing
                        let channel_name = ctx.channel.as_deref().unwrap_or("cli");
                        let chat_id = ctx.chat_id.as_deref().unwrap_or(channel_name);
                        if let crate::hooks::HookResult::Block(msg) =
                            hooks.before_tool(&name, &args, channel_name, chat_id)
                        {
                            return ToolObservation::pre_execution(
                                id,
                                name.clone(),
                                format!("Tool '{}' blocked by hook: {}", name, msg),
                                ToolObservationKind::HardError,
                            );
                        }

                        // Agent mode enforcement (before approval gate).
                        // RequiresApproval: blocks the tool unless ApprovalGate is
                        // already configured to gate this tool name. In practice, this
                        // means Assistant mode blocks Shell/Hardware/Destructive tools
                        // unless the operator has explicitly listed them in
                        // `approval.require_approval_for`. This is "fail-closed" by design.
                        {
                            let mode_policy = crate::security::ModePolicy::new(agent_mode);
                            let tools_guard = tools.read().await;
                            if let Some(tool) = tools_guard.get(&name) {
                                let tool_category = tool.category();
                                match mode_policy.check(tool_category) {
                                    crate::security::CategoryPermission::Blocked => {
                                        info!(tool = %name, mode = %agent_mode, category = ?tool_category, "Tool blocked by agent mode");
                                        return ToolObservation::pre_execution(
                                            id,
                                            name.clone(),
                                            format!(
                                                "Tool '{}' is blocked in {} mode (category: {})",
                                                name, agent_mode, tool_category
                                            ),
                                            ToolObservationKind::HardError,
                                        );
                                    }
                                    crate::security::CategoryPermission::RequiresApproval => {
                                        if trusted_local_session {
                                            info!(tool = %name, mode = %agent_mode, category = ?tool_category, "Trusted local session bypassed approval-gated tool");
                                        } else if !gate.requires_approval(&name) {
                                            info!(tool = %name, mode = %agent_mode, category = ?tool_category, "Tool requires approval per agent mode");
                                            return ToolObservation::pre_execution(
                                                id,
                                                name.clone(),
                                                format!(
                                                    "Tool '{}' requires approval in {} mode (category: {}). Not executed.",
                                                    name, agent_mode, tool_category
                                                ),
                                                ToolObservationKind::ApprovalRequired,
                                            );
                                        }
                                        // Fall through to approval gate — it will prompt for approval
                                    }
                                    crate::security::CategoryPermission::Allowed => {}
                                }
                            }
                        }

                        // Check approval gate before executing
                        if !trusted_local_session {
                            if let Some(message) = resolve_tool_approval(
                                &gate,
                                approval_handler.as_ref(),
                                thread_identity.as_ref(),
                                &name,
                                &args,
                                ctx.channel.as_deref(),
                                ctx.chat_id.as_deref(),
                            )
                            .await
                            {
                                info!(tool = %name, "Tool requires approval, blocking execution");
                                return ToolObservation::pre_execution(
                                    id,
                                    name,
                                    message,
                                    ToolObservationKind::ApprovalRequired,
                                );
                            }
                        }

                        // Dry-run mode: describe what would happen without executing
                        if dry_run {
                            return ToolObservation::pre_execution(
                                id,
                                name.clone(),
                                AgentLoop::dry_run_result(&name, &args, &raw_args, budget),
                                ToolObservationKind::Success,
                            );
                        }
                        let file_artifact_candidate =
                            prepare_file_artifact_candidate(&name, &args, &ctx);
                        let pretty_args = prettify_tool_arguments(&raw_args);

                        // Send tool starting feedback
                        if let Some(tx) = tool_feedback_tx.read().await.as_ref() {
                            let _ = tx.send(ToolFeedback {
                                tool_name: name.clone(),
                                phase: ToolFeedbackPhase::Starting,
                                args_json: Some(raw_args.clone()),
                            });
                        }
                        publish_tool_call_started_event(
                            &bus_for_tools,
                            &ctx,
                            &id,
                            &name,
                            &pretty_args,
                        )
                        .await;
                        #[cfg(feature = "panel")]
                        if let Some(bus) = &event_bus {
                            bus.send(crate::api::events::PanelEvent::ToolStarted {
                                tool: name.clone(),
                            });
                        }
                        let tool_start = std::time::Instant::now();
                        let execution = std::panic::AssertUnwindSafe(async {
                            let tools_guard = tools.read().await;
                            crate::kernel::execute_tool(
                                &tools_guard,
                                &name,
                                args,
                                &ctx,
                                safety.as_ref().map(|s| s.as_ref()),
                                &metrics_collector,
                                taint.as_ref().map(|t| t.as_ref()),
                            )
                            .await
                        })
                        .catch_unwind();
                        // Phase 3 state-machine refactor: classify execution
                        // outcome into a `ToolObservationKind`.
                        //   - Success         : tool ran, `is_error == false`.
                        //   - SoftError       : tool ran, `is_error == true`
                        //                       (e.g. web_fetch 404 — model
                        //                       can recover from the in-band
                        //                       message).
                        //   - HardError       : `execute_tool` returned Err,
                        //                       panic, or timeout.
                        // `success` is kept as a derived bool so the rest of
                        // the closure's hook / event / metrics branches
                        // remain a 2-state decision; `kind` is what the
                        // outer loop reads.
                        let (result, success, tool_output, kind) =
                            match tokio::time::timeout(tool_timeout, execution).await {
                                Ok(Ok(Ok(output))) => {
                                    let is_err = output.is_error;
                                    let kind = if is_err {
                                        ToolObservationKind::SoftError
                                    } else {
                                        ToolObservationKind::Success
                                    };
                                    let for_llm = output.for_llm.clone();
                                    (for_llm, !is_err, Some(output), kind)
                                }
                                Ok(Ok(Err(e))) => (
                                    format!("Error: {}", e),
                                    false,
                                    None,
                                    ToolObservationKind::HardError,
                                ),
                                Ok(Err(_panic)) => {
                                    error!(tool = %name, "Tool panicked during execution");
                                    (
                                        format!(
                                            "Error: Tool '{}' panicked during execution",
                                            name
                                        ),
                                        false,
                                        None,
                                        ToolObservationKind::HardError,
                                    )
                                }
                                Err(_) => {
                                    error!(tool = %name, timeout_secs = tool_timeout.as_secs(), "Tool execution timed out");
                                    (
                                        format!(
                                            "Error: Tool '{}' timed out after {}s",
                                            name,
                                            tool_timeout.as_secs()
                                        ),
                                        false,
                                        None,
                                        ToolObservationKind::HardError,
                                    )
                                }
                            };

                        let pause = tool_output.as_ref().is_some_and(|o| o.pause_for_input);
                        let elapsed = tool_start.elapsed();
                        let latency_ms = elapsed.as_millis() as u64;
                        let (sanitized_result, result_preview) =
                            build_tool_result_payload(&result, budget);
                        // Send to user if tool opted in
                        if let Some(ref output) = tool_output {
                            if let Some(ref user_msg) = output.for_user {
                                let mut outbound = crate::bus::OutboundMessage::new(
                                    ctx.channel.as_deref().unwrap_or(""),
                                    ctx.chat_id.as_deref().unwrap_or(""),
                                    user_msg,
                                );
                                // Propagate routing metadata (e.g. telegram_thread_id, telegram_message_id)
                                if let Some(tid) = inbound_meta.get("telegram_thread_id") {
                                    outbound
                                        .metadata
                                        .insert("telegram_thread_id".to_string(), tid.clone());
                                }
                                if let Some(mid) = inbound_meta.get("telegram_message_id") {
                                    outbound
                                        .metadata
                                        .insert("telegram_message_id".to_string(), mid.clone());
                                }
                                // Keep typing indicator alive — agent is still working
                                outbound
                                    .metadata
                                    .insert("keep_typing".to_string(), "true".to_string());
                                let _ = bus_for_tools.publish_outbound(outbound).await;
                            }
                        }
                        if success {
                            if let Some(candidate) = file_artifact_candidate.as_ref() {
                                if let Some(payload) = build_file_artifact_payload(candidate, &ctx) {
                                    publish_file_artifact_event(&bus_for_tools, &ctx, &payload).await;
                                }
                            }
                            debug!(tool = %name, latency_ms = latency_ms, "Tool executed successfully");
                            hooks.after_tool(&name, &result, elapsed, channel_name, chat_id);
                            if let Some(tx) = tool_feedback_tx.read().await.as_ref() {
                                let _ = tx.send(ToolFeedback {
                                    tool_name: name.clone(),
                                    phase: ToolFeedbackPhase::Done { elapsed_ms: latency_ms },
                                    args_json: Some(raw_args.clone()),
                                });
                            }
                            publish_tool_call_finished_event(
                                &bus_for_tools,
                                &ctx,
                                &id,
                                &name,
                                ToolCallOutcome::Done,
                                latency_ms,
                                &pretty_args,
                                &sanitized_result,
                                result_preview.as_deref(),
                            )
                            .await;
                            #[cfg(feature = "panel")]
                            if let Some(bus) = &event_bus {
                                bus.send(crate::api::events::PanelEvent::ToolDone {
                                    tool: name.clone(),
                                    duration_ms: latency_ms,
                                });
                            }
                        } else {
                            error!(tool = %name, latency_ms = latency_ms, error = %result, "Tool execution failed");
                            hooks.on_error(&name, &result, channel_name, chat_id);
                            if let Some(metrics) = usage_metrics.as_ref() {
                                metrics.record_error();
                            }
                            if let Some(tx) = tool_feedback_tx.read().await.as_ref() {
                                let _ = tx.send(ToolFeedback {
                                    tool_name: name.clone(),
                                    phase: ToolFeedbackPhase::Failed {
                                        elapsed_ms: latency_ms,
                                        error: result.clone(),
                                    },
                                    args_json: Some(raw_args.clone()),
                                });
                            }
                            publish_tool_call_finished_event(
                                &bus_for_tools,
                                &ctx,
                                &id,
                                &name,
                                ToolCallOutcome::Failed { raw_error: &result },
                                latency_ms,
                                &pretty_args,
                                &sanitized_result,
                                result_preview.as_deref(),
                            )
                            .await;
                            #[cfg(feature = "panel")]
                            if let Some(bus) = &event_bus {
                                bus.send(crate::api::events::PanelEvent::ToolFailed {
                                    tool: name.clone(),
                                    error: result.clone(),
                                });
                            }
                        }

                        ToolObservation::new(
                            id,
                            name,
                            sanitized_result,
                            kind,
                            latency_ms,
                            pause,
                        )
                    }
                })
                .collect();

            let results = if run_sequential {
                let mut out = Vec::with_capacity(tool_futures.len());
                for fut in tool_futures {
                    out.push(fut.await);
                }
                out
            } else {
                futures::future::join_all(tool_futures).await
            };

            // Record tool names for chain alerting
            let tool_names: Vec<String> = response
                .tool_calls
                .iter()
                .map(|tc| tc.name.clone())
                .collect();
            chain_tracker.record(&tool_names);

            let results: Vec<ToolObservation> = results;
            let should_pause = results.iter().any(|obs| obs.pause_for_input);
            for obs in &results {
                session.add_message(Message::tool_result(&obs.call_id, &obs.content));
            }

            // In-loop compaction: check if tool results pushed context over threshold
            if let Some(ref monitor) = self.agent.context_monitor {
                if let Some(urgency) = monitor.urgency(&session.messages) {
                    debug!(urgency = ?urgency, "In-loop compaction triggered after tool results");
                    let ctx_limit = self.agent.config.compaction.context_limit;
                    let cap = self.agent.config.agents.defaults.max_tool_result_bytes;
                    let (recovered, tier) =
                        crate::agent::compaction::try_recover_context_with_urgency(
                            session.messages,
                            ctx_limit,
                            urgency,
                            8,
                            cap,
                            self.agent.config.compaction.safety_margin,
                        );
                    if tier > 0 {
                        debug!(tier = tier, "In-loop context recovered via tier {}", tier);
                    }
                    session.messages = recovered;
                }
            }

            if should_pause {
                break;
            }

            // Increment tool call counter after execution.
            self.agent
                .tool_call_limit
                .increment(response.tool_calls.len() as u32);
            // If the limit is now hit, make one final LLM call WITHOUT tools
            // so the model can synthesize the tool results into a proper answer
            // instead of returning the stale tool-call stub content.
            if self.agent.tool_call_limit.is_exceeded() {
                info!(
                    count = self.agent.tool_call_limit.count(),
                    limit = ?self.agent.tool_call_limit.limit(),
                    "Tool call limit reached, making final synthesis call"
                );
                // Respect token budget — skip the synthesis call if already over.
                if self.agent.token_budget.is_exceeded() {
                    info!(budget = %self.agent.token_budget.summary(), "Token budget also exceeded, skipping synthesis call");
                    response.content =
                        "Tool call limit reached. Token budget exceeded.".to_string();
                    break;
                }
                let mut messages = self
                    .agent
                    .build_resolved_messages(msg, &session, memory_override.as_deref())
                    .await;
                // Pre-flight guard for synthesis call
                if let Some(ref monitor) = self.agent.context_monitor {
                    if let PreflightAction::NeedsCompaction =
                        monitor.preflight_check(&mut messages, &[])
                    {
                        let ctx_limit = self.agent.config.compaction.context_limit;
                        let cap = self.agent.config.agents.defaults.max_tool_result_bytes;
                        let (recovered, _) =
                            crate::agent::compaction::try_recover_context_with_urgency(
                                session.messages,
                                ctx_limit,
                                CompactionUrgency::Emergency,
                                5,
                                cap,
                                self.agent.config.compaction.safety_margin,
                            );
                        session.messages = recovered;
                        messages = self
                            .agent
                            .build_resolved_messages(msg, &session, memory_override.as_deref())
                            .await;
                    }
                }
                response = {
                    let max_retries = self.agent.config.compaction.overflow_retries;
                    let mut last_messages = messages;
                    let mut result = provider
                        .chat(last_messages.clone(), vec![], model, options.clone())
                        .await;
                    let mut attempt = 0u32;
                    while let Err(ref e) = result {
                        if !AgentLoop::is_context_overflow(e) || attempt >= max_retries {
                            break;
                        }
                        if self.agent.context_monitor.is_none() {
                            break; // compaction disabled
                        }
                        warn!(
                            attempt = attempt + 1,
                            max = max_retries,
                            "Context overflow in synthesis call, compacting and retrying"
                        );
                        let urgency = AgentLoop::overflow_retry_urgency(attempt);
                        let ctx_limit = self.agent.config.compaction.context_limit;
                        let cap = self.agent.config.agents.defaults.max_tool_result_bytes;
                        let (recovered, _) =
                            crate::agent::compaction::try_recover_context_with_urgency(
                                session.messages,
                                ctx_limit,
                                urgency,
                                5,
                                cap,
                                self.agent.config.compaction.safety_margin,
                            );
                        session.messages = recovered;
                        last_messages = self
                            .agent
                            .build_resolved_messages(msg, &session, memory_override.as_deref())
                            .await;
                        result = provider
                            .chat(last_messages.clone(), vec![], model, options.clone())
                            .await;
                        attempt += 1;
                    }
                    result?
                };
                if let (Some(metrics), Some(usage)) =
                    (usage_metrics.as_ref(), response.usage.as_ref())
                {
                    metrics.record_tokens_with_cache(
                        usage.prompt_tokens as u64,
                        usage.completion_tokens as u64,
                        usage.cached_tokens as u64,
                        usage.cache_creation_tokens as u64,
                    );
                }
                if let Some(usage) = response.usage.as_ref() {
                    metrics_collector.record_tokens_with_cache(
                        usage.prompt_tokens as u64,
                        usage.completion_tokens as u64,
                        usage.cached_tokens as u64,
                        usage.cache_creation_tokens as u64,
                    );
                    self.agent
                        .token_budget
                        .record(usage.prompt_tokens as u64, usage.completion_tokens as u64);
                }
                break;
            }

            if let Some(guard) = loop_guard.as_mut() {
                if check_loop_guard(guard, &response.tool_calls, &mut session) {
                    response.content =
                        "Stopped tool loop due to repeated tool-call pattern.".to_string();
                    break;
                }

                // Record outcomes for outcome-aware blocking.
                let results_for_guard: Vec<(String, String)> = results
                    .iter()
                    .map(|obs| (obs.call_id.clone(), obs.content.clone()))
                    .collect();
                if check_loop_guard_outcomes(
                    guard,
                    &response.tool_calls,
                    &results_for_guard,
                    &mut session,
                ) {
                    response.content =
                        "Stopped tool loop due to repeated identical outcomes.".to_string();
                    break;
                }
            }

            // Get fresh tool definitions for the next LLM call
            let tool_definitions = {
                let tools = self.agent.tools.read().await;
                tools.definitions_for_mode(
                    self.agent.config.agents.defaults.lazy_tool_schema,
                    self.agent.config.agents.defaults.compact_tools,
                )
            };

            // Check token budget before next LLM call
            if self.agent.token_budget.is_exceeded() {
                info!(budget = %self.agent.token_budget.summary(), "Token budget exceeded during tool loop");
                break;
            }

            // Call LLM again with tool results -- provider lock NOT held
            let mut messages = self
                .agent
                .build_resolved_messages(msg, &session, memory_override.as_deref())
                .await;

            // Pre-flight context guard (tool loop)
            if let Some(ref monitor) = self.agent.context_monitor {
                match monitor.preflight_check(&mut messages, &tool_definitions) {
                    PreflightAction::Ok => {}
                    PreflightAction::Trimmed => {
                        debug!("Pre-flight guard trimmed tool results (tool loop)");
                        sync_trimmed_tool_results(&mut session.messages, &messages);
                    }
                    PreflightAction::NeedsCompaction => {
                        warn!("Pre-flight: context too large in tool loop, emergency compaction");
                        let ctx_limit = self.agent.config.compaction.context_limit;
                        let cap = self.agent.config.agents.defaults.max_tool_result_bytes;
                        let (recovered, _) =
                            crate::agent::compaction::try_recover_context_with_urgency(
                                session.messages,
                                ctx_limit,
                                CompactionUrgency::Emergency,
                                5,
                                cap,
                                self.agent.config.compaction.safety_margin,
                            );
                        session.messages = recovered;
                        messages = self
                            .agent
                            .build_resolved_messages(msg, &session, memory_override.as_deref())
                            .await;
                    }
                }
            }

            // Send thinking feedback for tool-loop LLM call
            if let Some(tx) = self.agent.tool_feedback_tx.read().await.as_ref() {
                let _ = tx.send(ToolFeedback {
                    tool_name: String::new(),
                    phase: ToolFeedbackPhase::Thinking,
                    args_json: None,
                });
            }
            let thinking_scope =
                ThinkingScope::start(Arc::clone(&self.agent.bus), &msg.channel, &msg.chat_id).await;

            response = {
                let max_retries = self.agent.config.compaction.overflow_retries;
                let mut last_messages = messages;
                let mut last_tool_defs = tool_definitions;
                let mut result = provider
                    .chat(
                        last_messages.clone(),
                        last_tool_defs.clone(),
                        model,
                        options.clone(),
                    )
                    .await;
                let mut attempt = 0u32;
                while let Err(ref e) = result {
                    if !AgentLoop::is_context_overflow(e) || attempt >= max_retries {
                        break;
                    }
                    if self.agent.context_monitor.is_none() {
                        break; // compaction disabled
                    }
                    warn!(
                        attempt = attempt + 1,
                        max = max_retries,
                        "Context overflow in tool loop, compacting and retrying"
                    );
                    let urgency = AgentLoop::overflow_retry_urgency(attempt);
                    let ctx_limit = self.agent.config.compaction.context_limit;
                    let cap = self.agent.config.agents.defaults.max_tool_result_bytes;
                    let (recovered, _) = crate::agent::compaction::try_recover_context_with_urgency(
                        session.messages,
                        ctx_limit,
                        urgency,
                        8,
                        cap,
                        self.agent.config.compaction.safety_margin,
                    );
                    session.messages = recovered;
                    last_messages = self
                        .agent
                        .build_resolved_messages(msg, &session, memory_override.as_deref())
                        .await;
                    last_tool_defs = {
                        let tools = self.agent.tools.read().await;
                        tools.definitions_for_mode(
                            self.agent.config.agents.defaults.lazy_tool_schema,
                            self.agent.config.agents.defaults.compact_tools,
                        )
                    };
                    result = provider
                        .chat(
                            last_messages.clone(),
                            last_tool_defs.clone(),
                            model,
                            options.clone(),
                        )
                        .await;
                    attempt += 1;
                }
                result?
            };

            // Send thinking done feedback
            if let Some(tx) = self.agent.tool_feedback_tx.read().await.as_ref() {
                let _ = tx.send(ToolFeedback {
                    tool_name: String::new(),
                    phase: ToolFeedbackPhase::ThinkingDone,
                    args_json: None,
                });
            }
            let thinking_detail = build_thinking_detail(&response);
            thinking_scope.finish(thinking_detail.as_deref()).await;

            if let (Some(metrics), Some(usage)) = (usage_metrics.as_ref(), response.usage.as_ref())
            {
                metrics.record_tokens_with_cache(
                    usage.prompt_tokens as u64,
                    usage.completion_tokens as u64,
                    usage.cached_tokens as u64,
                    usage.cache_creation_tokens as u64,
                );
            }
            if let Some(usage) = response.usage.as_ref() {
                metrics_collector.record_tokens_with_cache(
                    usage.prompt_tokens as u64,
                    usage.completion_tokens as u64,
                    usage.cached_tokens as u64,
                    usage.cache_creation_tokens as u64,
                );
                self.agent
                    .token_budget
                    .record(usage.prompt_tokens as u64, usage.completion_tokens as u64);
            }
        }

        let max_iter_reached = iteration >= max_iterations && response.has_tool_calls();
        if max_iter_reached {
            info!(
                iterations = iteration,
                "Tool loop reached maximum iterations, attempting final synthesis"
            );
        }

        // Signal that tools are done and response is ready
        if let Some(tx) = self.agent.tool_feedback_tx.read().await.as_ref() {
            let _ = tx.send(ToolFeedback {
                tool_name: String::new(),
                phase: ToolFeedbackPhase::ResponseReady,
                args_json: None,
            });
        }

        // Phase 1: classify the raw final content. ToolCalls cannot occur
        // here (classify_final_content does not return that variant) so we
        // only need to look at content shape.
        let initial_outcome = classify_final_content(&response.content);
        let cfg_defaults = &self.agent.config.agents.defaults;
        let synthesis_trigger = classify_synthesis_trigger(
            &initial_outcome,
            max_iter_reached,
            cfg_defaults.final_synthesis_on_empty,
            cfg_defaults.final_synthesis_on_tool_limit,
        );

        // Phase 2: when the loop produces no usable answer but synthesis is
        // enabled, run one tools-disabled synthesis turn before deciding
        // the final outcome. The synthesis result is itself classified, so
        // an empty / markup-only synthesis response still fails explicitly.
        let outcome = if let Some(trigger) = synthesis_trigger {
            let provider_opt = self
                .agent
                .provider
                .read()
                .await
                .as_ref()
                .map(Arc::clone);
            match provider_opt {
                Some(provider) => {
                    info!(
                        trigger = trigger,
                        iterations = iteration,
                        "agent_turn: running final synthesis"
                    );
                    let synthesis_messages = self
                        .agent
                        .build_resolved_messages(msg, &session, memory_override.as_deref())
                        .await;
                    let chat_options = ChatOptions::new()
                        .with_max_tokens(cfg_defaults.max_tokens)
                        .with_temperature(cfg_defaults.temperature);
                    let model = Some(cfg_defaults.model.clone());
                    match crate::agent::synthesis::run_final_synthesis(
                        provider,
                        synthesis_messages,
                        model,
                        chat_options,
                    )
                    .await
                    {
                        Ok(resp) => classify_final_content(&resp.content),
                        Err(e) => {
                            warn!(
                                trigger = trigger,
                                error = %e,
                                "agent_turn: final synthesis call failed; falling back to initial outcome"
                            );
                            initial_outcome
                        }
                    }
                }
                None => {
                    warn!(
                        trigger = trigger,
                        "agent_turn: final synthesis configured but no provider attached; falling back to initial outcome"
                    );
                    initial_outcome
                }
            }
        } else {
            initial_outcome
        };

        match outcome {
            TurnOutcome::FinalAnswer(text) => {
                session.add_message(Message::assistant(&text));
                self.agent.session_manager.save(&session).await?;
                Ok(text)
            }
            TurnOutcome::EmptyAnswer => {
                warn!(
                    iterations = iteration,
                    tool_call_count = response.tool_calls.len(),
                    "agent_turn: empty final answer after synthesis attempt"
                );
                Err(ZeptoError::Provider(
                    "模型返回了空白最终答案。请重试或缩小任务范围。".to_string(),
                ))
            }
            TurnOutcome::ProviderMarkupOnly => {
                warn!(
                    iterations = iteration,
                    "agent_turn: provider tool markup leaked into final content after synthesis attempt"
                );
                Err(ZeptoError::Provider(
                    "模型未生成可展示的最终答案(只返回了内部工具调用标记)。请重试。"
                        .to_string(),
                ))
            }
            TurnOutcome::ToolCalls(_) => unreachable!(
                "classify_final_content cannot return ToolCalls; only classify_turn_outcome can"
            ),
        }
    }
}
