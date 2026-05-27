//! Context builder for agent conversations
//!
//! This module provides the `ContextBuilder` for constructing the system prompt
//! and message history for LLM conversations. It also provides `RuntimeContext`
//! for injecting environment-awareness into the agent's system prompt.

use chrono::Local;

use crate::session::Message;

/// Format a timestamp envelope for a user message.
///
/// Returns a string like "[Monday 2026-02-16 12:51 +08:00]" to prepend to user messages.
/// Uses the system's local timezone via `chrono::Local`.
/// Full day-of-week name (%A) is used to prevent LLM day-of-week hallucination.
pub fn format_message_envelope() -> String {
    format!("[{}]", Local::now().format("%A %Y-%m-%d %H:%M %:z"))
}

/// Default system prompt for ZeptoClaw agent
const DEFAULT_SYSTEM_PROMPT: &str = r#"You are ZeptoClaw, a concise personal AI assistant. Use tools when useful.

Memory: use longterm_memory to save durable facts, user preferences, decisions, and critical pinned facts; recall them with action "search" when relevant.

Use ask_clarification for missing required info, real ambiguity, or destructive/irreversible actions; not trivial choices.

Scheduled messages: `Reminder:` means the scheduler delivered a reminder; notify the user in the reminder's voice. For heartbeat workspace checks, reply `HEARTBEAT_OK` when nothing needs action, otherwise act.

Visuals on a text-only channel: answer charts/dashboards with concise prose and, when useful, a markdown table. Do not emit JSON UI, Mermaid, or plotting code unless requested."#;

/// Optional system prompt segment appended only when the active channel can
/// render A2UI surfaces (e.g. the Tauri client over `acp_http`). Channels
/// without a renderer (Discord / Telegram / WhatsApp / CLI) skip this so the
/// model does not produce raw JSON the user would see verbatim.
///
/// Body lives in `zeptoclaw/prompts/a2ui_v0_9.md` so prompt edits can be
/// reviewed as a markdown diff and don't churn `context.rs`. The caller
/// (`build_messages`) is responsible for inserting the blank-line
/// separator from the preceding system_prompt section, so the file
/// itself starts cleanly with a `## ...` heading.
const A2UI_RENDERING_PROMPT_SUFFIX: &str = include_str!("../../prompts/a2ui_v0_9.md");

/// System prompt suffix for first-run persona guidance.
// Wired in by the persona override extraction task (common.rs); suppress
// the dead_code lint until that integration step is complete.
#[allow(dead_code)]
pub const FIRST_RUN_PERSONA_PROMPT: &str = r#"

## First Conversation Setup

This appears to be a new chat. Take a moment to introduce yourself briefly and ask the user what kind of assistant they'd like you to be. Offer these options:

1. **Default** — balanced and helpful
2. **Concise** — short, direct answers
3. **Friendly** — warm and conversational
4. **Professional** — formal and structured
5. **Creative** — playful and imaginative
6. **Technical** — detailed expert mode

Say something like: "Hi! I'm your AI assistant. I can adapt my style to suit you. Would you like me to be concise, friendly, professional, creative, or technical? Or just say 'default' for a balanced approach. You can also describe any custom style you'd like!"

After the user responds, save their preference using longterm_memory with key "persona_pref:{chat_id}" and apply it going forward."#;

/// Runtime context injected into the system prompt to make agents environment-aware.
///
/// This struct captures information about the agent's runtime environment such as
/// the channel it is running on, available tools, current time, workspace path,
/// and OS/platform details. When rendered, it produces a `## Runtime Context`
/// section appended to the system prompt.
///
/// # Example
///
/// ```rust
/// use zeptoclaw::agent::RuntimeContext;
///
/// let ctx = RuntimeContext::new()
///     .with_channel("telegram")
///     .with_tools(vec!["shell".to_string(), "web_search".to_string()])
///     .with_workspace("/home/user/project")
///     .with_os_info();
///
/// let rendered = ctx.render().unwrap();
/// assert!(rendered.contains("Channel: telegram"));
/// assert!(rendered.contains("shell, web_search"));
/// ```
#[derive(Debug, Clone, Default)]
pub struct RuntimeContext {
    /// The channel the agent is running on (e.g., "telegram", "cli", "whatsapp", "discord")
    pub channel: Option<String>,
    /// Names of available tools
    pub available_tools: Vec<String>,
    /// Timezone label (e.g., "Asia/Kuala_Lumpur"). When set, `render()` emits a
    /// **day-granular** `Today is: ...` line in the system prompt (computed live
    /// via `chrono::Local`).
    ///
    /// Minute-level time and timezone offset are intentionally **not** emitted:
    /// they would change every minute and invalidate the prompt-cache prefix on
    /// every call. Channel UIs (Telegram/Discord/CLI/WhatsApp) already display
    /// message timestamps to the user; the model retains day-of-week awareness
    /// via the `Today is:` label.
    pub timezone: Option<String>,
    /// Workspace path
    pub workspace: Option<String>,
    /// OS/platform info (e.g., "linux aarch64", "macos aarch64")
    pub os_info: Option<String>,
}

impl RuntimeContext {
    /// Create a new empty runtime context.
    ///
    /// # Example
    /// ```rust
    /// use zeptoclaw::agent::RuntimeContext;
    ///
    /// let ctx = RuntimeContext::new();
    /// assert!(ctx.is_empty());
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the channel name.
    ///
    /// # Arguments
    /// * `channel` - The channel identifier (e.g., "telegram", "cli", "discord")
    pub fn with_channel(mut self, channel: &str) -> Self {
        self.channel = Some(channel.to_string());
        self
    }

    /// Set the list of available tool names.
    ///
    /// # Arguments
    /// * `tools` - Vector of tool name strings
    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.available_tools = tools;
        self
    }

    /// Set the current time to now (UTC).
    #[deprecated(note = "Use with_timezone() instead for live time computation")]
    pub fn with_current_time(mut self) -> Self {
        self.timezone = Some("UTC".to_string());
        self
    }

    /// Set the timezone label and enable the day-granular `Today is:` line in
    /// the system prompt.
    ///
    /// When a timezone is set, `render()` emits `- Today is: <Day>, <Month> <D>, <YYYY>`
    /// computed live via `chrono::Local`. Minute-level time and timezone offset
    /// are intentionally **not** emitted to keep the prompt-cache prefix stable
    /// (changing every minute would invalidate every cached prefix). The `tz`
    /// label itself is currently used only as an "enable" flag; the rendered
    /// date follows the host's local timezone.
    ///
    /// # Arguments
    /// * `tz` - Timezone label (e.g., "Asia/Kuala_Lumpur", "US/Pacific", "UTC")
    pub fn with_timezone(mut self, tz: &str) -> Self {
        self.timezone = Some(tz.to_string());
        self
    }

    /// Set the workspace path.
    ///
    /// # Arguments
    /// * `workspace` - The workspace directory path
    pub fn with_workspace(mut self, workspace: &str) -> Self {
        self.workspace = Some(workspace.to_string());
        self
    }

    /// Set the OS/platform info from the current environment.
    pub fn with_os_info(mut self) -> Self {
        self.os_info = Some(format!(
            "{} {}",
            std::env::consts::OS,
            std::env::consts::ARCH
        ));
        self
    }

    /// Check if any context field is set.
    ///
    /// Returns `true` if no fields have been populated.
    ///
    /// # Example
    /// ```rust
    /// use zeptoclaw::agent::RuntimeContext;
    ///
    /// assert!(RuntimeContext::new().is_empty());
    /// assert!(!RuntimeContext::new().with_channel("cli").is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.channel.is_none()
            && self.available_tools.is_empty()
            && self.timezone.is_none()
            && self.workspace.is_none()
            && self.os_info.is_none()
    }

    /// Render the context as a markdown section for the system prompt.
    ///
    /// Returns `None` if no context fields are set.
    ///
    /// Kept for backward compatibility: emits a single `## Runtime Context`
    /// section containing both stable fields (Channel / Tools / Workspace /
    /// Platform) and the day-granular `Today is:` line at the tail of the
    /// section. The layered builder (`ContextBuilder::build_system_layered`)
    /// uses `render_stable` and `render_volatile` instead so the volatile
    /// line lands past the memory section in the final prompt.
    ///
    /// # Example
    /// ```rust
    /// use zeptoclaw::agent::RuntimeContext;
    ///
    /// let ctx = RuntimeContext::new().with_channel("cli");
    /// let rendered = ctx.render().unwrap();
    /// assert!(rendered.starts_with("## Runtime Context"));
    /// assert!(rendered.contains("Channel: cli"));
    /// ```
    pub fn render(&self) -> Option<String> {
        if self.is_empty() {
            return None;
        }

        let mut parts = Vec::new();
        if let Some(ref channel) = self.channel {
            parts.push(format!("- Channel: {}", channel));
        }
        if !self.available_tools.is_empty() {
            parts.push(format!(
                "- Available tools: {}",
                self.available_tools.join(", ")
            ));
        }
        if let Some(ref workspace) = self.workspace {
            parts.push(format!("- Workspace: {}", workspace));
            parts.push(
                "- Workspace path rules: filesystem tools are scoped here. Prefer relative paths like `.` or `dir/file`; do not invent host paths outside the configured workspace.".to_string(),
            );
        }
        if let Some(ref os) = self.os_info {
            parts.push(format!("- Platform: {}", os));
        }
        // P0a: emit only the day-granular `Today is:` label. Minute-level
        // time and timezone offset are intentionally NOT emitted — they
        // would change every minute and invalidate the prompt-cache prefix
        // on every LLM call. Channel UIs already show message timestamps;
        // the model still has day-of-week awareness via this label.
        //
        // P3: in render() the Today line trails the stable fields so the
        // single-section legacy callers still see one `## Runtime Context`
        // block. The layered builder splits it out to L4.
        if self.timezone.is_some() {
            let now = Local::now();
            parts.push(format!("- Today is: {}", now.format("%A, %B %-d, %Y")));
        }

        Some(format!("## Runtime Context\n\n{}", parts.join("\n")))
    }

    /// Render only the **stable** portion of runtime context (L2 semi-static).
    ///
    /// Emits a `## Runtime Context` section with Channel + Available tools +
    /// Workspace (and path rules) + Platform. **Excludes** the day-granular
    /// `Today is:` line — that lives in `render_volatile` (L4) so the prompt
    /// prefix can be shared across days as long as the layered prompt is
    /// reassembled in L1→L2→L3→L4 order. See plan
    /// `docs/plans/doing/2026-05-21-token-cost-optimization.md §6.4`.
    ///
    /// Returns `None` when no stable field is set (timezone alone does not
    /// qualify this section as renderable; it only contributes to the
    /// volatile section).
    pub fn render_stable(&self) -> Option<String> {
        let has_stable = self.channel.is_some()
            || !self.available_tools.is_empty()
            || self.workspace.is_some()
            || self.os_info.is_some();
        if !has_stable {
            return None;
        }
        let mut parts = Vec::new();
        if let Some(ref channel) = self.channel {
            parts.push(format!("- Channel: {}", channel));
        }
        if !self.available_tools.is_empty() {
            parts.push(format!(
                "- Available tools: {}",
                self.available_tools.join(", ")
            ));
        }
        if let Some(ref workspace) = self.workspace {
            parts.push(format!("- Workspace: {}", workspace));
            parts.push(
                "- Workspace path rules: filesystem tools are scoped here. Prefer relative paths like `.` or `dir/file`; do not invent host paths outside the configured workspace.".to_string(),
            );
        }
        if let Some(ref os) = self.os_info {
            parts.push(format!("- Platform: {}", os));
        }
        Some(format!("## Runtime Context\n\n{}", parts.join("\n")))
    }

    /// Render only the **volatile** portion of runtime context (L4).
    ///
    /// Emits a `## Session Now` section with the day-granular `Today is:`
    /// line. Returns `None` when no timezone is configured.
    ///
    /// Day rollovers invalidate this section; placing it at the prompt tail
    /// keeps the L1+L2+L3 prefix cacheable across the rollover.
    pub fn render_volatile(&self) -> Option<String> {
        self.timezone.as_ref()?;
        let now = Local::now();
        Some(format!(
            "## Session Now\n\n- Today is: {}",
            now.format("%A, %B %-d, %Y")
        ))
    }
}

/// Per-message prompt capabilities decided by the caller (typically the
/// agent loop) based on the inbound channel. Sections of the system prompt
/// that depend on a renderer (e.g. A2UI surfaces) are gated on these flags
/// so non-rendering channels do not see them.
#[derive(Debug, Clone, Copy, Default)]
pub struct PromptCapabilities {
    /// True when the active channel can render A2UI v0.9 surfaces.
    pub a2ui_capable: bool,
}

impl PromptCapabilities {
    /// Convenience constructor for the A2UI-capable variant.
    pub fn with_a2ui() -> Self {
        Self { a2ui_capable: true }
    }
}

/// Structured layered view of the system prompt for cache-aware assembly.
///
/// The four layers are concatenated in L1→L2→L3→L4 order with blank-line
/// separators (`join`). Layout follows plan
/// `docs/plans/doing/2026-05-21-token-cost-optimization.md §6.4`:
///
/// - **L1 `base_static`** — persona (SOUL), base system prompt, A2UI suffix
///   when the channel can render A2UI surfaces. Static across the
///   conversation; lives at the top of the prompt where prompt caches anchor.
/// - **L2 `semi_static`** — Available Skills section and the stable runtime
///   fields (Channel / Available tools / Workspace / Platform). These drift
///   slowly (a skill enabled, a workspace renamed) so they sit just below
///   the static layer.
/// - **L3 `memory`** — long-term memory (pinned + recall). Tied to the user
///   session; sits below the workspace fingerprint so memory churn doesn't
///   invalidate the L1/L2 prefix.
/// - **L4 `volatile`** — day-granular `Today is:` and any other per-turn
///   volatile signals. Always tail of the prompt so a day rollover only
///   shifts the suffix.
///
/// Used by `ContextBuilder::build_system_layered` and also returned to
/// callers that want layer-level measurements (planned: P1.5
/// `cache_baseline` token accounting).
#[derive(Debug, Clone)]
pub struct LayeredSystemPrompt {
    /// L1: base static persona + system prompt + (optional) A2UI suffix.
    /// Always populated; the system prompt itself is mandatory.
    pub l1_base: String,
    /// L2: semi-static (Skills + stable runtime context).
    /// `None` when neither skills nor stable runtime fields are configured.
    pub l2_semi_static: Option<String>,
    /// L3: memory section, already pre-formatted with its own `## Memory`
    /// heading by the caller. `None` when no memory context is configured.
    pub l3_memory: Option<String>,
    /// L4: per-turn volatile section (currently the day-granular `Today is:`
    /// label, headed `## Session Now`). `None` when no timezone is set.
    pub l4_volatile: Option<String>,
}

impl LayeredSystemPrompt {
    /// Join the populated layers in L1→L2→L3→L4 order with blank-line
    /// separators. Returns the full system-prompt string ready to attach to
    /// a `Message::system`.
    pub fn join(&self) -> String {
        let mut s = self.l1_base.clone();
        if let Some(ref l2) = self.l2_semi_static {
            s.push_str("\n\n");
            s.push_str(l2);
        }
        if let Some(ref l3) = self.l3_memory {
            s.push_str("\n\n");
            s.push_str(l3);
        }
        if let Some(ref l4) = self.l4_volatile {
            s.push_str("\n\n");
            s.push_str(l4);
        }
        s
    }
}

/// Builder for constructing conversation context for LLM calls.
///
/// The `ContextBuilder` helps construct the full message list including
/// system prompts, skills information, conversation history, and user input.
///
/// # Example
///
/// ```rust
/// use zeptoclaw::agent::ContextBuilder;
/// use zeptoclaw::session::Message;
///
/// let builder = ContextBuilder::new()
///     .with_skills("- /help: Show help information");
///
/// let messages = builder.build_messages(&[], "Hello!");
/// assert_eq!(messages.len(), 2); // system + user message
/// ```
pub struct ContextBuilder {
    /// The system prompt to use
    system_prompt: String,
    /// Optional SOUL.md content prepended before system prompt
    soul_prompt: Option<String>,
    /// Optional skills content to append to system prompt
    skills_prompt: Option<String>,
    /// Optional runtime context to append to system prompt
    runtime_context: Option<RuntimeContext>,
    /// Optional memory context to append to system prompt
    memory_context: Option<String>,
    /// Optional anchored rolling summary (token-cost-optimization §P5).
    ///
    /// When set, `build_messages_with_overrides` inserts a single
    /// `system` message containing the summary right after the main
    /// system message and before conversation history. `None` (default)
    /// is byte-for-byte equivalent to the pre-P5.1 prompt. P5.1 ships
    /// dormant — no caller writes this slot yet.
    anchored_summary: Option<String>,
}

impl ContextBuilder {
    /// Create a new context builder with the default system prompt.
    ///
    /// # Example
    /// ```rust
    /// use zeptoclaw::agent::ContextBuilder;
    ///
    /// let builder = ContextBuilder::new();
    /// let system = builder.build_system_message();
    /// assert!(system.content.contains("ZeptoClaw"));
    /// ```
    pub fn new() -> Self {
        Self {
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
            soul_prompt: None,
            skills_prompt: None,
            runtime_context: None,
            memory_context: None,
            anchored_summary: None,
        }
    }

    /// Set a custom system prompt.
    ///
    /// # Arguments
    /// * `prompt` - The custom system prompt to use
    ///
    /// # Example
    /// ```rust
    /// use zeptoclaw::agent::ContextBuilder;
    ///
    /// let builder = ContextBuilder::new()
    ///     .with_system_prompt("You are a helpful assistant.");
    /// let system = builder.build_system_message();
    /// assert!(system.content.contains("helpful assistant"));
    /// ```
    pub fn with_system_prompt(mut self, prompt: &str) -> Self {
        self.system_prompt = prompt.to_string();
        self
    }

    /// Set SOUL.md identity content, prepended before the system prompt.
    ///
    /// SOUL.md defines the agent's personality, values, and behavioral
    /// constraints. Content is prepended to the system prompt so it takes
    /// priority in the LLM's context.
    ///
    /// # Arguments
    /// * `content` - The SOUL.md content to prepend
    ///
    /// # Example
    /// ```rust
    /// use zeptoclaw::agent::ContextBuilder;
    ///
    /// let builder = ContextBuilder::new()
    ///     .with_soul("You are kind and empathetic.");
    /// let system = builder.build_system_message();
    /// assert!(system.content.starts_with("You are kind"));
    /// ```
    pub fn with_soul(mut self, content: &str) -> Self {
        self.soul_prompt = Some(content.to_string());
        self
    }

    /// Add skills information to the system prompt.
    ///
    /// Skills content is appended to the system prompt under an
    /// "Available Skills" section.
    ///
    /// # Arguments
    /// * `skills_content` - The skills documentation to include
    ///
    /// # Example
    /// ```rust
    /// use zeptoclaw::agent::ContextBuilder;
    ///
    /// let builder = ContextBuilder::new()
    ///     .with_skills("- /search: Search the web\n- /help: Show help");
    /// let system = builder.build_system_message();
    /// assert!(system.content.contains("Available Skills"));
    /// assert!(system.content.contains("/search"));
    /// ```
    pub fn with_skills(mut self, skills_content: &str) -> Self {
        self.skills_prompt = Some(skills_content.to_string());
        self
    }

    /// Add runtime context to the system prompt.
    ///
    /// Runtime context provides the agent with awareness of its environment
    /// including the channel, available tools, current time, workspace, and
    /// platform information.
    ///
    /// If the provided context is empty (no fields set), it is ignored.
    ///
    /// # Arguments
    /// * `ctx` - The runtime context to inject
    ///
    /// # Example
    /// ```rust
    /// use zeptoclaw::agent::{ContextBuilder, RuntimeContext};
    ///
    /// let ctx = RuntimeContext::new()
    ///     .with_channel("discord")
    ///     .with_os_info();
    /// let builder = ContextBuilder::new().with_runtime_context(ctx);
    /// let system = builder.build_system_message();
    /// assert!(system.content.contains("Runtime Context"));
    /// assert!(system.content.contains("discord"));
    /// ```
    pub fn with_runtime_context(mut self, ctx: RuntimeContext) -> Self {
        if !ctx.is_empty() {
            self.runtime_context = Some(ctx);
        }
        self
    }

    /// Add memory context to the system prompt.
    ///
    /// Injects long-term memory content (pinned + relevant entries) as a
    /// `## Memory` section. If the provided string is empty, it is ignored.
    ///
    /// # Arguments
    /// * `memory_context` - Pre-built memory injection string
    ///
    /// # Example
    /// ```rust
    /// use zeptoclaw::agent::ContextBuilder;
    ///
    /// let builder = ContextBuilder::new()
    ///     .with_memory_context("## Memory\n\n### Pinned\n- user:name: Alice".to_string());
    /// let system = builder.build_system_message();
    /// assert!(system.content.contains("## Memory"));
    /// ```
    pub fn with_memory_context(mut self, memory_context: String) -> Self {
        if !memory_context.is_empty() {
            self.memory_context = Some(memory_context);
        }
        self
    }

    /// Append a suffix to the system prompt.
    ///
    /// Used for injecting additional instructions like first-run persona prompts.
    pub fn with_system_prompt_suffix(mut self, suffix: &str) -> Self {
        self.system_prompt.push_str(suffix);
        self
    }

    /// Build the system message with all configured content.
    ///
    /// # Returns
    /// A `Message` with role `System` containing the full system prompt.
    ///
    /// # Example
    /// ```rust
    /// use zeptoclaw::agent::ContextBuilder;
    /// use zeptoclaw::session::Role;
    ///
    /// let builder = ContextBuilder::new();
    /// let system = builder.build_system_message();
    /// assert_eq!(system.role, Role::System);
    /// ```
    pub fn build_system_message(&self) -> Message {
        let layered = self.build_system_layered(PromptCapabilities::default(), None);
        Message::system(&layered.join())
    }

    /// Internal builder that supports both memory override and channel
    /// capability gating (e.g. the A2UI suffix is only injected when the
    /// caller signals the channel can render A2UI surfaces).
    fn build_system_message_with_overrides(
        &self,
        memory_override: Option<&str>,
        caps: PromptCapabilities,
    ) -> Message {
        let layered = self.build_system_layered(caps, memory_override);
        Message::system(&layered.join())
    }

    /// Build the system prompt as a structured `LayeredSystemPrompt` so
    /// callers (and metrics code) can reason about cache-friendly layers.
    ///
    /// Layer assignment follows plan
    /// `docs/plans/doing/2026-05-21-token-cost-optimization.md §6.4`:
    ///
    /// - L1: SOUL (if any) + base `system_prompt` + A2UI suffix (when
    ///   `caps.a2ui_capable`).
    /// - L2: `## Available Skills` (if configured) and the stable runtime
    ///   context fields (Channel / Available tools / Workspace / Platform).
    ///   Excludes the `Today is:` line — that drifts daily so it belongs in
    ///   L4 to keep the L1+L2+L3 prefix cacheable across day rollovers.
    /// - L3: memory section (with optional per-call override; `Some("")`
    ///   explicitly suppresses memory for this build).
    /// - L4: the day-granular volatile runtime context (`## Session Now`).
    ///
    /// `LayeredSystemPrompt::join` concatenates the populated layers with
    /// blank-line separators in L1→L2→L3→L4 order.
    pub fn build_system_layered(
        &self,
        caps: PromptCapabilities,
        memory_override: Option<&str>,
    ) -> LayeredSystemPrompt {
        // L1: persona + base + (optional) A2UI suffix.
        let mut l1 = String::new();
        if let Some(ref soul) = self.soul_prompt {
            l1.push_str(soul);
            l1.push_str("\n\n");
        }
        l1.push_str(&self.system_prompt);
        if caps.a2ui_capable {
            l1.push_str("\n\n");
            l1.push_str(A2UI_RENDERING_PROMPT_SUFFIX);
        }

        // L2: skills section + stable runtime context.
        let mut l2_parts: Vec<String> = Vec::new();
        if let Some(ref skills) = self.skills_prompt {
            l2_parts.push(format!("## Available Skills\n\n{}", skills));
        }
        if let Some(ref ctx) = self.runtime_context {
            if let Some(stable) = ctx.render_stable() {
                l2_parts.push(stable);
            }
        }
        let l2_semi_static = if l2_parts.is_empty() {
            None
        } else {
            Some(l2_parts.join("\n\n"))
        };

        // L3: memory section, with per-call override.
        let l3_memory = match memory_override {
            Some("") => None,
            Some(memory) => Some(memory.to_string()),
            None => self.memory_context.clone(),
        };

        // L4: day-granular volatile runtime context (Today is:).
        let l4_volatile = self
            .runtime_context
            .as_ref()
            .and_then(|c| c.render_volatile());

        LayeredSystemPrompt {
            l1_base: l1,
            l2_semi_static,
            l3_memory,
            l4_volatile,
        }
    }

    /// Build the full message list for an LLM call.
    ///
    /// This constructs a message list with:
    /// 1. System message (with skills if configured)
    /// 2. Conversation history
    /// 3. New user input (if non-empty), with timestamp envelope when timezone is set
    ///
    /// # Arguments
    /// * `history` - The conversation history to include
    /// * `user_input` - The new user message (empty string is ignored)
    ///
    /// # Returns
    /// A vector of messages ready for the LLM.
    ///
    /// # Example
    /// ```rust
    /// use zeptoclaw::agent::ContextBuilder;
    /// use zeptoclaw::session::Message;
    ///
    /// let builder = ContextBuilder::new();
    /// let history = vec![
    ///     Message::user("Hello"),
    ///     Message::assistant("Hi there!"),
    /// ];
    /// let messages = builder.build_messages(&history, "How are you?");
    /// assert_eq!(messages.len(), 4); // system + 2 history + new user
    /// ```
    pub fn build_messages(&self, history: &[Message], user_input: &str) -> Vec<Message> {
        let mut messages = vec![self.build_system_message()];
        messages.extend(history.iter().cloned());
        if !user_input.is_empty() {
            // Prepend timestamp envelope to user message so the LLM knows
            // exactly when this message arrived (prevents stale-time bugs).
            let content = if let Some(ref ctx) = self.runtime_context {
                if ctx.timezone.is_some() {
                    let envelope = format_message_envelope();
                    format!("{} {}", envelope, user_input)
                } else {
                    user_input.to_string()
                }
            } else {
                user_input.to_string()
            };
            messages.push(Message::user(&content));
        }
        messages
    }

    /// Build the full message list with an optional per-message memory override.
    ///
    /// Works like `build_messages`, but `memory_override` can replace the
    /// stored memory section for this specific message build.
    pub fn build_messages_with_memory_override(
        &self,
        history: &[Message],
        user_input: &str,
        memory_override: Option<&str>,
    ) -> Vec<Message> {
        self.build_messages_with_overrides(
            history,
            user_input,
            memory_override,
            PromptCapabilities::default(),
        )
    }

    /// Build the full message list, gating channel-specific prompt sections
    /// (currently the A2UI rendering suffix) on `caps`. The agent loop sets
    /// `caps.a2ui_capable` based on the channel of the inbound message so
    /// non-A2UI channels (Discord, CLI, ...) never see the A2UI guidance.
    pub fn build_messages_with_overrides(
        &self,
        history: &[Message],
        user_input: &str,
        memory_override: Option<&str>,
        caps: PromptCapabilities,
    ) -> Vec<Message> {
        let mut messages = vec![self.build_system_message_with_overrides(memory_override, caps)];
        // Anchored rolling summary slot (token-cost-optimization §P5).
        // Inserted between the main system message and history so the
        // LLM treats it as authoritative compressed context. P5.1 keeps
        // this dormant (`anchored_summary` is always `None` in current
        // callers); P5.2 wires the Harness to populate it.
        if let Some(summary) = &self.anchored_summary {
            messages.push(Message::system(summary));
        }
        messages.extend(history.iter().cloned());
        if !user_input.is_empty() {
            let content = if let Some(ref ctx) = self.runtime_context {
                if ctx.timezone.is_some() {
                    let envelope = format_message_envelope();
                    format!("{} {}", envelope, user_input)
                } else {
                    user_input.to_string()
                }
            } else {
                user_input.to_string()
            };
            messages.push(Message::user(&content));
        }
        messages
    }

    /// Set or clear the anchored rolling summary slot
    /// (token-cost-optimization §P5).
    ///
    /// When `Some(text)`, `build_messages_with_overrides` inserts one
    /// extra `Message::system(text)` right after the main system
    /// message and before history. `None` (default) keeps the prompt
    /// byte-for-byte identical to pre-P5.1.
    ///
    /// P5.1 ships this method dormant — no caller in `agent::loop` or
    /// `agent::harness` sets it yet. P5.2 wires it up.
    ///
    /// # Example
    /// ```rust
    /// use zeptoclaw::agent::ContextBuilder;
    ///
    /// let builder = ContextBuilder::new()
    ///     .with_anchored_summary(Some("Previous discussion: ...".into()));
    /// assert!(builder.anchored_summary().is_some());
    /// ```
    pub fn with_anchored_summary(mut self, summary: Option<String>) -> Self {
        self.anchored_summary = summary;
        self
    }

    /// Get the current system prompt.
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    /// Check if a SOUL.md identity is configured.
    pub fn has_soul(&self) -> bool {
        self.soul_prompt.is_some()
    }

    /// Check if skills are configured.
    pub fn has_skills(&self) -> bool {
        self.skills_prompt.is_some()
    }

    /// Get the configured anchored rolling summary, if any.
    pub fn anchored_summary(&self) -> Option<&str> {
        self.anchored_summary.as_deref()
    }
}

impl Default for ContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::Role;

    #[test]
    fn test_context_builder_new() {
        let builder = ContextBuilder::new();
        assert!(builder.system_prompt().contains("ZeptoClaw"));
        assert!(!builder.has_skills());
    }

    #[test]
    fn test_context_builder_default() {
        let builder = ContextBuilder::default();
        assert!(builder.system_prompt().contains("ZeptoClaw"));
    }

    #[test]
    fn test_context_builder_custom_system_prompt() {
        let builder = ContextBuilder::new().with_system_prompt("Custom prompt here");
        assert_eq!(builder.system_prompt(), "Custom prompt here");
    }

    #[test]
    fn test_context_builder_with_skills() {
        let builder = ContextBuilder::new().with_skills("- /test: Test skill");
        assert!(builder.has_skills());

        let system = builder.build_system_message();
        assert!(system.content.contains("Available Skills"));
        assert!(system.content.contains("/test"));
    }

    #[test]
    fn test_build_system_message() {
        let builder = ContextBuilder::new();
        let system = builder.build_system_message();

        assert_eq!(system.role, Role::System);
        assert!(system.content.contains("ZeptoClaw"));
    }

    #[test]
    fn test_build_messages_empty_input() {
        let builder = ContextBuilder::new();
        let messages = builder.build_messages(&[], "");

        // Only system message when input is empty
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, Role::System);
    }

    #[test]
    fn test_build_messages_with_input() {
        let builder = ContextBuilder::new();
        let messages = builder.build_messages(&[], "Hello");

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[1].role, Role::User);
        assert_eq!(messages[1].content, "Hello");
    }

    #[test]
    fn test_build_messages_with_history() {
        let builder = ContextBuilder::new();
        let history = vec![
            Message::user("Previous message"),
            Message::assistant("Previous response"),
        ];
        let messages = builder.build_messages(&history, "New message");

        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[1].role, Role::User);
        assert_eq!(messages[1].content, "Previous message");
        assert_eq!(messages[2].role, Role::Assistant);
        assert_eq!(messages[3].role, Role::User);
        assert_eq!(messages[3].content, "New message");
    }

    #[test]
    fn test_build_messages_preserves_history_order() {
        let builder = ContextBuilder::new();
        let history = vec![
            Message::user("First"),
            Message::assistant("Second"),
            Message::user("Third"),
            Message::assistant("Fourth"),
        ];
        let messages = builder.build_messages(&history, "");

        // System + 4 history messages (no new input since it's empty)
        assert_eq!(messages.len(), 5);
        assert_eq!(messages[1].content, "First");
        assert_eq!(messages[2].content, "Second");
        assert_eq!(messages[3].content, "Third");
        assert_eq!(messages[4].content, "Fourth");
    }

    #[test]
    fn test_context_builder_with_soul() {
        let builder = ContextBuilder::new().with_soul("You are a pirate captain.");
        assert!(builder.has_soul());

        let system = builder.build_system_message();
        assert!(system.content.starts_with("You are a pirate captain."));
        assert!(system.content.contains("ZeptoClaw"));
    }

    #[test]
    fn test_soul_prepended_before_system_prompt() {
        let builder = ContextBuilder::new()
            .with_soul("SOUL: Be kind.")
            .with_system_prompt("SYSTEM: Do tasks.");
        let system = builder.build_system_message();

        let soul_pos = system.content.find("SOUL: Be kind.").unwrap();
        let system_pos = system.content.find("SYSTEM: Do tasks.").unwrap();
        assert!(soul_pos < system_pos);
    }

    #[test]
    fn test_soul_with_skills() {
        let builder = ContextBuilder::new()
            .with_soul("Identity: helper")
            .with_skills("- /test: Test");
        let system = builder.build_system_message();

        assert!(system.content.starts_with("Identity: helper"));
        assert!(system.content.contains("ZeptoClaw"));
        assert!(system.content.contains("Available Skills"));
        assert!(system.content.contains("/test"));
    }

    #[test]
    fn test_no_soul_by_default() {
        let builder = ContextBuilder::new();
        assert!(!builder.has_soul());

        let system = builder.build_system_message();
        assert!(system.content.starts_with("You are ZeptoClaw"));
    }

    #[test]
    fn test_context_builder_chaining() {
        let builder = ContextBuilder::new()
            .with_system_prompt("Custom prompt")
            .with_skills("- /skill1: Do something");

        let system = builder.build_system_message();
        assert!(system.content.contains("Custom prompt"));
        assert!(system.content.contains("/skill1"));
    }

    #[test]
    fn test_build_messages_with_tool_calls_in_history() {
        use crate::session::ToolCall;

        let builder = ContextBuilder::new();
        let history = vec![
            Message::user("Search for rust"),
            Message::assistant_with_tools(
                "Let me search for that.",
                vec![ToolCall::new("call_1", "search", r#"{"q": "rust"}"#)],
            ),
            Message::tool_result("call_1", "Found 100 results"),
            Message::assistant("I found 100 results about Rust."),
        ];
        let messages = builder.build_messages(&history, "Thanks!");

        // System + 4 history + new user message
        assert_eq!(messages.len(), 6);
        assert!(messages[2].has_tool_calls());
        assert!(messages[3].is_tool_result());
    }

    // ---- RuntimeContext tests ----

    #[test]
    fn test_runtime_context_empty() {
        let ctx = RuntimeContext::new();
        assert!(ctx.is_empty());
        assert!(ctx.render().is_none());
    }

    #[test]
    fn test_runtime_context_default() {
        let ctx = RuntimeContext::default();
        assert!(ctx.is_empty());
        assert!(ctx.channel.is_none());
        assert!(ctx.available_tools.is_empty());
        assert!(ctx.timezone.is_none());
        assert!(ctx.workspace.is_none());
        assert!(ctx.os_info.is_none());
    }

    #[test]
    fn test_runtime_context_with_channel() {
        let ctx = RuntimeContext::new().with_channel("telegram");
        assert!(!ctx.is_empty());
        let rendered = ctx.render().unwrap();
        assert!(rendered.contains("Channel: telegram"));
    }

    #[test]
    fn test_runtime_context_with_tools() {
        let ctx =
            RuntimeContext::new().with_tools(vec!["shell".to_string(), "web_search".to_string()]);
        assert!(!ctx.is_empty());
        let rendered = ctx.render().unwrap();
        assert!(rendered.contains("shell, web_search"));
    }

    #[test]
    fn test_runtime_context_with_empty_tools() {
        let ctx = RuntimeContext::new().with_tools(vec![]);
        assert!(ctx.is_empty());
        assert!(ctx.render().is_none());
    }

    #[test]
    fn test_runtime_context_with_timezone() {
        let ctx = RuntimeContext::new().with_timezone("Asia/Kuala_Lumpur");
        assert!(!ctx.is_empty());
        let rendered = ctx.render().unwrap();
        // P0a: only the day-granular `Today is:` label is emitted; minute-level
        // `Current time:` and `Timezone:` lines were removed to keep the system
        // prompt prefix stable for prompt-cache hits.
        assert!(
            rendered.contains("Today is:"),
            "should emit day-granular label: {}",
            rendered
        );
        assert!(
            !rendered.contains("Current time:"),
            "must NOT emit minute-level Current time line (cache-killer): {}",
            rendered
        );
        assert!(
            !rendered.contains("Timezone:"),
            "must NOT emit Timezone offset line (cache-killer): {}",
            rendered
        );
    }

    #[test]
    fn test_runtime_context_with_utc_label() {
        let ctx = RuntimeContext::new().with_timezone("UTC");
        let rendered = ctx.render().unwrap();
        // P0a: regardless of timezone label, render() must produce only the
        // day-granular line and no minute-level / offset content.
        assert!(rendered.contains("Today is:"));
        assert!(!rendered.contains("Current time:"));
        assert!(!rendered.contains("Timezone:"));
    }

    #[test]
    fn test_runtime_context_time_format() {
        // Verify render produces the expected day-granular format: "Today is: <Day>, <Month> <D>, <YYYY>"
        let ctx = RuntimeContext::new().with_timezone("UTC");
        let rendered = ctx.render().unwrap();
        let today_line = rendered
            .lines()
            .find(|l| l.contains("Today is:"))
            .expect("render should emit a Today is: line when timezone is set");
        // Should contain a 4-digit year (no minute-level info anywhere on the line).
        assert!(
            today_line.contains("202"),
            "today line should contain year: {}",
            today_line
        );
    }

    #[test]
    fn test_runtime_context_today_label_present() {
        // Renamed from test_runtime_context_time_is_live (P0a):
        // we no longer expose minute-level "Current time", so the live-ness
        // we care about is the day-granular Today is: label and the absence
        // of minute / timezone-offset content.
        let ctx = RuntimeContext::new().with_timezone("local");
        let rendered = ctx.render().unwrap();
        assert!(
            rendered.contains("Today is:"),
            "should emit Today is: label: {}",
            rendered
        );
        assert!(
            !rendered.contains("Current time:"),
            "must NOT emit Current time: (cache-killer): {}",
            rendered
        );
        assert!(
            !rendered.contains("Timezone:"),
            "must NOT emit Timezone: (cache-killer): {}",
            rendered
        );
    }

    #[test]
    fn test_runtime_context_cache_friendly_render() {
        // Hard guarantee for prompt-cache prefix stability: render() output
        // must NOT contain any minute-level time substring or timezone offset.
        // This is the structural cause of cache stability — we assert the cause,
        // not the effect (e.g. no `sleep` based equality test, which would
        // flake on CI and time out on slow runners).
        let ctx = RuntimeContext::new()
            .with_channel("cli")
            .with_timezone("Asia/Kuala_Lumpur")
            .with_workspace("/tmp/ws")
            .with_os_info();
        let rendered = ctx.render().unwrap();

        // 1) No "HH:MM" minute marker anywhere in the rendered text.
        let has_minute_marker = rendered.as_bytes().windows(5).any(|w| {
            w.len() == 5
                && w[0].is_ascii_digit()
                && w[1].is_ascii_digit()
                && w[2] == b':'
                && w[3].is_ascii_digit()
                && w[4].is_ascii_digit()
        });
        assert!(
            !has_minute_marker,
            "render() must not contain HH:MM minute markers: {}",
            rendered
        );

        // 2) No timezone offset pattern "[+-]HH:MM" anywhere.
        let has_offset_pattern = rendered.as_bytes().windows(6).any(|w| {
            (w[0] == b'+' || w[0] == b'-')
                && w[1].is_ascii_digit()
                && w[2].is_ascii_digit()
                && w[3] == b':'
                && w[4].is_ascii_digit()
                && w[5].is_ascii_digit()
        });
        assert!(
            !has_offset_pattern,
            "render() must not contain timezone offset like +HH:MM: {}",
            rendered
        );

        // 3) Sanity: the day-granular label is still present.
        assert!(
            rendered.contains("Today is:"),
            "render() should still emit Today is: label: {}",
            rendered
        );
    }

    #[test]
    fn test_runtime_context_with_os_info() {
        let ctx = RuntimeContext::new().with_os_info();
        assert!(!ctx.is_empty());
        let rendered = ctx.render().unwrap();
        assert!(rendered.contains("Platform:"));
        // Should contain the current OS
        assert!(rendered.contains(std::env::consts::OS));
    }

    #[test]
    fn test_runtime_context_with_workspace() {
        let ctx = RuntimeContext::new().with_workspace("/home/user/project");
        assert!(!ctx.is_empty());
        let rendered = ctx.render().unwrap();
        assert!(rendered.contains("Workspace: /home/user/project"));
        assert!(rendered.contains("Workspace path rules:"));
        assert!(rendered.contains("Prefer relative paths"));
    }

    #[test]
    fn test_runtime_context_full() {
        let ctx = RuntimeContext::new()
            .with_channel("whatsapp")
            .with_tools(vec!["shell".to_string()])
            .with_workspace("/tmp/test")
            .with_os_info();
        let rendered = ctx.render().unwrap();
        assert!(rendered.contains("## Runtime Context"));
        assert!(rendered.contains("Channel: whatsapp"));
        assert!(rendered.contains("Available tools: shell"));
        assert!(rendered.contains("Workspace: /tmp/test"));
        assert!(rendered.contains("Platform:"));
    }

    #[test]
    fn test_runtime_context_render_ordering() {
        let ctx = RuntimeContext::new()
            .with_channel("cli")
            .with_tools(vec!["echo".to_string()])
            .with_workspace("/work");
        let rendered = ctx.render().unwrap();
        let channel_pos = rendered.find("Channel:").unwrap();
        let tools_pos = rendered.find("Available tools:").unwrap();
        let workspace_pos = rendered.find("Workspace:").unwrap();
        // Channel comes before tools, tools before workspace
        assert!(channel_pos < tools_pos);
        assert!(tools_pos < workspace_pos);
    }

    #[test]
    fn test_runtime_context_clone() {
        let ctx = RuntimeContext::new()
            .with_channel("discord")
            .with_workspace("/tmp");
        let cloned = ctx.clone();
        assert_eq!(ctx.channel, cloned.channel);
        assert_eq!(ctx.workspace, cloned.workspace);
    }

    // ---- P3 layered prompt tests ----
    //
    // The split between `render_stable` (L2) and `render_volatile` (L4) is the
    // structural enabler for prompt-cache hits across day rollovers. These
    // tests pin the layer-membership contract.

    #[test]
    fn p3_render_stable_excludes_today_line() {
        let ctx = RuntimeContext::new()
            .with_channel("cli")
            .with_tools(vec!["echo".to_string()])
            .with_workspace("/work")
            .with_timezone("UTC");
        let stable = ctx
            .render_stable()
            .expect("stable section should render with channel/tools/workspace");
        assert!(stable.starts_with("## Runtime Context"));
        assert!(stable.contains("Channel: cli"));
        assert!(stable.contains("Available tools: echo"));
        assert!(stable.contains("Workspace: /work"));
        assert!(
            !stable.contains("Today is:"),
            "stable section must NOT contain Today (cache-killer if reused across days): {}",
            stable
        );
    }

    #[test]
    fn p3_render_stable_none_when_only_timezone() {
        // Timezone alone is volatile-only signal; the stable section must
        // collapse to None so the layered builder doesn't emit an empty
        // `## Runtime Context` header.
        let ctx = RuntimeContext::new().with_timezone("UTC");
        assert!(ctx.render_stable().is_none());
    }

    #[test]
    fn p3_render_volatile_only_emits_today_line() {
        let ctx = RuntimeContext::new()
            .with_channel("cli")
            .with_workspace("/work")
            .with_timezone("UTC");
        let volatile = ctx
            .render_volatile()
            .expect("volatile should render when timezone is set");
        assert!(volatile.starts_with("## Session Now"));
        assert!(volatile.contains("Today is:"));
        assert!(
            !volatile.contains("Channel:"),
            "volatile must NOT contain Channel (channel lives in L2 stable): {}",
            volatile
        );
        assert!(
            !volatile.contains("Workspace:"),
            "volatile must NOT contain Workspace (workspace is L2 stable): {}",
            volatile
        );
    }

    #[test]
    fn p3_render_volatile_none_without_timezone() {
        let ctx = RuntimeContext::new()
            .with_channel("cli")
            .with_workspace("/work");
        assert!(ctx.render_volatile().is_none());
    }

    #[test]
    fn p3_layered_join_order_l1_l2_l3_l4() {
        // Full layered prompt: SOUL → system → skills + stable runtime → memory → volatile.
        let ctx = RuntimeContext::new()
            .with_channel("cli")
            .with_workspace("/work")
            .with_timezone("UTC");
        let builder = ContextBuilder::new()
            .with_soul("SOUL_ANCHOR")
            .with_skills("- /help: show help")
            .with_runtime_context(ctx)
            .with_memory_context("## Memory\n\n### Pinned\n- mem:anchor".to_string());
        let layered = builder.build_system_layered(PromptCapabilities::default(), None);

        // Each layer should be populated.
        assert!(layered.l1_base.contains("SOUL_ANCHOR"));
        assert!(layered.l1_base.contains("ZeptoClaw"));
        let l2 = layered.l2_semi_static.as_ref().expect("L2 should exist");
        assert!(l2.contains("## Available Skills"));
        assert!(l2.contains("## Runtime Context"));
        assert!(l2.contains("Channel: cli"));
        assert!(l2.contains("Workspace: /work"));
        assert!(!l2.contains("Today is:"));
        let l3 = layered.l3_memory.as_ref().expect("L3 should exist");
        assert!(l3.contains("mem:anchor"));
        let l4 = layered.l4_volatile.as_ref().expect("L4 should exist");
        assert!(l4.contains("Today is:"));

        // Joined order must be L1→L2→L3→L4.
        let joined = layered.join();
        let soul_pos = joined.find("SOUL_ANCHOR").unwrap();
        let skills_pos = joined.find("## Available Skills").unwrap();
        let runtime_pos = joined.find("## Runtime Context").unwrap();
        let memory_pos = joined.find("## Memory").unwrap();
        let today_pos = joined.find("Today is:").unwrap();
        assert!(soul_pos < skills_pos, "SOUL before Skills");
        assert!(skills_pos < runtime_pos, "Skills before Runtime Context");
        assert!(runtime_pos < memory_pos, "Runtime Context before Memory");
        assert!(memory_pos < today_pos, "Memory before Today is: (P3 core invariant)");
    }

    #[test]
    fn p3_built_system_message_places_today_after_memory() {
        // Behavioral test against the legacy entry point used by the agent
        // loop. Today must trail Memory in the final system prompt string so
        // a day rollover doesn't invalidate the prefix.
        let ctx = RuntimeContext::new()
            .with_channel("discord")
            .with_timezone("Asia/Kuala_Lumpur");
        let builder = ContextBuilder::new()
            .with_runtime_context(ctx)
            .with_memory_context("## Memory\n\n### Pinned\n- k: v".to_string());
        let system = builder.build_system_message();
        let memory_pos = system.content.find("## Memory").unwrap();
        let today_pos = system.content.find("Today is:").unwrap();
        assert!(
            memory_pos < today_pos,
            "P3 invariant: Today is: must come AFTER memory section. \
             memory_pos={} today_pos={} content=\n{}",
            memory_pos,
            today_pos,
            system.content
        );
    }

    #[test]
    fn p3_built_system_message_no_today_in_stable_runtime_block() {
        // Tighter contract: within the `## Runtime Context` section (stable
        // L2), there must be no `Today is:` line. We bound the search to the
        // region between `## Runtime Context` and the next blank-line section
        // header.
        let ctx = RuntimeContext::new()
            .with_channel("cli")
            .with_workspace("/work")
            .with_timezone("UTC");
        let builder = ContextBuilder::new().with_runtime_context(ctx);
        let system = builder.build_system_message();
        let rc_start = system.content.find("## Runtime Context").unwrap();
        // Slice from the runtime context heading to the next section heading.
        let tail = &system.content[rc_start..];
        let rc_end = tail[1..]
            .find("\n## ")
            .map(|p| rc_start + 1 + p)
            .unwrap_or(system.content.len());
        let runtime_block = &system.content[rc_start..rc_end];
        assert!(
            !runtime_block.contains("Today is:"),
            "stable L2 Runtime Context block must NOT contain Today is:. \
             block=\n{}",
            runtime_block
        );
    }

    #[test]
    fn p3_layered_optional_layers_collapse_cleanly() {
        // Only the base + system prompt should be present; no L2/L3/L4.
        let builder = ContextBuilder::new();
        let layered = builder.build_system_layered(PromptCapabilities::default(), None);
        assert!(layered.l1_base.contains("ZeptoClaw"));
        assert!(layered.l2_semi_static.is_none());
        assert!(layered.l3_memory.is_none());
        assert!(layered.l4_volatile.is_none());
        // join() should not introduce trailing whitespace from absent layers.
        let joined = layered.join();
        assert_eq!(joined.trim_end(), joined);
    }

    #[test]
    fn p3_layered_memory_override_empty_suppresses_l3() {
        let builder = ContextBuilder::new()
            .with_memory_context("## Memory\n\n- stored".to_string());
        let layered = builder.build_system_layered(PromptCapabilities::default(), Some(""));
        assert!(
            layered.l3_memory.is_none(),
            "Some(\"\") override must collapse L3 to None"
        );
    }

    #[test]
    fn p3_layered_a2ui_lives_in_l1() {
        // When the channel is A2UI-capable, the A2UI rendering suffix is
        // part of the static L1 layer (it does not drift per turn).
        let builder = ContextBuilder::new();
        let layered = builder.build_system_layered(PromptCapabilities::with_a2ui(), None);
        assert!(
            layered.l1_base.contains("A2UI"),
            "L1 should carry A2UI suffix when channel is A2UI-capable"
        );
    }

    // ---- Message envelope tests ----

    #[test]
    fn test_message_envelope_format() {
        let envelope = format_message_envelope();
        assert!(envelope.starts_with('['));
        assert!(envelope.ends_with(']'));
        // Should contain a date and UTC offset
        assert!(envelope.contains("202"), "should contain year");
        assert!(
            envelope.contains('+') || envelope.contains('-'),
            "should contain UTC offset sign"
        );
    }

    #[test]
    fn test_message_envelope_contains_time_components() {
        let envelope = format_message_envelope();
        // Format: [Day YYYY-MM-DD HH:MM +HH:MM]
        // Should contain a colon (from HH:MM time)
        assert!(envelope.contains(':'), "should contain time separator");
        // Should be bracketed
        assert!(envelope.len() > 10, "envelope should not be empty");
    }

    #[test]
    fn test_build_messages_with_timezone_envelope() {
        let ctx = RuntimeContext::new().with_timezone("UTC");
        let builder = ContextBuilder::new().with_runtime_context(ctx);
        let messages = builder.build_messages(&[], "Hello");
        assert_eq!(messages.len(), 2);
        // User message should start with timestamp envelope
        assert!(messages[1].content.starts_with('['));
        assert!(messages[1].content.contains("] Hello"));
    }

    #[test]
    fn test_build_messages_without_timezone_no_envelope() {
        let ctx = RuntimeContext::new().with_channel("cli");
        let builder = ContextBuilder::new().with_runtime_context(ctx);
        let messages = builder.build_messages(&[], "Hello");
        assert_eq!(messages[1].content, "Hello");
    }

    // ---- ContextBuilder + RuntimeContext integration tests ----

    #[test]
    fn test_context_builder_with_runtime_context() {
        let ctx = RuntimeContext::new().with_channel("discord");
        let builder = ContextBuilder::new().with_runtime_context(ctx);
        let system = builder.build_system_message();
        assert!(system.content.contains("Runtime Context"));
        assert!(system.content.contains("discord"));
    }

    #[test]
    fn test_context_builder_empty_runtime_context_adds_nothing() {
        let ctx = RuntimeContext::new();
        let builder = ContextBuilder::new().with_runtime_context(ctx);
        let system = builder.build_system_message();
        assert!(!system.content.contains("Runtime Context"));
    }

    #[test]
    fn test_context_builder_all_sections() {
        let ctx = RuntimeContext::new().with_channel("cli");
        let builder = ContextBuilder::new()
            .with_skills("- /help: Show help")
            .with_runtime_context(ctx);
        let system = builder.build_system_message();
        assert!(system.content.contains("ZeptoClaw"));
        assert!(system.content.contains("Available Skills"));
        assert!(system.content.contains("## Runtime Context"));
        assert!(system.content.contains("cli"));
    }

    #[test]
    fn test_context_builder_section_ordering() {
        let ctx = RuntimeContext::new().with_channel("slack");
        let builder = ContextBuilder::new()
            .with_skills("- /deploy: Deploy app")
            .with_runtime_context(ctx);
        let system = builder.build_system_message();
        let skills_pos = system.content.find("Available Skills").unwrap();
        let runtime_pos = system.content.find("Runtime Context").unwrap();
        // Skills section should come before runtime context
        assert!(skills_pos < runtime_pos);
    }

    #[test]
    fn test_context_builder_runtime_context_in_messages() {
        let ctx = RuntimeContext::new()
            .with_channel("telegram")
            .with_os_info();
        let builder = ContextBuilder::new().with_runtime_context(ctx);
        let messages = builder.build_messages(&[], "Hello");
        assert_eq!(messages.len(), 2);
        assert!(messages[0].content.contains("Runtime Context"));
        assert!(messages[0].content.contains("telegram"));
    }

    // ---- Memory context tests ----

    #[test]
    fn test_context_builder_with_memory_context() {
        let builder = ContextBuilder::new()
            .with_memory_context("## Memory\n\n### Pinned\n- user:name: Alice".to_string());
        let system = builder.build_system_message();
        assert!(system.content.contains("## Memory"));
        assert!(system.content.contains("user:name: Alice"));
    }

    #[test]
    fn test_context_builder_empty_memory_context_skipped() {
        let builder = ContextBuilder::new().with_memory_context(String::new());
        let system = builder.build_system_message();
        assert!(!system.content.contains("## Memory"));
    }

    #[test]
    fn test_context_builder_memory_after_runtime() {
        let ctx = RuntimeContext::new().with_channel("cli");
        let builder = ContextBuilder::new()
            .with_runtime_context(ctx)
            .with_memory_context("## Memory\n\n### Pinned\n- k: v".to_string());
        let system = builder.build_system_message();
        let runtime_pos = system.content.find("Runtime Context").unwrap();
        let memory_pos = system.content.find("## Memory").unwrap();
        assert!(runtime_pos < memory_pos);
    }

    #[test]
    fn test_build_messages_with_memory_override_some() {
        let builder = ContextBuilder::new()
            .with_memory_context("## Memory\n\n### Pinned\n- old: data".to_string());
        let override_ctx =
            "## Memory\n\n### Pinned\n- user:name: Alice\n\n### Relevant\n- fact:project: ZeptoClaw";
        let messages =
            builder.build_messages_with_memory_override(&[], "Hello", Some(override_ctx));
        assert!(messages[0].content.contains("fact:project: ZeptoClaw"));
        assert!(!messages[0].content.contains("old: data"));
        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn test_build_messages_with_memory_override_none_falls_back() {
        let builder = ContextBuilder::new()
            .with_memory_context("## Memory\n\n### Pinned\n- stored: value".to_string());
        let messages = builder.build_messages_with_memory_override(&[], "Hello", None);
        assert!(messages[0].content.contains("stored: value"));
    }

    #[test]
    fn test_build_messages_with_memory_override_empty_string() {
        let builder = ContextBuilder::new()
            .with_memory_context("## Memory\n\n### Pinned\n- old: data".to_string());
        let messages = builder.build_messages_with_memory_override(&[], "Hello", Some(""));
        assert!(!messages[0].content.contains("## Memory"));
    }

    // ---- Memory system prompt tests ----

    #[test]
    fn test_system_prompt_mentions_memory() {
        // DEFAULT_SYSTEM_PROMPT must reference longterm_memory so the LLM
        // knows the tool exists and uses it proactively.
        assert!(
            DEFAULT_SYSTEM_PROMPT.contains("longterm_memory"),
            "DEFAULT_SYSTEM_PROMPT should mention longterm_memory tool"
        );
    }

    #[test]
    fn test_system_prompt_mentions_save_and_recall() {
        // The prompt must tell the LLM to both save and recall information.
        assert!(
            DEFAULT_SYSTEM_PROMPT.contains("Save") || DEFAULT_SYSTEM_PROMPT.contains("save"),
            "DEFAULT_SYSTEM_PROMPT should mention saving facts"
        );
        assert!(
            DEFAULT_SYSTEM_PROMPT.contains("Recall") || DEFAULT_SYSTEM_PROMPT.contains("recall"),
            "DEFAULT_SYSTEM_PROMPT should mention recalling information"
        );
    }

    #[test]
    fn test_system_prompt_memory_instructions_in_built_message() {
        // Verify the memory instructions survive into the built system message.
        let builder = ContextBuilder::new();
        let system = builder.build_system_message();
        assert!(
            system.content.contains("longterm_memory"),
            "Built system message should contain longterm_memory instructions"
        );
    }

    #[test]
    fn test_system_prompt_contains_reminder_guidance() {
        assert!(
            DEFAULT_SYSTEM_PROMPT.contains("Reminder:"),
            "System prompt must instruct how to handle cron-delivered 'Reminder:' messages"
        );
    }

    #[test]
    fn test_system_prompt_contains_heartbeat_guidance() {
        assert!(
            DEFAULT_SYSTEM_PROMPT.contains("HEARTBEAT_OK"),
            "System prompt must instruct how to handle heartbeat messages"
        );
    }

    #[test]
    fn test_a2ui_suffix_contains_full_guidance() {
        assert!(
            A2UI_RENDERING_PROMPT_SUFFIX.contains("A2UI"),
            "A2UI suffix must include rendering guidance"
        );
        assert!(
            A2UI_RENDERING_PROMPT_SUFFIX.contains("Do NOT output Mermaid"),
            "A2UI suffix must discourage Mermaid fallback for UI rendering requests"
        );
        assert!(
            A2UI_RENDERING_PROMPT_SUFFIX.contains("```a2ui"),
            "A2UI suffix must include a valid ```a2ui shape example"
        );
        assert!(
            A2UI_RENDERING_PROMPT_SUFFIX.contains("Do NOT call `shell`"),
            "A2UI suffix must forbid using shell/python tools to render charts"
        );
    }

    #[test]
    fn test_default_prompt_excludes_a2ui_guidance() {
        // A2UI guidance is gated per-channel; the always-on default must
        // never carry it so non-rendering channels (Discord/CLI/etc) don't
        // receive instructions that would make the model emit raw JSON.
        assert!(
            !DEFAULT_SYSTEM_PROMPT.contains("```a2ui"),
            "Default prompt must not contain the A2UI block — gate it on capability instead"
        );
        assert!(
            !DEFAULT_SYSTEM_PROMPT.contains("A2UI Rendering (HARD RULE)"),
            "Default prompt must not contain the HARD RULE A2UI section"
        );
    }

    #[test]
    fn test_default_prompt_keeps_text_only_visual_guidance() {
        // Non-A2UI channels still need a hint to fall back to plain text /
        // markdown table rather than emitting JSON or mermaid.
        assert!(
            DEFAULT_SYSTEM_PROMPT.contains("text-only channel"),
            "Default prompt must include the text-only visual rendering note"
        );
    }

    #[test]
    fn test_prompt_size_budgets() {
        assert!(
            DEFAULT_SYSTEM_PROMPT.len() <= 768,
            "DEFAULT_SYSTEM_PROMPT is {} bytes; keep it <= 768",
            DEFAULT_SYSTEM_PROMPT.len()
        );
        assert!(
            A2UI_RENDERING_PROMPT_SUFFIX.len() <= 1700,
            "A2UI_RENDERING_PROMPT_SUFFIX is {} bytes; keep it <= 1700",
            A2UI_RENDERING_PROMPT_SUFFIX.len()
        );
    }

    #[test]
    fn test_build_messages_with_a2ui_capability_includes_suffix() {
        let builder = ContextBuilder::new();
        let messages = builder.build_messages_with_overrides(
            &[],
            "draw a chart",
            None,
            PromptCapabilities::with_a2ui(),
        );
        assert!(messages[0].content.contains("```a2ui"));
        assert!(messages[0].content.contains("A2UI Rendering (HARD RULE)"));
    }

    #[test]
    fn test_build_messages_without_capability_omits_suffix() {
        let builder = ContextBuilder::new();
        let messages = builder.build_messages_with_overrides(
            &[],
            "draw a chart",
            None,
            PromptCapabilities::default(),
        );
        assert!(!messages[0].content.contains("```a2ui"));
        assert!(!messages[0].content.contains("A2UI Rendering (HARD RULE)"));
    }

    #[test]
    fn test_legacy_build_messages_with_memory_override_omits_a2ui() {
        // Backwards-compatibility: callers that don't opt in must default
        // to a2ui_capable=false so non-rendering channels stay clean.
        let builder = ContextBuilder::new();
        let messages = builder.build_messages_with_memory_override(&[], "Hello", None);
        assert!(!messages[0].content.contains("```a2ui"));
    }

    #[test]
    fn test_with_system_prompt_suffix() {
        let builder = ContextBuilder::new().with_system_prompt_suffix("\n\nExtra instructions.");
        let system = builder.build_system_message();
        assert!(system.content.contains("ZeptoClaw"));
        assert!(system.content.contains("Extra instructions."));
    }

    #[test]
    fn test_first_run_persona_prompt_content() {
        assert!(FIRST_RUN_PERSONA_PROMPT.contains("First Conversation Setup"));
        assert!(FIRST_RUN_PERSONA_PROMPT.contains("concise"));
        assert!(FIRST_RUN_PERSONA_PROMPT.contains("persona_pref"));
    }

    // ── P5.1 anchored rolling summary slot ────────────────────────────

    #[test]
    fn p5_anchored_summary_default_is_none() {
        let builder = ContextBuilder::new();
        assert!(builder.anchored_summary().is_none());
    }

    #[test]
    fn p5_with_anchored_summary_sets_and_clears() {
        let builder = ContextBuilder::new().with_anchored_summary(Some("sum-text".into()));
        assert_eq!(builder.anchored_summary(), Some("sum-text"));

        let cleared = builder.with_anchored_summary(None);
        assert!(cleared.anchored_summary().is_none());
    }

    #[test]
    fn p5_build_with_overrides_none_summary_matches_baseline() {
        // Zero-behavior-change invariant: an unset anchored_summary must
        // produce the exact same message slice as a pre-P5.1 build. The
        // surrounding test suite is the regression net; this case just
        // pins the byte-for-byte equivalence for callers reading the
        // module.
        let history = vec![Message::user("hi"), Message::assistant("hello")];
        let baseline = ContextBuilder::new().build_messages_with_overrides(
            &history,
            "next",
            None,
            PromptCapabilities::default(),
        );
        let with_none = ContextBuilder::new()
            .with_anchored_summary(None)
            .build_messages_with_overrides(
                &history,
                "next",
                None,
                PromptCapabilities::default(),
            );

        assert_eq!(baseline.len(), with_none.len());
        for (a, b) in baseline.iter().zip(with_none.iter()) {
            assert_eq!(a.role, b.role);
            assert_eq!(a.content, b.content);
        }
    }

    #[test]
    fn p5_build_with_overrides_inserts_summary_between_system_and_history() {
        let history = vec![Message::user("hi"), Message::assistant("hello")];
        let messages = ContextBuilder::new()
            .with_anchored_summary(Some("[summary] previous turns covered X, Y, Z.".into()))
            .build_messages_with_overrides(
                &history,
                "next",
                None,
                PromptCapabilities::default(),
            );

        // Expect: [system, summary(system), user(hi), assistant(hello), user(next)]
        assert_eq!(messages.len(), 5);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[1].role, Role::System);
        assert_eq!(
            messages[1].content,
            "[summary] previous turns covered X, Y, Z."
        );
        assert_eq!(messages[2].role, Role::User);
        assert_eq!(messages[2].content, "hi");
        assert_eq!(messages[3].role, Role::Assistant);
        assert_eq!(messages[4].role, Role::User);
        assert!(messages[4].content.ends_with("next"));
    }

    #[test]
    fn p5_legacy_build_messages_ignores_summary() {
        // P5.1 only wires the summary slot into `_with_overrides`. The
        // simplified `build_messages` entry point is unused by the
        // production path (Harness goes through `_with_overrides`) and
        // is intentionally left alone for now. This test pins that
        // contract so future readers don't change it accidentally.
        let history = vec![Message::user("hi")];
        let messages = ContextBuilder::new()
            .with_anchored_summary(Some("ignored in legacy path".into()))
            .build_messages(&history, "next");

        // Just the system message, history, and user input — no summary.
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[1].role, Role::User);
        assert_eq!(messages[1].content, "hi");
        assert_eq!(messages[2].role, Role::User);
        assert!(messages[2].content.ends_with("next"));
    }
}
