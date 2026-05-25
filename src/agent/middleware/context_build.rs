//! Middleware for building the resolved message list, tool definitions,
//! and chat options.
//!
//! Assembles the full message list from the session history + system prompt
//! (via `ContextBuilder`), resolves image file paths to base64, filters
//! empty messages, and populates `ctx.messages`, `ctx.tool_definitions`,
//! and `ctx.chat_options`.

use async_trait::async_trait;

use super::{Middleware, PipelineContext, PipelineOutput};
use crate::agent::pipeline::Next;
use crate::error::Result;
use crate::providers::ChatOptions;
use crate::session::Role;

const LAZY_TOOL_SCHEMA_SYSTEM_HINT: &str = r#"## Lazy Tool Schema Usage

When tool schemas are lazy-loaded, use these common tools directly when their arguments are obvious:
- internal__web_search: {"query": "...", "count": 10}
- internal__web_fetch: {"url": "https://..."}
- internal__shell: {"command": "..."}

Before using a tool for the first time when you are not certain about its exact argument names or types, call internal__get_tool_schema with the tool's exposed name. If a tool returns Invalid args, call internal__get_tool_schema immediately and retry with the schema."#;

/// Builds the resolved messages, tool definitions, and chat options.
///
/// Requires `ctx.session` to be populated (runs after session + compaction).
/// After this middleware:
/// - `ctx.messages` contains the full message list for the LLM call
/// - `ctx.tool_definitions` has the tool schemas
/// - `ctx.chat_options` is configured from agent defaults
#[derive(Debug)]
pub struct ContextBuildMiddleware;

impl ContextBuildMiddleware {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ContextBuildMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Middleware for ContextBuildMiddleware {
    fn name(&self) -> &'static str {
        "context_build"
    }

    async fn handle(&self, ctx: &mut PipelineContext, next: Next<'_>) -> Result<PipelineOutput> {
        // Build messages from session history + system prompt + memory override.
        if let Some(ref session) = ctx.session {
            let mut msgs = ctx
                .subsystems
                .context_builder
                .build_messages_with_memory_override(
                    &session.messages,
                    "", // user input already in session
                    ctx.memory_override.as_deref(),
                );

            // TODO(phase-4): Resolve image file paths to base64.
            // The `resolve_images_to_base64` utility currently lives as a
            // private function in loop.rs.  Phase 4 will extract it and
            // wire it here.  For now, image-bearing messages pass through
            // with their original ImageSource representation.

            // Filter out empty user messages (e.g. after failed image resolution).
            msgs.retain(|m| !(m.role == Role::User && m.content.is_empty() && !m.has_images()));

            if ctx.config.agents.defaults.lazy_tool_schema {
                append_lazy_tool_schema_hint(&mut msgs);
            }

            ctx.messages = Some(msgs);
        }

        // Get tool definitions (short-lived read lock).
        let tool_definitions = {
            let tools = ctx.subsystems.tools.read().await;
            tools.definitions_for_mode(
                ctx.config.agents.defaults.lazy_tool_schema,
                ctx.config.agents.defaults.compact_tools,
            )
        };
        ctx.tool_definitions = Some(tool_definitions);

        // Build chat options from config defaults.
        let options = ChatOptions::new()
            .with_max_tokens(ctx.config.agents.defaults.max_tokens)
            .with_temperature(ctx.config.agents.defaults.temperature);
        ctx.chat_options = Some(options);

        next.run(ctx).await
    }
}

fn append_lazy_tool_schema_hint(messages: &mut [crate::session::Message]) {
    if let Some(system) = messages.iter_mut().find(|m| m.role == Role::System) {
        system.content.push_str("\n\n");
        system.content.push_str(LAZY_TOOL_SCHEMA_SYSTEM_HINT);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::middleware::test_helpers::*;
    use crate::session::{Message, Session};
    #[tokio::test]
    async fn builds_messages_from_session() {
        let mw = ContextBuildMiddleware::new();
        let terminal = MockTerminal::with_response("ok");
        let pipeline = crate::agent::pipeline::Pipeline::builder()
            .add(mw)
            .build(terminal);

        let subsystems = test_subsystems();
        let mut ctx = test_context(subsystems);
        let mut session = Session::new("test");
        session.add_message(Message::user("Hello"));
        ctx.session = Some(session);

        let _ = pipeline.execute(&mut ctx).await.unwrap();

        let messages = ctx.messages.as_ref().expect("messages should be set");
        // Should have at least the system message + user message
        assert!(
            messages.len() >= 2,
            "Expected >= 2 messages, got {}",
            messages.len()
        );
        // First message should be system
        assert_eq!(messages[0].role, Role::System);
    }

    #[tokio::test]
    async fn sets_tool_definitions() {
        let mw = ContextBuildMiddleware::new();
        let terminal = MockTerminal::with_response("ok");
        let pipeline = crate::agent::pipeline::Pipeline::builder()
            .add(mw)
            .build(terminal);

        let subsystems = test_subsystems();
        let mut ctx = test_context(subsystems);
        ctx.session = Some(Session::new("test"));

        let _ = pipeline.execute(&mut ctx).await.unwrap();
        assert!(ctx.tool_definitions.is_some());
    }

    #[tokio::test]
    async fn sets_chat_options() {
        let mw = ContextBuildMiddleware::new();
        let terminal = MockTerminal::with_response("ok");
        let pipeline = crate::agent::pipeline::Pipeline::builder()
            .add(mw)
            .build(terminal);

        let subsystems = test_subsystems();
        let mut ctx = test_context(subsystems);
        ctx.session = Some(Session::new("test"));

        let _ = pipeline.execute(&mut ctx).await.unwrap();
        assert!(ctx.chat_options.is_some());
    }

    #[tokio::test]
    async fn no_session_skips_messages() {
        let mw = ContextBuildMiddleware::new();
        let terminal = MockTerminal::with_response("ok");
        let pipeline = crate::agent::pipeline::Pipeline::builder()
            .add(mw)
            .build(terminal);

        let subsystems = test_subsystems();
        let mut ctx = test_context(subsystems);
        // session is None

        let _ = pipeline.execute(&mut ctx).await.unwrap();
        // messages not built, but tool_definitions and chat_options should still be set
        assert!(ctx.messages.is_none());
        assert!(ctx.tool_definitions.is_some());
        assert!(ctx.chat_options.is_some());
    }

    #[tokio::test]
    async fn memory_override_included() {
        let mw = ContextBuildMiddleware::new();
        let terminal = MockTerminal::with_response("ok");
        let pipeline = crate::agent::pipeline::Pipeline::builder()
            .add(mw)
            .build(terminal);

        let subsystems = test_subsystems();
        let mut ctx = test_context(subsystems);
        ctx.memory_override = Some("User prefers dark mode".to_string());
        let mut session = Session::new("test");
        session.add_message(Message::user("What are my preferences?"));
        ctx.session = Some(session);

        let _ = pipeline.execute(&mut ctx).await.unwrap();

        let messages = ctx.messages.as_ref().expect("messages should be set");
        // The system message should contain the memory override
        let system_content = &messages[0].content;
        assert!(
            system_content.contains("dark mode"),
            "System prompt should include memory override, got: {}",
            &system_content[..system_content.len().min(200)]
        );
    }

    #[tokio::test]
    async fn lazy_tool_schema_hint_included_when_enabled() {
        let mw = ContextBuildMiddleware::new();
        let terminal = MockTerminal::with_response("ok");
        let pipeline = crate::agent::pipeline::Pipeline::builder()
            .add(mw)
            .build(terminal);

        let subsystems = test_subsystems();
        let mut ctx = test_context(subsystems);
        let mut config = crate::config::Config::default();
        config.agents.defaults.lazy_tool_schema = true;
        ctx.config = std::sync::Arc::new(config);
        let mut session = Session::new("test");
        session.add_message(Message::user("Search something"));
        ctx.session = Some(session);

        let _ = pipeline.execute(&mut ctx).await.unwrap();

        let messages = ctx.messages.as_ref().expect("messages should be set");
        let system_content = &messages[0].content;
        assert!(system_content.contains("Lazy Tool Schema Usage"));
        assert!(system_content.contains("internal__get_tool_schema"));
        assert!(system_content.contains("internal__web_search"));
    }
}
