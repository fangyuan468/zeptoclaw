//! Tool execution feedback types surfaced to CLI and other channels.
//!
//! Extracted verbatim from `agent::loop` as part of the Phase 4 alternative
//! helper module split. No behavior changes; this is a mechanical move.

/// Tool execution feedback event for CLI display.
#[derive(Debug, Clone)]
pub struct ToolFeedback {
    /// Name of the tool being executed.
    pub tool_name: String,
    /// Current phase of execution.
    pub phase: ToolFeedbackPhase,
    /// Raw JSON arguments for extracting display hints.
    pub args_json: Option<String>,
}

/// Phase of tool execution feedback.
#[derive(Debug, Clone)]
pub enum ToolFeedbackPhase {
    /// LLM is processing (shimmer should start).
    Thinking,
    /// LLM finished thinking (shimmer should stop).
    ThinkingDone,
    /// Tool execution is starting.
    Starting,
    /// Tool execution completed successfully.
    Done {
        /// Elapsed time in milliseconds.
        elapsed_ms: u64,
    },
    /// Tool execution failed.
    Failed {
        /// Elapsed time in milliseconds.
        elapsed_ms: u64,
        /// Error description.
        error: String,
    },
    /// All tool execution and LLM processing complete; final response follows.
    ResponseReady,
}
