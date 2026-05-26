//! Tool runtime observations — phase 3 of the agent loop state-machine
//! refactor.
//!
//! The tool loop in `agent/loop.rs` previously returned `(String, String,
//! bool)` tuples — `(call_id, sanitized_content, pause_for_input)` — for
//! each executed tool call. That collapses several distinct outcomes into
//! one channel: a tool that succeeded, a tool that returned a soft error
//! the model can recover from, a hard runtime failure, and an
//! approval-gated rejection all looked identical to the loop driver.
//!
//! [`ToolObservation`] is the typed replacement. The wire format written
//! to the conversation (`Message::tool_result(call_id, content)`) is
//! deliberately unchanged: only the *internal* loop-driver type
//! evolves, so future phases can branch on
//! [`ToolObservationKind`] without rewriting persistence or transport.
//!
//! See `docs/plans/doing/agent-loop-state-machine-refactor.md` §5.5
//! (soft vs hard tool errors) and §9 phase 3 (runtime result
//! normalisation).

/// Coarse classification of how a single tool call concluded.
///
/// The variants intentionally mirror the four outcomes the agent loop
/// needs to react to differently — neither the LLM nor the persisted
/// transcript sees this enum directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ToolObservationKind {
    /// Tool executed cleanly. `Message::tool_result.content` reflects the
    /// model-visible payload exactly as the tool produced it.
    Success,
    /// Tool ran to completion but reported an in-band failure the model
    /// can be asked to recover from (e.g. `web_fetch` returning HTTP 404,
    /// a search returning zero results). The conversation continues; the
    /// failure is just useful evidence for the next model turn.
    SoftError,
    /// Tool failed in a way the model cannot recover from: timeout,
    /// panic, sandbox/permission violation, or an internal runtime error
    /// raised by `kernel::execute_tool` itself. The loop may still
    /// surface the message to the model but should not pretend the call
    /// succeeded for metrics/observability purposes.
    HardError,
    /// Tool was rejected before execution because the current agent mode
    /// requires approval (e.g. Assistant-mode Shell tools) or the
    /// approval gate denied the call. This is distinct from a hard
    /// error: the user / operator can re-issue the request with
    /// approval, so retry semantics and metrics differ.
    ApprovalRequired,
}

impl ToolObservationKind {
    /// Returns the kind as a stable string suitable for logs / metrics
    /// dimensions. The labels are part of the observability surface and
    /// must not change without a coordinated dashboard update.
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::SoftError => "soft_error",
            Self::HardError => "hard_error",
            Self::ApprovalRequired => "approval_required",
        }
    }

    /// Returns `true` for any non-success kind, so existing call sites
    /// that only care whether the call worked can keep using a bool
    /// without losing the finer-grained dimension.
    pub fn is_error(self) -> bool {
        !matches!(self, Self::Success)
    }
}

/// Result of executing one tool call from the agent loop's perspective.
///
/// `call_id`, `tool_name`, and `content` together still satisfy the
/// session wire format — callers persist results with
/// `Message::tool_result(&obs.call_id, &obs.content)` — while `kind`
/// and `elapsed_ms` feed loop control / metrics decisions that the
/// previous tuple representation could not express.
///
/// `pause_for_input` preserves the behaviour of the previous third
/// tuple field (some tools, e.g. approval prompts, asked the loop to
/// pause until the user responds).
#[derive(Debug, Clone)]
pub struct ToolObservation {
    pub call_id: String,
    pub tool_name: String,
    pub content: String,
    pub kind: ToolObservationKind,
    pub elapsed_ms: u64,
    pub pause_for_input: bool,
}

impl ToolObservation {
    /// Convenience constructor used by `agent/loop.rs` at the bottom of
    /// each tool execution closure. Kept tiny so the call sites in the
    /// loop stay readable.
    pub fn new(
        call_id: impl Into<String>,
        tool_name: impl Into<String>,
        content: impl Into<String>,
        kind: ToolObservationKind,
        elapsed_ms: u64,
        pause_for_input: bool,
    ) -> Self {
        Self {
            call_id: call_id.into(),
            tool_name: tool_name.into(),
            content: content.into(),
            kind,
            elapsed_ms,
            pause_for_input,
        }
    }

    /// Build an observation for a pre-execution rejection (hook block,
    /// agent-mode block, approval gate, dry-run, lazy schema lookup,
    /// etc.). These paths never actually invoked the underlying tool, so
    /// `elapsed_ms` is 0 and `pause_for_input` is `false`.
    pub fn pre_execution(
        call_id: impl Into<String>,
        tool_name: impl Into<String>,
        content: impl Into<String>,
        kind: ToolObservationKind,
    ) -> Self {
        Self::new(call_id, tool_name, content, kind, 0, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_as_label_is_stable() {
        assert_eq!(ToolObservationKind::Success.as_label(), "success");
        assert_eq!(ToolObservationKind::SoftError.as_label(), "soft_error");
        assert_eq!(ToolObservationKind::HardError.as_label(), "hard_error");
        assert_eq!(
            ToolObservationKind::ApprovalRequired.as_label(),
            "approval_required"
        );
    }

    #[test]
    fn kind_is_error_true_for_all_non_success() {
        assert!(!ToolObservationKind::Success.is_error());
        assert!(ToolObservationKind::SoftError.is_error());
        assert!(ToolObservationKind::HardError.is_error());
        assert!(ToolObservationKind::ApprovalRequired.is_error());
    }

    #[test]
    fn new_preserves_fields() {
        let obs = ToolObservation::new(
            "call_42",
            "web_fetch",
            "hello world",
            ToolObservationKind::Success,
            123,
            true,
        );
        assert_eq!(obs.call_id, "call_42");
        assert_eq!(obs.tool_name, "web_fetch");
        assert_eq!(obs.content, "hello world");
        assert_eq!(obs.kind, ToolObservationKind::Success);
        assert_eq!(obs.elapsed_ms, 123);
        assert!(obs.pause_for_input);
    }

    #[test]
    fn pre_execution_defaults_elapsed_and_pause() {
        let obs = ToolObservation::pre_execution(
            "call_1",
            "rm_rf",
            "blocked by hook",
            ToolObservationKind::HardError,
        );
        assert_eq!(obs.elapsed_ms, 0);
        assert!(!obs.pause_for_input);
        assert_eq!(obs.kind, ToolObservationKind::HardError);
        assert_eq!(obs.content, "blocked by hook");
    }
}
