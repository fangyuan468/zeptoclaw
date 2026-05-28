use std::collections::VecDeque;

use serde::Deserialize;

use super::observations::{ToolObservation, ToolObservationKind};

pub(super) const PROPOSE_PLAN_TOOL_NAME: &str = "propose_plan";
pub(super) const REVISE_PLAN_TOOL_NAME: &str = "revise_plan";

const DEFAULT_TOOL_BUDGET: u32 = 5;
const MAX_SUBTASKS: usize = 8;
const MAX_TOOL_BUDGET: u32 = 15;
const ERROR_WINDOW: usize = 5;

const DISORDER_ADVISORY: &str = "Heads up: this turn shows signs of churn, such as repeated tool calls, repeated errors, or uncertainty. If you have not planned, you may call propose_plan to structure the remaining steps. If you already have a plan, you may call revise_plan to change direction. Continuing without restructuring is acceptable if you are confident in the current approach. This message is informational, not a directive.";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum DisorderSignal {
    ConsecutiveSameTool,
    ToolErrorRate,
    LoopGuard,
    SelfDoubt,
    PlanStagnation,
    SubtaskBudget,
}

impl DisorderSignal {
    pub(super) fn as_label(self) -> &'static str {
        match self {
            Self::ConsecutiveSameTool => "consecutive_same_tool",
            Self::ToolErrorRate => "tool_error_rate",
            Self::LoopGuard => "loop_guard",
            Self::SelfDoubt => "self_doubt",
            Self::PlanStagnation => "plan_stagnation",
            Self::SubtaskBudget => "subtask_budget",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(super) struct ProposedSubtask {
    description: String,
    acceptance: String,
    #[serde(default = "default_tool_budget")]
    tool_budget: u32,
}

fn default_tool_budget() -> u32 {
    DEFAULT_TOOL_BUDGET
}

#[derive(Debug, Deserialize)]
struct ProposePlanArgs {
    subtasks: Vec<ProposedSubtask>,
    #[allow(dead_code)]
    rationale: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ReviseDecision {
    CompleteCurrent,
    AdvanceTo,
    MarkBlocked,
    ReplacePlan,
}

#[derive(Debug, Deserialize)]
struct RevisePlanArgs {
    decision: ReviseDecision,
    target_subtask_idx: Option<usize>,
    new_subtasks: Option<Vec<ProposedSubtask>>,
    #[allow(dead_code)]
    reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SubtaskStatus {
    Pending,
    InProgress,
    Completed,
    Blocked,
}

#[derive(Debug, Clone)]
struct Subtask {
    description: String,
    acceptance: String,
    tool_budget: u32,
    tool_calls_used: u32,
    budget_advisory_injected: bool,
    status: SubtaskStatus,
}

#[derive(Debug, Clone)]
struct TurnPlan {
    subtasks: Vec<Subtask>,
    current_subtask_idx: usize,
}

#[derive(Debug, Default)]
struct DisorderTracker {
    consecutive_same_tool: u32,
    last_tool_name: Option<String>,
    recent_observation_errors: VecDeque<bool>,
}

#[derive(Debug, Default)]
pub(super) struct HarnessTurnState {
    plan: Option<TurnPlan>,
    plan_last_revised_iteration: u32,
    disorder: DisorderTracker,
    disorder_advisory_pending: Option<DisorderSignal>,
    disorder_advisory_injected: bool,
    disorder_advisory_awaiting_response: bool,
}

impl HarnessTurnState {
    pub(super) fn handle_propose_plan(
        &mut self,
        call_id: impl Into<String>,
        args: &str,
        iteration: u32,
    ) -> ToolObservation {
        let call_id = call_id.into();
        if self.plan.is_some() {
            return ToolObservation::pre_execution(
                call_id,
                PROPOSE_PLAN_TOOL_NAME,
                "A plan is already active. Use revise_plan to update it.",
                ToolObservationKind::SoftError,
            );
        }

        match serde_json::from_str::<ProposePlanArgs>(args) {
            Ok(parsed) => {
                self.plan = Some(TurnPlan::new(parsed.subtasks));
                self.plan_last_revised_iteration = iteration;
                ToolObservation::pre_execution(
                    call_id,
                    PROPOSE_PLAN_TOOL_NAME,
                    self.render_plan_ack("Plan accepted."),
                    ToolObservationKind::Success,
                )
            }
            Err(error) => ToolObservation::pre_execution(
                call_id,
                PROPOSE_PLAN_TOOL_NAME,
                format!("Invalid propose_plan arguments: {error}"),
                ToolObservationKind::SoftError,
            ),
        }
    }

    pub(super) fn handle_revise_plan(
        &mut self,
        call_id: impl Into<String>,
        args: &str,
        iteration: u32,
    ) -> ToolObservation {
        let call_id = call_id.into();
        let Some(plan) = self.plan.as_mut() else {
            return ToolObservation::pre_execution(
                call_id,
                REVISE_PLAN_TOOL_NAME,
                "No plan is active. Use propose_plan before revise_plan.",
                ToolObservationKind::SoftError,
            );
        };

        let parsed = match serde_json::from_str::<RevisePlanArgs>(args) {
            Ok(parsed) => parsed,
            Err(error) => {
                return ToolObservation::pre_execution(
                    call_id,
                    REVISE_PLAN_TOOL_NAME,
                    format!("Invalid revise_plan arguments: {error}"),
                    ToolObservationKind::SoftError,
                );
            }
        };

        let result = match parsed.decision {
            ReviseDecision::CompleteCurrent => {
                plan.complete_current();
                Ok("Plan updated: current subtask completed.")
            }
            ReviseDecision::AdvanceTo => {
                let Some(idx) = parsed.target_subtask_idx else {
                    return ToolObservation::pre_execution(
                        call_id,
                        REVISE_PLAN_TOOL_NAME,
                        "Invalid revise_plan arguments: target_subtask_idx is required for advance_to.",
                        ToolObservationKind::SoftError,
                    );
                };
                plan.advance_to(idx)
                    .map(|()| "Plan updated: advanced to requested subtask.")
            }
            ReviseDecision::MarkBlocked => {
                let Some(idx) = parsed.target_subtask_idx else {
                    return ToolObservation::pre_execution(
                        call_id,
                        REVISE_PLAN_TOOL_NAME,
                        "Invalid revise_plan arguments: target_subtask_idx is required for mark_blocked.",
                        ToolObservationKind::SoftError,
                    );
                };
                plan.mark_blocked(idx)
                    .map(|()| "Plan updated: subtask marked blocked.")
            }
            ReviseDecision::ReplacePlan => {
                let Some(subtasks) = parsed.new_subtasks else {
                    return ToolObservation::pre_execution(
                        call_id,
                        REVISE_PLAN_TOOL_NAME,
                        "Invalid revise_plan arguments: new_subtasks is required for replace_plan.",
                        ToolObservationKind::SoftError,
                    );
                };
                *plan = TurnPlan::new(subtasks);
                Ok("Plan replaced.")
            }
        };

        match result {
            Ok(prefix) => {
                self.plan_last_revised_iteration = iteration;
                ToolObservation::pre_execution(
                    call_id,
                    REVISE_PLAN_TOOL_NAME,
                    self.render_plan_ack(prefix),
                    ToolObservationKind::Success,
                )
            }
            Err(message) => ToolObservation::pre_execution(
                call_id,
                REVISE_PLAN_TOOL_NAME,
                message,
                ToolObservationKind::SoftError,
            ),
        }
    }

    pub(super) fn render_plan_prompt(&self) -> Option<String> {
        self.plan.as_ref().map(|plan| {
            format!(
                "{}\n\nYou may continue the current subtask, call revise_plan to update progress or direction, or call final_answer when the overall plan is complete.",
                plan.render("Current plan:")
            )
        })
    }

    pub(super) fn take_disorder_advisory(&mut self) -> Option<(DisorderSignal, String)> {
        if self.disorder_advisory_injected {
            return None;
        }
        let signal = self.disorder_advisory_pending.take()?;
        let advisory = self.render_disorder_advisory(signal);
        self.disorder_advisory_injected = true;
        self.disorder_advisory_awaiting_response = true;
        Some((signal, advisory))
    }

    pub(super) fn note_llm_response_after_advisory(&mut self, used_plan_tool: bool) -> bool {
        if !self.disorder_advisory_awaiting_response {
            return false;
        }
        self.disorder_advisory_awaiting_response = false;
        !used_plan_tool
    }

    pub(super) fn record_tool_calls(&mut self, tool_names: &[String]) {
        if let Some(plan) = self.plan.as_mut() {
            if plan.record_tool_calls(tool_names.len() as u32) {
                self.mark_disorder(DisorderSignal::SubtaskBudget);
            }
        }

        for name in tool_names {
            if self.disorder.last_tool_name.as_deref() == Some(name.as_str()) {
                self.disorder.consecutive_same_tool += 1;
            } else {
                self.disorder.last_tool_name = Some(name.clone());
                self.disorder.consecutive_same_tool = 1;
            }

            if self.disorder.consecutive_same_tool >= 4 {
                self.mark_disorder(DisorderSignal::ConsecutiveSameTool);
            }
        }
    }

    pub(super) fn record_observations(&mut self, observations: &[ToolObservation]) {
        for obs in observations {
            if self.disorder.recent_observation_errors.len() == ERROR_WINDOW {
                self.disorder.recent_observation_errors.pop_front();
            }
            self.disorder
                .recent_observation_errors
                .push_back(obs.kind.is_error());
        }

        if self.disorder.recent_observation_errors.len() == ERROR_WINDOW {
            let errors = self
                .disorder
                .recent_observation_errors
                .iter()
                .filter(|is_error| **is_error)
                .count();
            if errors >= 3 {
                self.mark_disorder(DisorderSignal::ToolErrorRate);
            }
        }
    }

    pub(super) fn record_model_content(&mut self, content: &str) {
        let content = content.to_ascii_lowercase();
        let hit = [
            "let me reconsider",
            "i'm not sure how to proceed",
            "let me try a different",
            "actually, wait",
            "i'm stuck",
            "i don't know what to do next",
            "\u{8ba9}\u{6211}\u{91cd}\u{65b0}\u{8003}\u{8651}",
            "\u{6211}\u{4e0d}\u{786e}\u{5b9a}",
            "\u{8ba9}\u{6211}\u{6362}\u{4e2a}\u{601d}\u{8def}",
            "\u{6211}\u{6709}\u{70b9}\u{5361}\u{4f4f}",
            "\u{6211}\u{4e0d}\u{77e5}\u{9053}\u{4e0b}\u{4e00}\u{6b65}",
        ]
        .iter()
        .any(|needle| content.contains(needle));

        if hit {
            self.mark_disorder(DisorderSignal::SelfDoubt);
        }
    }

    pub(super) fn record_loop_guard_signal(&mut self) {
        self.mark_disorder(DisorderSignal::LoopGuard);
    }

    pub(super) fn check_plan_stagnation(&mut self, iteration: u32) {
        if self.plan.is_some() && iteration.saturating_sub(self.plan_last_revised_iteration) >= 6 {
            self.mark_disorder(DisorderSignal::PlanStagnation);
        }
    }

    fn mark_disorder(&mut self, signal: DisorderSignal) {
        if !self.disorder_advisory_injected && self.disorder_advisory_pending.is_none() {
            self.disorder_advisory_pending = Some(signal);
        }
    }

    fn render_disorder_advisory(&self, signal: DisorderSignal) -> String {
        match signal {
            DisorderSignal::SubtaskBudget => self
                .plan
                .as_ref()
                .and_then(TurnPlan::render_budget_advisory)
                .unwrap_or_else(|| DISORDER_ADVISORY.to_string()),
            _ => DISORDER_ADVISORY.to_string(),
        }
    }

    fn render_plan_ack(&self, prefix: &str) -> String {
        match &self.plan {
            Some(plan) => format!("{}\n\n{}", prefix, plan.render("Current plan:")),
            None => prefix.to_string(),
        }
    }
}

impl TurnPlan {
    fn new(subtasks: Vec<ProposedSubtask>) -> Self {
        let mut subtasks: Vec<Subtask> = subtasks
            .into_iter()
            .take(MAX_SUBTASKS)
            .map(Subtask::from)
            .collect();
        if subtasks.is_empty() {
            subtasks.push(Subtask::from(ProposedSubtask {
                description: "Complete the user's request.".to_string(),
                acceptance: "A useful final answer is ready.".to_string(),
                tool_budget: DEFAULT_TOOL_BUDGET,
            }));
        }
        subtasks[0].status = SubtaskStatus::InProgress;
        Self {
            subtasks,
            current_subtask_idx: 0,
        }
    }

    fn complete_current(&mut self) {
        if let Some(current) = self.subtasks.get_mut(self.current_subtask_idx) {
            current.status = SubtaskStatus::Completed;
        }
        if let Some(next_idx) = self
            .subtasks
            .iter()
            .position(|subtask| matches!(subtask.status, SubtaskStatus::Pending))
        {
            self.current_subtask_idx = next_idx;
            self.subtasks[next_idx].status = SubtaskStatus::InProgress;
        }
    }

    fn advance_to(&mut self, idx: usize) -> std::result::Result<(), String> {
        if idx >= self.subtasks.len() {
            return Err(format!(
                "Invalid revise_plan target: subtask index {idx} is out of range."
            ));
        }
        if let Some(current) = self.subtasks.get_mut(self.current_subtask_idx) {
            if matches!(current.status, SubtaskStatus::InProgress) {
                current.status = SubtaskStatus::Pending;
            }
        }
        self.current_subtask_idx = idx;
        self.subtasks[idx].status = SubtaskStatus::InProgress;
        Ok(())
    }

    fn mark_blocked(&mut self, idx: usize) -> std::result::Result<(), String> {
        if idx >= self.subtasks.len() {
            return Err(format!(
                "Invalid revise_plan target: subtask index {idx} is out of range."
            ));
        }
        self.subtasks[idx].status = SubtaskStatus::Blocked;
        if idx == self.current_subtask_idx {
            if let Some(next_idx) = self
                .subtasks
                .iter()
                .position(|subtask| matches!(subtask.status, SubtaskStatus::Pending))
            {
                self.current_subtask_idx = next_idx;
                self.subtasks[next_idx].status = SubtaskStatus::InProgress;
            }
        }
        Ok(())
    }

    fn record_tool_calls(&mut self, count: u32) -> bool {
        if let Some(current) = self.subtasks.get_mut(self.current_subtask_idx) {
            current.tool_calls_used = current.tool_calls_used.saturating_add(count);
            if current.tool_calls_used >= current.tool_budget && !current.budget_advisory_injected {
                current.budget_advisory_injected = true;
                return true;
            }
        }
        false
    }

    fn render_budget_advisory(&self) -> Option<String> {
        let current = self.subtasks.get(self.current_subtask_idx)?;
        Some(format!(
            "Subtask \"{}\" has used its declared tool budget ({} calls used, budget {}). If you have enough information, call revise_plan with decision=complete_current. If you need more work, you may continue, or call revise_plan with decision=replace_plan to update the budget. This is a suggestion, not a restriction.",
            current.description, current.tool_calls_used, current.tool_budget
        ))
    }

    fn render(&self, heading: &str) -> String {
        let mut lines = vec![heading.to_string()];
        for (idx, subtask) in self.subtasks.iter().enumerate() {
            let marker = match subtask.status {
                SubtaskStatus::Completed => "x",
                SubtaskStatus::InProgress => ">",
                SubtaskStatus::Blocked => "!",
                SubtaskStatus::Pending => " ",
            };
            let current = if idx == self.current_subtask_idx {
                format!(
                    " ({}/{} tool calls used)",
                    subtask.tool_calls_used, subtask.tool_budget
                )
            } else {
                String::new()
            };
            lines.push(format!(
                "- [{}] subtask {}{}: {} Acceptance: {}",
                marker,
                idx + 1,
                current,
                subtask.description,
                subtask.acceptance
            ));
        }
        lines.join("\n")
    }
}

impl From<ProposedSubtask> for Subtask {
    fn from(value: ProposedSubtask) -> Self {
        Self {
            description: value.description.trim().to_string(),
            acceptance: value.acceptance.trim().to_string(),
            tool_budget: value.tool_budget.clamp(1, MAX_TOOL_BUDGET),
            tool_calls_used: 0,
            budget_advisory_injected: false,
            status: SubtaskStatus::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn propose_plan_creates_renderable_plan() {
        let mut state = HarnessTurnState::default();
        let obs = state.handle_propose_plan(
            "call_plan",
            r#"{"subtasks":[{"description":"Search sources","acceptance":"Sources collected","tool_budget":2},{"description":"Compare","acceptance":"Table ready"}]}"#,
            1,
        );

        assert_eq!(obs.kind, ToolObservationKind::Success);
        let prompt = state.render_plan_prompt().expect("plan should render");
        assert!(prompt.contains("Current plan:"));
        assert!(prompt.contains("[>] subtask 1 (0/2 tool calls used): Search sources"));
        assert!(prompt.contains("[ ] subtask 2: Compare"));
    }

    #[test]
    fn revise_plan_complete_current_advances() {
        let mut state = HarnessTurnState::default();
        state.handle_propose_plan(
            "call_plan",
            r#"{"subtasks":[{"description":"One","acceptance":"Done"},{"description":"Two","acceptance":"Done"}]}"#,
            1,
        );

        let obs = state.handle_revise_plan(
            "call_revise",
            r#"{"decision":"complete_current","reason":"first done"}"#,
            2,
        );

        assert_eq!(obs.kind, ToolObservationKind::Success);
        let prompt = state.render_plan_prompt().expect("plan should render");
        assert!(prompt.contains("[x] subtask 1: One"));
        assert!(prompt.contains("[>] subtask 2 (0/5 tool calls used): Two"));
    }

    #[test]
    fn revise_plan_replace_plan_rebuilds_subtasks() {
        let mut state = HarnessTurnState::default();
        state.handle_propose_plan(
            "call_plan",
            r#"{"subtasks":[{"description":"Old","acceptance":"Done"}]}"#,
            1,
        );

        let obs = state.handle_revise_plan(
            "call_revise",
            r#"{"decision":"replace_plan","new_subtasks":[{"description":"New first","acceptance":"Ready","tool_budget":3},{"description":"New second","acceptance":"Ready"}],"reason":"direction changed"}"#,
            2,
        );

        assert_eq!(obs.kind, ToolObservationKind::Success);
        let prompt = state.render_plan_prompt().expect("plan should render");
        assert!(!prompt.contains("Old"));
        assert!(prompt.contains("[>] subtask 1 (0/3 tool calls used): New first"));
        assert!(prompt.contains("[ ] subtask 2: New second"));
    }

    #[test]
    fn consecutive_same_tool_marks_advisory() {
        let mut state = HarnessTurnState::default();
        for _ in 0..4 {
            state.record_tool_calls(&["web_search".to_string()]);
        }

        let advisory = state
            .take_disorder_advisory()
            .expect("disorder advisory should be pending");
        assert_eq!(advisory.0, DisorderSignal::ConsecutiveSameTool);
        assert!(advisory.1.contains("informational"));
    }

    #[test]
    fn tool_error_rate_marks_advisory() {
        let mut state = HarnessTurnState::default();
        let observations = vec![
            ToolObservation::pre_execution(
                "call_1",
                "web_fetch",
                "ok",
                ToolObservationKind::Success,
            ),
            ToolObservation::pre_execution(
                "call_2",
                "web_fetch",
                "error",
                ToolObservationKind::SoftError,
            ),
            ToolObservation::pre_execution(
                "call_3",
                "web_fetch",
                "error",
                ToolObservationKind::HardError,
            ),
            ToolObservation::pre_execution(
                "call_4",
                "web_fetch",
                "ok",
                ToolObservationKind::Success,
            ),
            ToolObservation::pre_execution(
                "call_5",
                "web_fetch",
                "approval",
                ToolObservationKind::ApprovalRequired,
            ),
        ];

        state.record_observations(&observations);

        assert_eq!(
            state
                .take_disorder_advisory()
                .expect("tool error rate should trigger")
                .0,
            DisorderSignal::ToolErrorRate
        );
    }

    #[test]
    fn plan_stagnation_marks_advisory() {
        let mut state = HarnessTurnState::default();
        state.handle_propose_plan(
            "call_plan",
            r#"{"subtasks":[{"description":"Search","acceptance":"Done"}]}"#,
            1,
        );

        state.check_plan_stagnation(7);

        assert_eq!(
            state
                .take_disorder_advisory()
                .expect("plan stagnation should trigger")
                .0,
            DisorderSignal::PlanStagnation
        );
    }

    #[test]
    fn subtask_budget_marks_advisory_when_plan_is_active() {
        let mut state = HarnessTurnState::default();
        state.handle_propose_plan(
            "call_plan",
            r#"{"subtasks":[{"description":"Search","acceptance":"Sources found","tool_budget":2}]}"#,
            1,
        );

        state.record_tool_calls(&["web_search".to_string()]);
        assert!(state.take_disorder_advisory().is_none());

        state.record_tool_calls(&["web_fetch".to_string()]);

        let advisory = state
            .take_disorder_advisory()
            .expect("subtask budget should trigger");
        assert_eq!(advisory.0, DisorderSignal::SubtaskBudget);
        assert!(advisory.1.contains("Subtask \"Search\""));
        assert!(advisory.1.contains("budget 2"));
        assert!(advisory.1.contains("suggestion, not a restriction"));
    }

    #[test]
    fn subtask_budget_does_not_repeat_for_same_subtask() {
        let mut state = HarnessTurnState::default();
        state.handle_propose_plan(
            "call_plan",
            r#"{"subtasks":[{"description":"Search","acceptance":"Sources found","tool_budget":1}]}"#,
            1,
        );

        state.record_tool_calls(&["web_search".to_string()]);
        state.record_tool_calls(&["web_fetch".to_string()]);

        assert_eq!(
            state
                .take_disorder_advisory()
                .expect("subtask budget should trigger once")
                .0,
            DisorderSignal::SubtaskBudget
        );
        assert!(state.take_disorder_advisory().is_none());
    }

    #[test]
    fn next_subtask_budget_starts_at_zero_after_complete_current() {
        let mut state = HarnessTurnState::default();
        state.handle_propose_plan(
            "call_plan",
            r#"{"subtasks":[{"description":"Search","acceptance":"Sources found","tool_budget":1},{"description":"Compare","acceptance":"Table ready","tool_budget":3}]}"#,
            1,
        );

        state.record_tool_calls(&["web_search".to_string()]);
        state.handle_revise_plan(
            "call_revise",
            r#"{"decision":"complete_current","reason":"sources found"}"#,
            2,
        );

        let prompt = state.render_plan_prompt().expect("plan should render");
        assert!(prompt.contains("[x] subtask 1: Search"));
        assert!(prompt.contains("[>] subtask 2 (0/3 tool calls used): Compare"));
    }

    #[test]
    fn subtask_budget_is_inactive_without_plan() {
        let mut state = HarnessTurnState::default();
        state.record_tool_calls(&["web_search".to_string(), "web_fetch".to_string()]);

        assert!(state.take_disorder_advisory().is_none());
    }

    #[test]
    fn advisory_is_only_injected_once_per_turn() {
        let mut state = HarnessTurnState::default();
        for _ in 0..4 {
            state.record_tool_calls(&["web_search".to_string()]);
        }

        assert!(state.take_disorder_advisory().is_some());
        state.record_loop_guard_signal();
        state.record_model_content("actually, wait");

        assert!(state.take_disorder_advisory().is_none());
    }

    #[test]
    fn self_doubt_matching_is_conservative() {
        let mut state = HarnessTurnState::default();
        state.record_model_content("I'm not sure if this URL is correct.");
        assert!(state.take_disorder_advisory().is_none());

        state.record_model_content("Actually, wait, let me reconsider the plan.");
        assert_eq!(
            state
                .take_disorder_advisory()
                .expect("self doubt should trigger")
                .0,
            DisorderSignal::SelfDoubt
        );
    }

    #[test]
    fn self_doubt_matching_supports_chinese_phrases() {
        let mut state = HarnessTurnState::default();
        state.record_model_content("\u{6211}\u{4e0d}\u{77e5}\u{9053}\u{4e0b}\u{4e00}\u{6b65}");

        assert_eq!(
            state
                .take_disorder_advisory()
                .expect("Chinese self-doubt should trigger")
                .0,
            DisorderSignal::SelfDoubt
        );
    }
}
