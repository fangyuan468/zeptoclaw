use async_trait::async_trait;
use serde_json::{json, Value};

use crate::error::Result;

use super::{Tool, ToolCategory, ToolContext, ToolOutput};

/// Meta-tool used by the harness to let the model update its active turn plan.
pub struct RevisePlanTool;

#[async_trait]
impl Tool for RevisePlanTool {
    fn name(&self) -> &str {
        "revise_plan"
    }

    fn description(&self) -> &str {
        "Revise the active turn plan: complete the current subtask, advance to another subtask, mark a subtask blocked, or replace the plan."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["complete_current", "advance_to", "mark_blocked", "replace_plan"]
                },
                "target_subtask_idx": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Required for advance_to and mark_blocked."
                },
                "new_subtasks": {
                    "type": "array",
                    "description": "Required for replace_plan; uses the same subtask shape as propose_plan.",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "acceptance": {"type": "string"},
                            "tool_budget": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 15,
                                "default": 5
                            }
                        },
                        "required": ["description", "acceptance"],
                        "additionalProperties": false
                    }
                },
                "reason": {
                    "type": "string",
                    "description": "Why this plan revision is appropriate."
                }
            },
            "required": ["decision", "reason"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, _args: Value, _ctx: &ToolContext) -> Result<ToolOutput> {
        Ok(ToolOutput::error(
            "revise_plan is a harness meta-tool and must not be executed directly.",
        ))
    }

    fn category(&self) -> ToolCategory {
        // The harness intercepts this meta-tool before execution; Messaging is
        // the closest existing category without widening the tool taxonomy.
        ToolCategory::Messaging
    }
}
