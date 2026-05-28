use async_trait::async_trait;
use serde_json::{json, Value};

use crate::error::Result;

use super::{Tool, ToolCategory, ToolContext, ToolOutput};

/// Meta-tool used by the harness to let the model externalize a turn plan.
pub struct ProposePlanTool;

#[async_trait]
impl Tool for ProposePlanTool {
    fn name(&self) -> &str {
        "propose_plan"
    }

    fn description(&self) -> &str {
        "Propose an optional multi-step plan for this turn when the path is not obvious. Use this only when planning would help; deterministic workflows can proceed without it."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "subtasks": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": "A concise description of this subtask."
                            },
                            "acceptance": {
                                "type": "string",
                                "description": "How you will know this subtask is done."
                            },
                            "tool_budget": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 15,
                                "default": 5,
                                "description": "Suggested number of real tool calls for this subtask."
                            }
                        },
                        "required": ["description", "acceptance"],
                        "additionalProperties": false
                    }
                },
                "rationale": {
                    "type": "string",
                    "description": "Why planning is useful now."
                }
            },
            "required": ["subtasks"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, _args: Value, _ctx: &ToolContext) -> Result<ToolOutput> {
        Ok(ToolOutput::error(
            "propose_plan is a harness meta-tool and must not be executed directly.",
        ))
    }

    fn category(&self) -> ToolCategory {
        // The harness intercepts this meta-tool before execution; Messaging is
        // the closest existing category without widening the tool taxonomy.
        ToolCategory::Messaging
    }
}
