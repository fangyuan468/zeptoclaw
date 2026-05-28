use async_trait::async_trait;
use serde_json::{json, Value};

use crate::error::Result;

use super::{Tool, ToolCategory, ToolContext, ToolOutput};

/// Meta-tool used by the harness as the explicit turn termination signal.
pub struct FinalAnswerTool;

#[async_trait]
impl Tool for FinalAnswerTool {
    fn name(&self) -> &str {
        "final_answer"
    }

    fn description(&self) -> &str {
        "Yield the final user-visible answer for this turn. Call this when you have enough information to answer the user. Do not emit the answer as free text once you have started calling other tools."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Final answer text. Plain natural language; no XML, no provider tool markup."
                },
                "status": {
                    "type": "string",
                    "enum": ["complete", "partial", "blocked"],
                    "description": "complete: task finished. partial: partially finished and explains the gap. blocked: blocked by permission, approval, tooling, network, or budget."
                }
            },
            "required": ["content", "status"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, _args: Value, _ctx: &ToolContext) -> Result<ToolOutput> {
        Ok(ToolOutput::error(
            "final_answer is a harness meta-tool and must not be executed directly.",
        ))
    }

    fn category(&self) -> ToolCategory {
        // The harness intercepts this meta-tool before execution; Messaging is
        // the closest existing category without widening the tool taxonomy.
        ToolCategory::Messaging
    }
}
