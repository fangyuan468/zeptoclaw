//! Tool registry for ZeptoClaw
//!
//! This module provides the `ToolRegistry` struct for managing and executing tools.
//! Tools can be registered, looked up by name, and executed with context.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use serde_json::json;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tracing::{error, info};

use crate::error::Result;
use crate::providers::ToolDefinition;

use super::{Tool, ToolContext, ToolOutput};

const GET_TOOL_SCHEMA_NAME: &str = "get_tool_schema";
const GET_TOOL_SCHEMA_EXPOSED_NAME: &str = "internal__get_tool_schema";
const MAX_EXPOSED_TOOL_NAME_LEN: usize = 64;

/// The kind of tool behind an exposed lazy-schema name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolHandleKind {
    Internal,
    Mcp,
}

/// A provider-safe tool name mapped back to the registry's real tool key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolHandle {
    pub exposed_name: String,
    pub tool_name: String,
    pub kind: ToolHandleKind,
}

/// Returns a setup hint for tools that are opt-in (not registered by default).
fn opt_in_tool_hint(name: &str) -> &'static str {
    match name {
        "grep" | "find" => {
            " (coding tool — enable with `--template coder` or set `tools.coding_tools: true` in config)"
        }
        _ => "",
    }
}

/// A registry that holds and manages tools.
///
/// The registry allows tools to be registered, looked up by name,
/// and executed with proper logging and error handling.
///
/// # Example
///
/// ```rust
/// use zeptoclaw::tools::{ToolRegistry, EchoTool};
/// use serde_json::json;
///
/// # tokio_test::block_on(async {
/// let mut registry = ToolRegistry::new();
/// registry.register(Box::new(EchoTool));
///
/// assert!(registry.has("echo"));
///
/// let result = registry.execute("echo", json!({"message": "hello"})).await;
/// assert!(result.is_ok());
/// # });
/// ```
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    /// Create a new empty tool registry.
    ///
    /// # Example
    /// ```
    /// use zeptoclaw::tools::ToolRegistry;
    ///
    /// let registry = ToolRegistry::new();
    /// assert_eq!(registry.names().len(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a new tool in the registry.
    ///
    /// If a tool with the same name already exists, it will be replaced.
    ///
    /// # Arguments
    /// * `tool` - The tool to register
    ///
    /// # Example
    /// ```
    /// use zeptoclaw::tools::{ToolRegistry, EchoTool};
    ///
    /// let mut registry = ToolRegistry::new();
    /// registry.register(Box::new(EchoTool));
    /// assert!(registry.has("echo"));
    /// ```
    pub fn register(&mut self, tool: Box<dyn Tool>) {
        let name = tool.name().to_string();
        info!(tool = %name, "Registering tool");
        self.tools.insert(name, tool);
    }

    /// Get a tool by name.
    ///
    /// # Arguments
    /// * `name` - The name of the tool to retrieve
    ///
    /// # Returns
    /// A reference to the tool if found, or `None` if not found.
    ///
    /// # Example
    /// ```
    /// use zeptoclaw::tools::{ToolRegistry, EchoTool};
    ///
    /// let mut registry = ToolRegistry::new();
    /// registry.register(Box::new(EchoTool));
    ///
    /// let tool = registry.get("echo");
    /// assert!(tool.is_some());
    /// assert_eq!(tool.unwrap().name(), "echo");
    /// ```
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    /// Execute a tool by name with default context.
    ///
    /// # Arguments
    /// * `name` - The name of the tool to execute
    /// * `args` - The JSON arguments for the tool
    ///
    /// # Returns
    /// A `ToolOutput` with dual-audience content, or an error if execution fails.
    /// Tool-not-found returns `Ok(ToolOutput::error(...))`.
    ///
    /// # Example
    /// ```
    /// use zeptoclaw::tools::{ToolRegistry, EchoTool};
    /// use serde_json::json;
    ///
    /// # tokio_test::block_on(async {
    /// let mut registry = ToolRegistry::new();
    /// registry.register(Box::new(EchoTool));
    ///
    /// let result = registry.execute("echo", json!({"message": "hello"})).await;
    /// assert!(result.is_ok());
    /// assert_eq!(result.unwrap().for_llm, "hello");
    /// # });
    /// ```
    pub async fn execute(&self, name: &str, args: Value) -> Result<ToolOutput> {
        self.execute_with_context(name, args, &ToolContext::default())
            .await
    }

    /// Execute a tool by name with a specific context.
    ///
    /// # Arguments
    /// * `name` - The name of the tool to execute
    /// * `args` - The JSON arguments for the tool
    /// * `ctx` - The execution context
    ///
    /// # Returns
    /// A `ToolOutput` with dual-audience content, or an error if execution fails.
    /// Tool-not-found returns `Ok(ToolOutput::error(...))`.
    ///
    /// # Example
    /// ```
    /// use zeptoclaw::tools::{ToolRegistry, ToolContext, EchoTool};
    /// use serde_json::json;
    ///
    /// # tokio_test::block_on(async {
    /// let mut registry = ToolRegistry::new();
    /// registry.register(Box::new(EchoTool));
    ///
    /// let ctx = ToolContext::new().with_channel("telegram", "123");
    /// let result = registry.execute_with_context("echo", json!({"message": "hi"}), &ctx).await;
    /// assert!(result.is_ok());
    /// # });
    /// ```
    pub async fn execute_with_context(
        &self,
        name: &str,
        args: Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput> {
        if name == GET_TOOL_SCHEMA_NAME || name == GET_TOOL_SCHEMA_EXPOSED_NAME {
            return Ok(self.execute_get_tool_schema(args));
        }

        let tool = match self.tools.get(name) {
            Some(t) => t,
            None => {
                let hint = opt_in_tool_hint(name);
                return Ok(ToolOutput::error(format!(
                    "Tool not found: {}{}",
                    name, hint
                )));
            }
        };

        let start = Instant::now();

        match tool.execute(args, ctx).await {
            Ok(output) => {
                info!(
                    tool = name,
                    duration_ms = start.elapsed().as_millis() as u64,
                    "Tool executed successfully"
                );
                Ok(output)
            }
            Err(e) => {
                error!(
                    tool = name,
                    error = %e,
                    duration_ms = start.elapsed().as_millis() as u64,
                    "Tool execution failed"
                );
                Err(e)
            }
        }
    }

    /// Get all tool definitions for use with LLM providers.
    ///
    /// This returns a list of `ToolDefinition` structs that can be passed
    /// to an LLM provider's chat method.
    ///
    /// # Returns
    /// A vector of tool definitions.
    ///
    /// # Example
    /// ```
    /// use zeptoclaw::tools::{ToolRegistry, EchoTool};
    ///
    /// let mut registry = ToolRegistry::new();
    /// registry.register(Box::new(EchoTool));
    ///
    /// let definitions = registry.definitions();
    /// assert_eq!(definitions.len(), 1);
    /// assert_eq!(definitions[0].name, "echo");
    /// ```
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|t| ToolDefinition {
                name: t.name().to_string(),
                description: t.description().to_string(),
                parameters: t.parameters(),
            })
            .collect()
    }

    /// Get tool definitions, optionally using compact descriptions.
    ///
    /// When `compact` is true, tools that override `compact_description()`
    /// will use their shorter descriptions, saving tokens for constrained contexts.
    pub fn definitions_with_options(&self, compact: bool) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|t| ToolDefinition {
                name: t.name().to_string(),
                description: if compact {
                    t.compact_description().to_string()
                } else {
                    t.description().to_string()
                },
                parameters: t.parameters(),
            })
            .collect()
    }

    /// Get tool definitions for the configured schema exposure mode.
    pub fn definitions_for_mode(&self, lazy_schema: bool, compact: bool) -> Vec<ToolDefinition> {
        if lazy_schema {
            self.definitions_lazy()
        } else {
            self.definitions_with_options(compact)
        }
    }

    /// Get lazy tool definitions using provider-safe exposed names and placeholder schemas.
    pub fn definitions_lazy(&self) -> Vec<ToolDefinition> {
        let mut definitions: Vec<ToolDefinition> = self
            .lazy_tool_handles()
            .into_iter()
            .filter_map(|handle| {
                self.tools
                    .get(&handle.tool_name)
                    .map(|tool| ToolDefinition {
                        name: handle.exposed_name,
                        description: tool.compact_description().to_string(),
                        parameters: placeholder_parameters(),
                    })
            })
            .collect();

        definitions.push(ToolDefinition {
            name: GET_TOOL_SCHEMA_EXPOSED_NAME.to_string(),
            description: "Get the full JSON Schema for a tool by its exposed name. Use this when a tool call fails validation or when arguments are unclear.".to_string(),
            parameters: get_tool_schema_parameters(),
        });

        definitions
    }

    /// Resolve a provider-facing lazy exposed name to the registry's real tool key.
    pub fn resolve_exposed(&self, name: &str) -> Option<ToolHandle> {
        if name == GET_TOOL_SCHEMA_EXPOSED_NAME {
            return Some(ToolHandle {
                exposed_name: GET_TOOL_SCHEMA_EXPOSED_NAME.to_string(),
                tool_name: GET_TOOL_SCHEMA_NAME.to_string(),
                kind: ToolHandleKind::Internal,
            });
        }

        self.lazy_tool_handles()
            .into_iter()
            .find(|handle| handle.exposed_name == name)
    }

    /// Return the exposed name for a real registry tool key.
    pub fn exposed_name_for_tool(&self, name: &str) -> Option<String> {
        if name == GET_TOOL_SCHEMA_NAME {
            return Some(GET_TOOL_SCHEMA_EXPOSED_NAME.to_string());
        }

        self.lazy_tool_handles()
            .into_iter()
            .find(|handle| handle.tool_name == name)
            .map(|handle| handle.exposed_name)
    }

    /// Return the real schema for an exposed lazy tool name.
    pub fn schema_for_exposed(&self, name: &str) -> Option<Value> {
        if name == GET_TOOL_SCHEMA_EXPOSED_NAME {
            return Some(get_tool_schema_parameters());
        }

        let handle = self.resolve_exposed(name)?;
        self.tools
            .get(&handle.tool_name)
            .map(|tool| tool.parameters())
    }

    /// Validate tool arguments for lazy-schema mode and return a fail-soft error if invalid.
    pub fn validate_tool_args_lazy(
        &self,
        name: &str,
        args: &Value,
        exposed_hint: &str,
    ) -> Option<ToolOutput> {
        let schema = if name == GET_TOOL_SCHEMA_NAME {
            get_tool_schema_parameters()
        } else {
            self.tools.get(name)?.parameters()
        };

        match validate_json_value(&schema, args, "") {
            Ok(()) => None,
            Err(reason) => Some(ToolOutput::error(format!(
                "Invalid args for tool '{}': {}. Run get_tool_schema('{}') for the spec.",
                name, reason, exposed_hint
            ))),
        }
    }

    /// Get tool definitions for specific tool names only.
    ///
    /// Returns definitions only for tools whose names are in the provided list.
    /// Tools not found in the registry are silently skipped.
    ///
    /// # Arguments
    /// * `names` - Slice of tool names to include
    ///
    /// # Returns
    /// A vector of tool definitions for the matching tools.
    ///
    /// # Example
    /// ```
    /// use zeptoclaw::tools::{ToolRegistry, EchoTool};
    ///
    /// let mut registry = ToolRegistry::new();
    /// registry.register(Box::new(EchoTool));
    ///
    /// let defs = registry.definitions_for_tools(&["echo"]);
    /// assert_eq!(defs.len(), 1);
    /// assert_eq!(defs[0].name, "echo");
    ///
    /// let empty = registry.definitions_for_tools(&["nonexistent"]);
    /// assert!(empty.is_empty());
    /// ```
    pub fn definitions_for_tools(&self, names: &[&str]) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .filter(|(key, _)| names.contains(&key.as_str()))
            .map(|(_, t)| ToolDefinition {
                name: t.name().to_string(),
                description: t.description().to_string(),
                parameters: t.parameters(),
            })
            .collect()
    }

    /// Get the names of all registered tools.
    ///
    /// # Returns
    /// A vector of tool names.
    ///
    /// # Example
    /// ```
    /// use zeptoclaw::tools::{ToolRegistry, EchoTool};
    ///
    /// let mut registry = ToolRegistry::new();
    /// registry.register(Box::new(EchoTool));
    ///
    /// let names = registry.names();
    /// assert!(names.contains(&"echo"));
    /// ```
    pub fn names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a tool exists in the registry.
    ///
    /// # Arguments
    /// * `name` - The name of the tool to check
    ///
    /// # Returns
    /// `true` if the tool exists, `false` otherwise.
    ///
    /// # Example
    /// ```
    /// use zeptoclaw::tools::{ToolRegistry, EchoTool};
    ///
    /// let mut registry = ToolRegistry::new();
    /// assert!(!registry.has("echo"));
    ///
    /// registry.register(Box::new(EchoTool));
    /// assert!(registry.has("echo"));
    /// ```
    pub fn has(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Get the number of registered tools.
    ///
    /// # Returns
    /// The number of tools in the registry.
    ///
    /// # Example
    /// ```
    /// use zeptoclaw::tools::{ToolRegistry, EchoTool};
    ///
    /// let mut registry = ToolRegistry::new();
    /// assert_eq!(registry.len(), 0);
    ///
    /// registry.register(Box::new(EchoTool));
    /// assert_eq!(registry.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if the registry is empty.
    ///
    /// # Returns
    /// `true` if no tools are registered, `false` otherwise.
    ///
    /// # Example
    /// ```
    /// use zeptoclaw::tools::{ToolRegistry, EchoTool};
    ///
    /// let mut registry = ToolRegistry::new();
    /// assert!(registry.is_empty());
    ///
    /// registry.register(Box::new(EchoTool));
    /// assert!(!registry.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Drain all tools from `other` into this registry, consuming the other registry.
    ///
    /// Tools in `other` that have the same name as tools in `self` will replace
    /// the existing tool.
    pub fn merge(&mut self, other: ToolRegistry) {
        self.tools.extend(other.tools);
    }

    fn execute_get_tool_schema(&self, args: Value) -> ToolOutput {
        let Some(name) = args.get("name").and_then(|v| v.as_str()) else {
            return ToolOutput::error(
                "Invalid args: missing string field 'name' for get_tool_schema".to_string(),
            );
        };

        let Some(schema) = self.schema_for_exposed(name) else {
            return ToolOutput::error(format!("Tool schema not found for exposed name: {}", name));
        };

        let payload = json!({
            "name": name,
            "parameters": schema,
        });
        let rendered =
            serde_json::to_string_pretty(&payload).unwrap_or_else(|_| payload.to_string());
        ToolOutput::llm_only(rendered)
    }

    fn lazy_tool_handles(&self) -> Vec<ToolHandle> {
        let mut tools: Vec<(&str, &dyn Tool)> = self
            .tools
            .iter()
            .map(|(name, tool)| (name.as_str(), tool.as_ref()))
            .collect();
        tools.sort_by_key(|(name, _)| *name);

        let mut used = HashSet::from([GET_TOOL_SCHEMA_EXPOSED_NAME.to_string()]);
        let mut handles = Vec::with_capacity(tools.len());

        for (tool_name, tool) in tools {
            let candidate = sanitize_exposed_name(&tool.lazy_exposed_name_candidate());
            let kind = if candidate.starts_with("mcp__") {
                ToolHandleKind::Mcp
            } else {
                ToolHandleKind::Internal
            };
            let exposed_name = unique_exposed_name(&candidate, tool_name, &mut used);
            handles.push(ToolHandle {
                exposed_name,
                tool_name: tool_name.to_string(),
                kind,
            });
        }

        handles
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

fn placeholder_parameters() -> Value {
    json!({
        "type": "object",
        "additionalProperties": true,
    })
}

fn get_tool_schema_parameters() -> Value {
    json!({
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The exposed tool name, for example internal__shell"
            }
        },
        "required": ["name"]
    })
}

fn sanitize_exposed_name(name: &str) -> String {
    let sanitized: String = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect();

    if sanitized.is_empty() {
        "tool".to_string()
    } else {
        sanitized
    }
}

fn unique_exposed_name(candidate: &str, tool_name: &str, used: &mut HashSet<String>) -> String {
    if candidate.len() <= MAX_EXPOSED_TOOL_NAME_LEN && !used.contains(candidate) {
        used.insert(candidate.to_string());
        return candidate.to_string();
    }

    let hash = short_hash(&format!("{}:{}", candidate, tool_name));
    let namespace = if candidate.starts_with("mcp__") {
        "mcp"
    } else {
        "internal"
    };
    let safe_tool = sanitize_exposed_name(tool_name);
    let reserved = namespace.len() + 2 + 2 + hash.len();
    let prefix_len = MAX_EXPOSED_TOOL_NAME_LEN.saturating_sub(reserved);
    let prefix: String = safe_tool.chars().take(prefix_len).collect();
    let mut fallback = format!("{}__{}__{}", namespace, prefix, hash);

    let mut collision_index = 1usize;
    while used.contains(&fallback) {
        let suffix = short_hash(&format!("{}:{}:{}", candidate, tool_name, collision_index));
        fallback = format!("{}__{}__{}", namespace, prefix, suffix);
        collision_index += 1;
    }

    used.insert(fallback.clone());
    fallback
}

fn short_hash(input: &str) -> String {
    let digest = Sha256::digest(input.as_bytes());
    hex::encode(&digest[..6])
}

fn validate_json_value(
    schema: &Value,
    value: &Value,
    path: &str,
) -> std::result::Result<(), String> {
    if let Some(allowed) = schema.get("enum").and_then(|v| v.as_array()) {
        if !allowed.iter().any(|item| item == value) {
            return Err(format!(
                "{}must be one of the declared enum values",
                path_prefix(path)
            ));
        }
    }

    if let Some(type_error) = validate_schema_type(schema, value, path) {
        return Err(type_error);
    }

    let treats_as_object = schema.get("properties").is_some() || schema.get("required").is_some();
    if schema_type_includes(schema, "object") || (schema.get("type").is_none() && treats_as_object)
    {
        let Some(object) = value.as_object() else {
            return Err(format!("{}must be an object", path_prefix(path)));
        };

        if let Some(required) = schema.get("required").and_then(|v| v.as_array()) {
            for field in required.iter().filter_map(|v| v.as_str()) {
                if !object.contains_key(field) {
                    return Err(format!(
                        "{}missing required field '{}'",
                        path_prefix(path),
                        field
                    ));
                }
            }
        }

        if let Some(properties) = schema.get("properties").and_then(|v| v.as_object()) {
            for (field, field_schema) in properties {
                if let Some(field_value) = object.get(field) {
                    validate_json_value(field_schema, field_value, &child_path(path, field))?;
                }
            }

            if schema.get("additionalProperties").and_then(|v| v.as_bool()) == Some(false) {
                for field in object.keys() {
                    if !properties.contains_key(field) {
                        return Err(format!(
                            "{}contains unknown field '{}'",
                            path_prefix(path),
                            field
                        ));
                    }
                }
            }
        }
    }

    if schema_type_includes(schema, "array") {
        let Some(items) = schema.get("items") else {
            return Ok(());
        };
        let Some(array) = value.as_array() else {
            return Err(format!("{}must be an array", path_prefix(path)));
        };
        for (index, item) in array.iter().enumerate() {
            validate_json_value(items, item, &indexed_path(path, index))?;
        }
    }

    Ok(())
}

fn validate_schema_type(schema: &Value, value: &Value, path: &str) -> Option<String> {
    let declared = schema.get("type")?;
    let valid = match declared {
        Value::String(kind) => value_matches_type(value, kind),
        Value::Array(kinds) => kinds
            .iter()
            .filter_map(|kind| kind.as_str())
            .any(|kind| value_matches_type(value, kind)),
        _ => true,
    };

    if valid {
        None
    } else {
        Some(format!("{}has the wrong type", path_prefix(path)))
    }
}

fn path_prefix(path: &str) -> String {
    if path.is_empty() {
        String::new()
    } else {
        format!("{} ", path)
    }
}

fn child_path(path: &str, field: &str) -> String {
    if path.is_empty() {
        field.to_string()
    } else {
        format!("{}.{}", path, field)
    }
}

fn indexed_path(path: &str, index: usize) -> String {
    if path.is_empty() {
        format!("[{}]", index)
    } else {
        format!("{}[{}]", path, index)
    }
}

fn schema_type_includes(schema: &Value, expected: &str) -> bool {
    match schema.get("type") {
        Some(Value::String(kind)) => kind == expected,
        Some(Value::Array(kinds)) => kinds.iter().any(|kind| kind.as_str() == Some(expected)),
        _ => false,
    }
}

fn value_matches_type(value: &Value, kind: &str) -> bool {
    match kind {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::EchoTool;
    use async_trait::async_trait;
    use serde_json::{json, Value};

    struct CandidateTool {
        name: &'static str,
        exposed: &'static str,
    }

    #[async_trait]
    impl Tool for CandidateTool {
        fn name(&self) -> &str {
            self.name
        }

        fn description(&self) -> &str {
            "Candidate tool"
        }

        fn parameters(&self) -> Value {
            json!({
                "type": "object",
                "properties": {
                    "value": {"type": "string"}
                },
                "required": ["value"]
            })
        }

        fn lazy_exposed_name_candidate(&self) -> String {
            self.exposed.to_string()
        }

        async fn execute(&self, _args: Value, _ctx: &ToolContext) -> Result<ToolOutput> {
            Ok(ToolOutput::llm_only("ok"))
        }
    }

    #[test]
    fn test_registry_new() {
        let registry = ToolRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_registry_default() {
        let registry = ToolRegistry::default();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_registry_register() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));

        assert!(registry.has("echo"));
        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
    }

    #[test]
    fn test_registry_get() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));

        let tool = registry.get("echo");
        assert!(tool.is_some());
        assert_eq!(tool.unwrap().name(), "echo");

        let missing = registry.get("nonexistent");
        assert!(missing.is_none());
    }

    #[tokio::test]
    async fn test_registry_register_and_execute() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));

        assert!(registry.has("echo"));

        let result = registry.execute("echo", json!({"message": "hello"})).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().for_llm, "hello");
    }

    #[tokio::test]
    async fn test_registry_execute_with_context() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));

        let ctx = ToolContext::new()
            .with_channel("telegram", "123456")
            .with_workspace("/tmp/test");

        let result = registry
            .execute_with_context("echo", json!({"message": "world"}), &ctx)
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap().for_llm, "world");
    }

    #[test]
    fn test_registry_definitions() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));

        let definitions = registry.definitions();
        assert_eq!(definitions.len(), 1);
        assert_eq!(definitions[0].name, "echo");
        assert_eq!(
            definitions[0].description,
            "Echoes back the provided message"
        );
        assert!(definitions[0].parameters.is_object());
    }

    #[test]
    fn test_registry_names() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));

        let names = registry.names();
        assert_eq!(names.len(), 1);
        assert!(names.contains(&"echo"));
    }

    #[tokio::test]
    async fn test_tool_not_found() {
        let registry = ToolRegistry::new();
        let result = registry.execute("nonexistent", json!({})).await;

        // Tool-not-found returns Ok(ToolOutput::error(...))
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.is_error);
        assert!(output.for_llm.contains("Tool not found: nonexistent"));
    }

    #[tokio::test]
    async fn test_registry_execute_missing_message() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));

        // Execute without message argument - should return default
        let result = registry.execute("echo", json!({})).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().for_llm, "(no message)");
    }

    #[tokio::test]
    async fn test_registry_execute_null_message() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));

        // Execute with null message - should return default
        let result = registry.execute("echo", json!({"message": null})).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().for_llm, "(no message)");
    }

    #[test]
    fn test_registry_replace_tool() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));
        registry.register(Box::new(EchoTool)); // Register again

        // Should still have only one tool
        assert_eq!(registry.len(), 1);
        assert!(registry.has("echo"));
    }

    #[test]
    fn test_definitions_for_tools() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));
        let defs = registry.definitions_for_tools(&["echo"]);
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "echo");

        let empty = registry.definitions_for_tools(&["nonexistent"]);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_definitions_for_tools_multiple() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));
        let defs = registry.definitions_for_tools(&["echo", "nonexistent"]);
        assert_eq!(defs.len(), 1);
    }

    #[test]
    fn test_definitions_with_options_normal() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));
        let defs = registry.definitions_with_options(false);
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].description, "Echoes back the provided message");
    }

    #[test]
    fn test_definitions_with_options_compact() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));
        let defs = registry.definitions_with_options(true);
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].description, "Echo message");
    }

    #[test]
    fn test_definitions_lazy_uses_exposed_names_and_placeholder_schema() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));

        let defs = registry.definitions_lazy();
        let echo = defs
            .iter()
            .find(|def| def.name == "internal__echo")
            .unwrap();
        assert_eq!(echo.description, "Echo message");
        assert_eq!(
            echo.parameters,
            json!({"type": "object", "additionalProperties": true})
        );
        assert!(defs
            .iter()
            .any(|def| def.name == "internal__get_tool_schema"));
    }

    #[test]
    fn test_resolve_exposed_returns_real_tool_name() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));

        let handle = registry.resolve_exposed("internal__echo").unwrap();
        assert_eq!(handle.tool_name, "echo");
        assert_eq!(handle.kind, ToolHandleKind::Internal);
    }

    #[test]
    fn test_exposed_names_are_sanitized_limited_and_unique() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(CandidateTool {
            name: "first_tool",
            exposed: "mcp__server.with.dots__tool/with/slashes",
        }));
        registry.register(Box::new(CandidateTool {
            name: "second_tool_with_a_very_long_name_that_forces_hash_fallback",
            exposed: "mcp__server.with.dots__tool/with/slashes",
        }));

        let defs = registry.definitions_lazy();
        let exposed: Vec<&str> = defs
            .iter()
            .filter(|def| def.name != "internal__get_tool_schema")
            .map(|def| def.name.as_str())
            .collect();
        assert_eq!(exposed.len(), 2);
        assert_ne!(exposed[0], exposed[1]);
        assert!(exposed.iter().all(|name| name.len() <= 64));
        assert!(exposed.iter().all(|name| name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')));
    }

    #[tokio::test]
    async fn test_get_tool_schema_returns_real_schema() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));

        let output = registry
            .execute("get_tool_schema", json!({"name": "internal__echo"}))
            .await
            .unwrap();
        assert!(!output.is_error);
        assert!(output.for_llm.contains("\"message\""));
        assert!(output.for_llm.contains("\"required\""));
    }

    #[test]
    fn test_validate_tool_args_lazy_fail_soft() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));

        let output = registry
            .validate_tool_args_lazy("echo", &json!({}), "internal__echo")
            .unwrap();
        assert!(output.is_error);
        assert!(output.for_llm.contains("missing required field 'message'"));
        assert!(output.for_llm.contains("get_tool_schema('internal__echo')"));
    }

    #[test]
    fn test_opt_in_tool_hint_grep() {
        assert!(opt_in_tool_hint("grep").contains("--template coder"));
    }

    #[test]
    fn test_opt_in_tool_hint_find() {
        assert!(opt_in_tool_hint("find").contains("--template coder"));
    }

    #[test]
    fn test_opt_in_tool_hint_unknown() {
        assert_eq!(opt_in_tool_hint("unknown"), "");
    }
}
