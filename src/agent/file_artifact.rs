//! File artifact detection and event publishing extracted from `agent::loop`.
//!
//! Phase 4 alternative helper extraction: mechanical move only.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::Serialize;

use crate::agent::agui_events;
use crate::bus::MessageBus;
use crate::tools::ToolContext;

use super::loop_events::{publish_custom_ui_event, supports_custom_ui_channel};

static FILE_ARTIFACT_EVENT_SEQ: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy)]
pub(super) enum FileArtifactOperation {
    Write,
    Edit,
}

#[derive(Debug, Clone)]
pub(super) struct FileArtifactCandidate {
    pub(super) raw_path: String,
    pub(super) existed_before: bool,
    pub(super) operation: FileArtifactOperation,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct FileArtifactPayload {
    pub(super) event_id: String,
    pub(super) path: String,
    pub(super) name: String,
    pub(super) size_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) mime: Option<String>,
    pub(super) operation: &'static str,
}

pub(super) fn resolve_workspace_path(workspace: &Path, raw_path: &str) -> PathBuf {
    let path = Path::new(raw_path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        workspace.join(path)
    }
}

pub(super) fn prepare_file_artifact_candidate(
    tool_name: &str,
    args: &serde_json::Value,
    ctx: &ToolContext,
) -> Option<FileArtifactCandidate> {
    if !ctx
        .channel
        .as_deref()
        .is_some_and(supports_custom_ui_channel)
    {
        return None;
    }
    let workspace = Path::new(ctx.workspace.as_deref()?);
    let raw_path = args.get("path")?.as_str()?.to_string();
    let operation = match tool_name {
        "write_file" => FileArtifactOperation::Write,
        "edit_file" => FileArtifactOperation::Edit,
        _ => return None,
    };
    let full_path = resolve_workspace_path(workspace, &raw_path);
    let existed_before = full_path.exists();
    Some(FileArtifactCandidate {
        raw_path,
        existed_before,
        operation,
    })
}

pub(super) fn infer_file_mime(path: &Path) -> Option<&'static str> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    Some(match ext.as_str() {
        "md" => "text/markdown",
        "txt" => "text/plain",
        "json" => "application/json",
        "py" => "text/x-python",
        "rs" => "text/plain",
        "toml" => "application/toml",
        "yaml" | "yml" => "application/yaml",
        "html" | "htm" => "text/html",
        "js" => "text/javascript",
        "ts" | "tsx" => "text/plain",
        "css" => "text/css",
        "csv" => "text/csv",
        _ => return None,
    })
}

pub(super) fn build_file_artifact_payload(
    candidate: &FileArtifactCandidate,
    ctx: &ToolContext,
) -> Option<FileArtifactPayload> {
    let workspace = Path::new(ctx.workspace.as_deref()?);
    let workspace_canon = workspace.canonicalize().ok()?;
    let full_path = resolve_workspace_path(workspace, &candidate.raw_path);
    let full_path_canon = full_path.canonicalize().ok()?;
    if !full_path_canon.starts_with(&workspace_canon) {
        return None;
    }
    let metadata = std::fs::metadata(&full_path_canon).ok()?;
    if !metadata.is_file() {
        return None;
    }
    let relative = full_path_canon
        .strip_prefix(&workspace_canon)
        .ok()?
        .to_string_lossy()
        .replace('\\', "/");
    let name = full_path_canon.file_name()?.to_string_lossy().to_string();
    let operation = match candidate.operation {
        FileArtifactOperation::Edit => "modified",
        FileArtifactOperation::Write => {
            if candidate.existed_before {
                "modified"
            } else {
                "created"
            }
        }
    };
    Some(FileArtifactPayload {
        event_id: format!(
            "file_{}",
            FILE_ARTIFACT_EVENT_SEQ.fetch_add(1, Ordering::Relaxed)
        ),
        path: relative,
        name,
        size_bytes: metadata.len(),
        mime: infer_file_mime(&full_path_canon).map(str::to_string),
        operation,
    })
}

pub(super) async fn publish_file_artifact_event(
    bus: &Arc<MessageBus>,
    ctx: &ToolContext,
    payload: &FileArtifactPayload,
) {
    publish_custom_ui_event(
        bus,
        ctx.channel.as_deref(),
        ctx.chat_id.as_deref(),
        agui_events::FILE_ARTIFACT,
        payload,
        Some(&format!("[file] {} {}", payload.operation, payload.path)),
    )
    .await;
}
