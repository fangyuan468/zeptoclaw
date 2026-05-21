//! Per-`(user, agent)` HITL approval mode + persistence.
//!
//! PR2 introduces two thread-level modes (the `HardFloor` rule layer
//! lands in PR3 as an orthogonal escalation, not a third mode):
//!
//! - `RequireApproval` (default): every tool that the `ApprovalGate`
//!   marks dangerous still prompts the user. This mirrors the pre-PR2
//!   behaviour.
//! - `AutoApprove`: the agent loop bypasses the handler for
//!   non-HardFloor-class tools and records the decision via the
//!   `audit::approval` tracing target. Users opt in with
//!   `/skip-approval` and revert with `/require-approval`.
//!
//! State is persisted as a single JSON file under the bind-mounted
//! zeptoclaw config dir, alongside `thread_meta.json` and `config.json`.
//! The file format is versioned and forward-compatible (unknown fields
//! tolerated, missing fields default), so a PR3 upgrade that adds e.g.
//! per-thread HardFloor overrides can land without breaking older
//! deployments.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::tools::approval_broker::ThreadKey;

/// File name (inside the zeptoclaw config dir) carrying persisted thread
/// approval state. Sits next to `thread_meta.json` and `config.json`.
pub const APPROVAL_THREAD_FILENAME: &str = "approval_thread.json";

/// Current persisted file format version. Bumped only on incompatible
/// shape changes; new optional fields are added under the existing
/// version using `serde(default)`.
const FILE_VERSION: u32 = 1;

/// Per-thread HITL approval mode.
///
/// The default is `RequireApproval` — fail-closed at the thread level
/// matches the global `ApprovalGate::requires_approval` default policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThreadApprovalMode {
    #[default]
    RequireApproval,
    AutoApprove,
}

/// Persisted state for a single `(user, agent)` thread.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadApprovalState {
    pub mode: ThreadApprovalMode,
    /// Wall-clock at the last state change. Used in `/approval-status`
    /// responses and in `audit::approval` log entries so operators can
    /// correlate "when did this thread go auto-approve?".
    pub updated_at: DateTime<Utc>,
    /// Channel that triggered the last mode change ("discord",
    /// "telegram", "acp_http", …). Empty for the synthesised default
    /// returned by `get` on an unknown thread.
    #[serde(default)]
    pub updated_by: String,
}

impl Default for ThreadApprovalState {
    fn default() -> Self {
        Self {
            mode: ThreadApprovalMode::default(),
            updated_at: Utc::now(),
            updated_by: String::new(),
        }
    }
}

/// On-disk envelope. The `threads` list is keyed by `(user_id,
/// agent_id)`; we use a list-of-records instead of a `HashMap<String,
/// _>` so the JSON stays self-describing without a custom key encoder.
#[derive(Debug, Default, Serialize, Deserialize)]
struct PersistedFile {
    #[serde(default = "default_version")]
    version: u32,
    #[serde(default)]
    threads: Vec<PersistedThread>,
}

fn default_version() -> u32 {
    FILE_VERSION
}

#[derive(Debug, Serialize, Deserialize)]
struct PersistedThread {
    user_id: String,
    agent_id: String,
    mode: ThreadApprovalMode,
    updated_at: DateTime<Utc>,
    #[serde(default)]
    updated_by: String,
}

/// In-memory + on-disk store of `ThreadApprovalState`. Cheap to clone
/// behind an `Arc`; reads are wait-free except for an `RwLock` read,
/// writes serialise on a full lock + atomic file rewrite.
pub struct ThreadApprovalStore {
    /// In-memory authoritative copy. Always reflects the on-disk file
    /// after `open()`. `RwLock` over a `HashMap` is adequate: this is a
    /// per-`(user, agent)` sandbox so contention is effectively zero;
    /// the `RwLock` is for the in-process / dev case where multiple
    /// threads exist.
    states: RwLock<HashMap<ThreadKey, ThreadApprovalState>>,
    /// Bind-mounted config dir where the JSON file lives.
    root: PathBuf,
}

impl ThreadApprovalStore {
    /// Open (or create) a store rooted at `root`. Existing state is
    /// loaded on a best-effort basis; a missing or corrupt file logs a
    /// `warn!` and the store starts empty (all threads default to
    /// `RequireApproval`).
    pub fn open(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let states = load_from_disk(&root);
        Self {
            states: RwLock::new(states),
            root,
        }
    }

    /// In-memory-only store, for tests that don't want to touch disk.
    #[cfg(test)]
    pub fn ephemeral() -> Self {
        Self {
            states: RwLock::new(HashMap::new()),
            root: PathBuf::new(),
        }
    }

    /// Read the state for `thread`. Returns the persisted record if
    /// one exists; otherwise a synthesised default (RequireApproval,
    /// `updated_at = now`, empty `updated_by`).
    pub fn get(&self, thread: &ThreadKey) -> ThreadApprovalState {
        self.states
            .read()
            .expect("ThreadApprovalStore poisoned")
            .get(thread)
            .cloned()
            .unwrap_or_default()
    }

    /// True when `thread` is currently configured to skip the
    /// `ApprovalGate` interactively. Convenience wrapper used by the
    /// approval handler hot path.
    pub fn is_auto_approve(&self, thread: &ThreadKey) -> bool {
        matches!(self.get(thread).mode, ThreadApprovalMode::AutoApprove)
    }

    /// Mutate `thread`'s mode and persist atomically. The on-disk file
    /// is rewritten via temp-file + rename; a failed rename leaves the
    /// previous canonical file intact. `source` records the channel
    /// that triggered the change (audit trail).
    pub fn set(
        &self,
        thread: &ThreadKey,
        mode: ThreadApprovalMode,
        source: &str,
    ) -> std::io::Result<ThreadApprovalState> {
        let state = ThreadApprovalState {
            mode,
            updated_at: Utc::now(),
            updated_by: source.to_string(),
        };
        let mut guard = self.states.write().expect("ThreadApprovalStore poisoned");
        guard.insert(thread.clone(), state.clone());
        if !self.root.as_os_str().is_empty() {
            save_to_disk(&self.root, &guard)?;
        }
        Ok(state)
    }
}

fn load_from_disk(root: &Path) -> HashMap<ThreadKey, ThreadApprovalState> {
    let path = root.join(APPROVAL_THREAD_FILENAME);
    match std::fs::read_to_string(&path) {
        Ok(body) => match serde_json::from_str::<PersistedFile>(&body) {
            Ok(file) => file
                .threads
                .into_iter()
                .map(|t| {
                    (
                        ThreadKey::new(t.user_id, t.agent_id),
                        ThreadApprovalState {
                            mode: t.mode,
                            updated_at: t.updated_at,
                            updated_by: t.updated_by,
                        },
                    )
                })
                .collect(),
            Err(e) => {
                warn!(
                    path = %path.display(),
                    error = %e,
                    "approval_thread.json is corrupt; starting from empty store \
                     (all threads will default to RequireApproval)"
                );
                HashMap::new()
            }
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => HashMap::new(),
        Err(e) => {
            warn!(
                path = %path.display(),
                error = %e,
                "failed to read approval_thread.json; starting from empty store"
            );
            HashMap::new()
        }
    }
}

fn save_to_disk(
    root: &Path,
    states: &HashMap<ThreadKey, ThreadApprovalState>,
) -> std::io::Result<()> {
    let payload = PersistedFile {
        version: FILE_VERSION,
        threads: states
            .iter()
            .map(|(k, v)| PersistedThread {
                user_id: k.user.clone(),
                agent_id: k.agent.clone(),
                mode: v.mode,
                updated_at: v.updated_at,
                updated_by: v.updated_by.clone(),
            })
            .collect(),
    };
    let body = serde_json::to_vec_pretty(&payload).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("serializing approval_thread.json: {}", e),
        )
    })?;
    let final_path = root.join(APPROVAL_THREAD_FILENAME);
    let tmp_path = root.join(format!("{}.tmp", APPROVAL_THREAD_FILENAME));
    std::fs::write(&tmp_path, &body)?;
    std::fs::rename(&tmp_path, &final_path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(user: &str, agent: &str) -> ThreadKey {
        ThreadKey::new(user, agent)
    }

    #[test]
    fn fresh_store_returns_require_approval_default() {
        let store = ThreadApprovalStore::ephemeral();
        let st = store.get(&key("alice", "zc"));
        assert_eq!(st.mode, ThreadApprovalMode::RequireApproval);
        assert!(!store.is_auto_approve(&key("alice", "zc")));
    }

    #[test]
    fn set_then_get_roundtrips_mode() {
        let store = ThreadApprovalStore::ephemeral();
        let thread = key("alice", "zc");
        store
            .set(&thread, ThreadApprovalMode::AutoApprove, "discord")
            .expect("set");
        let st = store.get(&thread);
        assert_eq!(st.mode, ThreadApprovalMode::AutoApprove);
        assert_eq!(st.updated_by, "discord");
        assert!(store.is_auto_approve(&thread));
    }

    #[test]
    fn set_isolates_threads() {
        let store = ThreadApprovalStore::ephemeral();
        let a = key("alice", "zc");
        let b = key("bob", "zc");
        let c = key("alice", "claude");
        store
            .set(&a, ThreadApprovalMode::AutoApprove, "cli")
            .unwrap();
        assert!(store.is_auto_approve(&a));
        assert!(!store.is_auto_approve(&b));
        assert!(!store.is_auto_approve(&c));
    }

    #[test]
    fn persist_and_reload_survives_process_restart() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let thread = key("alice", "zc");
        {
            let store = ThreadApprovalStore::open(tmp.path());
            store
                .set(&thread, ThreadApprovalMode::AutoApprove, "discord")
                .expect("set");
        }
        // Simulate a fresh process — new Store instance reads from
        // disk and recovers the in-memory map.
        let reloaded = ThreadApprovalStore::open(tmp.path());
        assert!(reloaded.is_auto_approve(&thread));
        let st = reloaded.get(&thread);
        assert_eq!(st.updated_by, "discord");
    }

    #[test]
    fn reloaded_state_carries_updated_at() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let thread = key("alice", "zc");
        let store = ThreadApprovalStore::open(tmp.path());
        let written = store
            .set(&thread, ThreadApprovalMode::AutoApprove, "cli")
            .unwrap();
        let reloaded = ThreadApprovalStore::open(tmp.path());
        let loaded = reloaded.get(&thread);
        // Timestamps survive serialization (millisecond fidelity is
        // enough for audit / UI; chrono RFC3339 preserves to ns).
        assert_eq!(loaded.updated_at, written.updated_at);
    }

    #[test]
    fn corrupt_file_falls_back_to_empty_store() {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            tmp.path().join(APPROVAL_THREAD_FILENAME),
            b"not json at all",
        )
        .unwrap();
        let store = ThreadApprovalStore::open(tmp.path());
        // No panic; defaults apply.
        assert!(!store.is_auto_approve(&key("alice", "zc")));
        // Subsequent writes still succeed (corrupt file gets
        // overwritten by the atomic rename on first set).
        store
            .set(&key("alice", "zc"), ThreadApprovalMode::AutoApprove, "cli")
            .unwrap();
        let reloaded = ThreadApprovalStore::open(tmp.path());
        assert!(reloaded.is_auto_approve(&key("alice", "zc")));
    }

    #[test]
    fn missing_file_returns_empty_store_silently() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let store = ThreadApprovalStore::open(tmp.path());
        assert!(!store.is_auto_approve(&key("alice", "zc")));
    }

    #[test]
    fn forward_compat_tolerates_unknown_fields() {
        // PR3 may add e.g. per-thread HardFloor overrides under the
        // existing version. The reader must NOT choke on them.
        let tmp = tempfile::tempdir().expect("tempdir");
        let body = serde_json::json!({
            "version": 1,
            "threads": [{
                "user_id": "alice",
                "agent_id": "zc",
                "mode": "auto_approve",
                "updated_at": "2026-05-17T04:00:00Z",
                "updated_by": "discord",
                "hard_floor_overrides": ["custom_rule"]
            }],
            "future_top_level_field": {"foo": 1}
        });
        std::fs::write(
            tmp.path().join(APPROVAL_THREAD_FILENAME),
            serde_json::to_string(&body).unwrap(),
        )
        .unwrap();
        let store = ThreadApprovalStore::open(tmp.path());
        assert!(store.is_auto_approve(&key("alice", "zc")));
    }

    #[test]
    fn switching_back_to_require_approval_persists() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let thread = key("alice", "zc");
        let store = ThreadApprovalStore::open(tmp.path());
        store
            .set(&thread, ThreadApprovalMode::AutoApprove, "discord")
            .unwrap();
        store
            .set(&thread, ThreadApprovalMode::RequireApproval, "discord")
            .unwrap();
        let reloaded = ThreadApprovalStore::open(tmp.path());
        assert!(!reloaded.is_auto_approve(&thread));
    }

    #[test]
    fn atomic_save_clears_temp_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let store = ThreadApprovalStore::open(tmp.path());
        store
            .set(&key("alice", "zc"), ThreadApprovalMode::AutoApprove, "cli")
            .unwrap();
        let tmp_path = tmp.path().join(format!("{}.tmp", APPROVAL_THREAD_FILENAME));
        assert!(
            !tmp_path.exists(),
            "rename should remove the temp file ({})",
            tmp_path.display()
        );
    }
}
