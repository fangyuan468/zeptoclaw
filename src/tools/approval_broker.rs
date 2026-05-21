//! Shared approval broker for coordinating tool approval across async boundaries.
//!
//! The `ApprovalBroker` manages pending approval requests using oneshot channels,
//! allowing the approval handler (agent loop) to wait for a user response while
//! the inbound message interceptor (on the `MessageBus`) resolves it.
//!
//! # Routing key
//!
//! Pending entries are keyed by a [`ThreadKey`] = `(user_id, agent_id)`, which is
//! stable across the lifetime of an `(user, agent)` sandbox container. Earlier
//! versions used `chat_id` (= ACP `session_id` in the containerised path); when
//! the scheduler rebuilds the ACP session mid-flight (transport hiccup, SSE
//! truncation, …) the freshly assigned `session_id` no longer matches the
//! pending entry, and the user's "yes" reply hits `resolve_by_id` with the
//! wrong key — surfacing as `RequestIdNotFound` and "Approval window already
//! closed". Anchoring on `ThreadKey` makes the broker oblivious to session
//! churn.
//!
//! `ThreadKey { user: "", agent: "" }` is reserved as the "unknown thread"
//! fallback used by legacy callers (in-process / dev mode that doesn't have
//! `ThreadIdentity` set up yet). It still works because all such callers share
//! the same chat_id namespace they had before; they just resolve under a
//! single fallback bucket instead of per-`(user, agent)`.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tokio::sync::oneshot;

/// Identifies a logical conversation thread for approval routing.
///
/// In the containerised deployment this is a one-to-one with a sandbox
/// process; in the in-process / dev path it falls back to whatever the
/// caller used to drive the `chat_id` (typically the channel-level chat id),
/// surfaced via [`ThreadKey::from_chat_id_fallback`].
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ThreadKey {
    pub user: String,
    pub agent: String,
}

impl ThreadKey {
    pub fn new(user: impl Into<String>, agent: impl Into<String>) -> Self {
        Self {
            user: user.into(),
            agent: agent.into(),
        }
    }

    /// Build a `ThreadKey` for callers that don't (yet) have a stable thread
    /// identity. The `chat_id` is parked in the `user` slot so different
    /// chats still get separate FIFO queues; the empty `agent` marks the
    /// entry as "legacy / unscoped" so we can later distinguish it from
    /// real per-`(user, agent)` entries during migration.
    pub fn from_chat_id_fallback(chat_id: &str) -> Self {
        Self {
            user: chat_id.to_string(),
            agent: String::new(),
        }
    }

    pub fn is_fallback(&self) -> bool {
        self.agent.is_empty()
    }
}

struct PendingEntry {
    request_id: String,
    /// Marks the entry as a HardFloor-class approval. PR3 enforces the
    /// bit in `resolve_all`: HardFloor entries are **never** drained
    /// by `yes all` / `no all` — they always require an explicit
    /// per-request decision.
    hard_floor: bool,
    created_at: Instant,
    tx: oneshot::Sender<bool>,
}

/// Coordinates pending tool-approval requests.
///
/// A single `ApprovalBroker` is shared (via `Arc`) between:
/// 1. The **approval handler** registered on the agent, which calls [`register`]
///    and then awaits the returned receiver.
/// 2. The **inbound interceptor** on `MessageBus`, which calls [`resolve_by_id`]
///    (when the user clicked a specific approval card) or [`resolve_fifo`]
///    (when the user typed a plain "yes" / "no" with no request context,
///    e.g. on Discord or codex stdio).
///
/// Pending entries are stored as `PendingEntry`s in FIFO order per
/// [`ThreadKey`]. [`resolve_by_id`] removes the matching entry by id;
/// [`resolve_fifo`] pops the oldest (the text-fallback path); [`resolve_all`]
/// drains everything for "yes all" / "no all".
///
/// Why both paths: AG-UI cards always carry a `requestId` and MUST resolve the
/// exact pending entry — without this the user could click "approve" on card B
/// and silently approve request A. Plain text channels (Discord raw "yes",
/// codex stdio prompt) have no id to thread through, so they fall back to FIFO.
pub struct ApprovalBroker {
    pending: Mutex<HashMap<ThreadKey, VecDeque<PendingEntry>>>,
}

impl ApprovalBroker {
    pub fn new() -> Self {
        Self {
            pending: Mutex::new(HashMap::new()),
        }
    }

    /// Register a pending approval for `(thread, request_id)`.
    ///
    /// Returns a receiver that completes when one of the resolve paths fires
    /// for this entry. The caller is responsible for generating a unique
    /// `request_id` (typically a ULID). `hard_floor=true` marks the entry as
    /// not eligible for batch (`yes all`/`no all`) resolution — PR1 records
    /// the bit; PR3 will enforce it.
    pub fn register(
        &self,
        thread: &ThreadKey,
        request_id: &str,
        hard_floor: bool,
    ) -> oneshot::Receiver<bool> {
        let (tx, rx) = oneshot::channel();
        self.pending
            .lock()
            .expect("ApprovalBroker mutex poisoned")
            .entry(thread.clone())
            .or_default()
            .push_back(PendingEntry {
                request_id: request_id.to_string(),
                hard_floor,
                created_at: Instant::now(),
                tx,
            });
        rx
    }

    /// Resolve the pending approval whose `request_id` matches.
    ///
    /// Returns `true` if a matching entry was found and the receiver was still
    /// alive. Returns `false` if no entry matched (e.g. the request already
    /// timed out or the sandbox restarted) — callers should treat this as
    /// "not handled" and let the message continue downstream rather than
    /// silently resolving an unrelated request.
    pub fn resolve_by_id(&self, thread: &ThreadKey, request_id: &str, approved: bool) -> bool {
        let mut guard = self.pending.lock().expect("ApprovalBroker mutex poisoned");
        let Some(queue) = guard.get_mut(thread) else {
            return false;
        };
        let Some(idx) = queue.iter().position(|e| e.request_id == request_id) else {
            return false;
        };
        let entry = queue.remove(idx).expect("position was just found");
        let empty = queue.is_empty();
        if empty {
            guard.remove(thread);
        }
        entry.tx.send(approved).is_ok()
    }

    /// Resolve the oldest pending approval for `thread`, regardless of id.
    ///
    /// Used by text-only channels (Discord raw "yes", codex stdio) where the
    /// user reply doesn't carry a request id. AG-UI / desktop callers MUST
    /// use [`resolve_by_id`] instead; using FIFO there leads to wrong-card
    /// approvals when multiple requests are pending.
    ///
    /// Returns `true` if a pending entry was found and resolved.
    pub fn resolve_fifo(&self, thread: &ThreadKey, approved: bool) -> bool {
        let mut guard = self.pending.lock().expect("ApprovalBroker mutex poisoned");
        let Some(queue) = guard.get_mut(thread) else {
            return false;
        };
        let Some(entry) = queue.pop_front() else {
            return false;
        };
        if queue.is_empty() {
            guard.remove(thread);
        }
        entry.tx.send(approved).is_ok()
    }

    /// Resolve ALL non-HardFloor pending approvals for `thread` with the
    /// same decision.
    ///
    /// Returns the number of requests **actually drained**. HardFloor
    /// entries (registered with `hard_floor=true`) are skipped — they
    /// always require an explicit per-request decision, so `yes all` /
    /// `no all` can't accidentally approve a destructive op. Skipped
    /// entries remain pending in the queue for `resolve_by_id` / next
    /// `resolve_fifo`.
    pub fn resolve_all(&self, thread: &ThreadKey, approved: bool) -> usize {
        let mut guard = self.pending.lock().expect("ApprovalBroker mutex poisoned");
        let Some(queue) = guard.get_mut(thread) else {
            return 0;
        };

        // Partition: drain non-hard-floor entries, keep hard-floor in
        // the queue. `VecDeque::retain` would consume in place but
        // wouldn't give us ownership of the drained `tx`s, so we walk
        // the queue manually instead.
        let mut drained = 0usize;
        let mut i = 0;
        while i < queue.len() {
            if queue[i].hard_floor {
                i += 1;
            } else {
                let entry = queue.remove(i).expect("index just checked");
                let _ = entry.tx.send(approved);
                drained += 1;
            }
        }
        if queue.is_empty() {
            guard.remove(thread);
        }
        drained
    }

    /// Return the number of pending approvals for `thread`.
    pub fn pending_count(&self, thread: &ThreadKey) -> usize {
        self.pending
            .lock()
            .expect("ApprovalBroker mutex poisoned")
            .get(thread)
            .map(|q| q.len())
            .unwrap_or(0)
    }

    /// Check whether any pending approval exists for `thread`.
    pub fn has_pending(&self, thread: &ThreadKey) -> bool {
        self.pending_count(thread) > 0
    }

    /// Drop pending entries older than `max_age` and return how many were
    /// swept. Used by a background task to reclaim entries whose handler
    /// future got cancelled out-of-band (e.g. SSE stream torn down by the
    /// channel layer) without ever firing the normal timeout path.
    ///
    /// Receivers backed by the dropped senders observe `RecvError`; the
    /// resolve-by-id and resolve-fifo paths skip them naturally because
    /// they've been removed from the map.
    pub fn sweep_expired(&self, max_age: Duration) -> usize {
        let now = Instant::now();
        let mut guard = self.pending.lock().expect("ApprovalBroker mutex poisoned");
        let mut swept = 0usize;
        guard.retain(|_thread, queue| {
            queue.retain(|entry| {
                let aged_out = now.duration_since(entry.created_at) > max_age;
                if aged_out {
                    swept += 1;
                }
                !aged_out
            });
            !queue.is_empty()
        });
        swept
    }
}

impl Default for ApprovalBroker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    fn tk(user: &str, agent: &str) -> ThreadKey {
        ThreadKey::new(user, agent)
    }

    #[test]
    fn register_and_resolve_approved() {
        let broker = ApprovalBroker::new();
        let key = tk("alice", "zc");
        let mut rx = broker.register(&key, "req1", false);
        assert!(broker.has_pending(&key));
        assert!(broker.resolve_fifo(&key, true));
        assert!(!broker.has_pending(&key));
        assert_eq!(rx.try_recv(), Ok(true));
    }

    #[test]
    fn register_and_resolve_denied() {
        let broker = ApprovalBroker::new();
        let key = tk("alice", "zc");
        let mut rx = broker.register(&key, "req1", false);
        assert!(broker.resolve_fifo(&key, false));
        assert_eq!(rx.try_recv(), Ok(false));
    }

    #[test]
    fn resolve_without_pending_returns_false() {
        let broker = ApprovalBroker::new();
        let key = tk("nobody", "zc");
        assert!(!broker.resolve_fifo(&key, true));
    }

    #[test]
    fn has_pending_returns_false_when_empty() {
        let broker = ApprovalBroker::new();
        assert!(!broker.has_pending(&tk("alice", "zc")));
    }

    #[test]
    fn double_register_queues_both() {
        let broker = ApprovalBroker::new();
        let key = tk("alice", "zc");
        let mut rx1 = broker.register(&key, "req1", false);
        let mut rx2 = broker.register(&key, "req2", false);
        assert_eq!(broker.pending_count(&key), 2);
        assert!(broker.resolve_fifo(&key, true));
        assert_eq!(rx1.try_recv(), Ok(true));
        assert_eq!(broker.pending_count(&key), 1);
        assert!(broker.resolve_fifo(&key, false));
        assert_eq!(rx2.try_recv(), Ok(false));
        assert!(!broker.has_pending(&key));
    }

    #[test]
    fn resolve_all_approves_every_pending() {
        let broker = ApprovalBroker::new();
        let key = tk("alice", "zc");
        let mut rx1 = broker.register(&key, "req1", false);
        let mut rx2 = broker.register(&key, "req2", false);
        let mut rx3 = broker.register(&key, "req3", false);
        assert_eq!(broker.resolve_all(&key, true), 3);
        assert_eq!(rx1.try_recv(), Ok(true));
        assert_eq!(rx2.try_recv(), Ok(true));
        assert_eq!(rx3.try_recv(), Ok(true));
        assert!(!broker.has_pending(&key));
    }

    // ---- PR3: resolve_all skips HardFloor entries ----------------------

    #[test]
    fn resolve_all_skips_hard_floor_entries() {
        let broker = ApprovalBroker::new();
        let key = tk("alice", "zc");
        let mut rx_normal_1 = broker.register(&key, "req_normal_1", false);
        let mut rx_hard = broker.register(&key, "req_hard", true);
        let mut rx_normal_2 = broker.register(&key, "req_normal_2", false);

        // `yes all` drains the two normal entries but leaves the
        // HardFloor entry alive in the queue.
        assert_eq!(broker.resolve_all(&key, true), 2);
        assert_eq!(rx_normal_1.try_recv(), Ok(true));
        assert_eq!(rx_normal_2.try_recv(), Ok(true));
        assert_eq!(
            rx_hard.try_recv(),
            Err(oneshot::error::TryRecvError::Empty),
            "HardFloor entry must NOT be drained by yes-all"
        );
        // Pending count reflects the remaining HardFloor entry.
        assert_eq!(broker.pending_count(&key), 1);

        // Per-request resolution still works on the survivor.
        assert!(broker.resolve_by_id(&key, "req_hard", true));
        assert_eq!(rx_hard.try_recv(), Ok(true));
        assert!(!broker.has_pending(&key));
    }

    #[test]
    fn resolve_all_returns_zero_when_only_hard_floor_pending() {
        let broker = ApprovalBroker::new();
        let key = tk("alice", "zc");
        let mut rx_hard = broker.register(&key, "req_hard", true);
        assert_eq!(broker.resolve_all(&key, true), 0);
        assert_eq!(rx_hard.try_recv(), Err(oneshot::error::TryRecvError::Empty));
        assert!(broker.has_pending(&key));
    }

    #[test]
    fn resolve_all_drops_thread_key_only_when_queue_fully_empty() {
        // After a partial drain, the bucket must remain so future
        // resolve_by_id calls can find the HardFloor entry.
        let broker = ApprovalBroker::new();
        let key = tk("alice", "zc");
        let _rx_normal = broker.register(&key, "req_normal", false);
        let _rx_hard = broker.register(&key, "req_hard", true);
        broker.resolve_all(&key, false);
        assert!(
            broker.has_pending(&key),
            "HardFloor survivor must stay in the map"
        );
    }

    #[test]
    fn resolve_all_preserves_hard_floor_order_for_subsequent_fifo() {
        // Mixed sequence — verify HardFloor entries keep their relative
        // order after non-HardFloor entries are drained.
        let broker = ApprovalBroker::new();
        let key = tk("alice", "zc");
        let _h1 = broker.register(&key, "h1", true);
        let _n1 = broker.register(&key, "n1", false);
        let _h2 = broker.register(&key, "h2", true);
        let _n2 = broker.register(&key, "n2", false);
        assert_eq!(broker.resolve_all(&key, true), 2);
        assert_eq!(broker.pending_count(&key), 2);
        // FIFO on the survivors must pop h1 before h2.
        assert!(broker.resolve_fifo(&key, true));
        assert_eq!(broker.pending_count(&key), 1);
        // The remaining entry must be h2.
        assert!(broker.resolve_by_id(&key, "h2", true));
        assert!(!broker.has_pending(&key));
    }

    #[test]
    fn resolve_by_id_targets_exact_request_not_fifo() {
        // Regression: previously broker was FIFO-only; an AG-UI card click
        // for the SECOND request would silently resolve the FIRST. Verify
        // resolve_by_id removes the entry by id, leaving the other intact.
        let broker = ApprovalBroker::new();
        let key = tk("alice", "zc");
        let mut rx_a = broker.register(&key, "req_a", false);
        let mut rx_b = broker.register(&key, "req_b", false);
        assert_eq!(broker.pending_count(&key), 2);

        assert!(broker.resolve_by_id(&key, "req_b", true));

        assert_eq!(rx_b.try_recv(), Ok(true));
        assert_eq!(rx_a.try_recv(), Err(oneshot::error::TryRecvError::Empty));
        assert_eq!(broker.pending_count(&key), 1);

        assert!(broker.resolve_by_id(&key, "req_a", false));
        assert_eq!(rx_a.try_recv(), Ok(false));
        assert!(!broker.has_pending(&key));
    }

    #[test]
    fn resolve_by_id_returns_false_for_unknown_id() {
        let broker = ApprovalBroker::new();
        let key = tk("alice", "zc");
        let mut rx = broker.register(&key, "req_a", false);
        assert!(!broker.resolve_by_id(&key, "req_does_not_exist", true));
        assert!(!broker.resolve_by_id(&tk("bob", "zc"), "req_a", true));
        assert_eq!(rx.try_recv(), Err(oneshot::error::TryRecvError::Empty));
        assert_eq!(broker.pending_count(&key), 1);
    }

    #[test]
    fn resolve_by_id_cleans_up_thread_entry_when_emptied() {
        let broker = ApprovalBroker::new();
        let key = tk("alice", "zc");
        let _rx = broker.register(&key, "req_a", false);
        assert!(broker.has_pending(&key));
        assert!(broker.resolve_by_id(&key, "req_a", true));
        assert!(!broker.has_pending(&key));
    }

    // ---- ThreadKey stability across chat_id churn ----

    #[test]
    fn resolve_by_id_survives_chat_id_churn() {
        // The scenario PR1 exists to fix: the approval handler registered a
        // request under `(user, agent)`; the ACP session_id (which old code
        // used as the routing key) then changes — e.g. scheduler tears down
        // and rebuilds the session due to SSE truncation. With the broker
        // anchored to ThreadKey, the new resolve still finds the pending
        // entry regardless of the chat_id swap.
        let broker = ApprovalBroker::new();
        let key = tk("alice", "zc");
        let mut rx = broker.register(&key, "req1", false);
        // Some unrelated "old chat_id" must not match anymore (it never did
        // — we just want to demonstrate the legacy fallback bucket is
        // empty since registration used the real ThreadKey).
        assert!(!broker.has_pending(&ThreadKey::from_chat_id_fallback("acp-session-1")));
        // Replicated user reply on the rebuilt session resolves cleanly.
        assert!(broker.resolve_by_id(&key, "req1", true));
        assert_eq!(rx.try_recv(), Ok(true));
    }

    #[test]
    fn fallback_bucket_separates_chats() {
        // Legacy callers without a real ThreadIdentity still get isolation
        // because each chat_id lands in its own fallback bucket.
        let broker = ApprovalBroker::new();
        let key_a = ThreadKey::from_chat_id_fallback("discord:guild-123");
        let key_b = ThreadKey::from_chat_id_fallback("discord:guild-456");
        let mut rx_a = broker.register(&key_a, "req_a", false);
        let _rx_b = broker.register(&key_b, "req_b", false);

        // Resolving in key_b's bucket must NOT touch key_a's pending entry.
        assert!(broker.resolve_fifo(&key_b, true));
        assert_eq!(rx_a.try_recv(), Err(oneshot::error::TryRecvError::Empty));
        assert!(broker.has_pending(&key_a));
        assert!(!broker.has_pending(&key_b));
    }

    // ---- sweep_expired ----

    #[test]
    fn sweep_expired_removes_aged_entries() {
        let broker = ApprovalBroker::new();
        let key = tk("alice", "zc");
        let _rx_old = broker.register(&key, "req_old", false);
        // Let entry age past the threshold.
        sleep(Duration::from_millis(50));
        let _rx_new = broker.register(&key, "req_new", false);

        let swept = broker.sweep_expired(Duration::from_millis(20));
        assert_eq!(swept, 1, "should have swept the old entry only");
        // New entry stays.
        assert!(broker.has_pending(&key));
        assert!(broker.resolve_by_id(&key, "req_new", true));
        assert!(!broker.has_pending(&key));
    }

    #[test]
    fn sweep_expired_drops_empty_threads() {
        let broker = ApprovalBroker::new();
        let key = tk("alice", "zc");
        let _rx = broker.register(&key, "req1", false);
        sleep(Duration::from_millis(30));
        let swept = broker.sweep_expired(Duration::from_millis(10));
        assert_eq!(swept, 1);
        assert!(!broker.has_pending(&key));
        // resolve_by_id on a now-empty thread must return false (entry dropped).
        assert!(!broker.resolve_by_id(&key, "req1", true));
    }

    #[test]
    fn sweep_expired_no_op_when_all_fresh() {
        let broker = ApprovalBroker::new();
        let key = tk("alice", "zc");
        let _rx = broker.register(&key, "req1", false);
        let swept = broker.sweep_expired(Duration::from_secs(60));
        assert_eq!(swept, 0);
        assert!(broker.has_pending(&key));
    }
}
