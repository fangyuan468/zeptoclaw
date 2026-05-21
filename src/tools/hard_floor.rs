//! HardFloor: an orthogonal "always require human" escalation layer.
//!
//! HardFloor sits OUTSIDE the per-thread `ApprovalApprovalMode` axis: even
//! when a thread is `AutoApprove`, a HardFloor-matching tool invocation
//! still pops an approval card AND the resulting card is not eligible
//! for batch resolution (`yes all` / `no all` skip HardFloor entries).
//! HardFloor approval timeouts collapse to **Denied** rather than
//! `TimedOut` so a stalled user reply can never silently approve a
//! destructive op.
//!
//! Design notes:
//!
//! - This module is read-only / pure: callers build a `HardFloorMatcher`
//!   once (default ruleset baked in via `once_cell`) and call
//!   `check(tool_name, args)` per tool invocation. No I/O, no async, no
//!   global state.
//! - We currently only inspect the `shell` tool's `command` arg because
//!   that's where the high-cardinality destructive surface lives. Other
//!   tools (`write_file`, `edit_file`) are already gated by the regular
//!   `ApprovalGate`'s dangerous-tools list and don't have a "destroy
//!   the filesystem" failure mode that's worth a separate HardFloor
//!   rule. PR3 ships the shell-side ruleset; future PRs can layer in
//!   non-shell rules without touching the agent loop.
//! - Rules use `regex::Regex` for command matching (no shell parser —
//!   regex is good enough for the canonical "this is fatal" patterns
//!   the ruleset targets, and false negatives are acceptable here since
//!   HardFloor is a defence-in-depth net, not a sandbox).
//! - The regex set is **English-only** today. The plan calls out
//!   non-English shell aliases as out-of-scope; we revisit in a later
//!   PR if needed.

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value;

/// A single HardFloor rule. Compiled once at startup; checked per tool
/// invocation. `reason` is surfaced verbatim on the approval card and
/// in audit logs.
pub struct HardFloorRule {
    /// Stable identifier (snake_case). Used in audit logs for grepability.
    pub id: &'static str,
    /// Human-readable rationale rendered on the approval card.
    pub reason: &'static str,
    /// Compiled regex. Lazy so we only pay the compile cost on first
    /// access (and avoid panicking at module load on malformed input —
    /// the default set is unit-tested below).
    pub pattern: Lazy<Regex>,
}

impl HardFloorRule {
    /// Returns true if `command` matches this rule's pattern.
    pub fn matches_command(&self, command: &str) -> bool {
        self.pattern.is_match(command)
    }
}

// Helper to keep the rule table readable.
macro_rules! rule {
    ($id:literal, $reason:literal, $pattern:literal $(,)?) => {
        HardFloorRule {
            id: $id,
            reason: $reason,
            pattern: Lazy::new(|| {
                Regex::new($pattern)
                    .expect(concat!("hard_floor rule pattern failed to compile: ", $id))
            }),
        }
    };
}

/// Default ruleset (8 rules). Order is not load-bearing — `check` returns
/// the **first** match, so put broader patterns later if you ever extend
/// this list. Each rule has a corresponding unit test below with 3
/// positives + 3 negatives.
///
/// Pattern conventions:
///
/// - `\b` (word boundary) — keeps `rm` from firing on `trim`, `mkfs`
///   from firing on `mkfsutil`, `reboot` from firing on `rebootloader`,
///   etc. Crucially: regex's `\b` is a *transition* between word and
///   non-word chars, so a leading `sudo ` or `; ` doesn't break it.
/// - "command + at least one arg" — most rules require `\s+\S` after
///   the verb so that `man mkfs`, `cat /usr/share/mkfs.txt` or
///   `which reboot` don't trigger HardFloor when the verb is just
///   *mentioned*. Verbs that are catastrophic *with no argument*
///   (`reboot`, `halt`, the canonical fork-bomb) ignore this rule and
///   anchor on `\b` alone.
/// - Trailing `(\s|\*|$)` — the matched directive must end at
///   whitespace, EOL, or the glob `*` so `/` matches `/`, `/ `, `/*`,
///   but not `/tmp/...`.
///
/// All patterns are case-sensitive (shell commands are case-sensitive
/// on every platform we target). Non-English shell aliases / wrappers
/// are out of scope — see the module doc.
static DEFAULT_RULES: Lazy<Vec<HardFloorRule>> = Lazy::new(|| {
    vec![
        rule!(
            "rm_rf_root",
            "`rm -rf /` or `rm -rf /*` would wipe the entire filesystem.",
            r"\brm\s+(-[rRfF]+\s+)+/(\s|\*|$)",
        ),
        rule!(
            "rm_rf_home",
            "`rm -rf` against $HOME / ~ recursively deletes the user's data.",
            r"\brm\s+(-[rRfF]+\s+)+(\$HOME|\$\{HOME\}|~)(\s|$)",
        ),
        rule!(
            "mkfs_any",
            "`mkfs.*` formats a filesystem, destroying all data on the target.",
            r"\bmkfs(\.\w+)?\s+\S",
        ),
        rule!(
            "dd_to_block_device",
            "`dd of=/dev/...` writes raw bytes to a block device.",
            r"\bdd\s+([^|;]*\s)?of=/dev/(sd[a-z]\d*|nvme\d+n\d+(p\d+)?|hd[a-z]\d*|xvd[a-z]\d*|vd[a-z]\d*|disk\d*)(\s|$)",
        ),
        rule!(
            "wipefs_or_partition_destroy",
            "`wipefs` / destructive `parted` removes partition tables.",
            r"\b(wipefs\s+\S|parted\s+[^|;]*\b(mklabel|rm)\b)",
        ),
        rule!(
            "fork_bomb",
            "Classic fork-bomb pattern exhausts process limits.",
            r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
        ),
        rule!(
            "chmod_777_root_recursive",
            "`chmod -R 777 /` makes the entire filesystem world-writable.",
            r"\bchmod\s+(-R\s+)?[0-7]*7[0-7]*7[0-7]*7\s+/(\s|$)",
        ),
        rule!(
            "shutdown_or_reboot",
            "`shutdown` / `reboot` / `halt` takes the host offline.",
            r"\b(shutdown\s+\S|reboot\b|halt\b|poweroff\b|init\s+[06]\b)",
        ),
    ]
});

/// HardFloor checker.
///
/// `HardFloorMatcher::default()` returns a matcher backed by
/// `DEFAULT_RULES`. The matcher holds a `&'static` slice so it's cheap
/// to clone and ship around (`Default + Clone + Send + Sync`). Custom
/// rulesets are out of scope for PR3; the API is shaped so we can add a
/// `with_rules(&[HardFloorRule])` constructor later.
#[derive(Clone, Copy)]
pub struct HardFloorMatcher {
    rules: &'static [HardFloorRule],
}

impl HardFloorMatcher {
    /// Look up a HardFloor match for the given tool invocation.
    ///
    /// Returns `Some(reason)` for the first matching rule (in
    /// `DEFAULT_RULES` order). `None` means no HardFloor escalation.
    ///
    /// Only the `shell` tool's `command` argument is inspected today —
    /// see the module doc for rationale.
    pub fn check(&self, tool_name: &str, args: &Value) -> Option<&'static HardFloorRule> {
        if tool_name != "shell" {
            return None;
        }
        let command = args.get("command").and_then(Value::as_str)?;
        // `command` strings can legitimately span multiple lines (heredocs,
        // multi-line scripts). `regex::Regex` is single-line by default
        // unless we toggle `(?m)`; our patterns are line-anchored via
        // the `(^|[\s;&|...])` group, so we match against the whole
        // string verbatim.
        self.rules.iter().find(|r| r.matches_command(command))
    }

    /// Total rule count. Useful in tests and for `/approval-status`-like
    /// introspection — not currently surfaced over the wire.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Iterate over the matcher's rules. Lets callers (e.g. a future
    /// `/hardfloor-rules` command, or test fixtures) enumerate the
    /// ruleset without leaking the static slice.
    pub fn rules(&self) -> impl Iterator<Item = &'static HardFloorRule> {
        self.rules.iter()
    }
}

impl Default for HardFloorMatcher {
    fn default() -> Self {
        Self {
            rules: &DEFAULT_RULES,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn matcher() -> HardFloorMatcher {
        HardFloorMatcher::default()
    }

    fn check_shell(m: &HardFloorMatcher, cmd: &str) -> Option<&'static str> {
        m.check("shell", &json!({"command": cmd})).map(|r| r.id)
    }

    // ---- non-shell tools never trigger ---------------------------------

    #[test]
    fn check_returns_none_for_non_shell_tool() {
        let m = matcher();
        assert!(m
            .check("write_file", &json!({"path": "/", "content": "x"}))
            .is_none());
        assert!(m
            .check("edit_file", &json!({"path": "/", "edits": []}))
            .is_none());
        assert!(m.check("google", &json!({"query": "rm -rf /"})).is_none());
    }

    #[test]
    fn check_returns_none_for_shell_without_command_arg() {
        let m = matcher();
        assert!(m.check("shell", &json!({})).is_none());
        assert!(m.check("shell", &json!({"timeout": 60})).is_none());
        // Non-string command.
        assert!(m.check("shell", &json!({"command": 42})).is_none());
    }

    // ---- rule_count smoke test -----------------------------------------

    #[test]
    fn default_ruleset_has_eight_rules() {
        let m = matcher();
        assert_eq!(m.rule_count(), 8);
        // Sanity: every rule has a unique id (audit log greppability).
        let ids: Vec<&str> = m.rules().map(|r| r.id).collect();
        let unique: std::collections::HashSet<_> = ids.iter().copied().collect();
        assert_eq!(unique.len(), ids.len(), "duplicate rule ids: {:?}", ids);
    }

    // ---- rule 1: rm_rf_root --------------------------------------------

    #[test]
    fn rm_rf_root_positive() {
        let m = matcher();
        assert_eq!(check_shell(&m, "rm -rf /"), Some("rm_rf_root"));
        assert_eq!(check_shell(&m, "sudo rm -rf /*"), Some("rm_rf_root"));
        assert_eq!(check_shell(&m, "cd /tmp && rm -rf /"), Some("rm_rf_root"));
    }

    #[test]
    fn rm_rf_root_negative() {
        let m = matcher();
        // Targeted path, not bare root.
        assert_eq!(check_shell(&m, "rm -rf /tmp/build"), None);
        // No -rf flag.
        assert_eq!(check_shell(&m, "rm /etc/hosts"), None);
        // Different binary name (don't trip on prefixes).
        assert_eq!(check_shell(&m, "trim -rf / output.txt"), None);
    }

    // ---- rule 2: rm_rf_home --------------------------------------------

    #[test]
    fn rm_rf_home_positive() {
        let m = matcher();
        assert_eq!(check_shell(&m, "rm -rf $HOME"), Some("rm_rf_home"));
        assert_eq!(check_shell(&m, "rm -rf ${HOME}"), Some("rm_rf_home"));
        assert_eq!(check_shell(&m, "rm -rf ~"), Some("rm_rf_home"));
    }

    #[test]
    fn rm_rf_home_negative() {
        let m = matcher();
        // `~/foo/bar` is a sub-path (still recursively deletes a subtree
        // but not the whole home dir; left to ApprovalGate per design).
        // For now the rule deliberately accepts `~/single` but not deeper —
        // we keep the test honest: matched 1-segment paths fire.
        assert_eq!(check_shell(&m, "rm -rf $HOME/cache/foo"), None);
        // ${HOME}-like prefix that isn't HOME.
        assert_eq!(check_shell(&m, "rm -rf $HOMETOWN"), None);
        // Different command entirely.
        assert_eq!(check_shell(&m, "ls ~"), None);
    }

    // ---- rule 3: mkfs_any ----------------------------------------------

    #[test]
    fn mkfs_any_positive() {
        let m = matcher();
        assert_eq!(check_shell(&m, "mkfs /dev/sdb1"), Some("mkfs_any"));
        assert_eq!(check_shell(&m, "mkfs.ext4 /dev/sdb1"), Some("mkfs_any"));
        assert_eq!(
            check_shell(&m, "sudo mkfs.xfs /dev/nvme0n1p1"),
            Some("mkfs_any")
        );
    }

    #[test]
    fn mkfs_any_negative() {
        let m = matcher();
        // Substring match must NOT fire on unrelated binary names.
        assert_eq!(check_shell(&m, "mkfsutil --help"), None);
        // Reading the man page for mkfs is not destructive.
        assert_eq!(check_shell(&m, "man mkfs"), None);
        // Tool name in a path argument.
        assert_eq!(check_shell(&m, "cat /usr/share/mkfs.txt"), None);
    }

    // ---- rule 4: dd_to_block_device ------------------------------------

    #[test]
    fn dd_to_block_device_positive() {
        let m = matcher();
        assert_eq!(
            check_shell(&m, "dd if=/dev/zero of=/dev/sda bs=1M"),
            Some("dd_to_block_device")
        );
        assert_eq!(
            check_shell(&m, "sudo dd if=image.iso of=/dev/nvme0n1"),
            Some("dd_to_block_device")
        );
        assert_eq!(
            check_shell(&m, "dd if=/dev/zero of=/dev/sdb1 count=100"),
            Some("dd_to_block_device")
        );
    }

    #[test]
    fn dd_to_block_device_negative() {
        let m = matcher();
        // Output to a regular file is fine.
        assert_eq!(
            check_shell(&m, "dd if=/dev/zero of=./hello.bin bs=1k count=1"),
            None
        );
        // No `of=` arg at all (dd reading to stdout).
        assert_eq!(check_shell(&m, "dd if=/dev/urandom bs=64 count=1"), None);
        // /dev/null is not a block device.
        assert_eq!(check_shell(&m, "dd if=/etc/hosts of=/dev/null"), None);
    }

    // ---- rule 5: wipefs_or_partition_destroy ---------------------------

    #[test]
    fn wipefs_or_partition_destroy_positive() {
        let m = matcher();
        assert_eq!(
            check_shell(&m, "wipefs -a /dev/sdb"),
            Some("wipefs_or_partition_destroy")
        );
        assert_eq!(
            check_shell(&m, "parted /dev/sda mklabel gpt"),
            Some("wipefs_or_partition_destroy")
        );
        assert_eq!(
            check_shell(&m, "sudo parted -s /dev/sdb rm 1"),
            Some("wipefs_or_partition_destroy")
        );
    }

    #[test]
    fn wipefs_or_partition_destroy_negative() {
        let m = matcher();
        // parted *print* / *help* are read-only.
        assert_eq!(check_shell(&m, "parted /dev/sda print"), None);
        // Mentioning `wipefs` in an unrelated context (man page).
        assert_eq!(check_shell(&m, "man wipefs"), None);
        // `mklabel` not preceded by parted.
        assert_eq!(check_shell(&m, "echo mklabel"), None);
    }

    // ---- rule 6: fork_bomb ---------------------------------------------

    #[test]
    fn fork_bomb_positive() {
        let m = matcher();
        // Canonical: `:(){ :|:& };:`
        assert_eq!(check_shell(&m, ":(){ :|:& };:"), Some("fork_bomb"));
        // With irregular whitespace.
        assert_eq!(
            check_shell(&m, ": ( )  {  :  |  :  &  }  ;  :"),
            Some("fork_bomb")
        );
        // Embedded mid-line.
        assert_eq!(
            check_shell(&m, "echo ready && :(){ :|:& };: && echo unreachable"),
            Some("fork_bomb")
        );
    }

    #[test]
    fn fork_bomb_negative() {
        let m = matcher();
        // Random colons / braces — no recursive call pattern.
        assert_eq!(check_shell(&m, "echo ':)' "), None);
        // Function definition but no recursive pipe self-call.
        assert_eq!(check_shell(&m, "f(){ echo hi; }; f"), None);
        // A python `:` slice — completely unrelated.
        assert_eq!(check_shell(&m, "python -c 'a=[1,2,3]; print(a[:])'"), None);
    }

    // ---- rule 7: chmod_777_root_recursive ------------------------------

    #[test]
    fn chmod_777_root_recursive_positive() {
        let m = matcher();
        assert_eq!(
            check_shell(&m, "chmod -R 777 /"),
            Some("chmod_777_root_recursive")
        );
        assert_eq!(
            check_shell(&m, "sudo chmod -R 777 /"),
            Some("chmod_777_root_recursive")
        );
        // Non-recursive form: `chmod 777 /` is still catastrophic
        // (changes / itself but not children) — we treat it the same.
        assert_eq!(
            check_shell(&m, "chmod 777 /"),
            Some("chmod_777_root_recursive")
        );
    }

    #[test]
    fn chmod_777_root_recursive_negative() {
        let m = matcher();
        // Targeted path is fine.
        assert_eq!(check_shell(&m, "chmod -R 777 /tmp/build"), None);
        // Conservative permission isn't HardFloor.
        assert_eq!(check_shell(&m, "chmod -R 755 /"), None);
        // Command targeting a file under root, not root itself.
        assert_eq!(check_shell(&m, "chmod 777 /etc/hosts"), None);
    }

    // ---- rule 8: shutdown_or_reboot ------------------------------------

    #[test]
    fn shutdown_or_reboot_positive() {
        let m = matcher();
        assert_eq!(
            check_shell(&m, "shutdown -h now"),
            Some("shutdown_or_reboot")
        );
        assert_eq!(check_shell(&m, "sudo reboot"), Some("shutdown_or_reboot"));
        assert_eq!(check_shell(&m, "init 0"), Some("shutdown_or_reboot"));
    }

    #[test]
    fn shutdown_or_reboot_negative() {
        let m = matcher();
        // Substring in a different binary.
        assert_eq!(check_shell(&m, "rebootloader --help"), None);
        // Word in a path or unrelated arg.
        assert_eq!(check_shell(&m, "cat /var/log/shutdown.log"), None);
        // `init` with non-runlevel arg is not a fatal init transition.
        assert_eq!(check_shell(&m, "init.sh --setup"), None);
    }

    // ---- order independence: first match wins, but unique rules first ---

    #[test]
    fn first_match_wins_when_command_could_match_multiple_rules() {
        // Construct a (somewhat contrived) command where the first
        // ruleset entry to match dominates — important because the
        // `reason` rendered to the user must be deterministic.
        let m = matcher();
        let cmd = "rm -rf / && mkfs.ext4 /dev/sda1";
        // `rm_rf_root` precedes `mkfs_any` in DEFAULT_RULES.
        assert_eq!(check_shell(&m, cmd), Some("rm_rf_root"));
    }

    // ---- defaults are clone-able / cheap to pass around ----------------

    #[test]
    fn matcher_default_is_copy_and_cheap() {
        let m = matcher();
        let m_copy = m;
        assert_eq!(m.rule_count(), m_copy.rule_count());
    }

    // ---- rule reason / id are exposed on match -------------------------

    #[test]
    fn match_carries_id_and_reason() {
        let m = matcher();
        let rule = m
            .check("shell", &json!({"command": "rm -rf /"}))
            .expect("rule should match");
        assert_eq!(rule.id, "rm_rf_root");
        assert!(rule.reason.contains("filesystem"));
    }
}
