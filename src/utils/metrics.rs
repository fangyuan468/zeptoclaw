//! Tool execution metrics collector.
//!
//! Provides a lightweight, thread-safe metrics collector for tracking tool
//! execution statistics within a session. Uses interior mutability via
//! `Mutex` so all recording methods take `&self`.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Per-tool execution statistics.
#[derive(Debug, Clone, Default)]
pub struct ToolMetrics {
    /// Total number of calls made to this tool.
    pub call_count: u64,
    /// Number of calls that resulted in an error.
    pub error_count: u64,
    /// Cumulative duration of all calls.
    pub total_duration: Duration,
    /// Shortest call duration observed.
    pub min_duration: Option<Duration>,
    /// Longest call duration observed.
    pub max_duration: Option<Duration>,
}

impl ToolMetrics {
    /// Returns the average call duration, or `None` if no calls have been recorded.
    pub fn average_duration(&self) -> Option<Duration> {
        if self.call_count == 0 {
            return None;
        }
        Some(self.total_duration / self.call_count as u32)
    }

    /// Returns the success rate as a value between 0.0 and 1.0.
    ///
    /// If no calls have been recorded, returns 1.0 (100%).
    pub fn success_rate(&self) -> f64 {
        if self.call_count == 0 {
            return 1.0;
        }
        (self.call_count - self.error_count) as f64 / self.call_count as f64
    }
}

/// Session-level metrics collector.
///
/// Thread-safe via interior `Mutex`. All recording methods take `&self`,
/// making it easy to share across async tasks via `Arc<MetricsCollector>`.
#[derive(Debug)]
pub struct MetricsCollector {
    tools: Mutex<HashMap<String, ToolMetrics>>,
    session_start: Instant,
    total_tokens_in: Mutex<u64>,
    total_tokens_out: Mutex<u64>,
    /// Cumulative subset of `total_tokens_in` that hit prompt-cache.
    /// Populated by `record_tokens_with_cache`; `record_tokens` leaves it 0.
    total_tokens_cached: Mutex<u64>,
    /// Cumulative `cache_creation_input_tokens` (Anthropic only).
    total_tokens_cache_creation: Mutex<u64>,
    /// Number of turns that accepted valid free text after entering the tool loop.
    harness_implicit_termination_total: Mutex<u64>,
    /// Number of times the model proposed a turn plan.
    harness_plan_proposed_total: Mutex<u64>,
    /// Number of times the model revised an active turn plan.
    harness_plan_revised_total: Mutex<u64>,
    /// Number of disorder advisories injected, keyed by signal label.
    harness_disorder_advisory_total: Mutex<HashMap<String, u64>>,
    /// Number of injected disorder advisories followed by a response that did not use plan tools.
    harness_disorder_advisory_ignored_total: Mutex<u64>,
    /// Number of turns that reached the max tool iteration crash guard.
    harness_max_iter_reached_total: Mutex<u64>,
    /// Number of times the legacy final synthesis fallback was invoked.
    synthesis_legacy_invoked_total: Mutex<u64>,
}

impl MetricsCollector {
    /// Creates a new metrics collector. The session clock starts immediately.
    pub fn new() -> Self {
        Self {
            tools: Mutex::new(HashMap::new()),
            session_start: Instant::now(),
            total_tokens_in: Mutex::new(0),
            total_tokens_out: Mutex::new(0),
            total_tokens_cached: Mutex::new(0),
            total_tokens_cache_creation: Mutex::new(0),
            harness_implicit_termination_total: Mutex::new(0),
            harness_plan_proposed_total: Mutex::new(0),
            harness_plan_revised_total: Mutex::new(0),
            harness_disorder_advisory_total: Mutex::new(HashMap::new()),
            harness_disorder_advisory_ignored_total: Mutex::new(0),
            harness_max_iter_reached_total: Mutex::new(0),
            synthesis_legacy_invoked_total: Mutex::new(0),
        }
    }

    /// Records a single tool call.
    ///
    /// Updates the per-tool `ToolMetrics` entry, creating it if this is the
    /// first call for the given tool name.
    pub fn record_tool_call(&self, tool_name: &str, duration: Duration, success: bool) {
        let mut tools = self.tools.lock().unwrap();
        let metrics = tools.entry(tool_name.to_string()).or_default();

        metrics.call_count += 1;
        if !success {
            metrics.error_count += 1;
        }
        metrics.total_duration += duration;

        metrics.min_duration = Some(match metrics.min_duration {
            Some(current) => current.min(duration),
            None => duration,
        });

        metrics.max_duration = Some(match metrics.max_duration {
            Some(current) => current.max(duration),
            None => duration,
        });
    }

    /// Adds to the running token totals.
    ///
    /// Treats this call as having no cache stats; `total_tokens_cached`
    /// and `total_tokens_cache_creation` are left unchanged. Use
    /// [`record_tokens_with_cache`] to record cache breakdowns.
    pub fn record_tokens(&self, input_tokens: u64, output_tokens: u64) {
        self.record_tokens_with_cache(input_tokens, output_tokens, 0, 0);
    }

    /// Adds to the running token totals including cache breakdown.
    ///
    /// `cached_tokens` and `cache_creation_tokens` are *subsets* of the
    /// reported `input_tokens` value, not additional input. Pass them
    /// straight from `crate::providers::Usage` (after PR2's
    /// `Usage::with_cache` parsing).
    pub fn record_tokens_with_cache(
        &self,
        input_tokens: u64,
        output_tokens: u64,
        cached_tokens: u64,
        cache_creation_tokens: u64,
    ) {
        *self.total_tokens_in.lock().unwrap() += input_tokens;
        *self.total_tokens_out.lock().unwrap() += output_tokens;
        *self.total_tokens_cached.lock().unwrap() += cached_tokens;
        *self.total_tokens_cache_creation.lock().unwrap() += cache_creation_tokens;
    }

    /// Records an implicit final answer accepted after at least one real tool call.
    pub fn record_harness_implicit_termination(&self) {
        *self.harness_implicit_termination_total.lock().unwrap() += 1;
    }

    /// Records that the model proposed a plan.
    pub fn record_harness_plan_proposed(&self) {
        *self.harness_plan_proposed_total.lock().unwrap() += 1;
    }

    /// Records that the model revised a plan.
    pub fn record_harness_plan_revised(&self) {
        *self.harness_plan_revised_total.lock().unwrap() += 1;
    }

    /// Records that a disorder advisory was injected for a signal.
    pub fn record_harness_disorder_advisory(&self, signal: &str) {
        let mut totals = self.harness_disorder_advisory_total.lock().unwrap();
        *totals.entry(signal.to_string()).or_insert(0) += 1;
    }

    /// Records that the next model response ignored a disorder advisory.
    pub fn record_harness_disorder_advisory_ignored(&self) {
        *self.harness_disorder_advisory_ignored_total.lock().unwrap() += 1;
    }

    /// Records that the max tool iteration crash guard was reached.
    pub fn record_harness_max_iter_reached(&self) {
        *self.harness_max_iter_reached_total.lock().unwrap() += 1;
    }

    /// Records that the legacy final synthesis fallback was invoked.
    pub fn record_synthesis_legacy_invoked(&self) {
        *self.synthesis_legacy_invoked_total.lock().unwrap() += 1;
    }

    /// Returns a clone of the metrics for a specific tool, or `None` if the
    /// tool has never been called.
    pub fn tool_metrics(&self, tool_name: &str) -> Option<ToolMetrics> {
        let tools = self.tools.lock().unwrap();
        tools.get(tool_name).cloned()
    }

    /// Returns a snapshot of all per-tool metrics.
    pub fn all_tool_metrics(&self) -> HashMap<String, ToolMetrics> {
        self.tools.lock().unwrap().clone()
    }

    /// Returns the sum of `call_count` across all tools.
    pub fn total_tool_calls(&self) -> u64 {
        let tools = self.tools.lock().unwrap();
        tools.values().map(|m| m.call_count).sum()
    }

    /// Returns the running token totals as `(input, output)`.
    pub fn total_tokens(&self) -> (u64, u64) {
        let input = *self.total_tokens_in.lock().unwrap();
        let output = *self.total_tokens_out.lock().unwrap();
        (input, output)
    }

    /// Returns the cumulative cached input tokens (subset of `input` from
    /// [`total_tokens`]).
    pub fn total_cached_tokens(&self) -> u64 {
        *self.total_tokens_cached.lock().unwrap()
    }

    /// Returns the cumulative cache-creation input tokens (subset of
    /// `input` from [`total_tokens`], Anthropic-only).
    pub fn total_cache_creation_tokens(&self) -> u64 {
        *self.total_tokens_cache_creation.lock().unwrap()
    }

    /// Returns the number of implicit tool-loop terminations accepted.
    pub fn harness_implicit_termination_total(&self) -> u64 {
        *self.harness_implicit_termination_total.lock().unwrap()
    }

    /// Returns the number of plans proposed.
    pub fn harness_plan_proposed_total(&self) -> u64 {
        *self.harness_plan_proposed_total.lock().unwrap()
    }

    /// Returns the number of plan revisions.
    pub fn harness_plan_revised_total(&self) -> u64 {
        *self.harness_plan_revised_total.lock().unwrap()
    }

    /// Returns a snapshot of disorder advisory counts keyed by signal label.
    pub fn harness_disorder_advisory_totals(&self) -> HashMap<String, u64> {
        self.harness_disorder_advisory_total.lock().unwrap().clone()
    }

    /// Returns the number of ignored disorder advisories.
    pub fn harness_disorder_advisory_ignored_total(&self) -> u64 {
        *self.harness_disorder_advisory_ignored_total.lock().unwrap()
    }

    /// Returns the number of max-iteration crash guard hits.
    pub fn harness_max_iter_reached_total(&self) -> u64 {
        *self.harness_max_iter_reached_total.lock().unwrap()
    }

    /// Returns the number of legacy final synthesis fallback invocations.
    pub fn synthesis_legacy_invoked_total(&self) -> u64 {
        *self.synthesis_legacy_invoked_total.lock().unwrap()
    }

    /// Cumulative cache-hit ratio: `total_tokens_cached / total_tokens_in`
    /// in `[0.0, 1.0]`. Returns 0.0 when no input tokens have been
    /// recorded.
    pub fn cache_hit_ratio(&self) -> f64 {
        let input = *self.total_tokens_in.lock().unwrap();
        if input == 0 {
            return 0.0;
        }
        let cached = *self.total_tokens_cached.lock().unwrap();
        cached as f64 / input as f64
    }

    /// Returns the elapsed time since the collector was created.
    pub fn session_duration(&self) -> Duration {
        self.session_start.elapsed()
    }

    /// Compute the worst-case tool latency across all tools.
    ///
    /// Uses `avg + 2 * (max - avg)` clamped to max as a rough upper-bound estimate.
    /// Returns `None` if no tool calls have been recorded.
    pub fn worst_case_tool_latency(&self) -> Option<Duration> {
        let tools = self.tools.lock().unwrap();
        let mut max_p95 = Duration::ZERO;
        let mut found_any = false;

        for metrics in tools.values() {
            if let (Some(avg), Some(max)) = (metrics.average_duration(), metrics.max_duration) {
                found_any = true;
                let spread = max.saturating_sub(avg);
                // p95 ≈ avg + 2*(max-avg), clamped to max
                let p95 = (avg + spread.mul_f32(2.0)).min(max);
                if p95 > max_p95 {
                    max_p95 = p95;
                }
            }
        }

        if found_any {
            Some(max_p95)
        } else {
            None
        }
    }

    /// Compute the aggregate success rate across all tools.
    ///
    /// Returns 1.0 if no calls have been recorded.
    pub fn aggregate_success_rate(&self) -> f64 {
        let tools = self.tools.lock().unwrap();
        let total_calls: u64 = tools.values().map(|m| m.call_count).sum();
        let total_errors: u64 = tools.values().map(|m| m.error_count).sum();

        if total_calls == 0 {
            return 1.0;
        }
        (total_calls - total_errors) as f64 / total_calls as f64
    }

    /// Produces a human-readable summary of the session metrics.
    ///
    /// Example output:
    /// ```text
    /// Session: 45s | Tools: 12 calls (2 errors) | Tokens: 1500 in / 800 out
    ///   shell: 5 calls, avg 200ms, 100% success
    ///   read_file: 4 calls, avg 5ms, 100% success
    ///   web_fetch: 3 calls, avg 1.2s, 67% success
    /// ```
    pub fn summary(&self) -> String {
        let tools = self.tools.lock().unwrap();
        let (tokens_in, tokens_out) = self.total_tokens();
        let cached = self.total_cached_tokens();
        let cache_creation = self.total_cache_creation_tokens();
        let session_secs = self.session_duration().as_secs();

        let total_calls: u64 = tools.values().map(|m| m.call_count).sum();
        let total_errors: u64 = tools.values().map(|m| m.error_count).sum();

        let mut summary = format!(
            "Session: {}s | Tools: {} calls ({} errors) | Tokens: {} in / {} out",
            session_secs, total_calls, total_errors, tokens_in, tokens_out,
        );

        if cached > 0 || cache_creation > 0 {
            let hit_pct = (self.cache_hit_ratio() * 100.0).round() as u64;
            summary.push_str(&format!(
                " | Cache: {} hit ({}%) / {} created",
                cached, hit_pct, cache_creation
            ));
        }

        // Sort tools by call_count descending.
        let mut entries: Vec<_> = tools.iter().collect();
        entries.sort_by_key(|b| std::cmp::Reverse(b.1.call_count));

        for (name, metrics) in entries {
            let avg = match metrics.average_duration() {
                Some(d) => format_duration(d),
                None => "N/A".to_string(),
            };
            let success_pct = (metrics.success_rate() * 100.0).round() as u64;
            summary.push_str(&format!(
                "\n  {}: {} calls, avg {}, {}% success",
                name, metrics.call_count, avg, success_pct,
            ));
        }

        summary
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Formats a duration in a human-friendly way.
///
/// - Under 1ms: shows microseconds (e.g. "500us")
/// - Under 1s: shows milliseconds (e.g. "200ms")
/// - 1s or more: shows seconds with one decimal (e.g. "1.2s")
fn format_duration(d: Duration) -> String {
    let micros = d.as_micros();
    if micros < 1_000 {
        format!("{}us", micros)
    } else if micros < 1_000_000 {
        format!("{}ms", d.as_millis())
    } else {
        format!("{:.1}s", d.as_secs_f64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_metrics_collector_new() {
        let collector = MetricsCollector::new();

        assert!(collector.all_tool_metrics().is_empty());
        assert_eq!(collector.total_tool_calls(), 0);
        assert_eq!(collector.total_tokens(), (0, 0));
        assert_eq!(collector.harness_max_iter_reached_total(), 0);
        assert_eq!(collector.synthesis_legacy_invoked_total(), 0);
    }

    #[test]
    fn test_record_tool_call_success() {
        let collector = MetricsCollector::new();
        let duration = Duration::from_millis(100);

        collector.record_tool_call("shell", duration, true);

        let metrics = collector.tool_metrics("shell").unwrap();
        assert_eq!(metrics.call_count, 1);
        assert_eq!(metrics.error_count, 0);
        assert_eq!(metrics.total_duration, duration);
        assert_eq!(metrics.min_duration, Some(duration));
        assert_eq!(metrics.max_duration, Some(duration));
    }

    #[test]
    fn test_record_tool_call_failure() {
        let collector = MetricsCollector::new();
        let duration = Duration::from_millis(50);

        collector.record_tool_call("web_fetch", duration, false);

        let metrics = collector.tool_metrics("web_fetch").unwrap();
        assert_eq!(metrics.call_count, 1);
        assert_eq!(metrics.error_count, 1);
    }

    #[test]
    fn test_record_multiple_calls() {
        let collector = MetricsCollector::new();

        collector.record_tool_call("shell", Duration::from_millis(100), true);
        collector.record_tool_call("shell", Duration::from_millis(200), true);
        collector.record_tool_call("shell", Duration::from_millis(300), true);

        let metrics = collector.tool_metrics("shell").unwrap();
        assert_eq!(metrics.call_count, 3);
        assert_eq!(metrics.error_count, 0);
        assert_eq!(metrics.total_duration, Duration::from_millis(600));
        assert_eq!(metrics.min_duration, Some(Duration::from_millis(100)));
        assert_eq!(metrics.max_duration, Some(Duration::from_millis(300)));

        let avg = metrics.average_duration().unwrap();
        assert_eq!(avg, Duration::from_millis(200));
    }

    #[test]
    fn test_record_tokens() {
        let collector = MetricsCollector::new();

        collector.record_tokens(500, 200);
        collector.record_tokens(1000, 600);

        assert_eq!(collector.total_tokens(), (1500, 800));
        // record_tokens (no-cache variant) must not touch cache counters.
        assert_eq!(collector.total_cached_tokens(), 0);
        assert_eq!(collector.total_cache_creation_tokens(), 0);
        assert_eq!(collector.cache_hit_ratio(), 0.0);
    }

    #[test]
    fn test_record_harness_crash_guard_metrics() {
        let collector = MetricsCollector::new();

        collector.record_harness_max_iter_reached();
        collector.record_harness_max_iter_reached();
        collector.record_synthesis_legacy_invoked();

        assert_eq!(collector.harness_max_iter_reached_total(), 2);
        assert_eq!(collector.synthesis_legacy_invoked_total(), 1);
    }

    #[test]
    fn test_record_tokens_with_cache_accumulates_all_buckets() {
        let collector = MetricsCollector::new();

        // First request: cache miss (write to cache).
        collector.record_tokens_with_cache(2000, 100, 0, 1500);
        // Second request: cache hit on the same prefix.
        collector.record_tokens_with_cache(2000, 80, 1500, 0);
        // Third request: pure no-cache call.
        collector.record_tokens_with_cache(500, 50, 0, 0);

        assert_eq!(collector.total_tokens(), (4500, 230));
        assert_eq!(collector.total_cached_tokens(), 1500);
        assert_eq!(collector.total_cache_creation_tokens(), 1500);
        // 1500 / 4500 = 1/3
        let ratio = collector.cache_hit_ratio();
        assert!(
            (ratio - (1500.0 / 4500.0)).abs() < 1e-9,
            "expected ~0.333, got {}",
            ratio
        );
    }

    #[test]
    fn test_record_tokens_delegates_to_with_cache() {
        // Verify the legacy `record_tokens` is just `record_tokens_with_cache`
        // with zero cache buckets (no double-count, no implicit assumptions).
        let collector = MetricsCollector::new();
        collector.record_tokens(1000, 300);
        assert_eq!(collector.total_tokens(), (1000, 300));
        assert_eq!(collector.total_cached_tokens(), 0);
        assert_eq!(collector.cache_hit_ratio(), 0.0);
    }

    #[test]
    fn test_cache_hit_ratio_zero_input() {
        let collector = MetricsCollector::new();
        assert_eq!(collector.cache_hit_ratio(), 0.0);
    }

    #[test]
    fn test_summary_omits_cache_section_when_no_cache_activity() {
        let collector = MetricsCollector::new();
        collector.record_tokens(100, 50);
        collector.record_tool_call("shell", Duration::from_millis(10), true);
        let summary = collector.summary();
        assert!(!summary.contains("Cache:"), "got: {}", summary);
    }

    #[test]
    fn test_summary_includes_cache_section_when_cache_seen() {
        let collector = MetricsCollector::new();
        collector.record_tokens_with_cache(1000, 100, 800, 0);
        let summary = collector.summary();
        assert!(summary.contains("Cache: 800 hit"), "got: {}", summary);
        // 800/1000 = 80%
        assert!(summary.contains("(80%)"), "got: {}", summary);
    }

    #[test]
    fn test_total_tool_calls() {
        let collector = MetricsCollector::new();

        collector.record_tool_call("shell", Duration::from_millis(10), true);
        collector.record_tool_call("shell", Duration::from_millis(20), true);
        collector.record_tool_call("read_file", Duration::from_millis(5), true);
        collector.record_tool_call("web_fetch", Duration::from_millis(1000), false);

        assert_eq!(collector.total_tool_calls(), 4);
    }

    #[test]
    fn test_tool_metrics_unknown_tool() {
        let collector = MetricsCollector::new();

        assert!(collector.tool_metrics("nonexistent").is_none());
    }

    #[test]
    fn test_all_tool_metrics() {
        let collector = MetricsCollector::new();

        collector.record_tool_call("shell", Duration::from_millis(10), true);
        collector.record_tool_call("read_file", Duration::from_millis(5), true);
        collector.record_tool_call("web_fetch", Duration::from_millis(1000), false);

        let all = collector.all_tool_metrics();
        assert_eq!(all.len(), 3);
        assert!(all.contains_key("shell"));
        assert!(all.contains_key("read_file"));
        assert!(all.contains_key("web_fetch"));
    }

    #[test]
    fn test_average_duration() {
        let mut metrics = ToolMetrics::default();
        assert!(metrics.average_duration().is_none());

        metrics.call_count = 4;
        metrics.total_duration = Duration::from_millis(400);
        assert_eq!(metrics.average_duration(), Some(Duration::from_millis(100)));
    }

    #[test]
    fn test_success_rate() {
        let collector = MetricsCollector::new();

        collector.record_tool_call("web_fetch", Duration::from_millis(100), true);
        collector.record_tool_call("web_fetch", Duration::from_millis(200), true);
        collector.record_tool_call("web_fetch", Duration::from_millis(300), true);
        collector.record_tool_call("web_fetch", Duration::from_millis(400), false);

        let metrics = collector.tool_metrics("web_fetch").unwrap();
        let rate = metrics.success_rate();
        assert!((rate - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_success_rate_zero_calls() {
        let metrics = ToolMetrics::default();
        assert!((metrics.success_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_summary_format() {
        let collector = MetricsCollector::new();

        collector.record_tool_call("shell", Duration::from_millis(200), true);
        collector.record_tool_call("shell", Duration::from_millis(200), true);
        collector.record_tool_call("read_file", Duration::from_millis(5), true);
        collector.record_tool_call("web_fetch", Duration::from_millis(1200), false);

        collector.record_tokens(1500, 800);

        let summary = collector.summary();

        assert!(summary.contains("Session:"));
        assert!(summary.contains("Tools: 4 calls (1 errors)"));
        assert!(summary.contains("Tokens: 1500 in / 800 out"));
        assert!(summary.contains("shell: 2 calls"));
        assert!(summary.contains("read_file: 1 calls"));
        assert!(summary.contains("web_fetch: 1 calls"));
        assert!(summary.contains("% success"));
    }

    #[test]
    fn test_session_duration() {
        let collector = MetricsCollector::new();
        thread::sleep(Duration::from_millis(10));

        let duration = collector.session_duration();
        assert!(duration >= Duration::from_millis(10));
    }

    #[test]
    fn test_worst_case_tool_latency_no_calls() {
        let collector = MetricsCollector::new();
        assert!(collector.worst_case_tool_latency().is_none());
    }

    #[test]
    fn test_worst_case_tool_latency_single_tool() {
        let collector = MetricsCollector::new();
        collector.record_tool_call("shell", Duration::from_millis(100), true);
        collector.record_tool_call("shell", Duration::from_millis(500), true);

        let p95 = collector.worst_case_tool_latency().unwrap();
        assert!(p95 >= Duration::from_millis(100));
        assert!(p95 <= Duration::from_millis(500));
    }

    #[test]
    fn test_aggregate_success_rate_no_calls() {
        let collector = MetricsCollector::new();
        assert!((collector.aggregate_success_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_aggregate_success_rate_mixed() {
        let collector = MetricsCollector::new();
        collector.record_tool_call("a", Duration::from_millis(10), true);
        collector.record_tool_call("a", Duration::from_millis(10), true);
        collector.record_tool_call("b", Duration::from_millis(10), true);
        collector.record_tool_call("b", Duration::from_millis(10), false);

        let rate = collector.aggregate_success_rate();
        assert!((rate - 0.75).abs() < f64::EPSILON);
    }
}
