//! Telemetry exporter for MetricsCollector.
//!
//! Renders session metrics in Prometheus text exposition format or JSON.
//! This module provides only rendering functions — no HTTP server or
//! transport logic.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::metrics::MetricsCollector;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Telemetry output format.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TelemetryFormat {
    #[default]
    Prometheus,
    Json,
}

/// Telemetry configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TelemetryConfig {
    /// Whether telemetry export is enabled.
    pub enabled: bool,
    /// Output format (prometheus or json).
    pub format: TelemetryFormat,
    /// HTTP endpoint path for serving metrics.
    pub endpoint: String,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            format: TelemetryFormat::default(),
            endpoint: "/metrics".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Renderers
// ---------------------------------------------------------------------------

/// Dispatches to the correct renderer based on the configured format.
pub fn render(collector: &MetricsCollector, format: &TelemetryFormat) -> String {
    match format {
        TelemetryFormat::Prometheus => render_prometheus(collector),
        TelemetryFormat::Json => render_json(collector),
    }
}

/// Renders all metrics in Prometheus text exposition format.
///
/// Metric families emitted:
/// - `zeptoclaw_tool_calls_total` (counter)
/// - `zeptoclaw_tool_errors_total` (counter)
/// - `zeptoclaw_tool_duration_seconds_sum` (counter)
/// - `zeptoclaw_tool_duration_seconds_min` (gauge)
/// - `zeptoclaw_tool_duration_seconds_max` (gauge)
/// - `zeptoclaw_tokens_input_total` (counter)
/// - `zeptoclaw_tokens_output_total` (counter)
/// - `zeptoclaw_harness_implicit_termination_total` (counter)
/// - `zeptoclaw_session_duration_seconds` (gauge)
pub fn render_prometheus(collector: &MetricsCollector) -> String {
    let mut out = String::new();

    // Collect per-tool metrics into a BTreeMap for deterministic ordering.
    let tools: BTreeMap<_, _> = collector.all_tool_metrics().into_iter().collect();

    // --- tool calls total ---
    out.push_str("# HELP zeptoclaw_tool_calls_total Total number of tool calls.\n");
    out.push_str("# TYPE zeptoclaw_tool_calls_total counter\n");
    for (name, m) in &tools {
        out.push_str(&format!(
            "zeptoclaw_tool_calls_total{{tool=\"{}\"}} {}\n",
            name, m.call_count,
        ));
    }

    // --- tool errors total ---
    out.push_str("# HELP zeptoclaw_tool_errors_total Total number of tool call errors.\n");
    out.push_str("# TYPE zeptoclaw_tool_errors_total counter\n");
    for (name, m) in &tools {
        out.push_str(&format!(
            "zeptoclaw_tool_errors_total{{tool=\"{}\"}} {}\n",
            name, m.error_count,
        ));
    }

    // --- tool duration sum ---
    out.push_str(
        "# HELP zeptoclaw_tool_duration_seconds_sum Cumulative tool call duration in seconds.\n",
    );
    out.push_str("# TYPE zeptoclaw_tool_duration_seconds_sum counter\n");
    for (name, m) in &tools {
        out.push_str(&format!(
            "zeptoclaw_tool_duration_seconds_sum{{tool=\"{}\"}} {:.6}\n",
            name,
            m.total_duration.as_secs_f64(),
        ));
    }

    // --- tool duration min ---
    out.push_str(
        "# HELP zeptoclaw_tool_duration_seconds_min Minimum observed tool call duration in seconds.\n",
    );
    out.push_str("# TYPE zeptoclaw_tool_duration_seconds_min gauge\n");
    for (name, m) in &tools {
        let val = m.min_duration.map_or(0.0, |d| d.as_secs_f64());
        out.push_str(&format!(
            "zeptoclaw_tool_duration_seconds_min{{tool=\"{}\"}} {:.6}\n",
            name, val,
        ));
    }

    // --- tool duration max ---
    out.push_str(
        "# HELP zeptoclaw_tool_duration_seconds_max Maximum observed tool call duration in seconds.\n",
    );
    out.push_str("# TYPE zeptoclaw_tool_duration_seconds_max gauge\n");
    for (name, m) in &tools {
        let val = m.max_duration.map_or(0.0, |d| d.as_secs_f64());
        out.push_str(&format!(
            "zeptoclaw_tool_duration_seconds_max{{tool=\"{}\"}} {:.6}\n",
            name, val,
        ));
    }

    // --- tokens ---
    let (tokens_in, tokens_out) = collector.total_tokens();

    out.push_str("# HELP zeptoclaw_tokens_input_total Total input tokens consumed.\n");
    out.push_str("# TYPE zeptoclaw_tokens_input_total counter\n");
    out.push_str(&format!("zeptoclaw_tokens_input_total {}\n", tokens_in));

    out.push_str("# HELP zeptoclaw_tokens_output_total Total output tokens produced.\n");
    out.push_str("# TYPE zeptoclaw_tokens_output_total counter\n");
    out.push_str(&format!("zeptoclaw_tokens_output_total {}\n", tokens_out));

    out.push_str("# HELP zeptoclaw_harness_implicit_termination_total Total accepted implicit final answers after a tool loop.\n");
    out.push_str("# TYPE zeptoclaw_harness_implicit_termination_total counter\n");
    out.push_str(&format!(
        "zeptoclaw_harness_implicit_termination_total {}\n",
        collector.harness_implicit_termination_total()
    ));

    // --- session duration ---
    out.push_str("# HELP zeptoclaw_session_duration_seconds Session uptime in seconds.\n");
    out.push_str("# TYPE zeptoclaw_session_duration_seconds gauge\n");
    out.push_str(&format!(
        "zeptoclaw_session_duration_seconds {:.6}\n",
        collector.session_duration().as_secs_f64(),
    ));

    out
}

/// Renders all metrics as a JSON string.
///
/// The output structure:
/// ```json
/// {
///   "tools": {
///     "shell": {
///       "call_count": 5,
///       "error_count": 1,
///       "total_duration_seconds": 1.234,
///       "min_duration_seconds": 0.1,
///       "max_duration_seconds": 0.5
///     }
///   },
///   "tokens_input_total": 1500,
///   "tokens_output_total": 800,
///   "session_duration_seconds": 45.0
/// }
/// ```
pub fn render_json(collector: &MetricsCollector) -> String {
    let tools_raw = collector.all_tool_metrics();
    // Use BTreeMap for deterministic key order.
    let mut tools_json: BTreeMap<String, serde_json::Value> = BTreeMap::new();

    for (name, m) in &tools_raw {
        tools_json.insert(
            name.clone(),
            serde_json::json!({
                "call_count": m.call_count,
                "error_count": m.error_count,
                "total_duration_seconds": m.total_duration.as_secs_f64(),
                "min_duration_seconds": m.min_duration.map(|d| d.as_secs_f64()),
                "max_duration_seconds": m.max_duration.map(|d| d.as_secs_f64()),
            }),
        );
    }

    let (tokens_in, tokens_out) = collector.total_tokens();

    let root = serde_json::json!({
        "tools": tools_json,
        "tokens_input_total": tokens_in,
        "tokens_output_total": tokens_out,
        "zeptoclaw_harness_implicit_termination_total": collector.harness_implicit_termination_total(),
        "session_duration_seconds": collector.session_duration().as_secs_f64(),
    });

    serde_json::to_string_pretty(&root).expect("metrics JSON serialization should never fail")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // -- TelemetryConfig defaults --

    #[test]
    fn test_telemetry_config_defaults() {
        let config = TelemetryConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.format, TelemetryFormat::Prometheus);
        assert_eq!(config.endpoint, "/metrics");
    }

    // -- TelemetryFormat default --

    #[test]
    fn test_telemetry_format_default_is_prometheus() {
        assert_eq!(TelemetryFormat::default(), TelemetryFormat::Prometheus);
    }

    // -- Serde roundtrip for TelemetryConfig --

    #[test]
    fn test_telemetry_config_serde_roundtrip() {
        let config = TelemetryConfig {
            enabled: true,
            format: TelemetryFormat::Json,
            endpoint: "/custom-metrics".to_string(),
        };
        let json = serde_json::to_string(&config).unwrap();
        let restored: TelemetryConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.enabled, config.enabled);
        assert_eq!(restored.format, config.format);
        assert_eq!(restored.endpoint, config.endpoint);
    }

    #[test]
    fn test_telemetry_config_serde_uses_defaults_for_missing_fields() {
        let json = "{}";
        let config: TelemetryConfig = serde_json::from_str(json).unwrap();
        assert!(!config.enabled);
        assert_eq!(config.format, TelemetryFormat::Prometheus);
        assert_eq!(config.endpoint, "/metrics");
    }

    // -- Serde variants for TelemetryFormat --

    #[test]
    fn test_telemetry_format_serde_prometheus() {
        let json = serde_json::to_string(&TelemetryFormat::Prometheus).unwrap();
        assert_eq!(json, "\"prometheus\"");
        let restored: TelemetryFormat = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, TelemetryFormat::Prometheus);
    }

    #[test]
    fn test_telemetry_format_serde_json() {
        let json = serde_json::to_string(&TelemetryFormat::Json).unwrap();
        assert_eq!(json, "\"json\"");
        let restored: TelemetryFormat = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, TelemetryFormat::Json);
    }

    // -- render_prometheus with empty metrics --

    #[test]
    fn test_render_prometheus_empty_metrics() {
        let collector = MetricsCollector::new();
        let output = render_prometheus(&collector);

        // Should still contain HELP/TYPE headers for token and session metrics.
        assert!(output.contains("# HELP zeptoclaw_tokens_input_total"));
        assert!(output.contains("# TYPE zeptoclaw_tokens_input_total counter"));
        assert!(output.contains("zeptoclaw_tokens_input_total 0"));
        assert!(output.contains("zeptoclaw_tokens_output_total 0"));
        assert!(output.contains("# HELP zeptoclaw_session_duration_seconds"));
        assert!(output.contains("zeptoclaw_session_duration_seconds"));

        // No tool-level data lines (only HELP/TYPE headers for tool metrics).
        assert!(!output.contains("tool=\""));
    }

    // -- render_prometheus with populated metrics --

    #[test]
    fn test_render_prometheus_populated_metrics() {
        let collector = MetricsCollector::new();
        collector.record_tool_call("shell", Duration::from_millis(100), true);
        collector.record_tool_call("shell", Duration::from_millis(300), false);
        collector.record_tool_call("read_file", Duration::from_millis(5), true);
        collector.record_tokens(1500, 800);

        let output = render_prometheus(&collector);

        // Tool call counts.
        assert!(output.contains("zeptoclaw_tool_calls_total{tool=\"shell\"} 2"));
        assert!(output.contains("zeptoclaw_tool_calls_total{tool=\"read_file\"} 1"));

        // Error counts.
        assert!(output.contains("zeptoclaw_tool_errors_total{tool=\"shell\"} 1"));
        assert!(output.contains("zeptoclaw_tool_errors_total{tool=\"read_file\"} 0"));

        // Tokens.
        assert!(output.contains("zeptoclaw_tokens_input_total 1500"));
        assert!(output.contains("zeptoclaw_tokens_output_total 800"));
    }

    // -- render_prometheus contains expected metric names and labels --

    #[test]
    fn test_render_prometheus_contains_expected_metric_families() {
        let collector = MetricsCollector::new();
        collector.record_tool_call("web_fetch", Duration::from_millis(200), true);

        let output = render_prometheus(&collector);

        let expected_families = [
            "zeptoclaw_tool_calls_total",
            "zeptoclaw_tool_errors_total",
            "zeptoclaw_tool_duration_seconds_sum",
            "zeptoclaw_tool_duration_seconds_min",
            "zeptoclaw_tool_duration_seconds_max",
            "zeptoclaw_tokens_input_total",
            "zeptoclaw_tokens_output_total",
            "zeptoclaw_harness_implicit_termination_total",
            "zeptoclaw_session_duration_seconds",
        ];

        for family in &expected_families {
            assert!(
                output.contains(&format!("# HELP {}", family)),
                "Missing HELP for {}",
                family,
            );
            assert!(
                output.contains(&format!("# TYPE {}", family)),
                "Missing TYPE for {}",
                family,
            );
        }

        // Labels present.
        assert!(output.contains("tool=\"web_fetch\""));
    }

    // -- render_json with empty metrics --

    #[test]
    fn test_render_json_empty_metrics() {
        let collector = MetricsCollector::new();
        let output = render_json(&collector);
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();

        assert_eq!(parsed["tools"], serde_json::json!({}));
        assert_eq!(parsed["tokens_input_total"], 0);
        assert_eq!(parsed["tokens_output_total"], 0);
        assert_eq!(parsed["zeptoclaw_harness_implicit_termination_total"], 0);
        assert!(parsed["session_duration_seconds"].as_f64().unwrap() >= 0.0);
    }

    // -- render_json with populated metrics --

    #[test]
    fn test_render_json_populated_metrics() {
        let collector = MetricsCollector::new();
        collector.record_tool_call("shell", Duration::from_millis(100), true);
        collector.record_tool_call("shell", Duration::from_millis(300), false);
        collector.record_tokens(500, 200);

        let output = render_json(&collector);
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();

        let shell = &parsed["tools"]["shell"];
        assert_eq!(shell["call_count"], 2);
        assert_eq!(shell["error_count"], 1);
        assert!(shell["total_duration_seconds"].as_f64().unwrap() > 0.0);
        assert!(shell["min_duration_seconds"].as_f64().unwrap() > 0.0);
        assert!(shell["max_duration_seconds"].as_f64().unwrap() > 0.0);

        assert_eq!(parsed["tokens_input_total"], 500);
        assert_eq!(parsed["tokens_output_total"], 200);
        assert_eq!(parsed["zeptoclaw_harness_implicit_termination_total"], 0);
    }

    // -- render dispatches correctly --

    #[test]
    fn test_render_dispatches_prometheus() {
        let collector = MetricsCollector::new();
        collector.record_tool_call("test_tool", Duration::from_millis(50), true);

        let output = render(&collector, &TelemetryFormat::Prometheus);
        // Prometheus format uses HELP/TYPE comments.
        assert!(output.contains("# HELP"));
        assert!(output.contains("# TYPE"));
        assert!(output.contains("tool=\"test_tool\""));
    }

    #[test]
    fn test_render_dispatches_json() {
        let collector = MetricsCollector::new();
        collector.record_tool_call("test_tool", Duration::from_millis(50), true);

        let output = render(&collector, &TelemetryFormat::Json);
        // JSON format parses as valid JSON.
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert!(parsed["tools"]["test_tool"].is_object());
    }
}
