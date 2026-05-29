//! Internal telemetry custom event names.
//!
//! These events are consumed by the outer Gateway and must not be forwarded
//! to user-facing AG-UI clients.

pub const HARNESS_MAX_ITER_REACHED: &str = "telemetry:harness_max_iter_reached";
pub const SYNTHESIS_LEGACY_INVOKED: &str = "telemetry:synthesis_legacy_invoked";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_names_use_telemetry_namespace() {
        for name in [HARNESS_MAX_ITER_REACHED, SYNTHESIS_LEGACY_INVOKED] {
            assert!(
                name.starts_with("telemetry:"),
                "telemetry custom event must use telemetry:* namespace: {name}"
            );
        }
    }
}
