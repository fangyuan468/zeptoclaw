//! LLM API cost estimation and tracking.
//!
//! Provides model pricing data, per-call cost estimation, and a thread-safe
//! `CostTracker` that accumulates spend across providers and models within
//! a session. Uses interior mutability via `Mutex` so all recording methods
//! take `&self`.

use std::collections::HashMap;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

/// Pricing for a single LLM model, expressed in USD per million tokens.
///
/// `cached_input_cost_per_million` and `cache_creation_cost_per_million`
/// are optional; when `None`, the corresponding token bucket is billed at
/// the full `input_cost_per_million` rate (i.e. behaves as if the
/// provider has no cache discount). For DeepSeek/SiliconFlow the cached
/// rate is typically 10% of input; for Anthropic prompt-caching the
/// cached rate is 10% and cache_creation is ~125% of input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPricing {
    /// Cost per 1 000 000 input (prompt) tokens in USD, full rate.
    pub input_cost_per_million: f64,
    /// Cost per 1 000 000 output (completion) tokens in USD.
    pub output_cost_per_million: f64,
    /// Cost per 1 000 000 cached input tokens in USD (prompt-cache hit).
    /// `None` ⇒ falls back to `input_cost_per_million`.
    #[serde(default)]
    pub cached_input_cost_per_million: Option<f64>,
    /// Cost per 1 000 000 cache-creation input tokens in USD (Anthropic
    /// prompt-cache write rate). `None` ⇒ falls back to
    /// `input_cost_per_million`.
    #[serde(default)]
    pub cache_creation_cost_per_million: Option<f64>,
}

/// Returns a static map of known model pricing.
///
/// Prices are in USD per million tokens and reflect public list prices at
/// the time of writing. Cache-rate fields are populated only for providers
/// where the discount is publicly documented.
pub fn default_pricing() -> HashMap<String, ModelPricing> {
    let mut m = HashMap::new();

    // Anthropic Claude models. Anthropic prompt-cache: read = 10% of input,
    // write = 125% of input (Sonnet/Opus tier; Haiku follows the same ratio).
    m.insert(
        "claude-sonnet-4-6".to_string(),
        ModelPricing {
            input_cost_per_million: 3.0,
            output_cost_per_million: 15.0,
            cached_input_cost_per_million: Some(0.30),
            cache_creation_cost_per_million: Some(3.75),
        },
    );
    m.insert(
        "claude-sonnet-4-5-20250929".to_string(),
        ModelPricing {
            input_cost_per_million: 3.0,
            output_cost_per_million: 15.0,
            cached_input_cost_per_million: Some(0.30),
            cache_creation_cost_per_million: Some(3.75),
        },
    );
    m.insert(
        "claude-3-5-sonnet-20241022".to_string(),
        ModelPricing {
            input_cost_per_million: 3.0,
            output_cost_per_million: 15.0,
            cached_input_cost_per_million: Some(0.30),
            cache_creation_cost_per_million: Some(3.75),
        },
    );
    m.insert(
        "claude-opus-4-6".to_string(),
        ModelPricing {
            input_cost_per_million: 15.0,
            output_cost_per_million: 75.0,
            cached_input_cost_per_million: Some(1.50),
            cache_creation_cost_per_million: Some(18.75),
        },
    );
    m.insert(
        "claude-3-opus-20240229".to_string(),
        ModelPricing {
            input_cost_per_million: 15.0,
            output_cost_per_million: 75.0,
            cached_input_cost_per_million: Some(1.50),
            cache_creation_cost_per_million: Some(18.75),
        },
    );
    m.insert(
        "claude-3-haiku-20240307".to_string(),
        ModelPricing {
            input_cost_per_million: 0.25,
            output_cost_per_million: 1.25,
            cached_input_cost_per_million: Some(0.025),
            cache_creation_cost_per_million: Some(0.3125),
        },
    );

    // OpenAI models. OpenAI cached-input rate is 50% of input on
    // cached-prefix-enabled models; cache_creation is N/A on this protocol.
    m.insert(
        "gpt-5.1".to_string(),
        ModelPricing {
            input_cost_per_million: 2.5,
            output_cost_per_million: 10.0,
            cached_input_cost_per_million: Some(1.25),
            cache_creation_cost_per_million: None,
        },
    );
    m.insert(
        "gpt-4o-mini".to_string(),
        ModelPricing {
            input_cost_per_million: 0.15,
            output_cost_per_million: 0.6,
            cached_input_cost_per_million: Some(0.075),
            cache_creation_cost_per_million: None,
        },
    );
    m.insert(
        "gpt-4-turbo".to_string(),
        ModelPricing {
            input_cost_per_million: 10.0,
            output_cost_per_million: 30.0,
            cached_input_cost_per_million: None,
            cache_creation_cost_per_million: None,
        },
    );

    m
}

/// Estimate the cost of a single LLM call in USD (no cache breakdown).
///
/// Looks up pricing in `custom_pricing` first, then falls back to
/// [`default_pricing`]. Returns `None` if the model is unknown in both.
///
/// All `prompt_tokens` are billed at the full input rate. To take the
/// per-bucket cache discount into account, use [`estimate_cost_with_cache`].
pub fn estimate_cost(
    model: &str,
    prompt_tokens: u32,
    completion_tokens: u32,
    custom_pricing: &HashMap<String, ModelPricing>,
) -> Option<f64> {
    estimate_cost_with_cache(
        model,
        prompt_tokens,
        completion_tokens,
        0,
        0,
        custom_pricing,
    )
}

/// Estimate the cost of a single LLM call in USD, billing each input
/// bucket at its own rate.
///
/// `cached_tokens` and `cache_creation_tokens` are *subsets* of
/// `prompt_tokens` (matching the [`crate::providers::Usage`] convention):
/// `non_cached = prompt_tokens − cached_tokens − cache_creation_tokens`.
/// The non-cached portion is billed at `input_cost_per_million`; cached
/// and cache-creation portions fall back to `input_cost_per_million` when
/// the model's pricing entry leaves the corresponding optional rate as
/// `None` (i.e. no discount declared).
///
/// If `cached + cache_creation` exceeds `prompt_tokens` (malformed input
/// from the upstream API), `non_cached` is clamped to 0 to avoid a
/// negative bill.
pub fn estimate_cost_with_cache(
    model: &str,
    prompt_tokens: u32,
    completion_tokens: u32,
    cached_tokens: u32,
    cache_creation_tokens: u32,
    custom_pricing: &HashMap<String, ModelPricing>,
) -> Option<f64> {
    // Resolve the lookup in two steps so the owned `defaults` HashMap
    // lives long enough for the borrow returned by `.get()`.
    let defaults = default_pricing();
    let pricing = custom_pricing.get(model).or_else(|| defaults.get(model))?;

    let cached = cached_tokens as u64;
    let cache_creation = cache_creation_tokens as u64;
    let prompt = prompt_tokens as u64;
    // Defensive clamp for malformed upstream payloads.
    let non_cached = prompt.saturating_sub(cached).saturating_sub(cache_creation);

    let cached_rate = pricing
        .cached_input_cost_per_million
        .unwrap_or(pricing.input_cost_per_million);
    let cache_creation_rate = pricing
        .cache_creation_cost_per_million
        .unwrap_or(pricing.input_cost_per_million);

    let non_cached_cost = (non_cached as f64 / 1_000_000.0) * pricing.input_cost_per_million;
    let cached_cost = (cached as f64 / 1_000_000.0) * cached_rate;
    let cache_creation_cost = (cache_creation as f64 / 1_000_000.0) * cache_creation_rate;
    let output_cost = (completion_tokens as f64 / 1_000_000.0) * pricing.output_cost_per_million;
    Some(non_cached_cost + cached_cost + cache_creation_cost + output_cost)
}

/// Internal mutable state guarded by the `CostTracker` mutex.
#[derive(Debug, Default)]
struct CostState {
    total_cost: f64,
    per_provider: HashMap<String, f64>,
    per_model: HashMap<String, f64>,
    call_count: u64,
}

/// Thread-safe, session-level cost accumulator.
///
/// All recording methods take `&self` (interior mutability via `Mutex`),
/// making it easy to share across async tasks via `Arc<CostTracker>`.
#[derive(Debug)]
pub struct CostTracker {
    state: Mutex<CostState>,
    custom_pricing: HashMap<String, ModelPricing>,
}

impl CostTracker {
    /// Creates a new tracker with default model pricing only.
    pub fn new() -> Self {
        Self {
            state: Mutex::new(CostState::default()),
            custom_pricing: HashMap::new(),
        }
    }

    /// Creates a new tracker with additional custom model pricing.
    ///
    /// Custom entries take precedence over the built-in defaults.
    pub fn new_with_pricing(custom: HashMap<String, ModelPricing>) -> Self {
        Self {
            state: Mutex::new(CostState::default()),
            custom_pricing: custom,
        }
    }

    /// Record a single LLM call (no cache breakdown).
    ///
    /// Estimates cost at full input rate (if the model is known) and
    /// accumulates it under both the provider name and the model name.
    /// Equivalent to `record_with_cache(provider, model, prompt, completion, 0, 0)`.
    pub fn record(&self, provider: &str, model: &str, prompt_tokens: u32, completion_tokens: u32) {
        self.record_with_cache(provider, model, prompt_tokens, completion_tokens, 0, 0);
    }

    /// Record a single LLM call with cache-bucket breakdown.
    ///
    /// `cached_tokens` and `cache_creation_tokens` are *subsets* of
    /// `prompt_tokens` and are billed at their respective per-million
    /// rates from [`ModelPricing`] (falling back to the full input rate
    /// when not configured).
    pub fn record_with_cache(
        &self,
        provider: &str,
        model: &str,
        prompt_tokens: u32,
        completion_tokens: u32,
        cached_tokens: u32,
        cache_creation_tokens: u32,
    ) {
        let cost = estimate_cost_with_cache(
            model,
            prompt_tokens,
            completion_tokens,
            cached_tokens,
            cache_creation_tokens,
            &self.custom_pricing,
        )
        .unwrap_or(0.0);

        let mut state = self.state.lock().unwrap();
        state.total_cost += cost;
        *state
            .per_provider
            .entry(provider.to_string())
            .or_insert(0.0) += cost;
        *state.per_model.entry(model.to_string()).or_insert(0.0) += cost;
        state.call_count += 1;
    }

    /// Returns the total accumulated cost in USD.
    pub fn total_cost(&self) -> f64 {
        self.state.lock().unwrap().total_cost
    }

    /// Returns a snapshot of accumulated cost per provider.
    pub fn cost_by_provider(&self) -> HashMap<String, f64> {
        self.state.lock().unwrap().per_provider.clone()
    }

    /// Returns a snapshot of accumulated cost per model.
    pub fn cost_by_model(&self) -> HashMap<String, f64> {
        self.state.lock().unwrap().per_model.clone()
    }

    /// Returns the total number of LLM calls recorded.
    pub fn call_count(&self) -> u64 {
        self.state.lock().unwrap().call_count
    }

    /// Produces a human-readable cost summary.
    ///
    /// Example output:
    /// ```text
    /// Total: $0.0150 (3 calls) | anthropic: $0.0120, openai: $0.0030
    /// ```
    pub fn summary(&self) -> String {
        let state = self.state.lock().unwrap();

        let mut summary = format!(
            "Total: ${:.4} ({} calls)",
            state.total_cost, state.call_count,
        );

        if !state.per_provider.is_empty() {
            let mut providers: Vec<_> = state.per_provider.iter().collect();
            providers.sort_by(|a, b| a.0.cmp(b.0));

            let parts: Vec<String> = providers
                .iter()
                .map(|(name, cost)| format!("{}: ${:.4}", name, cost))
                .collect();

            summary.push_str(" | ");
            summary.push_str(&parts.join(", "));
        }

        summary
    }
}

impl Default for CostTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for cost tracking, suitable for embedding in the main
/// ZeptoClaw config file.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct CostConfig {
    /// Whether cost tracking is enabled.
    pub enabled: bool,
    /// Custom per-model pricing overrides.
    pub custom_pricing: HashMap<String, ModelPricing>,
}

// We need Copy-like semantics for the lookup in estimate_cost where we clone
// out of a temporary HashMap. Derive Copy if the fields allow it (f64 is Copy).
impl Copy for ModelPricing {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_pricing_contains_claude_sonnet() {
        let prices = default_pricing();
        assert!(prices.contains_key("claude-sonnet-4-5-20250929"));
        let p = &prices["claude-sonnet-4-5-20250929"];
        assert!((p.input_cost_per_million - 3.0).abs() < f64::EPSILON);
        assert!((p.output_cost_per_million - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_default_pricing_contains_all_expected_models() {
        let prices = default_pricing();
        let expected = [
            "claude-sonnet-4-6",
            "claude-sonnet-4-5-20250929",
            "claude-3-5-sonnet-20241022",
            "claude-opus-4-6",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
            "gpt-5.1",
            "gpt-4o-mini",
            "gpt-4-turbo",
        ];
        for model in &expected {
            assert!(prices.contains_key(*model), "missing model: {}", model);
        }
        assert_eq!(prices.len(), expected.len());
    }

    #[test]
    fn test_estimate_cost_known_model() {
        let custom = HashMap::new();
        // claude-sonnet-4-5: $3/M input, $15/M output
        // 1000 input tokens = 1000/1_000_000 * 3.0 = 0.003
        // 500 output tokens  = 500/1_000_000 * 15.0 = 0.0075
        let cost = estimate_cost("claude-sonnet-4-5-20250929", 1000, 500, &custom).unwrap();
        assert!((cost - 0.0105).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_cost_gpt4o() {
        let custom = HashMap::new();
        // gpt-5.1: $2.5/M input, $10/M output
        // 2000 input  = 2000/1_000_000 * 2.5 = 0.005
        // 1000 output = 1000/1_000_000 * 10  = 0.01
        let cost = estimate_cost("gpt-5.1", 2000, 1000, &custom).unwrap();
        assert!((cost - 0.015).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_cost_unknown_model_returns_none() {
        let custom = HashMap::new();
        assert!(estimate_cost("unknown-model-xyz", 1000, 500, &custom).is_none());
    }

    #[test]
    fn test_estimate_cost_custom_pricing_overrides_default() {
        let mut custom = HashMap::new();
        custom.insert(
            "gpt-5.1".to_string(),
            ModelPricing {
                input_cost_per_million: 100.0,
                output_cost_per_million: 200.0,
                cached_input_cost_per_million: None,
                cache_creation_cost_per_million: None,
            },
        );
        // With custom pricing: 1000/1M * 100 + 500/1M * 200 = 0.1 + 0.1 = 0.2
        let cost = estimate_cost("gpt-5.1", 1000, 500, &custom).unwrap();
        assert!((cost - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_cost_custom_new_model() {
        let mut custom = HashMap::new();
        custom.insert(
            "my-custom-model".to_string(),
            ModelPricing {
                input_cost_per_million: 1.0,
                output_cost_per_million: 2.0,
                cached_input_cost_per_million: None,
                cache_creation_cost_per_million: None,
            },
        );
        let cost = estimate_cost("my-custom-model", 1_000_000, 1_000_000, &custom).unwrap();
        assert!((cost - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_cost_with_cache_full_hit_anthropic() {
        // claude-sonnet-4-5: input $3/M, output $15/M, cached $0.30/M, cache_creation $3.75/M
        // 1000 input total: 800 cached, 200 non-cached
        // cost = 200/1M * 3.0 + 800/1M * 0.30 + 0 + 50/1M * 15.0
        //      = 0.0006 + 0.00024 + 0 + 0.00075
        //      = 0.00159
        let custom = HashMap::new();
        let cost =
            estimate_cost_with_cache("claude-sonnet-4-5-20250929", 1000, 50, 800, 0, &custom)
                .unwrap();
        assert!((cost - 0.00159).abs() < 1e-10, "got {}", cost);
    }

    #[test]
    fn test_estimate_cost_with_cache_creation_anthropic() {
        // claude-sonnet-4-5: cache_creation $3.75/M
        // 1000 prompt: 0 cached, 1000 cache_creation, 0 non_cached
        // cost = 0 + 0 + 1000/1M * 3.75 + 50/1M * 15.0
        //      = 0.00375 + 0.00075 = 0.0045
        let custom = HashMap::new();
        let cost =
            estimate_cost_with_cache("claude-sonnet-4-5-20250929", 1000, 50, 0, 1000, &custom)
                .unwrap();
        assert!((cost - 0.0045).abs() < 1e-10, "got {}", cost);
    }

    #[test]
    fn test_estimate_cost_with_cache_zero_buckets_matches_estimate_cost() {
        // estimate_cost(...) is just estimate_cost_with_cache with zero
        // buckets — verify they produce identical results.
        let custom = HashMap::new();
        let plain = estimate_cost("gpt-5.1", 1000, 500, &custom).unwrap();
        let with_cache = estimate_cost_with_cache("gpt-5.1", 1000, 500, 0, 0, &custom).unwrap();
        assert!((plain - with_cache).abs() < 1e-12);
    }

    #[test]
    fn test_estimate_cost_with_cache_falls_back_when_rate_unset() {
        // gpt-4-turbo declares cached_input_cost_per_million = None, so a
        // cached request should bill cached tokens at the full input rate.
        let custom = HashMap::new();
        let with_cache = estimate_cost_with_cache("gpt-4-turbo", 1000, 0, 500, 0, &custom).unwrap();
        let plain = estimate_cost("gpt-4-turbo", 1000, 0, &custom).unwrap();
        assert!((plain - with_cache).abs() < 1e-12);
    }

    #[test]
    fn test_estimate_cost_with_cache_clamps_overflow() {
        // Malformed upstream: cached + cache_creation > prompt. We should
        // clamp non_cached to 0 rather than producing negative cost.
        let custom = HashMap::new();
        let cost = estimate_cost_with_cache(
            "claude-sonnet-4-5-20250929",
            500,
            0,
            800, // cached > prompt
            0,
            &custom,
        )
        .unwrap();
        // non_cached clamped to 0; cost = 0 + 800/1M * 0.30 + 0 + 0 = 0.00024
        assert!((cost - 0.00024).abs() < 1e-10, "got {}", cost);
    }

    #[test]
    fn test_estimate_cost_with_cache_unknown_model_returns_none() {
        let custom = HashMap::new();
        assert!(
            estimate_cost_with_cache("unknown-model-xyz", 1000, 500, 100, 0, &custom).is_none()
        );
    }

    #[test]
    fn test_record_with_cache_uses_cache_pricing() {
        let tracker = CostTracker::new();
        // 1000 prompt: 800 cached, 0 cache_creation, 200 non_cached on
        // claude-sonnet-4-5; expected cost = 0.00159 (see earlier test).
        tracker.record_with_cache("anthropic", "claude-sonnet-4-5-20250929", 1000, 50, 800, 0);
        assert!((tracker.total_cost() - 0.00159).abs() < 1e-10);
        assert_eq!(tracker.call_count(), 1);
    }

    #[test]
    fn test_record_delegates_to_record_with_cache() {
        // Verify the legacy `record` is just `record_with_cache` with
        // zero cache buckets — no double-count, identical totals.
        let a = CostTracker::new();
        let b = CostTracker::new();
        a.record("openai", "gpt-5.1", 1000, 500);
        b.record_with_cache("openai", "gpt-5.1", 1000, 500, 0, 0);
        assert!((a.total_cost() - b.total_cost()).abs() < 1e-12);
    }

    #[test]
    fn test_cost_tracker_new_starts_at_zero() {
        let tracker = CostTracker::new();
        assert!((tracker.total_cost() - 0.0).abs() < f64::EPSILON);
        assert_eq!(tracker.call_count(), 0);
        assert!(tracker.cost_by_provider().is_empty());
        assert!(tracker.cost_by_model().is_empty());
    }

    #[test]
    fn test_cost_tracker_record_accumulates() {
        let tracker = CostTracker::new();
        // gpt-5.1: $2.5/M input, $10/M output
        tracker.record("openai", "gpt-5.1", 1000, 500);
        // 1000/1M * 2.5 + 500/1M * 10 = 0.0025 + 0.005 = 0.0075
        assert!((tracker.total_cost() - 0.0075).abs() < 1e-10);
        assert_eq!(tracker.call_count(), 1);

        tracker.record("openai", "gpt-5.1", 1000, 500);
        assert!((tracker.total_cost() - 0.015).abs() < 1e-10);
        assert_eq!(tracker.call_count(), 2);
    }

    #[test]
    fn test_cost_tracker_multiple_providers() {
        let tracker = CostTracker::new();
        // anthropic call
        tracker.record("anthropic", "claude-sonnet-4-5-20250929", 1000, 500);
        // openai call
        tracker.record("openai", "gpt-5.1", 1000, 500);

        let by_provider = tracker.cost_by_provider();
        assert_eq!(by_provider.len(), 2);
        assert!(by_provider.contains_key("anthropic"));
        assert!(by_provider.contains_key("openai"));

        // anthropic: 1000/1M*3 + 500/1M*15 = 0.003 + 0.0075 = 0.0105
        assert!((by_provider["anthropic"] - 0.0105).abs() < 1e-10);
        // openai: 1000/1M*2.5 + 500/1M*10 = 0.0025 + 0.005 = 0.0075
        assert!((by_provider["openai"] - 0.0075).abs() < 1e-10);
    }

    #[test]
    fn test_cost_tracker_multiple_models() {
        let tracker = CostTracker::new();
        tracker.record("openai", "gpt-5.1", 1000, 500);
        tracker.record("openai", "gpt-4o-mini", 1000, 500);

        let by_model = tracker.cost_by_model();
        assert_eq!(by_model.len(), 2);
        assert!(by_model.contains_key("gpt-5.1"));
        assert!(by_model.contains_key("gpt-4o-mini"));

        // gpt-5.1: 0.0075
        assert!((by_model["gpt-5.1"] - 0.0075).abs() < 1e-10);
        // gpt-4o-mini: 1000/1M*0.15 + 500/1M*0.6 = 0.00015 + 0.0003 = 0.00045
        assert!((by_model["gpt-4o-mini"] - 0.00045).abs() < 1e-10);
    }

    #[test]
    fn test_cost_tracker_summary_format() {
        let tracker = CostTracker::new();
        tracker.record("anthropic", "claude-sonnet-4-5-20250929", 1000, 500);
        tracker.record("openai", "gpt-5.1", 2000, 1000);
        tracker.record("openai", "gpt-5.1", 2000, 1000);

        let summary = tracker.summary();

        assert!(summary.contains("Total: $"), "missing Total prefix");
        assert!(summary.contains("(3 calls)"), "missing call count");
        assert!(summary.contains("anthropic: $"), "missing anthropic");
        assert!(summary.contains("openai: $"), "missing openai");
    }

    #[test]
    fn test_cost_tracker_call_count() {
        let tracker = CostTracker::new();
        assert_eq!(tracker.call_count(), 0);

        tracker.record("anthropic", "claude-3-haiku-20240307", 100, 50);
        assert_eq!(tracker.call_count(), 1);

        tracker.record("anthropic", "claude-3-haiku-20240307", 100, 50);
        tracker.record("openai", "gpt-4o-mini", 100, 50);
        assert_eq!(tracker.call_count(), 3);
    }

    #[test]
    fn test_cost_tracker_unknown_model_zero_cost() {
        let tracker = CostTracker::new();
        tracker.record("custom", "unknown-model", 10000, 5000);

        // Unknown model should record 0.0 cost but still count the call
        assert!((tracker.total_cost() - 0.0).abs() < f64::EPSILON);
        assert_eq!(tracker.call_count(), 1);
        assert!(tracker.cost_by_provider().contains_key("custom"));
    }

    #[test]
    fn test_cost_config_default() {
        let config = CostConfig::default();
        assert!(!config.enabled);
        assert!(config.custom_pricing.is_empty());
    }

    #[test]
    fn test_cost_config_serde_roundtrip() {
        let mut custom = HashMap::new();
        custom.insert(
            "my-model".to_string(),
            ModelPricing {
                input_cost_per_million: 5.0,
                output_cost_per_million: 20.0,
                cached_input_cost_per_million: Some(0.5),
                cache_creation_cost_per_million: None,
            },
        );
        let config = CostConfig {
            enabled: true,
            custom_pricing: custom,
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: CostConfig = serde_json::from_str(&json).unwrap();

        assert!(parsed.enabled);
        assert_eq!(parsed.custom_pricing.len(), 1);
        let p = &parsed.custom_pricing["my-model"];
        assert!((p.input_cost_per_million - 5.0).abs() < f64::EPSILON);
        assert!((p.output_cost_per_million - 20.0).abs() < f64::EPSILON);
        assert!((p.cached_input_cost_per_million.unwrap() - 0.5).abs() < f64::EPSILON);
        assert!(p.cache_creation_cost_per_million.is_none());
    }

    #[test]
    fn test_model_pricing_serde_roundtrip() {
        let pricing = ModelPricing {
            input_cost_per_million: 3.0,
            output_cost_per_million: 15.0,
            cached_input_cost_per_million: Some(0.30),
            cache_creation_cost_per_million: Some(3.75),
        };

        let json = serde_json::to_string(&pricing).unwrap();
        let parsed: ModelPricing = serde_json::from_str(&json).unwrap();

        assert!((parsed.input_cost_per_million - 3.0).abs() < f64::EPSILON);
        assert!((parsed.output_cost_per_million - 15.0).abs() < f64::EPSILON);
        assert!((parsed.cached_input_cost_per_million.unwrap() - 0.30).abs() < f64::EPSILON);
        assert!((parsed.cache_creation_cost_per_million.unwrap() - 3.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_model_pricing_serde_legacy_payload_deserializes() {
        // Older config files won't have the cache fields; serde defaults
        // them to None.
        let json = r#"{"input_cost_per_million": 1.0, "output_cost_per_million": 2.0}"#;
        let parsed: ModelPricing = serde_json::from_str(json).unwrap();
        assert!((parsed.input_cost_per_million - 1.0).abs() < f64::EPSILON);
        assert!(parsed.cached_input_cost_per_million.is_none());
        assert!(parsed.cache_creation_cost_per_million.is_none());
    }

    #[test]
    fn test_cost_config_serde_defaults_on_missing_fields() {
        let json = "{}";
        let parsed: CostConfig = serde_json::from_str(json).unwrap();
        assert!(!parsed.enabled);
        assert!(parsed.custom_pricing.is_empty());
    }

    #[test]
    fn test_cost_tracker_with_custom_pricing() {
        let mut custom = HashMap::new();
        custom.insert(
            "my-llm".to_string(),
            ModelPricing {
                input_cost_per_million: 10.0,
                output_cost_per_million: 50.0,
                cached_input_cost_per_million: None,
                cache_creation_cost_per_million: None,
            },
        );
        let tracker = CostTracker::new_with_pricing(custom);

        tracker.record("custom-provider", "my-llm", 1_000_000, 1_000_000);
        // 1M/1M * 10 + 1M/1M * 50 = 60.0
        assert!((tracker.total_cost() - 60.0).abs() < 1e-10);
    }
}
