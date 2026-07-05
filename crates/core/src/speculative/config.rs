#![allow(clippy::module_name_repetitions)]
//! Configuration for speculative decoding
//!
//! `SpeculationConfig` controls how speculative decoding operates,
//! including draft token count, depth limits, and sampling parameters.

use std::sync::Arc;

/// Speculative-decoding configuration: number of draft tokens, acceptance threshold, max draft size, draft-model name. Built via [`SpeculationConfig::builder`].
#[derive(Clone, Debug)]
pub struct SpeculationConfig {
    /// Draft tokens produced per speculative step.
    pub draft_count: usize,
    /// Maximum tree depth (number of draft rounds) before forcing verification.
    pub max_depth: usize,
    /// Sampling temperature for the draft model.
    pub temperature: f32,
    /// Top-k sampling cutoff for the draft model (`0` = disabled).
    pub top_k: usize,
    /// Top-p (nucleus) sampling cutoff for the draft model.
    pub top_p: f32,
    /// Identifier of the target model that verifies the draft tokens.
    pub target_model: Arc<String>,
    /// Layer count for the draft model (`None` = use the registry default).
    pub draft_layers: Option<usize>,
    /// Whether to use self-speculation (target-model also acts as the drafter).
    pub self_speculation: bool,
}

impl Default for SpeculationConfig {
    fn default() -> Self {
        Self {
            draft_count: 4,
            max_depth: 8,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            target_model: Arc::new(String::new()),
            draft_layers: None,
            self_speculation: true,
        }
    }
}

impl SpeculationConfig {
    #[must_use]
    pub fn builder() -> SpeculationConfigBuilder {
        SpeculationConfigBuilder::default()
    }

    #[must_use]
    pub fn from_env() -> Self {
        Self {
            draft_count: std::env::var("VLLM_SPECULATIVE_DRAFT_COUNT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4),
            max_depth: std::env::var("VLLM_SPECULATIVE_MAX_DEPTH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8),
            temperature: std::env::var("VLLM_SPECULATIVE_TEMPERATURE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
            top_k: std::env::var("VLLM_SPECULATIVE_TOP_K")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            top_p: std::env::var("VLLM_SPECULATIVE_TOP_P")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.0),
            target_model: Arc::new(
                std::env::var("VLLM_SPECULATIVE_TARGET_MODEL").unwrap_or_default(),
            ),
            draft_layers: std::env::var("VLLM_SPECULATIVE_DRAFT_LAYERS")
                .ok()
                .and_then(|v| v.parse().ok()),
            self_speculation: std::env::var("VLLM_SPECULATIVE_SELF")
                .ok()
                .is_none_or(|v| v != "false"),
        }
    }
}

/// Builder for `SpeculationConfig`. Use `with_*` methods to override defaults, then call `.build()` to produce the final value.
#[derive(Debug, Default)]
pub struct SpeculationConfigBuilder {
    config: SpeculationConfig,
}

impl SpeculationConfigBuilder {
    #[must_use]
    pub const fn draft_count(mut self, count: usize) -> Self {
        self.config.draft_count = count;
        self
    }

    #[must_use]
    pub const fn max_depth(mut self, depth: usize) -> Self {
        self.config.max_depth = depth;
        self
    }

    #[must_use]
    pub const fn temperature(mut self, temp: f32) -> Self {
        self.config.temperature = temp;
        self
    }

    #[must_use]
    pub const fn top_k(mut self, k: usize) -> Self {
        self.config.top_k = k;
        self
    }

    #[must_use]
    pub const fn top_p(mut self, p: f32) -> Self {
        self.config.top_p = p;
        self
    }

    #[must_use]
    pub fn target_model(mut self, model: String) -> Self {
        self.config.target_model = Arc::new(model);
        self
    }

    #[must_use]
    pub const fn draft_layers(mut self, layers: usize) -> Self {
        self.config.draft_layers = Some(layers);
        self
    }

    #[must_use]
    pub const fn self_speculation(mut self, enabled: bool) -> Self {
        self.config.self_speculation = enabled;
        self
    }

    #[must_use]
    pub fn build(self) -> SpeculationConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SpeculationConfig::default();
        assert_eq!(config.draft_count, 4);
        assert_eq!(config.max_depth, 8);
        assert!(config.temperature.abs() < 1e-6);
        assert!(config.self_speculation);
    }

    #[test]
    fn test_builder_pattern() {
        let config = SpeculationConfig::builder()
            .draft_count(6)
            .max_depth(10)
            .temperature(0.5)
            .build();
        assert_eq!(config.draft_count, 6);
        assert_eq!(config.max_depth, 10);
        assert!((config.temperature - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_from_env_not_set() {
        let config = SpeculationConfig::from_env();
        assert_eq!(config.draft_count, 4);
        assert_eq!(config.max_depth, 8);
    }
}
