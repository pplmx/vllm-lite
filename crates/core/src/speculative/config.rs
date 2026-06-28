//! Configuration for speculative decoding
//!
//! `SpeculationConfig` controls how speculative decoding operates,
//! including draft token count, depth limits, and sampling parameters.

use std::sync::Arc;

/// `SpeculationConfig`: speculation configuration.
#[derive(Clone, Debug)]
pub struct SpeculationConfig {
    pub draft_count: usize,
    pub max_depth: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub target_model: Arc<String>,
    pub draft_layers: Option<usize>,
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

/// `SpeculationConfigBuilder`: speculation config builder.
#[derive(Default)]
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
        assert_eq!(config.temperature, 0.0);
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
        assert_eq!(config.temperature, 0.5);
    }

    #[test]
    fn test_from_env_not_set() {
        let config = SpeculationConfig::from_env();
        assert_eq!(config.draft_count, 4);
        assert_eq!(config.max_depth, 8);
    }
}
