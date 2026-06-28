//! Sequence-packing optimisation knobs.

/// Configuration for sequence packing optimization
#[derive(Clone, Debug)]
pub struct SequencePackingConfig {
    /// Enable sequence packing optimization
    pub enabled: bool,
    /// Target batch size for packing
    pub target_batch_size: usize,
    /// Maximum batch size (hard limit)
    pub max_batch_size: usize,
    /// Length similarity threshold (0.0-1.0)
    pub similarity_threshold: f32,
}

impl Default for SequencePackingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            target_batch_size: 32,
            max_batch_size: 256,
            similarity_threshold: 0.2,
        }
    }
}

impl SequencePackingConfig {
    /// Returns a builder for configuring this type with the documented field defaults.
    /// Use `with_*(...)` to override individual fields, then `build()` to produce the type.
    #[must_use]
    pub fn builder() -> SequencePackingConfigBuilder {
        SequencePackingConfigBuilder::default()
    }
}

/// Builder for [`SequencePackingConfig`].
#[derive(Debug, Clone, Default)]
pub struct SequencePackingConfigBuilder {
    inner: SequencePackingConfig,
}

impl SequencePackingConfigBuilder {
    #[must_use]
    pub const fn with_enabled(mut self, v: bool) -> Self {
        self.inner.enabled = v;
        self
    }
    #[must_use]
    pub const fn with_target_batch_size(mut self, v: usize) -> Self {
        self.inner.target_batch_size = v;
        self
    }
    #[must_use]
    pub const fn with_max_batch_size(mut self, v: usize) -> Self {
        self.inner.max_batch_size = v;
        self
    }
    #[must_use]
    pub const fn with_similarity_threshold(mut self, v: f32) -> Self {
        self.inner.similarity_threshold = v;
        self
    }
    /// build: build the [`SequencePackingConfig`].
    #[must_use]
    pub const fn build(self) -> SequencePackingConfig {
        self.inner
    }
}

impl SequencePackingConfig {
    /// Create config from environment variables
    #[must_use]
    pub fn from_env() -> Self {
        let enabled = std::env::var("VLLM_SEQ_PACKING_ENABLED")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(true);
        let target_batch_size = std::env::var("VLLM_SEQ_PACKING_TARGET_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32);
        let max_batch_size = std::env::var("VLLM_SEQ_PACKING_MAX_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256);
        let similarity_threshold = std::env::var("VLLM_SEQ_PACKING_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.2);
        Self {
            enabled,
            target_batch_size,
            max_batch_size,
            similarity_threshold,
        }
    }
}
