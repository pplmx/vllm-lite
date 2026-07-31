//! Sequence-packing optimisation knobs.

/// Configuration for sequence packing optimization
#[derive(Clone, Debug, PartialEq)]
#[allow(clippy::derive_partial_eq_without_eq)]
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

/// Builder for [`SequencePackingConfig`].
#[derive(Debug, Clone, Default)]
pub struct SequencePackingConfigBuilder {
    inner: SequencePackingConfig,
}

impl SequencePackingConfigBuilder {
    /// Enable or disable sequence packing.
    #[must_use]
    pub const fn with_enabled(mut self, v: bool) -> Self {
        self.inner.enabled = v;
        self
    }
    /// Set the target number of sequences to pack into a batch.
    #[must_use]
    pub const fn with_target_batch_size(mut self, v: usize) -> Self {
        self.inner.target_batch_size = v;
        self
    }
    /// Set the maximum number of sequences in a packed batch.
    #[must_use]
    pub const fn with_max_batch_size(mut self, v: usize) -> Self {
        self.inner.max_batch_size = v;
        self
    }
    /// Set the similarity threshold for grouping sequences into a batch.
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

#[cfg(test)]
#[allow(clippy::float_cmp)]
// SAFETY: the test module mutates process-wide env vars
// (VLLM_SEQ_PACKING_*) which is `unsafe` in the Rust 2024 edition.
// The per-block SAFETY comments above each `unsafe` block document why
// the specific env-var names cannot race with other tests.
#[allow(unsafe_code)]
mod tests {
    use super::*;

    #[test]
    fn default_values() {
        let cfg = SequencePackingConfig::default();
        assert!(cfg.enabled);
        assert_eq!(cfg.target_batch_size, 32);
        assert_eq!(cfg.max_batch_size, 256);
        assert!((cfg.similarity_threshold - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn builder_overrides() {
        let cfg = SequencePackingConfig::builder()
            .with_enabled(false)
            .with_target_batch_size(16)
            .with_max_batch_size(64)
            .with_similarity_threshold(0.5)
            .build();
        assert!(!cfg.enabled);
        assert_eq!(cfg.target_batch_size, 16);
        assert_eq!(cfg.max_batch_size, 64);
        assert!((cfg.similarity_threshold - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn builder_defaults_match_struct_default() {
        assert_eq!(
            SequencePackingConfig::builder().build(),
            SequencePackingConfig::default()
        );
    }

    #[test]
    fn from_env_uses_defaults_when_vars_unset() {
        // Ensure env vars are not set.
        // SAFETY: These env var names are unique to this test and won't race
        // with other tests because no other test in the workspace writes to
        // VLLM_SEQ_PACKING_* variables. Single-threaded test execution also
        // prevents concurrent access.
        unsafe {
            std::env::remove_var("VLLM_SEQ_PACKING_ENABLED");
            std::env::remove_var("VLLM_SEQ_PACKING_TARGET_BATCH");
            std::env::remove_var("VLLM_SEQ_PACKING_MAX_BATCH");
            std::env::remove_var("VLLM_SEQ_PACKING_THRESHOLD");
        }
        let cfg = SequencePackingConfig::from_env();
        assert_eq!(cfg, SequencePackingConfig::default());
    }

    #[test]
    fn from_env_reads_env_vars() {
        // SAFETY: Same reasoning as from_env_uses_defaults_when_vars_unset — these
        // env var names are unique to this test module and won't race with other
        // tests that don't write to VLLM_SEQ_PACKING_* variables.
        unsafe {
            std::env::set_var("VLLM_SEQ_PACKING_ENABLED", "false");
            std::env::set_var("VLLM_SEQ_PACKING_TARGET_BATCH", "8");
            std::env::set_var("VLLM_SEQ_PACKING_MAX_BATCH", "16");
            std::env::set_var("VLLM_SEQ_PACKING_THRESHOLD", "0.35");
        }
        let cfg = SequencePackingConfig::from_env();
        assert!(!cfg.enabled);
        assert_eq!(cfg.target_batch_size, 8);
        assert_eq!(cfg.max_batch_size, 16);
        assert!((cfg.similarity_threshold - 0.35).abs() < f32::EPSILON);
        // Clean up.
        // SAFETY: Same reasoning as above — unique env var names ensure no races with other tests.
        unsafe {
            std::env::remove_var("VLLM_SEQ_PACKING_ENABLED");
            std::env::remove_var("VLLM_SEQ_PACKING_TARGET_BATCH");
            std::env::remove_var("VLLM_SEQ_PACKING_MAX_BATCH");
            std::env::remove_var("VLLM_SEQ_PACKING_THRESHOLD");
        }
    }

    #[test]
    fn from_env_falls_back_on_parse_failure() {
        // SAFETY: Same reasoning as above — unique env var names ensure no races with other tests.
        unsafe {
            std::env::set_var("VLLM_SEQ_PACKING_ENABLED", "not-a-bool");
            std::env::set_var("VLLM_SEQ_PACKING_TARGET_BATCH", "not-a-number");
        }
        let cfg = SequencePackingConfig::from_env();
        assert!(cfg.enabled); // default
        assert_eq!(cfg.target_batch_size, 32); // default
        // SAFETY: Same reasoning as above — unique env var names ensure no races with other tests.
        unsafe {
            std::env::remove_var("VLLM_SEQ_PACKING_ENABLED");
            std::env::remove_var("VLLM_SEQ_PACKING_TARGET_BATCH");
        }
    }
}
