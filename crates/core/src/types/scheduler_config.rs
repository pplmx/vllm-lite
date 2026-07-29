//! Top-level scheduler configuration.

use crate::scheduler::cuda_graph::SchedulerCudaGraphConfig;
use crate::types::sequence_packing::SequencePackingConfig;

/// Configuration for the request scheduler.
///
/// Controls batching behavior, prefill/decode separation, and priority handling.
#[derive(Clone, Debug, PartialEq)]
#[allow(clippy::derive_partial_eq_without_eq)]
pub struct SchedulerConfig {
    /// Maximum number of sequences that can be scheduled in a single batch.
    pub max_num_seqs: usize,
    /// Maximum number of tokens (including prompt and generated) in a batch.
    pub max_num_batched_tokens: usize,
    /// Maximum consecutive decode iterations before forcing a prefill.
    pub max_consecutive_decode: u32,
    /// Enable separation of prefill and decode phases into different batches.
    pub enable_pd_separation: bool,
    /// Maximum number of prompt tokens to process in a single prefill chunk.
    pub prefill_chunk_size: usize,
    /// Ratio of decode-to-prefill tokens when batching mixed phases (0.0-1.0).
    pub decode_preference_ratio: f32,
    /// Enable priority-based scheduling (higher priority requests first).
    pub enable_priority_scheduling: bool,
    /// Enable dynamic batching (grouping similar requests automatically).
    pub enable_dynamic_batching: bool,
    /// Minimum batch size for dynamic batching.
    pub min_batch_size: usize,
    /// Maximum batch size for dynamic batching.
    pub max_batch_size: usize,
    /// CUDA Graph configuration
    pub cuda_graph: SchedulerCudaGraphConfig,
    /// Sequence packing configuration
    pub packing: SequencePackingConfig,
}

impl SchedulerConfig {
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    /// Construct a `SchedulerConfig` from explicit parameters, validating the
    /// cross-field invariants in one place. Prefer this over the literal
    /// struct expression when you want panicking validation; prefer
    /// [`SchedulerConfig::builder`] when you want to override only a few
    /// fields of the default.
    ///
    /// # Panics
    ///
    /// Panics if any of the following invariants is violated:
    /// - `max_num_seqs > 0`
    /// - `max_num_batched_tokens > 0`
    /// - `prefill_chunk_size > 0`
    /// - `0.0 <= decode_preference_ratio <= 1.0`
    /// - `min_batch_size > 0`
    /// - `max_batch_size >= min_batch_size`
    /// - `max_num_batched_tokens >= max_batch_size`
    pub fn new(
        max_num_seqs: usize,
        max_num_batched_tokens: usize,
        max_consecutive_decode: u32,
        enable_pd_separation: bool,
        prefill_chunk_size: usize,
        decode_preference_ratio: f32,
        enable_priority_scheduling: bool,
        enable_dynamic_batching: bool,
        min_batch_size: usize,
        max_batch_size: usize,
        packing: SequencePackingConfig,
    ) -> Self {
        assert!(max_num_seqs > 0, "max_num_seqs must be > 0");
        assert!(
            max_num_batched_tokens > 0,
            "max_num_batched_tokens must be > 0"
        );
        assert!(prefill_chunk_size > 0, "prefill_chunk_size must be > 0");
        assert!(
            (0.0..=1.0).contains(&decode_preference_ratio),
            "decode_preference_ratio must be between 0.0 and 1.0"
        );
        assert!(min_batch_size > 0, "min_batch_size must be > 0");
        assert!(
            max_batch_size >= min_batch_size,
            "max_batch_size must be >= min_batch_size"
        );
        assert!(
            max_num_batched_tokens >= max_batch_size,
            "max_num_batched_tokens must be >= max_batch_size"
        );

        Self {
            max_num_seqs,
            max_num_batched_tokens,
            max_consecutive_decode,
            enable_pd_separation,
            prefill_chunk_size,
            decode_preference_ratio,
            enable_priority_scheduling,
            enable_dynamic_batching,
            min_batch_size,
            max_batch_size,
            cuda_graph: SchedulerCudaGraphConfig::default(),
            packing,
        }
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 256,
            max_num_batched_tokens: 4096,
            max_consecutive_decode: 10,
            enable_pd_separation: true,
            prefill_chunk_size: 512,
            decode_preference_ratio: 0.7,
            enable_priority_scheduling: false,
            enable_dynamic_batching: true,
            min_batch_size: 1,
            max_batch_size: 256,
            cuda_graph: SchedulerCudaGraphConfig::default(),
            packing: SequencePackingConfig::default(),
        }
    }
}

impl SchedulerConfig {
    /// Returns a builder for configuring this type with the documented field defaults.
    /// Use `with_*(...)` to override individual fields, then `build()` to produce the type.
    #[must_use]
    pub fn builder() -> SchedulerConfigBuilder {
        SchedulerConfigBuilder::default()
    }
}

/// Builder for [`SchedulerConfig`].
#[derive(Debug, Clone, Default)]
pub struct SchedulerConfigBuilder {
    inner: SchedulerConfig,
}

impl SchedulerConfigBuilder {
    /// Set the maximum number of sequences per batch.
    #[must_use]
    pub const fn with_max_num_seqs(mut self, v: usize) -> Self {
        self.inner.max_num_seqs = v;
        self
    }
    /// Set the maximum number of tokens (prompt + generated) per batch.
    #[must_use]
    pub const fn with_max_num_batched_tokens(mut self, v: usize) -> Self {
        self.inner.max_num_batched_tokens = v;
        self
    }
    /// Set the maximum consecutive decode iterations before forcing a prefill.
    #[must_use]
    pub const fn with_max_consecutive_decode(mut self, v: u32) -> Self {
        self.inner.max_consecutive_decode = v;
        self
    }
    /// Toggle prefill/decode batch separation.
    #[must_use]
    pub const fn with_enable_pd_separation(mut self, v: bool) -> Self {
        self.inner.enable_pd_separation = v;
        self
    }
    /// Set the prefill chunk size — the maximum prompt tokens processed in a
    /// single prefill step.
    #[must_use]
    pub const fn with_prefill_chunk_size(mut self, v: usize) -> Self {
        self.inner.prefill_chunk_size = v;
        self
    }
    /// Set the decode-vs-prefill preference ratio (0.0–1.0). Higher values
    /// weight decode latency more heavily when assembling mixed-phase batches.
    #[must_use]
    pub const fn with_decode_preference_ratio(mut self, v: f32) -> Self {
        self.inner.decode_preference_ratio = v;
        self
    }
    /// Toggle priority-based scheduling.
    #[must_use]
    pub const fn with_enable_priority_scheduling(mut self, v: bool) -> Self {
        self.inner.enable_priority_scheduling = v;
        self
    }
    /// Toggle dynamic batching (group similar requests automatically).
    #[must_use]
    pub const fn with_enable_dynamic_batching(mut self, v: bool) -> Self {
        self.inner.enable_dynamic_batching = v;
        self
    }
    /// Set the minimum batch size for dynamic batching.
    #[must_use]
    pub const fn with_min_batch_size(mut self, v: usize) -> Self {
        self.inner.min_batch_size = v;
        self
    }
    /// Set the maximum batch size for dynamic batching.
    #[must_use]
    pub const fn with_max_batch_size(mut self, v: usize) -> Self {
        self.inner.max_batch_size = v;
        self
    }
    /// Override the sequence-packing sub-config.
    #[must_use]
    pub const fn with_packing(mut self, v: SequencePackingConfig) -> Self {
        self.inner.packing = v;
        self
    }
    /// Finalize the builder into a [`SchedulerConfig`].
    #[must_use]
    pub fn build(self) -> SchedulerConfig {
        self.inner
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::must_use_candidate)]
mod tests {
    use super::*;

    #[test]
    fn default_has_documented_values() {
        let cfg = SchedulerConfig::default();
        assert_eq!(cfg.max_num_seqs, 256);
        assert_eq!(cfg.max_num_batched_tokens, 4096);
        assert_eq!(cfg.max_consecutive_decode, 10);
        assert!(cfg.enable_pd_separation);
        assert_eq!(cfg.prefill_chunk_size, 512);
        assert!((cfg.decode_preference_ratio - 0.7).abs() < f32::EPSILON);
        assert!(!cfg.enable_priority_scheduling);
        assert!(cfg.enable_dynamic_batching);
        assert_eq!(cfg.min_batch_size, 1);
        assert_eq!(cfg.max_batch_size, 256);
    }

    #[test]
    fn new_valid_construction() {
        let cfg = SchedulerConfig::new(
            128,
            2048,
            5,
            false,
            256,
            0.5,
            true,
            false,
            1,
            128,
            SequencePackingConfig::default(),
        );
        assert_eq!(cfg.max_num_seqs, 128);
        assert_eq!(cfg.max_num_batched_tokens, 2048);
        assert_eq!(cfg.max_consecutive_decode, 5);
        assert!(!cfg.enable_pd_separation);
        assert_eq!(cfg.prefill_chunk_size, 256);
        assert!((cfg.decode_preference_ratio - 0.5).abs() < f32::EPSILON);
        assert!(cfg.enable_priority_scheduling);
        assert!(!cfg.enable_dynamic_batching);
        assert_eq!(cfg.min_batch_size, 1);
        assert_eq!(cfg.max_batch_size, 128);
    }

    #[test]
    #[should_panic(expected = "max_num_seqs must be > 0")]
    fn new_panics_when_max_num_seqs_is_zero() {
        let _ = SchedulerConfig::new(
            0,
            4096,
            10,
            true,
            512,
            0.7,
            false,
            true,
            1,
            256,
            SequencePackingConfig::default(),
        );
    }

    #[test]
    #[should_panic(expected = "max_num_batched_tokens must be > 0")]
    fn new_panics_when_max_num_batched_tokens_is_zero() {
        let _ = SchedulerConfig::new(
            256,
            0,
            10,
            true,
            512,
            0.7,
            false,
            true,
            1,
            256,
            SequencePackingConfig::default(),
        );
    }

    #[test]
    #[should_panic(expected = "prefill_chunk_size must be > 0")]
    fn new_panics_when_prefill_chunk_size_is_zero() {
        let _ = SchedulerConfig::new(
            256,
            4096,
            10,
            true,
            0,
            0.7,
            false,
            true,
            1,
            256,
            SequencePackingConfig::default(),
        );
    }

    #[test]
    #[should_panic(expected = "decode_preference_ratio must be between 0.0 and 1.0")]
    fn new_panics_when_decode_preference_ratio_is_below_zero() {
        let _ = SchedulerConfig::new(
            256,
            4096,
            10,
            true,
            512,
            -0.1,
            false,
            true,
            1,
            256,
            SequencePackingConfig::default(),
        );
    }

    #[test]
    #[should_panic(expected = "decode_preference_ratio must be between 0.0 and 1.0")]
    fn new_panics_when_decode_preference_ratio_is_above_one() {
        let _ = SchedulerConfig::new(
            256,
            4096,
            10,
            true,
            512,
            1.1,
            false,
            true,
            1,
            256,
            SequencePackingConfig::default(),
        );
    }

    #[test]
    #[should_panic(expected = "min_batch_size must be > 0")]
    fn new_panics_when_min_batch_size_is_zero() {
        let _ = SchedulerConfig::new(
            256,
            4096,
            10,
            true,
            512,
            0.7,
            false,
            true,
            0,
            256,
            SequencePackingConfig::default(),
        );
    }

    #[test]
    #[should_panic(expected = "max_batch_size must be >= min_batch_size")]
    fn new_panics_when_max_batch_size_below_min() {
        let _ = SchedulerConfig::new(
            256,
            4096,
            10,
            true,
            512,
            0.7,
            false,
            true,
            10,
            5,
            SequencePackingConfig::default(),
        );
    }

    #[test]
    #[should_panic(expected = "max_num_batched_tokens must be >= max_batch_size")]
    fn new_panics_when_batched_tokens_below_max_batch() {
        let _ = SchedulerConfig::new(
            256,
            10,
            10,
            true,
            512,
            0.7,
            false,
            true,
            1,
            100,
            SequencePackingConfig::default(),
        );
    }

    #[test]
    fn boundary_ratio_values_accepted() {
        // 0.0 and 1.0 are the boundaries — they should be accepted.
        let _ = SchedulerConfig::new(
            256,
            4096,
            10,
            true,
            512,
            0.0,
            false,
            true,
            1,
            256,
            SequencePackingConfig::default(),
        );
        let _ = SchedulerConfig::new(
            256,
            4096,
            10,
            true,
            512,
            1.0,
            false,
            true,
            1,
            256,
            SequencePackingConfig::default(),
        );
    }

    #[test]
    fn builder_produces_config_with_defaults() {
        let cfg = SchedulerConfig::builder().build();
        assert_eq!(cfg, SchedulerConfig::default());
    }

    #[test]
    fn builder_overrides_individual_fields() {
        let cfg = SchedulerConfig::builder()
            .with_max_num_seqs(64)
            .with_max_num_batched_tokens(512)
            .with_enable_pd_separation(false)
            .with_enable_priority_scheduling(true)
            .build();
        assert_eq!(cfg.max_num_seqs, 64);
        assert_eq!(cfg.max_num_batched_tokens, 512);
        assert!(!cfg.enable_pd_separation);
        assert!(cfg.enable_priority_scheduling);
    }
}
