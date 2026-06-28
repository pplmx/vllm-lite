//! Top-level scheduler configuration.

use crate::scheduler::cuda_graph::SchedulerCudaGraphConfig;
use crate::types::sequence_packing::SequencePackingConfig;

/// Configuration for the request scheduler.
///
/// Controls batching behavior, prefill/decode separation, and priority handling.
#[derive(Clone, Debug)]
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
    #[must_use]
    pub const fn with_max_num_seqs(mut self, v: usize) -> Self {
        self.inner.max_num_seqs = v;
        self
    }
    #[must_use]
    pub const fn with_max_num_batched_tokens(mut self, v: usize) -> Self {
        self.inner.max_num_batched_tokens = v;
        self
    }
    #[must_use]
    pub const fn with_max_consecutive_decode(mut self, v: u32) -> Self {
        self.inner.max_consecutive_decode = v;
        self
    }
    #[must_use]
    pub const fn with_enable_pd_separation(mut self, v: bool) -> Self {
        self.inner.enable_pd_separation = v;
        self
    }
    #[must_use]
    pub const fn with_prefill_chunk_size(mut self, v: usize) -> Self {
        self.inner.prefill_chunk_size = v;
        self
    }
    #[must_use]
    pub const fn with_decode_preference_ratio(mut self, v: f32) -> Self {
        self.inner.decode_preference_ratio = v;
        self
    }
    #[must_use]
    pub const fn with_enable_priority_scheduling(mut self, v: bool) -> Self {
        self.inner.enable_priority_scheduling = v;
        self
    }
    #[must_use]
    pub const fn with_enable_dynamic_batching(mut self, v: bool) -> Self {
        self.inner.enable_dynamic_batching = v;
        self
    }
    #[must_use]
    pub const fn with_min_batch_size(mut self, v: usize) -> Self {
        self.inner.min_batch_size = v;
        self
    }
    #[must_use]
    pub const fn with_max_batch_size(mut self, v: usize) -> Self {
        self.inner.max_batch_size = v;
        self
    }
    #[must_use]
    pub fn with_cuda_graph(mut self, v: SchedulerCudaGraphConfig) -> Self {
        self.inner.cuda_graph = v;
        self
    }
    #[must_use]
    pub const fn with_packing(mut self, v: SequencePackingConfig) -> Self {
        self.inner.packing = v;
        self
    }
    /// build: build the [`SchedulerConfig`].
    #[must_use]
    pub fn build(self) -> SchedulerConfig {
        self.inner
    }
}
