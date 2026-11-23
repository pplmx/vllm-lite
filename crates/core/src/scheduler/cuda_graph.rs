//! CUDA Graph integration for the scheduler
//!
//! This module provides integration between the `PhaseScheduler` and CUDA Graph
//! execution, routing decode batches through captured graphs when available.

use vllm_traits::Batch;

/// Graph-aware batch wrapper
#[derive(Debug)]
pub enum GraphBatch {
    /// Batch that can be executed via CUDA Graph
    Graph(GraphPreparedBatch),
    /// Regular batch requiring standard execution
    Regular(Batch),
}

impl GraphBatch {
    /// Convert to regular batch
    #[must_use]
    pub fn into_regular(self) -> Batch {
        match self {
            Self::Graph(prepared) => prepared.into_batch(),
            Self::Regular(batch) => batch,
        }
    }

    /// Check if this is a graph batch
    #[must_use]
    pub const fn is_graph(&self) -> bool {
        matches!(self, Self::Graph(_))
    }

    /// Get batch size
    #[must_use]
    pub const fn batch_size(&self) -> usize {
        match self {
            Self::Graph(prepared) => prepared.batch_size,
            Self::Regular(batch) => batch.seq_ids.len(),
        }
    }
}

/// Batch prepared for CUDA Graph execution
#[derive(Debug)]
pub struct GraphPreparedBatch {
    /// Original batch
    pub batch: Batch,
    /// Batch size (cached for lookup)
    pub batch_size: usize,
}

impl GraphPreparedBatch {
    /// Wrap a `Batch` for graph execution, caching its size to avoid
    /// recomputing it on every dispatch.
    #[must_use]
    pub const fn new(batch: Batch) -> Self {
        let batch_size = batch.seq_ids.len();
        Self { batch, batch_size }
    }

    /// Unwrap back into the underlying [`Batch`].
    #[must_use]
    pub fn into_batch(self) -> Batch {
        self.batch
    }
}

/// Configuration for CUDA Graph in scheduler
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchedulerCudaGraphConfig {
    /// Enable CUDA Graph capture/replay for decode batches.
    pub enabled: bool,
    /// Batch sizes to pre-capture (must match `CudaGraphExecutor`).
    pub batch_sizes: Vec<usize>,
}

impl Default for SchedulerCudaGraphConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            batch_sizes: vec![1, 4, 8, 16, 32, 64],
        }
    }
}

impl SchedulerCudaGraphConfig {
    /// Check if batch size is supported
    #[must_use]
    pub fn supports_batch_size(&self, batch_size: usize) -> bool {
        self.batch_sizes.contains(&batch_size)
    }

    /// Returns a builder for configuring this type with the documented field defaults.
    /// Use `with_*(...)` to override individual fields, then `build()` to produce the type.
    #[must_use]
    pub fn builder() -> SchedulerCudaGraphConfigBuilder {
        SchedulerCudaGraphConfigBuilder::default()
    }
}

/// Builder for [`SchedulerCudaGraphConfig`].
#[derive(Debug, Clone, Default)]
pub struct SchedulerCudaGraphConfigBuilder {
    inner: SchedulerCudaGraphConfig,
}

impl SchedulerCudaGraphConfigBuilder {
    /// Set [`SchedulerCudaGraphConfig::enabled`].
    #[must_use]
    pub const fn with_enabled(mut self, v: bool) -> Self {
        self.inner.enabled = v;
        self
    }
    /// Finalize the builder into a [`SchedulerCudaGraphConfig`].
    #[must_use]
    pub fn build(self) -> SchedulerCudaGraphConfig {
        self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vllm_traits::{Batch, BatchPhase, SamplingParams, SeqId};

    fn create_test_batch(batch_size: usize) -> Batch {
        Batch {
            seq_ids: (0..batch_size as u64).collect::<Vec<SeqId>>(),
            input_tokens: vec![vec![1u32]; batch_size],
            positions: vec![vec![0usize]; batch_size],
            kv_block_ids: vec![vec![]; batch_size],
            num_computed_tokens: vec![0; batch_size],
            is_prefill: vec![false; batch_size],
            sampling_params: vec![SamplingParams::default(); batch_size],
            phase: BatchPhase::Decode,
            total_tokens: batch_size,
            max_seq_len: 1,
        }
    }

    #[test]
    fn test_graph_batch_is_graph() {
        let batch = create_test_batch(4);
        let graph_batch = GraphBatch::Graph(GraphPreparedBatch::new(batch));
        assert!(graph_batch.is_graph());
    }

    #[test]
    fn test_regular_batch_is_not_graph() {
        let batch = create_test_batch(4);
        let graph_batch = GraphBatch::Regular(batch);
        assert!(!graph_batch.is_graph());
    }

    /// `batch_size()` must report the batch's real size for BOTH variants.
    ///
    /// The Graph arm reads its cached `batch_size` and the Regular arm counts
    /// `seq_ids`. `cargo mutants` replaced the whole method with the constants
    /// 0 and 1 and both survived: the only caller in the suite is
    /// `cuda_graph_integration.rs`, whose single assertion uses a size-3
    /// Regular batch and therefore misses both constants. That matters because
    /// `graph_step` uses `batch_size() == 0` as its empty-batch guard — a
    /// constant 0 would skip every real batch and a constant 1 would dispatch
    /// an empty one. Three sizes across BOTH variants make each constant fail.
    #[test]
    fn test_batch_size_reports_the_real_size_for_both_variants() {
        for size in [1usize, 3, 7] {
            let batch = create_test_batch(size);
            let regular = GraphBatch::Regular(batch);
            assert_eq!(
                regular.batch_size(),
                size,
                "Regular must count seq_ids, not return a constant"
            );

            let batch = create_test_batch(size);
            let graph = GraphBatch::Graph(GraphPreparedBatch::new(batch));
            assert_eq!(
                graph.batch_size(),
                size,
                "Graph must report its cached size, not return a constant"
            );
        }
    }

    /// `builder().build()` must carry every `with_*` override through.
    ///
    /// `cargo mutants` replaced `build` with `Default::default()` and it
    /// survived: no test used the builder at all, so a scheduler configured
    /// through it would silently run with the DEFAULT cuda-graph settings —
    /// `with_enabled(false)` ignored, i.e. captured graphs replayed for an
    /// operator who explicitly disabled them.
    #[test]
    fn test_builder_applies_overrides_and_defaults() {
        let built = SchedulerCudaGraphConfig::builder()
            .with_enabled(false)
            .build();
        assert!(!built.enabled, "with_enabled(false) must reach the config");

        let built = SchedulerCudaGraphConfig::builder()
            .with_enabled(true)
            .build();
        assert!(built.enabled);

        // The fields `build` must NOT reset stay at their documented defaults.
        let default = SchedulerCudaGraphConfig::default();
        assert_eq!(
            built.batch_sizes, default.batch_sizes,
            "untouched fields keep the documented defaults"
        );

        // An unconfigured builder is the default config.
        let plain = SchedulerCudaGraphConfig::builder().build();
        assert_eq!(plain.enabled, default.enabled);
        assert_eq!(plain.batch_sizes, default.batch_sizes);
    }

    #[test]
    fn test_config_supports_batch_size() {
        let config = SchedulerCudaGraphConfig::default();
        assert!(config.supports_batch_size(1));
        assert!(config.supports_batch_size(4));
        assert!(config.supports_batch_size(8));
        assert!(!config.supports_batch_size(3));
    }
}
