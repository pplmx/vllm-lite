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
#[derive(Clone, Debug)]
pub struct SchedulerCudaGraphConfig {
    /// Enable CUDA Graph for decode
    pub enabled: bool,
    /// Batch sizes to capture (must match `CudaGraphExecutor`)
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
    /// Override [`SchedulerCudaGraphConfig::batch_sizes`].
    #[must_use]
    pub fn with_batch_sizes(mut self, v: Vec<usize>) -> Self {
        self.inner.batch_sizes = v;
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
    use vllm_traits::{Batch, BatchPhase, SeqId};

    fn create_test_batch(batch_size: usize) -> Batch {
        Batch {
            seq_ids: (0..batch_size as u64).collect::<Vec<SeqId>>(),
            input_tokens: vec![vec![1u32]; batch_size],
            positions: vec![vec![0usize]; batch_size],
            kv_block_ids: vec![vec![]; batch_size],
            num_computed_tokens: vec![0; batch_size],
            is_prefill: vec![false; batch_size],
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

    #[test]
    fn test_config_supports_batch_size() {
        let config = SchedulerCudaGraphConfig::default();
        assert!(config.supports_batch_size(1));
        assert!(config.supports_batch_size(4));
        assert!(config.supports_batch_size(8));
        assert!(!config.supports_batch_size(3));
    }
}
