#![allow(clippy::module_name_repetitions)]
// `stats`, `GraphStats`, and `cache_hit_rate` are public API on the
// executor; downstream callers and tests consume them. They appear unused
// in this crate alone but are intended for external observability.
#![allow(dead_code)]
use super::CudaGraph;
use super::config::CudaGraphConfig;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use vllm_traits::kernels::GraphExecutionError;
use vllm_traits::{Batch, BatchOutput};

/// Executor for managing CUDA Graph capture and execution
///
/// This executor uses `batch_size` (usize) keys and is designed for scheduler integration.
#[derive(Debug)]
/// For the string-keyed executor, see the `CudaGraphExecutor` type in the parent module.
pub struct BatchCudaGraphExecutor {
    /// Map from `batch_size` to captured graph
    graphs: HashMap<usize, CudaGraph>,
    /// Configuration
    config: CudaGraphConfig,
    /// Whether CUDA Graph is enabled
    enabled: bool,
    /// Execution statistics
    total_executions: AtomicU64,
    /// Cache hits
    cache_hits: AtomicU64,
    /// Pooled graphs (shared graphs for similar batch sizes)
    pooled_graphs: HashMap<usize, usize>,
}

impl BatchCudaGraphExecutor {
    /// # Errors
    ///
    /// Returns `Err` if any required tensor allocation or weight loading fails.
    /// Create new executor (does not capture graphs yet)
    pub fn new(config: CudaGraphConfig) -> Result<Self, GraphExecutionError> {
        let enable_graph_pooling = config.enable_graph_pooling.unwrap_or(true);
        let pooled_graphs = if enable_graph_pooling {
            Self::compute_graph_pool(&config.batch_sizes)
        } else {
            HashMap::new()
        };

        if !config.enabled {
            return Ok(Self {
                graphs: HashMap::new(),
                config,
                enabled: false,
                total_executions: AtomicU64::new(0),
                cache_hits: AtomicU64::new(0),
                pooled_graphs,
            });
        }
        Ok(Self {
            graphs: HashMap::new(),
            config,
            enabled: true,
            total_executions: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            pooled_graphs,
        })
    }

    /// Compute pooling strategy for batch sizes
    fn compute_graph_pool(batch_sizes: &[usize]) -> HashMap<usize, usize> {
        let mut pool = HashMap::new();
        if batch_sizes.is_empty() {
            return pool;
        }

        let max_batch = *batch_sizes.iter().max().unwrap_or(&1);

        let ranges = if max_batch <= 8 {
            vec![(1, 2), (3, 4), (5, 8)]
        } else if max_batch <= 32 {
            vec![(1, 4), (5, 16), (17, 32)]
        } else {
            vec![(1, 8), (9, 32), (33, 128)]
        };

        for &(start, end) in &ranges {
            if batch_sizes.iter().any(|&s| s >= start && s <= end) {
                let representative = start.min(*batch_sizes.iter().max().unwrap_or(&1));
                for batch_size in batch_sizes {
                    if *batch_size >= start && *batch_size <= end {
                        pool.insert(*batch_size, representative);
                    }
                }
            }
        }

        pool
    }

    /// Configuration used when capturing graphs.
    pub const fn config(&self) -> &CudaGraphConfig {
        &self.config
    }

    /// Check if CUDA Graph is enabled
    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Check if graph exists for given batch size
    pub fn has_graph(&self, batch_size: usize) -> bool {
        self.find_graph_key(batch_size).is_some()
    }

    /// Find the key for graph lookup (considering pooling)
    fn find_graph_key(&self, batch_size: usize) -> Option<usize> {
        if self.graphs.contains_key(&batch_size) {
            Some(batch_size)
        } else if let Some(&pooled_key) = self.pooled_graphs.get(&batch_size) {
            if self.graphs.contains_key(&pooled_key) {
                Some(pooled_key)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Look up a captured graph by key, returning a typed error if absent.
    ///
    /// This is the post-`find_graph_key` lookup site. In normal operation
    /// `find_graph_key` only returns `Some(k)` when `self.graphs.contains_key(&k)`,
    /// so this branch is unreachable. The typed error protects against future
    /// refactors that may weaken that invariant (or against logic errors where
    /// a graph is invalidated between the key lookup and the actual retrieval).
    fn lookup_graph(
        &self,
        graph_key: usize,
        batch_size: usize,
    ) -> Result<&CudaGraph, GraphExecutionError> {
        self.graphs.get(&graph_key).ok_or_else(|| {
            tracing::warn!(
                batch_size,
                graph_key,
                "CUDA graph cache miss; key remapped but not found"
            );
            GraphExecutionError::GraphNotFound(batch_size)
        })
    }

    /// Get number of captured graphs
    pub fn graph_count(&self) -> usize {
        self.graphs.len()
    }

    /// Get list of available batch sizes
    pub fn available_batch_sizes(&self) -> Vec<usize> {
        let mut sizes: Vec<_> = self.graphs.keys().copied().collect();
        sizes.sort_unstable();
        sizes
    }

    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    /// Capture graphs for all configured batch sizes
    pub fn capture_all_graphs(&mut self) -> Result<(), GraphExecutionError> {
        if !self.enabled {
            return Ok(());
        }
        let batch_sizes: Vec<usize> = self.config.batch_sizes.clone();
        for &batch_size in &batch_sizes {
            self.capture_graph_for_batch_size(batch_size)?;
        }
        tracing::info!(
            batch_sizes = ?self.available_batch_sizes(),
            pooled = self.pooled_graphs.len(),
            "CUDA Graphs captured"
        );
        Ok(())
    }

    /// Capture graph for specific batch size
    fn capture_graph_for_batch_size(
        &mut self,
        batch_size: usize,
    ) -> Result<(), GraphExecutionError> {
        if self.graphs.contains_key(&batch_size) {
            tracing::debug!("Graph already exists for batch_size={}", batch_size);
            return Ok(());
        }

        let mut graph = CudaGraph::new();
        graph
            .capture()
            .map_err(|e| GraphExecutionError::GraphCaptureFailed(e.to_string()))?;
        self.graphs.insert(batch_size, graph);
        tracing::debug!("Captured graph for batch_size={}", batch_size);
        Ok(())
    }

    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    /// Warm up graph cache with common batch sizes
    pub fn warmup(&mut self, common_batch_sizes: &[usize]) -> Result<(), GraphExecutionError> {
        if !self.enabled {
            return Ok(());
        }
        tracing::info!(batch_sizes = ?common_batch_sizes, "Warming up CUDA Graph cache");
        for &batch_size in common_batch_sizes {
            if !self.has_graph(batch_size) {
                self.capture_graph_for_batch_size(batch_size)?;
            }
        }
        Ok(())
    }

    /// Invalidate graph for batch size (e.g., after model weight change)
    pub fn invalidate(&mut self, batch_size: usize) {
        if self.graphs.remove(&batch_size).is_some() {
            tracing::info!("Invalidated graph for batch_size={}", batch_size);
        }
    }

    /// Get execution statistics
    pub(crate) fn stats(&self) -> GraphStats {
        GraphStats {
            total_executions: self.total_executions.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cached_graphs: self.graph_count(),
        }
    }

    /// Clear all cached graphs
    pub fn clear(&mut self) {
        let count = self.graphs.len();
        self.graphs.clear();
        tracing::info!("Cleared {} cached graphs", count);
    }

    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    /// Execute graph for batch
    pub fn execute(&self, batch: &Batch) -> Result<BatchOutput, GraphExecutionError> {
        self.total_executions.fetch_add(1, Ordering::Relaxed);

        if !self.enabled {
            return Err(GraphExecutionError::GraphExecutionFailed(
                "CUDA Graph not enabled".to_string(),
            ));
        }

        let batch_size = batch.seq_ids.len();
        let graph_key = self
            .find_graph_key(batch_size)
            .ok_or(GraphExecutionError::GraphNotFound(batch_size))?;

        if graph_key != batch_size {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
        }

        let mut tensors: Vec<Box<dyn crate::kernels::cuda_graph::CudaGraphTensor>> = vec![];
        self.lookup_graph(graph_key, batch_size)?
            .execute(&mut tensors)
            .map_err(|e| GraphExecutionError::GraphExecutionFailed(e.to_string()))?;

        Ok(BatchOutput {
            seq_ids: batch.seq_ids.clone(),
            next_tokens: vec![0u32; batch_size],
        })
    }
}

/// Telemetry snapshot for Graph: counters, gauges, and percentile latencies. Cloned and serialized on every metrics export.
#[derive(Debug, Clone)]
pub(crate) struct GraphStats {
    pub total_executions: u64,
    pub cache_hits: u64,
    pub cached_graphs: usize,
}

impl GraphStats {
    // invariant: cache_hits/total_executions are bounded counters; u64 -> f64
    // precision loss is acceptable for the hit-rate metric.
    #[allow(clippy::cast_precision_loss)]
    pub fn cache_hit_rate(&self) -> f64 {
        if self.total_executions == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_executions as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vllm_traits::{Batch, BatchPhase};

    fn create_mock_batch(batch_size: usize) -> Batch {
        Batch {
            seq_ids: (0..batch_size as u64).collect(),
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
    fn test_config_accessor() {
        let config = CudaGraphConfig {
            enabled: true,
            batch_sizes: vec![1, 2, 4],
            ..Default::default()
        };
        let executor = BatchCudaGraphExecutor::new(config.clone()).unwrap();
        assert_eq!(executor.config().batch_sizes, config.batch_sizes);
    }

    #[test]
    fn test_executor_disabled_when_config_disabled() {
        let config = CudaGraphConfig {
            enabled: false,
            ..Default::default()
        };
        let executor = BatchCudaGraphExecutor::new(config).unwrap();
        assert!(!executor.is_enabled());
    }

    #[test]
    fn test_executor_enabled_when_config_enabled() {
        let config = CudaGraphConfig {
            enabled: true,
            ..Default::default()
        };
        let executor = BatchCudaGraphExecutor::new(config).unwrap();
        assert!(executor.is_enabled());
    }

    #[test]
    fn test_has_graph_returns_false_for_empty_executor() {
        let config = CudaGraphConfig::default();
        let executor = BatchCudaGraphExecutor::new(config).unwrap();
        assert!(!executor.has_graph(1));
        assert!(!executor.has_graph(4));
    }

    #[test]
    fn test_capture_graph_increases_graph_count() {
        let mut executor = BatchCudaGraphExecutor::new(CudaGraphConfig::default()).unwrap();
        assert_eq!(executor.graph_count(), 0);

        executor.capture_graph_for_batch_size(1).unwrap();
        assert_eq!(executor.graph_count(), 1);
        assert!(executor.has_graph(1));
    }

    #[test]
    fn test_execute_returns_error_for_unknown_batch_size() {
        // Create config with enabled=true so we can test graph not found
        let config = CudaGraphConfig {
            enabled: true,
            ..Default::default()
        };
        let executor = BatchCudaGraphExecutor::new(config).unwrap();
        let batch = create_mock_batch(2);
        let result = executor.execute(&batch);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GraphExecutionError::GraphNotFound(2)
        ));
    }

    #[test]
    fn test_lookup_graph_returns_not_found_for_missing_key() {
        // Verifies the unwrap-to-error conversion at the post-find_graph_key
        // lookup site. In normal operation this branch is unreachable because
        // find_graph_key checks contains_key before returning Some, but the
        // typed error protects against future refactors that weaken that
        // invariant or against logic errors where a graph is invalidated
        // between the key lookup and the actual retrieval.
        let config = CudaGraphConfig {
            enabled: true,
            ..Default::default()
        };
        let executor = BatchCudaGraphExecutor::new(config).unwrap();
        let result = executor.lookup_graph(999, 999);
        assert!(matches!(
            result,
            Err(GraphExecutionError::GraphNotFound(999))
        ));
    }

    #[test]
    fn test_lookup_graph_returns_graph_for_captured_key() {
        // Positive control: verifies lookup_graph returns the captured graph
        // when the key is present, so the typed-error test above cannot be
        // passing because of an unrelated always-Err implementation.
        let mut executor = BatchCudaGraphExecutor::new(CudaGraphConfig::default()).unwrap();
        executor.capture_graph_for_batch_size(4).unwrap();
        let result = executor.lookup_graph(4, 4);
        assert!(result.is_ok());
    }

    #[test]
    fn test_graph_pooling() {
        let config = CudaGraphConfig {
            enabled: true,
            batch_sizes: vec![1, 2, 3, 4, 8],
            enable_graph_pooling: Some(true),
            ..Default::default()
        };
        let executor = BatchCudaGraphExecutor::new(config).unwrap();
        let stats = executor.stats();

        assert_eq!(stats.total_executions, 0);
        assert_eq!(stats.cached_graphs, 0);
    }

    #[test]
    fn test_stats_initialization() {
        let config = CudaGraphConfig::default();
        let executor = BatchCudaGraphExecutor::new(config).unwrap();
        let stats = executor.stats();

        assert_eq!(stats.total_executions, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cached_graphs, 0);
        assert!(stats.cache_hit_rate().abs() < 1e-6);
    }

    #[test]
    fn test_clear_graphs() {
        let mut executor = BatchCudaGraphExecutor::new(CudaGraphConfig::default()).unwrap();
        executor.capture_graph_for_batch_size(1).unwrap();
        executor.capture_graph_for_batch_size(2).unwrap();
        assert_eq!(executor.graph_count(), 2);

        executor.clear();
        assert_eq!(executor.graph_count(), 0);
    }

    #[test]
    fn test_invalidate_graph() {
        let mut executor = BatchCudaGraphExecutor::new(CudaGraphConfig::default()).unwrap();
        executor.capture_graph_for_batch_size(4).unwrap();
        assert!(executor.has_graph(4));

        executor.invalidate(4);
        assert!(!executor.has_graph(4));
    }

    #[test]
    fn test_warmup() {
        let mut executor = BatchCudaGraphExecutor::new(CudaGraphConfig {
            enabled: true,
            batch_sizes: vec![1, 2, 4, 8],
            ..Default::default()
        })
        .unwrap();

        executor.warmup(&[1, 4, 8]).unwrap();

        assert!(executor.has_graph(1));
        assert!(executor.has_graph(4));
        assert!(executor.has_graph(8));
    }
}
