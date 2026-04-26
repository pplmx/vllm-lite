use super::CudaGraph;
use super::config::CudaGraphConfig;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use vllm_traits::kernels::GraphExecutionError;
use vllm_traits::{Batch, BatchOutput};

/// Executor for managing CUDA Graph capture and execution
///
/// This executor uses batch_size (usize) keys and is designed for scheduler integration.
/// For the string-keyed executor, see the `CudaGraphExecutor` type in the parent module.
pub struct BatchCudaGraphExecutor {
    /// Map from batch_size to captured graph
    graphs: HashMap<usize, CudaGraph>,
    /// Configuration
    #[allow(dead_code)]
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

    /// Check if CUDA Graph is enabled
    pub fn is_enabled(&self) -> bool {
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

    /// Get number of captured graphs
    pub fn graph_count(&self) -> usize {
        self.graphs.len()
    }

    /// Get list of available batch sizes
    pub fn available_batch_sizes(&self) -> Vec<usize> {
        let mut sizes: Vec<_> = self.graphs.keys().copied().collect();
        sizes.sort();
        sizes
    }

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
    pub fn stats(&self) -> GraphStats {
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

        let graph = self.graphs.get(&graph_key).unwrap();
        let mut tensors: Vec<Box<dyn crate::kernels::cuda_graph::CudaGraphTensor>> = vec![];
        graph
            .execute(&mut tensors)
            .map_err(|e| GraphExecutionError::GraphExecutionFailed(e.to_string()))?;

        Ok(BatchOutput {
            seq_ids: batch.seq_ids.clone(),
            next_tokens: vec![0u32; batch_size],
        })
    }
}

#[derive(Debug, Clone)]
pub struct GraphStats {
    pub total_executions: u64,
    pub cache_hits: u64,
    pub cached_graphs: usize,
}

impl GraphStats {
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
        assert_eq!(stats.cache_hit_rate(), 0.0);
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
