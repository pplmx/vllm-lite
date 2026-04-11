use super::config::CudaGraphConfig;
use super::CudaGraph;
use std::collections::HashMap;
use vllm_traits::{Batch, BatchOutput, ModelBackend};

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
}

/// Errors that can occur during graph operations
#[derive(Debug, Clone)]
pub enum GraphExecutionError {
    GraphNotFound(usize),
    GraphCaptureFailed(String),
    GraphExecutionFailed(String),
    CudaError(String),
}

impl std::fmt::Display for GraphExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphExecutionError::GraphNotFound(batch_size) => {
                write!(f, "graph not found for batch size {}", batch_size)
            }
            GraphExecutionError::GraphCaptureFailed(msg) => {
                write!(f, "graph capture failed: {}", msg)
            }
            GraphExecutionError::GraphExecutionFailed(msg) => {
                write!(f, "graph execution failed: {}", msg)
            }
            GraphExecutionError::CudaError(msg) => {
                write!(f, "CUDA error: {}", msg)
            }
        }
    }
}

impl std::error::Error for GraphExecutionError {}

impl BatchCudaGraphExecutor {
    /// Create new executor (does not capture graphs yet)
    pub fn new(config: CudaGraphConfig) -> Result<Self, GraphExecutionError> {
        if !config.enabled {
            return Ok(Self {
                graphs: HashMap::new(),
                config,
                enabled: false,
            });
        }
        Ok(Self {
            graphs: HashMap::new(),
            config,
            enabled: true,
        })
    }

    /// Check if CUDA Graph is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Check if graph exists for given batch size
    pub fn has_graph(&self, batch_size: usize) -> bool {
        self.graphs.contains_key(&batch_size)
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
    pub fn capture_all_graphs<M: ModelBackend>(
        &mut self,
        model: &mut M,
    ) -> Result<(), GraphExecutionError> {
        if !self.enabled {
            return Ok(());
        }
        let batch_sizes: Vec<usize> = self.config.batch_sizes.clone();
        for &batch_size in &batch_sizes {
            self.capture_graph_for_batch_size(batch_size, model)?;
        }
        tracing::info!(
            "CUDA Graphs captured for batch sizes: {:?}",
            self.available_batch_sizes()
        );
        Ok(())
    }

    /// Capture graph for specific batch size
    fn capture_graph_for_batch_size<M: ModelBackend>(
        &mut self,
        batch_size: usize,
        _model: &mut M,
    ) -> Result<(), GraphExecutionError> {
        // For now, create mock graph (actual CUDA integration in future phase)
        let mut graph = CudaGraph::new();
        graph
            .capture()
            .map_err(|e| GraphExecutionError::GraphCaptureFailed(e.to_string()))?;
        self.graphs.insert(batch_size, graph);
        tracing::debug!("Captured graph for batch_size={}", batch_size);
        Ok(())
    }

    /// Execute graph for batch
    pub fn execute(&self, batch: &Batch) -> Result<BatchOutput, GraphExecutionError> {
        if !self.enabled {
            return Err(GraphExecutionError::GraphExecutionFailed(
                "CUDA Graph not enabled".to_string(),
            ));
        }
        let batch_size = batch.seq_ids.len();
        let graph = self
            .graphs
            .get(&batch_size)
            .ok_or_else(|| GraphExecutionError::GraphNotFound(batch_size))?;
        // For now, return mock output (actual execution in future phase)
        let mut tensors: Vec<Box<dyn crate::kernels::cuda_graph::CudaGraphTensor>> = vec![];
        graph
            .execute(&mut tensors)
            .map_err(|e| GraphExecutionError::GraphExecutionFailed(e.to_string()))?;
        // Return dummy output
        Ok(BatchOutput {
            seq_ids: batch.seq_ids.clone(),
            next_tokens: vec![0u32; batch_size],
        })
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
        use vllm_traits::{BatchOutput, ModelBackend, ModelError};

        let mut executor = BatchCudaGraphExecutor::new(CudaGraphConfig::default()).unwrap();
        assert_eq!(executor.graph_count(), 0);

        // Create mock model
        struct MockModel;

        impl ModelBackend for MockModel {
            fn forward(
                &mut self,
                _seq_ids: &[u64],
                _input_tokens: &[Vec<u32>],
                _positions: &[Vec<usize>],
                _kv_block_ids: &[Vec<usize>],
                _num_computed_tokens: &[usize],
                _is_prefill: &[bool],
            ) -> Result<BatchOutput, ModelError> {
                Ok(BatchOutput {
                    seq_ids: vec![],
                    next_tokens: vec![],
                })
            }

            fn forward_logits(
                &mut self,
                _seq_ids: &[u64],
                _input_tokens: &[Vec<u32>],
                _positions: &[Vec<usize>],
                _kv_block_ids: &[Vec<usize>],
                _num_computed_tokens: &[usize],
                _is_prefill: &[bool],
            ) -> Result<Vec<Vec<f32>>, ModelError> {
                Ok(vec![])
            }

            fn embed(
                &mut self,
                _input_tokens: &[Vec<u32>],
                _positions: &[Vec<usize>],
            ) -> Result<Vec<Vec<f32>>, ModelError> {
                Ok(vec![])
            }
        }

        let mut model = MockModel;
        executor
            .capture_graph_for_batch_size(1, &mut model)
            .unwrap();
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
}
