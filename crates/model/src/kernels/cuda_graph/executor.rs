use super::config::CudaGraphConfig;
use super::CudaGraph;
use std::collections::HashMap;

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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
