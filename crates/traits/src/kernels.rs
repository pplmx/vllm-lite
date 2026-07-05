//! Kernel and CUDA-Graph trait definitions consumed by `vllm-core` and `vllm-model`.
//!
//! Pure-data: configuration structs and an error enum. The actual capture /
//! replay logic lives in `vllm-model::kernels::cuda_graph`. This module is
//! the workspace-wide wire format for those configs.
use serde::{Deserialize, Serialize};

/// Configuration for CudaGraph. Constructed via the `builder()` associated function or by deserializing from JSON / TOML. Pass-by-value to construction APIs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CudaGraphConfig {
    /// Whether CUDA Graph capture/replay is enabled.
    pub enabled: bool,
    /// Batch sizes to capture (one graph per size).
    pub batch_sizes: Vec<usize>,
    /// Model-graph sub-config (seq len, KV blocks, etc.).
    pub model_config: ModelGraphConfig,
    /// Enable memory-pool sharing across graphs (default true; None = use runtime default).
    pub enable_graph_pooling: Option<bool>,
}

impl Default for CudaGraphConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            batch_sizes: vec![1, 4, 8, 16, 32, 64],
            model_config: ModelGraphConfig::default(),
            enable_graph_pooling: Some(true),
        }
    }
}

/// Configuration for ModelGraph. Constructed via the `builder()` associated function or by deserializing from JSON / TOML. Pass-by-value to construction APIs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelGraphConfig {
    /// Maximum sequence length the captured graph supports.
    pub max_seq_len: usize,
    /// KV-cache block count for the captured graph.
    pub num_kv_blocks: usize,
    /// Whether attention is captured separately (memory savings vs. flexibility).
    pub capture_attention_separate: bool,
}

impl Default for ModelGraphConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 8192,
            num_kv_blocks: 1024,
            capture_attention_separate: false,
        }
    }
}

impl CudaGraphConfig {
    #[must_use]
    pub fn from_env() -> Self {
        let enabled = std::env::var("VLLM_CUDA_GRAPH_ENABLED")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(false);

        let batch_sizes = std::env::var("VLLM_CUDA_GRAPH_BATCH_SIZES")
            .ok()
            .map_or_else(
                || vec![1, 4, 8, 16, 32, 64],
                |v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect(),
            );

        let enable_graph_pooling = std::env::var("VLLM_CUDA_GRAPH_POOLING")
            .ok()
            .and_then(|v| v.parse().ok());

        Self {
            enabled,
            batch_sizes,
            model_config: ModelGraphConfig::default(),
            enable_graph_pooling,
        }
    }
}

/// Error type for GraphExecution. Returned from every fallible public API; covers I/O, validation, and resource-limit failures. Use [`Result<T>`] alias in the same module.
#[derive(Debug, Clone, thiserror::Error)]
pub enum GraphExecutionError {
    #[error("graph not found for batch size {0}")]
    GraphNotFound(usize),
    #[error("graph capture failed: {0}")]
    GraphCaptureFailed(String),
    #[error("graph execution failed: {0}")]
    GraphExecutionFailed(String),
    #[error("CUDA error: {0}")]
    CudaError(String),
}
