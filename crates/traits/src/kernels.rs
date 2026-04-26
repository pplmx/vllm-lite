use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CudaGraphConfig {
    pub enabled: bool,
    pub batch_sizes: Vec<usize>,
    pub model_config: ModelGraphConfig,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelGraphConfig {
    pub max_seq_len: usize,
    pub num_kv_blocks: usize,
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
    pub fn from_env() -> Self {
        let enabled = std::env::var("VLLM_CUDA_GRAPH_ENABLED")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(false);

        let batch_sizes = std::env::var("VLLM_CUDA_GRAPH_BATCH_SIZES")
            .ok()
            .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
            .unwrap_or_else(|| vec![1, 4, 8, 16, 32, 64]);

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
