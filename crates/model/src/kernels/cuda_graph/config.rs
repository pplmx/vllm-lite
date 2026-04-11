/// Configuration for CUDA Graph integration
#[derive(Clone, Debug)]
pub struct CudaGraphConfig {
    /// Enable CUDA Graph execution
    pub enabled: bool,
    /// Predefined batch sizes to capture graphs for
    pub batch_sizes: Vec<usize>,
    /// Model-specific configuration for graph capture
    pub model_config: ModelGraphConfig,
}

/// Model-specific CUDA Graph configuration
#[derive(Clone, Debug)]
pub struct ModelGraphConfig {
    /// Maximum sequence length supported
    pub max_seq_len: usize,
    /// Number of KV blocks
    pub num_kv_blocks: usize,
    /// Whether to capture attention separately
    pub capture_attention_separate: bool,
}

impl Default for CudaGraphConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            batch_sizes: vec![1, 4, 8, 16, 32, 64],
            model_config: ModelGraphConfig::default(),
        }
    }
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
    /// Create config from environment variables
    pub fn from_env() -> Self {
        let enabled = std::env::var("VLLM_CUDA_GRAPH_ENABLED")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(false);

        let batch_sizes = std::env::var("VLLM_CUDA_GRAPH_BATCH_SIZES")
            .ok()
            .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
            .unwrap_or_else(|| vec![1, 4, 8, 16, 32, 64]);

        Self {
            enabled,
            batch_sizes,
            ..Default::default()
        }
    }
}
