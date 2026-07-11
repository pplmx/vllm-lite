//! Kernel and CUDA-Graph trait definitions consumed by `vllm-core` and `vllm-model`.
//!
//! Pure-data: configuration structs, an error enum, and the
//! [`CudaGraphExecutor`] trait. The actual capture / replay logic lives in
//! `vllm-model::kernels::cuda_graph`. This module is the workspace-wide wire
//! format for those configs and the abstract interface every engine talks to.
//!
//! See [`CudaGraphExecutor`] for the trait that hides `BatchCudaGraphExecutor`
//! behind a trait object — the contract `vllm-core` depends on. Phase 18
//! ARCH-06 closes the `core → model` upward dependency via cuda-graph.
use serde::{Deserialize, Serialize};

use crate::types::{Batch, BatchOutput};

/// Configuration for `CudaGraph`. Constructed via the `builder()` associated function or by deserializing from JSON / TOML. Pass-by-value to construction APIs.
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

/// Configuration for `ModelGraph`. Constructed via the `builder()` associated function or by deserializing from JSON / TOML. Pass-by-value to construction APIs.
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

/// Error type for `GraphExecution`. Returned from every fallible public API; covers I/O, validation, and resource-limit failures. Use [`Result<T>`] alias in the same module.
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

/// Abstract interface over a CUDA-Graph executor.
///
/// `vllm-core::engine::Engine` holds an `Option<Box<dyn CudaGraphExecutor + Send>>`
/// rather than a concrete `vllm_model::kernels::BatchCudaGraphExecutor`. The
/// concrete type lives below the `core → model` boundary; the trait above it
/// is what every consumer in `core` talks to.
///
/// This is the abstraction that closes the `core → model` upward dependency
/// surfaced by the `cuda-graph` feature (Phase 18 ARCH-06). Without it, every
/// `core` call site that touched the executor had to know the concrete model
/// type. With it, the only place that imports `BatchCudaGraphExecutor` is
/// the engine constructor that builds the concrete value before boxing it
/// into the trait object.
///
/// # Object safety
///
/// `+ Send` is required so the boxed executor can move between threads if a
/// future caller spawns the engine loop on a worker thread. The trait has no
/// generic methods and no `Self` in return position, so it is object-safe.
pub trait CudaGraphExecutor: Send {
    /// Whether graph capture has completed and the executor can replay.
    ///
    /// The engine's run loop checks this every step to decide between the
    /// fast-path (`step_with_graph`) and the regular `step`. Implementations
    /// must be cheap — this is on the hot path.
    fn is_enabled(&self) -> bool;

    /// Replay the captured graph for `batch` and return the model output.
    ///
    /// # Errors
    ///
    /// Returns [`GraphExecutionError::GraphNotFound`] if no graph has been
    /// captured for the batch's shape, and `GraphExecutionFailed` if the
    /// underlying GPU launch failed. Callers are expected to fall back to a
    /// regular forward pass on any error.
    fn execute(&self, batch: &Batch) -> Result<BatchOutput, GraphExecutionError>;

    /// Capture one graph per configured batch size.
    ///
    /// Called once after model load and before serving traffic. Subsequent
    /// calls are no-ops in most implementations.
    ///
    /// # Errors
    ///
    /// Returns [`GraphExecutionError::GraphCaptureFailed`] if capture fails
    /// on any of the configured batch sizes. Capture failure aborts the
    /// call early; later sizes are not attempted.
    fn capture_all_graphs(&mut self) -> Result<(), GraphExecutionError>;
}
