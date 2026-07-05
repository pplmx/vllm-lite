//! Compute kernels for the model layer: CUDA-Graph capture/replay, FlashAttention (CPU + GPU variants), and fused MLP.
//!
//! Each submodule exposes a `forward` / `capture` function the model
//! layer calls into; the surrounding trait surface is in `vllm-traits`.
pub mod cuda_graph;
pub mod flash_attention;
pub mod fused_mlp;

pub use cuda_graph::{
    BatchCudaGraphExecutor, CudaGraph, CudaGraphConfig, CudaGraphError, CudaGraphExecutor,
    GraphExecutionError, ModelGraphConfig,
};
pub use flash_attention::{
    AttentionVariant, FlashAttention, FlashAttentionConfig, FlashAttentionKernel,
};
pub use fused_mlp::{fused_attention_layer, fused_mlp_layer};
