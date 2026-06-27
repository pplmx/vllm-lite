//! mod: module.

/// cuda_graph: cuda graph module.
pub mod cuda_graph;
/// flash_attention: flash attention module.
pub mod flash_attention;
/// fused_mlp: fused mlp module.
pub mod fused_mlp;

pub use cuda_graph::{
    BatchCudaGraphExecutor, CudaGraph, CudaGraphConfig, CudaGraphError, CudaGraphExecutor,
    GraphExecutionError, ModelGraphConfig,
};
pub use flash_attention::{
    AttentionVariant, FlashAttention, FlashAttentionConfig, FlashAttentionKernel,
};
pub use fused_mlp::{fused_attention_layer, fused_mlp_layer};
