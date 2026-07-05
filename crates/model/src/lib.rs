//! `vllm-model` — architecture implementations and shared model components.
//!
//! This crate owns the inference code that `vllm-core` calls: every
//! supported LLM architecture (Llama, Qwen3, Mistral, Gemma, Mixtral,
//! Qwen3.5 Mamba hybrid), the architecture-registry that picks among
//! them at load time, the shared transformer components (attention,
//! MLP, norm, `RoPE`), the kernels (`FlashAttention`, fused MLP, CUDA
//! graph capture), and the checkpoint loader (safetensors + GGUF).
pub mod arch;
pub mod causal_lm;
pub mod components;
pub mod config;
/// gemma3: gemma3 module.
pub mod gemma3;
/// gemma4: gemma4 module.
pub mod gemma4;
pub mod kernels;
pub mod kv_cache;
pub mod llama;
/// llama4: llama4 module.
pub mod llama4;
pub mod loader;
pub mod mistral;
pub mod mistral_small;
pub mod mixtral;
pub mod paged_tensor;
/// phi4: phi4 module.
pub mod phi4;
pub mod quantize;
/// qwen3: qwen3 module.
pub mod qwen3;
/// `qwen3_5`: qwen3 5 module.
pub mod qwen3_5;
pub mod tokenizer;

pub use arch::{
    ARCHITECTURE_REGISTRY, ArchCapabilities, Architecture, ArchitectureRegistry, register_all_archs,
};
pub use causal_lm::{BlockWrapper, CausalLm};
pub use config::ModelConfig;
pub use kernels::{
    AttentionVariant, CudaGraph, FlashAttention, FlashAttentionConfig, FlashAttentionKernel,
    fused_attention_layer, fused_mlp_layer,
};
pub use loader::{ModelLoader, ModelLoaderBuilder};
pub use quantize::{QuantizationConfig, QuantizationFormat, QuantizedTensor, StorageTensor};
pub use tokenizer::Tokenizer;
