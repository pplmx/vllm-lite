//! `vllm-model` — architecture implementations and shared model components.
//!
//! This crate owns the inference code that `vllm-core` calls: every
//! supported LLM architecture (Llama, Qwen3, Mistral, Gemma, Mixtral,
//! Qwen3.5 Mamba hybrid), the architecture-registry that picks among
//! them at load time, the shared transformer components (attention,
//! MLP, norm, `RoPE`), the kernels (`FlashAttention`, fused MLP, CUDA
//! graph capture), and the checkpoint loader (safetensors + GGUF).
/// Dynamic architecture registry and per-model `Architecture` trait impls.
pub mod arch;
/// Causal-LM trait surface (`CausalLm`) and decoder block wrappers.
pub mod causal_lm;
/// Shared transformer primitives (attention, MLP, norm, RoPE, SSM).
pub mod components;
/// Model configuration types and deserialization helpers.
pub mod config;
/// Gemma 4 hybrid attention architecture.
pub mod gemma4;
/// GPU kernels (`FlashAttention`, fused MLP, CUDA graph capture).
pub mod kernels;
/// Logical KV-cache block types used by model forward paths.
pub mod kv_cache;
/// Llama-family architecture implementation.
pub mod llama;
/// Checkpoint loader (safetensors, GGUF) and `ModelLoader` builder.
pub mod loader;
/// Mistral architecture with sliding-window attention.
pub mod mistral;
/// Mixtral sparse MoE architecture.
pub mod mixtral;
/// Physical paged KV tensor store and quantization helpers.
pub mod paged_tensor;
/// Weight quantization formats and `StorageTensor` abstraction.
pub mod quantize;
/// Qwen2/3 architecture (GQA, MLA, QK-Norm).
pub mod qwen3;
/// Qwen3.5 Mamba / hybrid SSM architecture.
pub mod qwen3_5;
/// Tokenizer wrapper (HuggingFace `tokenizers` + tiktoken backends).
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
