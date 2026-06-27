//! model: crate root.

/// arch: arch module.
pub mod arch;
/// causal_lm: causal lm module.
pub mod causal_lm;
/// components: components module.
pub mod components;
/// config: config module.
pub mod config;
/// gemma3: gemma3 module.
pub mod gemma3;
/// gemma4: gemma4 module.
pub mod gemma4;
/// kernels: kernels module.
pub mod kernels;
/// kv_cache: kv cache module.
pub mod kv_cache;
/// llama: llama module.
pub mod llama;
/// llama4: llama4 module.
pub mod llama4;
/// loader: loader module.
pub mod loader;
/// mistral: mistral module.
pub mod mistral;
/// mistral_small: mistral small module.
pub mod mistral_small;
/// mixtral: mixtral module.
pub mod mixtral;
/// paged_tensor: paged tensor module.
pub mod paged_tensor;
/// phi4: phi4 module.
pub mod phi4;
/// quantize: quantize module.
pub mod quantize;
/// qwen3: qwen3 module.
pub mod qwen3;
/// qwen3_5: qwen3 5 module.
pub mod qwen3_5;
/// tokenizer: tokenizer module.
pub mod tokenizer;

// Deprecated alias: `qwen3_config` module moved to `qwen3::config` in v21.1.
// Retained for one minor release as a re-export shim for backward compatibility.
#[deprecated(
    since = "0.21.0",
    note = "use `vllm_model::qwen3::config` instead; this module path is preserved as a re-export shim"
)]
pub mod qwen3_config {
    pub use crate::qwen3::config::*;
}

pub use arch::{ARCHITECTURE_REGISTRY, ArchCapabilities, ArchitectureRegistry, register_all_archs};
pub use causal_lm::{BlockWrapper, CausalLm};
pub use kernels::{
    AttentionVariant, CudaGraph, FlashAttention, FlashAttentionConfig, FlashAttentionKernel,
    fused_attention_layer, fused_mlp_layer,
};
pub use quantize::{QuantizationConfig, QuantizationFormat, QuantizedTensor, StorageTensor};
