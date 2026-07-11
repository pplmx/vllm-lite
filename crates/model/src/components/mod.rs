//! Shared transformer building blocks.
//!
//! Composable attention (GQA + MLA + paged), MLP (`SwiGLU`), norm
//! (`RMSNorm` + `LayerNorm`), positional (`RoPE` + `MRoPE`), SSM
//! (Mamba + gated-delta), and a vision encoder placeholder. Each
//! architecture in `model::llama`, `model::qwen3`, etc. composes these
//! primitives rather than re-implementing them. The re-exports below
//! surface the most-used items at `vllm_model::*` for ergonomics.
/// GQA, MLA, and paged attention implementations.
pub mod attention;
/// Generic transformer block composing attention + MLP + norm.
pub mod block;
/// Paged-KV-aware decoder blocks for causal LM stacks.
pub mod decoder_block;
/// Gated-delta (linear attention) layers for Qwen3.5 hybrid models.
pub mod gated_delta;
/// FP8 KV-cache quantization helpers.
pub mod kv_cache_fp8;
/// SwiGLU feed-forward network primitives.
pub mod mlp;
/// RMSNorm and LayerNorm building blocks.
pub mod norm;
/// RoPE and multi-axis MRoPE positional encodings.
pub mod positional;

pub use positional::{MRoPE, RoPE, apply_rope, precompute_rope_cache};
/// Mamba SSM and harmonic hybrid SSM layers.
pub mod ssm;
/// Vision encoder placeholder for multimodal models.
pub mod vision;

pub use super::kernels::{fused_attention_layer, fused_mlp_layer};
pub use attention::{
    AttentionConfig, GqaAttention, MlaAttention, causal_mask, causal_mask_tile, expand_kv,
    paged_attention, tiled_attention,
};
pub use block::TransformerBlock;
pub use decoder_block::{PagedDecoderBlock, RopeGqaDecoderBlock};
pub use gated_delta::{GatedDeltaConfig, GatedDeltaNet, GatedDeltaState};
pub use kv_cache_fp8::{Fp8Quantizer, KvCacheDtype};
pub use mlp::{SwiGLU, swiglu_forward};
pub use norm::{LnLayerNorm, RmsNorm, layer_norm, rms_norm};
pub use ssm::{MambaBlock, SSMConfig, SSMError, SSMHarmonicSSMLayer, SSMLayer, softplus};
pub use vision::{PatchEmbed, VisionConfig, VisionEncoder};
