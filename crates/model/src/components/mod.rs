//! Shared transformer building blocks: attention (GQA + MLA + paged), MLP (SwiGLU), norm (RMSNorm + LayerNorm), positional (RoPE + MRoPE), SSM (Mamba + gated-delta), and a vision encoder placeholder.
//!
//! Each architecture in `model::llama`, `model::qwen3`, etc. composes
//! these primitives rather than re-implementing them. The re-exports
//! below surface the most-used items at `vllm_model::*` for ergonomics.
pub mod attention;
pub mod block;
pub mod decoder_block;
pub mod gated_delta;
/// kv_cache_fp8: kv cache fp8 module.
pub mod kv_cache_fp8;
pub mod mlp;
pub mod norm;
pub mod positional;

pub use positional::{MRoPE, RoPE, apply_rope, precompute_rope_cache};
pub mod ssm;
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
