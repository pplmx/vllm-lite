#![allow(clippy::module_name_repetitions)]
//! Attention: submodules for grouped-query, multi-head latent, and
//! paged/tiled attention primitives, plus shared utilities.
//!
//! Most call sites should `use` the specific submodule (`gqa`, `mla`,
//! `paged_gqa`, etc.). The utility helpers (`expand_kv`, `causal_mask`,
//! `paged_attention`, `tiled_attention`, `AttentionConfig`) live in
//! the `util` module and are re-exported here for convenience.

/// Flash attention v1 reference implementation.
pub mod flash;
/// flash_attention_v3: flash attention v3 module.
pub mod flash_attention_v3;
/// Grouped-Query Attention (GQA): shared KV heads across query head groups.
pub mod gqa;
/// Multi-head Latent Attention (MLA) for DeepSeek-style compression.
pub mod mla;
/// Paged GQA: KV-cache stored in physical blocks with indirection.
pub mod paged_gqa;
/// RoPE-wrapped GQA attention for Qwen3/Qwen3.5 (adds QK-norm + RoPE).
pub mod rope_gqa;
/// Shared attention utilities (expand_kv, causal_mask, paged/tiled attention).
pub mod util;

pub use flash_attention_v3::{
    FlashAttentionV3, FlashAttentionV3Config, GqaFlashAttention, MqaFlashAttention,
};
pub use gqa::GqaAttention;
pub use mla::MlaAttention;
pub use paged_gqa::{
    QkRotaryEmb, compute_gqa_attention, prefill_causal_mask, prefill_continue_causal_mask,
    project_attention_output, read_decode_kv, write_prefill_kv,
};
pub use rope_gqa::RopeGqaAttention;
pub use util::{
    AttentionConfig, causal_mask, causal_mask_tile, expand_kv, paged_attention, tiled_attention,
};
