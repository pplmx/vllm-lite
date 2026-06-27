//! Attention: submodules for grouped-query, multi-head latent, and
//! paged/tiled attention primitives, plus shared utilities.
//!
//! Most call sites should `use` the specific submodule (`gqa`, `mla`,
//! `paged_gqa`, etc.). The utility helpers (`expand_kv`, `causal_mask`,
//! `paged_attention`, `tiled_attention`, `AttentionConfig`) live in
//! the `util` module and are re-exported here for convenience.

/// flash: flash module.
pub mod flash;
/// flash_attention_v3: flash attention v3 module.
pub mod flash_attention_v3;
/// gqa: gqa module.
pub mod gqa;
/// mla: mla module.
pub mod mla;
/// paged_gqa: paged gqa module.
pub mod paged_gqa;
/// rope_gqa: rope gqa module.
pub mod rope_gqa;
/// util: shared attention utilities (expand_kv, causal_mask, paged/tiled attention).
pub mod util;

pub use flash_attention_v3::{
    FlashAttentionV3, FlashAttentionV3Config, GqaFlashAttention, MqaFlashAttention,
};
pub use gqa::GqaAttention;
pub use mla::MlaAttention;
pub use paged_gqa::{
    QkRotaryEmb, compute_gqa_attention, prefill_causal_mask, project_attention_output,
    read_decode_kv, write_prefill_kv,
};
pub use rope_gqa::RopeGqaAttention;
pub use util::{
    AttentionConfig, causal_mask, causal_mask_tile, expand_kv, paged_attention, tiled_attention,
};
