//! FlashAttention namespace: facade re-exporting the `config` (variant + tile-size helpers), `kernel` (impls), and `util` (softmax + math helpers).
//!
//! Selected by the architecture's `model.json` field; the `forward`
//! function in `kernel.rs` dispatches to the right impl.
#![allow(clippy::module_name_repetitions)]
// crates/model/src/kernels/flash_attention/mod.rs
//
// Facade for the flash attention subsystem. Sub-modules:
// - `config` — `AttentionVariant`, `FlashAttentionConfig`, tile-size helpers.
// - `util`   — `AttentionStats`, `softmax_last_dim` helpers.
// - `kernel` — `FlashAttention` trait, `ScaledDotProductAttention`,
//              `FlashAttentionV2`, `FlashAttentionKernel`.

mod config;
mod kernel;
mod util;

pub use config::{AttentionVariant, FlashAttentionConfig, select_tile_size, should_use_tiled};
pub use kernel::{
    FlashAttention, FlashAttentionKernel, FlashAttentionV2, ScaledDotProductAttention,
};
pub use util::{AttentionStats, softmax_last_dim};
