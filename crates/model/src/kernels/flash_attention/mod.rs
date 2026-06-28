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

pub use config::{select_tile_size, should_use_tiled, AttentionVariant, FlashAttentionConfig};
pub use kernel::{FlashAttention, FlashAttentionKernel, FlashAttentionV2, ScaledDotProductAttention};
pub use util::{softmax_last_dim, AttentionStats};
