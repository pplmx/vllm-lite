//! `FlashAttention` facade: dispatch entry-point plus the trait that
//! every kernel impl satisfies.
//!
//! The two concrete implementations live in sibling files:
//! - [`flash_attention_v2`] — `FlashAttentionV2` (tiling + online softmax)
//! - [`scaled_dot_product`] — `ScaledDotProductAttention` (CPU reference,
//!   also used as the dispatch target for the `Tiled`, `Flash`, and
//!   `Standard` variants).
//!
//! `FlashAttentionKernel::new` selects the impl based on
//! [`AttentionVariant`] and routes `forward` calls accordingly.
//!
//! Tests for the facade + impls live in `tests.rs`.

// crates/model/src/kernels/flash_attention/kernel.rs
//
// Facade: `FlashAttention` trait + `FlashAttentionKernel` dispatcher.
// Impls: `flash_attention_v2.rs`, `scaled_dot_product.rs`.

// invariant: tensor-dimension casts (head_dim/block_idx -> f32) are bounded by
// model architecture constants; precision loss / truncation is intentional.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use super::config::{AttentionVariant, FlashAttentionConfig, select_tile_size};
use candle_core::{Result, Tensor};

mod flash_attention_v2;
mod scaled_dot_product;
#[cfg(test)]
mod tests;

pub use flash_attention_v2::FlashAttentionV2;
pub use scaled_dot_product::ScaledDotProductAttention;

/// `FlashAttention`. See the type definition for fields and behavior.
pub trait FlashAttention: Send + Sync + std::fmt::Debug {
    /// Run the layer forward pass over the input.
    /// # Errors
    ///
    /// Returns `Err` if any tensor operation fails (shape mismatch, out-of-memory, dtype incompatibility, or kernel error).
    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor>;
    /// Run attention with an explicit mask tensor.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    fn forward_with_mask(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor>;
    /// Run attention tile-by-tile for memory efficiency.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    fn forward_tiled(&self, q: &Tensor, k: &Tensor, v: &Tensor, tile_size: usize)
    -> Result<Tensor>;
}

/// `FlashAttentionKernel`. See the type definition for fields and behavior.
#[derive(Debug)]
pub struct FlashAttentionKernel {
    attention: Box<dyn FlashAttention>,
    config: FlashAttentionConfig,
}

impl FlashAttentionKernel {
    #[must_use]
    pub fn new(num_heads: usize, head_dim: usize, config: FlashAttentionConfig) -> Self {
        let attention: Box<dyn FlashAttention> = match config.variant {
            AttentionVariant::Tiled => Box::new(
                ScaledDotProductAttention::new(head_dim).with_tile_size(config.flash_block_size),
            ),
            AttentionVariant::FlashV2 => Box::new(
                FlashAttentionV2::new(num_heads, head_dim).with_block_size(config.flash_block_size),
            ),
            AttentionVariant::Flash | AttentionVariant::Standard => {
                Box::new(ScaledDotProductAttention::new(head_dim))
            }
        };

        Self { attention, config }
    }

    /// Run the layer forward pass over the input.
    /// # Errors
    ///
    /// Returns `Err` if any tensor operation fails (shape mismatch, out-of-memory, dtype incompatibility, or kernel error).
    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        if self.config.variant == AttentionVariant::Tiled {
            return self.forward_tiled(q, k, v);
        }
        if self.config.use_sliding_window && self.config.sliding_window_size > 0 {
            self.forward_sliding_window(q, k, v)
        } else {
            self.attention.forward(q, k, v)
        }
    }

    fn forward_tiled(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let seq_len = q.dims()[2];
        let tile_size = select_tile_size(seq_len, &self.config);
        self.attention.forward_tiled(q, k, v, tile_size)
    }

    fn forward_sliding_window(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let window_size = self.config.sliding_window_size;
        let k_len = k.dims()[2];

        if k_len <= window_size {
            return self.attention.forward(q, k, v);
        }

        let k_window = k.narrow(2, k_len.saturating_sub(window_size), window_size)?;
        let v_window = v.narrow(2, k_len.saturating_sub(window_size), window_size)?;

        self.attention.forward(q, &k_window, &v_window)
    }
}
