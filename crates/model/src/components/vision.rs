#![allow(clippy::module_name_repetitions)]
//! Vision encoder components for Sam/Vision models.
//!
//! `VisionEncoder` is a placeholder for future VLM integration. It preserves tensor
//! shapes today and does not run a real `ViT` stack yet.

use candle_core::{Module, Result as CandleResult, Tensor};
use candle_nn::{Linear, VarBuilder};

/// Configuration for Vision. Constructed via the `builder()` associated function or by deserializing from JSON / TOML. Pass-by-value to construction APIs.
#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub embed_dim: usize,
    pub depth: usize,
}

impl VisionConfig {
    #[must_use]
    pub const fn new(image_size: usize, patch_size: usize) -> Self {
        Self {
            image_size,
            patch_size,
            embed_dim: 768,
            depth: 12,
        }
    }

    #[must_use]
    pub const fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
    }
}

#[derive(Debug)]
/// Patch-embedding projection for a vision transformer.
///
/// Splits an input image into non-overlapping `patch_size × patch_size`
/// patches (RGB channels flattened per patch), then linearly projects
/// each patch into the model's `embed_dim`. The result is a sequence
/// of `(num_patches, embed_dim)` token embeddings ready for the
/// transformer stack.
///
/// Input shape to [`forward`](PatchEmbed::forward) is expected to be
/// `[batch, num_patches, patch_size * patch_size * 3]` (i.e. the
/// caller has already flattened the patches; a real ViT pipeline
/// would have a separate patch-extraction step before this layer).
pub struct PatchEmbed {
    proj: Linear,
}

impl PatchEmbed {
    /// Construct a new `PatchEmbed` from a [`VisionConfig`] and a
    /// `VarBuilder` rooted at the `proj` prefix.
    ///
    /// The projection matrix has shape
    /// `(patch_size * patch_size * 3) → embed_dim`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the underlying `Linear` weight tensor cannot
    /// be allocated or loaded from the `VarBuilder`.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(config: &VisionConfig, vb: VarBuilder<'_>) -> CandleResult<Self> {
        let proj = candle_nn::linear(
            config.patch_size * config.patch_size * 3,
            config.embed_dim,
            vb.pp("proj"),
        )?;
        Ok(Self { proj })
    }

    /// Run the patch-embedding projection on a pre-flattened patch
    /// tensor of shape `[batch, num_patches, patch_size²·3]`.
    ///
    /// # Errors
    ///
    /// Returns `Err` on shape mismatch (last dim ≠
    /// `patch_size²·3`), out-of-memory, dtype incompatibility, or
    /// any kernel error from the underlying matmul.
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        self.proj.forward(x)
    }
}
#[derive(Debug)]

/// Placeholder vision tower.
///
/// Owns a [`VisionConfig`] and currently passes its input through
/// unchanged on [`forward`](VisionEncoder::forward). This exists so
/// that vision-language model loaders can resolve a `vision_tower`
/// module today and dispatch through a stable API; the real ViT
/// (patch-embed + transformer + pooling) will replace the
/// pass-through body without changing the public surface.
pub struct VisionEncoder {
    config: VisionConfig,
}

impl VisionEncoder {
    /// Construct a new `VisionEncoder` from a [`VisionConfig`] and a
    /// `VarBuilder` (currently unused — placeholder).
    ///
    /// # Errors
    ///
    /// Returns `Err` if any required tensor allocation or weight
    /// loading fails. Today this always returns `Ok`.
    pub fn new(config: &VisionConfig, _vb: VarBuilder<'_>) -> CandleResult<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    /// Borrow the underlying [`VisionConfig`].
    #[must_use]
    pub const fn config(&self) -> &VisionConfig {
        &self.config
    }

    /// Pass-through placeholder forward.
    ///
    /// Returns the input tensor unchanged. The shape contract is
    /// `output.dims() == input.dims()` — future implementations will
    /// preserve this by mapping input tokens to output embeddings.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any tensor operation fails. Today this
    /// always returns `Ok` since the implementation is a `clone`.
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        Ok(x.clone())
    }
}

// Unit tests are extracted to `tests.rs` (sibling) to keep this
// module under the 800-line soft cap. They cover VisionConfig
// patch-count math, PatchEmbed construction under zero-init, and
// the VisionEncoder pass-through shape contract.
#[cfg(test)]
mod tests;
