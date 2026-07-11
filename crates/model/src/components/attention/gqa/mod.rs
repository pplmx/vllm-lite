//! Grouped-Query Attention implementation: `num_heads` query heads share `num_kv_heads` key/value heads with a configurable ratio.
//!
//! The default impl is the reference GQA used by Llama/Mistral; the
//! `RopeGqaAttention` wrapper (in `rope_gqa.rs`) layers `RoPE` + QK-norm
//! on top. Reads/writes KV through the paged tensor store so memory
//! stays contiguous regardless of sequence length.
//!
//! Module layout:
//!
//! - [`self`] (`mod.rs`) — `GqaAttention` struct + construction + private QK-norm helpers + getters
//! - [`forward`] — `forward` + the production dispatchers (`paged_attention_fn`, `tiled_attention_fn`, `flash_attention_fn`, `run_attention_fn`)
//! - [`norm`] — public QK-norm API (`project_qkv`, `apply_q_norm_impl_flattened`, `apply_k_norm_impl_flattened`) used by external callers that need finer-grained control
//! - [`tests`] — unit tests (sibling file)

#![allow(clippy::too_many_arguments, clippy::module_name_repetitions)]
// invariant: tensor-dimension casts (head_dim/seq_len -> f32) are bounded by
// model architecture constants; precision loss is intentional.
#![allow(clippy::cast_precision_loss)]

mod forward;
mod norm;

use candle_core::{Module, Result, Tensor};
use candle_nn::{LayerNorm, Linear};

use super::{AttentionConfig, expand_kv};

#[derive(Debug)]
/// `GqaAttention`. See the type definition for fields and behavior.
pub struct GqaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    config: AttentionConfig,
    q_norm: Option<LayerNorm>,
    k_norm: Option<LayerNorm>,
    /// YaRN attention-temperature scaling factor. When `Some(f)`,
    /// attention scores in the standard `forward()` are additionally
    /// divided by `f` before softmax. `None` or `Some(1.0)` = no scaling.
    /// Currently only honoured by the standard forward path; paged/tiled/
    /// flash attention paths silently ignore this value (documented
    /// limitation; follow-up phase will thread it through).
    pub attn_factor: Option<f32>,
}

impl GqaAttention {
    /// Construct a new instance from the given configuration.
    /// # Errors
    ///
    /// Returns `Err` if any required tensor allocation or weight loading fails.
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        vb: Option<candle_nn::VarBuilder<'_>>,
        config: AttentionConfig,
        has_qk_norm: bool,
    ) -> Result<Self> {
        let vb = vb.unwrap_or_else(|| {
            candle_nn::VarBuilder::zeros(candle_core::DType::F32, &candle_core::Device::Cpu)
        });

        let q_proj = candle_nn::linear(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        let q_norm = if has_qk_norm {
            Some(candle_nn::layer_norm(head_dim, 1e-6, vb.pp("q_norm"))?)
        } else {
            None
        };
        let k_norm = if has_qk_norm {
            Some(candle_nn::layer_norm(head_dim, 1e-6, vb.pp("k_norm"))?)
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            config,
            q_norm,
            k_norm,
            attn_factor: None,
        })
    }

    /// Construct a new instance from already-loaded weight tensors.
    /// # Errors
    ///
    /// Returns `Err` if any required tensor allocation or weight loading fails.
    pub fn new_with_weights(
        _hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        q_weight: Tensor,
        k_weight: Tensor,
        v_weight: Tensor,
        o_weight: Tensor,
        config: AttentionConfig,
        has_qk_norm: bool,
        q_norm_weight: Option<Tensor>,
        k_norm_weight: Option<Tensor>,
    ) -> Result<Self> {
        let q_proj = Linear::new(q_weight, None);
        let k_proj = Linear::new(k_weight, None);
        let v_proj = Linear::new(v_weight, None);
        let o_proj = Linear::new(o_weight, None);

        let q_norm = if has_qk_norm {
            let q_norm_weight =
                q_norm_weight.ok_or_else(|| candle_core::Error::msg("Missing q_norm weight"))?;
            let q_norm_bias =
                Tensor::zeros(head_dim, q_norm_weight.dtype(), q_norm_weight.device())?;
            Some(LayerNorm::new(q_norm_weight, q_norm_bias, 1e-6))
        } else {
            None
        };
        let k_norm = if has_qk_norm {
            let k_norm_weight =
                k_norm_weight.ok_or_else(|| candle_core::Error::msg("Missing k_norm weight"))?;
            let k_norm_bias =
                Tensor::zeros(head_dim, k_norm_weight.dtype(), k_norm_weight.device())?;
            Some(LayerNorm::new(k_norm_weight, k_norm_bias, 1e-6))
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            config,
            q_norm,
            k_norm,
            attn_factor: None,
        })
    }

    /// Expand KV heads along the head axis to match the query head count.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn expand_kv(
        &self,
        kv: &Tensor,
        num_q_heads: usize,
        num_kv_heads: usize,
    ) -> Result<Tensor> {
        expand_kv(kv, num_q_heads, num_kv_heads)
    }

    /// Private QK-norm helper used by `forward` (transposes the head axis
    /// in/out around the layer-norm).
    fn apply_q_norm(&self, q: Tensor, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        if let Some(ref q_norm) = self.q_norm {
            let q = q.transpose(1, 2)?;
            let reshape_size = batch_size * self.num_heads * seq_len;
            let q = q.reshape((reshape_size, self.head_dim))?;
            let q = q_norm.forward(&q)?;
            let q = q.reshape((batch_size, self.num_heads, seq_len, self.head_dim))?;
            let q = q.transpose(1, 2)?;
            Ok(q)
        } else {
            Ok(q)
        }
    }

    /// Private QK-norm helper used by `forward` (K variant).
    fn apply_k_norm(&self, k: Tensor, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        if let Some(ref k_norm) = self.k_norm {
            let k = k.transpose(1, 2)?;
            let reshape_size = batch_size * self.num_kv_heads * seq_len;
            let k = k.reshape((reshape_size, self.head_dim))?;
            let k = k_norm.forward(&k)?;
            let k = k.reshape((batch_size, self.num_kv_heads, seq_len, self.head_dim))?;
            let k = k.transpose(1, 2)?;
            Ok(k)
        } else {
            Ok(k)
        }
    }

    #[must_use]
    pub const fn num_heads(&self) -> usize {
        self.num_heads
    }

    #[must_use]
    pub const fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    #[must_use]
    pub(crate) const fn o_proj_linear(&self) -> &Linear {
        &self.o_proj
    }

    #[must_use]
    pub const fn has_q_norm(&self) -> bool {
        self.q_norm.is_some()
    }

    #[must_use]
    pub const fn has_k_norm(&self) -> bool {
        self.k_norm.is_some()
    }

    #[must_use]
    pub const fn config(&self) -> &AttentionConfig {
        &self.config
    }

    /// Device on which the projection weights live.
    ///
    /// Used by wrappers (e.g. `RopeGqaAttention`) that need to construct
    /// a `RoPE` matching the underlying attention's device.
    #[must_use]
    pub fn device(&self) -> &candle_core::Device {
        self.q_proj.weight().device()
    }
}

// Unit tests live in a separate file to keep this implementation file under
// the 800-line soft cap. The tests are included via `mod tests;` so they
// share the same scope (and `use super::*;` access) as if they were inline.
#[cfg(test)]
mod tests;
