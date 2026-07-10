//! Public QK-norm + projection helpers for [`GqaAttention`].
//!
//! These are the externally-visible building blocks used by callers
//! that need finer-grained control than [`GqaAttention::forward`]
//! (e.g. when wiring QK-norm manually or chaining custom projections):
//!
//! - [`GqaAttention::project_qkv`] — split the input through Q/K/V
//!   linear projections without reshape or norm.
//! - [`GqaAttention::apply_q_norm_impl`] / [`GqaAttention::apply_k_norm_impl`]
//!   — apply QK-norm on already-projected tensors with caller-supplied
//!   shape params.
//! - [`GqaAttention::apply_q_norm_impl_flattened`] /
//!   [`GqaAttention::apply_k_norm_impl_flattened`] — same but on a
//!   pre-flattened `(N, head_dim)` tensor.

use super::GqaAttention;
use candle_core::{Module, Result, Tensor};

impl GqaAttention {
    /// Project the input through Q/K/V linear projections.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn project_qkv(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        Ok((q, k, v))
    }

    /// Run the operation (see signature for params and return type).
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn apply_q_norm_impl(
        &self,
        q: Tensor,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        if let Some(ref q_norm) = self.q_norm {
            let q = q.reshape((batch_size * num_heads * seq_len, head_dim))?;
            let q = q_norm.forward(&q)?;
            let q = q.reshape((batch_size, num_heads, seq_len, head_dim))?;
            Ok(q)
        } else {
            Ok(q)
        }
    }

    /// Run the operation (see signature for params and return type).
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn apply_k_norm_impl(
        &self,
        k: Tensor,
        batch_size: usize,
        num_kv_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        if let Some(ref k_norm) = self.k_norm {
            let k = k.reshape((batch_size * num_kv_heads * seq_len, head_dim))?;
            let k = k_norm.forward(&k)?;
            let k = k.reshape((batch_size, num_kv_heads, seq_len, head_dim))?;
            Ok(k)
        } else {
            Ok(k)
        }
    }

    /// Run the operation (see signature for params and return type).
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn apply_q_norm_impl_flattened(&self, q: Tensor) -> Result<Tensor> {
        if let Some(ref q_norm) = self.q_norm {
            let q = q_norm.forward(&q)?;
            Ok(q)
        } else {
            Ok(q)
        }
    }

    /// Run the operation (see signature for params and return type).
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn apply_k_norm_impl_flattened(&self, k: Tensor) -> Result<Tensor> {
        if let Some(ref k_norm) = self.k_norm {
            let k = k_norm.forward(&k)?;
            Ok(k)
        } else {
            Ok(k)
        }
    }
}
