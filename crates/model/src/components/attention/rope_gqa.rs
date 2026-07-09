//! GQA + `RoPE` + QK-norm fused attention layer used by Qwen3.
//!
//! Layers the rope embedding + Q/K normalisation on top of the
//! reference GQA so the forward call is one fused op. Reads/writes
//! through the paged tensor store; falls back to the baseline GQA on
//! unsupported head-dim / seq-len combinations.
#![allow(clippy::too_many_arguments, clippy::module_name_repetitions)]
// invariant: position-id casts (usize -> i64) are bounded by sequence length,
// well within i64 range.
#![allow(clippy::cast_possible_wrap)]

pub use crate::components::AttentionConfig;
use crate::components::attention::GqaAttention as SharedGqaAttention;
use crate::components::attention::paged_gqa::{read_decode_kv, write_prefill_kv};
use crate::components::positional::apply_rope;
use crate::paged_tensor::PagedKvCache;
use candle_core::{Result, Tensor};

#[derive(Debug)]
/// `RopeGqaAttention`. See the type definition for fields and behavior.
pub struct RopeGqaAttention {
    inner: SharedGqaAttention,
    theta: f32,
}

impl RopeGqaAttention {
    /// Construct a `RopeGqaAttention` from a `VarBuilder`. The inner
    /// `GqaAttention` is built with the same projection layout
    /// (`q`/`k`/`v`/`o` + optional QK-norm); this wrapper additionally
    /// captures the `RoPE` base `theta` so [`Self::forward_prefill`] /
    /// [`Self::forward_decode`] can apply rotary embeddings before the
    /// attention matmul.
    /// # Errors
    ///
    /// Returns `Err` if the inner `GqaAttention::new` weight load fails.
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        theta: f32,
        vb: Option<candle_nn::VarBuilder<'_>>,
        config: AttentionConfig,
        has_qk_norm: bool,
    ) -> Result<Self> {
        let inner = SharedGqaAttention::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            vb,
            config,
            has_qk_norm,
        )?;
        Ok(Self { inner, theta })
    }

    /// Construct a `RopeGqaAttention` from already-loaded projection tensors.
    /// Use this when sourcing weights from a non-HF checkpoint or when
    /// wiring weights from a custom loader.
    /// # Errors
    ///
    /// Returns `Err` if `GqaAttention::new_with_weights` fails (e.g.
    /// missing QK-norm weights when `has_qk_norm` is true).
    pub fn new_with_weights(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        theta: f32,
        q_weight: Tensor,
        k_weight: Tensor,
        v_weight: Tensor,
        o_weight: Tensor,
        config: AttentionConfig,
        has_qk_norm: bool,
        q_norm_weight: Option<Tensor>,
        k_norm_weight: Option<Tensor>,
    ) -> Result<Self> {
        let inner = SharedGqaAttention::new_with_weights(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            q_weight,
            k_weight,
            v_weight,
            o_weight,
            config,
            has_qk_norm,
            q_norm_weight,
            k_norm_weight,
        )?;
        Ok(Self { inner, theta })
    }

    /// Standard non-causal forward (delegates to `GqaAttention::forward`).
    /// # Errors
    ///
    /// Returns `Err` if any tensor operation fails (shape mismatch, out-of-memory, dtype incompatibility, or kernel error).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }

    fn apply_qk_norm(
        &self,
        q: Tensor,
        k: Tensor,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let num_heads = self.inner.num_heads();
        let num_kv_heads = self.inner.num_kv_heads();
        let head_dim = self.inner.head_dim();

        let q = if self.inner.has_q_norm() {
            let q = q.transpose(1, 2)?;
            let reshape_size = batch_size * num_heads * seq_len;
            let q = q.reshape((reshape_size, head_dim))?;
            let q = self.inner.apply_q_norm_impl_flattened(q)?;
            let q = q.reshape((batch_size, num_heads, seq_len, head_dim))?;
            q.transpose(1, 2)?
        } else {
            q
        };

        let k = if self.inner.has_k_norm() {
            let k = k.transpose(1, 2)?;
            let reshape_size = batch_size * num_kv_heads * seq_len;
            let k = k.reshape((reshape_size, head_dim))?;
            let k = self.inner.apply_k_norm_impl_flattened(k)?;
            let k = k.reshape((batch_size, num_kv_heads, seq_len, head_dim))?;
            k.transpose(1, 2)?
        } else {
            k
        };

        Ok((q, k))
    }

    /// Prefill path: project Q/K/V, apply optional QK-norm, apply `RoPE`,
    /// write the new KV into the paged cache, then run causal attention.
    ///
    /// Used when a brand-new prompt arrives and the full KV prefix has to
    /// be cached at once. The KV cache write happens before the attention
    /// matmul so subsequent decode steps can reuse the cache without
    /// re-encoding.
    /// # Errors
    ///
    /// Returns `Err` if any projection, `RoPE` rotation, KV-cache write,
    /// or attention matmul fails.
    pub fn forward_prefill(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        positions: &[usize],
    ) -> Result<Tensor> {
        let batch_size = x.dims()[0];
        let seq_len = x.dims()[1];
        let num_heads = self.inner.num_heads();
        let num_kv_heads = self.inner.num_kv_heads();
        let head_dim = self.inner.head_dim();

        let (q, k, v) = self.inner.project_qkv(x)?;

        let q = q.reshape((batch_size, seq_len, num_heads, head_dim))?;
        let k = k.reshape((batch_size, seq_len, num_kv_heads, head_dim))?;
        let v = v.reshape((batch_size, seq_len, num_kv_heads, head_dim))?;

        let (q, k) = self.apply_qk_norm(q, k, batch_size, seq_len)?;

        let position_ids: Vec<i64> = positions.iter().map(|&p| p as i64).collect();
        let q = apply_rope(&q, &position_ids, self.theta)?;
        let k = apply_rope(&k, &position_ids, self.theta)?;

        let k_expanded = self.inner.expand_kv(&k, num_heads, num_kv_heads)?;
        let v_expanded = self.inner.expand_kv(&v, num_heads, num_kv_heads)?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k_expanded = k_expanded.transpose(1, 2)?.contiguous()?;
        let v_expanded = v_expanded.transpose(1, 2)?.contiguous()?;

        write_prefill_kv(
            kv_cache,
            layer_idx,
            block_ids,
            seq_len,
            &k_expanded,
            &v_expanded,
        )?;

        self.inner.run_attention_fn(&q, &k_expanded, &v_expanded)
    }

    /// Decode path: project Q/K/V for one new token, apply optional QK-norm,
    /// apply `RoPE`, read the existing KV prefix from the paged cache, append
    /// the new K/V, then run causal attention over the full prefix.
    ///
    /// `num_computed_tokens` is the number of tokens already in the cache
    /// for this sequence at the start of this step; it tells the cache
    /// reader where to start writing the new entries.
    /// # Errors
    ///
    /// Returns `Err` if any projection, `RoPE` rotation, KV-cache read,
    /// or attention matmul fails.
    pub fn forward_decode(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        num_computed_tokens: usize,
        positions: &[usize],
    ) -> Result<Tensor> {
        let batch_size = x.dims()[0];
        let num_heads = self.inner.num_heads();
        let num_kv_heads = self.inner.num_kv_heads();
        let head_dim = self.inner.head_dim();

        let (q, k, v) = self.inner.project_qkv(x)?;

        let q = q.reshape((batch_size, 1, num_heads, head_dim))?;
        let k = k.reshape((batch_size, 1, num_kv_heads, head_dim))?;
        let v = v.reshape((batch_size, 1, num_kv_heads, head_dim))?;

        let (q, k) = self.apply_qk_norm(q, k, batch_size, 1)?;

        let position_ids: Vec<i64> = positions.iter().map(|&p| p as i64).collect();
        let q = apply_rope(&q, &position_ids, self.theta)?;
        let k = apply_rope(&k, &position_ids, self.theta)?;

        let k_expanded = self.inner.expand_kv(&k, num_heads, num_kv_heads)?;
        let v_expanded = self.inner.expand_kv(&v, num_heads, num_kv_heads)?;

        let q = q.transpose(1, 2)?;

        let k_for_cache = k_expanded.transpose(1, 2)?.squeeze(0)?.contiguous()?;
        let v_for_cache = v_expanded.transpose(1, 2)?.squeeze(0)?.contiguous()?;

        let (full_k, full_v) = read_decode_kv(
            kv_cache,
            layer_idx,
            block_ids,
            num_computed_tokens,
            &k_for_cache,
            &v_for_cache,
        )?;

        self.inner.run_attention_fn(&q, &full_k, &full_v)
    }
}

// Unit tests are extracted to `tests.rs` (sibling) to keep this
// implementation file under the 800-line soft cap. The sibling covers:
// forward / forward_decode / forward_prefill shape contracts, the
// qk_norm toggle, and the fused-vs-paged numerical equivalence
// (max abs diff < 1e-4).
#[cfg(test)]
mod tests;
