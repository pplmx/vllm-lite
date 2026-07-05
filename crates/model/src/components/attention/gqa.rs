//! Grouped-Query Attention implementation: `num_heads` query heads share `num_kv_heads` key/value heads with a configurable ratio.
//!
//! The default impl is the reference GQA used by Llama/Mistral; the
//! `RopeGqaAttention` wrapper (in `rope_gqa.rs`) layers `RoPE` + QK-norm
//! on top. Reads/writes KV through the paged tensor store so memory
//! stays contiguous regardless of sequence length.
#![allow(clippy::too_many_arguments, clippy::module_name_repetitions)]
// invariant: tensor-dimension casts (head_dim/seq_len -> f32) are bounded by
// model architecture constants; precision loss is intentional.
#![allow(clippy::cast_precision_loss)]

use candle_core::{Module, Result, Tensor};
use candle_nn::{LayerNorm, Linear};
use tracing::trace;

use super::{AttentionConfig, GqaFlashAttention, expand_kv, paged_attention, tiled_attention};

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
        })
    }

    /// Run the layer forward pass over the input.
    /// # Caution: No causal masking
    ///
    /// This is a low-level primitive. It does **NOT** apply causal masking.
    /// Use [`crate::components::attention::RopeGqaAttention::forward_prefill`] or
    /// [`crate::components::attention::RopeGqaAttention::forward_decode`] (which
    /// dispatch to `paged_attention_fn`, `tiled_attention_fn`, or
    /// `flash_attention_fn`, all of which apply causal masking) for
    /// production inference.
    ///
    /// The fused path (`config.use_fused = true`) hardcodes `causal = false`
    /// because this method is not used in production — see the `// invariant:`
    /// comment at the branch site.
    /// # Errors
    ///
    /// Returns `Err` if any tensor operation fails (shape mismatch, out-of-memory, dtype incompatibility, or kernel error).
    #[allow(clippy::many_single_char_names)]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.dims()[0];
        let seq_len = x.dims()[1];

        trace!(
            batch_size,
            seq_len,
            head_dim = self.head_dim,
            "GqaAttention forward started"
        );

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;

        let q = self.apply_q_norm(q, batch_size, seq_len)?;
        let k = self.apply_k_norm(k, batch_size, seq_len)?;

        // H-11 #3 (attempted, reverted): removing `.contiguous()?` here broke
        // `Tensor::matmul(&q, &k_t)` downstream — candle's CPU matmul kernel
        // rejects non-contiguous LHS (`MatMulUnexpectedStriding` error). Keep
        // the contiguous call; the cost is one copy of Q at (B, H, S, D).
        let q = q.transpose(1, 2)?.contiguous()?;

        if self.config.use_fused {
            // H-11 #3 (attempted, reverted): removing `.contiguous()?` here broke
            // `Tensor::matmul(&attn_weights, &v_expanded)` inside `flash.forward`.
            // Even though flash internally calls `.transpose(2, 3)?.contiguous()?`
            // for the k path, the v path (`Tensor::matmul(attn, v_expanded)`) has
            // no such copy — v_expanded is strided from k_heads.transpose(1, 2).
            // Same `MatMulUnexpectedStriding` failure mode as q/v on the standard path.
            let k_heads = k.transpose(1, 2)?.contiguous()?;
            let v_heads = v.transpose(1, 2)?.contiguous()?;
            // invariant: causal is hardcoded to false here because
            // GqaAttention::forward is a low-level primitive; production code
            // paths route through RopeGqaAttention::forward_prefill/forward_decode
            // which use flash_attention_fn(..., causal=true) instead.
            let flash =
                GqaFlashAttention::new(self.num_heads, self.num_kv_heads, self.head_dim, false);
            let attn_output = flash.forward(&q, &k_heads, &v_heads)?;
            let attn_output = attn_output.transpose(1, 2)?;
            let attn_output =
                attn_output.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;
            let o = self.o_proj.forward(&attn_output)?;
            trace!(output_shape = ?o.dims(), "GqaAttention fused forward completed");
            return Ok(o);
        }

        // H-11 #1 (attempted, reverted): tried to skip `expand_kv` materialization
        // by reshaping q from `(B, H_q, S_q, D)` to `(B, H_kv, repeat, S_q, D)`
        // and using `Tensor::matmul` against `k_t = (B, H_kv, D, S_k)`.
        //
        // FAILED because candle's `Tensor::matmul` (cpu_backend/mod.rs:1329-1351)
        // requires the batch-dim product to match between LHS and RHS — the
        // `(B, H_kv, repeat)` on q_r vs `(B, H_kv, 1)` on k_t (after unsqueeze)
        // gives products `B*H_kv*repeat` vs `B*H_kv*1` (e.g. 16 vs 8 for the
        // standard 8/4 test case), and matmul rejects with
        // "shape mismatch in matmul". `Tensor::broadcast_matmul` would handle
        // it but forces `.contiguous()?` on the broadcast side, which would
        // materialize `(B, H_kv, repeat, D, S_k)` — i.e. `repeat` × the original
        // k_t size, defeating the optimization entirely.
        //
        // The expected savings (`~12×` less K/V memory traffic on qwen3-7B
        // with repeat=7) do not justify the refactor risk without a custom
        // fused matmul kernel that supports implicit GQA broadcasting.
        // Deferred to follow-up work (H-12+ or a custom kernel PR).
        let k = self.expand_kv(&k, self.num_heads, self.num_kv_heads)?;
        let v = self.expand_kv(&v, self.num_heads, self.num_kv_heads)?;

        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let k_t = k.transpose(2, 3)?.contiguous()?;
        let qk = Tensor::matmul(&q, &k_t)?;

        // H-11 #2: replaced `qk.mul(broadcast(scalar_tensor))` with `qk.affine(scale, 0.0)`.
        // The scalar tensor was re-allocated and broadcast to O(B*H*S*S) every forward;
        // `affine` fuses the scaling into the existing kernel without materializing a broadcast tensor.
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let qk = qk.affine(f64::from(scale), 0.0)?;
        // H-11 #3: `candle_nn::ops::softmax` already returns a contiguous tensor
        // (verified in candle-nn 0.10.2 src/ops.rs:22-29 — final op is
        // `broadcast_div`, which produces a fresh contiguous tensor). The
        // explicit `.contiguous()?` is a redundant is_contiguous check + clone.
        let attn_weights = candle_nn::ops::softmax(&qk, 3)?;

        // H-11 #3 (attempted, reverted): removing `.contiguous()?` on v broke
        // `Tensor::matmul(&attn_weights, &v)` — candle's CPU matmul kernel
        // rejects non-contiguous batch dimensions in the RHS
        // (`MatMulUnexpectedStriding`). Same constraint as q.contiguous() above.
        let attn_output = Tensor::matmul(&attn_weights, &v.contiguous()?)?;

        let attn_output = attn_output.transpose(1, 2)?;

        let attn_output =
            attn_output.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        let o = self.o_proj.forward(&attn_output)?;

        trace!(output_shape = ?o.dims(), "GqaAttention forward completed");

        Ok(o)
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

    /// Run the operation (see signature for params and return type).
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn paged_attention_fn(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let attn_output = paged_attention(q, k, v, self.num_heads, self.head_dim)?;
        let o = self.o_proj.forward(&attn_output)?;
        Ok(o)
    }

    /// Run the operation (see signature for params and return type).
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn tiled_attention_fn(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let tile_size = self.config.tile_size.unwrap_or(16);
        let attn_output = tiled_attention(q, k, v, self.num_heads, tile_size)?;
        let o = self.o_proj.forward(&attn_output)?;
        Ok(o)
    }

    /// Run the operation (see signature for params and return type).
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn flash_attention_fn(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let flash = GqaFlashAttention::new(self.num_heads, self.num_kv_heads, self.head_dim, true);
        let attn_output = flash.forward(q, k, v)?;
        let batch_size = q.dims()[0];
        let seq_len = q.dims()[2];
        let attn_output = attn_output.transpose(1, 2)?;
        let attn_output =
            attn_output.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&attn_output)
    }

    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    /// Prefill/decode attention: flash when `use_fused`, else tiled or paged matmul.
    pub fn run_attention_fn(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        if self.config.use_fused {
            return self.flash_attention_fn(q, k, v);
        }
        let tile_size = self.config.tile_size.unwrap_or(16);
        if q.dims()[2] > tile_size {
            self.tiled_attention_fn(q, k, v)
        } else {
            self.paged_attention_fn(q, k, v)
        }
    }

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

// Unit tests live in a separate file to keep this implementation file under
// the 800-line soft cap. The tests are included via `#[path = ...]` so they
// share the same scope (and `use super::*;` access) as if they were inline.
#[cfg(test)]
#[path = "gqa/tests.rs"]
mod tests;
