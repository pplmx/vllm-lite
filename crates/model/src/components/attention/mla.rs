//! Multi-head Latent Attention (MLA): compresses the KV cache by projecting keys + values into a shared low-rank latent, decompressing on the fly per attention step.
//!
//! Implements the DeepSeek-V2/V3 design. Achieves ~32× KV cache
//! compression at the cost of an extra low-rank projection per layer.
//! Used by Qwen3 (`Qwen3MlaAttention` wraps this) and DeepSeek-style
//! models.
//!
//! Tests for `MlaAttention` live in `tests.rs` (sibling file) to keep
//! this module under the 800-line soft cap.
#![allow(
    clippy::too_many_arguments,
    clippy::module_name_repetitions,
    clippy::similar_names
)]
// invariant: tensor-dimension casts (head_dim/seq_len -> f32) are bounded by
// model architecture constants; precision loss is intentional.
#![allow(clippy::cast_precision_loss)]

#[cfg(test)]
mod tests;

use candle_core::{Module, Result, Tensor};
use candle_nn::Linear;

use super::AttentionConfig;
use crate::components::positional::rope::{RopeScalingContext, apply_rope_with_scaling};

#[derive(Debug)]
/// `MlaAttention`. See the type definition for fields and behavior.
pub struct MlaAttention {
    q_proj: Linear,
    kv_proj: Linear,
    k_decompress: Linear,
    v_decompress: Linear,
    o_proj: Linear,
    q_lora_rank: usize,
    kv_lora_rank: usize,
    qk_nope_dim: usize,
    qk_rope_dim: usize,
    v_head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    config: AttentionConfig,
}

impl MlaAttention {
    /// Query attention head count.
    #[must_use]
    pub const fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Key/value head count (may differ from query heads in GQA-style MLA).
    #[must_use]
    pub const fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Combined per-head dimension after MLA decompression.
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Low-rank rank of the compressed KV latent.
    #[must_use]
    pub const fn kv_lora_rank(&self) -> usize {
        self.kv_lora_rank
    }

    /// Low-rank rank of the compressed query latent.
    #[must_use]
    pub const fn q_lora_rank(&self) -> usize {
        self.q_lora_rank
    }

    /// Shared attention hyper-parameters (dropout, scale, etc.).
    #[must_use]
    pub const fn config(&self) -> &AttentionConfig {
        &self.config
    }

    #[cfg(test)]
    /// Test-only accessor to the query projection linear layer.
    #[must_use]
    pub const fn q_proj_test(&self) -> &Linear {
        &self.q_proj
    }

    #[cfg(test)]
    /// Test-only accessor to the compressed KV projection linear layer.
    #[must_use]
    pub const fn kv_proj_test(&self) -> &Linear {
        &self.kv_proj
    }

    #[cfg(test)]
    /// Test-only accessor to the key decompression linear layer.
    #[must_use]
    pub const fn k_decompress_test(&self) -> &Linear {
        &self.k_decompress
    }

    #[cfg(test)]
    /// Test-only accessor to the value decompression linear layer.
    #[must_use]
    pub const fn v_decompress_test(&self) -> &Linear {
        &self.v_decompress
    }

    /// Split compressed MLA Q into position-agnostic and rotary parts.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    #[allow(dead_code)] // test-only helper; reachable under cfg(test) only
    pub(crate) fn split_q(
        &self,
        q_compressed: &Tensor,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let batch_size = q_compressed.dims()[0];
        let q_nope_dim = self.num_heads * self.qk_nope_dim;
        let q_rope_dim_total = self.num_heads * self.qk_rope_dim;

        let q_reshaped =
            q_compressed.reshape((batch_size, seq_len, q_nope_dim + q_rope_dim_total))?;
        let q_nope = q_reshaped.narrow(2, 0, q_nope_dim)?;
        let q_rope = q_reshaped.narrow(2, q_nope_dim, q_rope_dim_total)?;

        Ok((q_nope, q_rope))
    }

    /// Run the operation (see signature for params and return type).
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    #[allow(dead_code)]
    pub(crate) fn concat_q_nope_rope(&self, q_nope: &Tensor, q_rope: &Tensor) -> Result<Tensor> {
        let q = Tensor::cat(&[q_nope, q_rope], 2)?;
        let batch_size = q.dims()[0];
        let seq_len = q.dims()[1];
        let head_dim = self.qk_nope_dim + self.qk_rope_dim;
        let q = q.reshape((batch_size, seq_len, self.num_heads, head_dim))?;
        let q = q.transpose(1, 2)?;
        q.contiguous()
    }

    /// Run the operation (see signature for params and return type).
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    #[allow(dead_code)]
    pub(crate) fn reshape_k(
        &self,
        k_flat: &Tensor,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        let k = k_flat.reshape((batch_size, seq_len, self.num_kv_heads, self.v_head_dim))?;
        let k = k.transpose(1, 2)?;
        k.contiguous()
    }

    /// Run the operation (see signature for params and return type).
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    #[allow(dead_code)]
    pub(crate) fn reshape_v(
        &self,
        v_flat: &Tensor,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        let v = v_flat.reshape((batch_size, seq_len, self.num_kv_heads, self.v_head_dim))?;
        let v = v.transpose(1, 2)?;
        v.contiguous()
    }

    fn reshape_q_rope_for_rope(
        &self,
        q_rope_flat: &Tensor,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        q_rope_flat.reshape((batch_size, seq_len, self.num_heads, self.qk_rope_dim))
    }

    /// Run the layer forward pass over the input.
    /// # Caution: No causal masking
    ///
    /// This is a low-level primitive. It does **NOT** apply causal masking.
    /// Currently no production model architecture uses `MlaAttention`
    /// directly; the path is exposed for experimentation and benchmarking.
    /// When MLA is wired into a production decoder, callers must apply
    /// causal masking themselves (mirroring how `GqaAttention::forward`
    /// defers to `RopeGqaAttention::forward_prefill/forward_decode`).
    /// See [`crate::components::attention::util`] for causal-aware helpers.
    /// # Errors
    ///
    /// Returns `Err` if any tensor operation fails (shape mismatch, out-of-memory, dtype incompatibility, or kernel error).
    pub fn forward(&self, x: &Tensor, positions: &[i64]) -> Result<Tensor> {
        let batch_size = x.dims()[0];
        let seq_len = x.dims()[1];

        let q_compressed = self.q_proj.forward(x)?;
        let (q_nope, q_rope) = self.split_q(&q_compressed, seq_len)?;

        let q_rope_4d = self.reshape_q_rope_for_rope(&q_rope, batch_size, seq_len)?;
        // Use `apply_rope_with_scaling` with a default context so long-context
        // scaling can be wired through `RopeScaling` in a follow-up phase.
        // Default `RopeScalingContext` is a no-op (rope_type=Default,
        // scaling_factor=1.0) — identical numerical output to the previous
        // `apply_rope(&q, positions, 10000.0)` call.
        let q_rope_rotated_4d = apply_rope_with_scaling(
            &q_rope_4d,
            positions,
            10000.0,
            RopeScalingContext::default(),
        )?;
        let q_rope_rotated =
            q_rope_rotated_4d.reshape((batch_size, seq_len, self.num_heads * self.qk_rope_dim))?;

        let kv_compressed = self.kv_proj.forward(x)?;
        let k_decompressed = self.k_decompress.forward(&kv_compressed)?;
        let k = self.reshape_k(&k_decompressed, batch_size, seq_len)?;
        let v_decompressed = self.v_decompress.forward(&kv_compressed)?;
        let v = self.reshape_v(&v_decompressed, batch_size, seq_len)?;

        let q_nope_4d = self.reshape_q_nope_for_attn(&q_nope, batch_size, seq_len)?;
        let attn_output = self.attention_with_compressed_kv(&q_nope_4d, &k, &v)?;

        let q_concat = Tensor::cat(&[&attn_output, &q_rope_rotated], 2)?;
        let o = self.o_proj.forward(&q_concat)?;

        Ok(o)
    }

    fn reshape_q_nope_for_attn(
        &self,
        q_nope_flat: &Tensor,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        let q_nope_4d =
            q_nope_flat.reshape((batch_size, seq_len, self.num_heads, self.qk_nope_dim))?;
        let q_nope_transposed = q_nope_4d.transpose(1, 2)?;
        q_nope_transposed.contiguous()
    }

    fn attention_with_compressed_kv(
        &self,
        q_nope: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<Tensor> {
        let batch_size = q_nope.dims()[0];
        let seq_len = q_nope.dims()[2];
        let num_heads = self.num_heads;

        let q = q_nope.contiguous()?;
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let qk = q.matmul(&k_t)?;

        // H-12 #1: replaced `qk.mul(broadcast(scalar_tensor))` with `qk.affine(scale, 0.0)`.
        // Per H-9 profile (HIGH #1): `Tensor::new(&[scale], device)` was allocated and
        // broadcast to O(B*H*S*S) every forward; `affine` fuses the scaling into the
        // kernel without materializing a broadcast tensor.
        let scale = 1.0 / (self.v_head_dim as f32).sqrt();
        let qk = qk.affine(f64::from(scale), 0.0)?;

        // H-12 #3: `candle_nn::ops::softmax` already returns a contiguous tensor
        // (verified in candle-nn 0.10.2 src/ops.rs:22-29 — final op is
        // `broadcast_div`, which produces a fresh contiguous tensor). The
        // explicit `.contiguous()?` is a redundant is_contiguous check. Mirrors
        // H-11 #3 (GQA). `v.contiguous()?` on the next line is still required:
        // matmul rejects non-contiguous batch dims per H-11 #3 regression test.
        let attn_weights = candle_nn::ops::softmax(&qk, 3)?;

        let attn_output = attn_weights.matmul(&v.contiguous()?)?;
        let attn_output = attn_output.transpose(1, 2)?;
        let attn_output =
            attn_output.reshape((batch_size, seq_len, num_heads * self.v_head_dim))?;

        Ok(attn_output)
    }

    #[allow(clippy::too_many_arguments)]
    /// Construct a new instance from the given configuration.
    /// # Errors
    ///
    /// Returns `Err` if any required tensor allocation or weight loading fails.
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        q_lora_rank: usize,
        kv_lora_rank: usize,
        qk_nope_dim: usize,
        qk_rope_dim: usize,
        v_head_dim: usize,
        vb: Option<candle_nn::VarBuilder<'_>>,
        config: AttentionConfig,
    ) -> Result<Self> {
        let vb = vb.unwrap_or_else(|| {
            candle_nn::VarBuilder::zeros(candle_core::DType::F32, &candle_core::Device::Cpu)
        });

        let q_proj = candle_nn::linear(hidden_size, q_lora_rank, vb.pp("q_proj"))?;
        let kv_proj = candle_nn::linear(hidden_size, kv_lora_rank, vb.pp("kv_proj"))?;

        let k_decompress_out_dim = num_kv_heads * v_head_dim;
        let k_decompress =
            candle_nn::linear(kv_lora_rank, k_decompress_out_dim, vb.pp("k_decompress"))?;
        let v_decompress =
            candle_nn::linear(kv_lora_rank, k_decompress_out_dim, vb.pp("v_decompress"))?;

        let head_dim = qk_nope_dim + qk_rope_dim;
        let o_proj = candle_nn::linear(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            q_proj,
            kv_proj,
            k_decompress,
            v_decompress,
            o_proj,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_dim,
            qk_rope_dim,
            v_head_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            config,
        })
    }
}
