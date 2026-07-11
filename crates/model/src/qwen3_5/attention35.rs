#![allow(clippy::module_name_repetitions)]
//! Qwen3.5 full-attention layer with MRoPE and paged KV cache.

use candle_core::{Module, Result as CandleResult, Tensor};
use candle_nn::Linear;
use std::collections::HashMap;

use crate::components::attention::expand_kv;
use crate::components::attention::paged_gqa::{
    compute_gqa_attention, prefill_causal_mask, prefill_continue_causal_mask,
    project_attention_output, read_decode_kv, write_prefill_kv,
};
use crate::components::positional::MRoPE;
use crate::paged_tensor::PagedKvCache;

#[derive(Debug)]
/// `Attention35WithRoPE`. See the type definition for fields and behavior.
pub struct Attention35WithRoPE {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    rope: MRoPE,
}

impl Attention35WithRoPE {
    /// Construct a new instance from the given configuration.
    /// # Errors
    ///
    /// Returns `Err` if any required tensor allocation or weight loading fails.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope: MRoPE,
        vb: candle_nn::VarBuilder<'_>,
    ) -> CandleResult<Self> {
        Ok(Self {
            q_proj: candle_nn::linear(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?,
            k_proj: candle_nn::linear(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?,
            v_proj: candle_nn::linear(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?,
            o_proj: candle_nn::linear(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?,
            num_heads,
            num_kv_heads,
            head_dim,
            rope,
        })
    }

    /// Build from weights.
    /// # Errors
    ///
    /// Returns `Err` if reading or parsing the source fails.
    pub fn from_weights(
        prefix: &str,
        weights: &HashMap<String, Tensor>,
        _hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope: MRoPE,
    ) -> CandleResult<Self> {
        let q_proj_key = format!("{prefix}.self_attn.q_proj.weight");
        let k_proj_key = format!("{prefix}.self_attn.k_proj.weight");
        let v_proj_key = format!("{prefix}.self_attn.v_proj.weight");
        let o_proj_key = format!("{prefix}.self_attn.o_proj.weight");

        let q_w = weights
            .get(&q_proj_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {q_proj_key}")))?;
        let k_w = weights
            .get(&k_proj_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {k_proj_key}")))?;
        let v_w = weights
            .get(&v_proj_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {v_proj_key}")))?;
        let o_w = weights
            .get(&o_proj_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {o_proj_key}")))?;

        Ok(Self {
            q_proj: Linear::new(q_w, None),
            k_proj: Linear::new(k_w, None),
            v_proj: Linear::new(v_w, None),
            o_proj: Linear::new(o_w, None),
            num_heads,
            num_kv_heads,
            head_dim,
            rope,
        })
    }

    /// Run the layer forward pass over the input.
    /// # Errors
    ///
    /// Returns `Err` if any tensor operation fails (shape mismatch, out-of-memory, dtype incompatibility, or kernel error).
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        let positions: Vec<usize> = (0..seq_len).collect();
        let (q, k, v) = self.project_qkv(x, batch, seq_len)?;
        let (q, k) = self.apply_mrope(&q, &k, &positions)?;
        self.compute_attention(&q, &k, &v, batch, seq_len, true)
    }

    /// Run the prefill path: process the full prompt and cache its KV.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn forward_prefill(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        positions: &[usize],
    ) -> CandleResult<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        let (q, k, v) = self.project_qkv(x, batch, seq_len)?;
        let (q, k) = self.apply_mrope(&q, &k, positions)?;

        let k_expanded = expand_kv(&k, self.num_heads, self.num_kv_heads)?;
        let v_expanded = expand_kv(&v, self.num_heads, self.num_kv_heads)?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k_expanded = k_expanded.transpose(1, 2)?.contiguous()?;
        let v_expanded = v_expanded.transpose(1, 2)?.contiguous()?;

        write_prefill_kv(
            kv_cache,
            layer_idx,
            block_ids,
            positions,
            seq_len,
            &k_expanded,
            &v_expanded,
        )?;

        self.compute_paged_attention(&q, &k_expanded, &v_expanded, seq_len)
    }

    /// Chunked-prefill continuation: attend over the cached prefix plus new tokens.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn forward_prefill_continue(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        positions: &[usize],
        num_computed_tokens: usize,
    ) -> CandleResult<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        let (q, k, v) = self.project_qkv(x, batch, seq_len)?;
        let (q, k) = self.apply_mrope(&q, &k, positions)?;

        let k_expanded = expand_kv(&k, self.num_heads, self.num_kv_heads)?;
        let v_expanded = expand_kv(&v, self.num_heads, self.num_kv_heads)?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k_new = k_expanded.transpose(1, 2)?.contiguous()?;
        let v_new = v_expanded.transpose(1, 2)?.contiguous()?;

        let (cached_k, cached_v) =
            kv_cache.read_kv(layer_idx, block_ids, num_computed_tokens)?;
        let cached_k = cached_k
            .transpose(0, 1)?
            .unsqueeze(0)?
            .contiguous()?;
        let cached_v = cached_v
            .transpose(0, 1)?
            .unsqueeze(0)?
            .contiguous()?;

        write_prefill_kv(
            kv_cache,
            layer_idx,
            block_ids,
            positions,
            seq_len,
            &k_new,
            &v_new,
        )?;

        let full_k = Tensor::cat(&[&cached_k, &k_new], 2)?.contiguous()?;
        let full_v = Tensor::cat(&[&cached_v, &v_new], 2)?.contiguous()?;

        let mask = prefill_continue_causal_mask(seq_len, num_computed_tokens, q.device())?;
        let attn_output =
            compute_gqa_attention(&q, &full_k, &full_v, self.head_dim, Some(&mask))?;
        project_attention_output(
            &attn_output,
            batch,
            seq_len,
            self.num_heads,
            self.head_dim,
            &self.o_proj,
        )
    }

    /// Run the decode path: process one new token against cached KV.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn forward_decode(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        num_computed_tokens: usize,
        positions: &[usize],
    ) -> CandleResult<Tensor> {
        let batch_size = x.dims()[0];
        let seq_len = if x.dims().len() == 3 { x.dims()[1] } else { 1 };

        let (q, k, v) = self.project_qkv(x, batch_size, seq_len)?;
        let (q, k) = self.apply_mrope(&q, &k, positions)?;

        let k_expanded = expand_kv(&k, self.num_heads, self.num_kv_heads)?;
        let v_expanded = expand_kv(&v, self.num_heads, self.num_kv_heads)?;

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

        self.compute_paged_attention(&q, &full_k, &full_v, 1)
    }

    fn project_qkv(
        &self,
        x: &Tensor,
        batch: usize,
        seq_len: usize,
    ) -> CandleResult<(Tensor, Tensor, Tensor)> {
        let q = self
            .q_proj
            .forward(x)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?;
        let k =
            self.k_proj
                .forward(x)?
                .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;
        let v =
            self.v_proj
                .forward(x)?
                .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;
        Ok((q, k, v))
    }

    fn apply_mrope(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &[usize],
    ) -> CandleResult<(Tensor, Tensor)> {
        // invariant: positions are bounded by sequence length; usize -> i64
        // conversion is non-wrapping for any realistic sequence.
        #[allow(clippy::cast_possible_wrap)]
        let positions_i64: Vec<i64> = positions.iter().map(|&p| p as i64).collect();
        self.rope.apply(q, k, &positions_i64)
    }

    fn compute_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        batch: usize,
        seq_len: usize,
        apply_causal: bool,
    ) -> CandleResult<Tensor> {
        let k = expand_kv(k, self.num_heads, self.num_kv_heads)?;
        let v = expand_kv(v, self.num_heads, self.num_kv_heads)?;
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let mask = if apply_causal {
            prefill_causal_mask(seq_len, seq_len, q.device())?
        } else {
            None
        };
        let attn_output = compute_gqa_attention(&q, &k, &v, self.head_dim, mask.as_ref())?;
        project_attention_output(
            &attn_output,
            batch,
            seq_len,
            self.num_heads,
            self.head_dim,
            &self.o_proj,
        )
    }

    fn compute_paged_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        seq_len: usize,
    ) -> CandleResult<Tensor> {
        let batch_size = q.dims()[0];
        let kv_seq = k.dims()[2];
        let mask = prefill_causal_mask(seq_len, kv_seq, q.device())?;
        let attn_output = compute_gqa_attention(q, k, v, self.head_dim, mask.as_ref())?;
        project_attention_output(
            &attn_output,
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
            &self.o_proj,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    #[test]
    fn test_attention35_forward_shape() {
        let device = Device::Cpu;
        let rope = MRoPE::new(32, 10_000.0, vec![10, 10, 12], 0.25);
        let attn =
            Attention35WithRoPE::new(128, 2, 2, 32, rope, VarBuilder::zeros(DType::F32, &device))
                .unwrap();
        let x = Tensor::randn(0.0f32, 1.0, (1, 4, 128), &device).unwrap();
        let out = attn.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 4, 128]);
    }
}
