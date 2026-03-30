#![allow(clippy::all, unused)]
use crate::kv_cache::PagedKvCache;
use candle_core::{Device, Module, Result, Shape, Tensor};
use candle_nn::Linear;

pub struct GqaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl GqaAttention {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        vb: Option<candle_nn::VarBuilder>,
    ) -> Result<Self> {
        let vb = vb.unwrap_or_else(|| {
            candle_nn::VarBuilder::zeros(candle_core::DType::F32, &candle_core::Device::Cpu)
        });

        let q_proj = candle_nn::linear(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
        })
    }

    pub fn new_with_weights(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        q_weight: Tensor,
        k_weight: Tensor,
        v_weight: Tensor,
        o_weight: Tensor,
    ) -> Result<Self> {
        let q_proj = Linear::new(q_weight, None);
        let k_proj = Linear::new(k_weight, None);
        let v_proj = Linear::new(v_weight, None);
        let o_proj = Linear::new(o_weight, None);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.dims()[0];
        let seq_len = x.dims()[1];

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;

        let k = self.expand_kv(&k, self.num_heads, self.num_kv_heads)?;
        let v = self.expand_kv(&v, self.num_heads, self.num_kv_heads)?;

        let k = k.transpose(2, 3)?;
        let qk = Tensor::matmul(&q, &k)?;

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let qk = qk.mul(&Tensor::new(&[scale], q.device())?.broadcast_as(qk.dims())?)?;
        let attn_weights = candle_nn::ops::softmax(&qk, 3)?;

        let attn_output = Tensor::matmul(&attn_weights, &v)?;

        let attn_output =
            attn_output.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        let o = self.o_proj.forward(&attn_output)?;
        Ok(o)
    }

    pub fn expand_kv(
        &self,
        kv: &Tensor,
        num_q_heads: usize,
        num_kv_heads: usize,
    ) -> Result<Tensor> {
        if num_q_heads == num_kv_heads {
            return Ok(kv.clone());
        }

        let repeat_factor = num_q_heads / num_kv_heads;
        let (batch, seq, heads, dim) = kv.dims4()?;

        // Manual expansion: repeat each KV head repeat_factor times
        // Input: [batch, seq, num_kv_heads, head_dim]
        // Output: [batch, seq, num_q_heads, head_dim]

        // Flatten batch*seq, then repeat each head
        let kv = kv.reshape((batch * seq, heads, dim))?; // [batch*seq, heads, dim]

        // Create output by repeating each head
        let mut result_parts = Vec::with_capacity(batch * seq * heads * repeat_factor);
        for i in 0..(batch * seq) {
            for h in 0..heads {
                let head_data = kv.narrow(0, i, 1)?.squeeze(0)?; // [heads, dim]
                let head_data = head_data.narrow(0, h, 1)?.squeeze(0)?; // [dim]
                for _ in 0..repeat_factor {
                    result_parts.push(head_data.clone());
                }
            }
        }

        // Stack back: [batch*seq, heads*repeat_factor, dim]
        let expanded = Tensor::stack(&result_parts, 0)?;
        let expanded = expanded.reshape((batch, seq, heads * repeat_factor, dim))?;

        Ok(expanded)
    }

    pub fn forward_prefill(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
    ) -> Result<Tensor> {
        let batch_size = x.dims()[0];
        let seq_len = x.dims()[1];

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        for token_idx in 0..seq_len {
            let block_id = token_idx / crate::kv_cache::BLOCK_SIZE;
            let offset = token_idx % crate::kv_cache::BLOCK_SIZE;

            let k_slice = k.narrow(2, token_idx, 1)?.transpose(0, 1)?;
            let v_slice = v.narrow(2, token_idx, 1)?.transpose(0, 1)?;

            let k_slice = k_slice.reshape((1, self.num_kv_heads, self.head_dim))?;
            let v_slice = v_slice.reshape((1, self.num_kv_heads, self.head_dim))?;

            kv_cache.write_kv(layer_idx, block_id, offset, &k_slice, &v_slice)?;
        }

        let k_expanded = self.expand_kv(&k.transpose(1, 2)?, self.num_heads, self.num_kv_heads)?;
        let v_expanded = self.expand_kv(&v.transpose(1, 2)?, self.num_heads, self.num_kv_heads)?;
        let k_expanded = k_expanded.transpose(1, 2)?;
        let v_expanded = v_expanded.transpose(1, 2)?;

        self.paged_attention(&q, &k_expanded, &v_expanded, seq_len)
    }

    pub fn forward_decode(
        &self,
        x: &Tensor,
        kv_cache: &PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        num_computed_tokens: usize,
    ) -> Result<Tensor> {
        let batch_size = x.dims()[0];

        let q = self.q_proj.forward(x)?;
        let q = q
            .reshape((batch_size, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (k, v) = kv_cache.read_kv(layer_idx, block_ids, num_computed_tokens)?;

        let k = k.transpose(0, 1)?.transpose(1, 2)?;
        let v = v.transpose(0, 1)?.transpose(1, 2)?;

        let k_expanded = self.expand_kv(&k, self.num_heads, self.num_kv_heads)?;
        let v_expanded = self.expand_kv(&v, self.num_heads, self.num_kv_heads)?;
        let k_expanded = k_expanded.transpose(1, 2)?;
        let v_expanded = v_expanded.transpose(1, 2)?;

        self.paged_attention(&q, &k_expanded, &v_expanded, num_computed_tokens + 1)
    }

    fn paged_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        seq_len: usize,
    ) -> Result<Tensor> {
        let qk = Tensor::matmul(q, &k.transpose(2, 3)?)?;

        let mask = self.causal_mask(seq_len, q.device())?;
        let qk = (&qk + &mask)?;

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let qk = qk.mul(&Tensor::new(&[scale], q.device())?)?;
        let attn_weights = candle_nn::ops::softmax(&qk, 3)?;

        let attn_output = Tensor::matmul(&attn_weights, v)?;

        let attn_output = attn_output.transpose(1, 2)?;
        let attn_output = attn_output.reshape((q.dims()[0], 1, self.num_heads * self.head_dim))?;

        let o = self.o_proj.forward(&attn_output)?;
        Ok(o)
    }

    fn causal_mask(&self, seq_len: usize, device: &Device) -> Result<Tensor> {
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if i >= j { 0.0 } else { f32::NEG_INFINITY }))
            .collect();
        Tensor::from_slice(&mask, (1, 1, seq_len, seq_len), device)
    }
}
