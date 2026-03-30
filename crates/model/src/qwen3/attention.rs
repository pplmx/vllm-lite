#![allow(clippy::all, unused)]
use candle_core::{Module, Result, Shape, Tensor};
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
        let qk = qk.mul(&Tensor::new(&[scale], q.device())?)?;
        let attn_weights = candle_nn::ops::softmax(&qk, 3)?;

        let attn_output = Tensor::matmul(&attn_weights, &v)?;

        let attn_output =
            attn_output.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        let o = self.o_proj.forward(&attn_output)?;
        Ok(o)
    }

    fn expand_kv(&self, kv: &Tensor, num_q_heads: usize, num_kv_heads: usize) -> Result<Tensor> {
        if num_q_heads == num_kv_heads {
            return Ok(kv.clone());
        }

        let repeat_factor = num_q_heads / num_kv_heads;
        let (batch, seq, heads, dim) = kv.dims4()?;

        let new_shape = vec![batch, seq, heads * repeat_factor, dim];
        let kv = kv.repeat(Shape::from(new_shape.as_slice()))?;
        let kv = kv.reshape((batch, seq, num_q_heads, dim))?;

        Ok(kv)
    }
}
