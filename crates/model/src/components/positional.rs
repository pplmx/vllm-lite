use crate::config::Qwen3Config;
use candle_core::{Result, Tensor};

#[allow(dead_code)]
pub struct RoPE {
    pub(crate) theta: f32,
    pub(crate) head_dim: usize,
    pub(crate) scaling_factor: f32,
}

impl RoPE {
    pub fn new(theta: f32, head_dim: usize) -> Self {
        Self {
            theta,
            head_dim,
            scaling_factor: 1.0,
        }
    }

    pub fn new_with_config(config: &Qwen3Config) -> Self {
        let rope_scaling = config.rope_scaling();
        Self {
            theta: config.rope_theta(),
            head_dim: config.hidden_size() / config.num_attention_heads(),
            scaling_factor: rope_scaling.and_then(|r| r.factor).unwrap_or(1.0),
        }
    }

    pub fn scaling_factor(&self) -> f32 {
        self.scaling_factor
    }

    pub fn apply(&self, x: &Tensor, positions: &[i64]) -> Result<Tensor> {
        apply_rope(x, positions, self.theta)
    }
}

pub fn apply_rope(query: &Tensor, positions: &[i64], theta: f32) -> Result<Tensor> {
    let (batch, seq_len, num_heads, head_dim) = query.dims4()?;

    let query = query.transpose(1, 2)?;

    let half_dim = head_dim / 2;

    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| theta.powf(-2.0 * (i as f32) / (head_dim as f32)))
        .collect();

    let mut cos_matrix = Vec::with_capacity(seq_len * half_dim);
    let mut sin_matrix = Vec::with_capacity(seq_len * half_dim);

    for &pos in positions {
        let pos_f = pos as f32;
        for &freq in &inv_freq {
            let angle = pos_f * freq;
            cos_matrix.push(angle.cos());
            sin_matrix.push(angle.sin());
        }
    }

    let cos = Tensor::new(cos_matrix.as_slice(), query.device())?
        .reshape((1, seq_len, half_dim))?
        .broadcast_as((batch, num_heads, seq_len, half_dim))?;
    let sin = Tensor::new(sin_matrix.as_slice(), query.device())?
        .reshape((1, seq_len, half_dim))?
        .broadcast_as((batch, num_heads, seq_len, half_dim))?;

    let first_half = query.narrow(3, 0, half_dim)?;
    let second_half = query.narrow(3, half_dim, half_dim)?;

    let rotated_first = first_half
        .broadcast_mul(&cos)?
        .broadcast_add(&second_half.broadcast_mul(&sin)?)?;
    let rotated_second = second_half
        .broadcast_mul(&cos)?
        .broadcast_sub(&first_half.broadcast_mul(&sin)?)?;

    let result = Tensor::cat(&[&rotated_first, &rotated_second], 3)?;

    result.transpose(1, 2)
}
