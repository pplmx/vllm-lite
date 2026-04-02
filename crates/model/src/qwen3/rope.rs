#![allow(clippy::all, unused)]
use crate::config::Qwen3Config;
use candle_core::{Result, Tensor};

pub struct RoPE {
    theta: f32,
    dim: usize,
    scaling_factor: f32,
}

impl RoPE {
    pub fn new(config: &Qwen3Config) -> Self {
        let rope_scaling = config.rope_scaling();
        Self {
            theta: config.rope_theta(),
            dim: config.hidden_size() / config.num_attention_heads(),
            scaling_factor: rope_scaling.and_then(|r| r.factor).unwrap_or(1.0),
        }
    }

    pub fn scaling_factor(&self) -> f32 {
        self.scaling_factor
    }
}

pub fn apply_rope(query: &Tensor, position_ids: &Tensor, theta: f32) -> Result<Tensor> {
    // query shape: [batch, seq_len, num_heads, head_dim]
    // position_ids shape: [seq_len]

    let (batch, seq_len, num_heads, head_dim) = query.dims4()?;

    // Reshape to [batch, num_heads, seq_len, head_dim] for easier processing
    let query = query.transpose(1, 2)?;

    // Get positions
    let positions = position_ids.to_vec1::<i64>()?;

    let half_dim = head_dim / 2;

    // Compute inv_freq: theta^(-2i/d) for i in 0..half_dim
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| theta.powf(-2.0 * (i as f32) / (head_dim as f32)))
        .collect();

    // Compute cos and sin for each position
    // Result shape: [seq_len, half_dim]
    let mut cos_matrix = Vec::with_capacity(seq_len * half_dim);
    let mut sin_matrix = Vec::with_capacity(seq_len * half_dim);

    for &pos in &positions {
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

    // Split head_dim into first_half and second_half
    let first_half = query.narrow(3, 0, half_dim)?;
    let second_half = query.narrow(3, half_dim, half_dim)?;

    // first_half * cos + second_half * sin
    let rotated_first = first_half
        .broadcast_mul(&cos)?
        .broadcast_add(&second_half.broadcast_mul(&sin)?)?;
    // second_half * cos - first_half * sin
    let rotated_second = second_half
        .broadcast_mul(&cos)?
        .broadcast_sub(&first_half.broadcast_mul(&sin)?)?;

    let result = Tensor::cat(&[&rotated_first, &rotated_second], 3)?;

    // Back to [batch, seq_len, num_heads, head_dim]
    result.transpose(1, 2)
}

pub fn precompute_rope_cache(seq_len: usize, head_dim: usize, theta: f32) -> Vec<(f32, f32)> {
    // Precompute cos and sin values for positions 0..seq_len
    let mut cache = Vec::with_capacity(seq_len * head_dim / 2);
    for pos in 0..seq_len {
        for i in 0..head_dim / 2 {
            let freq = (pos as f32).powf(-2.0 * (i as f32) / (head_dim as f32)) * theta;
            cache.push((freq.cos(), freq.sin()));
        }
    }
    cache
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precompute_rope_cache_length() {
        let cache = precompute_rope_cache(10, 64, 10000.0);
        assert_eq!(cache.len(), 10 * 32);
    }

    #[test]
    fn test_precompute_rope_cache_first_position() {
        let cache = precompute_rope_cache(1, 64, 10000.0);
        assert_eq!(cache.len(), 32);
    }

    #[test]
    fn test_precompute_rope_cache_values() {
        let cache = precompute_rope_cache(10, 64, 10000.0);
        assert_eq!(cache.len(), 320);
    }

    #[test]
    fn test_apply_rope_returns_same_shape() -> Result<()> {
        use candle_core::{DType, Device, Tensor};

        let device = Device::Cpu;
        // [batch=2, seq=4, num_heads=2, head_dim=32] = 2*4*2*32 = 512
        let query = Tensor::ones((2, 4, 2, 32), DType::F32, &device)?;
        let position_ids = Tensor::new(&[0i64, 1i64, 2i64, 3i64], &device)?;

        let result = apply_rope(&query, &position_ids, 10000.0)?;
        assert_eq!(result.dims(), query.dims());

        Ok(())
    }

    #[test]
    fn test_apply_rope_different_positions() -> Result<()> {
        use candle_core::{DType, Device, Tensor};

        let device = Device::Cpu;
        let query = Tensor::ones((1, 2, 2, 4), DType::F32, &device)?;

        let pos0 = Tensor::new(&[0i64, 1i64], &device)?;
        let out0 = apply_rope(&query, &pos0, 10000.0)?;

        let pos1 = Tensor::new(&[10i64, 11i64], &device)?;
        let out1 = apply_rope(&query, &pos1, 10000.0)?;

        let diff = (out0 - out1)?.abs()?.sum_all()?;
        assert!(
            diff.to_scalar::<f32>()? > 1e-5,
            "RoPE should produce different outputs for different positions"
        );

        Ok(())
    }

    #[test]
    fn test_apply_rope_deterministic() -> Result<()> {
        use candle_core::{DType, Device, Tensor};

        let device = Device::Cpu;
        let query = Tensor::ones((1, 2, 2, 4), DType::F32, &device)?;
        let position_ids = Tensor::new(&[0i64, 1i64], &device)?;

        let out1 = apply_rope(&query, &position_ids, 10000.0)?;
        let out2 = apply_rope(&query, &position_ids, 10000.0)?;

        let diff = (out1 - out2)?.abs()?.sum_all()?;
        assert_eq!(
            diff.to_scalar::<f32>()?,
            0.0,
            "RoPE should be deterministic"
        );

        Ok(())
    }
}
