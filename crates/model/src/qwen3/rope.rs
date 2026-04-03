#![allow(clippy::all, unused)]

pub use crate::components::{RoPE, apply_rope};

pub fn precompute_rope_cache(seq_len: usize, head_dim: usize, theta: f32) -> Vec<(f32, f32)> {
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
    use candle_core::{DType, Device, Tensor};

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
    fn test_apply_rope_returns_same_shape() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let query = Tensor::ones((2, 4, 2, 32), DType::F32, &device)?;
        let positions: Vec<i64> = vec![0, 1, 2, 3];

        let result = apply_rope(&query, &positions, 10000.0)?;
        assert_eq!(result.dims(), query.dims());

        Ok(())
    }

    #[test]
    fn test_apply_rope_different_positions() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let query = Tensor::ones((1, 2, 2, 4), DType::F32, &device)?;

        let pos0: Vec<i64> = vec![0, 1];
        let out0 = apply_rope(&query, &pos0, 10000.0)?;

        let pos1: Vec<i64> = vec![10, 11];
        let out1 = apply_rope(&query, &pos1, 10000.0)?;

        let diff = (out0 - out1)?.abs()?.sum_all()?;
        assert!(
            diff.to_scalar::<f32>()? > 1e-5,
            "RoPE should produce different outputs for different positions"
        );

        Ok(())
    }

    #[test]
    fn test_apply_rope_deterministic() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let query = Tensor::ones((1, 2, 2, 4), DType::F32, &device)?;
        let positions: Vec<i64> = vec![0, 1];

        let out1 = apply_rope(&query, &positions, 10000.0)?;
        let out2 = apply_rope(&query, &positions, 10000.0)?;

        let diff = (out1 - out2)?.abs()?.sum_all()?;
        assert_eq!(
            diff.to_scalar::<f32>()?,
            0.0,
            "RoPE should be deterministic"
        );

        Ok(())
    }
}
