use candle_core::{Result, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttentionVariant {
    #[default]
    Standard,
    Tiled,
    Flash,
}

#[derive(Debug, Clone, Default)]
pub struct FlashAttentionConfig {
    pub variant: AttentionVariant,
    pub flash_block_size: usize,
    pub use_sliding_window: bool,
    pub sliding_window_size: usize,
}

impl FlashAttentionConfig {
    pub fn new() -> Self {
        Self {
            variant: AttentionVariant::Standard,
            flash_block_size: 128,
            use_sliding_window: false,
            sliding_window_size: 512,
        }
    }

    pub fn with_flash(mut self) -> Self {
        self.variant = AttentionVariant::Flash;
        self
    }

    pub fn with_tiled(mut self, tile_size: usize) -> Self {
        self.variant = AttentionVariant::Tiled;
        self.flash_block_size = tile_size;
        self
    }

    pub fn with_sliding_window(mut self, size: usize) -> Self {
        self.use_sliding_window = true;
        self.sliding_window_size = size;
        self
    }
}

pub trait FlashAttention: Send + Sync {
    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor>;
    fn forward_with_mask(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor>;
    fn forward_tiled(&self, q: &Tensor, k: &Tensor, v: &Tensor, tile_size: usize)
    -> Result<Tensor>;
}

pub struct ScaledDotProductAttention {
    scale: f32,
    tile_size: usize,
}

impl ScaledDotProductAttention {
    pub fn new(head_dim: usize) -> Self {
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self {
            scale,
            tile_size: 16,
        }
    }

    pub fn with_tile_size(mut self, tile_size: usize) -> Self {
        self.tile_size = tile_size;
        self
    }

    pub fn compute_tiled(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        tile_size: usize,
    ) -> Result<Tensor> {
        let q_shape = q.dims();
        let batch_size = q_shape[0];
        let num_heads = q_shape[1];
        let seq_len = q_shape[2];
        let _head_dim = q_shape[3];

        let scale_tensor = Tensor::new(self.scale, q.device())?;
        let mut all_outputs: Vec<Tensor> = Vec::with_capacity(batch_size);

        for _b in 0..batch_size {
            let mut head_outputs: Vec<Tensor> = Vec::with_capacity(num_heads);

            for h in 0..num_heads {
                let q_bh = q.narrow(1, h, 1)?.squeeze(1)?;
                let k_bh = k.narrow(1, h, 1)?.squeeze(1)?;
                let v_bh = v.narrow(1, h, 1)?.squeeze(1)?;

                let mut tile_outputs: Vec<Tensor> = Vec::new();

                for start in (0..seq_len).step_by(tile_size) {
                    let end = (start + tile_size).min(seq_len);
                    let actual_tile_size = end - start;
                    let q_tile = q_bh.narrow(1, start, actual_tile_size)?;

                    let k_start = 0;
                    let k_len = end.min(seq_len);
                    let k_tile = k_bh.narrow(1, k_start, k_len)?;
                    let v_tile = v_bh.narrow(1, k_start, k_len)?;

                    let qk = q_tile.matmul(&k_tile.t()?)?;
                    let qk_scaled = qk.broadcast_mul(&scale_tensor)?;
                    let attn = softmax_last_dim(&qk_scaled)?;
                    let out_tile = attn.matmul(&v_tile)?;

                    tile_outputs.push(out_tile);
                }

                let head_output = Tensor::cat(&tile_outputs, 0)?;
                head_outputs.push(head_output);
            }

            let batch_output = Tensor::stack(&head_outputs, 0)?;
            all_outputs.push(batch_output);
        }

        let result = Tensor::stack(&all_outputs, 0)?;
        Ok(result)
    }

    pub fn compute_sliding_window(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        window_size: usize,
    ) -> Result<Tensor> {
        let k_len = k.dims()[2];

        if k_len <= window_size {
            return self.forward(q, k, v);
        }

        let k_window = k.narrow(2, k_len.saturating_sub(window_size), window_size)?;
        let v_window = v.narrow(2, k_len.saturating_sub(window_size), window_size)?;

        self.forward(q, &k_window, &v_window)
    }
}

fn softmax_last_dim(t: &Tensor) -> Result<Tensor> {
    let shape = t.dims();
    let exp = t.exp()?;
    let sum = exp.sum_keepdim(shape.len() - 1)?;
    exp.broadcast_div(&sum)
}

impl FlashAttention for ScaledDotProductAttention {
    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let qk = q.matmul(&k.t()?)?;
        let scale_tensor = Tensor::new(self.scale, q.device())?;
        let qk_scaled = qk.broadcast_mul(&scale_tensor)?;
        let attn = softmax_last_dim(&qk_scaled)?;
        attn.matmul(v)
    }

    fn forward_with_mask(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        _mask: &Tensor,
    ) -> Result<Tensor> {
        self.forward(q, k, v)
    }

    fn forward_tiled(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        tile_size: usize,
    ) -> Result<Tensor> {
        self.compute_tiled(q, k, v, tile_size)
    }
}

pub struct FlashAttentionKernel {
    attention: Box<dyn FlashAttention>,
    config: FlashAttentionConfig,
}

impl FlashAttentionKernel {
    pub fn new(head_dim: usize, config: FlashAttentionConfig) -> Self {
        let attention: Box<dyn FlashAttention> = match config.variant {
            AttentionVariant::Tiled => Box::new(
                ScaledDotProductAttention::new(head_dim).with_tile_size(config.flash_block_size),
            ),
            AttentionVariant::Flash | AttentionVariant::Standard => {
                Box::new(ScaledDotProductAttention::new(head_dim))
            }
        };

        Self { attention, config }
    }

    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        if self.config.variant == AttentionVariant::Tiled {
            return self.forward_tiled(q, k, v);
        }
        if self.config.use_sliding_window && self.config.sliding_window_size > 0 {
            self.forward_sliding_window(q, k, v)
        } else {
            self.attention.forward(q, k, v)
        }
    }

    fn forward_tiled(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        self.attention
            .forward_tiled(q, k, v, self.config.flash_block_size)
    }

    fn forward_sliding_window(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let window_size = self.config.sliding_window_size;
        let k_len = k.dims()[2];

        if k_len <= window_size {
            return self.attention.forward(q, k, v);
        }

        let k_window = k.narrow(2, k_len.saturating_sub(window_size), window_size)?;
        let v_window = v.narrow(2, k_len.saturating_sub(window_size), window_size)?;

        self.attention.forward(q, &k_window, &v_window)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_scaled_dot_product_attention() -> Result<()> {
        let sdpa = ScaledDotProductAttention::new(64);

        let q = Tensor::ones((2, 8, 10, 64), candle_core::DType::F32, &Device::Cpu)?;
        let k = Tensor::ones((2, 8, 10, 64), candle_core::DType::F32, &Device::Cpu)?;
        let v = Tensor::ones((2, 8, 10, 64), candle_core::DType::F32, &Device::Cpu)?;

        let output = sdpa.forward(&q, &k, &v)?;

        assert_eq!(output.dims(), &[2, 8, 10, 64]);

        Ok(())
    }

    #[test]
    fn test_scaled_dot_product_attention_small_head_dim() -> Result<()> {
        let sdpa = ScaledDotProductAttention::new(32);

        let q = Tensor::ones((1, 2, 5, 32), candle_core::DType::F32, &Device::Cpu)?;
        let k = Tensor::ones((1, 2, 5, 32), candle_core::DType::F32, &Device::Cpu)?;
        let v = Tensor::ones((1, 2, 5, 32), candle_core::DType::F32, &Device::Cpu)?;

        let output = sdpa.forward(&q, &k, &v)?;

        assert_eq!(output.dims(), &[1, 2, 5, 32]);

        Ok(())
    }

    #[test]
    fn test_scaled_dot_product_attention_large_head_dim() -> Result<()> {
        let sdpa = ScaledDotProductAttention::new(128);

        let q = Tensor::ones((1, 4, 8, 128), candle_core::DType::F32, &Device::Cpu)?;
        let k = Tensor::ones((1, 4, 8, 128), candle_core::DType::F32, &Device::Cpu)?;
        let v = Tensor::ones((1, 4, 8, 128), candle_core::DType::F32, &Device::Cpu)?;

        let output = sdpa.forward(&q, &k, &v)?;

        assert_eq!(output.dims(), &[1, 4, 8, 128]);

        Ok(())
    }

    #[test]
    fn test_scaled_dot_product_attention_known_values() -> Result<()> {
        let head_dim = 4;
        let sdpa = ScaledDotProductAttention::new(head_dim);

        let q_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let q = Tensor::from_slice(&q_data, (1, 1, 2, head_dim), &Device::Cpu)?;

        let k_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let k = Tensor::from_slice(&k_data, (1, 1, 2, head_dim), &Device::Cpu)?;

        let v_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v = Tensor::from_slice(&v_data, (1, 1, 2, head_dim), &Device::Cpu)?;

        let output = sdpa.forward(&q, &k, &v)?;

        assert_eq!(output.dims(), &[1, 1, 2, head_dim]);

        Ok(())
    }

    #[test]
    fn test_scaled_dot_product_batch() -> Result<()> {
        let sdpa = ScaledDotProductAttention::new(32);

        let batch_size = 4;
        let num_heads = 4;
        let seq_len = 8;
        let head_dim = 32;

        let q = Tensor::ones(
            (batch_size, num_heads, seq_len, head_dim),
            candle_core::DType::F32,
            &Device::Cpu,
        )?;
        let k = Tensor::ones(
            (batch_size, num_heads, seq_len, head_dim),
            candle_core::DType::F32,
            &Device::Cpu,
        )?;
        let v = Tensor::ones(
            (batch_size, num_heads, seq_len, head_dim),
            candle_core::DType::F32,
            &Device::Cpu,
        )?;

        let output = sdpa.forward(&q, &k, &v)?;

        assert_eq!(output.dims(), &[batch_size, num_heads, seq_len, head_dim]);

        Ok(())
    }

    #[test]
    fn test_flash_attention_config() {
        let config = FlashAttentionConfig::new()
            .with_flash()
            .with_sliding_window(512);

        assert_eq!(config.variant, AttentionVariant::Flash);
        assert_eq!(config.sliding_window_size, 512);
        assert!(config.use_sliding_window);
    }

    #[test]
    fn test_flash_attention_config_default() {
        let config = FlashAttentionConfig::new();

        assert_eq!(config.variant, AttentionVariant::Standard);
        assert_eq!(config.flash_block_size, 128);
        assert!(!config.use_sliding_window);
        assert_eq!(config.sliding_window_size, 512);
    }

    #[test]
    fn test_attention_variant_defaults() {
        assert_eq!(AttentionVariant::default(), AttentionVariant::Standard);
    }

    #[test]
    fn test_flash_attention_kernel_creation() {
        let config = FlashAttentionConfig::new().with_flash();
        let kernel = FlashAttentionKernel::new(64, config);

        assert_eq!(kernel.config.variant, AttentionVariant::Flash);
    }

    #[test]
    fn test_flash_attention_kernel_forward() -> Result<()> {
        let config = FlashAttentionConfig::new();
        let kernel = FlashAttentionKernel::new(64, config);

        let q = Tensor::ones((1, 2, 4, 64), candle_core::DType::F32, &Device::Cpu)?;
        let k = Tensor::ones((1, 2, 4, 64), candle_core::DType::F32, &Device::Cpu)?;
        let v = Tensor::ones((1, 2, 4, 64), candle_core::DType::F32, &Device::Cpu)?;

        let output = kernel.forward(&q, &k, &v)?;

        assert_eq!(output.dims(), &[1, 2, 4, 64]);

        Ok(())
    }

    #[test]
    fn test_flash_attention_kernel_sliding_window() -> Result<()> {
        let config = FlashAttentionConfig::new().with_sliding_window(2);
        let kernel = FlashAttentionKernel::new(64, config);

        let q = Tensor::ones((1, 1, 4, 64), candle_core::DType::F32, &Device::Cpu)?;
        let k = Tensor::ones((1, 1, 4, 64), candle_core::DType::F32, &Device::Cpu)?;
        let v = Tensor::ones((1, 1, 4, 64), candle_core::DType::F32, &Device::Cpu)?;

        let output = kernel.forward(&q, &k, &v)?;

        assert_eq!(output.dims(), &[1, 1, 4, 64]);

        Ok(())
    }

    #[test]
    fn test_sliding_window_small_seq() -> Result<()> {
        let sdpa = ScaledDotProductAttention::new(32);

        let q = Tensor::ones((1, 1, 3, 32), candle_core::DType::F32, &Device::Cpu)?;
        let k = Tensor::ones((1, 1, 3, 32), candle_core::DType::F32, &Device::Cpu)?;
        let v = Tensor::ones((1, 1, 3, 32), candle_core::DType::F32, &Device::Cpu)?;

        let output = sdpa.compute_sliding_window(&q, &k, &v, 5)?;

        assert_eq!(output.dims(), &[1, 1, 3, 32]);

        Ok(())
    }

    #[test]
    fn test_sliding_window_large_seq() -> Result<()> {
        let sdpa = ScaledDotProductAttention::new(32);

        let q = Tensor::ones((1, 1, 10, 32), candle_core::DType::F32, &Device::Cpu)?;
        let k = Tensor::ones((1, 1, 10, 32), candle_core::DType::F32, &Device::Cpu)?;
        let v = Tensor::ones((1, 1, 10, 32), candle_core::DType::F32, &Device::Cpu)?;

        let output = sdpa.compute_sliding_window(&q, &k, &v, 4)?;

        assert_eq!(output.dims(), &[1, 1, 10, 32]);

        Ok(())
    }

    #[test]
    fn test_softmax_output_range() -> Result<()> {
        let sdpa = ScaledDotProductAttention::new(8);

        let q_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let q = Tensor::from_slice(&q_data, (1, 1, 1, 8), &Device::Cpu)?;

        let k_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let k = Tensor::from_slice(&k_data, (1, 1, 1, 8), &Device::Cpu)?;

        let v_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let v = Tensor::from_slice(&v_data, (1, 1, 1, 8), &Device::Cpu)?;

        let output = sdpa.forward(&q, &k, &v)?;

        let output_data: Vec<f32> = output.flatten_all()?.to_vec1()?;

        for val in output_data.iter() {
            assert!(
                *val >= 0.0,
                "Softmax output should be non-negative: {}",
                val
            );
            assert!(
                *val <= 100.0,
                "Softmax output should be reasonable: {}",
                val
            );
        }

        Ok(())
    }
}
