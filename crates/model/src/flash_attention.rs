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
}

pub struct ScaledDotProductAttention {
    scale: f32,
}

impl ScaledDotProductAttention {
    pub fn new(head_dim: usize) -> Self {
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self { scale }
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
}

pub struct FlashAttentionKernel {
    attention: Box<dyn FlashAttention>,
    config: FlashAttentionConfig,
}

impl FlashAttentionKernel {
    pub fn new(head_dim: usize, config: FlashAttentionConfig) -> Self {
        let attention: Box<dyn FlashAttention> = match config.variant {
            AttentionVariant::Flash | AttentionVariant::Tiled => {
                Box::new(ScaledDotProductAttention::new(head_dim))
            }
            AttentionVariant::Standard => Box::new(ScaledDotProductAttention::new(head_dim)),
        };

        Self { attention, config }
    }

    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        if self.config.use_sliding_window && self.config.sliding_window_size > 0 {
            self.forward_sliding_window(q, k, v)
        } else {
            self.attention.forward(q, k, v)
        }
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
    fn test_flash_attention_config() {
        let config = FlashAttentionConfig::new()
            .with_flash()
            .with_sliding_window(512);

        assert_eq!(config.variant, AttentionVariant::Flash);
        assert_eq!(config.sliding_window_size, 512);
        assert!(config.use_sliding_window);
    }
}
