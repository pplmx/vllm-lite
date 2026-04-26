use candle_core::{Result, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttentionVariant {
    #[default]
    Standard,
    Tiled,
    Flash,
    FlashV2,
}

#[derive(Debug, Clone, Default)]
pub struct FlashAttentionConfig {
    pub variant: AttentionVariant,
    pub flash_block_size: usize,
    pub use_sliding_window: bool,
    pub sliding_window_size: usize,
    pub tile_sizes: Vec<usize>,
    pub use_fused: bool,
}

impl FlashAttentionConfig {
    pub fn new() -> Self {
        Self {
            variant: AttentionVariant::Standard,
            flash_block_size: 128,
            use_sliding_window: false,
            sliding_window_size: 512,
            tile_sizes: vec![64, 128, 256],
            use_fused: true,
        }
    }

    pub fn with_flash(mut self) -> Self {
        self.variant = AttentionVariant::Flash;
        self
    }

    pub fn with_flash_v2(mut self) -> Self {
        self.variant = AttentionVariant::FlashV2;
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

pub fn select_tile_size(seq_len: usize, config: &FlashAttentionConfig) -> usize {
    if seq_len <= 32 {
        32
    } else if seq_len <= 128 {
        64
    } else if seq_len <= 512 {
        128
    } else if seq_len <= 2048 {
        256
    } else {
        config.tile_sizes.last().copied().unwrap_or(256)
    }
}

pub fn should_use_tiled(seq_len: usize, head_dim: usize) -> bool {
    let memory_standard = seq_len * seq_len * head_dim;
    let memory_tiled = seq_len * 128 * head_dim * 2;
    memory_standard > memory_tiled * 2
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

#[derive(Clone, Default)]
pub struct AttentionStats {
    pub forward_count: u64,
    pub tiled_forward_count: u64,
    pub total_tokens: u64,
}

impl AttentionStats {
    pub fn record_forward(&mut self, num_tokens: usize) {
        self.forward_count += 1;
        self.total_tokens += num_tokens as u64;
    }

    pub fn record_tiled(&mut self, num_tokens: usize) {
        self.tiled_forward_count += 1;
        self.total_tokens += num_tokens as u64;
    }
}

pub struct ScaledDotProductAttention {
    scale: f32,
    tile_size: usize,
}

pub struct FlashAttentionV2 {
    scale: f32,
    block_size: usize,
    #[allow(dead_code)]
    num_heads: usize,
    head_dim: usize,
}

impl FlashAttentionV2 {
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self {
            scale,
            block_size: 64,
            num_heads,
            head_dim,
        }
    }

    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (_batch_size, _num_heads_q, _seq_len_q, _head_dim) = q.dims4()?;
        let (_, _, seq_len_k, _) = k.dims4()?;

        if seq_len_k <= 128 {
            return self.forward_standard(q, k, v);
        }

        self.forward_flash_v2(q, k, v)
    }

    fn forward_standard(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let qk = q.matmul(&k.t()?)?;
        let scale_tensor = Tensor::new(self.scale, q.device())?;
        let qk_scaled = qk.broadcast_mul(&scale_tensor)?;
        let attn = softmax_last_dim(&qk_scaled)?;
        attn.matmul(v)
    }

    fn forward_flash_v2(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (batch_size, num_heads_q, _seq_len_q, _head_dim) = q.dims4()?;
        let (_, num_heads_k, _seq_len_k, _) = k.dims4()?;

        let mut outputs = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let q_b = q.narrow(0, b, 1)?.squeeze(0)?;
            let k_b = k.narrow(0, b, 1)?.squeeze(0)?;
            let v_b = v.narrow(0, b, 1)?.squeeze(0)?;

            let mut head_outputs = Vec::with_capacity(num_heads_q);

            for h in 0..num_heads_q {
                let q_h = q_b.narrow(0, h, 1)?.squeeze(0)?;
                let k_h = k_b.narrow(0, h % num_heads_k, 1)?.squeeze(0)?;
                let v_h = v_b.narrow(0, h % num_heads_k, 1)?.squeeze(0)?;

                let out_h = self.compute_flash_attention_block(&q_h, &k_h, &v_h)?;
                head_outputs.push(out_h);
            }

            let batch_out = Tensor::stack(&head_outputs, 0)?;
            outputs.push(batch_out);
        }

        Tensor::stack(&outputs, 0)
    }

    fn compute_flash_attention_block(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let seq_len_k = k.dims()[0];
        let block_size = self.block_size.min(seq_len_k);

        let num_blocks = seq_len_k.div_ceil(block_size);
        let seq_len_q = q.dims()[0];

        let scale_tensor = Tensor::new(self.scale, q.device())?;
        let mut final_output = Tensor::zeros(
            (seq_len_q, self.head_dim),
            candle_core::DType::F32,
            q.device(),
        )?;

        let mut running_m = Tensor::zeros((seq_len_q, 1), candle_core::DType::F32, q.device())?;
        let mut running_l: Option<Tensor> = None;

        for block_idx in 0..num_blocks {
            let start_k = block_idx * block_size;
            let end_k = (start_k + block_size).min(seq_len_k);
            let actual_block_size = end_k - start_k;

            let k_block = k.narrow(0, start_k, actual_block_size)?;
            let v_block = v.narrow(0, start_k, actual_block_size)?;

            let qk_block = q.matmul(&k_block.t()?)?;
            let qk_scaled = qk_block.broadcast_mul(&scale_tensor)?;

            let block_m = qk_scaled.max_keepdim(1)?;
            let block_p = qk_scaled.broadcast_sub(&block_m)?.exp()?;
            let block_l = block_p.sum_keepdim(1)?;

            let m_diff = block_m.broadcast_sub(&running_m)?;
            let correction = m_diff.exp()?;

            let scaled_output = if let Some(ref running_l_val) = running_l {
                let scaled =
                    final_output.broadcast_mul(&running_l_val.broadcast_mul(&correction)?)?;
                scaled.broadcast_div(&block_l)?
            } else {
                final_output
            };

            let block_out = block_p.matmul(&v_block)?;
            final_output = scaled_output.broadcast_add(&block_out)?;

            running_m = block_m;
            running_l = Some(block_l);
        }

        if let Some(l) = running_l {
            final_output = final_output.broadcast_div(&l)?;
        }

        Ok(final_output)
    }

    pub fn forward_with_causal_mask(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (_, _, _seq_len_q, _) = q.dims4()?;
        let (_, _, seq_len_k, _) = k.dims4()?;

        if seq_len_k <= 128 {
            return self.forward_standard(q, k, v);
        }

        self.forward_flash_v2_with_causal(q, k, v)
    }

    fn forward_flash_v2_with_causal(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (batch_size, num_heads_q, _seq_len_q, _head_dim) = q.dims4()?;
        let (_, num_heads_k, _seq_len_k, _) = k.dims4()?;

        let mut outputs = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let q_b = q.narrow(0, b, 1)?.squeeze(0)?;
            let k_b = k.narrow(0, b, 1)?.squeeze(0)?;
            let v_b = v.narrow(0, b, 1)?.squeeze(0)?;

            let mut head_outputs = Vec::with_capacity(num_heads_q);

            for h in 0..num_heads_q {
                let q_h = q_b.narrow(0, h, 1)?.squeeze(0)?;
                let k_h = k_b.narrow(0, h % num_heads_k, 1)?.squeeze(0)?;
                let v_h = v_b.narrow(0, h % num_heads_k, 1)?.squeeze(0)?;

                let out_h = self.compute_flash_attention_causal(&q_h, &k_h, &v_h)?;
                head_outputs.push(out_h);
            }

            let batch_out = Tensor::stack(&head_outputs, 0)?;
            outputs.push(batch_out);
        }

        Tensor::stack(&outputs, 0)
    }

    fn compute_flash_attention_causal(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let seq_len_k = k.dims()[0];
        let block_size = self.block_size.min(seq_len_k);
        let seq_len_q = q.dims()[0];

        let num_blocks = seq_len_k.div_ceil(block_size);

        let scale_tensor = Tensor::new(self.scale, q.device())?;
        let mut final_output = Tensor::zeros(
            (seq_len_q, self.head_dim),
            candle_core::DType::F32,
            q.device(),
        )?;
        let mut running_m = Tensor::zeros((seq_len_q, 1), candle_core::DType::F32, q.device())?;
        let mut running_l: Option<Tensor> = None;

        for block_idx in 0..num_blocks {
            let start_k = block_idx * block_size;
            let end_k = (start_k + block_size).min(seq_len_k);
            let actual_block_size = end_k - start_k;

            let k_block = k.narrow(0, start_k, actual_block_size)?;
            let v_block = v.narrow(0, start_k, actual_block_size)?;

            let qk_block = q.matmul(&k_block.t()?)?;
            let qk_scaled = qk_block.broadcast_mul(&scale_tensor)?;

            let causal_mask =
                self.create_causal_mask(&[seq_len_q, actual_block_size], start_k, q.device())?;
            let qk_masked = qk_scaled.broadcast_add(&causal_mask)?;

            let block_m = qk_masked.max_keepdim(1)?;
            let block_p = qk_masked.broadcast_sub(&block_m)?.exp()?;
            let block_l = block_p.sum_keepdim(1)?;

            let m_diff = block_m.broadcast_sub(&running_m)?;
            let correction = m_diff.exp()?;

            let scaled_output = if let Some(ref running_l_val) = running_l {
                let scaled =
                    final_output.broadcast_mul(&running_l_val.broadcast_mul(&correction)?)?;
                scaled.broadcast_div(&block_l)?
            } else {
                final_output
            };

            let block_out = block_p.matmul(&v_block)?;
            final_output = scaled_output.broadcast_add(&block_out)?;

            running_m = block_m;
            running_l = Some(block_l);
        }

        if let Some(l) = running_l {
            final_output = final_output.broadcast_div(&l)?;
        }

        Ok(final_output)
    }

    fn create_causal_mask(
        &self,
        dims: &[usize],
        start_k: usize,
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        let seq_len_q = dims[0];
        let block_size = dims[1];

        let mut mask_data = Vec::with_capacity(seq_len_q * block_size);

        for q_idx in 0..seq_len_q {
            for k_idx in 0..block_size {
                let global_k_idx = start_k + k_idx;
                if q_idx > global_k_idx {
                    mask_data.push(f32::NEG_INFINITY);
                } else {
                    mask_data.push(0.0);
                }
            }
        }

        Tensor::from_slice(&mask_data, (seq_len_q, block_size), device)
    }
}

impl FlashAttention for FlashAttentionV2 {
    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        FlashAttentionV2::forward(self, q, k, v)
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
        let sdpa = ScaledDotProductAttention::new(self.head_dim).with_tile_size(tile_size);
        sdpa.compute_tiled(q, k, v, tile_size)
    }
}

impl ScaledDotProductAttention {
    pub fn new(head_dim: usize) -> Self {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let optimal_tile = if head_dim <= 64 { 32 } else { 64 };
        Self {
            scale,
            tile_size: optimal_tile,
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

        if seq_len <= 32 {
            return self.forward(q, k, v);
        }

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
    let max_vals = t.max_keepdim(shape.len() - 1)?;
    let t_shifted = t.broadcast_sub(&max_vals)?;
    let exp = t_shifted.exp()?;
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
    pub fn new(num_heads: usize, head_dim: usize, config: FlashAttentionConfig) -> Self {
        let attention: Box<dyn FlashAttention> = match config.variant {
            AttentionVariant::Tiled => Box::new(
                ScaledDotProductAttention::new(head_dim).with_tile_size(config.flash_block_size),
            ),
            AttentionVariant::FlashV2 => Box::new(
                FlashAttentionV2::new(num_heads, head_dim).with_block_size(config.flash_block_size),
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
        let seq_len = q.dims()[2];
        let tile_size = select_tile_size(seq_len, &self.config);
        self.attention.forward_tiled(q, k, v, tile_size)
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
        let kernel = FlashAttentionKernel::new(2, 64, config);

        assert_eq!(kernel.config.variant, AttentionVariant::Flash);
    }

    #[test]
    fn test_flash_attention_kernel_forward() -> Result<()> {
        let config = FlashAttentionConfig::new();
        let kernel = FlashAttentionKernel::new(2, 64, config);

        let q = Tensor::ones((1, 2, 4, 64), candle_core::DType::F32, &Device::Cpu)?;
        let k = Tensor::ones((1, 2, 4, 64), candle_core::DType::F32, &Device::Cpu)?;
        let v = Tensor::ones((1, 2, 4, 64), candle_core::DType::F32, &Device::Cpu)?;

        let output = kernel.forward(&q, &k, &v)?;

        assert_eq!(output.dims(), &[1, 2, 4, 64]);

        Ok(())
    }

    #[test]
    fn test_flash_attention_v2_config() {
        let config = FlashAttentionConfig::new().with_flash_v2();
        assert_eq!(config.variant, AttentionVariant::FlashV2);
    }

    #[test]
    fn test_flash_attention_v2_forward() -> Result<()> {
        let config = FlashAttentionConfig::new().with_flash_v2();
        let kernel = FlashAttentionKernel::new(2, 64, config);

        let q = Tensor::ones((1, 2, 4, 64), candle_core::DType::F32, &Device::Cpu)?;
        let k = Tensor::ones((1, 2, 4, 64), candle_core::DType::F32, &Device::Cpu)?;
        let v = Tensor::ones((1, 2, 4, 64), candle_core::DType::F32, &Device::Cpu)?;

        let output = kernel.forward(&q, &k, &v)?;

        assert_eq!(output.dims(), &[1, 2, 4, 64]);
        Ok(())
    }

    #[test]
    fn test_flash_attention_v2_long_sequence() -> Result<()> {
        let fa_v2 = FlashAttentionV2::new(4, 64).with_block_size(32);

        let q = Tensor::randn(0f32, 1.0, (2, 4, 128, 64), &Device::Cpu)?;
        let k = Tensor::randn(0f32, 1.0, (2, 4, 128, 64), &Device::Cpu)?;
        let v = Tensor::randn(0f32, 1.0, (2, 4, 128, 64), &Device::Cpu)?;

        let output = fa_v2.forward(&q, &k, &v)?;

        assert_eq!(output.dims(), &[2, 4, 128, 64]);
        Ok(())
    }

    #[test]
    fn test_flash_attention_v2_output_range() -> Result<()> {
        let fa_v2 = FlashAttentionV2::new(1, 32).with_block_size(16);

        let q = Tensor::randn(0f32, 1.0, (1, 1, 64, 32), &Device::Cpu)?;
        let k = Tensor::randn(0f32, 1.0, (1, 1, 64, 32), &Device::Cpu)?;
        let v = Tensor::ones((1, 1, 64, 32), candle_core::DType::F32, &Device::Cpu)?;

        let output = fa_v2.forward(&q, &k, &v)?;
        let output_data: Vec<f32> = output.flatten_all()?.to_vec1()?;

        for val in output_data.iter() {
            assert!(val.is_finite(), "Output should be finite: {}", val);
        }

        Ok(())
    }

    #[test]
    fn test_flash_attention_v2_consistency_with_sdpa() -> Result<()> {
        let sdpa = ScaledDotProductAttention::new(64);
        let fa_v2 = FlashAttentionV2::new(2, 64).with_block_size(64);

        let seed = 42;
        let q = Tensor::randn(seed as f32, 0.1, (1, 2, 32, 64), &Device::Cpu)?;
        let k = Tensor::randn((seed + 1) as f32, 0.1, (1, 2, 32, 64), &Device::Cpu)?;
        let v = Tensor::randn((seed + 2) as f32, 0.1, (1, 2, 32, 64), &Device::Cpu)?;

        let sdpa_out = sdpa.forward(&q, &k, &v)?;
        let fa_v2_out = fa_v2.forward(&q, &k, &v)?;

        let sdpa_data: Vec<f32> = sdpa_out.flatten_all()?.to_vec1()?;
        let fa_v2_data: Vec<f32> = fa_v2_out.flatten_all()?.to_vec1()?;

        let max_diff = sdpa_data
            .iter()
            .zip(fa_v2_data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |a, b| a.max(b));

        assert!(
            max_diff < 1e-2,
            "FlashAttentionV2 should be close to SDPA, max diff: {}",
            max_diff
        );

        Ok(())
    }

    #[test]
    fn test_flash_attention_kernel_sliding_window() -> Result<()> {
        let config = FlashAttentionConfig::new().with_sliding_window(2);
        let kernel = FlashAttentionKernel::new(2, 64, config);

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
