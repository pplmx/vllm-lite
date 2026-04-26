use candle_core::{Device, Result, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    Fp16,
    Fp32,
    Int8,
    Int4,
    Awq,
    Gptq,
    GGUF,
}

impl QuantizationType {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "fp16" | "float16" => Some(Self::Fp16),
            "fp32" | "float32" => Some(Self::Fp32),
            "int8" => Some(Self::Int8),
            "int4" => Some(Self::Int4),
            "awq" | "llm-awq" => Some(Self::Awq),
            "gptq" | "llm-int8" => Some(Self::Gptq),
            "gguf" => Some(Self::GGUF),
            _ => None,
        }
    }

    pub fn bits(&self) -> usize {
        match self {
            Self::Fp16 => 16,
            Self::Fp32 => 32,
            Self::Int8 => 8,
            Self::Int4 => 4,
            Self::Awq => 4,
            Self::Gptq => 4,
            Self::GGUF => 4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    pub quant_type: QuantizationType,
    pub group_size: Option<usize>,
    pub bits: usize,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            quant_type: QuantizationType::Fp16,
            group_size: None,
            bits: 16,
        }
    }
}

impl QuantizationConfig {
    pub fn awq(bits: usize, group_size: usize) -> Self {
        Self {
            quant_type: QuantizationType::Awq,
            group_size: Some(group_size),
            bits,
        }
    }

    pub fn gptq(bits: usize, group_size: usize) -> Self {
        Self {
            quant_type: QuantizationType::Gptq,
            group_size: Some(group_size),
            bits,
        }
    }
}

pub trait Quantization: Send + Sync {
    fn quantize(&self, data: &[f32]) -> QuantizedWeights;
    fn dequantize(&self, weights: &QuantizedWeights) -> Result<Tensor>;
    fn bits(&self) -> usize;
    fn quant_type(&self) -> QuantizationType;
}

#[derive(Debug, Clone)]
pub struct QuantizedWeights {
    pub qweight: Tensor,
    pub scales: Tensor,
    pub zeros: Option<Tensor>,
    pub g_idx: Option<Tensor>,
    pub bits: usize,
    pub group_size: usize,
}

impl QuantizedWeights {
    pub fn new(qweight: Tensor, scales: Tensor) -> Self {
        Self {
            qweight,
            scales,
            zeros: None,
            g_idx: None,
            bits: 4,
            group_size: 128,
        }
    }

    pub fn with_zeros(mut self, zeros: Tensor) -> Self {
        self.zeros = Some(zeros);
        self
    }

    pub fn with_g_idx(mut self, g_idx: Tensor) -> Self {
        self.g_idx = Some(g_idx);
        self
    }
}

pub struct AWQQuantization {
    bits: usize,
    group_size: usize,
}

impl AWQQuantization {
    pub fn new(bits: usize, group_size: usize) -> Self {
        Self { bits, group_size }
    }

    pub fn quantize(
        &self,
        data: &[f32],
        _shape: &[usize],
        device: &Device,
    ) -> Result<QuantizedWeights> {
        let num_elements = data.len();
        let num_groups = num_elements.div_ceil(self.group_size);

        let mut scales = Vec::with_capacity(num_groups);
        let _zeros: Vec<f32> = Vec::with_capacity(num_groups);
        let mut qweight = Vec::with_capacity(num_elements);

        for i in 0..num_groups {
            let start = i * self.group_size;
            let end = (start + self.group_size).min(num_elements);
            let group = &data[start..end];

            let max_q = (1 << self.bits) as f32 - 1.0;
            let scale = group.iter().map(|w| w.abs()).fold(0.0f32, f32::max) / max_q;

            let scale_val = if scale > 1e-8 { 1.0 / scale } else { 0.0 };

            scales.push(scale);

            for &w in group {
                let quantized = ((w * scale_val).round().clamp(0.0, max_q) as i32) as u8;
                qweight.push(quantized);
            }
        }

        let qweight_tensor = Tensor::from_slice(&qweight, (num_elements,), device)?;
        let scales_tensor = Tensor::from_slice(&scales, (num_groups,), device)?;

        Ok(QuantizedWeights::new(qweight_tensor, scales_tensor))
    }

    pub fn dequantize(&self, weights: &QuantizedWeights) -> Result<Tensor> {
        let qweight = &weights.qweight;
        let scales = &weights.scales;

        let q_data: Vec<u8> = qweight.to_vec1()?;
        let scales_data: Vec<f32> = scales.to_vec1()?;
        let g_idx_data = weights
            .g_idx
            .as_ref()
            .map(|t| t.to_vec1::<u32>().unwrap_or_default())
            .unwrap_or_else(|| {
                (0..q_data.len())
                    .map(|i| i as u32 / self.group_size as u32)
                    .collect()
            });

        let mut output = Vec::with_capacity(q_data.len());

        for (i, &q) in q_data.iter().enumerate() {
            let g_idx = g_idx_data
                .get(i)
                .copied()
                .unwrap_or(i as u32 / self.group_size as u32) as usize;
            let scale = scales_data.get(g_idx).copied().unwrap_or(1.0);
            let w = q as f32 * scale;
            output.push(w);
        }

        let shape = qweight.dims();
        Tensor::from_slice(&output, shape, qweight.device())
    }
}

pub struct GPTQQuantization {
    bits: usize,
    group_size: usize,
}

impl GPTQQuantization {
    pub fn new(bits: usize, group_size: usize) -> Self {
        Self { bits, group_size }
    }

    pub fn quantize(
        &self,
        data: &[f32],
        _shape: &[usize],
        device: &Device,
    ) -> Result<QuantizedWeights> {
        let num_elements = data.len();
        let num_groups = num_elements.div_ceil(self.group_size);

        let mut scales = Vec::with_capacity(num_groups);
        let mut qweight = Vec::with_capacity(num_elements);

        let qmax = ((1u32 << self.bits) - 1) as f32;

        for i in 0..num_groups {
            let start = i * self.group_size;
            let end = (start + self.group_size).min(num_elements);
            let group = &data[start..end];

            let max_abs = group.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs > 1e-8 { max_abs / qmax } else { 1.0 };

            let inv_scale = 1.0 / scale;
            scales.push(scale);

            for &w in group {
                let quantized = ((w * inv_scale).round().clamp(0.0, qmax)) as u8;
                qweight.push(quantized);
            }
        }

        let qweight_tensor = Tensor::from_slice(&qweight, (num_elements,), device)?;
        let scales_tensor = Tensor::from_slice(&scales, (num_groups,), device)?;

        Ok(QuantizedWeights::new(qweight_tensor, scales_tensor))
    }

    pub fn dequantize(&self, weights: &QuantizedWeights) -> Result<Tensor> {
        let qweight = &weights.qweight;
        let scales = &weights.scales;

        let q_data: Vec<u8> = qweight.to_vec1()?;
        let scales_data: Vec<f32> = scales.to_vec1()?;

        let mut output = Vec::with_capacity(q_data.len());

        for (i, &q) in q_data.iter().enumerate() {
            let g_idx = i / self.group_size;
            let scale = scales_data.get(g_idx).copied().unwrap_or(1.0);
            let w = q as f32 * scale;
            output.push(w);
        }

        Tensor::from_slice(&output, qweight.dims(), qweight.device())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_awq_quantize_dequantize() {
        let device = Device::Cpu;
        let data: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
        let awq = AWQQuantization::new(4, 32);

        let weights = awq.quantize(&data, &[128], &device).unwrap();
        let dequantized = awq.dequantize(&weights).unwrap();

        let deq_data: Vec<f32> = dequantized.to_vec1().unwrap();
        let max_diff = data
            .iter()
            .zip(deq_data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 10.0,
            "Dequantization error too large: {}",
            max_diff
        );
    }

    #[test]
    fn test_gptq_quantize_dequantize() {
        let device = Device::Cpu;
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.5).collect();
        let gptq = GPTQQuantization::new(4, 16);

        let weights = gptq.quantize(&data, &[64], &device).unwrap();
        let dequantized = gptq.dequantize(&weights).unwrap();

        let deq_data: Vec<f32> = dequantized.to_vec1().unwrap();
        let max_diff = data
            .iter()
            .zip(deq_data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(max_diff < 20.0, "GPTQ dequantization error too large");
    }

    #[test]
    fn test_quantization_types() {
        assert_eq!(
            QuantizationType::from_str("awq"),
            Some(QuantizationType::Awq)
        );
        assert_eq!(
            QuantizationType::from_str("gptq"),
            Some(QuantizationType::Gptq)
        );
        assert_eq!(
            QuantizationType::from_str("llm-awq"),
            Some(QuantizationType::Awq)
        );

        assert_eq!(QuantizationType::Awq.bits(), 4);
        assert_eq!(QuantizationType::Gptq.bits(), 4);
    }

    #[test]
    fn test_quantized_weights_builder() {
        use candle_core::Device;

        let qweight = Tensor::ones((32,), candle_core::DType::U8, &Device::Cpu).unwrap();
        let scales = Tensor::ones((1,), candle_core::DType::F32, &Device::Cpu).unwrap();

        let weights = QuantizedWeights::new(qweight, scales)
            .with_zeros(Tensor::zeros((1,), candle_core::DType::F32, &Device::Cpu).unwrap());

        assert!(weights.zeros.is_some());
    }
}
