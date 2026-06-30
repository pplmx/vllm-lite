//! Quantization utilities for model weights.

// `Checkpoint` and its conversion helpers are reserved for the in-flight
// Q4_K_M / Q5_K / Q8_0 GGUF integration. External callers reach the
// quantization surface via the `gguf` module; the wrapper struct is kept
// for the planned unified-checkpoint API.
#![allow(dead_code)]

#[cfg(feature = "gguf")]
pub mod gguf;
pub mod types;
pub use types::{QuantizationConfig, QuantizationFormat, QuantizedTensor, StorageTensor};

use candle_core::{Result, Tensor};
use std::collections::HashMap;

/// Checkpoint: checkpoint.
#[derive(Debug)]
pub(crate) struct Checkpoint {
    pub tensors: HashMap<String, StorageTensor>,
    pub quantization_config: Option<QuantizationConfig>,
}

impl Checkpoint {
    pub fn new(tensors: HashMap<String, Tensor>) -> Self {
        let storage_tensors = tensors
            .into_iter()
            .map(|(k, v)| (k, StorageTensor::Fp32(v)))
            .collect();
        Self {
            tensors: storage_tensors,
            quantization_config: None,
        }
    }

    /// `into_f16`: into f16.
    pub fn into_f16(self) -> Result<HashMap<String, Tensor>> {
        let mut result = HashMap::new();
        for (name, storage) in self.tensors {
            let tensor = match storage {
                StorageTensor::Quantized(q) => q.dequantize_to_f16()?,
                StorageTensor::Fp16(t) => t,
                StorageTensor::Fp32(t) => t.to_dtype(candle_core::DType::F16)?,
            };
            result.insert(name, tensor);
        }
        Ok(result)
    }

    pub fn into_raw(self) -> HashMap<String, StorageTensor> {
        self.tensors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_checkpoint_new() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "test".to_string(),
            Tensor::zeros(1, candle_core::DType::F32, &Device::Cpu).unwrap(),
        );
        let checkpoint = Checkpoint::new(tensors);
        assert!(checkpoint.quantization_config.is_none());
        assert_eq!(checkpoint.tensors.len(), 1);
    }
}
