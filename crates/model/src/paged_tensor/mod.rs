//! Paged KV-cache abstraction: physical tensor storage (`tensor_store`) + quantization helpers (`quant`, `quantization`).
//!
//! Owns the layout, block pool, and on-disk format conversion. Used by
//! every attention layer that needs to read / write KV through
//! `PagedKvCache`.
pub mod quant;
pub mod quantization;
pub mod tensor_store;

pub use quant::{
    AWQQuantization, GPTQQuantization, QuantizationConfig, QuantizationType, QuantizedWeights,
};
pub use quantization::{QuantizedTensor, dequantize, quantize};
pub use tensor_store::PagedKvCache;
