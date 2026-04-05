#[deprecated(since = "0.2.0", note = "Use paged_tensor module instead")]
pub use crate::paged_tensor::{
    dequantize, quantization::QuantizedTensor, quantize, tensor_store::PagedKvCache,
};
