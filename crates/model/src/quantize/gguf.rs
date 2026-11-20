//! GGUF format integration: file detection (`is_gguf_file`) and the loader
//! seam.
//!
//! This module is the on-disk-format adapter the model loader calls when a
//! `.gguf` file is detected. **The GGUF parser and `Q4_K_M` dequantization
//! are not yet implemented** (see ADR-009 and the v22.0 GGUF-01
//! deferred-items entry): `load_gguf_tensors` reports that clearly instead
//! of returning an empty weight map, and `QuantizedTensor::dequantize_to_f32`
//! is still a zero-filling placeholder.
//! `is_gguf_file` is the entry point the model loader uses to detect GGUF
//! checkpoints.
#![allow(dead_code)]

use crate::quantize::StorageTensor;
use candle_core::{Device, Result};
use std::collections::HashMap;
use std::path::Path;

/// Detect whether a file path points at a GGUF checkpoint, based on
/// the `.gguf` extension. Cheap, sync, allocation-free; safe to call
/// as part of the format detection chain in
/// [`crate::loader::format`].
pub(crate) fn is_gguf_file(path: &Path) -> bool {
    path.extension().is_some_and(|ext| ext == "gguf")
}

/// Load GGUF tensors — unimplemented, and reported as such.
///
/// A full GGUF parser (`Q4_K_M`, `Q5_K`, `Q8_0` quantization types, tensor +
/// metadata parsing, integration with `StorageTensor`) is future work; see
/// the ADR-009 orphan-module decision and the v22.0 GGUF-01 deferred-items
/// entry.
///
/// # Errors
///
/// Always returns `Err` today. Returning an empty map instead made an
/// unparsed checkpoint indistinguishable from a successfully loaded one — a
/// `.gguf` file, valid or corrupt, produced a model object with no weights
/// rather than failing at load time.
pub(crate) fn load_gguf_tensors(
    path: &Path,
    _device: &Device,
) -> Result<HashMap<String, StorageTensor>> {
    Err(candle_core::Error::msg(format!(
        "GGUF checkpoints are detected but not yet loadable in this build: \
         the Q4_K_M/Q5_K/Q8_0 parser is deferred (ADR-009), so `{}` cannot be \
         served. Convert the checkpoint to safetensors and load that instead.",
        path.display()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_gguf_file() {
        assert!(is_gguf_file(Path::new("model.gguf")));
        assert!(!is_gguf_file(Path::new("model.safetensors")));
        assert!(!is_gguf_file(Path::new("model.bin")));
    }

    #[test]
    fn test_load_gguf_tensors_reports_deferred_parser() {
        let err = load_gguf_tensors(Path::new("model.gguf"), &Device::Cpu)
            .expect_err("the deferred GGUF parser must refuse, not return an empty map");
        let message = err.to_string();
        assert!(message.contains("GGUF"), "got: {message}");
        assert!(message.contains("model.gguf"), "got: {message}");
    }
}
