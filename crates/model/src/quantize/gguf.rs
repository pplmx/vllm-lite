// Reserved public-API surface for the GGUF integration; the stub
// loader returns an empty tensor map today. `is_gguf_file` is the entry
// point the model loader uses to detect GGUF checkpoints.
#![allow(dead_code)]

use crate::quantize::StorageTensor;
use candle_core::Device;
use std::collections::HashMap;
use std::path::Path;

/// Detect whether a file path points at a GGUF checkpoint, based on
/// the `.gguf` extension. Cheap, sync, allocation-free; safe to call
/// as part of the format detection chain in
/// [`crate::loader::format::load_checkpoint`].
pub(crate) fn is_gguf_file(path: &Path) -> bool {
    path.extension().is_some_and(|ext| ext == "gguf")
}

/// Stub loader for GGUF checkpoints — returns an empty tensor map today.
/// A full GGUF parser (Q4_K_M, Q5_K, Q8_0 quantization types, tensor +
/// metadata parsing, integration with `StorageTensor`) is future work;
/// see the ADR-009 orphan-module decision and the v22.0 GGUF-01
/// deferred-items entry.
///
/// Callers that receive an empty map fall back to an empty tensor set
/// (no weights), which is the documented "no-op" behavior used by the
/// feature-gated `GgufLoader::load` path.
pub(crate) fn load_gguf_tensors(_path: &Path, _device: &Device) -> HashMap<String, StorageTensor> {
    HashMap::new()
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
}
