use crate::quantize::StorageTensor;
use candle_core::{Device, Result};
use std::collections::HashMap;
use std::path::Path;

///
/// Stub loader — returns an empty tensor map. A full GGUF parser
/// (`Q4_K_M` / `Q5_K` / `Q8_0` quantization types, tensor + metadata parsing,
/// integration with `StorageTensor`) is future work; see the ADR-009
/// orphan-module decision and the v22.0 GGUF-01 deferred-items entry.
/// Callers that receive an empty map fall back to an empty tensor set
/// (no weights), which is the documented "no-op" behavior used by the
/// feature-gated `GgufLoader::load` path.
pub(crate) fn load_gguf_tensors(
    _path: &Path,
    _device: &Device,
) -> Result<HashMap<String, StorageTensor>> {
    Ok(HashMap::new())
}

pub(crate) fn is_gguf_file(path: &Path) -> bool {
    path.extension().is_some_and(|ext| ext == "gguf")
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
