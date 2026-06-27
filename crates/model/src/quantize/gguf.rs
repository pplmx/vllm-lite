//! gguf: gguf.

use crate::quantize::StorageTensor;
use candle_core::{Device, Result};
use std::collections::HashMap;
use std::path::Path;

/// load_gguf_tensors: load gguf tensors.
pub fn load_gguf_tensors(_path: &Path, _device: &Device) -> Result<HashMap<String, StorageTensor>> {
    // Placeholder: return empty for now
    Ok(HashMap::new())
}

/// is_gguf_file: is gguf file.
pub fn is_gguf_file(path: &Path) -> bool {
    path.extension().map(|ext| ext == "gguf").unwrap_or(false)
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
