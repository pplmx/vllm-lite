//! Format loader trait and implementations for checkpoint loading.
//!
//! Provides a pluggable architecture for loading model weights from different formats.

use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;
use std::path::Path;

use crate::loader::io::find_safetensors_files;

/// Trait for loading model checkpoints from different file formats.
///
/// Implementors should provide methods to check if a file can be loaded
/// and to actually perform the loading.
pub trait FormatLoader {
    /// Check if this loader can handle the given file path.
    fn can_load(path: &Path) -> bool
    where
        Self: Sized;

    /// Load weights from the given path.
    fn load(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>>
    where
        Self: Sized;
}

/// Loader for Safetensors format files.
pub struct SafetensorsLoader;

impl SafetensorsLoader {
    pub fn can_load(path: &Path) -> bool {
        if path.exists() {
            if path.is_file() {
                path.extension()
                    .map(|ext| ext == "safetensors")
                    .unwrap_or(false)
            } else if path.is_dir() {
                find_safetensors_files(path).is_ok()
            } else {
                false
            }
        } else {
            path.extension()
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        }
    }
}

impl FormatLoader for SafetensorsLoader {
    fn can_load(path: &Path) -> bool {
        Self::can_load(path)
    }

    fn load(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>>
    where
        Self: Sized,
    {
        crate::loader::checkpoint::load_safetensors(path, device)
    }
}

#[cfg(feature = "gguf")]
mod gguf_loader {
    use super::*;
    use crate::quantize::gguf::load_gguf_tensors;

    pub struct GgufLoader;

    impl FormatLoader for GgufLoader {
        fn can_load(path: &Path) -> bool
        where
            Self: Sized,
        {
            path.extension().map(|ext| ext == "gguf").unwrap_or(false)
        }

        fn load(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>>
        where
            Self: Sized,
        {
            let storage_tensors = load_gguf_tensors(path, device)?;
            let mut tensors = HashMap::new();
            for (name, storage) in storage_tensors {
                let tensor = match storage {
                    crate::quantize::StorageTensor::Quantized(q) => q.dequantize_to_f32()?,
                    crate::quantize::StorageTensor::Fp16(t) => {
                        t.to_dtype(candle_core::DType::F32)?
                    }
                    crate::quantize::StorageTensor::Fp32(t) => t,
                };
                tensors.insert(name, tensor);
            }
            Ok(tensors)
        }
    }
}

#[cfg(feature = "gguf")]
pub use gguf_loader::GgufLoader;

#[cfg(test)]
mod tests {
    use super::SafetensorsLoader;
    use std::fs;
    use std::path::Path;
    use tempfile::TempDir;

    #[test]
    fn test_can_load_safetensors_file() {
        let path = Path::new("model.safetensors");
        assert!(SafetensorsLoader::can_load(path));
    }

    #[test]
    fn test_cannot_load_bin_file() {
        let path = Path::new("model.bin");
        assert!(!SafetensorsLoader::can_load(path));
    }

    #[test]
    fn test_cannot_load_unknown_extension() {
        let path = Path::new("model.unknown");
        assert!(!SafetensorsLoader::can_load(path));
    }

    #[test]
    fn test_can_load_safetensors_directory() {
        let temp_dir = TempDir::new().unwrap();
        let model_dir = temp_dir.path();

        fs::write(model_dir.join("model.safetensors"), b"test").unwrap();

        assert!(SafetensorsLoader::can_load(model_dir));
    }

    #[test]
    fn test_cannot_load_empty_directory() {
        let temp_dir = TempDir::new().unwrap();
        let model_dir = temp_dir.path();

        assert!(!SafetensorsLoader::can_load(model_dir));
    }
}
