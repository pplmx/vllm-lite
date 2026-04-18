//! Format loader trait and implementations for checkpoint loading.
//!
//! Provides a pluggable architecture for loading model weights from different formats.

use candle_core::{Device, Result, Tensor};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

use half::{bf16, f16};

const MMAP_THRESHOLD_BYTES: u64 = 100 * 1024 * 1024;
const MAX_MMAP_SIZE: u64 = 10 * 1024 * 1024 * 1024;

/// Trait for loading model checkpoints from different file formats.
///
/// Implementors should provide methods to check if a file can be loaded
/// and to actually perform the loading.
pub trait FormatLoader {
    /// Check if this loader can handle the given file path.
    ///
    /// # Arguments
    /// * `path` - The path to the file to check
    ///
    /// # Returns
    /// `true` if this loader can handle the file, `false` otherwise
    fn can_load(path: &Path) -> bool
    where
        Self: Sized;

    /// Load weights from the given path.
    ///
    /// # Arguments
    /// * `path` - The path to load weights from (can be a directory or file)
    /// * `device` - The device to load tensors onto
    ///
    /// # Returns
    /// A map of tensor names to tensors
    fn load(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>>
    where
        Self: Sized;
}

/// Loader for Safetensors format files.
pub struct SafetensorsLoader;

impl SafetensorsLoader {
    /// Load a file using mmap for large files.
    fn load_file(path: &Path) -> Result<Vec<u8>> {
        let metadata = std::fs::metadata(path)?;
        let file_size = metadata.len();

        if (MMAP_THRESHOLD_BYTES..=MAX_MMAP_SIZE).contains(&file_size) {
            match Self::load_mmap(path) {
                Ok(mmap) => {
                    return Ok(mmap.to_vec());
                }
                Err(e) => {
                    eprintln!(
                        "mmap failed for {}, falling back to read(): {}",
                        path.display(),
                        e
                    );
                }
            }
        }

        std::fs::read(path).map_err(|e| candle_core::Error::msg(format!("read failed: {}", e)))
    }

    fn load_mmap(path: &Path) -> Result<Mmap> {
        let file = std::fs::File::open(path)?;
        unsafe { Mmap::map(&file) }
            .map_err(|e| candle_core::Error::msg(format!("mmap failed: {}", e)))
    }

    /// Find all safetensors files in the given directory.
    fn find_safetensors_files(model_dir: &Path) -> Result<Vec<std::path::PathBuf>> {
        let single = model_dir.join("model.safetensors");
        if single.exists() {
            return Ok(vec![single]);
        }

        let mut files = Vec::new();
        let entries = std::fs::read_dir(model_dir).map_err(|e| {
            candle_core::Error::msg(format!("Failed to read model directory: {}", e))
        })?;

        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if (name.starts_with("model-") || name.starts_with("model.safetensors-"))
                    && name.ends_with(".safetensors")
                {
                    files.push(path);
                }
            }
        }

        if files.is_empty() {
            return Err(candle_core::Error::msg(format!(
                "No model weights found in {}",
                model_dir.display()
            )));
        }

        files.sort();
        Ok(files)
    }

    /// Convert a safetensors tensor view to a Candle tensor.
    fn convert_tensor(view: &safetensors::tensor::TensorView, device: &Device) -> Result<Tensor> {
        use safetensors::Dtype;

        let tensor_data: &[u8] = view.data();
        let shape = view.shape().to_vec();
        let dtype = view.dtype();

        match dtype {
            Dtype::BF16 => {
                let n = tensor_data.len() / 2;
                let data_bf16: &[u16] =
                    unsafe { std::slice::from_raw_parts(tensor_data.as_ptr() as *const u16, n) };
                let data_f32: Vec<f32> = data_bf16
                    .iter()
                    .map(|&bits| bf16::from_bits(bits).to_f32())
                    .collect();
                candle_core::Tensor::from_slice(&data_f32, shape, device)
            }
            Dtype::F16 => {
                let n = tensor_data.len() / 2;
                let data_f16: &[u16] =
                    unsafe { std::slice::from_raw_parts(tensor_data.as_ptr() as *const u16, n) };
                let data_f32: Vec<f32> = data_f16
                    .iter()
                    .map(|&bits| f16::from_bits(bits).to_f32())
                    .collect();
                candle_core::Tensor::from_slice(&data_f32, shape, device)
            }
            Dtype::F32 => {
                let n = tensor_data.len() / 4;
                let data_f32: &[f32] =
                    unsafe { std::slice::from_raw_parts(tensor_data.as_ptr() as *const f32, n) };
                candle_core::Tensor::from_slice(data_f32, shape, device)
            }
            _ => Err(candle_core::Error::msg(format!(
                "Unsupported dtype {:?} for weight",
                dtype
            ))),
        }
        .map_err(|e| candle_core::Error::msg(format!("Failed to create tensor: {}", e)))
    }
}

impl FormatLoader for SafetensorsLoader {
    fn can_load(path: &Path) -> bool
    where
        Self: Sized,
    {
        if path.exists() {
            if path.is_file() {
                path.extension()
                    .map(|ext| ext == "safetensors")
                    .unwrap_or(false)
            } else if path.is_dir() {
                // Check if directory contains safetensors files
                Self::find_safetensors_files(path).is_ok()
            } else {
                false
            }
        } else {
            // For non-existent paths, check by extension pattern
            path.extension()
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        }
    }

    fn load(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>>
    where
        Self: Sized,
    {
        use rayon::prelude::*;

        let files = if path.is_file() {
            vec![path.to_path_buf()]
        } else {
            Self::find_safetensors_files(path)?
        };

        let mmap_results: Vec<Result<(std::path::PathBuf, Vec<u8>)>> = files
            .par_iter()
            .map(|path| {
                let data = Self::load_file(path)?;
                Ok((path.clone(), data))
            })
            .collect();

        let tensor_vec: Vec<Result<Vec<(String, Tensor)>>> = mmap_results
            .into_par_iter()
            .map(|result| {
                let (_path, data) = result?;
                let file = SafeTensors::deserialize(&data)
                    .map_err(|e| candle_core::Error::msg(format!("deserialize failed: {}", e)))?;
                file.tensors()
                    .into_iter()
                    .filter(|(name, _)| {
                        !name.contains("visual.")
                            && !name.contains("vision_")
                            && !name.contains("img_")
                    })
                    .map(|(name, view)| {
                        let tensor = Self::convert_tensor(&view, device)?;
                        Ok((name, tensor))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect();

        let total: usize = tensor_vec
            .iter()
            .filter_map(|r| r.as_ref().ok())
            .map(|tensors| tensors.len())
            .sum();

        let mut weights = HashMap::new();
        let mut loaded = 0;

        for result in tensor_vec {
            let tensors = result?;
            for (name, tensor) in tensors {
                if weights.insert(name.clone(), tensor).is_some() {
                    return Err(candle_core::Error::msg(format!(
                        "Duplicate weight '{}'",
                        name
                    )));
                }
                loaded += 1;
                if loaded % 20 == 0 {
                    eprintln!("Loading: {}/{}", loaded, total);
                }
            }
        }

        eprintln!("Loaded {} tensors total", loaded);
        Ok(weights)
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

/// Load a checkpoint from the given path using the appropriate loader.
///
/// # Arguments
/// * `path` - Path to the checkpoint (file or directory)
/// * `device` - Device to load tensors onto
///
/// # Returns
/// A map of tensor names to tensors
///
/// # Errors
/// Returns an error if no loader can handle the path or if loading fails
pub fn load_checkpoint(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    // Try SafetensorsLoader first
    if SafetensorsLoader::can_load(path) {
        return SafetensorsLoader::load(path, device);
    }
    #[cfg(feature = "gguf")]
    {
        use gguf_loader::GgufLoader;
        if GgufLoader::can_load(path) {
            return GgufLoader::load(path, device);
        }
    }

    Err(candle_core::Error::msg(format!(
        "No loader available for path: {}",
        path.display()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
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

        // Create a safetensors file in the directory
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
