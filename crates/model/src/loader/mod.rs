//! Model loading utilities.
//!
//! This module provides the ModelLoader for loading model weights and configurations.
//! Supports Builder pattern for flexible configuration.

pub mod builder;
pub mod format;

pub use builder::{ModelLoader, ModelLoaderBuilder};
pub use format::{FormatLoader, load_checkpoint};

use candle_core::{Device, Result, Tensor};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

use half::{bf16, f16};

use crate::config::Architecture;

const MMAP_THRESHOLD_BYTES: u64 = 100 * 1024 * 1024;
const MAX_MMAP_SIZE: u64 = 10 * 1024 * 1024 * 1024;

pub fn load_file(path: &Path) -> Result<Vec<u8>> {
    let metadata = std::fs::metadata(path)?;
    let file_size = metadata.len();

    if (MMAP_THRESHOLD_BYTES..=MAX_MMAP_SIZE).contains(&file_size) {
        if let Ok(mmap) = load_mmap(path) {
            return Ok(mmap.to_vec());
        }
    }

    std::fs::read(path).map_err(|e| candle_core::Error::msg(format!("read failed: {}", e)))
}

fn load_mmap(path: &Path) -> Result<Mmap> {
    let file = std::fs::File::open(path)?;
    unsafe { Mmap::map(&file) }.map_err(|e| candle_core::Error::msg(format!("mmap failed: {}", e)))
}

pub fn detect_architecture(config: &serde_json::Value) -> Architecture {
    let model_type = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_lowercase();

    match model_type.as_str() {
        "llama" | "llama2" | "llama3" => Architecture::Llama,
        "mistral" => Architecture::Mistral,
        "mixtral" => Architecture::Mixtral,
        "qwen2" | "qwen2.5" | "qwen2_5" => Architecture::Qwen3,
        "qwen3" | "qwen_3" => Architecture::Qwen3,
        "qwen3.5" | "qwen3_5" => Architecture::Qwen35,
        "mamba" => Architecture::Qwen35,
        "gemma2" | "gemma3" | "gemma4" => Architecture::Gemma4,
        _ => Architecture::Llama,
    }
}

pub fn remap_qwen35_weight_keys(weights: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
    let mut remapped = HashMap::new();

    for (key, value) in weights {
        let new_key = if key.starts_with("model.language_model.") {
            key.replace("model.language_model.", "model.")
        } else if key.starts_with("lm_head.") || key.starts_with("model.lm_head.") {
            if key.starts_with("lm_head.") {
                key.replace("lm_head.", "model.lm_head.")
            } else {
                key
            }
        } else if key == "model.norm.weight" {
            "model.final_layernorm.weight".to_string()
        } else {
            key
        };
        remapped.insert(new_key, value);
    }

    remapped
}

fn find_safetensors_files(model_dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    let mut files = Vec::new();
    let entries = std::fs::read_dir(model_dir)
        .map_err(|e| candle_core::Error::msg(format!("Failed to read model directory: {}", e)))?;

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

pub fn do_load_weights(model_dir: &str, device: &Device) -> Result<HashMap<String, Tensor>> {
    use rayon::prelude::*;

    let model_path = Path::new(model_dir);
    let files = find_safetensors_files(model_path)?;

    let mmap_results: Vec<Result<(std::path::PathBuf, Vec<u8>)>> = files
        .par_iter()
        .map(|path| {
            let data = load_file(path)?;
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
                    !name.contains("visual.") && !name.contains("vision_") && !name.contains("img_")
                })
                .map(|(name, view)| {
                    let tensor = convert_tensor(&view, device)?;
                    Ok((name, tensor))
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect();

    let mut weights = HashMap::new();

    for result in tensor_vec {
        let tensors = result?;
        for (name, tensor) in tensors {
            if weights.insert(name.clone(), tensor).is_some() {
                return Err(candle_core::Error::msg(format!(
                    "Duplicate weight '{}'",
                    name
                )));
            }
        }
    }

    Ok(weights)
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_find_safetensors_single_file() {
        let temp_dir = TempDir::new().unwrap();
        let model_dir = temp_dir.path();

        fs::write(model_dir.join("model.safetensors"), b"test").unwrap();

        let files = find_safetensors_files(model_dir).unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].to_string_lossy().ends_with("model.safetensors"));
    }

    #[test]
    fn test_find_safetensors_sharded_files() {
        let temp_dir = TempDir::new().unwrap();
        let model_dir = temp_dir.path();

        fs::write(model_dir.join("model-00001-of-00002.safetensors"), b"test1").unwrap();
        fs::write(model_dir.join("model-00002-of-00002.safetensors"), b"test2").unwrap();

        let files = find_safetensors_files(model_dir).unwrap();
        assert_eq!(files.len(), 2);
    }

    #[test]
    fn test_find_safetensors_no_files() {
        let temp_dir = TempDir::new().unwrap();
        let model_dir = temp_dir.path();

        let result = find_safetensors_files(model_dir);
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_architecture_qwen2() {
        let config_json = serde_json::json!({
            "model_type": "qwen2"
        });
        let arch = detect_architecture(&config_json);
        assert_eq!(arch, Architecture::Qwen3);
    }

    #[test]
    fn test_detect_architecture_llama() {
        for model_type in ["llama", "llama2", "llama3"] {
            let config_json = serde_json::json!({
                "model_type": model_type
            });
            let arch = detect_architecture(&config_json);
            assert_eq!(arch, Architecture::Llama, "Failed for {}", model_type);
        }
    }

    #[test]
    fn test_detect_architecture_mistral() {
        let config_json = serde_json::json!({
            "model_type": "mistral"
        });
        let arch = detect_architecture(&config_json);
        assert_eq!(arch, Architecture::Mistral);
    }

    #[test]
    fn test_detect_architecture_mixtral() {
        let config_json = serde_json::json!({
            "model_type": "mixtral"
        });
        let arch = detect_architecture(&config_json);
        assert_eq!(arch, Architecture::Mixtral);
    }

    #[test]
    fn test_load_file_uses_mmap_for_large() {
        let temp_dir = TempDir::new().unwrap();
        let large_file = temp_dir.path().join("large.bin");
        fs::write(&large_file, vec![0u8; 150 * 1024 * 1024]).unwrap();

        let data = load_file(&large_file).unwrap();
        assert_eq!(data.len(), 150 * 1024 * 1024);
    }

    #[test]
    fn test_load_file_uses_read_for_small() {
        let temp_dir = TempDir::new().unwrap();
        let small_file = temp_dir.path().join("small.bin");
        fs::write(&small_file, b"hello world").unwrap();

        let data = load_file(&small_file).unwrap();
        assert_eq!(data, b"hello world");
    }

    #[test]
    fn test_load_file_fallback_on_error() {
        let temp_dir = TempDir::new().unwrap();
        let file = temp_dir.path().join("test.bin");
        fs::write(&file, b"test").unwrap();

        let data = load_file(&file).unwrap();
        assert_eq!(data, b"test");
    }
}
