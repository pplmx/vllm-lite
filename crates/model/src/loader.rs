use candle_core::{Device, Result, Tensor};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

use crate::config::Qwen3Config;
use crate::qwen3::model::Qwen3Model;

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

pub struct ModelLoader {
    device: Device,
}

impl ModelLoader {
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    pub fn load_config(&self, model_dir: &str) -> Result<Qwen3Config> {
        let config_path = Path::new(model_dir).join("config.json");
        let content = std::fs::read_to_string(config_path)
            .map_err(|e| candle_core::Error::msg(format!("Failed to read config: {}", e)))?;
        let config: Qwen3Config = serde_json::from_str(&content)
            .map_err(|e| candle_core::Error::msg(format!("Failed to parse config: {}", e)))?;
        Ok(config)
    }

    pub fn load_weights(&self, model_dir: &str) -> Result<HashMap<String, Tensor>> {
        let model_path = Path::new(model_dir);
        let files = find_safetensors_files(model_path)?;

        let mut weights: HashMap<String, Tensor> = HashMap::new();

        for file_path in files {
            let data = std::fs::read(&file_path).map_err(|e| {
                candle_core::Error::msg(format!("Failed to read {}: {}", file_path.display(), e))
            })?;
            let file = SafeTensors::deserialize(&data).map_err(|e| {
                candle_core::Error::msg(format!("Failed to load {}: {}", file_path.display(), e))
            })?;

            let mut loaded = 0;
            for (name, view) in file.tensors() {
                if name.contains("visual.") || name.contains("vision_") || name.contains("img_") {
                    continue;
                }
                loaded += 1;
                if loaded % 20 == 0 {
                    eprintln!("Loaded {}/? weights...", loaded);
                }

                let tensor_data: &[u8] = view.data();
                let shape = view.shape().to_vec();
                let dtype = view.dtype();

                let tensor = match dtype {
                    safetensors::Dtype::BF16 | safetensors::Dtype::F16 => {
                        let n = tensor_data.len() / 2;
                        let data_u16: &[u16] = unsafe {
                            std::slice::from_raw_parts(tensor_data.as_ptr() as *const u16, n)
                        };
                        let mut data_f32_vec = Vec::with_capacity(n);
                        for &bits in data_u16 {
                            let f32_val = half::f16::from_bits(bits).to_f32();
                            data_f32_vec.push(f32_val);
                        }
                        candle_core::Tensor::from_slice(&data_f32_vec, shape, &self.device)
                    }
                    safetensors::Dtype::F32 => {
                        let n = tensor_data.len() / 4;
                        let data_f32: &[f32] = unsafe {
                            std::slice::from_raw_parts(tensor_data.as_ptr() as *const f32, n)
                        };
                        candle_core::Tensor::from_slice(data_f32, shape, &self.device)
                    }
                    _ => {
                        return Err(candle_core::Error::msg(format!(
                            "Unsupported dtype {:?} for weight {}",
                            dtype, name
                        )));
                    }
                }
                .map_err(|e| {
                    candle_core::Error::msg(format!("Failed to create tensor for {}: {}", name, e))
                })?;
                if weights.insert(name.clone(), tensor).is_some() {
                    return Err(candle_core::Error::msg(format!(
                        "Duplicate weight '{}' found in sharded files",
                        name
                    )));
                }
            }
        }

        Ok(weights)
    }

    pub fn load(&self, model_dir: &str) -> Result<(Qwen3Config, HashMap<String, Tensor>)> {
        let config = self.load_config(model_dir)?;
        let weights = self.load_weights(model_dir)?;
        Ok((config, weights))
    }

    pub fn load_model(&self, model_dir: &str, num_kv_blocks: usize) -> Result<Qwen3Model> {
        let config = self.load_config(model_dir)?;
        let weights = self.load_weights(model_dir)?;

        Qwen3Model::from_weights(config, self.device.clone(), weights, num_kv_blocks)
            .map_err(|e| candle_core::Error::msg(format!("Failed to create model: {}", e)))
    }

    pub fn print_weight_keys(weights: &HashMap<String, Tensor>) {
        let mut keys: Vec<_> = weights.keys().collect();
        keys.sort();
        println!("Loaded weight keys ({} total):", keys.len());
        for key in keys.iter() {
            if let Some(t) = weights.get(*key) {
                let dims: Vec<usize> = (0..t.dims().len()).map(|i| t.dims()[i]).collect();
                println!("  {}: {:?}", key, dims);
            }
        }
    }
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
}
