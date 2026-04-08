use candle_core::{Device, Result, Tensor};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

use crate::config::Architecture;
use crate::config::ModelConfig;
use crate::gemma4::Gemma4Model;
use crate::llama::LlamaModel;
use crate::mistral::MistralModel;
use crate::qwen3::model::Qwen3Model;
use crate::qwen3_config::Qwen3Config;

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
        "qwen2" | "qwen2.5" => Architecture::Qwen3,
        "gemma2" | "gemma3" | "gemma4" => Architecture::Gemma4,
        _ => Architecture::Llama,
    }
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

    pub fn load(
        &self,
        model_dir: &str,
        num_kv_blocks: usize,
    ) -> Result<Box<dyn vllm_traits::ModelBackend>> {
        let config_path = Path::new(model_dir).join("config.json");
        let content = std::fs::read_to_string(config_path)
            .map_err(|e| candle_core::Error::msg(format!("Failed to read config: {}", e)))?;
        let value: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| candle_core::Error::msg(format!("Failed to parse config: {}", e)))?;

        let config = ModelConfig::from_config_json(&value)
            .map_err(|e| candle_core::Error::msg(format!("Failed to parse model config: {}", e)))?;
        let weights = self.load_weights(model_dir)?;

        match config.architecture {
            Architecture::Llama => {
                let model =
                    LlamaModel::from_weights(config, self.device.clone(), weights, num_kv_blocks)?;
                Ok(Box::new(model))
            }
            Architecture::Mistral => {
                let model = MistralModel::from_weights(
                    config,
                    self.device.clone(),
                    weights,
                    num_kv_blocks,
                )?;
                Ok(Box::new(model))
            }
            Architecture::Mixtral => Err(candle_core::Error::msg(
                "Mixtral model loading not yet implemented",
            )),
            Architecture::Qwen3 => {
                let config = self.load_config(model_dir)?;
                let model = Qwen3Model::from_weights(
                    config,
                    self.device.clone(),
                    weights,
                    num_kv_blocks,
                    false,
                )?;
                Ok(Box::new(model))
            }
            Architecture::Gemma4 => {
                let model =
                    Gemma4Model::from_weights(config, self.device.clone(), weights, num_kv_blocks)?;
                Ok(Box::new(model))
            }
        }
    }

    pub fn load_model(
        &self,
        model_dir: &str,
        num_kv_blocks: usize,
        kv_quantization: bool,
    ) -> Result<Qwen3Model> {
        let config = self.load_config(model_dir)?;
        let weights = self.load_weights(model_dir)?;

        Qwen3Model::from_weights(
            config,
            self.device.clone(),
            weights,
            num_kv_blocks,
            kv_quantization,
        )
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

    #[test]
    #[ignore = "Requires model files at /models/Qwen2.5-0.5B-Instruct, ~60s runtime"]
    fn test_load_qwen2_model_integration() {
        // This test requires the model to exist at /models/Qwen2.5-0.5B-Instruct
        // Skip if model doesn't exist
        let model_path = "/models/Qwen2.5-0.5B-Instruct";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Model not found at {}, skipping test", model_path);
            return;
        }

        let loader = ModelLoader::new(Device::Cpu);

        // Test config loading
        let config = loader
            .load_config(model_path)
            .expect("Failed to load config");
        eprintln!(
            "Loaded Qwen3 config: hidden_size={}, num_layers={}, num_heads={}",
            config.hidden_size(),
            config.num_hidden_layers(),
            config.num_attention_heads()
        );

        // Test weight loading
        let weights = loader
            .load_weights(model_path)
            .expect("Failed to load weights");
        eprintln!("Loaded {} weight tensors", weights.len());

        // Verify some key weights exist
        assert!(weights.contains_key("model.embed_tokens.weight"));

        // Test full model loading
        let _model = loader
            .load_model(model_path, 128, false)
            .expect("Failed to create model");
        eprintln!("Model created successfully!");
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
}
