use candle_core::{Device, Result, Tensor};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

use crate::config::Qwen3Config;

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
        let model_path = Path::new(model_dir).join("model.safetensors");
        let data = std::fs::read(model_path)
            .map_err(|e| candle_core::Error::msg(format!("Failed to read safetensors: {}", e)))?;
        let file = SafeTensors::deserialize(&data)
            .map_err(|e| candle_core::Error::msg(format!("Failed to load safetensors: {}", e)))?;

        let mut weights: HashMap<String, Tensor> = HashMap::new();
        for (name, view) in file.tensors() {
            let tensor_data: &[u8] = view.data();
            let shape = view.shape().to_vec();
            let n = tensor_data.len() / 4;
            let data_f32 =
                unsafe { std::slice::from_raw_parts(tensor_data.as_ptr() as *const f32, n) };
            let tensor = candle_core::Tensor::from_slice(data_f32, shape, &self.device)
                .map_err(|e| candle_core::Error::msg(format!("Failed to create tensor: {}", e)))?;
            weights.insert(name.clone(), tensor);
        }
        Ok(weights)
    }

    pub fn load(&self, model_dir: &str) -> Result<(Qwen3Config, HashMap<String, Tensor>)> {
        let config = self.load_config(model_dir)?;
        let weights = self.load_weights(model_dir)?;
        Ok((config, weights))
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
