use candle_core::{Device, Result, Tensor};
use std::path::Path;
use std::sync::Arc;

use crate::config::Architecture;
use crate::config::ModelConfig;

pub struct ModelLoaderBuilder {
    device: Device,
    model_dir: Option<String>,
    num_kv_blocks: Option<usize>,
    kv_quantization: Option<bool>,
}

impl ModelLoaderBuilder {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            model_dir: None,
            num_kv_blocks: None,
            kv_quantization: None,
        }
    }

    pub fn with_model_dir(mut self, model_dir: String) -> Self {
        self.model_dir = Some(model_dir);
        self
    }

    pub fn with_kv_blocks(mut self, num_kv_blocks: usize) -> Self {
        self.num_kv_blocks = Some(num_kv_blocks);
        self
    }

    pub fn with_kv_quantization(mut self, enabled: bool) -> Self {
        self.kv_quantization = Some(enabled);
        self
    }

    pub fn build(self) -> Result<ModelLoader> {
        let model_dir = self
            .model_dir
            .ok_or_else(|| candle_core::Error::msg("model_dir is required"))?;
        let num_kv_blocks = self.num_kv_blocks.unwrap_or(1024);
        let kv_quantization = self.kv_quantization.unwrap_or(false);

        Ok(ModelLoader {
            inner: Arc::new(ModelLoaderInner::new(
                self.device,
                model_dir,
                num_kv_blocks,
                kv_quantization,
            )?),
        })
    }
}

pub struct ModelLoader {
    inner: Arc<ModelLoaderInner>,
}

impl Clone for ModelLoader {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

struct ModelLoaderInner {
    device: Device,
    model_dir: String,
    num_kv_blocks: usize,
    kv_quantization: bool,
    config_json: serde_json::Value,
    architecture: Architecture,
}

impl ModelLoaderInner {
    fn new(
        device: Device,
        model_dir: String,
        num_kv_blocks: usize,
        kv_quantization: bool,
    ) -> Result<Self> {
        let config_path = Path::new(&model_dir).join("config.json");
        let content = std::fs::read_to_string(&config_path)
            .map_err(|e| candle_core::Error::msg(format!("Failed to read config: {}", e)))?;
        let config_json: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| candle_core::Error::msg(format!("Failed to parse config: {}", e)))?;

        let architecture = crate::loader::detect_architecture(&config_json);
        eprintln!("Detected architecture: {:?}", architecture);

        Ok(Self {
            device,
            model_dir,
            num_kv_blocks,
            kv_quantization,
            config_json,
            architecture,
        })
    }
}

impl ModelLoader {
    pub fn new(device: Device) -> Self {
        Self {
            inner: Arc::new(ModelLoaderInner {
                device,
                model_dir: String::new(),
                num_kv_blocks: 1024,
                kv_quantization: false,
                config_json: serde_json::Value::Null,
                architecture: Architecture::Llama,
            }),
        }
    }

    pub fn builder(device: Device) -> ModelLoaderBuilder {
        ModelLoaderBuilder::new(device)
    }

    pub fn device(&self) -> &Device {
        &self.inner.device
    }

    pub fn architecture(&self) -> Architecture {
        self.inner.architecture
    }

    pub fn config_json(&self) -> &serde_json::Value {
        &self.inner.config_json
    }

    pub fn load_config(&self) -> Result<crate::qwen3_config::Qwen3Config> {
        let config_path = Path::new(&self.inner.model_dir).join("config.json");
        let content = std::fs::read_to_string(config_path)
            .map_err(|e| candle_core::Error::msg(format!("Failed to read config: {}", e)))?;
        let config: crate::qwen3_config::Qwen3Config = serde_json::from_str(&content)
            .map_err(|e| candle_core::Error::msg(format!("Failed to parse config: {}", e)))?;
        Ok(config)
    }

    pub fn load_weights(&self) -> Result<std::collections::HashMap<String, Tensor>> {
        // Use new unified loading
        let path = Path::new(&self.inner.model_dir);
        crate::loader::format::load_checkpoint(path, &self.inner.device)
    }

    pub fn load(&self) -> Result<Box<dyn vllm_traits::ModelBackend>> {
        let num_kv_blocks = self.inner.num_kv_blocks;

        let config = ModelConfig::from_config_json(&self.inner.config_json)
            .map_err(|e| candle_core::Error::msg(format!("Failed to parse model config: {}", e)))?;
        let mut weights = self.load_weights()?;

        match self.inner.architecture {
            Architecture::Llama => {
                let model = crate::llama::LlamaModel::from_weights(
                    config,
                    self.inner.device.clone(),
                    weights,
                    num_kv_blocks,
                )?;
                Ok(Box::new(model))
            }
            Architecture::Mistral => {
                let model = crate::mistral::MistralModel::from_weights(
                    config,
                    self.inner.device.clone(),
                    weights,
                    num_kv_blocks,
                )?;
                Ok(Box::new(model))
            }
            Architecture::Mixtral => {
                let model = crate::mixtral::MixtralModel::from_weights(
                    config,
                    self.inner.device.clone(),
                    weights,
                    num_kv_blocks,
                )?;
                Ok(Box::new(model))
            }
            Architecture::Qwen3 => {
                let config = self.load_config()?;
                let model = crate::qwen3::model::Qwen3Model::from_weights(
                    config,
                    self.inner.device.clone(),
                    weights,
                    num_kv_blocks,
                    self.inner.kv_quantization,
                )?;
                Ok(Box::new(model))
            }
            Architecture::Qwen35 => {
                let config = self.load_config()?;
                eprintln!("Qwen35: architecture detected, loading hybrid model");
                weights = crate::loader::remap_qwen35_weight_keys(weights);
                eprintln!("Qwen35: after remap, checking for embed_tokens...");
                if !weights.contains_key("model.embed_tokens.weight")
                    && !weights.contains_key("model.language_model.embed_tokens.weight")
                {
                    let embed_keys: Vec<_> =
                        weights.keys().filter(|k| k.contains("embed")).collect();
                    eprintln!("Qwen35 embed keys available: {:?}", embed_keys);
                }
                let model = crate::qwen3_5::Qwen35HybridModel::from_weights(
                    config,
                    self.inner.device.clone(),
                    weights,
                    num_kv_blocks,
                )?;
                Ok(Box::new(model))
            }
            Architecture::Gemma4 => {
                let model = crate::gemma4::Gemma4Model::from_weights(
                    config,
                    self.inner.device.clone(),
                    weights,
                    num_kv_blocks,
                )?;
                Ok(Box::new(model))
            }
        }
    }

    pub fn load_model(&self) -> Result<Box<dyn vllm_traits::ModelBackend>> {
        self.load()
    }

    pub fn print_weight_keys(weights: &std::collections::HashMap<String, Tensor>) {
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
