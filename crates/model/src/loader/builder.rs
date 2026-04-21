use candle_core::{Device, Result, Tensor};
use std::path::Path;
use std::sync::Arc;

use crate::arch::{ARCHITECTURE_REGISTRY, register_all_archs};
use crate::config::Architecture as ConfigArchitecture;
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
    #[allow(dead_code)]
    kv_quantization: bool,
    config_json: serde_json::Value,
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

        Ok(Self {
            device,
            model_dir,
            num_kv_blocks,
            kv_quantization,
            config_json,
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
            }),
        }
    }

    pub fn builder(device: Device) -> ModelLoaderBuilder {
        ModelLoaderBuilder::new(device)
    }

    pub fn device(&self) -> &Device {
        &self.inner.device
    }

    pub fn architecture(&self) -> ConfigArchitecture {
        ARCHITECTURE_REGISTRY
            .detect(&self.inner.config_json)
            .map(|name| match name.as_str() {
                "llama" => ConfigArchitecture::Llama,
                "mistral" => ConfigArchitecture::Mistral,
                "qwen3" | "qwen2" => ConfigArchitecture::Qwen3,
                "qwen3.5" => ConfigArchitecture::Qwen35,
                "gemma4" => ConfigArchitecture::Gemma4,
                "mixtral" => ConfigArchitecture::Mixtral,
                _ => ConfigArchitecture::Llama,
            })
            .unwrap_or(ConfigArchitecture::Llama)
    }

    pub fn config_json(&self) -> &serde_json::Value {
        &self.inner.config_json
    }

    pub fn load_config<T: serde::de::DeserializeOwned>(&self) -> Result<T> {
        let config_path = Path::new(&self.inner.model_dir).join("config.json");
        let content = std::fs::read_to_string(config_path)
            .map_err(|e| candle_core::Error::msg(format!("Failed to read config: {}", e)))?;
        serde_json::from_str(&content)
            .map_err(|e| candle_core::Error::msg(format!("Failed to parse config: {}", e)))
    }

    pub fn load_weights(&self) -> Result<std::collections::HashMap<String, Tensor>> {
        let path = Path::new(&self.inner.model_dir);
        super::checkpoint::load_checkpoint(path, &self.inner.device)
    }

    pub fn load(&self) -> Result<Box<dyn vllm_traits::ModelBackend>> {
        register_all_archs(&ARCHITECTURE_REGISTRY);

        let arch_name = ARCHITECTURE_REGISTRY
            .detect(&self.inner.config_json)
            .ok_or_else(|| candle_core::Error::msg("Unsupported architecture"))?;

        let arch = ARCHITECTURE_REGISTRY
            .get(&arch_name)
            .ok_or_else(|| candle_core::Error::msg("Architecture not found"))?;

        let config = ModelConfig::from_config_json(&self.inner.config_json)
            .map_err(|e| candle_core::Error::msg(format!("Failed to parse model config: {}", e)))?;
        let weights = self.load_weights()?;
        let weights = arch.remap_weights(weights);

        arch.create_model(
            config,
            self.inner.device.clone(),
            weights,
            self.inner.num_kv_blocks,
        )
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_builder_new() {
        let loader = ModelLoader::new(Device::Cpu);
        assert_eq!(loader.inner.num_kv_blocks, 1024);
    }

    #[test]
    fn test_builder_requires_model_dir() {
        let loader = ModelLoader::builder(Device::Cpu).build();
        assert!(loader.is_err());
    }

    #[test]
    fn test_builder_with_config_file() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        std::fs::write(&config_path, r#"{"model_type": "llama"}"#).unwrap();

        let loader = ModelLoader::builder(Device::Cpu)
            .with_model_dir(temp_dir.path().to_str().unwrap().to_string())
            .build()
            .unwrap();

        assert_eq!(loader.inner.model_dir, temp_dir.path().to_str().unwrap());
        assert_eq!(loader.inner.num_kv_blocks, 1024);
    }

    #[test]
    fn test_builder_with_kv_blocks() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        std::fs::write(&config_path, r#"{"model_type": "llama"}"#).unwrap();

        let loader = ModelLoader::builder(Device::Cpu)
            .with_model_dir(temp_dir.path().to_str().unwrap().to_string())
            .with_kv_blocks(2048)
            .build()
            .unwrap();
        assert_eq!(loader.inner.num_kv_blocks, 2048);
    }

    #[test]
    fn test_device_getter() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        std::fs::write(&config_path, r#"{"model_type": "llama"}"#).unwrap();

        let loader = ModelLoader::builder(Device::Cpu)
            .with_model_dir(temp_dir.path().to_str().unwrap().to_string())
            .build()
            .unwrap();
        let _device = loader.device();
    }

    #[test]
    fn test_config_json_getter() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        std::fs::write(&config_path, r#"{"model_type": "llama"}"#).unwrap();

        let loader = ModelLoader::builder(Device::Cpu)
            .with_model_dir(temp_dir.path().to_str().unwrap().to_string())
            .build()
            .unwrap();
        let json = loader.config_json();
        assert_eq!(json.get("model_type").and_then(|v| v.as_str()), Some("llama"));
    }

    #[test]
    fn test_load_config_generic() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        std::fs::write(&config_path, r#"{"test": "value"}"#).unwrap();

        let loader = ModelLoader::builder(Device::Cpu)
            .with_model_dir(temp_dir.path().to_str().unwrap().to_string())
            .build()
            .unwrap();

        #[derive(serde::Deserialize, Debug, PartialEq)]
        struct TestConfig {
            test: String,
        }

        let config: TestConfig = loader.load_config().unwrap();
        assert_eq!(config.test, "value");
    }

    #[test]
    fn test_load_config_invalid_json() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        std::fs::write(&config_path, "invalid json").unwrap();

        let loader = ModelLoader::builder(Device::Cpu)
            .with_model_dir(temp_dir.path().to_str().unwrap().to_string())
            .build();

        assert!(loader.is_err());
    }

    #[test]
    fn test_clone() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        std::fs::write(&config_path, r#"{"model_type": "llama"}"#).unwrap();

        let loader = ModelLoader::builder(Device::Cpu)
            .with_model_dir(temp_dir.path().to_str().unwrap().to_string())
            .build()
            .unwrap();
        let cloned = loader.clone();
        assert_eq!(loader.inner.model_dir, cloned.inner.model_dir);
        assert_eq!(loader.inner.num_kv_blocks, cloned.inner.num_kv_blocks);
    }

    #[test]
    fn test_architecture_getter() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        std::fs::write(&config_path, r#"{"model_type": "llama"}"#).unwrap();

        let loader = ModelLoader::builder(Device::Cpu)
            .with_model_dir(temp_dir.path().to_str().unwrap().to_string())
            .build()
            .unwrap();
        let arch = loader.architecture();
        assert_eq!(arch, ConfigArchitecture::Llama);
    }
}
