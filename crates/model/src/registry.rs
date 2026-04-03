use crate::config::ModelConfig as LlamaMistralConfig;
use crate::gemma4::Gemma4Model;
use crate::llama::LlamaModel;
use crate::mistral::MistralModel;
use crate::qwen3_config::Qwen3Config;
use candle_core::{Device, Result};
use std::path::Path;
use vllm_traits::ModelBackend;

pub trait ModelRegistry: Send + Sync {
    fn get_model(
        config: &ModelConfig,
        device: Device,
        max_seq_len: usize,
    ) -> Result<Box<dyn ModelBackend>>;
    fn supported_models() -> Vec<ModelInfo>;
    fn detect_model_type(config_json: &str) -> Option<String>;
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub architecture: String,
    pub max_seq_len: Option<usize>,
}

pub struct ModelFactory;

impl ModelFactory {
    pub fn create_model(
        model_type: &str,
        config: &ModelConfig,
        device: Device,
        max_seq_len: usize,
    ) -> Result<Box<dyn ModelBackend>> {
        match model_type.to_lowercase().as_str() {
            "qwen3" | "qwen2" => {
                let qwen_config = Qwen3Config {
                    vocab_size: config.vocab_size,
                    hidden_size: config.hidden_size,
                    num_hidden_layers: config.num_hidden_layers,
                    num_attention_heads: config.num_attention_heads,
                    num_key_value_heads: config.num_key_value_heads,
                    intermediate_size: config.intermediate_size,
                    sliding_window: config.sliding_window,
                    ..Default::default()
                };
                let model = crate::qwen3::model::Qwen3Model::new(qwen_config, device, max_seq_len)?;
                Ok(Box::new(model))
            }
            "qwen3.5" | "qwen2.5" => {
                let qwen_config = Qwen3Config {
                    vocab_size: config.vocab_size,
                    hidden_size: config.hidden_size,
                    num_hidden_layers: config.num_hidden_layers,
                    num_attention_heads: config.num_attention_heads,
                    num_key_value_heads: config.num_key_value_heads,
                    intermediate_size: config.intermediate_size,
                    sliding_window: config.sliding_window,
                    ..Default::default()
                };
                let model = crate::qwen3_5::model::Qwen35Model::new(qwen_config, device)?;
                Ok(Box::new(model))
            }
            _ => Err(candle_core::Error::msg(format!(
                "Unsupported model type: {}",
                model_type
            ))),
        }
    }

    pub fn detect_from_json(config_json: &str) -> Option<String> {
        if let Ok(config) = serde_json::from_str::<serde_json::Value>(config_json) {
            if let Some(model_type) = config.get("model_type").and_then(|v| v.as_str()) {
                return Some(model_type.to_string());
            }
            if let Some(architectures) = config.get("architectures").and_then(|v| v.as_array()) {
                if let Some(first) = architectures.first() {
                    if let Some(name) = first.as_str() {
                        return Some(name.to_string());
                    }
                }
            }
        }
        None
    }

    pub fn create_llama(
        config: LlamaMistralConfig,
        device: Device,
        num_kv_blocks: usize,
    ) -> Result<LlamaModel> {
        LlamaModel::new(config, device, num_kv_blocks)
    }

    pub fn create_mistral(
        config: LlamaMistralConfig,
        device: Device,
        num_kv_blocks: usize,
    ) -> Result<MistralModel> {
        MistralModel::new(config, device, num_kv_blocks)
    }

    pub fn create_gemma4(
        config: LlamaMistralConfig,
        device: Device,
        num_kv_blocks: usize,
    ) -> Result<Gemma4Model> {
        Gemma4Model::new(config, device, num_kv_blocks)
    }
}

impl ModelRegistry for ModelFactory {
    fn get_model(
        config: &ModelConfig,
        device: Device,
        max_seq_len: usize,
    ) -> Result<Box<dyn ModelBackend>> {
        let model_type = config.model_type.as_deref().unwrap_or("qwen3");
        Self::create_model(model_type, config, device, max_seq_len)
    }

    fn supported_models() -> Vec<ModelInfo> {
        vec![
            ModelInfo {
                name: "Qwen3".to_string(),
                architecture: "Qwen3ForCausalLM".to_string(),
                max_seq_len: Some(32768),
            },
            ModelInfo {
                name: "Qwen3.5".to_string(),
                architecture: "Qwen3ForCausalLM".to_string(),
                max_seq_len: Some(32768),
            },
        ]
    }

    fn detect_model_type(config_json: &str) -> Option<String> {
        Self::detect_from_json(config_json)
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_type: Option<String>,
    pub vocab_size: Option<usize>,
    pub hidden_size: Option<usize>,
    pub num_hidden_layers: Option<usize>,
    pub num_attention_heads: Option<usize>,
    pub num_key_value_heads: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub sliding_window: Option<usize>,
}

impl ModelConfig {
    pub fn from_file(model_dir: &str) -> Result<Self> {
        let config_path = Path::new(model_dir).join("config.json");
        let content = std::fs::read_to_string(config_path)
            .map_err(|e| candle_core::Error::msg(format!("Failed to read config: {}", e)))?;

        let qwen_config: Qwen3Config = serde_json::from_str(&content)
            .map_err(|e| candle_core::Error::msg(format!("Failed to parse config: {}", e)))?;

        Ok(Self {
            model_type: Some("qwen3".to_string()),
            vocab_size: Some(qwen_config.vocab_size()),
            hidden_size: Some(qwen_config.hidden_size()),
            num_hidden_layers: Some(qwen_config.num_hidden_layers()),
            num_attention_heads: Some(qwen_config.num_attention_heads()),
            num_key_value_heads: Some(qwen_config.num_key_value_heads()),
            intermediate_size: Some(qwen_config.intermediate_size()),
            sliding_window: qwen_config.sliding_window,
        })
    }
}
