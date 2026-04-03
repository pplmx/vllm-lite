//! Model configuration types.
//!
//! Provides unified configuration for multiple model architectures
//! including Llama, Mistral, and Qwen3.
//!
//! # Usage
//! ```rust
//! use crate::config::{ModelConfig, Architecture};
//!
//! // Use predefined configs
//! let llama = ModelConfig::llama_7b();
//! let mistral = ModelConfig::mistral_7b();
//!
//! // Or create from HuggingFace config.json
//! let config = ModelConfig::from_config_json(&value).unwrap();
//! ```

pub mod architecture;
pub mod model_config;

pub use architecture::{Architecture, AttentionType, MlpType, NormType};
pub use model_config::ModelConfig;
