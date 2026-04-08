#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::derivable_impls)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_log_level")]
    pub log_level: String,
    #[serde(default)]
    pub log_dir: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            log_level: default_log_level(),
            log_dir: None,
        }
    }
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8000
}

fn default_log_level() -> String {
    "info".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    #[serde(default)]
    pub api_keys: Vec<String>,
    #[serde(default)]
    pub api_keys_env: Option<String>,
    #[serde(default)]
    pub api_keys_file: Option<String>,
    #[serde(default = "default_rate_limit_requests")]
    pub rate_limit_requests: usize,
    #[serde(default = "default_rate_limit_window")]
    pub rate_limit_window_secs: u64,
}

impl AuthConfig {
    pub fn resolve_api_keys(&self) -> Vec<String> {
        let mut keys = self.api_keys.clone();

        if let Some(ref env_var) = self.api_keys_env {
            if let Ok(env_keys) = std::env::var(env_var) {
                for key in env_keys.split(',') {
                    let key = key.trim().to_string();
                    if !key.is_empty() {
                        keys.push(key);
                    }
                }
            }
        }

        if let Some(ref file_path) = self.api_keys_file {
            if let Ok(content) = std::fs::read_to_string(file_path) {
                for line in content.lines() {
                    let key = line.trim().to_string();
                    if !key.is_empty() && !key.starts_with('#') {
                        keys.push(key);
                    }
                }
            }
        }

        keys
    }
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            api_keys: vec![],
            api_keys_env: None,
            api_keys_file: None,
            rate_limit_requests: 100,
            rate_limit_window_secs: 60,
        }
    }
}

fn default_rate_limit_requests() -> usize {
    100
}

fn default_rate_limit_window() -> u64 {
    60
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::derivable_impls)]
pub struct EngineConfig {
    #[serde(default = "default_max_draft_tokens")]
    pub max_draft_tokens: usize,
    #[serde(default = "default_num_kv_blocks")]
    pub num_kv_blocks: usize,
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,
    #[serde(default = "default_max_waiting_batches")]
    pub max_waiting_batches: usize,
    #[serde(default = "default_tensor_parallel_size")]
    pub tensor_parallel_size: usize,
    #[serde(default = "default_kv_quantization")]
    pub kv_quantization: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_draft_tokens: default_max_draft_tokens(),
            num_kv_blocks: default_num_kv_blocks(),
            max_batch_size: default_max_batch_size(),
            max_waiting_batches: default_max_waiting_batches(),
            tensor_parallel_size: default_tensor_parallel_size(),
            kv_quantization: default_kv_quantization(),
        }
    }
}

fn default_max_draft_tokens() -> usize {
    8
}

fn default_num_kv_blocks() -> usize {
    1024
}

fn default_max_batch_size() -> usize {
    256
}

fn default_max_waiting_batches() -> usize {
    10
}

fn default_tensor_parallel_size() -> usize {
    1
}

fn default_kv_quantization() -> bool {
    false
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::derivable_impls)]
pub struct AppConfig {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub engine: EngineConfig,
    #[serde(default)]
    pub auth: AuthConfig,
}

impl Default for AppConfig {
    #[allow(clippy::derivable_impls)]
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            engine: EngineConfig::default(),
            auth: AuthConfig::default(),
        }
    }
}

impl AppConfig {
    pub fn load(path: Option<PathBuf>) -> Self {
        let mut config = Self::default();

        if let Some(config_path) = path {
            if config_path.exists() {
                if let Ok(contents) = std::fs::read_to_string(&config_path) {
                    if let Ok(loaded) = serde_yaml::from_str::<AppConfig>(&contents) {
                        config = loaded;
                    }
                }
            }
        }

        config
    }

    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        if self.server.port == 0 {
            errors.push("server.port must be > 0".to_string());
        }

        let valid_levels = ["trace", "debug", "info", "warn", "error"];
        if !valid_levels.contains(&self.server.log_level.as_str()) {
            errors.push(format!(
                "server.log_level must be one of: {:?}",
                valid_levels
            ));
        }

        if self.engine.max_draft_tokens == 0 {
            errors.push("engine.max_draft_tokens must be > 0".to_string());
        }
        if self.engine.max_draft_tokens > 64 {
            errors.push("engine.max_draft_tokens must be <= 64".to_string());
        }

        if self.engine.num_kv_blocks == 0 {
            errors.push("engine.num_kv_blocks must be > 0".to_string());
        }
        if self.engine.num_kv_blocks > 65536 {
            errors.push("engine.num_kv_blocks must be <= 65536".to_string());
        }

        if self.engine.max_batch_size == 0 {
            errors.push("engine.max_batch_size must be > 0".to_string());
        }

        if self.engine.tensor_parallel_size == 0 {
            errors.push("engine.tensor_parallel_size must be > 0".to_string());
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_config_defaults() {
        let config = AppConfig::default();
        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.server.port, 8000);
        assert_eq!(config.server.log_level, "info");
        assert_eq!(config.engine.max_draft_tokens, 8);
        assert_eq!(config.engine.num_kv_blocks, 1024);
        assert_eq!(config.engine.max_batch_size, 256);
        assert_eq!(config.engine.max_waiting_batches, 10);
    }

    #[test]
    fn test_app_config_validate_passes() {
        let config = AppConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_app_config_validate_fails_zero_port() {
        let mut config = AppConfig::default();
        config.server.port = 0;
        let errors = config.validate().unwrap_err();
        assert!(errors.iter().any(|e| e.contains("server.port")));
    }

    #[test]
    fn test_tensor_parallel_size_default() {
        let config = AppConfig::default();
        assert_eq!(config.engine.tensor_parallel_size, 1);
    }

    #[test]
    fn test_tensor_parallel_size_from_config() {
        let mut config = AppConfig::default();
        config.engine.tensor_parallel_size = 4;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_tensor_parallel_size_validate_fails_zero() {
        let mut config = AppConfig::default();
        config.engine.tensor_parallel_size = 0;
        let errors = config.validate().unwrap_err();
        assert!(errors.iter().any(|e| e.contains("tensor_parallel_size")));
    }

    #[test]
    fn test_kv_quantization_default() {
        let config = AppConfig::default();
        assert!(!config.engine.kv_quantization);
    }

    #[test]
    fn test_kv_quantization_from_config() {
        let mut config = AppConfig::default();
        config.engine.kv_quantization = true;
        assert!(config.validate().is_ok());
        assert!(config.engine.kv_quantization);
    }
}
