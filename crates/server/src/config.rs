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
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_draft_tokens: default_max_draft_tokens(),
            num_kv_blocks: default_num_kv_blocks(),
            max_batch_size: default_max_batch_size(),
            max_waiting_batches: default_max_waiting_batches(),
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::derivable_impls)]
pub struct AppConfig {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub engine: EngineConfig,
}

impl Default for AppConfig {
    #[allow(clippy::derivable_impls)]
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            engine: EngineConfig::default(),
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

        if let Ok(host) = std::env::var("VLLM_HOST") {
            config.server.host = host;
        }
        if let Ok(port) = std::env::var("VLLM_PORT") {
            if let Ok(port) = port.parse() {
                config.server.port = port;
            }
        }
        if let Ok(level) = std::env::var("VLLM_LOG_LEVEL") {
            config.server.log_level = level;
        }
        if let Ok(log_dir) = std::env::var("VLLM_LOG_DIR") {
            config.server.log_dir = Some(log_dir);
        }
        if let Ok(max_draft) = std::env::var("VLLM_MAX_DRAFT_TOKENS") {
            if let Ok(v) = max_draft.parse() {
                config.engine.max_draft_tokens = v;
            }
        }
        if let Ok(blocks) = std::env::var("VLLM_KV_BLOCKS") {
            if let Ok(v) = blocks.parse() {
                config.engine.num_kv_blocks = v;
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
}
