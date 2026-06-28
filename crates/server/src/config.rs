#![allow(clippy::module_name_repetitions)]
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// `ConfigValidationError`: config validation error.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConfigValidationError {
    #[error("server.port must be > 0")]
    PortZero,
    #[error("server.log_level must be one of: trace, debug, info, warn, error")]
    InvalidLogLevel,
    #[error("engine.max_draft_tokens must be <= 64")]
    MaxDraftTokensTooLarge,
    #[error("engine.num_kv_blocks must be > 0")]
    KvBlocksZero,
    #[error("engine.num_kv_blocks must be <= 65536")]
    KvBlocksTooLarge,
    #[error("engine.max_batch_size must be > 0")]
    MaxBatchSizeZero,
    #[error("engine.tensor_parallel_size must be > 0")]
    TensorParallelSizeZero,
    #[error("engine.vram_budget_bytes must be > 0 when set")]
    VramBudgetZero,
    #[error("engine.draft_specs[].id must not be empty")]
    EmptyDraftId,
    #[error("engine.draft_specs[].id duplicate: {0}")]
    DuplicateDraftId(String),
}

/// `ConfigValidationErrors`: config validation errors.
#[derive(Debug, thiserror::Error)]
#[error("config validation failed: {0:?}")]
pub struct ConfigValidationErrors(pub Vec<ConfigValidationError>);

/// `ServerConfig`: server configuration.
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

const fn default_port() -> u16 {
    8000
}

fn default_log_level() -> String {
    "info".to_string()
}

/// `AuthConfig`: auth configuration.
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
    /// Collect the effective set of API keys from all configured sources:
    ///   1. `api_keys` (literal list)
    ///   2. `api_keys_env` (comma-separated env var, parsed lazily)
    ///   3. `api_keys_file` (one key per line, `#` comments, blanks ignored)
    ///
    /// Missing/unreadable sources contribute nothing. Empty entries are
    /// filtered. Returns a deduplicated list in source order.
    #[must_use]
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

const fn default_rate_limit_requests() -> usize {
    100
}

const fn default_rate_limit_window() -> u64 {
    60
}

/// `DraftSpecConfig`: draft spec configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DraftSpecConfig {
    pub id: String,
    pub path: String,
    #[serde(default = "default_draft_layers")]
    pub num_layers: usize,
    #[serde(default)]
    pub weight_size_bytes: u64,
    #[serde(default)]
    pub architecture: Option<String>,
}

const fn default_draft_layers() -> usize {
    4
}

/// `EngineConfig`: engine configuration.
#[allow(clippy::derivable_impls)]
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
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
    #[serde(default = "default_enable_adaptive_speculative")]
    pub enable_adaptive_speculative: bool,
    /// v18.0: VRAM budget for speculative draft models in bytes. When set,
    /// the Engine is constructed with `with_budget_boxed` so all draft
    /// registrations share this budget. When `None`, drafts are unbounded.
    #[serde(default)]
    pub vram_budget_bytes: Option<u64>,
    /// v18.0: Pre-declared external draft model specs. Each becomes a
    /// `DraftSpec` registered with the Engine's draft registry. The server
    /// does NOT load weights at startup; lazy loading via `DraftLoader`.
    #[serde(default)]
    pub draft_specs: Vec<DraftSpecConfig>,
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
            enable_adaptive_speculative: default_enable_adaptive_speculative(),
            vram_budget_bytes: None,
            draft_specs: Vec::new(),
        }
    }
}

const fn default_max_draft_tokens() -> usize {
    8
}

const fn default_num_kv_blocks() -> usize {
    1024
}

const fn default_max_batch_size() -> usize {
    256
}

const fn default_max_waiting_batches() -> usize {
    10
}

const fn default_tensor_parallel_size() -> usize {
    1
}

const fn default_kv_quantization() -> bool {
    false
}

const fn default_enable_adaptive_speculative() -> bool {
    true
}

/// `AppConfig`: app configuration.
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
    /// Load an [`AppConfig`] starting from `Self::default()` and layering
    /// optional overrides:
    ///   1. YAML file at `path` (if `Some` and the file exists).
    ///   2. YAML file at `$VLLM_CONFIG_PATH` (if the env var is set and the
    ///      file exists; takes precedence over `path`).
    ///
    /// Missing files and parse errors are silently ignored — defaults win.
    /// Use [`AppConfig::validate`] after loading to surface invalid configs.
    #[must_use]
    pub fn load(path: Option<PathBuf>) -> Self {
        let mut config = Self::default();

        if let Some(config_path) = path {
            if config_path.exists() {
                if let Ok(contents) = std::fs::read_to_string(&config_path) {
                    if let Ok(loaded) = serde_saphyr::from_str::<Self>(&contents) {
                        config = loaded;
                    }
                }
            }
        }

        if let Ok(env_path) = std::env::var("VLLM_CONFIG_PATH") {
            let env_config_path = PathBuf::from(env_path);
            if env_config_path.exists() {
                if let Ok(contents) = std::fs::read_to_string(&env_config_path) {
                    if let Ok(loaded) = serde_saphyr::from_str::<Self>(&contents) {
                        config = loaded;
                    }
                }
            }
        }

        config
    }

    /// Check the loaded config against all invariants. Collects every
    /// violation (rather than failing on the first) and returns them as a
    /// single [`ConfigValidationErrors`]. Returns `Ok(())` when the config is
    /// usable.
    ///
    /// Invariants enforced:
    /// - `server.port > 0`
    /// - `server.log_level ∈ {trace, debug, info, warn, error}`
    /// - `engine.max_draft_tokens ≤ 64`
    /// - `0 < engine.num_kv_blocks ≤ 65536`
    /// - `engine.max_batch_size > 0`
    /// - `engine.tensor_parallel_size > 0`
    /// - `engine.vram_budget_bytes` is either `None` or `> 0`
    /// - every `draft_specs[].id` is non-empty and unique
    ///
    /// # Errors
    ///
    /// Returns [`ConfigValidationErrors`] containing every violation found.
    pub fn validate(&self) -> Result<(), ConfigValidationErrors> {
        let mut errors = Vec::new();

        if self.server.port == 0 {
            errors.push(ConfigValidationError::PortZero);
        }

        let valid_levels = ["trace", "debug", "info", "warn", "error"];
        if !valid_levels.contains(&self.server.log_level.as_str()) {
            errors.push(ConfigValidationError::InvalidLogLevel);
        }

        if self.engine.max_draft_tokens > 64 {
            errors.push(ConfigValidationError::MaxDraftTokensTooLarge);
        }

        if self.engine.num_kv_blocks == 0 {
            errors.push(ConfigValidationError::KvBlocksZero);
        }
        if self.engine.num_kv_blocks > 65536 {
            errors.push(ConfigValidationError::KvBlocksTooLarge);
        }

        if self.engine.max_batch_size == 0 {
            errors.push(ConfigValidationError::MaxBatchSizeZero);
        }

        if self.engine.tensor_parallel_size == 0 {
            errors.push(ConfigValidationError::TensorParallelSizeZero);
        }

        // v18.0 validation
        if let Some(b) = self.engine.vram_budget_bytes {
            if b == 0 {
                errors.push(ConfigValidationError::VramBudgetZero);
            }
        }
        let mut seen_draft_ids = std::collections::HashSet::new();
        for spec in &self.engine.draft_specs {
            if spec.id.is_empty() {
                errors.push(ConfigValidationError::EmptyDraftId);
            }
            if !seen_draft_ids.insert(&spec.id) {
                errors.push(ConfigValidationError::DuplicateDraftId(spec.id.clone()));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(ConfigValidationErrors(errors))
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
        assert!(
            errors
                .0
                .iter()
                .any(|e| matches!(e, ConfigValidationError::PortZero))
        );
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
        assert!(
            errors
                .0
                .iter()
                .any(|e| matches!(e, ConfigValidationError::TensorParallelSizeZero))
        );
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

    // ─────────────────── v18.0 validation tests ───────────────────

    #[test]
    fn test_validate_vram_budget_zero_fails() {
        let mut config = AppConfig::default();
        config.engine.vram_budget_bytes = Some(0);
        let errors = config.validate().unwrap_err();
        assert!(
            errors
                .0
                .iter()
                .any(|e| matches!(e, ConfigValidationError::VramBudgetZero))
        );
    }

    #[test]
    fn test_validate_vram_budget_nonzero_ok() {
        let mut config = AppConfig::default();
        config.engine.vram_budget_bytes = Some(1024);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_draft_spec_empty_id_fails() {
        let mut config = AppConfig::default();
        config.engine.draft_specs = vec![DraftSpecConfig {
            id: String::new(),
            path: "/nope".into(),
            num_layers: 4,
            weight_size_bytes: 0,
            architecture: None,
        }];
        let errors = config.validate().unwrap_err();
        assert!(
            errors
                .0
                .iter()
                .any(|e| matches!(e, ConfigValidationError::EmptyDraftId))
        );
    }

    #[test]
    fn test_validate_draft_spec_duplicate_id_fails() {
        let mut config = AppConfig::default();
        config.engine.draft_specs = vec![
            DraftSpecConfig {
                id: "a".into(),
                path: "/a".into(),
                num_layers: 4,
                weight_size_bytes: 0,
                architecture: None,
            },
            DraftSpecConfig {
                id: "a".into(),
                path: "/a2".into(),
                num_layers: 4,
                weight_size_bytes: 0,
                architecture: None,
            },
        ];
        let errors = config.validate().unwrap_err();
        assert!(
            errors
                .0
                .iter()
                .any(|e| matches!(e, ConfigValidationError::DuplicateDraftId(_)))
        );
    }

    #[test]
    fn test_validate_draft_spec_unique_ids_ok() {
        let mut config = AppConfig::default();
        config.engine.draft_specs = vec![
            DraftSpecConfig {
                id: "a".into(),
                path: "/a".into(),
                num_layers: 4,
                weight_size_bytes: 0,
                architecture: None,
            },
            DraftSpecConfig {
                id: "b".into(),
                path: "/b".into(),
                num_layers: 4,
                weight_size_bytes: 0,
                architecture: None,
            },
        ];
        assert!(config.validate().is_ok());
    }
}
