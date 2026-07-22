//! Server configuration: top-level `AppConfig` plus error types and
//! loading/validation. The three independent sections (`ServerConfig`,
//! `EngineConfig`, `AuthConfig`) live in sibling modules and are composed
//! into a single YAML/JSON document.
//!
//! Module layout:
//!
//! - `mod.rs` — `AppConfig` + `Default` + `load` + `validate` +
//!   `ConfigValidationError` / `ConfigValidationErrors`
//! - `server` — `ServerConfig` (bind address, port, log level)
//! - `engine` — `EngineConfig` + `DraftSpecConfig` (scheduler tuning, draft specs)
//! - `auth` — `AuthConfig` + `resolve_api_keys` (API keys + rate limit)

// `ConfigXxx` / `AppConfig` / `ServerConfig` etc. are intentional public
// API names — re-exported across the workspace and consumed by
// downstream tooling. The `ConfigValidationError` /
// `ConfigValidationErrors` aggregate type follows the same convention.
#![allow(clippy::module_name_repetitions)]

mod auth;
mod cors;
mod engine;
mod multi_node;
mod server;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Error type for `ConfigValidation`. Returned from every fallible public API; covers I/O, validation, and resource-limit failures. Use [`Result<T>`] alias in the same module.
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
    #[error("server.shutdown_drain_grace_secs must be <= 300")]
    ShutdownDrainGraceTooLarge,
}

/// Aggregated list of [`ConfigValidationError`]s returned by
/// [`AppConfig::validate`]. Always carries every violation found in
/// one pass rather than failing on the first; the inner `Vec` is
/// `pub` so callers can pattern-match or render their own summary.
#[derive(Debug, thiserror::Error)]
#[error("config validation failed: {0:?}")]
pub struct ConfigValidationErrors(pub Vec<ConfigValidationError>);

pub use auth::AuthConfig;
pub use cors::CorsConfigFile;
pub use engine::{DraftSpecConfig, EngineConfig};
pub use multi_node::MultiNodeConfig;
pub use server::{ServerConfig, is_loopback_address};

/// Top-level server configuration. Composes the three independent
/// sections ([`ServerConfig`], [`EngineConfig`], [`AuthConfig`]) that
/// are loaded as a single YAML/JSON document and validated together
/// at startup. See [`AppConfig::load`] for the loading precedence and
/// [`AppConfig::validate`] for the invariant check.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::derivable_impls)]
pub struct AppConfig {
    /// HTTP server section (bind address, port, log level).
    #[serde(default)]
    pub server: ServerConfig,
    /// Engine section (KV blocks, batch size, draft specs).
    #[serde(default)]
    pub engine: EngineConfig,
    /// Authentication / rate-limit section.
    #[serde(default)]
    pub auth: AuthConfig,
    /// CORS section. Closed by default — operators opt in via
    /// `cors.allow_origins` in the YAML/JSON config.
    #[serde(default)]
    pub cors: CorsConfigFile,
}

impl Default for AppConfig {
    #[allow(clippy::derivable_impls)]
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            engine: EngineConfig::default(),
            auth: AuthConfig::default(),
            cors: CorsConfigFile::default(),
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
        let config = path
            .filter(|config_path| config_path.exists())
            .and_then(|config_path| std::fs::read_to_string(config_path).ok())
            .and_then(|contents| serde_saphyr::from_str::<Self>(&contents).ok())
            .unwrap_or_default();

        std::env::var("VLLM_CONFIG_PATH")
            .ok()
            .map(PathBuf::from)
            .filter(|config_path| config_path.exists())
            .and_then(|config_path| std::fs::read_to_string(config_path).ok())
            .and_then(|contents| serde_saphyr::from_str::<Self>(&contents).ok())
            .unwrap_or(config)
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

        // Production-readiness §7: cap the drain grace so a typo
        // (e.g. `shutdown_drain_grace_secs: 3600`) can't block shutdown
        // for an hour. 5 minutes is the upper bound a patient operator
        // might reasonably want; the default is 5 seconds.
        if self.server.shutdown_drain_grace_secs > 300 {
            errors.push(ConfigValidationError::ShutdownDrainGraceTooLarge);
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
        if let Some(b) = self.engine.vram_budget_bytes
            && b == 0
        {
            errors.push(ConfigValidationError::VramBudgetZero);
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

// Unit tests are extracted to `tests.rs` to keep this file under the
// 800-line soft cap. See `tests.rs` for the test surface
// (AppConfig::default, AppConfig::validate, port / tensor-parallel /
// vram-budget / draft-spec invariants, kv_quantization toggle).
#[cfg(test)]
mod tests;
