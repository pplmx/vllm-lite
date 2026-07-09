//! `clap`-derived CLI argument structs that override fields in [`AppConfig`].
//!
//! Parsed once at startup; the resolved config is then handed to the
//! engine constructor. Use `--help` for the full list of flags.
use crate::config::AppConfig;
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

/// Error type for `CliValidation`. Returned from every fallible public API; covers I/O, validation, and resource-limit failures. Use [`Result<T>`] alias in the same module.
#[derive(Clone, Debug, thiserror::Error)]
pub enum CliValidationError {
    #[error("'{0}' is not a valid number")]
    NotANumber(String),
    #[error("value must be between {min} and {max}")]
    OutOfRange { min: usize, max: usize },
    #[error("'{0}' is not a valid port number")]
    InvalidPort(String),
    #[error("port must be between 1 and 65535")]
    PortOutOfRange,
}

/// `LogLevel`. See the type definition for fields and behavior.
#[derive(Clone, Debug, ValueEnum, PartialEq, Eq)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

#[allow(clippy::derivable_impls)]
impl Default for LogLevel {
    fn default() -> Self {
        Self::Info
    }
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Trace => write!(f, "trace"),
            Self::Debug => write!(f, "debug"),
            Self::Info => write!(f, "info"),
            Self::Warn => write!(f, "warn"),
            Self::Error => write!(f, "error"),
        }
    }
}

fn parse_usize_in_range(s: &str, min: usize, max: usize) -> Result<usize, CliValidationError> {
    let val: usize = s
        .parse()
        .map_err(|_| CliValidationError::NotANumber(s.to_string()))?;
    if val < min || val > max {
        Err(CliValidationError::OutOfRange { min, max })
    } else {
        Ok(val)
    }
}

fn validate_port(s: &str) -> Result<u16, CliValidationError> {
    let val: u16 = s
        .parse()
        .map_err(|_| CliValidationError::InvalidPort(s.to_string()))?;
    if val == 0 {
        Err(CliValidationError::PortOutOfRange)
    } else {
        Ok(val)
    }
}

fn validate_tensor_parallel_size(s: &str) -> Result<usize, CliValidationError> {
    parse_usize_in_range(s, 1, 64)
}

fn validate_kv_blocks(s: &str) -> Result<usize, CliValidationError> {
    parse_usize_in_range(s, 1, 65536)
}

fn validate_max_batch_size(s: &str) -> Result<usize, CliValidationError> {
    parse_usize_in_range(s, 1, 8192)
}

fn validate_max_waiting_batches(s: &str) -> Result<usize, CliValidationError> {
    parse_usize_in_range(s, 1, 100)
}

fn validate_max_draft_tokens(s: &str) -> Result<usize, CliValidationError> {
    parse_usize_in_range(s, 0, 64)
}

/// `CliArgs`. See the type definition for fields and behavior.
#[derive(Parser, Debug)]
#[command(name = "vllm-server")]
#[command(version = "0.1.0")]
#[command(about = "High-performance LLM inference server", long_about = None)]
pub struct CliArgs {
    #[command(flatten)]
    server: ServerArgs,

    #[command(flatten)]
    pub model: ModelArgs,

    #[command(flatten)]
    engine: EngineArgs,

    #[command(flatten)]
    auth: AuthArgs,

    #[command(flatten)]
    logging: LoggingArgs,

    #[command(flatten)]
    config: ConfigArgs,
}

#[derive(clap::Args, Debug, Clone)]
#[group(id = "server_args")]
struct ServerArgs {
    #[arg(long, default_value = "0.0.0.0", env = "VLLM_HOST", global = true)]
    pub host: String,

    #[arg(long, default_value = "8000", env = "VLLM_PORT", short = 'p', value_parser = validate_port)]
    pub port: u16,
}

/// `ModelArgs`. See the type definition for fields and behavior.
#[derive(clap::Args, Debug, Clone)]
#[group(id = "model_args", required = true)]
pub struct ModelArgs {
    #[arg(long, required = true, env = "VLLM_MODEL", short = 'm')]
    pub model: PathBuf,

    /// Allow loading stub architectures that do not perform real inference.
    #[arg(long, default_value = "false", env = "VLLM_ALLOW_STUB")]
    pub allow_stub: bool,
}

#[derive(clap::Args, Debug, Clone)]
#[group(id = "engine_args")]
struct EngineArgs {
    #[arg(long, default_value = "1", env = "VLLM_TENSOR_PARALLEL_SIZE", short = 't', value_parser = validate_tensor_parallel_size)]
    pub tensor_parallel_size: usize,

    #[arg(long, default_value = "1024", env = "VLLM_KV_BLOCKS", value_parser = validate_kv_blocks)]
    pub kv_blocks: usize,

    #[arg(long, default_value = "false", env = "VLLM_KV_QUANTIZATION")]
    pub kv_quantization: bool,

    #[arg(long, default_value = "256", env = "VLLM_MAX_BATCH_SIZE", value_parser = validate_max_batch_size)]
    pub max_batch_size: usize,

    #[arg(long, default_value = "10", env = "VLLM_MAX_WAITING_BATCHES", value_parser = validate_max_waiting_batches)]
    pub max_waiting_batches: usize,

    #[arg(long, default_value = "8", env = "VLLM_MAX_DRAFT_TOKENS", value_parser = validate_max_draft_tokens)]
    pub max_draft_tokens: usize,

    #[arg(long, default_value = "false", env = "VLLM_ADAPTIVE_SPECULATIVE")]
    pub enable_adaptive_speculative: bool,
}

#[derive(clap::Args, Debug, Clone)]
#[group(id = "auth_args")]
struct AuthArgs {
    #[arg(long, env = "VLLM_API_KEY")]
    pub api_key: Vec<String>,

    #[arg(long, env = "VLLM_API_KEYS_FILE")]
    pub api_key_file: Option<PathBuf>,
}

#[derive(clap::Args, Debug, Clone)]
#[group(id = "logging_args")]
struct LoggingArgs {
    #[arg(long, default_value = "info", env = "VLLM_LOG_LEVEL", value_enum)]
    pub log_level: LogLevel,

    #[arg(long, env = "VLLM_LOG_DIR")]
    pub log_dir: Option<PathBuf>,
}

#[derive(clap::Args, Debug, Clone)]
#[group(id = "config_args")]
struct ConfigArgs {
    #[arg(long, short = 'c')]
    pub config: Option<PathBuf>,
}

impl CliArgs {
    #[must_use]
    pub fn to_app_config(&self) -> AppConfig {
        let mut config = AppConfig::load(self.config.config.clone());

        config.server.host.clone_from(&self.server.host);
        config.server.port = self.server.port;

        config.engine.tensor_parallel_size = self.engine.tensor_parallel_size;
        config.engine.num_kv_blocks = self.engine.kv_blocks;
        config.engine.kv_quantization = self.engine.kv_quantization;
        config.engine.max_batch_size = self.engine.max_batch_size;
        config.engine.max_waiting_batches = self.engine.max_waiting_batches;
        config.engine.max_draft_tokens = self.engine.max_draft_tokens;
        config.engine.enable_adaptive_speculative = self.engine.enable_adaptive_speculative;

        if !self.auth.api_key.is_empty() {
            config.auth.api_keys.clone_from(&self.auth.api_key);
        }
        if let Some(ref path) = self.auth.api_key_file {
            config.auth.api_keys_file = Some(path.to_string_lossy().to_string());
        }

        config.server.log_level = self.logging.log_level.to_string();
        config.server.log_dir = self
            .logging
            .log_dir
            .as_ref()
            .map(|p| p.to_string_lossy().to_string());

        config
    }

    #[must_use]
    pub const fn model_path(&self) -> &PathBuf {
        &self.model.model
    }
}

// Unit tests are extracted to `args/tests.rs` (a child module of
// `cli::args`) so they retain full access to CliArgs's private fields
// via `use super::*;`. Lifting them up one level to `cli/tests.rs`
// would require making these fields `pub(crate)` or `pub` — orthogonal
// to the file-size split.
#[cfg(test)]
mod tests;
