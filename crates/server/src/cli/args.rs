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

fn validate_max_model_len(s: &str) -> Result<usize, CliValidationError> {
    parse_usize_in_range(s, 1, 4_000_000)
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
    pub security: SecurityArgs,

    #[command(flatten)]
    logging: LoggingArgs,

    #[command(flatten)]
    config: ConfigArgs,

    /// OTLP collector endpoint override (e.g. `http://localhost:4317`).
    /// When set, enables the OTLP exporter and overrides the
    /// `observability.otlp.endpoint` YAML setting. Only available when the
    /// `opentelemetry` Cargo feature is enabled.
    #[cfg(feature = "opentelemetry")]
    #[arg(long, env = "VLLM_OTLP_ENDPOINT")]
    pub otlp_endpoint: Option<String>,
}

// RIL ISS-081: these args are `Option<T>` rather than defaulted so that
// `to_app_config` can distinguish "operator passed a flag / set env" from
// "operator left it alone". With hardcoded `default_value` strings, clap
// always filled the struct and `to_app_config` unconditionally overwrote
// the YAML file's values — a config with `kv_blocks: 4096` was silently
// reset to 1024. The effective defaults now come from one place
// (`AppConfig::default()`); `--help` documents the env var names.
#[derive(clap::Args, Debug, Clone)]
#[group(id = "server_args")]
struct ServerArgs {
    #[arg(
        long,
        env = "VLLM_HOST",
        global = true,
        help = "Bind address (default 0.0.0.0, or from --config YAML)"
    )]
    pub host: Option<String>,

    #[arg(long, env = "VLLM_PORT", short = 'p', value_parser = validate_port, help = "Listen port (default 8000, or from --config YAML)")]
    pub port: Option<u16>,
}

/// `ModelArgs`. See the type definition for fields and behavior.
///
/// The `required = true` lives on `--model` itself (not on the group):
/// with a required group clap renders `--allow-stub` as an alternative to
/// `--model` in the usage line (`<--model <MODEL>|--allow-stub>`), which
/// misleadingly suggests running with only `--allow-stub` (no model) is
/// valid. Dropping the group-level requirement keeps `--model` mandatory
/// while making the usage line unambiguous.
#[derive(clap::Args, Debug, Clone)]
#[group(id = "model_args")]
pub struct ModelArgs {
    #[arg(long, required = true, env = "VLLM_MODEL", short = 'm')]
    pub model: PathBuf,

    /// Allow loading stub architectures that do not perform real inference.
    #[arg(long, default_value = "false", env = "VLLM_ALLOW_STUB")]
    pub allow_stub: bool,
}

// RIL ISS-081: `Option<T>` (see the note on `ServerArgs`) so YAML values
// survive unless the operator explicitly overrides via flag or env var.
// NB: with Option fields the hardcoded defaults are gone, so
// `enable_adaptive_speculative` is no longer forced to `false` — it now
// honors the documented config default `true` when unspecified.
#[derive(clap::Args, Debug, Clone)]
#[group(id = "engine_args")]
struct EngineArgs {
    #[arg(long, env = "VLLM_TENSOR_PARALLEL_SIZE", short = 't', value_parser = validate_tensor_parallel_size, help = "Tensor-parallel degree (default 1, or from --config YAML)")]
    pub tensor_parallel_size: Option<usize>,

    #[arg(long, env = "VLLM_KV_BLOCKS", value_parser = validate_kv_blocks, help = "KV-cache blocks to allocate (default 1024, or from --config YAML)")]
    pub kv_blocks: Option<usize>,

    #[arg(
        long,
        env = "VLLM_KV_QUANTIZATION",
        default_missing_value = "true",
        num_args = 0..=1,
        help = "Enable FP8 KV-cache quantization (default false)"
    )]
    pub kv_quantization: Option<bool>,

    #[arg(long, env = "VLLM_MAX_BATCH_SIZE", value_parser = validate_max_batch_size, help = "Max batch size (default 256, or from --config YAML)")]
    pub max_batch_size: Option<usize>,

    #[arg(long, env = "VLLM_MAX_WAITING_BATCHES", value_parser = validate_max_waiting_batches, help = "Max waiting batches (default 10, or from --config YAML)")]
    pub max_waiting_batches: Option<usize>,

    #[arg(long, env = "VLLM_MAX_DRAFT_TOKENS", value_parser = validate_max_draft_tokens, help = "Max draft tokens per speculative step (default 8, or from --config YAML)")]
    pub max_draft_tokens: Option<usize>,

    #[arg(
        long,
        env = "VLLM_ADAPTIVE_SPECULATIVE",
        default_missing_value = "true",
        num_args = 0..=1,
        help = "Adaptive speculative decoding (default true, or from --config YAML)"
    )]
    pub enable_adaptive_speculative: Option<bool>,

    /// Explicit model context length (tokens). Overrides the checkpoint's
    /// `max_position_embeddings` (if declared) in request context-length
    /// validation. When neither is set, request validation caps `max_tokens`
    /// at a hard ceiling so an unknown-context model cannot be driven into an
    /// unbounded generation (RIL ISS-044).
    #[arg(long, env = "VLLM_MAX_MODEL_LEN", value_parser = validate_max_model_len)]
    pub max_model_len: Option<usize>,
}

#[derive(clap::Args, Debug, Clone)]
#[group(id = "auth_args")]
struct AuthArgs {
    #[arg(long, env = "VLLM_API_KEY")]
    pub api_key: Vec<String>,

    #[arg(long, env = "VLLM_API_KEYS_FILE")]
    pub api_key_file: Option<PathBuf>,
}

/// Security posture flags — currently just the SEC-01 escape hatch.
///
/// SEC-01 (technical due diligence): when the server binds to a
/// non-loopback address with no API keys configured, anyone reachable
/// on the network can hit `/v1/chat/completions`, `/debug/*`, and
/// `/shutdown`. The default is to print a loud warning; operators
/// running a known-internal single-tenant instance on `0.0.0.0`
/// can pass `--insecure-allow-public-no-auth` to silence it. We
/// deliberately do NOT fail closed: that would break local dev
/// and CI, and the cost of a wrong warning (operator annoyance)
/// is much lower than the cost of a wrong refusal (engineer can't
/// smoke-test). The choice is documented in OPERATIONS.md.
#[derive(clap::Args, Debug, Clone)]
#[group(id = "security_args")]
pub struct SecurityArgs {
    #[arg(
        long,
        default_value = "false",
        env = "VLLM_INSECURE_ALLOW_PUBLIC_NO_AUTH"
    )]
    pub insecure_allow_public_no_auth: bool,
}

#[derive(clap::Args, Debug, Clone)]
#[group(id = "logging_args")]
struct LoggingArgs {
    #[arg(
        long,
        env = "VLLM_LOG_LEVEL",
        value_enum,
        help = "Log level (default info, or from --config YAML)"
    )]
    pub log_level: Option<LogLevel>,

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

        // RIL ISS-081: only override a field when the operator actually
        // specified it through a flag or env var (`Some`). The `Option<T>`
        // arg types make an unset value distinguishable from a defaulted one,
        // so values from the `--config` YAML (and `AppConfig::default()`)
        // survive unless explicitly overridden.
        if let Some(ref host) = self.server.host {
            config.server.host.clone_from(host);
        }
        if let Some(port) = self.server.port {
            config.server.port = port;
        }

        if let Some(tensor_parallel_size) = self.engine.tensor_parallel_size {
            config.engine.tensor_parallel_size = tensor_parallel_size;
        }
        if let Some(num_kv_blocks) = self.engine.kv_blocks {
            config.engine.num_kv_blocks = num_kv_blocks;
        }
        if let Some(kv_quantization) = self.engine.kv_quantization {
            config.engine.kv_quantization = kv_quantization;
        }
        if let Some(max_batch_size) = self.engine.max_batch_size {
            config.engine.max_batch_size = max_batch_size;
        }
        if let Some(max_waiting_batches) = self.engine.max_waiting_batches {
            config.engine.max_waiting_batches = max_waiting_batches;
        }
        if let Some(max_draft_tokens) = self.engine.max_draft_tokens {
            config.engine.max_draft_tokens = max_draft_tokens;
        }
        if let Some(enable_adaptive_speculative) = self.engine.enable_adaptive_speculative {
            config.engine.enable_adaptive_speculative = enable_adaptive_speculative;
        }
        if let Some(max_model_len) = self.engine.max_model_len {
            config.engine.max_model_len = Some(max_model_len);
        }

        if !self.auth.api_key.is_empty() {
            config.auth.api_keys.clone_from(&self.auth.api_key);
        }
        if let Some(ref path) = self.auth.api_key_file {
            config.auth.api_keys_file = Some(path.to_string_lossy().to_string());
        }

        if let Some(ref log_level) = self.logging.log_level {
            config.server.log_level = log_level.to_string();
        }
        config.server.log_dir = self
            .logging
            .log_dir
            .as_ref()
            .map(|p| p.to_string_lossy().to_string());

        // P43 T5: --otlp-endpoint overrides the YAML `observability.otlp`
        // section and implicitly enables the exporter. Only compiled when
        // the `opentelemetry` feature is on the server crate.
        #[cfg(feature = "opentelemetry")]
        if let Some(endpoint) = self.otlp_endpoint.as_ref() {
            config.observability.otlp.enabled = true;
            config.observability.otlp.endpoint.clone_from(endpoint);
        }

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
