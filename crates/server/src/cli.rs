use crate::config::AppConfig;
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

#[derive(Clone, Debug, ValueEnum, PartialEq)]
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
        LogLevel::Info
    }
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Trace => write!(f, "trace"),
            LogLevel::Debug => write!(f, "debug"),
            LogLevel::Info => write!(f, "info"),
            LogLevel::Warn => write!(f, "warn"),
            LogLevel::Error => write!(f, "error"),
        }
    }
}

fn parse_usize_in_range(s: &str, min: usize, max: usize) -> Result<usize, String> {
    let val: usize = s
        .parse()
        .map_err(|_| format!("'{}' is not a valid number", s))?;
    if val < min || val > max {
        Err(format!("must be between {} and {}", min, max))
    } else {
        Ok(val)
    }
}

fn validate_port(s: &str) -> Result<u16, String> {
    let val: u16 = s
        .parse()
        .map_err(|_| format!("'{}' is not a valid port number", s))?;
    if val == 0 {
        Err("port must be between 1 and 65535".to_string())
    } else {
        Ok(val)
    }
}

fn validate_tensor_parallel_size(s: &str) -> Result<usize, String> {
    parse_usize_in_range(s, 1, 64)
}

fn validate_kv_blocks(s: &str) -> Result<usize, String> {
    parse_usize_in_range(s, 1, 65536)
}

fn validate_max_batch_size(s: &str) -> Result<usize, String> {
    parse_usize_in_range(s, 1, 8192)
}

fn validate_max_waiting_batches(s: &str) -> Result<usize, String> {
    parse_usize_in_range(s, 1, 100)
}

fn validate_max_draft_tokens(s: &str) -> Result<usize, String> {
    parse_usize_in_range(s, 0, 64)
}

#[derive(Parser, Debug)]
#[command(name = "vllm-server")]
#[command(version = "0.1.0")]
#[command(about = "High-performance LLM inference server", long_about = None)]
pub struct CliArgs {
    #[command(flatten)]
    server: ServerArgs,

    #[command(flatten)]
    model: ModelArgs,

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

#[derive(clap::Args, Debug, Clone)]
#[group(id = "model_args", required = true)]
struct ModelArgs {
    #[arg(long, required = true, env = "VLLM_MODEL", short = 'm')]
    pub model: PathBuf,
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
    pub fn to_app_config(&self) -> AppConfig {
        let mut config = AppConfig::load(self.config.config.clone());

        config.server.host = self.server.host.clone();
        config.server.port = self.server.port;

        config.engine.tensor_parallel_size = self.engine.tensor_parallel_size;
        config.engine.num_kv_blocks = self.engine.kv_blocks;
        config.engine.kv_quantization = self.engine.kv_quantization;
        config.engine.max_batch_size = self.engine.max_batch_size;
        config.engine.max_waiting_batches = self.engine.max_waiting_batches;
        config.engine.max_draft_tokens = self.engine.max_draft_tokens;

        if !self.auth.api_key.is_empty() {
            config.auth.api_keys = self.auth.api_key.clone();
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

    pub fn model_path(&self) -> &PathBuf {
        &self.model.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_defaults() {
        let cli = CliArgs::parse_from(["vllm-server", "-m", "/test/model"]);

        assert_eq!(cli.server.host, "0.0.0.0");
        assert_eq!(cli.server.port, 8000u16);
        assert_eq!(cli.engine.tensor_parallel_size, 1usize);
        assert_eq!(cli.engine.kv_blocks, 1024usize);
        assert_eq!(cli.engine.max_batch_size, 256usize);
        assert_eq!(cli.engine.max_waiting_batches, 10usize);
        assert_eq!(cli.engine.max_draft_tokens, 8usize);
        assert!(!cli.engine.kv_quantization);
        assert!(cli.auth.api_key.is_empty());
        assert!(cli.auth.api_key_file.is_none());
        assert_eq!(cli.logging.log_level, LogLevel::Info);
        assert!(cli.logging.log_dir.is_none());
        assert!(cli.config.config.is_none());
    }

    #[test]
    fn test_cli_with_long_args() {
        let cli = CliArgs::parse_from([
            "vllm-server",
            "-m",
            "/test/model",
            "-p",
            "9000",
            "--host",
            "127.0.0.1",
            "--tensor-parallel-size",
            "4",
            "--kv-blocks",
            "2048",
            "--kv-quantization",
            "--max-batch-size",
            "128",
            "--max-waiting-batches",
            "5",
            "--max-draft-tokens",
            "16",
        ]);

        assert_eq!(cli.server.host, "127.0.0.1");
        assert_eq!(cli.server.port, 9000u16);
        assert_eq!(cli.engine.tensor_parallel_size, 4usize);
        assert_eq!(cli.engine.kv_blocks, 2048usize);
        assert!(cli.engine.kv_quantization);
        assert_eq!(cli.engine.max_batch_size, 128usize);
        assert_eq!(cli.engine.max_waiting_batches, 5usize);
        assert_eq!(cli.engine.max_draft_tokens, 16usize);
    }

    #[test]
    fn test_cli_short_args() {
        let cli =
            CliArgs::parse_from(["vllm-server", "-m", "/test/model", "-p", "8080", "-t", "2"]);

        assert_eq!(cli.server.port, 8080u16);
        assert_eq!(cli.engine.tensor_parallel_size, 2usize);
    }

    #[test]
    fn test_cli_required_model() {
        let result = CliArgs::try_parse_from(["vllm-server"]);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("--model"));
    }

    #[test]
    fn test_cli_api_key_vec() {
        let cli = CliArgs::parse_from([
            "vllm-server",
            "-m",
            "/test/model",
            "--api-key",
            "key1",
            "--api-key",
            "key2",
        ]);

        assert_eq!(cli.auth.api_key.len(), 2);
        assert_eq!(cli.auth.api_key[0], "key1");
        assert_eq!(cli.auth.api_key[1], "key2");
    }

    #[test]
    fn test_cli_log_level() {
        let cli = CliArgs::parse_from(["vllm-server", "-m", "/test/model", "--log-level", "debug"]);

        assert_eq!(cli.logging.log_level, LogLevel::Debug);
    }

    #[test]
    fn test_cli_log_level_case_insensitive() {
        let cli = CliArgs::parse_from(["vllm-server", "-m", "/test/model", "--log-level", "debug"]);

        assert_eq!(cli.logging.log_level, LogLevel::Debug);
    }

    #[test]
    fn test_cli_log_level_valid_values() {
        for level in ["trace", "debug", "info", "warn", "error"] {
            let cli =
                CliArgs::parse_from(["vllm-server", "-m", "/test/model", "--log-level", level]);
            assert_eq!(cli.logging.log_level.to_string(), level);
        }
    }

    #[test]
    fn test_to_app_config_basic() {
        let cli = CliArgs::parse_from(["vllm-server", "-m", "/test/model", "-p", "9000"]);

        let config = cli.to_app_config();

        assert_eq!(config.server.port, 9000);
    }

    #[test]
    fn test_to_app_config_all_fields() {
        let cli = CliArgs::parse_from([
            "vllm-server",
            "-m",
            "/test/model",
            "-p",
            "9000",
            "--host",
            "192.168.1.1",
            "--tensor-parallel-size",
            "4",
            "--kv-blocks",
            "2048",
            "--kv-quantization",
            "--max-batch-size",
            "128",
            "--max-waiting-batches",
            "5",
            "--max-draft-tokens",
            "10",
            "--log-level",
            "debug",
        ]);

        let config = cli.to_app_config();

        assert_eq!(config.server.host, "192.168.1.1");
        assert_eq!(config.server.port, 9000);
        assert_eq!(config.engine.tensor_parallel_size, 4);
        assert_eq!(config.engine.num_kv_blocks, 2048);
        assert!(config.engine.kv_quantization);
        assert_eq!(config.engine.max_batch_size, 128);
        assert_eq!(config.engine.max_waiting_batches, 5);
        assert_eq!(config.engine.max_draft_tokens, 10);
        assert_eq!(config.server.log_level, "debug");
    }

    #[test]
    fn test_to_app_config_with_api_keys() {
        let cli = CliArgs::parse_from([
            "vllm-server",
            "-m",
            "/test/model",
            "--api-key",
            "sk-test-key",
        ]);

        let config = cli.to_app_config();

        assert_eq!(config.auth.api_keys.len(), 1);
        assert_eq!(config.auth.api_keys[0], "sk-test-key");
    }

    #[test]
    fn test_cli_version() {
        let result = CliArgs::try_parse_from(["vllm-server", "--version", "-m", "/test"]);
        assert!(result.is_ok() || result.unwrap_err().to_string().contains("0.1.0"));
    }

    #[test]
    fn test_cli_help() {
        let result = CliArgs::try_parse_from(["vllm-server", "--help", "-m", "/test"]);
        assert!(result.is_ok() || result.unwrap_err().to_string().contains("Usage:"));
    }

    #[test]
    fn test_cli_config_file() {
        let cli =
            CliArgs::parse_from(["vllm-server", "-m", "/test/model", "-c", "/tmp/config.yaml"]);
        assert!(cli.config.config.is_some());
    }

    #[test]
    fn test_model_path_is_pathbuf() {
        let cli = CliArgs::parse_from(["vllm-server", "-m", "/models/llama-7b"]);
        assert_eq!(
            cli.model.model.file_name().unwrap().to_string_lossy(),
            "llama-7b"
        );
    }

    #[test]
    fn test_port_range_validation() {
        let result = CliArgs::try_parse_from(["vllm-server", "-m", "/test", "-p", "0"]);
        assert!(result.is_err());

        let result = CliArgs::try_parse_from(["vllm-server", "-m", "/test", "-p", "65535"]);
        assert!(result.is_ok());

        let result = CliArgs::try_parse_from(["vllm-server", "-m", "/test", "-p", "65536"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_kv_blocks_range_validation() {
        let result = CliArgs::try_parse_from(["vllm-server", "-m", "/test", "--kv-blocks", "0"]);
        assert!(result.is_err());

        let result =
            CliArgs::try_parse_from(["vllm-server", "-m", "/test", "--kv-blocks", "65536"]);
        assert!(result.is_ok());

        let result =
            CliArgs::try_parse_from(["vllm-server", "-m", "/test", "--kv-blocks", "65537"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_max_batch_size_range_validation() {
        let result =
            CliArgs::try_parse_from(["vllm-server", "-m", "/test", "--max-batch-size", "0"]);
        assert!(result.is_err());

        let result =
            CliArgs::try_parse_from(["vllm-server", "-m", "/test", "--max-batch-size", "8192"]);
        assert!(result.is_ok());

        let result =
            CliArgs::try_parse_from(["vllm-server", "-m", "/test", "--max-batch-size", "8193"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_max_waiting_batches_range_validation() {
        let result =
            CliArgs::try_parse_from(["vllm-server", "-m", "/test", "--max-waiting-batches", "0"]);
        assert!(result.is_err());

        let result =
            CliArgs::try_parse_from(["vllm-server", "-m", "/test", "--max-waiting-batches", "100"]);
        assert!(result.is_ok());

        let result =
            CliArgs::try_parse_from(["vllm-server", "-m", "/test", "--max-waiting-batches", "101"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_max_draft_tokens_range_validation() {
        let result =
            CliArgs::try_parse_from(["vllm-server", "-m", "/test", "--max-draft-tokens", "0"]);
        assert!(result.is_err());

        let result =
            CliArgs::try_parse_from(["vllm-server", "-m", "/test", "--max-draft-tokens", "64"]);
        assert!(result.is_ok());

        let result =
            CliArgs::try_parse_from(["vllm-server", "-m", "/test", "--max-draft-tokens", "65"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_parallel_size_range_validation() {
        let result =
            CliArgs::try_parse_from(["vllm-server", "-m", "/test", "--tensor-parallel-size", "0"]);
        assert!(result.is_err());

        let result =
            CliArgs::try_parse_from(["vllm-server", "-m", "/test", "--tensor-parallel-size", "64"]);
        assert!(result.is_ok());

        let result =
            CliArgs::try_parse_from(["vllm-server", "-m", "/test", "--tensor-parallel-size", "65"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_cli_api_key_file() {
        let cli = CliArgs::parse_from([
            "vllm-server",
            "-m",
            "/test/model",
            "--api-key-file",
            "/tmp/keys.txt",
        ]);

        assert!(cli.auth.api_key_file.is_some());
        assert_eq!(
            cli.auth.api_key_file.unwrap().to_string_lossy(),
            "/tmp/keys.txt"
        );
    }

    #[test]
    fn test_to_app_config_with_log_dir() {
        let cli = CliArgs::parse_from([
            "vllm-server",
            "-m",
            "/test/model",
            "--log-dir",
            "/var/log/vllm",
        ]);

        let config = cli.to_app_config();

        assert_eq!(config.server.log_dir, Some("/var/log/vllm".to_string()));
    }
}
