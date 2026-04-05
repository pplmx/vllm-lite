use crate::config::AppConfig;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "vllm-server")]
#[command(version = "0.1.0")]
#[command(about = "High-performance LLM inference server")]
pub struct CliArgs {
    // Server
    #[arg(long, default_value = "0.0.0.0", env = "VLLM_HOST")]
    pub host: String,

    #[arg(long, default_value = "8000", env = "VLLM_PORT", short = 'p')]
    pub port: u16,

    // Model - required
    #[arg(long, required = true, env = "VLLM_MODEL", short = 'm')]
    pub model: PathBuf,

    // Engine
    #[arg(
        long,
        default_value = "1",
        env = "VLLM_TENSOR_PARALLEL_SIZE",
        short = 't'
    )]
    pub tensor_parallel_size: usize,

    #[arg(long, default_value = "1024", env = "VLLM_KV_BLOCKS")]
    pub kv_blocks: usize,

    #[arg(long, default_value = "false", env = "VLLM_KV_QUANTIZATION")]
    pub kv_quantization: bool,

    #[arg(long, default_value = "256", env = "VLLM_MAX_BATCH_SIZE")]
    pub max_batch_size: usize,

    #[arg(long, default_value = "10", env = "VLLM_MAX_WAITING_BATCHES")]
    pub max_waiting_batches: usize,

    #[arg(long, default_value = "8", env = "VLLM_MAX_DRAFT_TOKENS")]
    pub max_draft_tokens: usize,

    // Auth
    #[arg(long, env = "VLLM_API_KEY")]
    pub api_key: Vec<String>,

    #[arg(long, env = "VLLM_API_KEYS_FILE")]
    pub api_key_file: Option<PathBuf>,

    // Logging
    #[arg(long, default_value = "info", env = "VLLM_LOG_LEVEL")]
    pub log_level: String,

    #[arg(long, env = "VLLM_LOG_DIR")]
    pub log_dir: Option<PathBuf>,

    // Config file
    #[arg(long, short = 'c')]
    pub config: Option<PathBuf>,
}

impl CliArgs {
    /// Convert CLI args to AppConfig, loading from config file first
    pub fn to_app_config(&self) -> AppConfig {
        // 1. Load from config file if specified
        let mut config = AppConfig::load(self.config.clone());

        // 2. CLI overrides config
        config.server.host = self.host.clone();
        config.server.port = self.port;

        config.engine.tensor_parallel_size = self.tensor_parallel_size;
        config.engine.num_kv_blocks = self.kv_blocks;
        config.engine.kv_quantization = self.kv_quantization;
        config.engine.max_batch_size = self.max_batch_size;
        config.engine.max_waiting_batches = self.max_waiting_batches;
        config.engine.max_draft_tokens = self.max_draft_tokens;

        // Auth - merge API keys
        if !self.api_key.is_empty() {
            config.auth.api_keys = self.api_key.clone();
        }
        if let Some(ref path) = self.api_key_file {
            config.auth.api_keys_file = Some(path.to_string_lossy().to_string());
        }

        // Logging
        config.server.log_level = self.log_level.clone();
        config.server.log_dir = self
            .log_dir
            .as_ref()
            .map(|p| p.to_string_lossy().to_string());

        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_defaults() {
        let cli = CliArgs::parse_from(["vllm-server", "-m", "/test/model"]);

        assert_eq!(cli.host, "0.0.0.0");
        assert_eq!(cli.port, 8000u16);
        assert_eq!(cli.tensor_parallel_size, 1usize);
        assert_eq!(cli.kv_blocks, 1024usize);
        assert_eq!(cli.max_batch_size, 256usize);
        assert_eq!(cli.max_waiting_batches, 10usize);
        assert_eq!(cli.max_draft_tokens, 8usize);
        assert!(!cli.kv_quantization);
        assert!(cli.api_key.is_empty());
        assert!(cli.api_key_file.is_none());
        assert_eq!(cli.log_level, "info");
        assert!(cli.log_dir.is_none());
        assert!(cli.config.is_none());
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

        assert_eq!(cli.host, "127.0.0.1");
        assert_eq!(cli.port, 9000u16);
        assert_eq!(cli.tensor_parallel_size, 4usize);
        assert_eq!(cli.kv_blocks, 2048usize);
        assert!(cli.kv_quantization);
        assert_eq!(cli.max_batch_size, 128usize);
        assert_eq!(cli.max_waiting_batches, 5usize);
        assert_eq!(cli.max_draft_tokens, 16usize);
    }

    #[test]
    fn test_cli_short_args() {
        let cli =
            CliArgs::parse_from(["vllm-server", "-m", "/test/model", "-p", "8080", "-t", "2"]);

        assert_eq!(cli.port, 8080u16);
        assert_eq!(cli.tensor_parallel_size, 2usize);
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

        assert_eq!(cli.api_key.len(), 2);
        assert_eq!(cli.api_key[0], "key1");
        assert_eq!(cli.api_key[1], "key2");
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

        assert!(cli.api_key_file.is_some());
        assert_eq!(cli.api_key_file.unwrap().to_string_lossy(), "/tmp/keys.txt");
    }

    #[test]
    fn test_cli_log_level() {
        let cli = CliArgs::parse_from(["vllm-server", "-m", "/test/model", "--log-level", "debug"]);

        assert_eq!(cli.log_level, "debug");
    }

    #[test]
    fn test_cli_log_dir() {
        let cli = CliArgs::parse_from([
            "vllm-server",
            "-m",
            "/test/model",
            "--log-dir",
            "/var/log/vllm",
        ]);

        assert!(cli.log_dir.is_some());
        assert_eq!(cli.log_dir.unwrap().to_string_lossy(), "/var/log/vllm");
    }

    #[test]
    fn test_cli_config_file() {
        let cli =
            CliArgs::parse_from(["vllm-server", "-m", "/test/model", "-c", "/tmp/config.yaml"]);

        assert!(cli.config.is_some());
        assert_eq!(cli.config.unwrap().to_string_lossy(), "/tmp/config.yaml");
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

    #[test]
    fn test_cli_version() {
        let result = CliArgs::try_parse_from(["vllm-server", "--version", "-m", "/test"]);
        // --version should either work or show version in error
        // clap handles this automatically
        assert!(result.is_ok() || result.unwrap_err().to_string().contains("0.1.0"));
    }

    #[test]
    fn test_cli_help() {
        let result = CliArgs::try_parse_from(["vllm-server", "--help", "-m", "/test"]);
        // --help should either work or show help in error
        assert!(result.is_ok() || result.unwrap_err().to_string().contains("Usage:"));
    }

    // Note: env var tests require unsafe and are skipped in regular test runs
    // They can be run manually with: cargo test -p vllm-server -- cli_env_var --nocapture -- --ignored

    #[test]
    fn test_config_file_option() {
        let cli = CliArgs::parse_from([
            "vllm-server",
            "-m",
            "/test/model",
            "--config",
            "/tmp/test.yaml",
        ]);

        assert!(cli.config.is_some());
    }

    #[test]
    fn test_model_path_is_pathbuf() {
        let cli = CliArgs::parse_from(["vllm-server", "-m", "/models/llama-7b"]);

        assert_eq!(cli.model.file_name().unwrap().to_string_lossy(), "llama-7b");
    }
}
