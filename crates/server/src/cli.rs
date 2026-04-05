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
        assert_eq!(cli.port, 8000);
        assert_eq!(cli.tensor_parallel_size, 1);
        assert_eq!(cli.kv_blocks, 1024);
        assert_eq!(cli.max_batch_size, 256);
    }

    #[test]
    fn test_cli_with_args() {
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
        ]);

        assert_eq!(cli.host, "127.0.0.1");
        assert_eq!(cli.port, 9000);
        assert_eq!(cli.tensor_parallel_size, 4);
        assert_eq!(cli.kv_blocks, 2048);
    }

    #[test]
    fn test_cli_short_args() {
        let cli =
            CliArgs::parse_from(["vllm-server", "-m", "/test/model", "-p", "8080", "-t", "2"]);

        assert_eq!(cli.port, 8080);
        assert_eq!(cli.tensor_parallel_size, 2);
    }

    #[test]
    fn test_cli_required_model() {
        let result = CliArgs::try_parse_from(["vllm-server"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_to_app_config() {
        let cli = CliArgs::parse_from(["vllm-server", "-m", "/test/model", "-p", "9000"]);

        let config = cli.to_app_config();

        assert_eq!(config.server.port, 9000);
    }
}
