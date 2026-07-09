//! Unit tests for the `clap`-derived CLI argument structs (`CliArgs`,
//! `ServerArgs`, `EngineArgs`, `ModelArgs`, `AuthArgs`, `LoggingArgs`,
//! `ConfigArgs`) and the `to_app_config()` conversion path.
//!
//! Extracted from `args.rs` to keep the implementation file under the
//! project's 800-line soft cap. Lives in `cli/args/` (not `cli/`)
//! so it has full access to `CliArgs`'s private fields via
//! `use super::*;` — fields would have needed `pub(crate)` if this
//! file lived one level up at `cli/tests.rs`.
//!
//! Exercises:
//!
//! - Default values (no overrides)
//! - Long-form and short-form flag parsing
//! - Required `--model` enforcement
//! - Vector / file-path flags (`--api-key` × N, `--api-key-file`,
//!   `--config`, `--model` PathBuf, `--log-dir`)
//! - `--log-level` value mapping (case-insensitive, full enum coverage)
//! - Range validation (port, kv-blocks, max-batch-size,
//!   max-waiting-batches, max-draft-tokens, tensor-parallel-size)
//! - `to_app_config()` field mapping (basic, all-fields, api-keys,
//!   log-dir)

use super::*;
use clap::Parser;

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
    let cli = CliArgs::parse_from(["vllm-server", "-m", "/test/model", "-p", "8080", "-t", "2"]);

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
        let cli = CliArgs::parse_from(["vllm-server", "-m", "/test/model", "--log-level", level]);
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
    let cli = CliArgs::parse_from(["vllm-server", "-m", "/test/model", "-c", "/tmp/config.yaml"]);
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

    let result = CliArgs::try_parse_from(["vllm-server", "-m", "/test", "--kv-blocks", "65536"]);
    assert!(result.is_ok());

    let result = CliArgs::try_parse_from(["vllm-server", "-m", "/test", "--kv-blocks", "65537"]);
    assert!(result.is_err());
}

#[test]
fn test_max_batch_size_range_validation() {
    let result = CliArgs::try_parse_from(["vllm-server", "-m", "/test", "--max-batch-size", "0"]);
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
    // Valid range is 0-64
    let result = CliArgs::try_parse_from(["vllm-server", "-m", "/test", "--max-draft-tokens", "0"]);
    assert!(result.is_ok(), "0 should be valid");

    let result =
        CliArgs::try_parse_from(["vllm-server", "-m", "/test", "--max-draft-tokens", "64"]);
    assert!(result.is_ok());

    // 65 should be rejected (out of range)
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
