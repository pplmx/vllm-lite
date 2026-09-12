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
//!   `--config`, `--model` `PathBuf`, `--log-dir`)
//! - `--log-level` value mapping (case-insensitive, full enum coverage)
//! - Range validation (port, kv-blocks, max-batch-size,
//!   max-waiting-batches, max-draft-tokens, tensor-parallel-size)
//! - `to_app_config()` field mapping (basic, all-fields, api-keys,
//!   log-dir)

use super::*;
use clap::{CommandFactory, Parser};

#[test]
fn test_cli_defaults() {
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
    // RIL ISS-081: unset CLI args are `None` (no hardcoded clap defaults), so
    // `to_app_config` can tell "operator left it alone" from "operator
    // overrode it" and preserve `--config` YAML / built-in defaults.
    let cli = CliArgs::parse_from(["vllm-server", "-m", "/test/model"]);

    assert!(cli.server.host.is_none());
    assert!(cli.server.port.is_none());
    assert!(cli.engine.tensor_parallel_size.is_none());
    assert!(cli.engine.kv_blocks.is_none());
    assert!(cli.engine.max_batch_size.is_none());
    assert!(cli.engine.max_waiting_batches.is_none());
    assert!(cli.engine.max_draft_tokens.is_none());
    assert!(cli.engine.kv_quantization.is_none());
    assert!(cli.engine.enable_adaptive_speculative.is_none());
    assert!(cli.auth.api_key.is_empty());
    assert!(cli.auth.api_key_file.is_none());
    assert!(cli.logging.log_level.is_none());
    assert!(cli.logging.log_dir.is_none());
    assert!(cli.config.config.is_none());
}

#[test]
fn test_cli_with_long_args() {
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
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

    assert_eq!(cli.server.host.as_deref(), Some("127.0.0.1"));
    assert_eq!(cli.server.port, Some(9000u16));
    assert_eq!(cli.engine.tensor_parallel_size, Some(4usize));
    assert_eq!(cli.engine.kv_blocks, Some(2048usize));
    assert_eq!(cli.engine.kv_quantization, Some(true));
    assert_eq!(cli.engine.max_batch_size, Some(128usize));
    assert_eq!(cli.engine.max_waiting_batches, Some(5usize));
    assert_eq!(cli.engine.max_draft_tokens, Some(16usize));
}

#[test]
fn test_cli_short_args() {
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
    let cli = CliArgs::parse_from(["vllm-server", "-m", "/test/model", "-p", "8080", "-t", "2"]);

    assert_eq!(cli.server.port, Some(8080u16));
    assert_eq!(cli.engine.tensor_parallel_size, Some(2usize));
}

#[test]
fn test_cli_required_model() {
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
    let result = CliArgs::try_parse_from(["vllm-server"]);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("--model"));
}

/// RIL TASK-107: the missing-`--model` message must not render
/// `--allow-stub` as an alternative to `--model` in the usage line
/// (`<--model <MODEL>|--allow-stub>` made it look like running with only
/// `--allow-stub`, no model, was valid). `--model` is required; `--allow-stub`
/// is an optional flag.
#[test]
fn test_cli_required_model_usage_does_not_offer_allow_stub_alternative() {
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
    let err_msg = CliArgs::try_parse_from(["vllm-server"])
        .unwrap_err()
        .to_string();
    assert!(
        err_msg.contains("--model"),
        "message must name --model: {err_msg}"
    );
    assert!(
        !err_msg.contains("|--allow-stub"),
        "usage must not present --allow-stub as an alternative to --model: {err_msg}"
    );
}

#[test]
fn test_cli_api_key_vec() {
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
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
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
    let cli = CliArgs::parse_from(["vllm-server", "-m", "/test/model", "--log-level", "debug"]);

    assert_eq!(cli.logging.log_level, Some(LogLevel::Debug));
}

#[test]
fn test_cli_log_level_case_insensitive() {
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
    let cli = CliArgs::parse_from(["vllm-server", "-m", "/test/model", "--log-level", "debug"]);

    assert_eq!(cli.logging.log_level, Some(LogLevel::Debug));
}

#[test]
fn test_cli_log_level_valid_values() {
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
    for level in ["trace", "debug", "info", "warn", "error"] {
        let cli = CliArgs::parse_from(["vllm-server", "-m", "/test/model", "--log-level", level]);
        assert_eq!(cli.logging.log_level.unwrap().to_string(), level);
    }
}

#[test]
fn test_to_app_config_basic() {
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
    let cli = CliArgs::parse_from(["vllm-server", "-m", "/test/model", "-p", "9000"]);

    let config = cli.to_app_config();

    assert_eq!(config.server.port, 9000);
}

#[test]
fn test_to_app_config_all_fields() {
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
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

/// RIL ISS-044: `--max-model-len` must flow from CLI into
/// `EngineConfig.max_model_len` (absent the flag, it stays `None`).
#[test]
fn test_to_app_config_max_model_len() {
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
    let cli = CliArgs::parse_from([
        "vllm-server",
        "-m",
        "/test/model",
        "--max-model-len",
        "32768",
    ]);
    let config = cli.to_app_config();
    assert_eq!(config.engine.max_model_len, Some(32768usize));

    let cli = CliArgs::parse_from(["vllm-server", "-m", "/test/model"]);
    let config = cli.to_app_config();
    assert_eq!(
        config.engine.max_model_len, None,
        "max_model_len must default to None so the checkpoint value is used"
    );
}

#[test]
fn test_to_app_config_with_api_keys() {
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
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
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
    let result = CliArgs::try_parse_from(["vllm-server", "--version", "-m", "/test"]);
    assert!(result.is_ok() || result.unwrap_err().to_string().contains("0.1.0"));
}

#[test]
fn test_cli_help() {
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
    let result = CliArgs::try_parse_from(["vllm-server", "--help", "-m", "/test"]);
    assert!(result.is_ok() || result.unwrap_err().to_string().contains("Usage:"));
}

#[test]
fn test_cli_config_file() {
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
    let cli = CliArgs::parse_from(["vllm-server", "-m", "/test/model", "-c", "/tmp/config.yaml"]);
    assert!(cli.config.config.is_some());
}

#[test]
fn test_model_path_is_pathbuf() {
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
    let cli = CliArgs::parse_from(["vllm-server", "-m", "/models/llama-7b"]);
    assert_eq!(
        cli.model.model.file_name().unwrap().to_string_lossy(),
        "llama-7b"
    );
}

#[test]
fn test_port_range_validation() {
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
    let result = CliArgs::try_parse_from(["vllm-server", "-m", "/test", "-p", "0"]);
    assert!(result.is_err());

    let result = CliArgs::try_parse_from(["vllm-server", "-m", "/test", "-p", "65535"]);
    assert!(result.is_ok());

    let result = CliArgs::try_parse_from(["vllm-server", "-m", "/test", "-p", "65536"]);
    assert!(result.is_err());
}

#[test]
fn test_kv_blocks_range_validation() {
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
    let result = CliArgs::try_parse_from(["vllm-server", "-m", "/test", "--kv-blocks", "0"]);
    assert!(result.is_err());

    let result = CliArgs::try_parse_from(["vllm-server", "-m", "/test", "--kv-blocks", "65536"]);
    assert!(result.is_ok());

    let result = CliArgs::try_parse_from(["vllm-server", "-m", "/test", "--kv-blocks", "65537"]);
    assert!(result.is_err());
}

#[test]
fn test_max_batch_size_range_validation() {
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
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
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
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
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
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
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
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
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
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
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
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

// ───────────────── RIL ISS-081: YAML-preservation regressions ─────────────────
//
// Every clap arg below carries a hardcoded `default_value`, so clap always fills
// the struct and `to_app_config` unconditionally overwrote the YAML-loaded
// values. Pre-fix an operator's `--- config.yaml` with `kv_blocks: 4096` was
// silently reset to the clap default 1024; `enable_adaptive_speculative: true`
// was even INVERTED to `false` (clap default) vs the documented config default.
// These two tests lock the corrected precedence: explicit CLI/env wins, YAML
// survives, and built-in defaults win otherwise.

/// RIL ISS-081: values set in the `--config` YAML file must survive
/// `to_app_config()` when no CLI flag overrides them.
#[test]
fn test_to_app_config_preserves_yaml_values() {
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
    let dir = tempfile::tempdir().expect("create tempdir");
    let config_path = dir.path().join("config.yaml");
    std::fs::write(
        &config_path,
        r#"
server:
  host: "127.0.0.1"
  port: 9999
  log_level: "debug"
engine:
  num_kv_blocks: 4096
  max_batch_size: 64
  max_draft_tokens: 16
  enable_adaptive_speculative: true
"#,
    )
    .expect("write config file");

    let cli = CliArgs::parse_from([
        "vllm-server",
        "-m",
        "/test/model",
        "-c",
        config_path.to_string_lossy().as_ref(),
    ]);
    let config = cli.to_app_config();

    assert_eq!(config.server.host, "127.0.0.1");
    assert_eq!(config.server.port, 9999);
    assert_eq!(config.server.log_level, "debug");
    assert_eq!(config.engine.num_kv_blocks, 4096);
    assert_eq!(config.engine.max_batch_size, 64);
    assert_eq!(config.engine.max_draft_tokens, 16);
    assert!(
        config.engine.enable_adaptive_speculative,
        "YAML true must not be inverted to false by the clap default"
    );
}

/// RIL ISS-081: an explicit CLI flag must STILL win over the `--config` YAML
/// value (precedence: CLI/env > YAML > defaults). This guards the override
/// direction after the clobber fix — removing the clobber must not also remove
/// legitimate flag-overrides-config behavior.
#[test]
fn test_to_app_config_flag_overrides_yaml() {
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
    let dir = tempfile::tempdir().expect("create tempdir");
    let config_path = dir.path().join("config.yaml");
    std::fs::write(
        &config_path,
        "engine:\n  num_kv_blocks: 4096\n  max_draft_tokens: 16\n",
    )
    .expect("write config file");

    let cli = CliArgs::parse_from([
        "vllm-server",
        "-m",
        "/test/model",
        "-c",
        config_path.to_string_lossy().as_ref(),
        "--kv-blocks",
        "2048",
        "--max-draft-tokens",
        "32",
    ]);
    let config = cli.to_app_config();

    assert_eq!(
        config.engine.num_kv_blocks, 2048,
        "explicit --kv-blocks must override the YAML value"
    );
    assert_eq!(
        config.engine.max_draft_tokens, 32,
        "explicit --max-draft-tokens must override the YAML value"
    );
}

/// RIL ISS-081: a bare `--enable-adaptive-speculative` flag still means
/// `true`, and `--enable-adaptive-speculative=false` still means `false`
/// (the `default_missing_value`/`num_args=0..=1` bool-flag pattern must not
/// have changed flag semantics).
#[test]
fn test_adaptive_speculative_bool_flag_semantics() {
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
    let on = CliArgs::parse_from([
        "vllm-server",
        "-m",
        "/test/model",
        "--enable-adaptive-speculative",
    ]);
    assert_eq!(on.engine.enable_adaptive_speculative, Some(true));

    let off = CliArgs::parse_from([
        "vllm-server",
        "-m",
        "/test/model",
        "--enable-adaptive-speculative=false",
    ]);
    assert_eq!(off.engine.enable_adaptive_speculative, Some(false));

    let unset = CliArgs::parse_from(["vllm-server", "-m", "/test/model"]);
    assert!(unset.engine.enable_adaptive_speculative.is_none());
}

/// RIL ISS-081: with neither a config file nor CLI overrides, the resolved
/// config must equal `AppConfig::default()`. In particular
/// `engine.enable_adaptive_speculative` keeps its documented default (`true`),
/// not the clap flag's inverted `false`.
#[test]
fn test_to_app_config_keeps_config_defaults_when_unset() {
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
    let cli = CliArgs::parse_from(["vllm-server", "-m", "/test/model"]);
    let config = cli.to_app_config();
    let defaults = AppConfig::default();

    assert_eq!(config.server.host, defaults.server.host);
    assert_eq!(config.server.port, defaults.server.port);
    assert_eq!(config.server.log_level, defaults.server.log_level);
    assert_eq!(
        config.engine.num_kv_blocks, defaults.engine.num_kv_blocks,
        "no CLI override must not reset num_kv_blocks"
    );
    assert_eq!(
        config.engine.max_batch_size, defaults.engine.max_batch_size,
        "no CLI override must not reset max_batch_size"
    );
    assert!(
        config.engine.enable_adaptive_speculative,
        "documented default is adaptive=true; the clap flag's false must not force it"
    );
}

// ─────────────────── P43 T5: --otlp-endpoint CLI override ───────────────────

#[cfg(feature = "opentelemetry")]
#[test]
fn test_otlp_endpoint_defaults_to_none() {
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
    let cli = CliArgs::parse_from(["vllm-server", "-m", "/test/model"]);
    assert!(cli.otlp_endpoint.is_none());
}

#[cfg(feature = "opentelemetry")]
#[test]
fn test_otlp_endpoint_long_flag() {
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
    let cli = CliArgs::parse_from([
        "vllm-server",
        "-m",
        "/test/model",
        "--otlp-endpoint",
        "http://otlp-collector:4317",
    ]);
    assert_eq!(
        cli.otlp_endpoint,
        Some("http://otlp-collector:4317".to_string())
    );
}

#[cfg(feature = "opentelemetry")]
#[test]
fn test_to_app_config_otlp_endpoint_overrides_yaml() {
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
    let cli = CliArgs::parse_from([
        "vllm-server",
        "-m",
        "/test/model",
        "--otlp-endpoint",
        "http://otlp-collector:4317",
    ]);
    let config = cli.to_app_config();

    // --otlp-endpoint implicitly enables OTLP and overrides the endpoint.
    assert!(config.observability.otlp.enabled);
    assert_eq!(
        config.observability.otlp.endpoint,
        "http://otlp-collector:4317"
    );
}

#[cfg(feature = "opentelemetry")]
#[test]
fn test_to_app_config_otlp_endpoint_no_override_when_not_set() {
    // RIL ISS-109: serialize against sibling tests that read/write VLLM_*
    // env vars (config/tests.rs etc.) — clap `env = "VLLM_*"` args read
    // std::env at parse time, so a concurrent set_var races this reader
    // under `cargo test` (single process). Shared crate-wide lock; nextest
    // (per-process) is immune.
    let _env_guard = crate::test_fixtures::ENV_TEST_LOCK
        .lock()
        .expect("env lock");
    let cli = CliArgs::parse_from(["vllm-server", "-m", "/test/model"]);
    let config = cli.to_app_config();

    // Without --otlp-endpoint, OTLP stays disabled (default).
    assert!(!config.observability.otlp.enabled);
}

#[test]
fn version_flag_reports_cargo_version() {
    // RIL ISS-119: `--version` must report `CARGO_PKG_VERSION` (the
    // workspace-package version) — previously hardcoded "0.1.0", which
    // would go stale the moment the crate version is bumped.
    assert_eq!(
        SERVER_VERSION,
        env!("CARGO_PKG_VERSION"),
        "SERVER_VERSION must track the crate version"
    );
    let rendered = CliArgs::command().render_version();
    assert!(
        rendered.contains(env!("CARGO_PKG_VERSION")),
        "--version must contain the crate version; got: {rendered}"
    );
}

#[test]
fn every_help_visible_cli_arg_has_help_text() {
    // RIL ISS-115: `--help` is an operator-facing contract — a flag that
    // renders with a blank help section (pre-fix: `--model`, `--api-key`,
    // `--api-key-file`, `--insecure-allow-public-no-auth`, `--log-dir`,
    // `--config`) is effectively undocumented. Walk the flattened clap
    // Command and require non-empty help on every arg except clap's own
    // `--help` / `--version`. Building the Command reads no env, so no
    // ENV_TEST_LOCK is needed.
    let cmd = CliArgs::command();
    let undocumented: Vec<String> = cmd
        .get_arguments()
        .filter(|arg| !matches!(arg.get_id().as_str(), "help" | "version"))
        .filter(|arg| {
            arg.get_help()
                .is_none_or(|help| help.to_string().trim().is_empty())
        })
        .map(|arg| arg.get_id().to_string())
        .collect();
    assert!(
        undocumented.is_empty(),
        "CLI args render with no help text in --help: {undocumented:?}"
    );
}
