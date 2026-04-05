# CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add comprehensive CLI support using clap with env var and config file support

**Architecture:** Use clap derive macros for CLI parsing, merge with existing config loading system. Flat structure for simplicity and Rust idioms.

**Tech Stack:** clap 4.x with derive and env features

---

## Overview

This plan adds clap-based CLI to replace the current manual argument parsing in main.rs. The implementation:
1. Adds clap dependency to Cargo.toml
2. Creates cli.rs with CliArgs struct
3. Updates main.rs to use clap
4. Updates config.rs to support CLI merging
5. Adds comprehensive tests

**Files to modify:**
- `crates/server/Cargo.toml` - add clap
- `crates/server/src/cli.rs` - new file
- `crates/server/src/main.rs` - use clap
- `crates/server/src/config.rs` - merge logic

---

## Task 1: Add clap dependency

**Files:**
- Modify: `crates/server/Cargo.toml`

- [ ] **Step 1: Add clap dependency**

Run: `cat crates/server/Cargo.toml`
Check current dependencies section, then add clap:

```toml
[dependencies]
# ... existing deps ...
clap = { version = "4", features = ["derive", "env"] }
```

- [ ] **Step 2: Verify dependency resolves**

Run: `cargo fetch -p vllm-server`
Expected: Downloads clap crate

- [ ] **Step 3: Commit**

```bash
git add crates/server/Cargo.toml
git commit -m "feat(server): add clap dependency for CLI"
```

---

## Task 2: Create CLI module

**Files:**
- Create: `crates/server/src/cli.rs`

- [ ] **Step 1: Create cli.rs with CliArgs**

```rust
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "vllm-server")]
#[command(version = "0.1.0")]
#[command(about = "High-performance LLM inference server")]
pub struct CliArgs {
    // Server
    #[arg(long, default_value = "0.0.0.0", env = "VLLM_HOST", short = 'h')]
    pub host: String,

    #[arg(long, default_value = "8000", env = "VLLM_PORT", short = 'p')]
    pub port: u16,

    // Model - required
    #[arg(long, required = true, env = "VLLM_MODEL", short = 'm')]
    pub model: PathBuf,

    // Engine
    #[arg(long, default_value = "1", env = "VLLM_TENSOR_PARALLEL_SIZE", short = 'tp')]
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

    #[arg(long, default_value = "false", env = "VLLM_ENFORCE_EAGER")]
    pub enforce_eager: bool,

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
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p vllm-server`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add crates/server/src/cli.rs
git commit -m "feat(server): add CLI module with clap"
```

---

## Task 3: Add to_app_config method

**Files:**
- Modify: `crates/server/src/cli.rs`

- [ ] **Step 1: Add to_app_config implementation**

Add this impl block to cli.rs after the struct definition:

```rust
use crate::config::AppConfig;

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
        config.engine.enforce_eager = self.enforce_eager;

        // Auth - merge API keys
        if !self.api_key.is_empty() {
            config.auth.api_keys = self.api_key.clone();
        }
        if self.api_key_file.is_some() {
            config.auth.api_keys_file = self.api_key_file.clone();
        }

        // Logging
        config.server.log_level = self.log_level.clone();
        config.server.log_dir = self.log_dir.clone();

        config
    }
}
```

- [ ] **Step 2: Add module declaration**

Run: `head -10 crates/server/src/main.rs`
Add after the existing mod declarations:

```rust
mod cli;
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p vllm-server`
Expected: No errors (may need to add use statement)

- [ ] **Step 4: Commit**

```bash
git add crates/server/src/cli.rs crates/server/src/main.rs
git commit -m "feat(server): add to_app_config method"
```

---

## Task 4: Update main.rs to use clap

**Files:**
- Modify: `crates/server/src/main.rs`

- [ ] **Step 1: Replace manual arg parsing with clap**

Find and replace the load_config(), get_model_path(), get_tensor_parallel_size() functions:

```rust
// Replace old manual parsing with:
fn load_config() -> config::AppConfig {
    let cli = cli::CliArgs::parse();
    let config = cli.to_app_config();

    if let Err(errors) = config.validate() {
        for err in &errors {
            tracing::error!(error = %err, "Config validation failed");
        }
        eprintln!("Config validation failed:");
        for err in errors {
            eprintln!("  - {}", err);
        }
        std::process::exit(1);
    }

    config
}

fn get_model_path() -> String {
    // Now handled by clap - model is required
    // This function can be removed or simplified
    let cli = cli::CliArgs::parse();
    cli.model.to_string_lossy().to_string()
}
```

Actually, better approach - parse once at start:

```rust
// At start of main(), replace all the manual parsing:
fn main() {
    let cli = cli::CliArgs::parse();
    let app_config = cli.to_app_config();
    // ... rest of main uses app_config
}
```

- [ ] **Step 2: Remove old get_model_path and get_tensor_parallel_size functions**

Delete these functions that are no longer needed:
- get_model_path()
- get_tensor_parallel_size()

- [ ] **Step 3: Update references**

Update code that uses get_model_path() to use app_config fields directly.

Looking at main.rs, we need to find where model_path is used and change it to get from app_config.

- [ ] **Step 4: Build and fix errors**

Run: `cargo build -p vllm-server 2>&1`
Fix any compilation errors

- [ ] **Step 5: Commit**

```bash
git add crates/server/src/main.rs
git commit -m "refactor(server): use clap for CLI parsing"
```

---

## Task 5: Add CLI tests

**Files:**
- Create: `crates/server/src/cli.rs` (add tests module)

- [ ] **Step 1: Add test module to cli.rs**

Add at the end of cli.rs:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_defaults() {
        // Parse with required args
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
            "-m", "/test/model",
            "-p", "9000",
            "--host", "127.0.0.1",
            "--tensor-parallel-size", "4",
            "--kv-blocks", "2048",
        ]);
        
        assert_eq!(cli.host, "127.0.0.1");
        assert_eq!(cli.port, 9000);
        assert_eq!(cli.tensor_parallel_size, 4);
        assert_eq!(cli.kv_blocks, 2048);
    }

    #[test]
    fn test_cli_short_args() {
        let cli = CliArgs::parse_from([
            "vllm-server",
            "-m", "/test/model",
            "-p", "8080",
            "-h", "0.0.0.0",
            "-tp", "2",
        ]);
        
        assert_eq!(cli.port, 8080);
        assert_eq!(cli.host, "0.0.0.0");
        assert_eq!(cli.tensor_parallel_size, 2);
    }

    #[test]
    fn test_cli_required_model() {
        let result = CliArgs::try_parse_from(["vllm-server"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_to_app_config() {
        let cli = CliArgs::parse_from([
            "vllm-server",
            "-m", "/test/model",
            "-p", "9000",
            "--log-level", "debug",
        ]);
        
        let config = cli.to_app_config();
        
        assert_eq!(config.server.port, 9000);
        assert_eq!(config.server.log_level, "debug");
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p vllm-server -- cli`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add crates/server/src/cli.rs
git commit -m "test(server): add CLI tests"
```

---

## Task 6: Integration test

**Files:**
- Test: Full CLI workflow

- [ ] **Step 1: Test --help**

Run: `cargo run -- -p vllm-server -- --help`
Expected: Help output with all options shown

- [ ] **Step 2: Test --version**

Run: `cargo run -- -p vllm-server -- --version`
Expected: Shows "vllm-server 0.1.0"

- [ ] **Step 3: Test with config file**

Create test config:
```yaml
# /tmp/test-config.yaml
host: 127.0.0.1
port: 9999

model: /tmp/test-model

engine:
  tensor_parallel_size: 2
  kv_blocks: 512

logging:
  log_level: debug
```

Run: `cargo run -- -p vllm-server -- -c /tmp/test-config.yaml --help`
(Just verify config loads without error)

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "test(server): CLI integration tests"
```

---

## Task 7: Final verification

**Files:**
- Verify: Full workspace

- [ ] **Step 1: Run clippy**

Run: `cargo clippy -p vllm-server -- -D warnings`
Expected: No warnings

- [ ] **Step 2: Run tests**

Run: `cargo test -p vllm-server`
Expected: All tests pass

- [ ] **Step 3: Format**

Run: `cargo fmt --all`
Expected: No changes needed

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat(server): complete CLI implementation with clap"
```

---

## Acceptance Criteria

- [ ] `--help` shows all options with defaults
- [ ] `--version` shows version
- [ ] All short args work: -m, -p, -tp, -h, -c
- [ ] All CLI options have corresponding env vars
- [ ] `--config` loads YAML config
- [ ] CLI args override config file values
- [ ] Required `--model` enforced
- [ ] All existing functionality works
- [ ] Tests pass
- [ ] Clippy clean
