# CLI Design Specification

**Date**: 2026-04-05
**Goal:** Comprehensive command-line interface using clap with full feature parity

---

## 1. Overview

### 1.1 Design Philosophy

- Follow vLLM upstream CLI patterns
- Support config file + CLI args + env vars (config wins)
- Provide sensible defaults
- Add `--help` with comprehensive usage info
- Use subcommands for grouping related options

### 1.2 CLI vs Config vs Env Priority

```
CLI args > Env vars > Config file > Defaults
```

---

## 2. CLI Structure

### 2.1 Main Command

```bash
vllm-server [OPTIONS]
```

### 2.2 Option Groups

```
Server Options:
  --host <IP>                 Bind address (default: 0.0.0.0)
  --port <PORT>               Bind port (default: 8000)
  --timeout <SECS>            Request timeout (default: 300)

Model Options:
  --model <PATH>              Model path (required)
  --tokenizer <PATH>          Tokenizer path (default: <model>/tokenizer.json)
  --trust-remote-code         Trust remote code in tokenizer

Engine Options:
  --tensor-parallel-size <N>  GPU count for TP (default: 1)
  --kv-blocks <N>             KV cache blocks (default: 1024)
  --kv-quantization           Enable KV cache INT8 quantization
  --max-batch-size <N>        Max batch size (default: 256)
  --max-waiting-batches <N>   Max waiting batches (default: 10)
  --max-model-len <N>         Max model sequence length
  --max-draft-tokens <N>      Max speculative draft tokens (default: 8)
  --enforce-eager             Disable CUDA graph (debug)

Authentication Options:
  --api-key <KEY>             API key (can be repeated)
  --api-key-file <PATH>       File with API keys (one per line)
  --api-keys-env <VAR>        Env var containing comma-separated keys

Rate Limiting Options:
  --rate-limit <N>            Max requests per window (default: 100)
  --rate-limit-window <SECS>  Rate limit window (default: 60)

Logging Options:
  --log-level <LEVEL>         Log level: trace,debug,info,warn,error (default: info)
  --log-dir <PATH>            Log directory
  --disable-log-stdout        Disable stdout logging

Config Options:
  --config <PATH>             Config file (YAML)

Utility Options:
  --version                   Show version
  --help                      Show help
```

---

## 3. Implementation Design

### 3.1 Dependencies

```toml
# Cargo.toml
[dependencies]
clap = { version = "4", features = ["derive", "env"] }
```

### 3.2 CLI Args Struct

```rust
use clap::{Parser, Args};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "vllm-server")]
#[command(version = "0.1.0")]
#[command(about = "High-performance LLM inference server", long_about = None)]
pub struct CliArgs {
    /// Server configuration
    #[command(flatten)]
    pub server: ServerArgs,

    /// Model configuration
    #[command(flatten)]
    pub model: ModelArgs,

    /// Engine configuration
    #[command(flatten)]
    pub engine: EngineArgs,

    /// Authentication configuration
    #[command(flatten)]
    pub auth: AuthArgs,

    /// Rate limiting configuration
    #[command(flatten)]
    pub rate_limit: RateLimitArgs,

    /// Logging configuration
    #[command(flatten)]
    pub logging: LoggingArgs,

    /// Config file
    #[clap(short, long)]
    pub config: Option<PathBuf>,
}

#[derive(Args, Debug, Clone)]
#[group(required = false)]
pub struct ServerArgs {
    /// Server host
    #[clap(long, default_value = "0.0.0.0", env = "VLLM_HOST")]
    pub host: String,

    /// Server port
    #[clap(long, default_value_t = 8000, env = "VLLM_PORT")]
    pub port: u16,

    /// Request timeout in seconds
    #[clap(long, default_value_t = 300, env = "VLLM_TIMEOUT")]
    pub timeout: u64,
}

#[derive(Args, Debug, Clone)]
#[group(required = true)]
pub struct ModelArgs {
    /// Model path (directory or HF hub ID)
    #[clap(long)]
    pub model: PathBuf,

    /// Tokenizer path (default: <model>/tokenizer.json)
    #[clap(long)]
    pub tokenizer: Option<PathBuf>,

    /// Trust remote code in tokenizer
    #[clap(long, default_value = "false")]
    pub trust_remote_code: bool,
}

#[derive(Args, Debug, Clone)]
pub struct EngineArgs {
    /// Tensor parallel size
    #[clap(long, default_value_t = 1, env = "VLLM_TENSOR_PARALLEL_SIZE")]
    pub tensor_parallel_size: usize,

    /// KV cache blocks
    #[clap(long, default_value_t = 1024, env = "VLLM_KV_BLOCKS")]
    pub kv_blocks: usize,

    /// Enable KV cache quantization
    #[clap(long, default_value = "false", env = "VLLM_KV_QUANTIZATION")]
    pub kv_quantization: bool,

    /// Max batch size
    #[clap(long, default_value_t = 256, env = "VLLM_MAX_BATCH_SIZE")]
    pub max_batch_size: usize,

    /// Max waiting batches
    #[clap(long, default_value_t = 10, env = "VLLM_MAX_WAITING_BATCHES")]
    pub max_waiting_batches: usize,

    /// Max model sequence length
    #[clap(long, env = "VLLM_MAX_MODEL_LEN")]
    pub max_model_len: Option<usize>,

    /// Max speculative draft tokens
    #[clap(long, default_value_t = 8, env = "VLLM_MAX_DRAFT_TOKENS")]
    pub max_draft_tokens: usize,

    /// Disable CUDA graph (for debugging)
    #[clap(long, default_value = "false")]
    pub enforce_eager: bool,
}

#[derive(Args, Debug, Clone)]
pub struct AuthArgs {
    /// API key (can be specified multiple times)
    #[clap(long, env = "VLLM_API_KEY")]
    pub api_key: Vec<String>,

    /// File containing API keys (one per line)
    #[clap(long, env = "VLLM_API_KEYS_FILE")]
    pub api_key_file: Option<PathBuf>,

    /// Environment variable with comma-separated API keys
    #[clap(long, env = "VLLM_API_KEYS_ENV")]
    pub api_keys_env: Option<String>,
}

#[derive(Args, Debug, Clone)]
pub struct RateLimitArgs {
    /// Max requests per window
    #[clap(long, default_value_t = 100, env = "VLLM_RATE_LIMIT")]
    pub rate_limit: usize,

    /// Rate limit window in seconds
    #[clap(long, default_value_t = 60, env = "VLLM_RATE_LIMIT_WINDOW")]
    pub rate_limit_window: u64,
}

#[derive(Args, Debug, Clone)]
pub struct LoggingArgs {
    /// Log level
    #[clap(long, default_value = "info", env = "VLLM_LOG_LEVEL", 
           value_parser = clap::builder::PossibleValuesParser::new(["trace", "debug", "info", "warn", "error"]))]
    pub log_level: String,

    /// Log directory
    #[clap(long, env = "VLLM_LOG_DIR")]
    pub log_dir: Option<PathBuf>,

    /// Disable stdout logging
    #[clap(long, default_value = "false")]
    pub disable_log_stdout: bool,
}
```

---

## 4. Config Merge Logic

```rust
impl CliArgs {
    /// Merge CLI args with config file
    /// Priority: CLI > Env > Config > Defaults
    pub fn to_app_config(&self, config_path: Option<PathBuf>) -> AppConfig {
        // 1. Load from config file
        let mut config = AppConfig::load(config_path);

        // 2. Override with CLI args
        config.server.host = self.server.host.clone();
        config.server.port = self.server.port;
        
        config.engine.tensor_parallel_size = self.engine.tensor_parallel_size;
        config.engine.num_kv_blocks = self.engine.kv_blocks;
        config.engine.kv_quantization = self.engine.kv_quantization;
        config.engine.max_batch_size = self.engine.max_batch_size;
        config.engine.max_waiting_batches = self.engine.max_waiting_batches;
        config.engine.max_draft_tokens = self.engine.max_draft_tokens;
        
        config.server.log_level = self.logging.log_level.clone();
        config.server.log_dir = self.logging.log_dir.clone();
        
        // Auth: merge API keys
        if !self.auth.api_key.is_empty() {
            config.auth.api_keys = self.auth.api_key.clone();
        }
        if self.auth.api_key_file.is_some() {
            config.auth.api_keys_file = self.auth.api_key_file.clone();
        }
        if self.auth.api_keys_env.is_some() {
            config.auth.api_keys_env = self.auth.api_keys_env.clone();
        }
        
        config.auth.rate_limit_requests = self.rate_limit.rate_limit;
        config.auth.rate_limit_window_secs = self.rate_limit.rate_limit_window;

        config
    }
}
```

---

## 5. Help Output

```bash
$ vllm-server --help

vllm-server 0.1.0
High-performance LLM inference server

Usage: vllm-server [OPTIONS]

Server Options:
  --host <IP>                 Bind address [default: 0.0.0.0] [env: VLLM_HOST]
  --port <PORT>               Bind port [default: 8000] [env: VLLM_PORT]
  --timeout <SECS>            Request timeout [default: 300] [env: VLLM_TIMEOUT]

Model Options:
  --model <PATH>              Model path (directory or HF hub ID) [required]
  --tokenizer <PATH>          Tokenizer path [env: VLLM_TOKENIZER]
  --trust-remote-code         Trust remote code in tokenizer [default: false]

Engine Options:
  --tensor-parallel-size <N>  GPU count for tensor parallelism [default: 1] [env: VLLM_TENSOR_PARALLEL_SIZE]
  --kv-blocks <N>             KV cache block count [default: 1024] [env: VLLM_KV_BLOCKS]
  --kv-quantization           Enable KV cache INT8 quantization [env: VLLM_KV_QUANTIZATION]
  --max-batch-size <N>        Max batch size [default: 256] [env: VLLM_MAX_BATCH_SIZE]
  --max-waiting-batches <N>   Max waiting batches [default: 10] [env: VLLM_MAX_WAITING_BATCHES]
  --max-model-len <N>         Max model sequence length [env: VLLM_MAX_MODEL_LEN]
  --max-draft-tokens <N>      Max speculative draft tokens [default: 8] [env: VLLM_MAX_DRAFT_TOKENS]
  --enforce-eager             Disable CUDA graph (for debugging)

Authentication Options:
  --api-key <KEY>             API key (can be repeated) [env: VLLM_API_KEY]
  --api-key-file <PATH>       File with API keys [env: VLLM_API_KEYS_FILE]
  --api-keys-env <VAR>        Env var with comma-separated keys [env: VLLM_API_KEYS_ENV]

Rate Limiting Options:
  --rate-limit <N>            Max requests per window [default: 100] [env: VLLM_RATE_LIMIT]
  --rate-limit-window <SECS>  Window duration [default: 60] [env: VLLM_RATE_LIMIT_WINDOW]

Logging Options:
  --log-level <LEVEL>         Log level: trace,debug,info,warn,error [default: info] [env: VLLM_LOG_LEVEL]
  --log-dir <PATH>            Log directory [env: VLLM_LOG_DIR]
  --disable-log-stdout        Disable stdout logging

Config Options:
  --config <PATH>             Config file (YAML)

Examples:
  # Basic usage
  vllm-server --model /models/llama-7b

  # With custom port
  vllm-server --model /models/llama-7b --port 8080

  # With API key
  vllm-server --model /models/llama-7b --api-key sk-12345

  # From config file
  vllm-server --config config.yaml

  # With environment variables
  VLLM_PORT=8080 VLLM_API_KEY=sk-123 vllm-server --model /models/llama-7b

  # Multi-GPU tensor parallel
  vllm-server --model /models/llama-7b --tensor-parallel-size 4

  # Production with rate limiting
  vllm-server --model /models/llama-7b \
    --api-key-file /etc/vllm/keys.txt \
    --rate-limit 1000 \
    --rate-limit-window 60 \
    --log-level warn

See also:
  Config file format: https://github.com/vllm-lite/vllm-lite
  Environment variables: All options can be set via VLLM_<UPPER_CASE> env vars
```

---

## 6. Environment Variables

| CLI Option | Env Variable | Type | Default |
|------------|--------------|------|---------|
| `--host` | `VLLM_HOST` | string | "0.0.0.0" |
| `--port` | `VLLM_PORT` | u16 | 8000 |
| `--timeout` | `VLLM_TIMEOUT` | u64 | 300 |
| `--model` | `VLLM_MODEL` | path | (required) |
| `--tokenizer` | `VLLM_TOKENIZER` | path | auto |
| `--trust-remote-code` | `VLLM_TRUST_REMOTE_CODE` | bool | false |
| `--tensor-parallel-size` | `VLLM_TENSOR_PARALLEL_SIZE` | usize | 1 |
| `--kv-blocks` | `VLLM_KV_BLOCKS` | usize | 1024 |
| `--kv-quantization` | `VLLM_KV_QUANTIZATION` | bool | false |
| `--max-batch-size` | `VLLM_MAX_BATCH_SIZE` | usize | 256 |
| `--max-waiting-batches` | `VLLM_MAX_WAITING_BATCHES` | usize | 10 |
| `--max-model-len` | `VLLM_MAX_MODEL_LEN` | usize | auto |
| `--max-draft-tokens` | `VLLM_MAX_DRAFT_TOKENS` | usize | 8 |
| `--enforce-eager` | `VLLM_ENFORCE_EAGER` | bool | false |
| `--api-key` | `VLLM_API_KEY` | string | none |
| `--api-key-file` | `VLLM_API_KEYS_FILE` | path | none |
| `--api-keys-env` | `VLLM_API_KEYS_ENV` | string | none |
| `--rate-limit` | `VLLM_RATE_LIMIT` | usize | 100 |
| `--rate-limit-window` | `VLLM_RATE_LIMIT_WINDOW` | u64 | 60 |
| `--log-level` | `VLLM_LOG_LEVEL` | string | "info" |
| `--log-dir` | `VLLM_LOG_DIR` | path | none |
| `--config` | `VLLM_CONFIG` | path | none |

---

## 7. Config File Format

```yaml
# config.yaml
server:
  host: 0.0.0.0
  port: 8000
  timeout: 300

model:
  # model path
  model: /models/Qwen2.5-0.5B-Instruct
  # tokenizer path (optional)
  tokenizer: null
  # trust remote code
  trust_remote_code: false

engine:
  tensor_parallel_size: 1
  kv_blocks: 1024
  kv_quantization: false
  max_batch_size: 256
  max_waiting_batches: 10
  max_model_len: null
  max_draft_tokens: 8
  enforce_eager: false

auth:
  api_keys: []
  api_keys_env: null
  api_keys_file: null

rate_limit:
  requests: 100
  window_secs: 60

logging:
  log_level: info
  log_dir: null
  disable_log_stdout: false
```

---

## 8. Backward Compatibility

### 8.1 Old CLI Format (Keep Support)

```bash
# Old format still works
vllm-server --model /path/to/model --tensor-parallel-size 2
vllm-server --config=config.yaml
```

### 8.2 Config File Loading

- `--config` argument loads YAML config
- Config values are overridden by CLI args and env vars

---

## 9. Error Handling

### 9.1 Missing Required Args

```bash
$ vllm-server
error: the following required argument was not provided:
  --model <PATH>

Usage: vllm-server [OPTIONS]
```

### 9.2 Invalid Values

```bash
$ vllm-server --model /path --port 0
error: invalid value '0' for '--port <PORT>': number must be >= 1

$ vllm-server --model /path --log-level invalid
error: invalid value 'invalid' for '--log-level <LEVEL>': 
  possible values: trace, debug, info, warn, error
```

### 9.3 Config File Errors

```bash
$ vllm-server --config /nonexistent/config.yaml
warning: config file not found: /nonexistent/config.yaml
```

---

## 10. Files to Modify

| File | Changes |
|------|---------|
| `crates/server/Cargo.toml` | Add `clap` dependency |
| `crates/server/src/main.rs` | Use clap for CLI parsing |
| `crates/server/src/cli.rs` | New: CLI args definitions |
| `crates/server/src/config.rs` | Merge CLI into config |

---

## 11. Acceptance Criteria

- [ ] `--help` shows all options with descriptions
- [ ] `--version` shows version
- [ ] All config options available as CLI args
- [ ] All CLI args available as env vars
- [ ] Config file can be specified with `--config`
- [ ] CLI args override config file values
- [ ] Env vars override config file values
- [ ] Invalid values show helpful errors
- [ ] Required args (`--model`) enforced
- [ ] Backward compatible with existing usage
