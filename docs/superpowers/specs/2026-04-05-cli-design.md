# CLI Design Specification

**Date**: 2026-04-05
**Goal:** Comprehensive CLI using clap, aligned with vLLM upstream

---

## 1. Overview

### 1.1 Design Philosophy

- Follow vLLM upstream CLI patterns (https://docs.vllm.ai/en/latest/cli/serve/)
- Support config file + CLI args + env vars (CLI wins)
- Provide sensible defaults
- Add `--help` with comprehensive usage info
- Add short aliases for common options

### 1.2 CLI vs Config vs Env Priority

```
CLI args > Env vars > Config file > Defaults
```

---

## 2. CLI Structure (vLLM-style)

### 2.1 Main Command

```bash
vllm-server [OPTIONS]
```

### 2.2 Option Groups (Following vLLM Pattern)

```
Server (Frontend):
  --host <IP>                 Bind address [default: 0.0.0.0] [-h]
  --port <PORT>               Bind port [default: 8000] [-p]
  --timeout <SECS>            Request timeout [default: 300]
  --uvicorn-log-level <LEVEL> uvicorn log level [default: info]
  --disable-access-log        Disable request access log

Model:
  --model <PATH>              Model path or HF model name [required] [-m]
  --tokenizer <PATH>          Tokenizer path [default: <model>/tokenizer.json]
  --tokenizer-mode <MODE>     Tokenizer mode: auto, hf, slow [default: auto]
  --trust-remote-code         Trust remote code in tokenizer
  --dtype <DTYPE>             Data type: auto, float16, bfloat16, float32 [default: auto]
  --max-model-len <N>         Max model sequence length (auto-detect if not set)
  --served-model-name <NAME>  Model name in API responses

Parallel:
  --tensor-parallel-size <N>  Tensor parallel size [default: 1] [-tp]
  --pipeline-parallel-size <N> Pipeline parallel size [default: 1] [-pp]

Cache:
  --gpu-memory-utilization <FLOAT>  GPU memory for KV cache [default: 0.9]
  --block-size <N>            KV cache block size [default: 16]
  --kv-cache-dtype <DTYPE>    KV cache dtype: auto, fp8, fp8_e4m3, int8 [default: auto]
  --max-num-seqs <N>          Max concurrent sequences [default: 256]

Scheduler:
  --max-batch-size <N>        Max batch size [default: 256]
  --max-waiting-batches <N>   Max waiting batches [default: 10]
  --enforce-eager             Disable CUDA graph
  --disable-sliding-window    Disable sliding window

Speculative Decoding:
  --max-draft-tokens <N>      Max speculative draft tokens [default: 8]

Auth:
  --api-key <KEY>             API key (can be repeated)
  --api-key-file <PATH>       File with API keys (one per line)

Rate Limiting:
  --limit-requests-per-minute <N>  Requests per minute [default: 100]
  --limit-tokens-per-minute <N>    Tokens per minute

Logging:
  --log-level <LEVEL>         Log level: trace,debug,info,warn,error [default: info]
  --log-dir <PATH>            Log directory
  --disable-log-stdout        Disable stdout logging

Config:
  --config <PATH>             Config file (YAML)

Utility:
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

### 3.2 CLI Args Struct (vLLM-aligned)

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

    /// Parallel configuration
    #[command(flatten)]
    pub parallel: ParallelArgs,

    /// Cache configuration
    #[command(flatten)]
    pub cache: CacheArgs,

    /// Scheduler configuration
    #[command(flatten)]
    pub scheduler: SchedulerArgs,

    /// Speculative decoding
    #[command(flatten)]
    pub speculative: SpeculativeArgs,

    /// Authentication
    #[command(flatten)]
    pub auth: AuthArgs,

    /// Rate limiting
    #[command(flatten)]
    pub rate_limit: RateLimitArgs,

    /// Logging
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
    #[clap(long, default_value = "0.0.0.0", env = "VLLM_HOST", short = 'h')]
    pub host: String,

    /// Server port
    #[clap(long, default_value_t = 8000, env = "VLLM_PORT", short = 'p')]
    pub port: u16,

    /// Request timeout in seconds
    #[clap(long, default_value_t = 300, env = "VLLM_TIMEOUT")]
    pub timeout: u64,

    /// uvicorn log level
    #[clap(long, default_value = "info", env = "VLLM_UVICORN_LOG_LEVEL")]
    pub uvicorn_log_level: String,

    /// Disable access log
    #[clap(long, default_value = "false", env = "VLLM_DISABLE_ACCESS_LOG")]
    pub disable_access_log: bool,
}

#[derive(Args, Debug, Clone)]
#[group(required = true)]
pub struct ModelArgs {
    /// Model path (directory or HF hub ID)
    #[clap(long, env = "VLLM_MODEL", short = 'm')]
    pub model: PathBuf,

    /// Tokenizer path
    #[clap(long, env = "VLLM_TOKENIZER")]
    pub tokenizer: Option<PathBuf>,

    /// Tokenizer mode
    #[clap(long, default_value = "auto", env = "VLLM_TOKENIZER_MODE")]
    pub tokenizer_mode: String,

    /// Trust remote code in tokenizer
    #[clap(long, default_value = "false", env = "VLLM_TRUST_REMOTE_CODE")]
    pub trust_remote_code: bool,

    /// Data type
    #[clap(long, default_value = "auto", env = "VLLM_DTYPE")]
    pub dtype: String,

    /// Max model sequence length
    #[clap(long, env = "VLLM_MAX_MODEL_LEN")]
    pub max_model_len: Option<usize>,

    /// Served model name
    #[clap(long, env = "VLLM_SERVED_MODEL_NAME")]
    pub served_model_name: Option<String>,
}

#[derive(Args, Debug, Clone)]
pub struct ParallelArgs {
    /// Tensor parallel size
    #[clap(long, default_value_t = 1, env = "VLLM_TENSOR_PARALLEL_SIZE", short = 'tp')]
    pub tensor_parallel_size: usize,

    /// Pipeline parallel size
    #[clap(long, default_value_t = 1, env = "VLLM_PIPELINE_PARALLEL_SIZE", short = 'pp')]
    pub pipeline_parallel_size: usize,
}

#[derive(Args, Debug, Clone)]
pub struct CacheArgs {
    /// GPU memory utilization for KV cache
    #[clap(long, default_value_t = 0.9, env = "VLLM_GPU_MEMORY_UTILIZATION")]
    pub gpu_memory_utilization: f64,

    /// KV cache block size
    #[clap(long, default_value_t = 16, env = "VLLM_BLOCK_SIZE")]
    pub block_size: usize,

    /// KV cache dtype
    #[clap(long, default_value = "auto", env = "VLLM_KV_CACHE_DTYPE")]
    pub kv_cache_dtype: String,

    /// Max concurrent sequences
    #[clap(long, default_value_t = 256, env = "VLLM_MAX_NUM_SEQS")]
    pub max_num_seqs: usize,
}

#[derive(Args, Debug, Clone)]
pub struct SchedulerArgs {
    /// Max batch size
    #[clap(long, default_value_t = 256, env = "VLLM_MAX_BATCH_SIZE")]
    pub max_batch_size: usize,

    /// Max waiting batches
    #[clap(long, default_value_t = 10, env = "VLLM_MAX_WAITING_BATCHES")]
    pub max_waiting_batches: usize,

    /// Disable CUDA graph
    #[clap(long, default_value = "false", env = "VLLM_ENFORCE_EAGER")]
    pub enforce_eager: bool,

    /// Disable sliding window
    #[clap(long, default_value = "false", env = "VLLM_DISABLE_SLIDING_WINDOW")]
    pub disable_sliding_window: bool,
}

#[derive(Args, Debug, Clone)]
pub struct SpeculativeArgs {
    /// Max speculative draft tokens
    #[clap(long, default_value_t = 8, env = "VLLM_MAX_DRAFT_TOKENS")]
    pub max_draft_tokens: usize,
}

#[derive(Args, Debug, Clone)]
pub struct AuthArgs {
    /// API key (can be specified multiple times)
    #[clap(long, env = "VLLM_API_KEY")]
    pub api_key: Vec<String>,

    /// File containing API keys (one per line)
    #[clap(long, env = "VLLM_API_KEYS_FILE")]
    pub api_key_file: Option<PathBuf>,
}

#[derive(Args, Debug, Clone)]
pub struct RateLimitArgs {
    /// Max requests per minute
    #[clap(long, default_value_t = 100, env = "VLLM_LIMIT_REQUESTS_PER_MINUTE")]
    pub limit_requests_per_minute: usize,

    /// Max tokens per minute
    #[clap(long, env = "VLLM_LIMIT_TOKENS_PER_MINUTE")]
    pub limit_tokens_per_minute: Option<usize>,
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
        config.server.timeout = self.server.timeout;
        config.server.uvicorn_log_level = self.server.uvicorn_log_level.clone();

        config.model.model = self.model.model.clone();
        config.model.tokenizer = self.model.tokenizer.clone();
        config.model.trust_remote_code = self.model.trust_remote_code;
        config.model.dtype = self.model.dtype.clone();
        config.model.max_model_len = self.model.max_model_len;
        config.model.served_model_name = self.model.served_model_name.clone();

        config.parallel.tensor_parallel_size = self.parallel.tensor_parallel_size;
        config.parallel.pipeline_parallel_size = self.parallel.pipeline_parallel_size;

        config.cache.gpu_memory_utilization = self.cache.gpu_memory_utilization;
        config.cache.block_size = self.cache.block_size;
        config.cache.kv_cache_dtype = self.cache.kv_cache_dtype.clone();
        config.cache.max_num_seqs = self.cache.max_num_seqs;

        config.scheduler.max_batch_size = self.scheduler.max_batch_size;
        config.scheduler.max_waiting_batches = self.scheduler.max_waiting_batches;
        config.scheduler.enforce_eager = self.scheduler.enforce_eager;

        config.speculative.max_draft_tokens = self.speculative.max_draft_tokens;

        // Auth: merge API keys
        if !self.auth.api_key.is_empty() {
            config.auth.api_keys = self.auth.api_key.clone();
        }
        if self.auth.api_key_file.is_some() {
            config.auth.api_keys_file = self.auth.api_key_file.clone();
        }

        config.rate_limit.requests_per_minute = self.rate_limit.limit_requests_per_minute;
        config.rate_limit.tokens_per_minute = self.rate_limit.limit_tokens_per_minute;

        config.logging.log_level = self.logging.log_level.clone();
        config.logging.log_dir = self.logging.log_dir.clone();
        config.logging.disable_log_stdout = self.logging.disable_log_stdout;

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
  --host <IP>                 Bind address [default: 0.0.0.0] [-h] [env: VLLM_HOST]
  --port <PORT>               Bind port [default: 8000] [-p] [env: VLLM_PORT]
  --timeout <SECS>            Request timeout [default: 300] [env: VLLM_TIMEOUT]
  --uvicorn-log-level <LEVEL> uvicorn log level [default: info]

Model Options:
  --model <PATH>              Model path [required] [-m] [env: VLLM_MODEL]
  --tokenizer <PATH>          Tokenizer path [env: VLLM_TOKENIZER]
  --dtype <DTYPE>             Data type [default: auto] [env: VLLM_DTYPE]
  --max-model-len <N>         Max model length [env: VLLM_MAX_MODEL_LEN]
  --trust-remote-code         Trust remote code [env: VLLM_TRUST_REMOTE_CODE]

Parallel Options:
  --tensor-parallel-size <N>  Tensor parallel size [default: 1] [-tp]
  --pipeline-parallel-size <N> Pipeline parallel size [default: 1] [-pp]

Cache Options:
  --gpu-memory-utilization <F> GPU memory utilization [default: 0.9]
  --block-size <N>            KV cache block size [default: 16]
  --kv-cache-dtype <DTYPE>    KV cache dtype [default: auto]

Scheduler Options:
  --max-batch-size <N>        Max batch size [default: 256]
  --max-waiting-batches <N>   Max waiting batches [default: 10]
  --enforce-eager             Disable CUDA graph

Speculative Options:
  --max-draft-tokens <N>      Max draft tokens [default: 8]

Auth Options:
  --api-key <KEY>             API key [env: VLLM_API_KEY]
  --api-key-file <PATH>       API keys file [env: VLLM_API_KEYS_FILE]

Rate Limit Options:
  --limit-requests-per-minute <N>  Requests/min [default: 100]

Logging Options:
  --log-level <LEVEL>         Log level [default: info]
  --log-dir <PATH>            Log directory
  --disable-log-stdout        Disable stdout

Config Options:
  --config <PATH>             Config file (YAML)

Examples:
  # Basic usage
  vllm-server -m /models/llama-7b

  # With custom port
  vllm-server -m /models/llama-7b -p 8080

  # Multi-GPU
  vllm-server -m /models/llama-7b --tp 4

  # With API key
  vllm-server -m /models/llama-7b --api-key sk-12345

  # Production
  vllm-server -m /models/llama-7b \
    --api-key-file /etc/vllm/keys.txt \
    --gpu-memory-utilization 0.95 \
    --log-level warn

  # From config
  vllm-server --config config.yaml

  # Environment variables
  VLLM_PORT=8080 VLLM_API_KEY=sk-123 vllm-server -m /models/llama-7b
```

---

## 6. Environment Variables

| CLI Option | Short | Env Variable | Type | Default |
|------------|-------|--------------|------|---------|
| `--host` | `-h` | `VLLM_HOST` | string | "0.0.0.0" |
| `--port` | `-p` | `VLLM_PORT` | u16 | 8000 |
| `--model` | `-m` | `VLLM_MODEL` | path | required |
| `--tensor-parallel-size` | `-tp` | `VLLM_TENSOR_PARALLEL_SIZE` | usize | 1 |
| `--pipeline-parallel-size` | `-pp` | `VLLM_PIPELINE_PARALLEL_SIZE` | usize | 1 |
| `--gpu-memory-utilization` | | `VLLM_GPU_MEMORY_UTILIZATION` | f64 | 0.9 |
| `--block-size` | | `VLLM_BLOCK_SIZE` | usize | 16 |
| `--max-batch-size` | | `VLLM_MAX_BATCH_SIZE` | usize | 256 |
| `--max-waiting-batches` | | `VLLM_MAX_WAITING_BATCHES` | usize | 10 |
| `--max-model-len` | | `VLLM_MAX_MODEL_LEN` | usize | auto |
| `--max-draft-tokens` | | `VLLM_MAX_DRAFT_TOKENS` | usize | 8 |
| `--enforce-eager` | | `VLLM_ENFORCE_EAGER` | bool | false |
| `--api-key` | | `VLLM_API_KEY` | string | none |
| `--api-key-file` | | `VLLM_API_KEYS_FILE` | path | none |
| `--limit-requests-per-minute` | | `VLLM_LIMIT_REQUESTS_PER_MINUTE` | usize | 100 |
| `--log-level` | | `VLLM_LOG_LEVEL` | string | "info" |
| `--log-dir` | | `VLLM_LOG_DIR` | path | none |
| `--config` | | `VLLM_CONFIG` | path | none |

---

## 7. Config File Format

```yaml
# config.yaml (vLLM-style)
host: 0.0.0.0
port: 8000
timeout: 300

model:
  model: /models/Qwen2.5-0.5B-Instruct
  tokenizer: null
  tokenizer_mode: auto
  trust_remote_code: false
  dtype: auto
  max_model_len: null

parallel:
  tensor_parallel_size: 1
  pipeline_parallel_size: 1

cache:
  gpu_memory_utilization: 0.9
  block_size: 16
  kv_cache_dtype: auto
  max_num_seqs: 256

scheduler:
  max_batch_size: 256
  max_waiting_batches: 10
  enforce_eager: false

speculative:
  max_draft_tokens: 8

auth:
  api_keys: []
  api_keys_file: null

rate_limit:
  requests_per_minute: 100
  tokens_per_minute: null

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

---

## 9. Files to Modify

| File | Changes |
|------|---------|
| `crates/server/Cargo.toml` | Add `clap` dependency |
| `crates/server/src/main.rs` | Use clap for CLI parsing |
| `crates/server/src/cli.rs` | New: CLI args definitions |
| `crates/server/src/config.rs` | Merge CLI into config, add new fields |

---

## 10. Acceptance Criteria

- [ ] `--help` shows all options with descriptions
- [ ] `--version` shows version
- [ ] Short aliases work (-m, -p, -tp, -pp)
- [ ] All CLI args available as env vars
- [ ] Config file can be specified with `--config`
- [ ] CLI args override config file values
- [ ] Invalid values show helpful errors
- [ ] Required args (`--model`) enforced
- [ ] Backward compatible with existing usage
- [ ] gpu-memory-utilization works (vLLM style)
