# CLI Design Specification

**Date**: 2026-04-05
**Goal:** Comprehensive CLI using clap (Rust-idiomatic, vLLM-inspired)

---

## 1. Overview

### 1.1 Design Philosophy

- Rust-idiomatic: use clap derive macros, simple flat structure
- vLLM-inspired: similar option names where applicable
- Only include options that are actually implemented
- CLI args > Env vars > Config file > Defaults

---

## 2. CLI Options (Implemented Features Only)

```
Server:
  --host <IP>                 Bind address [default: 0.0.0.0] [-h]
  --port <PORT>               Bind port [default: 8000] [-p]

Model:
  --model <PATH>              Model path [required] [-m]

Engine:
  --tensor-parallel-size <N>  Tensor parallel size [default: 1] [-tp]
  --kv-blocks <N>             KV cache blocks [default: 1024]
  --kv-quantization           Enable KV cache quantization
  --max-batch-size <N>        Max batch size [default: 256]
  --max-waiting-batches <N>   Max waiting batches [default: 10]
  --max-draft-tokens <N>      Max speculative draft tokens [default: 8]
  --enforce-eager             Disable CUDA graph

Auth:
  --api-key <KEY>             API key (can repeat)
  --api-key-file <PATH>       File with API keys

Logging:
  --log-level <LEVEL>         Log level [default: info]
  --log-dir <PATH>            Log directory

Config:
  --config <PATH>             Config file (YAML)

Utility:
  --version                   Show version
  --help                      Show help
```

---

## 3. Implementation

### 3.1 Cargo.toml

```toml
[dependencies]
clap = { version = "4", features = ["derive", "env"] }
```

### 3.2 CLI Struct (Simple, Flat)

```rust
use clap::{Parser};
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

    // Model
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

    // Config
    #[arg(long, short = 'c')]
    pub config: Option<PathBuf>,
}
```

---

## 4. Config Merge

```rust
impl CliArgs {
    pub fn to_app_config(&self) -> AppConfig {
        // Load from config file if specified
        let mut config = AppConfig::load(self.config.clone());

        // CLI overrides config
        config.server.host = self.host.clone();
        config.server.port = self.port;

        config.engine.tensor_parallel_size = self.tensor_parallel_size;
        config.engine.num_kv_blocks = self.kv_blocks;
        config.engine.kv_quantization = self.kv_quantization;
        config.engine.max_batch_size = self.max_batch_size;
        config.engine.max_waiting_batches = self.max_waiting_batches;
        config.engine.max_draft_tokens = self.max_draft_tokens;
        config.engine.enforce_eager = self.enforce_eager;

        if !self.api_key.is_empty() {
            config.auth.api_keys = self.api_key.clone();
        }
        if self.api_key_file.is_some() {
            config.auth.api_keys_file = self.api_key_file.clone();
        }

        config.server.log_level = self.log_level.clone();
        config.server.log_dir = self.log_dir.clone();

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

Options:
  -h, --host <IP>                 Bind address [default: 0.0.0.0]
  -p, --port <PORT>               Bind port [default: 8000]
  -m, --model <PATH>              Model path [required]
  -tp, --tensor-parallel-size <N> Tensor parallel size [default: 1]
  --kv-blocks <N>                 KV cache blocks [default: 1024]
  --kv-quantization               Enable KV quantization
  --max-batch-size <N>            Max batch size [default: 256]
  --max-waiting-batches <N>       Max waiting batches [default: 10]
  --max-draft-tokens <N>          Max draft tokens [default: 8]
  --enforce-eager                 Disable CUDA graph
  --api-key <KEY>                 API key (can repeat)
  --api-key-file <PATH>           API keys file
  --log-level <LEVEL>             Log level [default: info]
  --log-dir <PATH>                Log directory
  -c, --config <PATH>             Config file
  --version                       Show version
  --help                          Show help

Examples:
  vllm-server -m /models/llama-7b
  vllm-server -m /models/llama-7b -p 8080
  vllm-server -m /models/llama-7b --tp 4
  vllm-server -m /models/llama-7b --api-key sk-123
  vllm-server -m /models/llama-7b -c config.yaml
  VLLM_PORT=8080 vllm-server -m /models/llama-7b
```

---

## 6. Environment Variables

| Option | Short | Env | Default |
|--------|-------|-----|---------|
| `--host` | `-h` | `VLLM_HOST` | "0.0.0.0" |
| `--port` | `-p` | `VLLM_PORT` | 8000 |
| `--model` | `-m` | `VLLM_MODEL` | required |
| `--tensor-parallel-size` | `-tp` | `VLLM_TENSOR_PARALLEL_SIZE` | 1 |
| `--kv-blocks` | | `VLLM_KV_BLOCKS` | 1024 |
| `--kv-quantization` | | `VLLM_KV_QUANTIZATION` | false |
| `--max-batch-size` | | `VLLM_MAX_BATCH_SIZE` | 256 |
| `--max-waiting-batches` | | `VLLM_MAX_WAITING_BATCHES` | 10 |
| `--max-draft-tokens` | | `VLLM_MAX_DRAFT_TOKENS` | 8 |
| `--enforce-eager` | | `VLLM_ENFORCE_EAGER` | false |
| `--api-key` | | `VLLM_API_KEY` | - |
| `--api-key-file` | | `VLLM_API_KEYS_FILE` | - |
| `--log-level` | | `VLLM_LOG_LEVEL` | "info" |
| `--log-dir` | | `VLLM_LOG_DIR` | - |
| `--config` | `-c` | `VLLM_CONFIG` | - |

---

## 7. Config File

```yaml
# config.yaml
host: 0.0.0.0
port: 8000

model: /models/Qwen2.5-0.5B-Instruct

engine:
  tensor_parallel_size: 1
  kv_blocks: 1024
  kv_quantization: false
  max_batch_size: 256
  max_waiting_batches: 10
  max_draft_tokens: 8
  enforce_eager: false

auth:
  api_keys: []
  api_keys_file: null

logging:
  log_level: info
  log_dir: null
```

---

## 8. Files to Modify

| File | Changes |
|------|---------|
| `crates/server/Cargo.toml` | Add clap |
| `crates/server/src/cli.rs` | New file |
| `crates/server/src/main.rs` | Use clap |
| `crates/server/src/config.rs` | Merge CLI |

---

## 9. Acceptance Criteria

- [ ] `--help` works with all options
- [ ] `--version` shows version
- [ ] Short args: `-m`, `-p`, `-tp`, `-h`, `-c`
- [ ] Env vars work for all options
- [ ] `--config` loads YAML
- [ ] CLI overrides config
- [ ] Required `--model` enforced
- [ ] Backward compatible with old usage
