# 日志系统优化实现计划

**Goal:** 优化 vllm-lite 日志系统，实现控制台美化格式 + 文件 JSON 输出，规范化日志级别和内容

**Architecture:** 双输出策略 - 控制台使用彩色缩进格式，人类可读；文件使用 JSON 结构化格式，便于程序解析。日志内容精简，info 级别只保留生命周期和请求概览。

**Tech Stack:** tracing + tracing_subscriber + tracing_appender

---

## 文件清单

| 文件 | 操作 | 职责 |
|------|------|------|
| crates/server/src/logging.rs | 修改 | 双输出格式配置 |
| crates/server/src/main.rs | 修改 | 精简日志内容 |
| crates/server/src/openai/chat.rs | 修改 | 精简请求日志 |

---

## 实现任务

### Task 1: 修改日志配置 (logging.rs)

**文件**: `crates/server/src/logging.rs`

- [ ] **Step 1: 备份并替换 logging.rs 内容**

```rust
#![allow(dead_code)]

use std::path::PathBuf;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

pub fn init_logging(log_dir: Option<PathBuf>, log_level: &str) {
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level));

    let subscriber = tracing_subscriber::registry().with(env_filter);

    if let Some(dir) = log_dir {
        std::fs::create_dir_all(&dir).ok();
        let file_appender = RollingFileAppender::new(Rotation::DAILY, dir, "vllm-lite.log");
        let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

        // 文件输出: JSON 格式 (用于程序解析)
        let json_layer = fmt::layer()
            .with_writer(non_blocking)
            .with_ansi(false)
            .json()
            .with_target(true)
            .with_thread_ids(false)
            .with_file(true)
            .with_line_number(true);

        // 控制台输出: 美化格式 (人类可读)
        let console_layer = fmt::layer()
            .with_target(true)
            .with_thread_ids(false)
            .with_ansi(true)
            .compact();

        subscriber
            .with(json_layer)
            .with(console_layer)
            .init();
    } else {
        // 仅控制台输出: 美化格式
        subscriber
            .with(
                fmt::layer()
                    .with_target(true)
                    .with_ansi(true)
                    .compact(),
            )
            .init();
    }
}
```

- [ ] **Step 2: 验证编译**

```bash
cargo check -p vllm-server
```

Expected: 编译成功，无错误

- [ ] **Step 3: 提交**

```bash
git add crates/server/src/logging.rs
git commit -m "feat(server): add dual-output logging (console + JSON file)"
```

---

### Task 2: 精简 server/main.rs 日志

**文件**: `crates/server/src/main.rs:89-200`

- [ ] **Step 1: 修改启动日志**

查找并替换:
```rust
// 替换前:
tracing::info!(config = ?app_config, "Starting vllm-lite");
tracing::info!(device = ?device, "Using device");

// 替换后:
tracing::info!("Starting vllm-lite");
```

```rust
// 替换前:
tracing::info!(
    tensor_parallel_size = tensor_parallel_size,
    "Tensor parallel size"
);

// 替换后:
// 移除此行，或改为 debug
```

```rust
// 替换前:
tracing::info!(
    draft_model = ?draft_model_path,
    kv_blocks = kv_blocks,
    "Engine config"
);

// 替换后:
tracing::debug!(
    draft_enabled = app_config.engine.max_draft_tokens > 0,
    kv_blocks = kv_blocks,
    "Engine configured"
);
```

```rust
// 替换前:
tracing::info!("Loaded tokenizer from {:?}", tokenizer_path);
tracing::info!(
    vocab_size = state.tokenizer.vocab_size(),
    "Tokenizer loaded"
);

// 替换后:
tracing::info!("Tokenizer loaded");
```

- [ ] **Step 2: 移除 eprintln 残留**

删除:
```rust
// 删除此部分 (line 98-101):
eprintln!("Config validation failed:");
for err in errors {
    eprintln!("  - {}", err);
}
```

保留 `tracing::error` 即可。

- [ ] **Step 3: 验证编译并测试**

```bash
cargo check -p vllm-server
cargo test -p vllm-server --lib -- --test-threads=1 2>&1 | tail -20
```

- [ ] **Step 4: 提交**

```bash
git add crates/server/src/main.rs
git commit -m "refactor(server): simplify startup logs, remove eprintln"
```

---

### Task 3: 精简 chat.rs 请求日志

**文件**: `crates/server/src/openai/chat.rs`

- [ ] **Step 1: 修改请求处理日志**

查找 `chat_completion` 函数，修改日志:

```rust
// 替换前 (约 line 95-105):
let prompt = build_prompt_from_messages(&req.messages);
tracing::info!(prompt = %prompt, "Built prompt");

let prompt_tokens = state.tokenizer.encode(&prompt);
tracing::info!(prompt_tokens_len = prompt_tokens.len(), first_tokens = ?&prompt_tokens[..prompt_tokens.len().min(20)], "Prompt tokens");

// 替换后:
let request_id = format!("req_{}", uuid::Uuid::new_v4().to_string()[..8].to_uppercase());
let prompt = build_prompt_from_messages(&req.messages);
let prompt_tokens = state.tokenizer.encode(&prompt);
let prompt_tokens_len = prompt_tokens.len();

tracing::info!(
    request_id = %request_id,
    prompt_tokens = prompt_tokens_len,
    "Request started"
);
```

```rust
// 替换前 (约 line 135-142):
tracing::info!(token_count = tokens.len(), first_tokens = ?&tokens[..tokens.len().min(10)], "Received tokens from engine");
tracing::info!(raw_decode_len = raw_decode.len(), "Raw decode length");
tracing::info!(completion_text = %completion_text, "Final completion text");

// 替换后:
let duration_ms = start.elapsed().as_millis() as u64;
let output_tokens_len = tokens.len();

tracing::info!(
    request_id = %request_id,
    output_tokens = output_tokens_len,
    duration_ms = duration_ms,
    "Request completed"
);
```

注意: 需要在函数开头添加 `let start = std::time::Instant::now();`

- [ ] **Step 2: 确保 request_id 在整个请求中传递**

在函数签名中确保返回 request_id 用于日志关联，或使用 UUID 生成并记录。

- [ ] **Step 3: 验证编译**

```bash
cargo check -p vllm-server
```

- [ ] **Step 4: 提交**

```bash
git add crates/server/src/openai/chat.rs
git commit -m "refactor(server): simplify request logs with request_id tracking"
```

---

### Task 4: 验证和测试

- [ ] **Step 1: 运行完整测试**

```bash
cargo test --workspace --lib 2>&1 | tail -30
```

Expected: 所有测试通过

- [ ] **Step 2: 运行 clippy 检查**

```bash
cargo clippy -p vllm-server -- -D warnings 2>&1
```

- [ ] **Step 3: 运行 CI 完整检查**

```bash
just ci 2>&1 | tail -20
```

Expected: 所有检查通过

- [ ] **Step 4: 提交**

```bash
git commit -m "test: verify logging changes don't break existing tests"
```

---

### Task 5: 更新文档 (可选)

- [ ] **Step 1: 更新 README 或相关文档**

如果项目有日志配置文档，更新说明新的日志格式和行为。

- [ ] **Step 2: 提交**

```bash
git add docs/  # 如有修改
git commit -m "docs: update logging documentation"
```

---

## 验收检查清单

- [ ] 控制台输出彩色美化格式（带缩进）
- [ ] JSON 文件保留完整字段
- [ ] Info 级别不包含 prompt/completion 完整内容
- [ ] 请求有 request_id 可追踪
- [ ] 所有测试通过
- [ ] Clippy 无警告

---

## 风险与回滚

如果出现问题，可以通过以下命令回滚:

```bash
git revert HEAD~4..HEAD  # 回滚本次优化所有 commits
```

或者单独回滚某个文件:

```bash
git checkout HEAD~1 -- crates/server/src/logging.rs
```
