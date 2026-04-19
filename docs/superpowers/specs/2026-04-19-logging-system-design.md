# 日志系统优化设计

**日期**: 2026-04-19  
**状态**: 已批准  
**目标**: 优化 vllm-lite 日志系统，提高可读性和实用性

---

## 1. 概述

当前日志系统使用纯 JSON 格式输出，对人类可读性差，且日志内容过于冗长。本设计旨在提供一个对开发者和运维都友好的日志系统。

## 2. 输出格式

### 2.1 双输出策略

| 输出目标 | 格式 | 说明 |
|----------|------|------|
| **控制台** | 彩色 + 缩进 | 人类可读，支持 grep/less |
| **文件** | 结构化 JSON | 程序解析，日志聚合工具 |

### 2.2 控制台格式示例

```
2026-04-19 22:30:00 [INFO] vllm_server::main: Starting vllm-lite
2026-04-19 22:30:01 [INFO] vllm_server::main: Model loaded (device=cuda:0)
2026-04-19 22:30:05 [INFO] vllm_server::openai::chat: Request started (request_id=req_abc123, prompt_tokens=150)
2026-04-19 22:30:06 [INFO] vllm_server::openai::chat: Request completed (request_id=req_abc123, output_tokens=42, duration_ms=1234)
2026-04-19 22:30:06 [WARN] vllm_core::engine: CUDA Graph disabled, using regular step
```

### 2.3 JSON 文件格式

```json
{
  "timestamp": "2026-04-19T22:30:06.123Z",
  "level": "INFO",
  "target": "vllm_server::openai::chat",
  "message": "Request completed",
  "request_id": "req_abc123",
  "duration_ms": 1234,
  "prompt_tokens": 150,
  "output_tokens": 42
}
```

## 3. 日志级别规范

### 3.1 级别定义

| 级别 | 使用场景 | 示例 |
|------|----------|------|
| **ERROR** | 系统级失败，无法继续 | 配置验证失败、模型加载失败 |
| **WARN** | 降级运行、回退行为 | CUDA Graph 禁用、tokenizer 回退 |
| **INFO** | 服务生命周期 + 请求概览 | 启动、请求开始/结束 |
| **DEBUG** | 内部流程细节 | 批处理、调度决策 |
| **TRACE** | 详细调试信息 | Token 发送、KV 操作 |

### 3.2 Info 级别白名单

Info 级别**只允许**以下日志：

| 场景 | 必须字段 | 消息模板 |
|------|----------|----------|
| 服务启动 | model_path, device | "Starting vllm-lite" |
| 模型加载完成 | model_name, device | "Model loaded" |
| 服务关闭 | - | "Shutting down" |
| 请求开始 | request_id, prompt_tokens | "Request started" |
| 请求结束 | request_id, output_tokens, duration_ms | "Request completed" |
| 严重错误 | error | 自动使用 ERROR 级别 |

**禁止在 info 级别记录**：
- 完整 config 对象
- 完整 prompt/completion 文本
- 每个 token 的详细信息
- 内部数据结构

## 4. 标准字段

### 4.1 必需字段

| 字段 | 类型 | 说明 |
|------|------|------|
| timestamp | ISO 8601 | 时间戳 |
| level | string | ERROR/WARN/INFO/DEBUG/TRACE |
| target | string | 模块路径 (crate::module) |
| message | string | 人类可读消息 |

### 4.2 可选字段

| 字段 | 类型 | 说明 |
|------|------|------|
| request_id | string | 请求追踪 ID |
| duration_ms | u64 | 操作耗时 (毫秒) |
| error | string | 错误信息 |

### 4.3 请求相关字段

| 字段 | 类型 | 说明 |
|------|------|------|
| prompt_tokens | usize | 输入 token 数 |
| output_tokens | usize | 输出 token 数 |
| model_name | string | 模型名称 |

## 5. 实现任务

### 5.1 日志配置修改

**文件**: `crates/server/src/logging.rs`

```rust
pub fn init_logging(log_dir: Option<PathBuf>, log_level: &str) {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(log_level));

    let subscriber = tracing_subscriber::registry().with(env_filter);

    if let Some(dir) = log_dir {
        // 文件输出：JSON 格式
        let file_appender = RollingFileAppender::new(Rotation::DAILY, dir, "vllm-lite.log");
        let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

        subscriber
            .with(
                fmt::layer()
                    .with_writer(non_blocking)
                    .with_ansi(false)
                    .json()
                    .with_target(true)
                    .with_thread_ids(false)
                    .with_file(true)
                    .with_line_number(true),
            )
            // 控制台输出：美化格式
            .with(
                fmt::layer()
                    .with_target(true)
                    .with_thread_ids(false)
                    .with_ansi(true)
                    .compact(),
            )
            .init();
    } else {
        // 仅控制台输出：美化格式
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

### 5.2 日志内容调整

#### 5.2.1 Server Main (server/main.rs)

| 当前 | 修改后 |
|------|--------|
| `tracing::info!(config = ?app_config, "Starting vllm-lite")` | `tracing::info!("Starting vllm-lite")` |
| `tracing::info!(device = ?device, "Using device")` | `tracing::info!(device = ?device, "Model loaded")` |
| 移除 config 对象打印 | 改用 debug 级别 |

#### 5.2.2 Chat Handler (server/openai/chat.rs)

| 当前 | 修改后 |
|------|--------|
| `tracing::info!(prompt = %prompt, "Built prompt")` | 删除 |
| `tracing::info!(first_tokens = ..., "Prompt tokens")` | `tracing::info!(prompt_tokens = %len, "Request started")` |
| `tracing::info!(completion_text = ..., "Final completion text")` | `tracing::info!(output_tokens = %len, duration_ms = %elapsed, "Request completed")` |

#### 5.2.3 Engine (core/engine.rs)

| 当前 | 修改后 |
|------|--------|
| `tracing::debug!(batch_seq_ids = ?, ...)` | 保持 debug 级别 |
| token-by-token debug | 提升到 trace 级别 |

### 5.3 eprintln 清理

| 位置 | 处理 |
|------|------|
| server/main.rs:98-100 | 移除，保留 tracing::error |
| tests/engine_trace.rs | 保留（测试专用）|

## 6. 配置选项

### 6.1 命令行参数

```rust
#[derive(Parser)]
pub struct CliArgs {
    #[arg(long, default_value = "info")]
    pub log_level: String,

    #[arg(long)]
    pub log_dir: Option<PathBuf>,
}
```

### 6.2 环境变量

支持 `RUST_LOG` 环境变量覆盖默认级别。

## 7. 验收标准

1. ✅ 控制台输出人类可读（彩色、缩进）
2. ✅ JSON 文件保留完整结构化数据
3. ✅ Info 级别不包含敏感内容（prompt/completion）
4. ✅ 请求可追踪（request_id）
5. ✅ 日志性能开销 < 1%

## 8. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 移除日志后调试困难 | debug/trace 级别保留详细信息 |
| JSON 格式变更 | 主要用于文件，控制台不影响 |
