# Phase 7: OpenAI 兼容 API 设计

> 2026-04-01

## 目标

实现完整的 OpenAI 兼容 API 端点，简化现有代码结构。

## 目录结构

```text
crates/server/src/
├── main.rs           # 路由注册
├── api.rs            # health, metrics (保留)
├── logging.rs        # 日志
├── config.rs         # 配置
└── openai/
    ├── mod.rs        # 模块入口
    ├── types.rs      # 请求/响应类型 + 错误类型
    ├── chat.rs       # /v1/chat/completions
    ├── completions.rs # /v1/completions
    └── embeddings.rs # /v1/embeddings
```

## API 端点

| 端点                   | 方法 | 说明                  |
| ---------------------- | ---- | --------------------- |
| `/v1/chat/completions` | POST | Chat Completions      |
| `/v1/completions`      | POST | Text Completions      |
| `/v1/embeddings`       | POST | Embeddings (占位实现) |
| `/health`              | GET  | 健康检查              |
| `/metrics`             | GET  | Prometheus 指标       |

## 类型定义

### Chat

```rust
pub struct ChatMessage {
    role: String,      // "system", "user", "assistant"
    content: String,
    name: Option<String>,
}

pub struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<usize>,
    stream: Option<bool>,
    n: Option<usize>,
    stop: Option<Vec<String>>,
}

pub struct ChatResponse {
    id: String,          // "chatcmpl-xxx"
    object: String,      // "chat.completion"
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

pub struct ChatChoice {
    index: usize,
    message: ChatMessage,
    finish_reason: Option<String>,
}

pub struct ChatChunk {
    id: String,
    choices: Vec<ChunkChoice>,
}

pub struct ChunkChoice {
    delta: ChatMessage,
    index: usize,
    finish_reason: Option<String>,
}
```

### Completions

```rust
pub struct CompletionRequest {
    model: Option<String>,
    prompt: String,
    temperature: Option<f32>,
    max_tokens: Option<usize>,
    stream: Option<bool>,
    n: Option<usize>,
    stop: Option<Vec<String>>,
}

pub struct CompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

pub struct CompletionChoice {
    text: String,
    index: usize,
    finish_reason: Option<String>,
}
```

### Embeddings

```rust
pub struct EmbeddingsRequest {
    model: String,
    input: Vec<String>,
}

pub struct EmbeddingsResponse {
    object: String,
    data: Vec<EmbeddingData>,
    usage: Usage,
}

pub struct EmbeddingData {
    object: String,
    embedding: Vec<f32>,
    index: usize,
}
```

### 通用

```rust
pub struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

pub struct ErrorResponse {
    error: ErrorDetail,
}

pub struct ErrorDetail {
    message: String,
    type: String,
    code: Option<String>,
}
```

## 实现要点

### 1. Message 处理

- system message: 拼接为 "System: ...\n\n"
- user message: 拼接为 "User: ...\n\n"
- assistant message: 拼接为 "Assistant: ...\n\n" (历史上下文)
- 最终 prompt 以 "Assistant: " 结尾，等待模型续写

### 2. Streaming

- 使用 Server-Sent Events (SSE)
- 格式: `data: {...}\n\n`
- 结束: `data: {...}\n\ndata: [DONE]\n\n`

### 3. Embeddings

- 使用 model 的 hidden state 获取 embedding
- 直接返回向量，不经过 sampling
- 暂时返回占位数据

### 4. 错误处理

- 400: invalid_request_error (缺少字段、格式错误)
- 500: internal_error (引擎不可用)
- 错误格式: `{error: {message, type, code}}`

### 5. 请求验证

- model: 必填
- messages: 必填，非空
- prompt (completions): 必填
- input (embeddings): 必填，非空

## 测试

- 单元测试: 类型序列化
- 集成测试: 端到端请求验证
