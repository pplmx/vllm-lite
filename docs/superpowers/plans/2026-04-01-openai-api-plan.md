# OpenAI API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现完整的 OpenAI 兼容 API 端点，替换现有内部 API

**Architecture:** 创建独立的 openai 模块，包含 types/chat/completions/embeddings，新路由注册到 main.rs

**Tech Stack:** axum, serde, tokio

---

### Task 1: 创建 openai 模块目录和 types

**Files:**
- Create: `crates/server/src/openai/mod.rs`
- Create: `crates/server/src/openai/types.rs`
- Modify: `crates/server/src/main.rs` (添加模块声明)

- [ ] **Step 1: 创建 openai 目录**

```bash
mkdir -p crates/server/src/openai
```

- [ ] **Step 2: 创建 openai/mod.rs**

```rust
pub mod types;
pub mod chat;
pub mod completions;
pub mod embeddings;
```

- [ ] **Step 3: 创建 types.rs 基础类型**

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl Usage {
    pub fn new(prompt: usize, completion: usize) -> Self {
        Self {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

impl ErrorResponse {
    pub fn new(message: &str, error_type: &str) -> Self {
        Self {
            error: ErrorDetail {
                message: message.to_string(),
                error_type: error_type.to_string(),
                code: None,
            }
        }
    }
}
```

- [ ] **Step 4: 检查并添加 uuid 依赖**

```bash
grep -n "uuid" crates/server/Cargo.toml
```

如果没找到，在 `[dependencies]` 添加:
```toml
uuid = { version = "1", features = ["v4"] }
```

- [ ] **Step 5: 在 main.rs 添加模块声明**

在 `mod api;` 后添加:
```rust
pub mod openai;
```

- [ ] **Step 6: Commit**

```bash
git add crates/server/src/openai/ crates/server/src/main.rs crates/server/Cargo.toml
git commit -m "feat(server): create openai module structure"
```

---

### Task 2: 实现 Chat 类型定义

**Files:**
- Modify: `crates/server/src/openai/types.rs`

- [ ] **Step 1: 添加 Chat 消息类型**

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<usize>,
    pub stream: Option<bool>,
    pub n: Option<usize>,
    pub stop: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

impl ChatResponse {
    pub fn new(id: String, model: String, choices: Vec<ChatChoice>, usage: Usage) -> Self {
        Self {
            id,
            object: "chat.completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            model,
            choices,
            usage,
        }
    }
}
```

- [ ] **Step 2: 添加 Streaming Chunk 类型**

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

impl ChatChunk {
    pub fn new(id: String, model: String, choice: ChatChunkChoice) -> Self {
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            model,
            choices: vec![choice],
        }
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add crates/server/src/openai/types.rs
git commit -m "feat(server): add Chat types"
```

---

### Task 3: 实现 Completions 和 Embeddings 类型

**Files:**
- Modify: `crates/server/src/openai/types.rs`

- [ ] **Step 1: 添加 Completions 类型**

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: String,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub stream: Option<bool>,
    pub n: Option<usize>,
    pub stop: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

impl CompletionResponse {
    pub fn new(id: String, model: String, choices: Vec<CompletionChoice>, usage: Usage) -> Self {
        Self {
            id,
            object: "text_completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            model,
            choices,
            usage,
        }
    }
}
```

- [ ] **Step 2: 添加 Embeddings 类型**

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsRequest {
    pub model: String,
    pub input: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

impl EmbeddingsResponse {
    pub fn new(embeddings: Vec<Vec<f32>>, model: String) -> Self {
        let data: Vec<EmbeddingData> = embeddings
            .into_iter()
            .enumerate()
            .map(|(i, e)| EmbeddingData {
                object: "embedding".to_string(),
                embedding: e,
                index: i,
            })
            .collect();
        
        let total_tokens = data.iter().map(|d| d.embedding.len()).sum();
        
        Self {
            object: "list".to_string(),
            data,
            model,
            usage: Usage::new(total_tokens, 0),
        }
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add crates/server/src/openai/types.rs
git commit -m "feat(server): add Completions and Embeddings types"
```

---

### Task 4: 实现 Chat Completions Handler

**Files:**
- Create: `crates/server/src/openai/chat.rs`

- [ ] **Step 1: 创建 chat.rs 骨架**

```rust
use axum::{
    extract::State,
    response::sse::{Event, Sse},
    Json,
};
use futures::stream::{self, Stream};
use std::convert::Infallible;
use tokio::sync::mpsc;

use crate::ApiState;
use super::types::*;

pub async fn chat_completions(
    State(state): State<ApiState>,
    Json(req): Json<ChatRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (axum::http::StatusCode, Json<ErrorResponse>)> {
    // 实现见下
}
```

- [ ] **Step 2: 添加请求验证和消息处理辅助函数**

在 chat.rs 中添加:

```rust
fn build_prompt_from_messages(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    
    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str("System: ");
                prompt.push_str(&msg.content);
                prompt.push_str("\n\n");
            }
            "user" => {
                prompt.push_str("User: ");
                prompt.push_str(&msg.content);
                prompt.push_str("\n\n");
            }
            "assistant" => {
                prompt.push_str("Assistant: ");
                prompt.push_str(&msg.content);
                prompt.push_str("\n\n");
            }
            _ => {}
        }
    }
    
    prompt.push_str("Assistant: ");
    prompt
}

fn validate_chat_request(req: &ChatRequest) -> Result<(), (axum::http::StatusCode, Json<ErrorResponse>)> {
    if req.model.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new("model is required", "invalid_request_error")),
        ));
    }
    if req.messages.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new("messages is required", "invalid_request_error")),
        ));
    }
    Ok(())
}
```

- [ ] **Step 3: 实现非流式响应**

```rust
// 在 chat.rs 中添加
async fn handle_chat(
    state: &ApiState,
    req: ChatRequest,
) -> Result<ChatResponse, (axum::http::StatusCode, Json<ErrorResponse>)> {
    validate_chat_request(&req)?;
    
    // 构建 prompt (包含 system 和历史消息)
    let prompt = build_prompt_from_messages(&req.messages);

    let prompt_tokens = state.tokenizer.encode(&prompt);
    let max_tokens = req.max_tokens.unwrap_or(100);
    let total_max = prompt_tokens.len() + max_tokens;

    let mut request = vllm_core::types::Request::new(0, prompt_tokens, total_max);

    if let Some(temp) = req.temperature {
        request.sampling_params.temperature = temp;
    }

    let (response_tx, mut response_rx) = mpsc::unbounded_channel();

    state.engine_tx
        .send(vllm_core::types::EngineMessage::AddRequest {
            request,
            response_tx,
        })
        .map_err(|_| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::new("Engine unavailable", "internal_error")),
            )
        })?;

    // 收集 tokens
    let mut tokens = Vec::new();
    while let Some(token) = response_rx.recv().await {
        tokens.push(token);
    }

    let completion_text = state.tokenizer.decode(&tokens);
    let choice = ChatChoice {
        index: 0,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: completion_text,
            name: None,
        },
        finish_reason: Some("stop".to_string()),
    };

    let usage = Usage::new(prompt_tokens.len(), tokens.len());

    Ok(ChatResponse::new(
        format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        req.model,
        vec![choice],
        usage,
    ))
}
```

- [ ] **Step 4: 实现流式响应**

在 `chat_completions` 函数中添加:

```rust
let is_streaming = req.stream.unwrap_or(false);

if is_streaming {
    let (response_tx, response_rx) = mpsc::unbounded_channel();
    
    // ... 相同的 request 创建逻辑 ...

    let tokenizer = state.tokenizer.clone();
    let stream = stream::unfold(response_rx, move |mut rx| {
        let tokenizer = tokenizer.clone();
        async move {
            match rx.recv().await {
                Some(token) => {
                    let text = tokenizer.decode(&[token]);
                    if text.is_empty() {
                        return Some((Ok(Event::default().data("")), rx));
                    }
                    let chunk = ChatChunk::new(
                        "chatcmpl-stream".to_string(),
                        req.model.clone(),
                        ChatChunkChoice {
                            index: 0,
                            delta: ChatMessage {
                                role: "assistant".to_string(),
                                content: text,
                                name: None,
                            },
                            finish_reason: None,
                        },
                    );
                    let data = serde_json::to_string(&chunk).unwrap();
                    Some((Ok(Event::default().data(format!("data: {}\n\n", data))), rx))
                }
                None => {
                    // 发送结束 chunk
                    let chunk = ChatChunk::new(
                        "chatcmpl-stream".to_string(),
                        req.model.clone(),
                        ChatChunkChoice {
                            index: 0,
                            delta: ChatMessage {
                                role: "assistant".to_string(),
                                content: String::new(),
                                name: None,
                            },
                            finish_reason: Some("stop".to_string()),
                        },
                    );
                    let data = serde_json::to_string(&chunk).unwrap();
                    Some((Ok(Event::default().data(format!("data: {}\n\ndata: [DONE]\n\n", data))), rx))
                }
            }
        }
    });
    
    return Ok(Sse::new(stream));
}

// 非流式
let response = handle_chat(&state, req).await?;
Ok(Sse::new(stream::once(async move {
    let data = serde_json::to_string(&response).unwrap();
    Ok(Event::default().data(data))
})))
```

- [ ] **Step 5: Commit**

```bash
git add crates/server/src/openai/chat.rs
git commit -m "feat(server): implement chat completions handler"
```

---

### Task 5: 实现 Completions Handler

**Files:**
- Create: `crates/server/src/openai/completions.rs`

- [ ] **Step 1: 创建 completions.rs**

```rust
use axum::{
    extract::State,
    response::sse::{Event, Sse},
    Json,
};
use futures::stream::{self, Stream};
use std::convert::Infallible;
use tokio::sync::mpsc;

use crate::ApiState;
use super::types::*;

pub async fn completions(
    State(state): State<ApiState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (axum::http::StatusCode, Json<ErrorResponse>)> {
    if req.prompt.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new("prompt is required", "invalid_request_error")),
        ));
    }
    
    let is_streaming = req.stream.unwrap_or(false);
    let prompt = req.prompt;
    let prompt_tokens = state.tokenizer.encode(&prompt);
    let max_tokens = req.max_tokens.unwrap_or(100);
    let total_max = prompt_tokens.len() + max_tokens;

    let mut request = vllm_core::types::Request::new(0, prompt_tokens, total_max);

    if let Some(temp) = req.temperature {
        request.sampling_params.temperature = temp;
    }

    let (response_tx, mut response_rx) = mpsc::unbounded_channel();

    state.engine_tx
        .send(vllm_core::types::EngineMessage::AddRequest {
            request,
            response_tx,
        })
        .map_err(|_| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::new("Engine unavailable", "internal_error")),
            )
        })?;

    if is_streaming {
        let tokenizer = state.tokenizer.clone();
        let stream = stream::unfold(response_rx, move |mut rx| {
            let tokenizer = tokenizer.clone();
            async move {
                match rx.recv().await {
                    Some(token) => {
                        let text = tokenizer.decode(&[token]);
                        if text.is_empty() {
                            return Some((Ok(Event::default().data("")), rx));
                        }
                        let chunk = serde_json::json!({
                            "id": "cmpl-stream",
                            "object": "text_completion",
                            "choices": [{
                                "text": text,
                                "index": 0,
                            }]
                        });
                        let data = chunk.to_string();
                        Some((Ok(Event::default().data(format!("data: {}\n\n", data))), rx))
                    }
                    None => {
                        Some((Ok(Event::default().data("data: [DONE]\n\n")), rx))
                    }
                }
            }
        });
        
        return Ok(Sse::new(stream));
    }

    // 非流式
    let mut tokens = Vec::new();
    while let Some(token) = response_rx.recv().await {
        tokens.push(token);
    }

    let text = state.tokenizer.decode(&tokens);
    let choice = CompletionChoice {
        text,
        index: 0,
        finish_reason: Some("stop".to_string()),
    };

    let usage = Usage::new(prompt_tokens.len(), tokens.len());
    let response = CompletionResponse::new(
        format!("cmpl-{}", uuid::Uuid::new_v4()),
        "default".to_string(),
        vec![choice],
        usage,
    );

    Ok(Sse::new(stream::once(async move {
        let data = serde_json::to_string(&response).unwrap();
        Ok(Event::default().data(data))
    })))
}
```

- [ ] **Step 2: Commit**

```bash
git add crates/server/src/openai/completions.rs
git commit -m "feat(server): implement completions handler"
```

---

### Task 6: 实现 Embeddings Handler

**Files:**
- Create: `crates/server/src/openai/embeddings.rs`

- [ ] **Step 1: 创建 embeddings.rs**

```rust
use axum::{
    extract::State,
    Json,
};
use crate::ApiState;
use super::types::*;

pub async fn embeddings(
    State(state): State<ApiState>,
    Json(req): Json<EmbeddingsRequest>,
) -> Result<Json<EmbeddingsResponse>, (axum::http::StatusCode, Json<ErrorResponse>)> {
    if req.model.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new("model is required", "invalid_request_error")),
        ));
    }
    if req.input.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new("input is required", "invalid_request_error")),
        ));
    }
    
    // TODO: 实现 embedding 生成
    // 暂时返回占位数据
    
    let embedding_dim = 512; // 需要从 model 获取
    let embeddings: Vec<Vec<f32>> = req.input
        .iter()
        .map(|_| vec![0.0; embedding_dim])
        .collect();

    Ok(Json(EmbeddingsResponse::new(embeddings, req.model)))
}
```

- [ ] **Step 2: Commit**

```bash
git add crates/server/src/openai/embeddings.rs
git commit -m "feat(server): implement embeddings handler (placeholder)"
```

---

### Task 7: 更新 main.rs 路由

**Files:**
- Modify: `crates/server/src/main.rs`

- [ ] **Step 1: 更新路由**

现有 api.rs 保留 health 和 get_prometheus，新增 OpenAI API 路由:

```rust
use vllm_server::openai::{
    chat::chat_completions,
    completions::completions,
    embeddings::embeddings,
};

let app = Router::new()
    // OpenAI API
    .route("/v1/chat/completions", post(chat_completions))
    .route("/v1/completions", post(completions))
    .route("/v1/embeddings", post(embeddings))
    // 运维 (保留在 api.rs)
    .route("/metrics", get(api::get_prometheus))
    .route("/health", get(api::health))
    .with_state(state);
```

- 可选: 移除旧的 completions 路由或标记废弃
- 可选: 移除 shutdown 路由 (用信号关闭即可)

- [ ] **Step 2: 确保 ApiState 可访问**

修改 `main.rs` 中的 `ApiState` 为 pub:
```rust
#[derive(Clone)]
pub struct ApiState {
    pub engine_tx: api::EngineHandle,
    pub tokenizer: Arc<Tokenizer>,
}
```

- [ ] **Step 3: 运行 cargo check 验证**

```bash
cargo check -p vllm-server
```

- [ ] **Step 4: Commit**

```bash
git add crates/server/src/main.rs
git commit -m "feat(server): register OpenAI API routes"
```

---

### Task 8: 测试验证

**Files:**
- Run: 编译和基本测试

- [ ] **Step 1: 编译项目**

```bash
cargo build -p vllm-server
```

- [ ] **Step 2: 运行现有测试**

```bash
cargo test -p vllm-server
```

- [ ] **Step 3: 提交**

```bash
git commit -m "test(server): verify OpenAI API builds"
```

---

## 执行选项

**Plan complete and saved to `docs/superpowers/plans/2026-04-01-openai-api-plan.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**