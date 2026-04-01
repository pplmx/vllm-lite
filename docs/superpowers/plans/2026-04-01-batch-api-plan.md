# Batch API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现异步批量处理 API，支持一次提交多个请求，异步处理并返回结果

**Architecture:** 创建 BatchManager 管理批量任务，后台异步处理，存储在内存中

**Tech Stack:** axum, tokio, serde

---

### Task 1: 创建 Batch 类型定义

**Files:**
- Create: `crates/server/src/openai/batch/types.rs`
- Modify: `crates/server/src/openai/batch/mod.rs`

- [ ] **Step 1: 创建 batch 目录和模块**

```bash
mkdir -p crates/server/src/openai/batch
```

- [ ] **Step 2: 创建 types.rs**

```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleBatchRequest {
    pub prompts: Vec<String>,
    pub endpoint: String,       // "chat" 或 "completions"
    pub model: Option<String>,
    pub max_tokens: Option<i64>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResponse {
    pub id: String,
    pub object: String,
    pub endpoint: String,
    pub status: String,
    pub created_at: i64,
    pub expires_at: i64,
    pub completed_at: Option<i64>,
    pub request_counts: Option<RequestCounts>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestCounts {
    pub total: i32,
    pub completed: i32,
    pub failed: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResults {
    pub batch_id: String,
    pub status: String,
    pub results: Vec<BatchResultItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResultItem {
    pub index: usize,
    pub status: String,
    pub content: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub enum BatchStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

pub struct BatchJob {
    pub id: String,
    pub endpoint: String,
    pub prompts: Vec<String>,
    pub model: Option<String>,
    pub max_tokens: Option<i64>,
    pub temperature: Option<f32>,
    pub status: BatchStatus,
    pub results: Vec<BatchResultItem>,
    pub created_at: i64,
    pub completed_at: Option<i64>,
}

impl BatchJob {
    pub fn new(
        id: String,
        endpoint: String,
        prompts: Vec<String>,
        model: Option<String>,
        max_tokens: Option<i64>,
        temperature: Option<f32>,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        Self {
            id,
            endpoint,
            prompts,
            model,
            max_tokens,
            temperature,
            status: BatchStatus::Pending,
            results: Vec::new(),
            created_at: now,
            completed_at: None,
        }
    }
}
```

- [ ] **Step 3: 创建 mod.rs**

```rust
pub mod types;
pub mod handler;
pub mod manager;
```

- [ ] **Step 4: Commit**

```bash
git add crates/server/src/openai/batch/
git commit -m "feat(server): add Batch API types"
```

---

### Task 2: 实现 BatchManager

**Files:**
- Create: `crates/server/src/openai/batch/manager.rs`

- [ ] **Step 1: 创建 manager.rs**

```rust
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use super::types::{BatchJob, BatchResultItem, BatchStatus};

pub struct BatchManager {
    jobs: Arc<RwLock<HashMap<String, BatchJob>>>,
}

impl BatchManager {
    pub fn new() -> Self {
        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn create_job(
        &self,
        endpoint: String,
        prompts: Vec<String>,
        model: Option<String>,
        max_tokens: Option<i64>,
        temperature: Option<f32>,
    ) -> String {
        let id = format!("batch_{}", Uuid::new_v4());
        let job = BatchJob::new(
            id.clone(),
            endpoint,
            prompts,
            model,
            max_tokens,
            temperature,
        );
        self.jobs.write().await.insert(id.clone(), job);
        id
    }

    pub async fn get_job(&self, id: &str) -> Option<BatchJob> {
        self.jobs.read().await.get(id).cloned()
    }

    pub async fn get_all_jobs(&self) -> Vec<BatchJob> {
        self.jobs.read().await.values().cloned().collect()
    }

    pub async fn update_job(&self, job: BatchJob) {
        self.jobs.write().await.insert(job.id.clone(), job);
    }

    pub async fn add_result(&self, job_id: &str, result: BatchResultItem) {
        let mut jobs = self.jobs.write().await;
        if let Some(job) = jobs.get_mut(job_id) {
            job.results.push(result);
        }
    }

    pub async fn set_completed(&self, job_id: &str) {
        let mut jobs = self.jobs.write().await;
        if let Some(job) = jobs.get_mut(job_id) {
            job.status = BatchStatus::Completed;
            job.completed_at = Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as i64,
            );
        }
    }
}

impl Default for BatchManager {
    fn default() -> Self {
        Self::new()
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add crates/server/src/openai/batch/manager.rs
git commit -m "feat(server): add BatchManager"
```

---

### Task 3: 实现 Batch Handler

**Files:**
- Create: `crates/server/src/openai/batch/handler.rs`

- [ ] **Step 1: 创建 handler.rs**

```rust
use axum::{
    extract::State,
    response::IntoResponse,
    Json,
};
use std::sync::Arc;

use super::types::*;
use super::manager::BatchManager;

pub async fn create_batch(
    State(manager): State<Arc<BatchManager>>,
    Json(req): Json<SimpleBatchRequest>,
) -> Result<Json<BatchResponse>, (axum::http::StatusCode, Json<ErrorResponse>)> {
    if req.prompts.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new("prompts is required", "invalid_request_error")),
        ));
    }

    if req.endpoint != "chat" && req.endpoint != "completions" {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new("endpoint must be 'chat' or 'completions'", "invalid_request_error")),
        ));
    }

    let id = manager
        .create_job(
            req.endpoint.clone(),
            req.prompts,
            req.model,
            req.max_tokens,
            req.temperature,
        )
        .await;

    let job = manager.get_job(&id).await.unwrap();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    Ok(Json(BatchResponse {
        id: job.id,
        object: "batch".to_string(),
        endpoint: job.endpoint,
        status: "pending".to_string(),
        created_at: job.created_at,
        expires_at: now + 86400, // 24 hours
        completed_at: None,
        request_counts: Some(RequestCounts {
            total: job.prompts.len() as i32,
            completed: 0,
            failed: 0,
        }),
    }))
}

pub async fn get_batch(
    State(manager): State<Arc<BatchManager>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<BatchResponse>, (axum::http::StatusCode, Json<ErrorResponse>)> {
    let job = manager.get_job(&id).await.ok_or((
        axum::http::StatusCode::NOT_FOUND,
        Json(ErrorResponse::new("batch not found", "invalid_request_error")),
    ))?;

    let status = match job.status {
        BatchStatus::Pending => "pending",
        BatchStatus::InProgress => "in_progress",
        BatchStatus::Completed => "completed",
        BatchStatus::Failed => "failed",
    };

    let completed = job.results.iter().filter(|r| r.status == "success").count() as i32;
    let failed = job.results.iter().filter(|r| r.status == "error").count() as i32;

    Ok(Json(BatchResponse {
        id: job.id,
        object: "batch".to_string(),
        endpoint: job.endpoint,
        status: status.to_string(),
        created_at: job.created_at,
        expires_at: job.created_at + 86400,
        completed_at: job.completed_at,
        request_counts: Some(RequestCounts {
            total: job.prompts.len() as i32,
            completed,
            failed,
        }),
    }))
}

pub async fn get_batch_results(
    State(manager): State<Arc<BatchManager>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<BatchResults>, (axum::http::StatusCode, Json<ErrorResponse>)> {
    let job = manager.get_job(&id).await.ok_or((
        axum::http::StatusCode::NOT_FOUND,
        Json(ErrorResponse::new("batch not found", "invalid_request_error")),
    ))?;

    let status = match job.status {
        BatchStatus::Pending => "pending",
        BatchStatus::InProgress => "in_progress",
        BatchStatus::Completed => "completed",
        BatchStatus::Failed => "failed",
    };

    Ok(Json(BatchResults {
        batch_id: job.id,
        status: status.to_string(),
        results: job.results,
    }))
}

pub async fn list_batches(
    State(manager): State<Arc<BatchManager>>,
) -> Json<Vec<BatchResponse>> {
    let jobs = manager.get_all_jobs().await;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    let responses: Vec<BatchResponse> = jobs
        .into_iter()
        .map(|job| {
            let status = match job.status {
                BatchStatus::Pending => "pending",
                BatchStatus::InProgress => "in_progress",
                BatchStatus::Completed => "completed",
                BatchStatus::Failed => "failed",
            };
            let completed = job.results.iter().filter(|r| r.status == "success").count() as i32;
            let failed = job.results.iter().filter(|r| r.status == "error").count() as i32;

            BatchResponse {
                id: job.id,
                object: "batch".to_string(),
                endpoint: job.endpoint,
                status: status.to_string(),
                created_at: job.created_at,
                expires_at: now + 86400,
                completed_at: job.completed_at,
                request_counts: Some(RequestCounts {
                    total: job.prompts.len() as i32,
                    completed,
                    failed,
                }),
            }
        })
        .collect();

    Json(responses)
}
```

- [ ] **Step 2: 在 mod.rs 导出 handler**

```rust
pub mod types;
pub mod handler;
pub mod manager;
```

- [ ] **Step 3: Commit**

```bash
git add crates/server/src/openai/batch/
git commit -m "feat(server): add Batch handler"
```

---

### Task 4: 注册路由

**Files:**
- Modify: `crates/server/src/main.rs`
- Modify: `crates/server/src/openai/mod.rs`

- [ ] **Step 1: 在 openai/mod.rs 添加 batch**

```rust
pub mod types;
pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod batch;
```

- [ ] **Step 2: 在 main.rs 注册路由**

添加 BatchManager 到 ApiState:
```rust
use vllm_server::openai::batch::manager::BatchManager;

#[derive(Clone)]
pub struct ApiState {
    pub engine_tx: api::EngineHandle,
    pub tokenizer: Arc<Tokenizer>,
    pub batch_manager: Arc<BatchManager>,
}
```

修改 main.rs:
```rust
use openai::batch::handler::{
    create_batch, get_batch, get_batch_results, list_batches,
};
use openai::batch::manager::BatchManager;

let batch_manager = Arc::new(BatchManager::new());
let state = ApiState {
    engine_tx: msg_tx.clone(),
    tokenizer,
    batch_manager: batch_manager.clone(),
};

let app = Router::new()
    // OpenAI API
    .route("/v1/chat/completions", post(chat_completions))
    .route("/v1/completions", post(openai_completions))
    .route("/v1/embeddings", post(embeddings))
    // Batch API
    .route("/v1/batches", post(create_batch))
    .route("/v1/batches", get(list_batches))
    .route("/v1/batches/:id", get(get_batch))
    .route("/v1/batches/:id/results", get(get_batch_results))
    // 运维
    .route("/metrics", get(api::get_prometheus))
    .route("/health", get(api::health))
    .with_state(state);
```

- [ ] **Step 3: 运行 cargo check**

```bash
cargo check -p vllm-server
```

- [ ] **Step 4: Commit**

```bash
git add crates/server/src/main.rs crates/server/src/openai/mod.rs
git commit -m "feat(server): register Batch API routes"
```

---

### Task 5: 测试验证

**Files:**
- Run: 编译和测试

- [ ] **Step 1: 编译项目**

```bash
cargo build -p vllm-server
```

- [ ] **Step 2: 运行测试**

```bash
cargo test -p vllm-server
```

- [ ] **Step 3: Commit**

```bash
git commit -m "test(server): verify Batch API builds"
```

---

## 执行选项

**Plan complete and saved to `docs/superpowers/plans/2026-04-01-batch-api-plan.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**