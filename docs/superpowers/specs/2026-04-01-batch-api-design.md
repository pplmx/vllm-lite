# Batch API Design

> 2026-04-01

## Goal

实现异步批量处理 API，支持一次提交多个请求，异步处理并返回结果。

## API Endpoints

| Endpoint                   | Method | Description      |
| -------------------------- | ------ | ---------------- |
| `/v1/batches`              | POST   | 提交批量任务     |
| `/v1/batches/{id}`         | GET    | 查询任务状态     |
| `/v1/batches/{id}/results` | GET    | 获取任务结果     |
| `/v1/batches`              | GET    | 列出所有批量任务 |

## Request/Response Types

### POST /v1/batches

**Request:**

```rust
pub struct CreateBatchRequest {
    pub input_file_id: String,  // 或 inline 输入
    pub endpoint: String,       // "/v1/chat/completions" 或 "/v1/completions"
    pub completion_window: String, // "24h"
    pub metadata: Option<HashMap<String, String>>,
}

// 简化版：直接传 prompts
pub struct SimpleBatchRequest {
    pub prompts: Vec<String>,
    pub endpoint: String,       // "chat" 或 "completions"
    pub model: Option<String>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
}
```

**Response:**

```rust
pub struct BatchResponse {
    pub id: String,           // "batch_xxx"
    pub object: String,       // "batch"
    pub endpoint: String,
    pub status: String,       // "pending", "in_progress", "completed", "failed"
    pub input_file_id: String,
    pub output_file_id: Option<String>,
    pub error_file_id: Option<String>,
    pub created_at: i64,
    pub expires_at: i64,
    pub completed_at: Option<i64>,
    pub failed_at: Option<i64>,
    pub request_counts: Option<RequestCounts>,
}

pub struct RequestCounts {
    pub total: i32,
    pub completed: i32,
    pub failed: i32,
}
```

### GET /v1/batches/{id}

返回 BatchResponse

### GET /v1/batches/{id}/results

```rust
pub struct BatchResults {
    pub batch_id: String,
    pub results: Vec<BatchResultItem>,
}

pub struct BatchResultItem {
    pub prompt_index: usize,
    pub status: String,       // "success" 或 "error"
    pub response: Option<ChatResponse>, // 或 CompletionResponse
    pub error: Option<String>,
}
```

## Status Flow

```text
pending → in_progress → completed
                      → failed
```

## Implementation

### 1. Batch Manager

创建 `BatchManager` 管理批量任务：

- 存储任务状态 (内存或文件)
- 后台处理队列
- 定期写入结果

### 2. Handler

- `POST /v1/batches`: 创建任务，加入队列
- `GET /v1/batches/{id}`: 查询状态
- `GET /v1/batches/{id}/results`: 返回结果
- `GET /v1/batches`: 列出任务

### 3. Background Processing

- 异步处理队列中的任务
- 每个 prompt 独立调用 engine
- 收集结果或错误

## Error Handling

- 400: 无效请求格式
- 404: 任务不存在
- 500: 内部错误

## Testing

- 单元测试: 类型序列化
- 集成测试: 提交批量任务，验证结果
