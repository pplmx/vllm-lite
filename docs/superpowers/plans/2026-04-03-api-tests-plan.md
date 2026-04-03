# Server API Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add comprehensive HTTP-level integration tests for vllm-server API endpoints

**Architecture:** Use axum's testing utilities (tower::ServiceExt) to send HTTP requests to the router without starting a server

**Tech Stack:** Rust, Axum, Tower, tokio

---

## Task 1: Setup Test Infrastructure

**Files:**
- Modify: `crates/server/Cargo.toml`
- Create: `crates/server/tests/api/mod.rs` (test module)

- [ ] **Step 1: Add test dependencies**

Add to `crates/server/Cargo.toml`:
```toml
[dev-dependencies]
axum = { version = "0.8", features = ["tokio", "macros"] }
tower = { version = "0.4", features = ["util"] }
http-body-util = "0.1"
```

- [ ] **Step 2: Create test module structure**

Create directory `crates/server/tests/api/` with:
- `mod.rs` - Test module exports
- `common.rs` - Shared test helpers

Create `crates/server/tests/api/common.rs`:
```rust
use axum::{
    body::Body,
    routing::get,
    Router,
};
use http::{Request, StatusCode};
use tower::ServiceExt;

pub async fn send_request(app: &Router, request: Request<Body>) -> axum::response::Response {
    app.oneshot(request).await.unwrap()
}

pub fn create_test_router() -> Router {
    // Import routes from server
    use vllm_server::*;
    
    Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(openai::chat::chat))
        .route("/v1/completions", post(openai::completions::completions))
        .route("/v1/embeddings", post(openai::embeddings::embeddings))
}
```

- [ ] **Step 3: Run cargo build to verify dependencies**

```bash
cargo build -p vllm-server
```

- [ ] **Step 4: Commit**

```bash
git add crates/server/Cargo.toml crates/server/tests/
git commit -m "test(server): add API test infrastructure"
```

---

## Task 2: Health Endpoint Tests

**Files:**
- Create: `crates/server/tests/api/health.rs`

- [ ] **Step 1: Write health tests**

```rust
use crate::common::*;
use http::{Request, StatusCode};
use axum::body::Body;

#[tokio::test]
async fn test_health_ok() {
    let app = create_test_router();
    let response = send_request(
        &app,
        Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap()
    ).await;
    
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_health_json_response() {
    let app = create_test_router();
    let response = send_request(
        &app,
        Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap()
    ).await;
    
    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json_str = String::from_utf8_lossy(&body);
    assert!(json_str.contains("status"));
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p vllm-server health
```

- [ ] **Step 3: Commit**

```bash
git add crates/server/tests/api/health.rs
git commit -m "test(server): add health endpoint tests"
```

---

## Task 3: Chat Completions Tests

**Files:**
- Create: `crates/server/tests/api/chat.rs`

- [ ] **Step 1: Write chat tests**

```rust
use crate::common::*;
use http::{Request, StatusCode};
use axum::body::Body;
use serde_json::json;

#[tokio::test]
async fn test_chat_basic() {
    let app = create_test_router();
    
    let request_body = json!({
        "model": "qwen",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10
    });
    
    let response = send_request(
        &app,
        Request::builder()
            .uri("/v1/chat/completions")
            .method("POST")
            .header("Content-Type", "application/json")
            .body(Body::from(request_body.to_string()))
            .unwrap()
    ).await;
    
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_chat_empty_prompt() {
    let app = create_test_router();
    
    let request_body = json!({
        "model": "qwen",
        "messages": [],
        "max_tokens": 10
    });
    
    let response = send_request(
        &app,
        Request::builder()
            .uri("/v1/chat/completions")
            .method("POST")
            .header("Content-Type", "application/json")
            .body(Body::from(request_body.to_string()))
            .unwrap()
    ).await;
    
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_chat_streaming() {
    let app = create_test_router();
    
    let request_body = json!({
        "model": "qwen",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5,
        "stream": true
    });
    
    let response = send_request(
        &app,
        Request::builder()
            .uri("/v1/chat/completions")
            .method("POST")
            .header("Content-Type", "application/json")
            .body(Body::from(request_body.to_string()))
            .unwrap()
    ).await;
    
    assert_eq!(response.status(), StatusCode::OK);
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p vllm-server chat
```

- [ ] **Step 3: Fix any issues (compilation or logic)**

- [ ] **Step 4: Commit**

```bash
git add crates/server/tests/api/chat.rs
git commit -m "test(server): add chat completions tests"
```

---

## Task 4: Completions Tests

**Files:**
- Create: `crates/server/tests/api/completions.rs`

- [ ] **Step 1: Write completions tests**

```rust
use crate::common::*;
use http::{Request, StatusCode};
use axum::body::Body;
use serde_json::json;

#[tokio::test]
async fn test_completions_basic() {
    let app = create_test_router();
    
    let request_body = json!({
        "model": "qwen",
        "prompt": "Hello",
        "max_tokens": 10
    });
    
    let response = send_request(
        &app,
        Request::builder()
            .uri("/v1/completions")
            .method("POST")
            .header("Content-Type", "application/json")
            .body(Body::from(request_body.to_string()))
            .unwrap()
    ).await;
    
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_completions_empty_prompt() {
    let app = create_test_router();
    
    let request_body = json!({
        "model": "qwen",
        "prompt": "",
        "max_tokens": 10
    });
    
    let response = send_request(
        &app,
        Request::builder()
            .uri("/v1/completions")
            .method("POST")
            .header("Content-Type", "application/json")
            .body(Body::from(request_body.to_string()))
            .unwrap()
    ).await;
    
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_completions_streaming() {
    let app = create_test_router();
    
    let request_body = json!({
        "model": "qwen",
        "prompt": "Hi",
        "max_tokens": 5,
        "stream": true
    });
    
    let response = send_request(
        &app,
        Request::builder()
            .uri("/v1/completions")
            .method("POST")
            .header("Content-Type", "application/json")
            .body(Body::from(request_body.to_string()))
            .unwrap()
    ).await;
    
    assert_eq!(response.status(), StatusCode::OK);
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p vllm-server completions
```

- [ ] **Step 3: Commit**

```bash
git add crates/server/tests/api/completions.rs
git commit -m "test(server): add completions tests"
```

---

## Task 5: Embeddings Tests

**Files:**
- Create: `crates/server/tests/api/embeddings.rs`

- [ ] **Step 1: Write embeddings tests**

```rust
use crate::common::*;
use http::{Request, StatusCode};
use axum::body::Body;
use serde_json::json;

#[tokio::test]
async fn test_embeddings_basic() {
    let app = create_test_router();
    
    let request_body = json!({
        "model": "qwen",
        "input": ["hello world"]
    });
    
    let response = send_request(
        &app,
        Request::builder()
            .uri("/v1/embeddings")
            .method("POST")
            .header("Content-Type", "application/json")
            .body(Body::from(request_body.to_string()))
            .unwrap()
    ).await;
    
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_embeddings_empty_input() {
    let app = create_test_router();
    
    let request_body = json!({
        "model": "qwen",
        "input": []
    });
    
    let response = send_request(
        &app,
        Request::builder()
            .uri("/v1/embeddings")
            .method("POST")
            .header("Content-Type", "application/json")
            .body(Body::from(request_body.to_string()))
            .unwrap()
    ).await;
    
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_embeddings_batch() {
    let app = create_test_router();
    
    let request_body = json!({
        "model": "qwen",
        "input": ["hello", "world", "test"]
    });
    
    let response = send_request(
        &app,
        Request::builder()
            .uri("/v1/embeddings")
            .method("POST")
            .header("Content-Type", "application/json")
            .body(Body::from(request_body.to_string()))
            .unwrap()
    ).await;
    
    assert_eq!(response.status(), StatusCode::OK);
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p vllm-server embeddings
```

- [ ] **Step 3: Commit**

```bash
git add crates/server/tests/api/embeddings.rs
git commit -m "test(server): add embeddings tests"
```

---

## Task 6: Final Verification

- [ ] **Step 1: Run all server tests**

```bash
cargo test -p vllm-server
```

- [ ] **Step 2: Verify test count increased**

Should have ~15+ new tests

- [ ] **Step 3: Run clippy**

```bash
cargo clippy -p vllm-server -- -D warnings
```

- [ ] **Step 4: Commit final**

```bash
git add .
git commit -m "test(server): add complete API test suite"
```

---

## Success Criteria

- [ ] Test infrastructure setup
- [ ] Health endpoint: 2 tests
- [ ] Chat completions: 3 tests
- [ ] Completions: 3 tests
- [ ] Embeddings: 3 tests
- [ ] All tests pass
