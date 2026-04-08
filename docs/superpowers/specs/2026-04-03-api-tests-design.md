# Server API Tests Design

**Date**: 2026-04-03  
**Status**: Draft  
**Goal**: Add comprehensive API tests for vllm-server endpoints

---

## 1. Executive Summary

Add HTTP-level integration tests for the server API endpoints. Currently there are no API tests - only unit tests for auth and config.

---

## 2. Test Infrastructure

### Dependencies Needed

Add to `crates/server/Cargo.toml`:

```toml
[dev-dependencies]
axum = { version = "0.8", features = ["tokio", "macros"] }
tower = { version = "0.4", features = ["util"] }
http-body-util = "0.1"
```

### Test Setup

```rust
use axum::{
    body::Body,
    routing::get,
    Router,
};
use tower::ServiceExt;
use http::{Request, StatusCode};

// Helper to create test app
fn create_test_app() -> Router {
    // ... setup with mock engine
}

// Helper to send request
async fn send_request(app: &Router, request: Request<Body>) -> Response<Body> {
    app.oneshot(request).await.unwrap()
}
```

---

## 3. Test Cases by Endpoint

### 3.1 GET /health

| Test                        | Description        |
| --------------------------- | ------------------ |
| `test_health_ok`            | Returns 200 OK     |
| `test_health_json_response` | Returns valid JSON |

### 3.2 POST /v1/chat/completions

| Test                         | Description                |
| ---------------------------- | -------------------------- |
| `test_chat_basic`            | Basic chat request         |
| `test_chat_with_temperature` | With temperature parameter |
| `test_chat_streaming`        | Streaming response         |
| `test_chat_empty_prompt`     | Error: empty prompt        |
| `test_chat_missing_model`    | Error: missing model       |
| `test_chat_max_tokens`       | With max_tokens parameter  |

### 3.3 POST /v1/completions

| Test                            | Description         |
| ------------------------------- | ------------------- |
| `test_completions_basic`        | Basic completion    |
| `test_completions_streaming`    | Streaming           |
| `test_completions_empty_prompt` | Error: empty prompt |

### 3.4 POST /v1/embeddings

| Test                            | Description             |
| ------------------------------- | ----------------------- |
| `test_embeddings_basic`         | Basic embedding request |
| `test_embeddings_empty_input`   | Error: empty input      |
| `test_embeddings_single_string` | Single string input     |
| `test_embeddings_batch`         | Batch input             |

### 3.5 Auth Tests (existing +扩展)

| Test                | Description                        |
| ------------------- | ---------------------------------- |
| `test_no_auth`      | Without API key - should fail      |
| `test_with_auth`    | With valid API key - should pass   |
| `test_invalid_auth` | With invalid API key - should fail |

---

## 4. Implementation Order

1. **Phase 1**: Setup test infrastructure
2. **Phase 2**: Health endpoint tests
3. **Phase 3**: Chat completions tests
4. **Phase 4**: Completions tests
5. **Phase 5**: Embeddings tests

---

## 5. Success Criteria

- [ ] Health endpoint tested
- [ ] Chat completions tested (6 tests)
- [ ] Completions tested (3 tests)
- [ ] Embeddings tested (3 tests)
- [ ] Auth integration tested
- [ ] All new tests pass
