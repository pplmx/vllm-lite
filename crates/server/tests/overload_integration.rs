//! HTTP-level overload tests for the chat handler — Phase REL-01.
//!
//! Verifies that when the bounded engine mailbox is saturated, the
//! chat handler returns 503 with the machine-readable `engine_overloaded`
//! error code (distinct from `engine_unavailable`) so clients can
//! implement smarter retry (backoff + jitter) for transient overload.
//!
//! Pattern:
//! 1. Build an `api_state_with_mock_engine` (the mock engine drains
//!    messages but doesn't reply to the AddRequest — so messages
//!    accumulate in the mailbox).
//! 2. Fill the mailbox beyond capacity by sending N `AddRequest`s
//!    directly.
//! 3. Send a real chat request via HTTP and assert the response is
//!    `503 engine_overloaded`.
//!
//! Also covers the closed-channel `engine_unavailable` path as a
//! regression guard for the `try_send` error mapping.

use std::sync::Arc;

use axum::{Router, body::Body, http::Request, routing::post};
use tokio::sync::mpsc;
use tower::ServiceExt;
use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_core::types::EngineMessage;
use vllm_model::config::Architecture;
use vllm_model::tokenizer::Tokenizer;
use vllm_server::ApiState;
use vllm_server::health::HealthChecker;
use vllm_server::openai::batch::BatchManager;
use vllm_server::openai::chat::chat_completions;

use axum::http::StatusCode;
use http_body_util::BodyExt;

fn router(state: ApiState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

fn chat_request_json() -> String {
    serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 3
    })
    .to_string()
}

/// Regression guard: closed `engine_tx` channel surfaces as
/// `503 engine_unavailable`. Confirms the type change from
/// `UnboundedSender` to `Sender` still maps `TrySendError::Closed`
/// to the existing error code (and not, e.g., a generic 500).
#[tokio::test]
async fn test_closed_channel_returns_engine_unavailable_503() {
    // Build an ApiState with a closed channel: drop the receiver
    // immediately so any `try_send` returns `Closed`.
    let (engine_tx, engine_rx) = mpsc::channel::<EngineMessage>(16);
    drop(engine_rx);

    let state = ApiState {
        engine_tx,
        tokenizer: Arc::new(Tokenizer::new()),
        architecture: Architecture::Qwen3,
        batch_manager: Arc::new(BatchManager::new()),
        auth: None,
        health: Arc::new(std::sync::RwLock::new(HealthChecker::new(true, true))),
        metrics: Arc::new(EnhancedMetricsCollector::new()),
    };

    let app = router(state);
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(chat_request_json()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        json["error"]["code"].as_str(),
        Some("engine_unavailable"),
        "closed channel must surface as engine_unavailable, not overloaded"
    );
}

/// REL-01: saturated `engine_tx` channel surfaces as
/// `503 engine_overloaded`. We pre-fill the mailbox by sending
/// `AddRequest` messages via `try_send` until `Full`, then send a
/// real chat HTTP request which must hit the `Full` path and return
/// `engine_overloaded` (distinct from `engine_unavailable`).
#[tokio::test]
async fn test_saturated_mailbox_returns_engine_overloaded_503() {
    // `api_state_with_mock_engine` returns a mock engine that
    // processes AddRequests but never replies with tokens. The
    // bounded mailbox will fill up because the mock engine drains
    // messages but doesn't reply.
    let (state, _handle) = vllm_server::test_fixtures::api_state_with_mock_engine(
        Architecture::Qwen3,
        vec![], // mock engine "replies" with no tokens
    );

    // Pre-fill the mailbox to capacity. We use `try_send` so this
    // works in async context. The mock engine drains messages but
    // doesn't produce tokens; the response channels accumulate
    // (and are dropped via `_response_rx`). After 256 messages
    // (the default mailbox capacity), the next `try_send` returns
    // `Full`.
    let mut filled = 0usize;
    for _ in 0..1024 {
        let (response_tx, _response_rx) = mpsc::channel::<vllm_traits::TokenId>(8);
        let req = vllm_core::types::Request::new(0, vec![1, 2, 3], 4);
        match state.engine_tx.try_send(EngineMessage::AddRequest {
            request: req,
            response_tx,
            seq_id_tx: None,
        }) {
            Ok(()) => filled += 1,
            Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => break,
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => break,
        }
    }
    assert!(
        filled >= 200,
        "expected to fill at least 200 messages; only filled {filled}"
    );

    // Now the mailbox is saturated. A real chat HTTP request must
    // fail fast with `503 engine_overloaded`.
    let app = router(state);
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(chat_request_json()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "saturated mailbox must yield 503"
    );
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        json["error"]["code"].as_str(),
        Some("engine_overloaded"),
        "saturated mailbox must surface as engine_overloaded, not engine_unavailable"
    );
}

/// Negative control: when the mailbox has capacity, the chat handler
/// reaches the engine and we get a successful response (the mock
/// engine still replies with the configured tokens). This proves
/// the `engine_overloaded` test above isn't a false positive caused
/// by an unrelated handler bug.
#[tokio::test]
async fn test_under_capacity_mailbox_succeeds() {
    let (state, _handle) =
        vllm_server::test_fixtures::api_state_with_mock_engine(Architecture::Qwen3, vec![0, 1, 2]);
    let app = router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(chat_request_json()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "uncrowded mailbox must yield 200 (regression guard)"
    );
}
