//! Embeddings capability gate test.
//!
//! Production-readiness recommendation §10: not every causal LM
//! checkpoint can produce quality-usable, normalised,
//! dimension-stable embeddings. The handler must refuse with
//! `501 Not Implemented` (and the machine-readable
//! `embeddings_unsupported` code) when the loaded model is a
//! stub or when architecture capabilities couldn't be detected,
//! rather than silently returning meaningless all-zero vectors
//! from `StubModel::embed()`.
//!
//! Two invariants are checked:
//!
//! 1. **`arch_capabilities = None`** (couldn't detect) → 501 with
//!    `embeddings_unsupported`.
//! 2. **`arch_capabilities.is_stub()`** (loaded model is a
//!    placeholder) → 501 with `embeddings_unsupported`.
//! 3. **`arch_capabilities.inference = true`** (real model) →
//!    request reaches the engine (we can't easily mount a real
//!    engine here, so we expect the closed-channel
//!    `engine_unavailable` error rather than `501`).
//!
//! The success-path (request → engine → JSON) is covered by
//! `error_contract.rs` when `arch_capabilities` is `None` (the
//! fixtures default) — that test currently asserts
//! `engine_unavailable` which is the engine-channel-closed case
//! we also exercise here.

#![cfg(test)]

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request as HttpRequest, StatusCode};
use axum::routing::post;
use tower::ServiceExt;
use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_model::arch::ArchCapabilities;
use vllm_model::config::Architecture;
use vllm_model::tokenizer::Tokenizer;
use vllm_server::ApiState;
use vllm_server::health::HealthChecker;
use vllm_server::openai::batch::BatchManager;
use vllm_server::openai::embeddings::embeddings;

fn build_state(arch_capabilities: Option<ArchCapabilities>) -> ApiState {
    let (engine_tx, _engine_rx) = tokio::sync::mpsc::channel(16);
    ApiState {
        engine_tx,
        tokenizer: Arc::new(Tokenizer::new()),
        architecture: Architecture::Qwen3,
        batch_manager: Arc::new(BatchManager::new()),
        auth: None,
        audit: Arc::new(vllm_server::security::audit::AuditLogger::new(1000)),
        health: Arc::new(std::sync::RwLock::new(HealthChecker::new(true, true))),
        metrics: Arc::new(EnhancedMetricsCollector::new()),
        max_model_len: None,
        arch_capabilities,
    }
}

fn router(state: ApiState) -> Router {
    Router::new()
        .route("/v1/embeddings", post(embeddings))
        .with_state(state)
}

async fn body_json(response: axum::response::Response) -> serde_json::Value {
    use http_body_util::BodyExt;
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
}

fn embeddings_request(model: &str, input: &str) -> String {
    serde_json::json!({
        "model": model,
        "input": [input],
    })
    .to_string()
}

#[tokio::test]
async fn embeddings_returns_501_when_capabilities_unknown() {
    // Most defensive case: the loader couldn't detect the
    // architecture. Don't guess — refuse with 501 so the
    // operator knows to fix the model / loader pairing.
    let state = build_state(None);
    let app = router(state);

    let req = HttpRequest::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from(embeddings_request("qwen3", "hello")))
        .unwrap();
    let resp = app.oneshot(req).await.expect("response");

    assert_eq!(
        resp.status(),
        StatusCode::NOT_IMPLEMENTED,
        "embeddings must refuse with 501 when capabilities are unknown"
    );
    let body = body_json(resp).await;
    assert_eq!(
        body["error"]["code"].as_str(),
        Some("embeddings_unsupported"),
        "501 body must carry the machine-readable 'embeddings_unsupported' code"
    );
}

#[tokio::test]
async fn embeddings_returns_501_when_model_is_stub() {
    // Loaded model is a stub architecture (returns all-zero
    // embeddings — meaningless noise). Refuse with 501 so
    // clients don't accidentally use zeros as a real signal.
    let state = build_state(Some(ArchCapabilities::STUB));
    let app = router(state);

    let req = HttpRequest::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from(embeddings_request("qwen3", "hello")))
        .unwrap();
    let resp = app.oneshot(req).await.expect("response");

    assert_eq!(
        resp.status(),
        StatusCode::NOT_IMPLEMENTED,
        "embeddings must refuse with 501 for stub models"
    );
    let body = body_json(resp).await;
    assert_eq!(
        body["error"]["code"].as_str(),
        Some("embeddings_unsupported")
    );
}

#[tokio::test]
async fn embeddings_reaches_engine_when_capabilities_real() {
    // A real (production) capability set must NOT trigger the
    // 501 gate — the handler should fall through to the
    // engine. The engine_tx has no receiver here, so we expect
    // the closed-channel error (503 engine_unavailable), NOT
    // 501.
    let state = build_state(Some(ArchCapabilities::PRODUCTION));
    let app = router(state);

    let req = HttpRequest::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from(embeddings_request("qwen3", "hello")))
        .unwrap();
    let resp = app.oneshot(req).await.expect("response");

    assert_ne!(
        resp.status(),
        StatusCode::NOT_IMPLEMENTED,
        "real capabilities must NOT trigger the embeddings_unsupported gate"
    );
    // The handler proceeded past the gate; with no engine
    // receiver the channel is closed → 503 engine_unavailable.
    assert_eq!(
        resp.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "without an engine receiver the request must surface as engine_unavailable"
    );
}
