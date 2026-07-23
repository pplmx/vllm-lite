//! HTTP → engine `request_id` propagation (production-readiness §6).
//!
//! `correlation_id_middleware` (P1 batch) mints or honours an
//! `X-Request-ID` header and installs a [`CorrelationId`] extension
//! on every incoming request. P10 wires that id through
//! `EngineMessage::AddRequest` so the engine run loop can attach
//! it to every synchronous log line via
//! `tracing::info_span!("engine.add_request", request_id)`.
//!
//! These tests verify the HTTP → engine boundary using a
//! capturing mock engine that records the `request_id` field of
//! the first `AddRequest` it observes, then asserts the value
//! round-trips from the `X-Request-ID` request header (or minted
//! id) all the way to the engine side.
//!
//! The engine-side tracing span itself is exercised by the unit
//! tests in `vllm_core::engine::run` (the `tracing::info_span!`
//! entry is unconditional — when the field is `None` the span
//! still enters with `request_id = null`; verifying the span
//! enters is mostly a smoke test on `tracing` itself). The
//! valuable invariant this file covers is the HTTP → engine
//! boundary, where the actual id byte string is forwarded.

use std::sync::Arc;

use axum::{
    Router,
    body::Body,
    http::{Request, StatusCode},
    middleware::from_fn,
    routing::post,
};
use tokio::sync::Mutex;
use tower::ServiceExt;
use vllm_core::types::EngineMessage;
use vllm_model::config::Architecture;
use vllm_server::ApiState;
use vllm_server::openai::chat::chat_completions;
use vllm_server::openai::completions::completions;
use vllm_server::security::correlation::correlation_id_middleware;

/// Captured `request_id` from the first `EngineMessage::AddRequest`
/// the mock engine receives.
type Captured = Arc<Mutex<Option<String>>>;

/// Spawn a mock engine that captures the `request_id` of the first
/// `AddRequest` it sees, then replies with a single synthetic token
/// (token 10) and exits. Returns the engine sender, the task
/// handle, and the captured `request_id` cell.
fn spawn_capturing_mock_engine() -> (
    vllm_server::api::EngineHandle,
    tokio::task::JoinHandle<()>,
    Captured,
) {
    let (engine_tx, mut engine_rx) = tokio::sync::mpsc::channel::<EngineMessage>(8);
    let captured: Captured = Arc::new(Mutex::new(None));
    let captured_clone = Arc::clone(&captured);
    let handle = tokio::spawn(async move {
        while let Some(msg) = engine_rx.recv().await {
            match msg {
                EngineMessage::AddRequest {
                    request_id,
                    response_tx,
                    seq_id_tx,
                    finish_reason_tx,
                    ..
                } => {
                    if let Some(tx) = seq_id_tx {
                        let _ = tx.send(1);
                    }
                    drop(finish_reason_tx);
                    *captured_clone.lock().await = request_id;
                    let _ = response_tx
                        .send(vllm_traits::SampledToken {
                            token: 10u32,
                            logprob: 0.0,
                            top_logprobs: vec![],
                        })
                        .await;
                    break;
                }
                EngineMessage::Shutdown => break,
                _ => {}
            }
        }
    });
    (engine_tx, handle, captured)
}

/// Build an `ApiState` whose engine channel is wired to the
/// capturing mock. The mock only handles one request, so each
/// test needs its own state.
fn state_with_capturing_engine() -> (ApiState, Captured) {
    let (engine_tx, _handle, captured) = spawn_capturing_mock_engine();
    let state = ApiState {
        engine_tx,
        tokenizer: Arc::new(vllm_model::tokenizer::Tokenizer::new()),
        architecture: Architecture::Qwen3,
        batch_manager: Arc::new(vllm_server::openai::batch::BatchManager::new()),
        auth: None,
        audit: Arc::new(vllm_server::security::audit::AuditLogger::new(1000)),
        health: Arc::new(std::sync::RwLock::new(
            vllm_server::health::HealthChecker::new(true, true),
        )),
        metrics: Arc::new(vllm_core::metrics::EnhancedMetricsCollector::new()),
        max_model_len: None,
        arch_capabilities: None,
    };
    (state, captured)
}

/// Build a router that mounts `correlation_id_middleware` as the
/// OUTERMOST layer (so the response carries `X-Request-ID`) and
/// the chat handler underneath. Mirrors the production wiring in
/// `main.rs`.
fn chat_router(state: ApiState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
        .layer(from_fn(correlation_id_middleware))
}

/// Build a router that mounts `correlation_id_middleware` as the
/// OUTERMOST layer and the completions handler underneath.
fn completions_router(state: ApiState) -> Router {
    Router::new()
        .route("/v1/completions", post(completions))
        .with_state(state)
        .layer(from_fn(correlation_id_middleware))
}

/// Minimal chat request body for the propagation tests. The mock
/// engine only needs enough structure to construct an
/// `EngineMessage::AddRequest` and reply with token 10.
fn chat_request_json(model: &str) -> String {
    serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1,
    })
    .to_string()
}

/// Minimal completion request body for the propagation tests.
fn completions_request_json(model: &str) -> String {
    serde_json::json!({
        "model": model,
        "prompt": "Hello",
        "max_tokens": 1,
    })
    .to_string()
}

/// `X-Request-ID` supplied by the client must be forwarded
/// unchanged through `chat_completions` → `EngineMessage::AddRequest`.
/// This proves the handler doesn't re-mint the id (pre-fix bug
/// from the audit-middleware review: a fresh id was minted at
/// every layer, breaking cross-layer correlation).
#[tokio::test]
async fn chat_handler_forwards_client_supplied_request_id() {
    let (state, captured) = state_with_capturing_engine();
    let app = chat_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("X-Request-ID", "client-supplied-trace-42")
                .body(Body::from(chat_request_json("test-model")))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // Response must echo the same X-Request-ID (correlation
    // middleware contract; sanity check that the middleware ran).
    let echoed = response
        .headers()
        .get("X-Request-ID")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(echoed, "client-supplied-trace-42");

    // Engine must have observed the same id in AddRequest.request_id.
    let captured_id = captured
        .lock()
        .await
        .clone()
        .expect("capturing mock must have observed the AddRequest");
    assert_eq!(
        captured_id.as_str(),
        "client-supplied-trace-42",
        "engine-side request_id must match the client-supplied X-Request-ID",
    );
}

/// When the client omits `X-Request-ID`, the middleware mints a
/// fresh id and the handler must forward that minted id (NOT
/// `None` and NOT a different value).
#[tokio::test]
async fn chat_handler_forwards_minted_request_id_when_client_omits_header() {
    let (state, captured) = state_with_capturing_engine();
    let app = chat_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                // NOTE: no X-Request-ID header.
                .body(Body::from(chat_request_json("test-model")))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // Response carries a minted id (non-empty).
    let echoed = response
        .headers()
        .get("X-Request-ID")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(!echoed.is_empty(), "middleware must mint a request id");

    // Engine sees the SAME minted id, proving the middleware
    // minted it once and downstream layers reused it.
    let captured_id = captured
        .lock()
        .await
        .clone()
        .expect("capturing mock must have observed the AddRequest");
    assert_eq!(
        captured_id.as_str(),
        echoed.as_str(),
        "engine-side request_id must equal the middleware-minted id on the response",
    );
}

/// `/v1/completions` must also forward the request_id. Mirrors
/// the chat tests; the field was added at the same time as the
/// chat forwarding fix.
#[tokio::test]
async fn completions_handler_forwards_client_supplied_request_id() {
    let (state, captured) = state_with_capturing_engine();
    let app = completions_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .header("X-Request-ID", "legacy-trace-7")
                .body(Body::from(completions_request_json("test-model")))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let echoed = response
        .headers()
        .get("X-Request-ID")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(echoed, "legacy-trace-7");

    let captured_id = captured
        .lock()
        .await
        .clone()
        .expect("capturing mock must have observed the AddRequest");
    assert_eq!(
        captured_id.as_str(),
        "legacy-trace-7",
        "engine-side request_id must match the client-supplied X-Request-ID",
    );
}

/// `/v1/completions` must also forward the minted id when the
/// client omits the header. Companion to the chat variant.
#[tokio::test]
async fn completions_handler_forwards_minted_request_id_when_client_omits_header() {
    let (state, captured) = state_with_capturing_engine();
    let app = completions_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                // NOTE: no X-Request-ID header.
                .body(Body::from(completions_request_json("test-model")))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let echoed = response
        .headers()
        .get("X-Request-ID")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(!echoed.is_empty());

    let captured_id = captured
        .lock()
        .await
        .clone()
        .expect("capturing mock must have observed the AddRequest");
    assert_eq!(captured_id.as_str(), echoed.as_str());
}
