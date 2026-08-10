//! Context-length validation wiring test.
//!
//! Production-readiness §4: tokenization after the fact, the chat and
//! completions handlers compare `prompt_tokens + max_tokens` against
//! the model's `max_position_embeddings` and return `400
//! context_length_exceeded` (OpenAI-compatible error code) when the
//! request would exceed the limit. Without this gate a 10× oversize
//! prompt can exhaust KV blocks before any application-level
//! validation runs.
//!
//! Three invariants are checked:
//!
//! 1. **chat (non-streaming)** rejects an over-budget request with
//!    `400` and `error.code = "context_length_exceeded"`.
//! 2. **chat (streaming)** rejects an over-budget request with the
//!    same `400 + context_length_exceeded` (so SSE clients don't
//!    get a hung-up connection that opens then dies on the first
//!    forward pass).
//! 3. **completions** rejects the same way.
//! 4. **`/v1/models`** exposes `max_model_len` so OpenAI-style
//!    clients can size their prompts before sending.
//! 5. **No `max_model_len` configured** → the handler stays
//!    fail-open for bounded requests (stub models / GGUF
//!    without the field still serve), but `max_tokens` above a
//!    hard ceiling is rejected with `400 context_length_exceeded`
//!    (RIL ISS-044) — previously such a request was admitted
//!    unconditionally and drove the engine into an unbounded
//!    generation that held the scheduler thread and OOM'd.
//!
//! We don't exercise the success path (request fits) because
//! every other handler test already covers that — the gate
//! failing open on `max_model_len = None` is the regression
//! risk worth locking in.

#![cfg(test)]

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request as HttpRequest, StatusCode};
use axum::routing::{get, post};
use tower::ServiceExt;
use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_model::config::Architecture;
use vllm_model::tokenizer::Tokenizer;
use vllm_server::ApiState;
use vllm_server::health::HealthChecker;
use vllm_server::openai::batch::BatchManager;
use vllm_server::openai::chat::chat_completions;
use vllm_server::openai::completions::completions as openai_completions;
use vllm_server::openai::models::models_handler;

fn build_state(max_model_len: Option<usize>) -> ApiState {
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
        max_model_len,
        arch_capabilities: None,
    }
}

fn router(state: ApiState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(openai_completions))
        .route("/v1/models", get(models_handler))
        .with_state(state)
        // chat_completions / completions require
        // `Extension<CorrelationId>` (P10 / §6). Mount the same
        // middleware the production router uses so tests exercise
        // the real boundary.
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ))
}

async fn body_json(response: axum::response::Response) -> serde_json::Value {
    use http_body_util::BodyExt;
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
}

fn chat_request(model: &str, max_tokens: u32, stream: bool) -> String {
    serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": max_tokens,
        "stream": stream,
    })
    .to_string()
}

fn completion_request(model: &str, max_tokens: u32) -> String {
    serde_json::json!({
        "model": model,
        "prompt": "hello",
        "max_tokens": max_tokens,
    })
    .to_string()
}

// ---------------------------------------------------------------------------
// /v1/chat/completions — context_length_exceeded
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_rejects_over_budget_request() {
    // max_model_len = 4. A "hello" prompt tokenises to 1 token +
    // max_tokens 10 → 11 > 4. The handler must reject before the
    // engine is touched.
    let state = build_state(Some(4));
    let app = router(state);

    let req = HttpRequest::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(chat_request("qwen3", 10, false)))
        .unwrap();
    let resp = app.oneshot(req).await.expect("response");

    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "over-budget chat must surface 400 (context_length_exceeded)"
    );
    let body = body_json(resp).await;
    assert_eq!(
        body["error"]["code"].as_str(),
        Some("context_length_exceeded"),
        "body must carry the OpenAI-standard 'context_length_exceeded' code"
    );
    assert_eq!(
        body["error"]["type"].as_str(),
        Some("invalid_request_error"),
        "body must carry the OpenAI-standard 'invalid_request_error' type"
    );
}

#[tokio::test]
async fn chat_streaming_rejects_over_budget_request() {
    // Same setup as above but with `stream: true`. SSE clients
    // should get the same 400 instead of a hung-up connection.
    let state = build_state(Some(4));
    let app = router(state);

    let req = HttpRequest::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(chat_request("qwen3", 10, true)))
        .unwrap();
    let resp = app.oneshot(req).await.expect("response");

    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "streaming over-budget chat must surface 400 before any SSE frame"
    );
    let body = body_json(resp).await;
    assert_eq!(
        body["error"]["code"].as_str(),
        Some("context_length_exceeded")
    );
}

// ---------------------------------------------------------------------------
// /v1/completions — context_length_exceeded
// ---------------------------------------------------------------------------

#[tokio::test]
async fn completions_rejects_over_budget_request() {
    let state = build_state(Some(4));
    let app = router(state);

    let req = HttpRequest::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(completion_request("qwen3", 10)))
        .unwrap();
    let resp = app.oneshot(req).await.expect("response");

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp).await;
    assert_eq!(
        body["error"]["code"].as_str(),
        Some("context_length_exceeded")
    );
}

// ---------------------------------------------------------------------------
// /v1/models — exposes max_model_len when configured
// ---------------------------------------------------------------------------

#[tokio::test]
async fn models_endpoint_exposes_max_model_len() {
    let state = build_state(Some(4096));
    let app = router(state);

    let req = HttpRequest::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.expect("response");

    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp).await;
    let models = body["data"].as_array().expect("data is array");
    assert_eq!(models.len(), 1);
    assert_eq!(
        models[0]["max_model_len"].as_u64(),
        Some(4096),
        "configured max_model_len must surface on /v1/models"
    );
}

#[tokio::test]
async fn models_endpoint_omits_max_model_len_when_unconfigured() {
    // When the model didn't declare max_position_embeddings
    // (stub models, some GGUF variants) the field is absent
    // — NOT a hard error.
    let state = build_state(None);
    let app = router(state);

    let req = HttpRequest::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.expect("response");

    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp).await;
    let models = body["data"].as_array().expect("data is array");
    assert_eq!(models.len(), 1);
    assert!(
        models[0].get("max_model_len").is_none(),
        "max_model_len must be skipped (not null) when unconfigured"
    );
}

// ---------------------------------------------------------------------------
// max_model_len = None → fail-open for bounded, hard ceiling for unbounded
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_with_unconfigured_max_model_len_rejects_unbounded_max_tokens() {
    // RIL ISS-044: with no max_position_embeddings and no --max-model-len,
    // an unbounded max_tokens would drive the engine (single scheduler
    // thread) into a generation that never terminates — growing the
    // sequence's token list forever until OOM. It must now be rejected at
    // the boundary with 400 context_length_exceeded instead of admitted.
    let state = build_state(None);
    let app = router(state);

    let req = HttpRequest::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(chat_request("qwen3", 10_000_000, false)))
        .unwrap();
    let resp = app.oneshot(req).await.expect("response");

    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "unbounded max_tokens with no configured context must be rejected"
    );
    let body = body_json(resp).await;
    assert_eq!(
        body["error"]["code"].as_str(),
        Some("context_length_exceeded")
    );
}

#[tokio::test]
async fn chat_with_unconfigured_max_model_len_stays_fail_open_for_bounded() {
    // Stub models / GGUF without the field must still serve *bounded*
    // requests: a modest max_tokens below the ceiling is not blocked by
    // the gate — it proceeds to the engine (a stub channel here, so the
    // closed engine surface reports 503 engine_unavailable, not 400).
    let state = build_state(None);
    let app = router(state);

    let req = HttpRequest::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(chat_request("qwen3", 100, false)))
        .unwrap();
    let resp = app.oneshot(req).await.expect("response");

    assert_ne!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "a bounded request must still pass the gate when max_model_len is unconfigured"
    );
    // Either the engine is unreachable (503) or the engine service is
    // unavailable — never a context-length rejection.
    assert_ne!(resp.status(), StatusCode::BAD_REQUEST);
}
