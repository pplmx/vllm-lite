//! HTTP-level legacy `/v1/completions` streaming tests with a mock
//! inference engine. The chat twin lives in `chat_integration_test.rs`;
//! this file covers the completions-specific SSE wire contract.
//!
//! Lock guards are intentionally held until the end of each test function
//! (borrow pattern matches the chat harness — `significant_drop_tightening`
//! prefers earlier drops, but the mock-captured parameter borrows must
//! outlive the assertions).
#![allow(clippy::significant_drop_tightening, clippy::type_complexity)]

use std::sync::Arc;

use axum::{
    Router,
    body::Body,
    http::{Request, StatusCode},
    routing::post,
};
use http_body_util::BodyExt;
use tower::ServiceExt;
use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_model::config::Architecture;
use vllm_server::ApiState;
use vllm_server::openai::completions::completions;
use vllm_server::test_fixtures::spawn_mock_engine;

fn router(state: ApiState) -> Router {
    Router::new()
        .route("/v1/completions", post(completions))
        .with_state(state)
        // completions requires `Extension<CorrelationId>`; mount the same
        // middleware the production router uses (P10).
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ))
}

fn streaming_state(
    engine_tx: tokio::sync::mpsc::Sender<vllm_core::types::EngineMessage>,
) -> ApiState {
    ApiState {
        engine_tx,
        tokenizer: Arc::new(vllm_model::tokenizer::Tokenizer::new()),
        architecture: Architecture::Llama,
        batch_manager: Arc::new(vllm_server::openai::batch::manager::BatchManager::new()),
        auth: None,
        audit: Arc::new(vllm_server::security::audit::AuditLogger::new(1000)),
        health: Arc::new(std::sync::RwLock::new(vllm_server::HealthChecker::new(
            true, true,
        ))),
        metrics: Arc::new(EnhancedMetricsCollector::new()),
        max_model_len: None,
        arch_capabilities: None,
    }
}

fn completions_request_json(stream: bool) -> String {
    serde_json::json!({
        "model": "llama-test",
        "prompt": "Hello",
        "stream": stream,
        "max_tokens": 3
    })
    .to_string()
}

/// Regression (RIL ISS-175): each `/v1/completions` SSE stream's chunks
/// must carry the SAME per-request `cmpl-<uuid>` id (OpenAI contract:
/// `id` identifies ONE completion). The n = 1 streaming handler hardcoded
/// `"cmpl-stream"` — every streamed completion shared the identical id,
/// so a client deduping/correlating by `id` collided across streams.
#[tokio::test]
async fn test_completions_streaming_chunks_carry_unique_id() {
    let (engine_tx, _handle) = spawn_mock_engine(vec![7, 8]);
    let app = router(streaming_state(engine_tx));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(completions_request_json(true)))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body_str = std::str::from_utf8(&body_bytes).unwrap();

    let ids: Vec<String> = body_str
        .split("\n\n")
        .filter(|s| !s.is_empty())
        .filter_map(|event| event.strip_prefix("data: "))
        .filter(|data| !data.starts_with('[')) // skip [DONE]
        .filter_map(|data| serde_json::from_str::<serde_json::Value>(data).ok())
        .filter_map(|chunk| chunk["id"].as_str().map(String::from))
        .collect();

    assert!(
        !ids.is_empty(),
        "SSE stream must emit at least one chunk with an id, body was: {body_str}"
    );
    assert_ne!(
        ids[0], "cmpl-stream",
        "chunk id must be unique per request, not the shared constant"
    );
    assert!(
        ids[0].starts_with("cmpl-"),
        "completion chunk id must carry the OpenAI cmpl- prefix, got: {}",
        ids[0]
    );
    assert!(
        ids.iter().all(|id| id == &ids[0]),
        "all chunks of one completion must carry the SAME id, got: {ids:?}"
    );
}
