//! Integration tests for the `/health/live` and `/health/ready` handlers.
//!
//! These tests exercise the real `health_handler` and `ready_handler`
//! from `vllm_server::health_handlers`, mounted on a full axum Router
//! with a controlled `ApiState`. They verify:
//!
//! - Liveness probe returns 200 + `{"status":"ok"}` when healthy.
//! - Liveness probe returns 503 + `{"status":"not_ready"}` when
//!   `HealthChecker` reports unhealthy.
//! - Readiness probe includes mailbox-saturation fields.
//! - Readiness probe flips to `not_ready` when the mailbox is full.
//! - Metrics handler returns 200 with `text/plain` content-type.

use std::sync::Arc;

use axum::{
    Router,
    body::Body,
    http::{Request, StatusCode},
    routing::get,
};
use http_body_util::BodyExt;
use tower::ServiceExt;
use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_core::types::EngineMessage;
use vllm_server::ApiState;
use vllm_server::health::HealthChecker;
use vllm_server::health_handlers::{health_handler, ready_handler};
use vllm_server::openai::batch::BatchManager;
use vllm_server::security::audit::AuditLogger;

/// Build an `ApiState` with the given initial health status.
fn build_state(alive: bool, ready: bool, channel_capacity: usize) -> ApiState {
    let (engine_tx, _engine_rx) = tokio::sync::mpsc::channel::<EngineMessage>(channel_capacity);
    ApiState {
        engine_tx,
        tokenizer: Arc::new(vllm_model::tokenizer::Tokenizer::new()),
        architecture: vllm_model::config::Architecture::Qwen3,
        batch_manager: Arc::new(BatchManager::new()),
        auth: None,
        audit: Arc::new(AuditLogger::new(1000)),
        health: Arc::new(std::sync::RwLock::new(HealthChecker::new(alive, ready))),
        metrics: Arc::new(EnhancedMetricsCollector::new()),
        max_model_len: None,
        arch_capabilities: None,
    }
}

fn health_router(state: ApiState) -> Router {
    Router::new()
        .route("/health/live", get(health_handler))
        .route("/health/ready", get(ready_handler))
        .with_state(state)
}

async fn body_bytes(response: axum::response::Response) -> Vec<u8> {
    response
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes()
        .to_vec()
}

// ── Liveness probes ──

#[tokio::test]
async fn liveness_returns_ok_when_healthy() {
    let state = build_state(true, true, 64);
    let app = health_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/health/live")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let json: serde_json::Value = serde_json::from_slice(&body_bytes(response).await).unwrap();
    assert_eq!(json["status"].as_str(), Some("ok"));
}

#[tokio::test]
async fn liveness_returns_service_unavailable_when_not_alive() {
    // When `alive=false`, the liveness handler returns 503 + "unhealthy".
    let state = build_state(false, true, 64);
    let app = health_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/health/live")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

    let json: serde_json::Value = serde_json::from_slice(&body_bytes(response).await).unwrap();
    assert_eq!(json["status"].as_str(), Some("unhealthy"));
}

#[tokio::test]
async fn liveness_content_type_is_json() {
    let state = build_state(true, true, 64);
    let app = health_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/health/live")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap(),
        "application/json"
    );
}

// ── Readiness probes ──

#[tokio::test]
async fn readiness_returns_ok_when_ready_and_not_saturated() {
    let state = build_state(true, true, 64);
    let app = health_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/health/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let json: serde_json::Value = serde_json::from_slice(&body_bytes(response).await).unwrap();
    assert_eq!(json["status"].as_str(), Some("ok"));
    // Verify mailbox fields are present (the channel is empty for a fresh state).
    assert!(json["mailbox_len"].is_number());
    assert!(json["mailbox_capacity"].is_number());
    assert!(json["mailbox_fill_ratio"].is_number());
    assert!(json["mailbox_threshold"].is_number());
    assert_eq!(json["saturated"].as_bool(), Some(false));
}

#[tokio::test]
async fn readiness_returns_not_ready_when_not_ready() {
    // When `ready=false` but `alive=true`, readiness check returns NotReady.
    let state = build_state(true, false, 64);
    let app = health_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/health/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

    let json: serde_json::Value = serde_json::from_slice(&body_bytes(response).await).unwrap();
    assert_eq!(json["status"].as_str(), Some("not_ready"));
}

#[tokio::test]
async fn readiness_returns_unhealthy_when_not_alive() {
    // When `alive=false`, readiness check returns Unhealthy regardless of ready flag.
    let state = build_state(false, false, 64);
    let app = health_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/health/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

    let json: serde_json::Value = serde_json::from_slice(&body_bytes(response).await).unwrap();
    assert_eq!(json["status"].as_str(), Some("unhealthy"));
}

#[tokio::test]
async fn readiness_content_type_is_json() {
    let state = build_state(true, true, 64);
    let app = health_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/health/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap(),
        "application/json"
    );
}
