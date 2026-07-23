//! SEC-01 wiring tests — admin endpoints refuse unauthorized callers.
//!
//! Background: previously the `/debug/*` and `/shutdown` endpoints
//! were reachable by anyone who could reach the HTTP server, even
//! when the server was bound to `0.0.0.0` without API keys. The
//! fix is a uniform admin gate on every handler in `debug.rs` and
//! the `/shutdown` handler in `api.rs`:
//!
//! 1. No API keys configured → `503 admin_disabled` (fail-closed
//!    even when no auth is set up, because exposing debug state to
//!    the network is worse than blocking local debugging).
//! 2. API keys configured, no `Authorization` header → `401`.
//! 3. API keys configured, valid `Bearer` token → handler runs.
//! 4. API keys configured, wrong `Bearer` token → `401`.
//!
//! These tests construct `ApiState` directly (no real engine) so
//! they exercise the handler-level gate rather than the global
//! auth middleware. The middleware has its own coverage in
//! `crates/server/src/auth.rs`.

use std::sync::Arc;

use axum::{
    Router,
    body::Body,
    http::{HeaderName, Request, StatusCode, header::AUTHORIZATION},
    routing::get,
};
use http_body_util::BodyExt;
use tower::ServiceExt;
use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_model::config::Architecture;
use vllm_model::tokenizer::Tokenizer;
use vllm_server::ApiState;
use vllm_server::api;
use vllm_server::auth::AuthMiddleware;
use vllm_server::debug;
use vllm_server::health::HealthChecker;
use vllm_server::openai::batch::BatchManager;

/// Build an `ApiState` with optional admin auth. The `engine_tx` is a
/// stub that immediately drops incoming messages — handlers under
/// test reach `require_admin` before they touch the channel, so the
/// stub never affects the test outcome.
fn state_with_auth(api_keys: Vec<String>) -> ApiState {
    let (engine_tx, _engine_rx) = tokio::sync::mpsc::channel::<vllm_core::types::EngineMessage>(1);
    let auth = if api_keys.is_empty() {
        None
    } else {
        Some(Arc::new(AuthMiddleware::new(api_keys, 100, 60)))
    };
    ApiState {
        engine_tx,
        tokenizer: Arc::new(Tokenizer::new()),
        architecture: Architecture::Qwen3,
        batch_manager: Arc::new(BatchManager::new()),
        auth,
        audit: Arc::new(vllm_server::security::audit::AuditLogger::new(1000)),
        health: Arc::new(std::sync::RwLock::new(HealthChecker::new(true, true))),
        metrics: Arc::new(EnhancedMetricsCollector::new()),
        max_model_len: None,
        arch_capabilities: None,
    }
}

fn debug_router(state: ApiState) -> Router {
    Router::new()
        .route("/debug/metrics", get(debug::metrics_snapshot))
        .route("/debug/kv-cache", get(debug::kv_cache_dump))
        .route("/debug/trace", get(debug::trace_status))
        .route("/debug/audit", get(debug::audit_dump))
        .with_state(state)
}

fn shutdown_router(state: ApiState) -> Router {
    Router::new()
        .route("/shutdown", get(api::shutdown))
        .with_state(state)
}

async fn body_json(response: axum::response::Response) -> serde_json::Value {
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
}

// ---------------------------------------------------------------------------
// Admin-disabled path (no API keys configured).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn debug_metrics_returns_503_admin_disabled_when_no_auth_configured() {
    let state = state_with_auth(vec![]);
    let app = debug_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/debug/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "debug/metrics must refuse with 503 when no auth is configured"
    );
    let json = body_json(response).await;
    assert_eq!(
        json["error"].as_str(),
        Some("admin_disabled"),
        "503 body must carry the machine-readable 'admin_disabled' code so \
         operators can distinguish it from a real outage"
    );
}

#[tokio::test]
async fn debug_kv_cache_returns_503_admin_disabled_when_no_auth_configured() {
    let state = state_with_auth(vec![]);
    let app = debug_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/debug/kv-cache")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let json = body_json(response).await;
    assert_eq!(json["error"].as_str(), Some("admin_disabled"));
}

#[tokio::test]
async fn debug_trace_returns_503_admin_disabled_when_no_auth_configured() {
    let state = state_with_auth(vec![]);
    let app = debug_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/debug/trace")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let json = body_json(response).await;
    assert_eq!(json["error"].as_str(), Some("admin_disabled"));
}

#[tokio::test]
async fn shutdown_returns_503_admin_disabled_when_no_auth_configured() {
    let state = state_with_auth(vec![]);
    let app = shutdown_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/shutdown")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "/shutdown must refuse with 503 when no auth is configured"
    );
    let json = body_json(response).await;
    assert_eq!(json["error"].as_str(), Some("admin_disabled"));
}

// ---------------------------------------------------------------------------
// Auth-configured paths: valid key succeeds, missing/wrong key returns 401.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn debug_metrics_with_valid_bearer_succeeds() {
    let state = state_with_auth(vec!["good-key".to_string()]);
    let app = debug_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/debug/metrics")
                .header(AUTHORIZATION, "Bearer good-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "valid Bearer token must reach the handler"
    );
}

#[tokio::test]
async fn debug_metrics_with_wrong_bearer_returns_401() {
    let state = state_with_auth(vec!["good-key".to_string()]);
    let app = debug_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/debug/metrics")
                .header(AUTHORIZATION, "Bearer wrong-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::UNAUTHORIZED,
        "wrong Bearer token must yield 401"
    );
    let json = body_json(response).await;
    assert_eq!(json["error"].as_str(), Some("unauthorized"));
}

#[tokio::test]
async fn debug_metrics_without_auth_header_returns_401() {
    let state = state_with_auth(vec!["good-key".to_string()]);
    let app = debug_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/debug/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::UNAUTHORIZED,
        "missing Authorization header must yield 401"
    );
}

#[tokio::test]
async fn shutdown_with_valid_bearer_succeeds() {
    let state = state_with_auth(vec!["good-key".to_string()]);
    let app = shutdown_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/shutdown")
                .header(AUTHORIZATION, "Bearer good-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "valid Bearer token must reach /shutdown"
    );
    let json = body_json(response).await;
    assert_eq!(json.as_str(), Some("Shutting down"));
}

// ---------------------------------------------------------------------------
// Header shape regression guard.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn authorization_header_must_use_bearer_scheme() {
    // A header without the "Bearer " prefix must NOT authenticate,
    // even if the raw value happens to equal a configured key. This
    // guards against a regression where someone "fixes" the
    // constant-time comparison by widening what counts as a match.
    let state = state_with_auth(vec!["good-key".to_string()]);
    let app = shutdown_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/shutdown")
                .header(HeaderName::from_static("authorization"), "good-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::UNAUTHORIZED,
        "Authorization header without 'Bearer ' prefix must not authenticate"
    );
}

// ---------------------------------------------------------------------------
// /debug/audit — admin-gated JSON dump of the in-memory audit ring buffer.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn debug_audit_returns_503_admin_disabled_when_no_auth_configured() {
    let state = state_with_auth(vec![]);
    let app = debug_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/debug/audit")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "/debug/audit must refuse with 503 when no auth is configured"
    );
    let json = body_json(response).await;
    assert_eq!(json["error"].as_str(), Some("admin_disabled"));
}

#[tokio::test]
async fn debug_audit_with_valid_bearer_returns_empty_buffer() {
    let state = state_with_auth(vec!["good-key".to_string()]);
    let app = debug_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/debug/audit")
                .header(AUTHORIZATION, "Bearer good-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "valid Bearer token must reach /debug/audit"
    );
    let json = body_json(response).await;
    // No requests have been processed by this router, so the
    // ring buffer is empty but the response shape is fixed.
    assert_eq!(json["count"].as_u64(), Some(0));
    assert_eq!(json["returned"].as_u64(), Some(0));
    assert!(
        json["events"].as_array().map(Vec::is_empty) == Some(true),
        "events array must be empty for an idle server, got {:?}",
        json["events"]
    );
}

#[tokio::test]
async fn debug_audit_with_wrong_bearer_returns_401() {
    let state = state_with_auth(vec!["good-key".to_string()]);
    let app = debug_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/debug/audit")
                .header(AUTHORIZATION, "Bearer wrong-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let json = body_json(response).await;
    assert_eq!(json["error"].as_str(), Some("unauthorized"));
}
