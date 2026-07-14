//! Production-readiness §7: graceful shutdown sequence flips readiness to
//! `NotReady` BEFORE the HTTP listener closes.
//!
//! Background: production-readiness §7 documents the recommended
//! shutdown flow:
//!
//! ```text
//! SIGTERM/admin request
//!   -> readiness=false          <-- this PR
//!   -> stop accepting new inference
//!   -> cancel or drain queued requests
//!   -> wait in-flight with deadline
//!   -> flush metrics/logs
//!   -> shutdown engine and join thread
//!   -> exit
//! ```
//!
//! Pre-fix, the `/shutdown` HTTP handler sent `EngineMessage::Shutdown`
//! to the engine and returned `200 OK` without touching the readiness
//! flag, so an external monitor or Kubernetes probe that polled
//! `/health/ready` between the `/shutdown` request and SIGTERM would
//! still see `Ok` and might route new traffic to a pod whose engine
//! was already tearing down.
//!
//! Post-fix both the `/shutdown` handler AND the SIGTERM/Ctrl+C
//! shutdown coordinator call `HealthChecker::mark_not_ready()` so
//! the next `/health/ready` probe returns `503 not_ready`. The
//! SIGTERM path additionally waits `shutdown_drain_grace_secs` (5
//! seconds by default) to give the orchestrator a chance to
//! observe the failed probe and remove the pod from the Service
//! endpoints list before the listener actually closes.

use std::sync::Arc;

use axum::{
    Router,
    body::Body,
    extract::State,
    http::{Request, StatusCode, header::AUTHORIZATION},
    response::IntoResponse,
    routing::get,
};
use http_body_util::BodyExt;
use tower::ServiceExt;
use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_core::types::EngineMessage;
use vllm_model::config::Architecture;
use vllm_model::tokenizer::Tokenizer;
use vllm_server::ApiState;
use vllm_server::api;
use vllm_server::auth::AuthMiddleware;
use vllm_server::health::{HealthChecker, HealthStatus};
use vllm_server::openai::batch::BatchManager;
use vllm_server::security::audit::AuditLogger;

/// Build an `ApiState` for the shutdown/readiness tests. Mirrors
/// `admin_gating::state_with_auth` (the engine channel is a 1-slot
/// stub — `try_send` either accepts or drops, which is fine because
/// the handler only verifies the readiness flag).
fn build_state(api_keys: Vec<String>) -> ApiState {
    let (engine_tx, _engine_rx) = tokio::sync::mpsc::channel::<EngineMessage>(1);
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
        audit: Arc::new(AuditLogger::new(1000)),
        health: Arc::new(std::sync::RwLock::new(HealthChecker::new(true, true))),
        metrics: Arc::new(EnhancedMetricsCollector::new()),
        max_model_len: None,
        arch_capabilities: None,
    }
}

/// Minimal readiness handler that mirrors
/// `vllm_server::health_handlers::ready_handler`'s read-side
/// contract: read the static readiness flag, return 200 + JSON
/// when ready, 503 otherwise. We don't need the mailbox-saturation
/// logic for these tests (the engine channel is a 1-slot stub).
async fn ready_probe(State(state): State<ApiState>) -> impl IntoResponse {
    let status = state.health.read().unwrap().check_readiness();
    let http = StatusCode::from_u16(status.http_status()).unwrap_or(StatusCode::OK);
    let body = serde_json::json!({ "status": status.as_str() });
    (http, axum::Json(body))
}

/// Mount a router with both the `/shutdown` endpoint and the
/// `/health/ready` endpoint so a single test can verify the
/// readiness flip.
fn shutdown_ready_router(state: ApiState) -> Router {
    Router::new()
        .route("/health/ready", get(ready_probe))
        .route("/shutdown", get(api::shutdown))
        .with_state(state)
}

async fn body_json(response: axum::response::Response) -> serde_json::Value {
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap_or_else(|e| {
        panic!(
            "response body is not JSON: {e}; raw = {:?}",
            String::from_utf8_lossy(&bytes)
        )
    })
}

#[tokio::test]
async fn readiness_starts_ok() {
    // Sanity: a freshly-constructed ApiState reports readiness=Ok.
    let state = build_state(vec![]);
    let app = shutdown_ready_router(state);

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
    let json = body_json(response).await;
    assert_eq!(json["status"].as_str(), Some("ok"));
}

#[tokio::test]
async fn shutdown_endpoint_flips_readiness_to_not_ready() {
    // The /shutdown handler must mark the health checker not-ready
    // before returning 200 so a Kubernetes readiness probe that polls
    // immediately after /shutdown observes NotReady and drains the
    // pod. Pre-fix the readiness flag was untouched.
    let state = build_state(vec!["admin-key".to_string()]);
    let health_arc = Arc::clone(&state.health);
    let app = shutdown_ready_router(state);

    // 1. Pre-shutdown: readiness=Ok.
    {
        let response = app
            .clone()
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
    }

    // 2. Trigger /shutdown. The handler still sends
    //    EngineMessage::Shutdown via try_send (the stub drops it,
    //    which is fine — we only care about the readiness flip).
    {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/shutdown")
                    .header(AUTHORIZATION, "Bearer admin-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    // 3. The readiness flag must now be NotReady in the shared
    //    state — verified directly on the Arc to avoid racing the
    //    router clone.
    let status = health_arc.read().unwrap().check_readiness();
    assert_eq!(status, HealthStatus::NotReady);

    // 4. Post-shutdown: a fresh /health/ready probe returns 503.
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
    let json = body_json(response).await;
    assert_eq!(json["status"].as_str(), Some("not_ready"));
}

#[tokio::test]
async fn shutdown_without_admin_key_does_not_flip_readiness() {
    // /shutdown requires admin auth. A rejected request must NOT
    // flip readiness — otherwise an unauthenticated probe could
    // grief a pod into permanent NotReady by spamming /shutdown.
    let state = build_state(vec!["admin-key".to_string()]);
    let health_arc = Arc::clone(&state.health);
    let app = shutdown_ready_router(state);

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
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    let status = health_arc.read().unwrap().check_readiness();
    assert_eq!(
        status,
        HealthStatus::Ok,
        "rejected /shutdown must not flip readiness"
    );
}

#[tokio::test]
async fn shutdown_with_admin_disabled_does_not_flip_readiness() {
    // SEC-01 fail-closed: when no API keys are configured, /shutdown
    // returns 503 admin_disabled. The readiness flag must stay Ok
    // — otherwise the test environment, which often runs without
    // API keys, would falsely advertise itself as draining.
    let state = build_state(vec![]);
    let health_arc = Arc::clone(&state.health);
    let app = shutdown_ready_router(state);

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
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

    let status = health_arc.read().unwrap().check_readiness();
    assert_eq!(status, HealthStatus::Ok);
}

#[tokio::test]
async fn health_checker_mark_not_ready_is_idempotent() {
    // Direct exercise of HealthChecker::mark_not_ready: calling it
    // twice on the same checker is a no-op so the SIGTERM handler
    // and the /shutdown handler can race without coordination.
    let checker = Arc::new(std::sync::RwLock::new(HealthChecker::new(true, true)));
    {
        let mut c = checker.write().unwrap();
        c.mark_not_ready();
        c.mark_not_ready();
    }
    assert_eq!(
        checker.read().unwrap().check_readiness(),
        HealthStatus::NotReady
    );
}
