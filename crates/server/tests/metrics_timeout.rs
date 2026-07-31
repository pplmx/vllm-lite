//! Regression tests: `/health/details`, `/metrics`, and
//! `/debug/kv-cache` must NOT hang when the engine cannot answer a
//! `GetMetrics` request (e.g. it is mid model-step).
//!
//! The `api_state()` fixture wires an engine channel with **no
//! reader**, so `GetMetrics` is never answered. Before the bounded
//! wait was added, these handlers blocked on `response_rx.recv().await`
//! forever; now they fall back to defaults after
//! `METRICS_RESPONSE_TIMEOUT` (2 s) and return promptly.

#![cfg(test)]

use axum::body::Body;
use axum::extract::Request;
use axum::response::Response;
use http_body_util::BodyExt;
use tower::ServiceExt;
use vllm_server::security::cors::CorsConfig;
use vllm_server::test_fixtures::api_state;

/// Build the production router with an engine mailbox that is **alive
/// but never read**: `try_send` succeeds (the message is buffered with
/// a live response sender), so the handler's `recv()` blocks until the
/// bounded wait fires — exercising the timeout fallback rather than
/// the channel-closed path. The returned receiver must stay bound for
/// the test's lifetime.
fn app_with_dead_engine() -> (
    axum::Router,
    tokio::sync::mpsc::Receiver<vllm_core::types::EngineMessage>,
) {
    let mut state = api_state(vllm_model::config::Architecture::Qwen3);
    let (engine_tx, engine_rx) = tokio::sync::mpsc::channel(256);
    state.engine_tx = engine_tx;
    let audit = std::sync::Arc::new(vllm_server::security::audit::AuditLogger::new(1000));
    let app = vllm_server::app::build_app(state, None, audit, &CorsConfig::default());
    (app, engine_rx)
}

async fn get(app: &axum::Router, uri: &str) -> Response {
    let req = Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .unwrap();
    app.clone().oneshot(req).await.unwrap()
}

async fn collect(resp: Response) -> (axum::http::StatusCode, String) {
    let (parts, body) = resp.into_parts();
    let bytes = BodyExt::collect(body).await.unwrap().to_bytes();
    (parts.status, String::from_utf8_lossy(&bytes).to_string())
}

#[tokio::test]
async fn health_details_returns_defaults_when_engine_unresponsive() {
    let (app, _engine_rx) = app_with_dead_engine();
    let (status, body) = collect(get(&app, "/health/details").await).await;
    assert_eq!(status, axum::http::StatusCode::OK);
    assert!(
        body.contains("\"status\":\"ok\""),
        "health/details must return defaults instead of hanging: {body}"
    );
}

#[tokio::test]
async fn debug_kv_cache_returns_defaults_when_engine_unresponsive() {
    // Debug endpoints require admin auth: configure API keys and send
    // a valid Authorization header so the request reaches the handler.
    let mut state = api_state(vllm_model::config::Architecture::Qwen3);
    let (engine_tx, _engine_rx) = tokio::sync::mpsc::channel(256);
    state.engine_tx = engine_tx;
    state.auth = Some(std::sync::Arc::new(vllm_server::auth::AuthMiddleware::new(
        vec!["sk-admin".to_string()],
        1000,
        60,
    )));
    let audit = std::sync::Arc::new(vllm_server::security::audit::AuditLogger::new(1000));
    let app = vllm_server::app::build_app(state, None, audit, &CorsConfig::default());
    let req = Request::builder()
        .method("GET")
        .uri("/debug/kv-cache")
        .header("authorization", "Bearer sk-admin")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let (status, body) = collect(resp).await;
    assert_eq!(status, axum::http::StatusCode::OK);
    assert!(
        body.contains("total_blocks"),
        "/debug/kv-cache must return defaults instead of hanging: {body}"
    );
}

#[tokio::test]
async fn prometheus_scrape_returns_defaults_when_engine_unresponsive() {
    // The routed /metrics handler reads the local collector (no engine
    // round-trip), so exercise `api::get_prometheus` directly — the
    // function the bounded wait protects.
    let mut state = api_state(vllm_model::config::Architecture::Qwen3);
    let (engine_tx, _engine_rx) = tokio::sync::mpsc::channel(256);
    state.engine_tx = engine_tx;
    let app = axum::Router::new()
        .route("/m", axum::routing::get(vllm_server::api::get_prometheus))
        .with_state(state);
    let req = Request::builder()
        .method("GET")
        .uri("/m")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let (parts, body) = resp.into_parts();
    let bytes = BodyExt::collect(body).await.unwrap().to_bytes();
    let body = String::from_utf8_lossy(&bytes).to_string();
    assert_eq!(parts.status, axum::http::StatusCode::OK);
    assert!(
        body.contains("vllm_tokens_total 0"),
        "get_prometheus must return the zero snapshot instead of hanging: {body}"
    );
}
