// crates/server/src/bootstrap/handlers.rs
//
// HTTP handlers for `/health/live`, `/health/ready`, and `/metrics`.
// These are wired in `main.rs` for both the canonical K8s probe paths
// (`/health/live`, `/health/ready`) and the legacy aliases (`/health`,
// `/ready`).

use axum::{extract::State, http::StatusCode, response::Response};
use serde_json::json;
use vllm_core::metrics::PrometheusExporter;
use vllm_server::ApiState;

/// Health check endpoint - liveness probe
pub async fn health_handler(State(state): State<ApiState>) -> Response {
    // invariant: lock is only held for synchronous field access; no panic possible while holding.
    let status = state.health.read().unwrap().check_liveness();
    let http_status = StatusCode::from_u16(status.http_status()).unwrap_or(StatusCode::OK);

    let body = json!({ "status": status.as_str() });
    Response::builder()
        .status(http_status)
        .header("content-type", "application/json")
        .body(serde_json::to_string(&body).unwrap_or_default().into())
        .unwrap_or_else(|_| Response::new(axum::body::Body::empty()))
}

/// Readiness check endpoint
pub async fn ready_handler(State(state): State<ApiState>) -> Response {
    // invariant: lock is only held for synchronous field access; no panic possible while holding.
    let status = state.health.read().unwrap().check_readiness();
    let http_status =
        StatusCode::from_u16(status.http_status()).unwrap_or(StatusCode::SERVICE_UNAVAILABLE);

    let body = json!({ "status": status.as_str() });
    Response::builder()
        .status(http_status)
        .header("content-type", "application/json")
        .body(serde_json::to_string(&body).unwrap_or_default().into())
        .unwrap_or_else(|_| Response::new(axum::body::Body::empty()))
}

/// Prometheus metrics endpoint
pub async fn metrics_handler(State(state): State<ApiState>) -> Response {
    let exporter = PrometheusExporter::new(state.metrics.clone(), 9090);
    let output = exporter.export_to_string().await;

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/plain; charset=utf-8")
        .body(output.into())
        .unwrap_or_else(|_| Response::new(axum::body::Body::empty()))
}
