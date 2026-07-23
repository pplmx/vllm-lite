//! Readiness wiring test — `/health/ready` reflects engine mailbox saturation.
//!
//! Production-readiness recommendation 7: readiness must NOT be a
//! static boolean. Previously `HealthChecker::check_readiness()`
//! only consulted a `ready: bool` flag set at startup; if the
//! engine's bounded mpsc mailbox (REL-01) saturated, the chat
//! handler started returning 503 but `/health/ready` kept saying
//! `ok` — orchestrators saw an "OK" pod that was actually rejecting
//! requests, so HPA + PDB decisions were made on stale data.
//!
//! The fix wires the `ready_handler` to the live mailbox:
//! `len = max_capacity - capacity`, and readiness flips to
//! `NotReady` once the fill ratio crosses 90 %.
//!
//! These tests assert the three observable invariants:
//!
//! 1. Empty mailbox → `status: ok`, HTTP 200, fill ratio 0.
//! 2. After draining all messages (filling and then consuming from
//!    the receiver) → fill ratio returns to 0, status `ok`.
//! 3. Saturating the mailbox (capacity = 2, fill with 2 `AddRequests`
//!    that the mock engine never drains) → fill ratio 1.0,
//!    status `not_ready`, HTTP 503.

#![cfg(test)]

use std::sync::Arc;

use axum::{Router, body::Body, http::Request, routing::get};
use tokio::sync::mpsc;
use tower::ServiceExt;
use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_core::types::EngineMessage;
use vllm_model::config::Architecture;
use vllm_model::tokenizer::Tokenizer;
use vllm_server::ApiState;
use vllm_server::health::HealthChecker;
use vllm_server::health_handlers::ready_handler;
use vllm_server::openai::batch::BatchManager;

fn build_state(engine_tx: mpsc::Sender<EngineMessage>) -> ApiState {
    // The readiness handler touches only: engine_tx (capacity probe),
    // health (static flag). The remaining fields are present in
    // `ApiState` for the production router but unused here.
    let metrics = Arc::new(EnhancedMetricsCollector::new());
    ApiState {
        engine_tx,
        tokenizer: Arc::new(Tokenizer::new()),
        architecture: Architecture::Qwen3,
        batch_manager: Arc::new(BatchManager::new()),
        auth: None,
        audit: Arc::new(vllm_server::security::audit::AuditLogger::new(1000)),
        health: Arc::new(std::sync::RwLock::new(HealthChecker::new(true, true))),
        metrics,
        max_model_len: None,
        arch_capabilities: None,
    }
}

fn router(state: ApiState) -> Router {
    Router::new()
        .route("/health/ready", get(ready_handler))
        .with_state(state)
}

async fn read_body(resp: axum::response::Response) -> serde_json::Value {
    let bytes = http_body_util::BodyExt::collect(resp.into_body())
        .await
        .expect("collect body")
        .to_bytes();
    serde_json::from_slice(&bytes).expect("json body")
}

#[tokio::test]
async fn ready_handler_reports_ok_when_mailbox_empty() {
    let (tx, _rx) = mpsc::channel::<EngineMessage>(16);
    let app = router(build_state(tx));

    let req = Request::builder()
        .uri("/health/ready")
        .body(Body::empty())
        .expect("build request");
    let resp = app.oneshot(req).await.expect("response");

    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let body = read_body(resp).await;
    assert_eq!(body["status"], "ok");
    assert_eq!(body["mailbox_capacity"], 16);
    assert_eq!(body["mailbox_len"], 0);
    assert_eq!(body["saturated"], false);
    // Fill ratio must be 0 (or close to 0 — the Sender holds one
    // initial permit, so max_capacity - capacity ≈ 0).
    let ratio = body["mailbox_fill_ratio"].as_f64().expect("ratio is f64");
    assert!(
        ratio < 0.01,
        "empty mailbox fill_ratio must be ≈0, got {ratio}"
    );
}

#[tokio::test]
async fn ready_handler_reports_not_ready_when_mailbox_saturated() {
    // capacity = 2 to make saturation fast; the mock engine never
    // pops messages so the mailbox stays full.
    let (tx, _rx) = mpsc::channel::<EngineMessage>(2);
    tx.try_send(EngineMessage::Shutdown).expect("send 1");
    tx.try_send(EngineMessage::Shutdown).expect("send 2");

    let app = router(build_state(tx));

    let req = Request::builder()
        .uri("/health/ready")
        .body(Body::empty())
        .expect("build request");
    let resp = app.oneshot(req).await.expect("response");

    assert_eq!(
        resp.status(),
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        "saturated mailbox must return 503 from /health/ready"
    );
    let body = read_body(resp).await;
    assert_eq!(body["status"], "not_ready");
    assert_eq!(body["mailbox_capacity"], 2);
    assert_eq!(body["mailbox_len"], 2);
    assert_eq!(body["saturated"], true);
    let ratio = body["mailbox_fill_ratio"].as_f64().expect("ratio is f64");
    assert!(
        ratio >= 0.90,
        "saturated mailbox fill_ratio must be ≥ 0.90, got {ratio}"
    );
}

#[tokio::test]
async fn ready_handler_recovers_when_mailbox_drains() {
    // Fill the mailbox to ≥ 90 %, drain via the receiver, then assert
    // the next probe recovers. Capacity 4 + 4 messages = 100 % fill.
    let (tx, mut rx) = mpsc::channel::<EngineMessage>(4);
    for _ in 0..4 {
        tx.try_send(EngineMessage::Shutdown).expect("send");
    }

    let app = router(build_state(tx.clone()));

    // First probe: full → 503.
    let req = Request::builder()
        .uri("/health/ready")
        .body(Body::empty())
        .expect("build request");
    let resp = app.clone().oneshot(req).await.expect("first response");
    assert_eq!(resp.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
    let body = read_body(resp).await;
    assert_eq!(body["saturated"], true);

    // Drain everything.
    for _ in 0..4 {
        rx.recv().await.expect("drain");
    }

    // Second probe: recovered → 200.
    let req = Request::builder()
        .uri("/health/ready")
        .body(Body::empty())
        .expect("build request");
    let resp = app.oneshot(req).await.expect("second response");
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let body = read_body(resp).await;
    assert_eq!(body["status"], "ok");
    assert_eq!(body["mailbox_len"], 0);
    assert_eq!(body["saturated"], false);
}
