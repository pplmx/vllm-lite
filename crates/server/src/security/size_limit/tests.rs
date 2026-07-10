//! Unit + integration tests for the request-body size limit.
//!
//! Locks in two contracts:
//!
//! 1. **Custom-limit layer**: a small JSON payload (< 1024 B limit)
//!    passes through to 200 OK; a 256 B raw payload against a 64 B
//!    limit is rejected with 413 `PAYLOAD_TOO_LARGE`.
//! 2. **Default 1 MiB helper**: `with_default_body_limit` accepts
//!    up to 1 MiB (512 KiB test → OK) and rejects anything above
//!    (2 MiB test → 413).
//!
//! Tests use a raw `Bytes` handler for the over-limit cases so
//! the `Json` extractor's 415 short-circuit doesn't fire before
//! the body-limit layer sees the request.
use super::*;
use axum::body::Body;
use axum::http::{Request as HttpRequest, StatusCode};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{Value, json};
use tower::ServiceExt;

async fn echo(Json(value): Json<Value>) -> Json<Value> {
    Json(value)
}

fn app_with_limit(limit: usize) -> Router {
    Router::new()
        .route("/echo", post(echo))
        .layer(RequestBodyLimitLayer::new(limit))
}

#[tokio::test]
async fn test_request_under_limit_succeeds() {
    let app = app_with_limit(1024);
    let body = axum::body::Body::from(json!({"hello": "world"}).to_string());
    let req = HttpRequest::builder()
        .method("POST")
        .uri("/echo")
        .header("content-type", "application/json")
        .body(body)
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_request_over_limit_returns_413() {
    // 256-byte payload > 64-byte limit. Send as raw body via a
    // raw-body handler so the Json extractor's 415 short-circuit
    // doesn't fire before the size limit.
    let raw_app = Router::new()
        .route(
            "/echo",
            post(|body: axum::body::Bytes| async move {
                let _ = body.len();
                axum::http::StatusCode::OK
            }),
        )
        .layer(RequestBodyLimitLayer::new(64));
    let large = "x".repeat(256);
    let body = axum::body::Body::from(large);
    let req = HttpRequest::builder()
        .method("POST")
        .uri("/echo")
        .header("content-type", "application/octet-stream")
        .body(body)
        .unwrap();
    let resp = raw_app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn test_default_limit_helper_uses_one_mib() {
    let raw_app = with_default_body_limit(Router::new().route(
        "/echo",
        post(|body: axum::body::Bytes| async move {
            let _ = body.len();
            axum::http::StatusCode::OK
        }),
    ));
    // 512 KiB body < 1 MiB default → OK
    let body_bytes = "x".repeat(512 * 1024);
    let body = Body::from(body_bytes);
    let req = HttpRequest::builder()
        .method("POST")
        .uri("/echo")
        .header("content-type", "application/octet-stream")
        .body(body)
        .unwrap();
    let resp = raw_app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_default_limit_rejects_over_one_mib() {
    let raw_app = with_default_body_limit(Router::new().route(
        "/echo",
        post(|body: axum::body::Bytes| async move {
            let _ = body.len();
            axum::http::StatusCode::OK
        }),
    ));
    // 2 MiB body > 1 MiB default → 413
    let body_bytes = "x".repeat(2 * 1024 * 1024);
    let body = Body::from(body_bytes);
    let req = HttpRequest::builder()
        .method("POST")
        .uri("/echo")
        .header("content-type", "application/octet-stream")
        .body(body)
        .unwrap();
    let resp = raw_app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
}
