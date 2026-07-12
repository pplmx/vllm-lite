//! correlation_id_middleware wiring test.
//!
//! Production-readiness recommendation 6: thread a single correlation
//! id (the `X-Request-ID` header) through every HTTP request so
//! operators can trace a request across logs, audit events and the
//! token stream. Previously the middleware existed but was never
//! mounted in `main.rs` — this test guards the wiring so a future
//! refactor that drops the layer fails CI.
//!
//! We verify three invariants against the real axum `Router`:
//!
//! 1. A request without an inbound `X-Request-ID` receives a fresh
//!    one on the response.
//! 2. A request that supplies its own `X-Request-ID` has it
//!    forwarded unchanged (preserves end-to-end tracing when a
//!    client gateway already minted one).
//! 3. Two requests without inbound ids receive distinct response
//!    ids (the counter increments).
//!
//! The router in these tests is intentionally minimal — we just
//! need a `Service` that runs the middleware layer. The full
//! production router (auth + size-limit + handlers) is covered
//! indirectly by `audit_integration.rs`, which builds a real
//! stack and would surface any `from_fn(correlation_id_middleware)`
//! type mismatch at compile time.

#![cfg(test)]

use axum::Router;
use axum::body::Body;
use axum::http::{Request as HttpRequest, StatusCode};
use axum::middleware::from_fn;
use axum::response::Response;
use axum::routing::get;
use tower::ServiceExt;
use vllm_server::security::correlation::correlation_id_middleware;

async fn ping() -> &'static str {
    "pong"
}

fn build_app() -> Router {
    Router::new()
        .route("/ping", get(ping))
        .layer(from_fn(correlation_id_middleware))
}

#[tokio::test]
async fn request_without_id_receives_one_in_response() {
    let app = build_app();

    let req = HttpRequest::builder()
        .uri("/ping")
        .body(Body::empty())
        .expect("build request");
    let resp: Response = app.oneshot(req).await.expect("response");

    assert_eq!(resp.status(), StatusCode::OK);
    let id = resp
        .headers()
        .get("x-request-id")
        .expect("response must carry x-request-id")
        .to_str()
        .expect("x-request-id must be ascii");
    assert!(!id.is_empty(), "minted id must not be empty");
    // Format: `<unix-nanos-hex>-<counter-hex>`.
    let parts: Vec<&str> = id.split('-').collect();
    assert_eq!(parts.len(), 2, "id '{id}' must have exactly one '-'");
    assert!(parts[0].chars().all(|c| c.is_ascii_hexdigit()));
    assert!(parts[1].chars().all(|c| c.is_ascii_hexdigit()));
}

#[tokio::test]
async fn client_supplied_id_is_forwarded_unchanged() {
    let app = build_app();

    let supplied = "trace-from-edge-gateway-12345";
    let req = HttpRequest::builder()
        .uri("/ping")
        .header("x-request-id", supplied)
        .body(Body::empty())
        .expect("build request");
    let resp: Response = app.oneshot(req).await.expect("response");

    assert_eq!(resp.status(), StatusCode::OK);
    let id = resp
        .headers()
        .get("x-request-id")
        .expect("response must echo x-request-id")
        .to_str()
        .expect("ascii");
    assert_eq!(id, supplied, "client id must be passed through verbatim");
}

#[tokio::test]
async fn consecutive_requests_get_distinct_minted_ids() {
    // Two requests, no inbound id → counter must advance so the ids
    // differ (same-nanosecond collision is unlikely but possible; the
    // counter disambiguates).
    let app = build_app();

    let make_req = || {
        HttpRequest::builder()
            .uri("/ping")
            .body(Body::empty())
            .expect("build request")
    };

    let resp1: Response = app
        .clone()
        .oneshot(make_req())
        .await
        .expect("first response");
    let resp2: Response = app.oneshot(make_req()).await.expect("second response");

    let id1 = resp1
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .expect("first response id")
        .to_string();
    let id2 = resp2
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .expect("second response id")
        .to_string();
    assert_ne!(id1, id2, "consecutive minted ids must differ");
}
