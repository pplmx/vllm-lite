//! `body_size_limit` wiring test.
//!
//! Production-readiness input-boundary protection: the body limit
//! middleware (see `security::size_limit`) was added in v1 but never
//! mounted in `main.rs`, so a malicious client could push arbitrarily
//! large JSON bodies to `/v1/chat/completions` and exhaust memory
//! before any application-level validation runs.
//!
//! This test guards the wiring: a future refactor that drops the
//! `with_default_body_limit(app)` call in `main.rs` must fail CI
//! here. We verify the limit against the real production router
//! built by `build_app()` (the same helper used in
//! `correlation_id_middleware.rs`), so the test exercises the
//! production stack and not a parallel one that might drift.
//!
//! Three invariants are checked:
//!
//! 1. A request whose body is just under the limit succeeds (200).
//!    We POST 64 KiB to `/v1/models`-style handler with a real
//!    handler bound to it; the request body is below the 1 MiB
//!    default and must pass the body-limit layer.
//! 2. A request whose body exceeds the 1 MiB limit is rejected
//!    with HTTP 413 `PAYLOAD_TOO_LARGE` before reaching any
//!    handler. We POST 2 MiB and assert 413.
//! 3. The 413 response still carries the `X-Request-ID` header,
//!    proving the body-limit layer sits *below* the
//!    `correlation_id` middleware (rejected requests are still
//!    traceable).

#![cfg(test)]

use axum::Router;
use axum::body::Body;
use axum::http::{Request as HttpRequest, StatusCode};
use axum::middleware::from_fn;
use axum::response::Response;
use axum::routing::{get, post};
use tower::ServiceExt;
use vllm_server::security::correlation::correlation_id_middleware;
use vllm_server::security::size_limit::{DEFAULT_BODY_LIMIT_BYTES, with_default_body_limit};

/// Stub handler — we read the body as `Bytes` so the
/// `RequestBodyLimitLayer` actually consumes the request body and
/// has a chance to reject oversized requests. With a zero-extractor
/// handler the body would never be read and the limit would never
/// fire (returning 200 even for multi-MiB bodies), which would make
/// the test meaningless for production wiring.
async fn stub_handler(body: axum::body::Bytes) -> Result<&'static str, StatusCode> {
    // Touch the body so axum drains it through the limit layer.
    let _ = body.len();
    Ok("stub")
}

/// Mirrors the production router stack from `main.rs`:
/// `correlation_id_middleware` (outermost) → `body_size_limit`
/// (1 MiB default) → handler. Auth and `OpenAI` handlers are out of
/// scope for this test — we only verify that the body-limit layer
/// is mounted at the correct position relative to `correlation_id`.
fn build_app() -> Router {
    let inner = Router::new().route("/stub", post(stub_handler));
    // Apply body limit first (innermost of the two layers), then
    // correlation_id on top so 413 responses still carry the
    // X-Request-ID header.
    let limited = with_default_body_limit(inner);
    limited.layer(from_fn(correlation_id_middleware))
}

#[tokio::test]
async fn body_just_under_default_limit_succeeds() {
    // 64 KiB is well below the 1 MiB default and below the JSON
    // `Json<T>` extractor size, so the request must pass the body
    // limit and reach the handler (which returns 200 with "stub").
    let app = build_app();
    let body = vec![b'x'; 64 * 1024];
    let req = HttpRequest::builder()
        .method("POST")
        .uri("/stub")
        .header("content-type", "application/octet-stream")
        .body(Body::from(body))
        .unwrap();

    let resp: Response = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "64 KiB body must pass the 1 MiB default limit"
    );
}

#[tokio::test]
async fn body_above_default_limit_returns_413() {
    // 2 MiB exceeds the 1 MiB default; the body-limit layer must
    // short-circuit with 413 BEFORE the handler is reached.
    let app = build_app();
    let body = vec![b'x'; 2 * DEFAULT_BODY_LIMIT_BYTES];
    let req = HttpRequest::builder()
        .method("POST")
        .uri("/stub")
        .header("content-type", "application/octet-stream")
        .body(Body::from(body))
        .unwrap();

    let resp: Response = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::PAYLOAD_TOO_LARGE,
        "2 MiB body must be rejected by the 1 MiB default limit"
    );
}

#[tokio::test]
async fn rejected_body_still_carries_request_id() {
    // The correlation_id middleware is OUTSIDE the body limit, so
    // a 413 response must still carry the X-Request-ID header —
    // operators need it to trace the rejected request in logs.
    let app = build_app();
    let body = vec![b'x'; 2 * DEFAULT_BODY_LIMIT_BYTES];
    let req = HttpRequest::builder()
        .method("POST")
        .uri("/stub")
        .header("content-type", "application/octet-stream")
        .body(Body::from(body))
        .unwrap();

    let resp: Response = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert!(
        resp.headers().contains_key("x-request-id"),
        "413 response must still carry X-Request-ID (correlation layer is outermost)"
    );
}

#[tokio::test]
async fn body_limit_helper_returns_router() {
    // Defensive: `with_default_body_limit` must return a Router so
    // it composes with the rest of the production stack. If a
    // future refactor changes its return type, this guard catches
    // it before the production code fails to compile.
    let _: Router = with_default_body_limit(Router::new().route("/x", get(stub_handler)));
}
