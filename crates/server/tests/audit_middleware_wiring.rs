//! `audit_middleware` wiring test.
//!
//! Production-readiness recommendation: every request should leave a
//! trail in the audit ring buffer so operators can read it via
//! `/debug/audit` (and aggregate it through the structured `tracing`
//! log stream). The [`AuditLogger`] type existed for a long time, but
//! nothing in `main.rs` actually called `log_api_request` for the
//! HTTP path; this test guards the wiring so a future refactor that
//! drops the layer fails CI.
//!
//! We verify three invariants against a real axum router built the
//! same way `main.rs` builds it (`correlation_id` outermost, audit
//! below it):
//!
//! 1. A successful (200) request produces exactly one audit event
//!    with `result = "success"`, the right method and path, and the
//!    correlation id from the inbound/outbound `X-Request-ID` header.
//! 2. A 4xx response is recorded as a failure (`result` starts with
//!    `"failure: "`) so the audit trail captures rejected requests
//!    too — operators need to know who tried what and was denied.
//! 3. Two distinct requests produce two distinct audit events (the
//!    logger is not somehow single-shot).
//!
//! The router intentionally omits auth so the test stays focused on
//! the audit wiring; auth's effect on the audit row (stamping the
//! `AuthenticatedUser` extension) is covered by the unit tests in
//! `security/audit.rs` and the manual end-to-end check at startup
//! time.

#![cfg(test)]

use axum::Router;
use axum::body::Body;
use axum::http::{Request as HttpRequest, StatusCode};
use axum::middleware::{from_fn, from_fn_with_state};
use axum::response::Response;
use axum::routing::get;
use std::sync::Arc;
use tower::ServiceExt;
use vllm_server::security::audit::AuditLogger;
use vllm_server::security::audit_middleware::audit_middleware;
use vllm_server::security::correlation::correlation_id_middleware;

async fn ok() -> &'static str {
    "ok"
}

/// Mirrors the production stack from `main.rs` for the layers we
/// care about: `correlation_id` (outermost) → `audit_middleware`
/// (below, sees every request including the ones `correlation_id`
/// stamped) → handler. We omit body-limit and auth because the audit
/// wiring doesn't depend on either — those are guarded by their own
/// integration tests (`body_limit_wiring.rs`, `admin_gating.rs`).
fn build_app(audit: Arc<AuditLogger>) -> Router {
    Router::new()
        .route("/ok", get(ok))
        // audit_middleware takes the logger via `State`, exactly
        // like `main.rs` mounts it.
        .layer(from_fn_with_state(audit, audit_middleware))
        // correlation_id is OUTSIDE so even requests that the audit
        // layer might error on still carry a stable id.
        .layer(from_fn(correlation_id_middleware))
}

#[tokio::test]
async fn successful_request_records_audit_event() {
    let audit = Arc::new(AuditLogger::new(100));
    let app = build_app(audit.clone());

    let supplied = "audit-test-trace-id-0001";
    let req = HttpRequest::builder()
        .method("GET")
        .uri("/ok")
        .header("x-request-id", supplied)
        .body(Body::empty())
        .expect("build request");
    let resp: Response = app.oneshot(req).await.expect("response");

    assert_eq!(resp.status(), StatusCode::OK);

    let events = audit.get_events().await;
    assert_eq!(events.len(), 1, "exactly one audit row per request");
    let ev = &events[0];
    assert_eq!(ev.action, "GET");
    assert_eq!(ev.resource, "/ok");
    assert_eq!(ev.result, "success");
    assert_eq!(
        ev.request_id, supplied,
        "audit row must carry the same correlation id the response echoes"
    );
    assert!(
        ev.user_id.is_none(),
        "unauthenticated request must record user_id = None (no Bearer token)"
    );
}

#[tokio::test]
async fn not_found_response_is_audited_as_failure() {
    // The audit layer must capture rejections too, otherwise
    // operators have no record of who probed which path. We use a
    // 404 (unknown route → `axum::handler::not_found`'s default
    // empty 404) rather than building a custom error handler.
    let audit = Arc::new(AuditLogger::new(100));
    let app = build_app(audit.clone());

    let req = HttpRequest::builder()
        .method("GET")
        .uri("/does-not-exist")
        .body(Body::empty())
        .expect("build request");
    let resp: Response = app.oneshot(req).await.expect("response");
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);

    let events = audit.get_events().await;
    assert_eq!(events.len(), 1);
    let ev = &events[0];
    assert_eq!(ev.action, "GET");
    assert_eq!(ev.resource, "/does-not-exist");
    assert!(
        ev.result.starts_with("failure: 404"),
        "404 must surface as 'failure: 404 ...', got {:?}",
        ev.result
    );
}

#[tokio::test]
async fn consecutive_requests_record_distinct_audit_events() {
    let audit = Arc::new(AuditLogger::new(100));
    let app = build_app(audit.clone());

    for n in 0..3 {
        let req = HttpRequest::builder()
            .method("GET")
            .uri("/ok")
            .header("x-request-id", format!("trace-{n}"))
            .body(Body::empty())
            .expect("build request");
        let resp: Response = app.clone().oneshot(req).await.expect("response");
        assert_eq!(resp.status(), StatusCode::OK);
    }

    let events = audit.get_events().await;
    assert_eq!(events.len(), 3, "every request must produce one audit row");
    let ids: Vec<&str> = events.iter().map(|e| e.request_id.as_str()).collect();
    assert_eq!(ids, vec!["trace-0", "trace-1", "trace-2"]);
}
