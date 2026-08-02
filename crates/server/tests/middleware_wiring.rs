//! Production middleware-stack ordering tests.
//!
//! `vllm_server::app::build_app` is the *real* production router
//! assembly (routes + middleware stack). These tests pin the stack
//! invariants documented in `app.rs`:
//!
//! 1. body-size limit is ABOVE auth — an oversized body is rejected
//!    with 413 *before* the auth middleware buffers it (auth's
//!    `to_bytes(body, 1 MiB)` would otherwise silently truncate the
//!    body and the request would proceed as an empty-body 400/200);
//! 2. correlation is OUTERMOST of the security stack — rejected
//!    responses (413 / 401 / 429) still carry `X-Request-ID`;
//! 3. audit is above both so rejections are recorded (indirectly
//!    verified here via the 413/401 paths — the ring buffer assertion
//!    lives in the audit wiring tests).

#![cfg(test)]

use std::sync::Arc;

use axum::body::Body;
use axum::extract::Request;
use axum::http::StatusCode;
use axum::response::Response;
use http_body_util::BodyExt;
use tower::ServiceExt;
use vllm_server::auth::AuthMiddleware;
use vllm_server::security::cors::CorsConfig;
use vllm_server::security::size_limit::DEFAULT_BODY_LIMIT_BYTES;
use vllm_server::test_fixtures::api_state;

const TEST_KEY: &str = "sk-test-1234";

/// Build the production router with API keys enabled.
fn production_app() -> axum::Router {
    let state = api_state(vllm_model::config::Architecture::Qwen3);
    let auth = Some(Arc::new(AuthMiddleware::new(
        vec![TEST_KEY.to_string()],
        100,
        60,
    )));
    let audit = Arc::new(vllm_server::security::audit::AuditLogger::new(1000));
    vllm_server::app::build_app(state, auth, audit, &CorsConfig::default())
}

fn post(uri: &str, body: Body, key: Option<&str>) -> Request<Body> {
    let mut builder = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json");
    if let Some(k) = key {
        builder = builder.header("authorization", format!("Bearer {k}"));
    }
    builder.body(body).unwrap()
}

async fn send(app: &axum::Router, req: Request<Body>) -> Response {
    app.clone().oneshot(req).await.unwrap()
}

#[tokio::test]
async fn oversized_body_returns_413_not_truncated_request() {
    let app = production_app();
    // 2 MiB > 1 MiB default limit. The body-size limit must reject
    // with 413 BEFORE auth buffers/truncates the body.
    let body = vec![b'x'; 2 * DEFAULT_BODY_LIMIT_BYTES];
    let resp = send(
        &app,
        post("/v1/completions", Body::from(body), Some(TEST_KEY)),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::PAYLOAD_TOO_LARGE,
        "oversized body must be rejected with 413 (not silently truncated by auth)"
    );
}

#[tokio::test]
async fn oversized_body_413_carries_request_id() {
    let app = production_app();
    let body = vec![b'x'; 2 * DEFAULT_BODY_LIMIT_BYTES];
    let resp = send(
        &app,
        post("/v1/completions", Body::from(body), Some(TEST_KEY)),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert!(
        resp.headers().contains_key("x-request-id"),
        "413 must carry X-Request-ID (correlation layer is outermost)"
    );
}

#[tokio::test]
async fn unauthorized_request_carries_request_id() {
    let app = production_app();
    // No Authorization header -> 401 from auth middleware. The 401
    // must still carry X-Request-ID (correlation runs outside auth).
    let resp = send(
        &app,
        post("/v1/completions", Body::from(r#"{"prompt": "hi"}"#), None),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    assert!(
        resp.headers().contains_key("x-request-id"),
        "401 must carry X-Request-ID (correlation layer is outermost)"
    );
}

#[tokio::test]
async fn under_limit_request_passes_size_limit_and_auth() {
    let app = production_app();
    // Small valid-shaped body with a valid key: must pass both auth
    // and the size limit, reaching the handler (which returns a
    // handler-level 4xx for a stub engine — NOT 401/413).
    let resp = send(
        &app,
        post(
            "/v1/completions",
            Body::from(r#"{"model":"test","prompt":"hi","max_tokens":4}"#),
            Some(TEST_KEY),
        ),
    )
    .await;
    let status = resp.status();
    assert!(
        status != StatusCode::UNAUTHORIZED && status != StatusCode::PAYLOAD_TOO_LARGE,
        "valid under-limit request must pass auth + size limit (got {status})"
    );
}

#[tokio::test]
async fn invalid_key_is_rejected_by_auth() {
    let app = production_app();
    let resp = send(
        &app,
        post(
            "/v1/completions",
            Body::from(r#"{"prompt": "hi"}"#),
            Some("sk-wrong-key"),
        ),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn request_id_still_propagates_on_success_path() {
    let app = production_app();
    let resp = send(
        &app,
        post(
            "/v1/completions",
            Body::from(r#"{"model":"test","prompt":"hi","max_tokens":4}"#),
            Some(TEST_KEY),
        ),
    )
    .await;
    assert!(
        resp.headers().contains_key("x-request-id"),
        "success-path responses must carry X-Request-ID"
    );
    // Drain the body so the response is fully consumed.
    let (_, body) = resp.into_parts();
    let _ = BodyExt::collect(body).await.unwrap().to_bytes();
}

/// Regression (RIL ISS-018): K8s liveness/readiness probes are
/// unauthenticated GETs (the Helm chart's probes carry no Authorization
/// header). When API-key auth is configured, the health/readiness paths
/// must be EXEMPT from auth so probes return 200 — otherwise every probe
/// gets 401 and the pod never becomes ready (restart loop). The security
/// boundary for unauthenticated deployment is the startup refusal to bind
/// a non-loopback address without keys (main.rs), not per-request auth on
/// health. Sensitive ops endpoints (/debug/*, /shutdown, /metrics) stay
/// authenticated.
#[tokio::test]
async fn health_probes_are_exempt_from_api_key_auth() {
    let app = production_app(); // auth configured with TEST_KEY
    for path in ["/health/live", "/health/ready", "/health", "/ready"] {
        let req = Request::builder()
            .method("GET")
            .uri(path)
            .body(Body::empty())
            .unwrap();
        let resp = send(&app, req).await;
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "{path} must be reachable WITHOUT auth so unauthenticated K8s \
             probes work (got {} => RIL ISS-018)",
            resp.status()
        );
    }
}

/// Companion to the exemption test: a NON-health endpoint must still
/// require auth (the exemption is narrow, not a wholesale auth bypass).
#[tokio::test]
async fn non_health_endpoints_still_require_auth() {
    let app = production_app();
    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let resp = send(&app, req).await;
    assert_eq!(
        resp.status(),
        StatusCode::UNAUTHORIZED,
        "non-health endpoints must still require API-key auth (got {})",
        resp.status()
    );
}
