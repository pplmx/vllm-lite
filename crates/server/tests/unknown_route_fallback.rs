//! Router-level 404 fallback (RIL ISS-125).
//!
//! Before the fix, an unmatched path (a typo'd route, an SDK pointed at
//! the wrong URL, or a `curl` miss) reached axum's default fallback —
//! a **plain-text** `404 Not Found` body. `OpenAI` SDKs parse every
//! non-2xx response as the `{error: {message, type, code}}` envelope, so
//! the plain-text body failed JSON decoding and surfaced as an opaque
//! client-side parse error instead of the actionable
//! `{"error":{"message":"Not Found","type":"invalid_request_error"}}`.
//!
//! This test drives the *real* production router (`vllm_server::app::
//! build_app`) through `tower::ServiceExt::oneshot` — not the handler in
//! isolation — because `fallback` is a router-level registration.

use axum::body::Body;
use axum::http::{Request as HttpRequest, StatusCode};
use tower::ServiceExt;
use vllm_model::config::Architecture;
use vllm_server::app::build_app;
use vllm_server::security::audit::AuditLogger;
use vllm_server::security::cors::CorsConfig;
use vllm_server::test_fixtures::api_state;

#[tokio::test]
async fn unmatched_path_returns_openai_error_envelope() {
    let state = api_state(Architecture::Qwen3);
    let app = build_app(
        state,
        None,
        std::sync::Arc::new(AuditLogger::new(100)),
        &CorsConfig::default(),
    );

    let response = app
        .oneshot(
            HttpRequest::builder()
                .uri("/v1/does-not-exist")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    assert!(
        content_type.starts_with("application/json"),
        "fallback must answer JSON, not axum's plain-text default (got {content_type:?})"
    );

    let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    let json: serde_json::Value =
        serde_json::from_slice(&body).expect("fallback body must be valid JSON");
    assert_eq!(json["error"]["message"], "Not Found");
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn known_route_still_dispatches_normally() {
    // The fallback must not shadow real routes: GET /v1/models still
    // answers 200 with the model list (RIL ISS-118 envelope).
    let state = api_state(Architecture::Qwen3);
    let app = build_app(
        state,
        None,
        std::sync::Arc::new(AuditLogger::new(100)),
        &CorsConfig::default(),
    );

    let response = app
        .oneshot(
            HttpRequest::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    let json: serde_json::Value =
        serde_json::from_slice(&body).expect("model list body must be valid JSON");
    assert_eq!(json["object"], "list");
}

// RIL ISS-145: a wrong HTTP verb on a known path (e.g. `POST /v1/models`,
// `GET /v1/chat/completions`) previously reached axum's default
// `405 Method Not Allowed` — a **plain-text** body like the pre-fix 404
// (RIL ISS-125) — which breaks the `{error: {...}}` parsing every OpenAI
// SDK performs on non-2xx responses. The envelope must be returned with
// the `405` status just like the 404 fallback.
#[tokio::test]
async fn wrong_method_returns_openai_error_envelope() {
    let state = api_state(Architecture::Qwen3);
    let app = build_app(
        state,
        None,
        std::sync::Arc::new(AuditLogger::new(100)),
        &CorsConfig::default(),
    );

    let response = app
        .oneshot(
            HttpRequest::builder()
                .method("POST")
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    assert!(
        content_type.starts_with("application/json"),
        "405 must answer JSON, not axum's plain-text default (got {content_type:?})"
    );

    let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    let json: serde_json::Value =
        serde_json::from_slice(&body).expect("405 body must be valid JSON");
    assert_eq!(json["error"]["message"], "Method Not Allowed");
    assert_eq!(json["error"]["type"], "invalid_request_error");
}
