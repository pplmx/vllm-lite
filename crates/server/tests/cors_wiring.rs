//! CORS layer wiring test.
//!
//! Production-readiness recommendation §9: the project ships no
//! CORS layer today. When browser-direct access is enabled, an
//! explicit allowlist is required (never `*` + credentials). This
//! test guards the wiring so a future refactor that drops the
//! layer fails CI.
//!
//! Three invariants are checked against a real axum router:
//!
//! 1. **Closed default**: a router built with `CorsConfig::default()`
//!    does NOT emit `Access-Control-Allow-Origin` in response to a
//!    preflight. Browser-direct callers will be blocked at the
//!    browser; server-to-server SDKs are unaffected (they don't
//!    send preflights).
//! 2. **Explicit allowlist**: an `allow_origins = ["https://app.example.com"]`
//!    config emits the matching header on a preflight.
//! 3. **`Origin` request header echoed**: when the request carries
//!    an `Origin` header that matches the allowlist, the response
//!    carries it back in `Access-Control-Allow-Origin`. The
//!    preflight also gets `Access-Control-Allow-Methods` populated
//!    from `CorsConfig::allow_methods`.
//!
//! We deliberately don't mount auth/body-limit/correlation layers
//! here — the CORS wiring test only needs to exercise the CORS
//! layer in isolation. The full router composition is covered by
//! `body_limit_wiring.rs` and the production-readiness tests.

#![cfg(test)]

use axum::Router;
use axum::body::Body;
use axum::http::{Request as HttpRequest, StatusCode};
use axum::routing::get;
use tower::ServiceExt;
use vllm_server::security::cors::{CorsConfig, with_cors};

async fn ping() -> &'static str {
    "pong"
}

fn build_app(config: CorsConfig) -> Router {
    with_cors(Router::new().route("/ping", get(ping)), config)
}

fn preflight_request(origin: &str) -> HttpRequest<Body> {
    HttpRequest::builder()
        .method("OPTIONS")
        .uri("/ping")
        .header("origin", origin)
        .header("access-control-request-method", "GET")
        .header("access-control-request-headers", "content-type")
        .body(Body::empty())
        .expect("build preflight")
}

#[tokio::test]
async fn closed_default_emits_no_allow_origin() {
    let app = build_app(CorsConfig::default());
    let req = preflight_request("https://app.example.com");
    let resp = app.oneshot(req).await.expect("response");

    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "preflight OPTIONS must reach the handler (router is closed, not the server)"
    );
    assert!(
        resp.headers().get("access-control-allow-origin").is_none(),
        "closed CORS layer must NOT emit Access-Control-Allow-Origin; \
         found it in {resp_headers:?}",
        resp_headers = resp.headers()
    );
}

#[tokio::test]
async fn explicit_allowlist_emits_matching_origin() {
    let mut config = CorsConfig::default();
    config.allow_origins = vec!["https://app.example.com".to_string()];
    config.allow_methods = vec!["GET".to_string(), "POST".to_string()];
    config.allow_headers = vec!["content-type".to_string()];
    let app = build_app(config);

    let req = preflight_request("https://app.example.com");
    let resp = app.oneshot(req).await.expect("response");

    assert_eq!(resp.status(), StatusCode::OK);
    let allow_origin = resp
        .headers()
        .get("access-control-allow-origin")
        .expect("explicit allowlist must emit Access-Control-Allow-Origin")
        .to_str()
        .expect("ascii");
    assert_eq!(allow_origin, "https://app.example.com");

    let allow_methods = resp
        .headers()
        .get("access-control-allow-methods")
        .expect("explicit methods must emit Access-Control-Allow-Methods")
        .to_str()
        .expect("ascii");
    // tower-http folds the list comma-separated.
    assert!(
        allow_methods.contains("GET") && allow_methods.contains("POST"),
        "expected GET and POST in {allow_methods}"
    );
}

#[tokio::test]
async fn origin_not_in_allowlist_is_dropped() {
    let mut config = CorsConfig::default();
    config.allow_origins = vec!["https://allowed.example.com".to_string()];
    let app = build_app(config);

    let req = preflight_request("https://attacker.example.com");
    let resp = app.oneshot(req).await.expect("response");

    // Either tower-http omits the header (browser blocks) OR the
    // header doesn't match — both fail closed. The key invariant
    // is that the attacker's origin is NOT echoed back.
    if let Some(value) = resp.headers().get("access-control-allow-origin") {
        let echoed = value.to_str().unwrap_or("");
        assert_ne!(
            echoed, "https://attacker.example.com",
            "non-allowlisted origin must NOT be echoed back as Access-Control-Allow-Origin"
        );
    }
}
