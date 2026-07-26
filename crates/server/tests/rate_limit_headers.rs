//! Integration tests for rate-limit response headers.
//!
//! Verifies that the token-bucket rate limiter in [`vllm_server::auth`]
//! emits the standard `Retry-After`, `X-RateLimit-Remaining`, and
//! `X-RateLimit-Limit` headers on both successful and rate-limited
//! responses.
//!
//! Run with: `cargo nextest run -p vllm-server --test rate_limit_headers`

use axum::{
    Router,
    http::{HeaderName, Request, StatusCode, header::AUTHORIZATION},
    middleware::from_fn_with_state,
    response::IntoResponse,
    routing::get,
};
use std::sync::Arc;
use tower::ServiceExt;
use vllm_server::auth::{AuthMiddleware, auth_middleware};

/// Build a minimal axum router with auth middleware mounted.
fn app(auth: Arc<AuthMiddleware>) -> Router {
    Router::new()
        .route("/", get(|| async { "ok".into_response() }))
        .route_layer(from_fn_with_state(auth, auth_middleware))
}

#[tokio::test]
async fn test_successful_request_includes_rate_limit_headers() {
    let auth = Arc::new(AuthMiddleware::new(vec!["test_key".to_string()], 10, 60));
    let app = app(auth);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/")
                .header(AUTHORIZATION, "Bearer test_key")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let remaining = response
        .headers()
        .get("x-ratelimit-remaining")
        .expect("X-RateLimit-Remaining header should be present");
    assert_eq!(remaining.to_str().unwrap(), "9");

    let limit = response
        .headers()
        .get("x-ratelimit-limit")
        .expect("X-RateLimit-Limit header should be present");
    assert_eq!(limit.to_str().unwrap(), "10");

    // No Retry-After on success.
    assert!(response.headers().get("retry-after").is_none());
}

#[tokio::test]
async fn test_rate_limited_response_includes_retry_after() {
    let auth = Arc::new(AuthMiddleware::new(vec!["test_key".to_string()], 2, 60));
    let app = app(auth);

    // First two requests succeed.
    for _ in 0..2 {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/")
                    .header(AUTHORIZATION, "Bearer test_key")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    // Third request should be rate-limited (429).
    let response = app
        .oneshot(
            Request::builder()
                .uri("/")
                .header(AUTHORIZATION, "Bearer test_key")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);

    // Retry-After header should be present on 429.
    let retry_after = response
        .headers()
        .get(HeaderName::from_static("retry-after"))
        .expect("Retry-After header should be present on 429");
    let retry_secs: u64 = retry_after.to_str().unwrap().parse().unwrap();
    assert!(retry_secs >= 1, "Retry-After should be at least 1 second");

    // X-RateLimit-Remaining should be 0 on 429.
    let remaining = response
        .headers()
        .get("x-ratelimit-remaining")
        .expect("X-RateLimit-Remaining header should be present on 429");
    assert_eq!(remaining.to_str().unwrap(), "0");

    // X-RateLimit-Limit should match the capacity.
    let limit = response
        .headers()
        .get("x-ratelimit-limit")
        .expect("X-RateLimit-Limit header should be present on 429");
    assert_eq!(limit.to_str().unwrap(), "2");
}

#[tokio::test]
async fn test_separate_keys_have_independent_rate_limits() {
    let auth = Arc::new(AuthMiddleware::new(
        vec!["key_a".to_string(), "key_b".to_string()],
        1,
        60,
    ));
    let app = app(auth);

    // key_a exhausts its quota.
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/")
                .header(AUTHORIZATION, "Bearer key_a")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/")
                .header(AUTHORIZATION, "Bearer key_a")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);

    // key_b should still be allowed (independent bucket).
    let response = app
        .oneshot(
            Request::builder()
                .uri("/")
                .header(AUTHORIZATION, "Bearer key_b")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let remaining_b = response
        .headers()
        .get("x-ratelimit-remaining")
        .unwrap()
        .to_str()
        .unwrap();
    assert_eq!(remaining_b, "0"); // 1 capacity, 1 consumed
}

#[tokio::test]
async fn test_unauthorized_request_has_no_rate_limit_headers() {
    let auth = Arc::new(AuthMiddleware::new(vec!["test_key".to_string()], 10, 60));
    let app = app(auth);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/")
                .header(AUTHORIZATION, "Bearer wrong_key")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    // No rate-limit headers on auth failures.
    assert!(response.headers().get("x-ratelimit-remaining").is_none());
    assert!(response.headers().get("x-ratelimit-limit").is_none());
}

#[tokio::test]
async fn test_cost_aware_rate_limiting_with_large_body() {
    // A request with a large prompt body should consume more tokens.
    // With capacity=5 and a body costing 4+1=5 tokens, one such request
    // should exhaust the bucket, while a small request would use only 1.
    let auth = Arc::new(AuthMiddleware::new(vec!["test_key".to_string()], 5, 60));
    let app = app(auth);

    // Large request: prompt has 3 words + max_tokens 2 = cost 5
    let large_body = r#"{"prompt": "hello world foo", "max_tokens": 2}"#;
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/")
                .header(AUTHORIZATION, "Bearer test_key")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(large_body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // Bucket is now empty (5 cost consumed). Next request should be 429
    // even though it's a small body costing only 1.
    let response = app
        .oneshot(
            Request::builder()
                .uri("/")
                .header(AUTHORIZATION, "Bearer test_key")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
}

#[tokio::test]
async fn test_small_body_costs_one_token() {
    // Empty body (GET-like) should cost 1 token.
    let auth = Arc::new(AuthMiddleware::new(vec!["test_key".to_string()], 3, 60));
    let app = app(auth);

    for _ in 0..3 {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/")
                    .header(AUTHORIZATION, "Bearer test_key")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    // Fourth request should be rate-limited
    let response = app
        .oneshot(
            Request::builder()
                .uri("/")
                .header(AUTHORIZATION, "Bearer test_key")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
}
