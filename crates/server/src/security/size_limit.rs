//!
//! Wraps `tower_http::limit::RequestBodyLimitLayer` so the limit can be
//! applied to the protected routes via `axum::Router::layer`. The layer
//! rejects requests whose body exceeds the configured byte count with
//! HTTP 413 Payload Too Large.

use axum::Router;
use tower_http::limit::RequestBodyLimitLayer;

/// Default body size limit: 1 MiB.
pub const DEFAULT_BODY_LIMIT_BYTES: usize = 1_048_576;

/// Apply the request-body-size limit layer to an Axum router.
///
/// `limit_bytes` is the maximum accepted body size; requests larger
/// than this are rejected with HTTP 413.
pub fn with_body_size_limit(router: Router, limit_bytes: usize) -> Router {
    router.layer(RequestBodyLimitLayer::new(limit_bytes))
}

/// Default-limit convenience wrapper using [`DEFAULT_BODY_LIMIT_BYTES`].
pub fn with_default_body_limit(router: Router) -> Router {
    with_body_size_limit(router, DEFAULT_BODY_LIMIT_BYTES)
}

#[cfg(test)]
mod tests {
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
}
