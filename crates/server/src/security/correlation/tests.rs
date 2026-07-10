//! Unit tests for the `correlation` middleware.
//!
//! Locks in two contracts:
//!
//! 1. **ID generation**: `generate_id()` returns non-empty,
//!    pairwise-distinct IDs (UUID-derived; collision probability
//!    is negligible but we still assert `id1 != id2`).
//! 2. **Header extraction**: `extract_id(&headers)` returns
//!    `Some(s)` when the `REQUEST_ID_HEADER` is present, and
//!    `None` when it is absent.
use super::*;

#[tokio::test]
async fn test_generate_id() {
    let middleware = CorrelationIdMiddleware::new();
    let id1 = middleware.generate_id().await;
    let id2 = middleware.generate_id().await;

    assert!(!id1.is_empty());
    assert!(!id2.is_empty());
    assert_ne!(id1, id2);
}

#[tokio::test]
async fn test_extract_id() {
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(REQUEST_ID_HEADER, "test-id-123".parse().unwrap());

    let id = CorrelationIdMiddleware::extract_id(&headers);
    assert_eq!(id, Some("test-id-123".to_string()));
}

#[tokio::test]
async fn test_extract_id_missing() {
    let headers = axum::http::HeaderMap::new();
    let id = CorrelationIdMiddleware::extract_id(&headers);
    assert_eq!(id, None);
}
