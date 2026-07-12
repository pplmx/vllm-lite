//! Unit tests for the `correlation` middleware.
//!
//! Locks in two contracts:
//!
//! 1. **ID generation**: `generate_id()` returns non-empty,
//!    pairwise-distinct IDs. The id format is
//!    `<unix-nanos-hex>-<counter-hex>`; the counter disambiguates
//!    ids minted in the same nanosecond.
//! 2. **Header extraction**: `extract_id(&headers)` returns
//!    `Some(s)` when the `REQUEST_ID_HEADER` is present, and
//!    `None` when it is absent.
use super::*;

#[test]
fn test_generate_id_non_empty_and_distinct() {
    let middleware = CorrelationIdMiddleware::new();
    let id1 = middleware.generate_id();
    let id2 = middleware.generate_id();

    assert!(!id1.is_empty());
    assert!(!id2.is_empty());
    assert_ne!(id1, id2);
}

#[test]
fn test_generate_id_format() {
    // `<unix-nanos-hex>-<counter-hex>` — both halves hex-encoded.
    let middleware = CorrelationIdMiddleware::new();
    let id = middleware.generate_id();
    let parts: Vec<&str> = id.split('-').collect();
    assert_eq!(
        parts.len(),
        2,
        "id '{id}' must have exactly one '-' separator"
    );
    assert!(
        parts[0].chars().all(|c| c.is_ascii_hexdigit()),
        "id '{id}' nanos half must be hex"
    );
    assert!(
        parts[1].chars().all(|c| c.is_ascii_hexdigit()),
        "id '{id}' counter half must be hex"
    );
}

#[test]
fn test_extract_id_present() {
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(REQUEST_ID_HEADER, "test-id-123".parse().unwrap());

    let id = CorrelationIdMiddleware::extract_id(&headers);
    assert_eq!(id, Some("test-id-123".to_string()));
}

#[test]
fn test_extract_id_missing() {
    let headers = axum::http::HeaderMap::new();
    let id = CorrelationIdMiddleware::extract_id(&headers);
    assert_eq!(id, None);
}
