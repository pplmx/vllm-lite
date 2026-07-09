#![allow(clippy::module_name_repetitions)]
//!
//! Wraps `tower_http::limit::RequestBodyLimitLayer` so the limit can be
//! applied to the protected routes via `axum::Router::layer`.
//!
//! The layer rejects requests whose body exceeds the configured byte
//! count with HTTP 413 Payload Too Large.

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

// Unit + integration tests are extracted to `tests.rs` (sibling)
// to keep this size-limit module under the 800-line soft cap.
// They cover the custom-limit layer (under-limit OK, over-limit
// 413) and the default 1 MiB helper (512 KiB OK, 2 MiB 413).
#[cfg(test)]
mod tests;
