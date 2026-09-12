#![allow(clippy::module_name_repetitions)]
//!
//! Wraps `tower_http::limit::RequestBodyLimitLayer` so the limit can be
//! applied to the protected routes via `axum::Router::layer`.
//!
//! The layer rejects requests whose body exceeds the configured byte
//! count with HTTP 413 Payload Too Large — and that 413 is rewritten
//! into the `OpenAI` error envelope (`{error:{message,type,code}}`) so an
//! `OpenAI` SDK client that parses `error.message` sees a structured error
//! instead of tower-http's plain-text `"length limit exceeded"` (RIL
//! ISS-118). The rewrite runs OUTSIDE the limit layer (so it sees the
//! 413 response), inside correlation/audit (so rejected requests still
//! carry `X-Request-ID` and are audited).

use axum::{
    Router,
    body::Body,
    extract::Request,
    http::{StatusCode, header::CONTENT_TYPE},
    middleware::{Next, from_fn},
    response::Response,
};
use tower_http::limit::RequestBodyLimitLayer;

use crate::openai::types::ErrorResponse;

/// Default body size limit: 1 MiB.
pub const DEFAULT_BODY_LIMIT_BYTES: usize = 1_048_576;

/// Apply the request-body-size limit layer to an Axum router.
///
/// `limit_bytes` is the maximum accepted body size; requests larger
/// than this are rejected with HTTP 413 carrying the `OpenAI` error
/// envelope (via [`size_limit_413_envelope`]).
pub fn with_body_size_limit(router: Router, limit_bytes: usize) -> Router {
    router
        .layer(RequestBodyLimitLayer::new(limit_bytes))
        .layer(from_fn(size_limit_413_envelope))
}

/// Rewrite `RequestBodyLimitLayer`'s 413 into the `OpenAI` error envelope.
///
/// tower-http rejects an over-limit body with `413 Payload Too Large`
/// and a `text/plain; charset=utf-8` body `"length limit exceeded"`.
/// Clients speaking the `OpenAI` contract parse `{error:{message,...}}`
/// and treat the plain body as opaque, so the 413 mirrors the envelope
/// the other error paths emit (RIL ISS-118). Headers from the inner
/// layers (e.g. `X-Request-ID`, added outside this layer) are preserved
/// by rebuilding the parts.
async fn size_limit_413_envelope(request: Request, next: Next) -> Response {
    let response = next.run(request).await;
    if response.status() == StatusCode::PAYLOAD_TOO_LARGE {
        let mut builder = Response::builder().status(StatusCode::PAYLOAD_TOO_LARGE);
        for (name, value) in response.headers() {
            builder = builder.header(name, value.clone());
        }
        let body = serde_json::to_string(&ErrorResponse::with_code(
            "request body exceeds the server limit",
            "invalid_request_error",
            "request_too_large",
        ))
        .expect("ErrorResponse always serializes");
        let mut rewritten = builder
            // invariant: a Response with a static string body cannot fail to build.
            .body(Body::from(body))
            .unwrap();
        // `builder.header` APPENDS, so the copied `text/plain` content-type
        // from tower-http's 413 would survive alongside ours; insert (which
        // replaces) makes application/json the sole content-type.
        rewritten
            .headers_mut()
            .insert(CONTENT_TYPE, "application/json".parse().unwrap());
        rewritten
    } else {
        response
    }
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
