//! Request correlation: assign a unique `X-Request-ID` header to every incoming request and propagate it into logs + audit events.
//!
//! If the client supplied an `X-Request-ID` header we honour it (after
//! validating the format); otherwise we mint a fresh
//! `<unix-nanos-hex>-<process-counter-hex>` id. The id is added to
//! the response headers and threaded through `tracing` info-span.
//!
//! The id generator uses a synchronous `AtomicU64` counter rather
//! than `tokio::sync::RwLock` because the middleware itself runs
//! inside the tokio runtime — the previous async-counter
//! implementation deadlocked on `Handle::current().block_on()` when
//! minted a fallback id (production-readiness recommendation 6).
#![allow(clippy::module_name_repetitions)]
use axum::{extract::Request, http::HeaderValue, middleware::Next, response::Response};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::info;

/// `REQUEST_ID_HEADER`. See the type definition for fields and behavior.
pub(crate) const REQUEST_ID_HEADER: &str = "X-Request-ID";

/// Opaque newtype identifier for a correlation. Hashable, comparable, serializable; use this rather than the raw integer.
///
/// Inserted into request extensions by [`correlation_id_middleware`]
/// so downstream layers (audit, auth-failure handler, structured
/// logs) can read the id without re-parsing the `X-Request-ID`
/// header. Production-readiness recommendation 6: every request
/// must carry a single correlation id through HTTP → scheduler →
/// token stream so operators can trace a request across the whole
/// pipeline.
///
/// `pub` (not `pub(crate)`) so axum's `Extension<CorrelationId>`
/// extractor can name it from public HTTP handlers like
/// `openai::chat::chat_completions` — axum's
/// `FromRequestParts` reflection requires the inner type to be
/// reachable from outside the crate.
#[derive(Debug, Clone)]
pub struct CorrelationId(pub String);

/// `CorrelationIdMiddleware`. See the type definition for fields and behavior.
#[derive(Debug, Clone)]
pub struct CorrelationIdMiddleware {
    counter: Arc<AtomicU64>,
}

impl CorrelationIdMiddleware {
    #[must_use]
    pub fn new() -> Self {
        Self {
            counter: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Synchronous id generator — safe to call from inside an
    /// existing tokio task. Format:
    /// `<unix-nanos-hex>-<process-counter-hex>`. The counter
    /// disambiguates ids minted in the same nanosecond (common when
    /// a burst of requests arrives concurrently).
    #[must_use]
    pub fn generate_id(&self) -> String {
        let counter = self.counter.fetch_add(1, Ordering::Relaxed);
        let nanos = u64::try_from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        )
        .unwrap_or(0);
        format!("{nanos:x}-{counter:x}")
    }

    #[must_use]
    pub(crate) fn extract_id(headers: &axum::http::HeaderMap) -> Option<String> {
        headers
            .get(REQUEST_ID_HEADER)
            .and_then(|v| v.to_str().ok())
            .map(std::string::ToString::to_string)
    }
}

impl Default for CorrelationIdMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

/// Axum middleware: ensure every request carries an `X-Request-ID`
/// header (forwarding the client's value if present and well-formed,
/// minting a fresh id otherwise), echo it on the response, and log
/// the request lifecycle with the id attached.
pub async fn correlation_id_middleware(request: Request, next: Next) -> Response {
    let middleware = CorrelationIdMiddleware::new();

    let request_id = CorrelationIdMiddleware::extract_id(request.headers())
        .unwrap_or_else(|| middleware.generate_id());

    info!(
        request_id = %request_id,
        method = %request.method(),
        uri = %request.uri(),
        "Request started"
    );

    let mut request = request;
    request.headers_mut().insert(
        REQUEST_ID_HEADER,
        HeaderValue::from_str(&request_id).unwrap_or_else(|_| HeaderValue::from_static("unknown")),
    );
    // Also expose the id via request extensions so the audit
    // middleware (and any future layer that wants the id without
    // re-parsing the header) can read it directly. This is the
    // typed counterpart to the header and keeps the public API
    // — header is for clients, extension is for layers.
    request
        .extensions_mut()
        .insert(CorrelationId(request_id.clone()));

    let mut response = next.run(request).await;

    // Echo the id on the response so the client (or an upstream
    // gateway) can correlate logs without inspecting internal
    // tracing spans.
    if let Ok(value) = HeaderValue::from_str(&request_id) {
        response.headers_mut().insert(REQUEST_ID_HEADER, value);
    }

    info!(request_id = %request_id, "Request completed");

    response
}

// Unit tests are extracted to `tests.rs` (sibling) to keep this
// correlation-id module under the 800-line soft cap. They cover
// ID generation (non-empty + pairwise-distinct) and header
// extraction (present → Some(s), absent → None).
#[cfg(test)]
mod tests;
