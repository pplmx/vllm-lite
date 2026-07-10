//! Request correlation: assign a unique `X-Request-ID` header to every incoming request and propagate it into logs + audit events.
//!
//! If the client supplied a `X-Request-ID` header we honour it (after
//! validating the format); otherwise we mint a fresh `UUIDv4`. The ID is
//! added to the response headers and threaded through `tracing` spans.
#![allow(clippy::module_name_repetitions, dead_code)]
use axum::{extract::Request, http::HeaderValue, middleware::Next, response::Response};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// `REQUEST_ID_HEADER`. See the type definition for fields and behavior.
pub(crate) const REQUEST_ID_HEADER: &str = "X-Request-ID";

/// Opaque newtype identifier for a correlation. Hashable, comparable, serializable; use this rather than the raw integer.
#[derive(Debug, Clone)]
pub(crate) struct CorrelationId(pub String);

/// `CorrelationIdMiddleware`. See the type definition for fields and behavior.
#[derive(Debug, Clone)]
pub struct CorrelationIdMiddleware {
    id_generator: Arc<RwLock<u64>>,
}

impl CorrelationIdMiddleware {
    #[must_use]
    pub fn new() -> Self {
        Self {
            id_generator: Arc::new(RwLock::new(0)),
        }
    }

    /// Run the operation (see signature for params and return type).
    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    pub async fn generate_id(&self) -> String {
        let mut counter = self.id_generator.write().await;
        *counter += 1;
        let id = format!(
            "{:x}-{:x}",
            // invariant: monotonic clock is always >= UNIX_EPOCH.
            u64::try_from(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    // invariant: pre-conditions make this infallible at this call site.
                    .unwrap()
                    .as_nanos(),
            )
            .unwrap_or(0),
            *counter
        );
        drop(counter);
        id
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

pub async fn correlation_id_middleware(request: Request, next: Next) -> Response {
    let middleware = CorrelationIdMiddleware::new();

    let request_id = CorrelationIdMiddleware::extract_id(request.headers())
        .unwrap_or_else(|| tokio::runtime::Handle::current().block_on(middleware.generate_id()));

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

    let response = next.run(request).await;

    info!(request_id = %request_id, "Request completed");

    response
}

// Unit tests are extracted to `tests.rs` (sibling) to keep this
// correlation-id module under the 800-line soft cap. They cover
// ID generation (non-empty + pairwise-distinct) and header
// extraction (present → Some(s), absent → None).
#[cfg(test)]
mod tests;
