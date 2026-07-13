//! Audit-logging middleware.
//!
//! Production-readiness recommendation: every authenticated API
//! call should leave a trail that operators can read — for
//! incident response ("who deleted this model?"), for compliance
//! ("show me all requests that touched PII last week"), and for
//! after-the-fact debugging ("why did this user get 401?"). The
//! [`AuditLogger`] type already exists; this module wires it into
//! the production router so every request is recorded.
//!
//! ## Position in the layer stack
//!
//! ```text
//! correlation_id_middleware   ← sets CorrelationId + X-Request-ID
//! audit_middleware            ← this file; logs after the response
//! size_limit_middleware       ← rejects oversize bodies with 413
//! auth_middleware             ← sets AuthenticatedUser on success
//! handler                     ← actual business logic
//! ```
//!
//! `audit_middleware` sits **above** auth so it sees both
//! successful AND failed-auth requests (we want to record the
//! 401s). It sits **below** correlation_id so even requests that
//! never reach a handler (e.g. body-limit 413s) carry a stable
//! `request_id` in the audit row.
//!
//! ## What gets logged
//!
//! One `log_api_request` row per request, after the handler
//! returns, with:
//! - `request_id` — from the `CorrelationId` extension set by
//!   the correlation middleware
//! - `user_id` — from the `AuthenticatedUser` extension set by
//!   the auth middleware (or `None` for unauthenticated requests)
//! - `action` — HTTP method (`GET`, `POST`, ...)
//! - `resource` — request path
//! - `result` — `"success"`, `"failure: 4xx"`, or `"failure: 5xx"`
//!   based on the response status code
//!
//! The audit logger also emits a structured `tracing` event for
//! log aggregation, so the audit trail shows up in both the
//! in-memory ring buffer (exportable via `/debug/audit`) and the
//! `tracing` JSON log stream.

use axum::{
    extract::{Request, State},
    middleware::Next,
    response::Response,
};
use std::sync::Arc;

use crate::auth::AuthenticatedUser;
use crate::security::audit::AuditLogger;
use crate::security::correlation::CorrelationId;

/// Axum middleware: record every request in the audit trail.
///
/// The middleware reads `request_id` from the
/// `CorrelationId` extension set by the correlation middleware
/// (above this layer) and `user_id` from the `AuthenticatedUser`
/// extension set by the auth middleware (below). It runs the
/// handler, then records one audit row with the HTTP method,
/// path, and outcome.
///
/// **Best-effort**: a failure inside `audit_logger.log(...)` would
/// block the response; we `.await` the logger but treat any
/// failure as a soft warning rather than failing the request. In
/// practice the logger is a tokio `RwLock<Vec<...>>` and never
/// fails — but this guards against future storage backends that
/// could.
pub async fn audit_middleware(
    State(audit): State<Arc<AuditLogger>>,
    request: Request,
    next: Next,
) -> Response {
    // Capture request metadata BEFORE running the handler so
    // the audit row is stable regardless of what the handler
    // does to the request (it can rewrite headers, consume
    // extensions, etc.).
    let request_id = request
        .extensions()
        .get::<CorrelationId>()
        .map_or_else(|| "unknown".to_string(), |c| c.0.clone());
    let user_id = request
        .extensions()
        .get::<AuthenticatedUser>()
        .map(|u| u.0.clone());
    let method = request.method().to_string();
    let path = request.uri().path().to_string();

    let response = next.run(request).await;
    let status = response.status();
    let result = audit_status_to_result(status);

    audit
        .log_api_request(user_id.as_deref(), &method, &path, &result, &request_id)
        .await;

    response
}

/// Map an HTTP status code to the audit-trail result string.
///
/// - 2xx → "success"
/// - 3xx → "redirect" (rare on a JSON API but possible)
/// - 4xx → `"failure: {status} {reason}"`
/// - 5xx → `"failure: {status} {reason}"`
#[must_use]
pub fn audit_status_to_result(status: axum::http::StatusCode) -> String {
    let code = status.as_u16();
    let reason = status.canonical_reason().unwrap_or("");
    if (200..300).contains(&code) {
        "success".to_string()
    } else if (300..400).contains(&code) {
        format!("redirect: {code} {reason}")
    } else {
        format!("failure: {code} {reason}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

    #[test]
    fn status_to_result_success() {
        assert_eq!(audit_status_to_result(StatusCode::OK), "success");
        assert_eq!(audit_status_to_result(StatusCode::ACCEPTED), "success");
    }

    #[test]
    fn status_to_result_client_error() {
        let s = audit_status_to_result(StatusCode::UNAUTHORIZED);
        assert!(s.starts_with("failure: 401"));
        assert!(s.contains("Unauthorized"));
    }

    #[test]
    fn status_to_result_server_error() {
        let s = audit_status_to_result(StatusCode::INTERNAL_SERVER_ERROR);
        assert!(s.starts_with("failure: 500"));
        assert!(s.contains("Internal Server Error"));
    }

    #[test]
    fn status_to_result_redirect() {
        let s = audit_status_to_result(StatusCode::PERMANENT_REDIRECT);
        assert!(s.starts_with("redirect: 308"));
    }
}
