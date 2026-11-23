//! Structured audit logging: in-memory ring buffer of security-relevant events plus a `/debug/audit` JSON export endpoint.
//!
//! Records login attempts, auth failures, rate-limit triggers, and admin
//! actions. Logged at INFO/WARN level via `tracing` and also retained in
//! memory for the rolling export.
#![allow(clippy::module_name_repetitions)]
use serde::Serialize;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// One event in the Audit stream. Variants cover the full state-machine of the subsystem.
#[derive(Debug, Clone, Serialize)]
pub struct AuditEvent {
    /// RFC-3339 timestamp when the event was recorded.
    pub timestamp: String,
    /// Authenticated user id, if any (None for failed-auth events).
    pub user_id: Option<String>,
    /// Action name (`"authenticate"`, `"rate_limit"`, `"admin_delete"`, ...).
    pub action: String,
    /// Resource the action targeted (model id, route, etc).
    pub resource: String,
    /// Outcome string (`"success"`, `"failure: <reason>"`, ...).
    pub result: String,
    /// Correlation id from the inbound HTTP request.
    pub request_id: String,
    /// Source IP if available.
    pub ip_address: Option<String>,
    /// User-Agent header if available.
    pub user_agent: Option<String>,
}

/// RFC-3339 UTC timestamp for an [`AuditEvent`].
///
/// Replaces `chrono::Utc::now().to_rfc3339()`: `chrono` is unmaintained
/// (RUSTSEC-2026-0543), and `time` is already in the dependency graph.
#[must_use]
fn rfc3339_now() -> String {
    time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        // invariant: `Rfc3339` is a well-known constant format and `now_utc()`
        // is always a representable date, so formatting cannot fail.
        .unwrap_or_else(|_| String::from("1970-01-01T00:00:00Z"))
}

/// Bounded in-memory ring buffer of [`AuditEvent`]s with structured tracing.
///
/// Each `log_*` call also emits a structured `tracing` event at INFO (success)
/// or WARN (failure) level. The oldest event is evicted once the buffer exceeds
/// `max_events`.
#[derive(Debug)]
pub struct AuditLogger {
    events: Arc<RwLock<VecDeque<AuditEvent>>>,
    max_events: usize,
}

impl AuditLogger {
    /// Build a logger that retains at most `max_events` entries.
    /// Larger values trade memory for a deeper audit-trail window
    /// available via the `/debug/audit` JSON export endpoint.
    #[must_use]
    pub fn new(max_events: usize) -> Self {
        Self {
            events: Arc::new(RwLock::new(VecDeque::with_capacity(max_events.min(1024)))),
            max_events,
        }
    }

    /// Append `event` to the ring buffer (evicting the oldest entry
    /// if full) and emit a structured `tracing` event for log
    /// aggregation.
    pub async fn log(&self, event: AuditEvent) {
        let mut events = self.events.write().await;
        events.push_back(event.clone());

        // `VecDeque::pop_front` is O(1); `Vec::remove(0)` shifted the whole
        // buffer, which at the production capacity (10_000) meant moving
        // 10_000 elements on every logged request once the ring filled.
        if events.len() > self.max_events {
            events.pop_front();
        }

        info!(
            request_id = %event.request_id,
            user_id = ?event.user_id,
            action = %event.action,
            resource = %event.resource,
            result = %event.result,
            "Audit event"
        );
        drop(events);
    }

    /// Convenience helper: record a successful authentication event
    /// (`action = "authenticate"`, `result = "success"`).
    pub async fn log_auth_success(&self, user_id: &str, request_id: &str) {
        self.log(AuditEvent {
            timestamp: rfc3339_now(),
            user_id: Some(user_id.to_string()),
            action: "authenticate".to_string(),
            resource: "api".to_string(),
            result: "success".to_string(),
            request_id: request_id.to_string(),
            ip_address: None,
            user_agent: None,
        })
        .await;
    }

    /// Convenience helper: record a failed authentication attempt
    /// (`action = "authenticate"`, `result = "failure: <reason>"`).
    pub async fn log_auth_failure(&self, reason: &str, request_id: &str) {
        self.log(AuditEvent {
            timestamp: rfc3339_now(),
            user_id: None,
            action: "authenticate".to_string(),
            resource: "api".to_string(),
            result: format!("failure: {reason}"),
            request_id: request_id.to_string(),
            ip_address: None,
            user_agent: None,
        })
        .await;

        warn!(
            request_id = %request_id,
            reason = %reason,
            "Authentication failed"
        );
    }

    /// Append an audit event for an API request (user, action, resource, result).
    pub async fn log_api_request(
        &self,
        user_id: Option<&str>,
        action: &str,
        resource: &str,
        result: &str,
        request_id: &str,
    ) {
        self.log_api_request_with_client(user_id, action, resource, result, request_id, None)
            .await;
    }

    /// Record an API request together with the caller's `User-Agent`.
    ///
    /// The audit row exposes a `user_agent` field (surfaced by
    /// `/debug/audit`), but every construction site passed `None`, so the
    /// field was permanently `null` and an operator investigating a run of
    /// requests could not tell which client produced them. `ip_address`
    /// stays `None`: this server does not see the peer address (no
    /// `ConnectInfo`) and does not trust `X-Forwarded-For`, so inventing a
    /// value there would be worse than leaving it out.
    pub async fn log_api_request_with_client(
        &self,
        user_id: Option<&str>,
        action: &str,
        resource: &str,
        result: &str,
        request_id: &str,
        user_agent: Option<&str>,
    ) {
        self.log(AuditEvent {
            timestamp: rfc3339_now(),
            user_id: user_id.map(std::string::ToString::to_string),
            action: action.to_string(),
            resource: resource.to_string(),
            result: result.to_string(),
            request_id: request_id.to_string(),
            ip_address: None,
            user_agent: user_agent.map(std::string::ToString::to_string),
        })
        .await;
    }

    /// Return a clone of all recorded audit events (oldest-first).
    pub async fn get_events(&self) -> Vec<AuditEvent> {
        self.events.read().await.iter().cloned().collect()
    }

    /// Remove all recorded audit events.
    pub async fn clear(&self) {
        self.events.write().await.clear();
    }
}

impl Default for AuditLogger {
    fn default() -> Self {
        Self::new(10000)
    }
}

// Unit tests are extracted to `tests.rs` (sibling) to keep this
// audit-logger module under the 800-line soft cap. They cover the
// single-event contract and the bounded-retention ring buffer.
#[cfg(test)]
mod tests;
