//! Structured audit logging: in-memory ring buffer of security-relevant events plus a `/debug/audit` JSON export endpoint.
//!
//! Records login attempts, auth failures, rate-limit triggers, and admin
//! actions. Logged at INFO/WARN level via `tracing` and also retained in
//! memory for the rolling export.
#![allow(clippy::module_name_repetitions)]
use serde::Serialize;
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

#[derive(Debug)]
/// `AuditLogger`. See the type definition for fields and behavior.
pub struct AuditLogger {
    events: Arc<RwLock<Vec<AuditEvent>>>,
    max_events: usize,
}

impl AuditLogger {
    #[must_use]
    pub fn new(max_events: usize) -> Self {
        Self {
            events: Arc::new(RwLock::new(Vec::new())),
            max_events,
        }
    }

    pub async fn log(&self, event: AuditEvent) {
        let mut events = self.events.write().await;
        events.push(event.clone());

        if events.len() > self.max_events {
            events.remove(0);
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

    pub async fn log_auth_success(&self, user_id: &str, request_id: &str) {
        self.log(AuditEvent {
            timestamp: chrono::Utc::now().to_rfc3339(),
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

    pub async fn log_auth_failure(&self, reason: &str, request_id: &str) {
        self.log(AuditEvent {
            timestamp: chrono::Utc::now().to_rfc3339(),
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

    pub async fn log_api_request(
        &self,
        user_id: Option<&str>,
        action: &str,
        resource: &str,
        result: &str,
        request_id: &str,
    ) {
        self.log(AuditEvent {
            timestamp: chrono::Utc::now().to_rfc3339(),
            user_id: user_id.map(std::string::ToString::to_string),
            action: action.to_string(),
            resource: resource.to_string(),
            result: result.to_string(),
            request_id: request_id.to_string(),
            ip_address: None,
            user_agent: None,
        })
        .await;
    }

    pub async fn get_events(&self) -> Vec<AuditEvent> {
        self.events.read().await.clone()
    }

    pub async fn clear(&self) {
        self.events.write().await.clear();
    }
}

impl Default for AuditLogger {
    fn default() -> Self {
        Self::new(10000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_audit_log() {
        let logger = AuditLogger::new(10);

        logger.log_auth_success("user1", "req-123").await;

        let events = logger.get_events().await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].action, "authenticate");
        assert_eq!(events[0].result, "success");
    }

    #[tokio::test]
    async fn test_audit_log_overflow() {
        let logger = AuditLogger::new(3);

        for i in 0..5 {
            logger
                .log(AuditEvent {
                    timestamp: chrono::Utc::now().to_rfc3339(),
                    user_id: None,
                    action: format!("action-{i}"),
                    resource: "test".to_string(),
                    result: "success".to_string(),
                    request_id: format!("req-{i}"),
                    ip_address: None,
                    user_agent: None,
                })
                .await;
        }

        let events = logger.get_events().await;
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].action, "action-2");
    }
}
