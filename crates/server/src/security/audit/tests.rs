//! Unit tests for the `audit` logger.
//!
//! Covers the ring-buffer contract:
//!
//! 1. **Single event**: `log_auth_success` records an
//!    `AuditEvent` with `action = "authenticate"` and
//!    `result = "success"`.
//! 2. **Bounded retention**: with capacity=3, logging 5 events keeps
//!    only the most recent 3; the oldest (`action-0`, `action-1`)
//!    are evicted, and `events[0].action == "action-2"`.
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
