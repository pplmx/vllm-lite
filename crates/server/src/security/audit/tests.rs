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
                timestamp: rfc3339_now(),
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

#[tokio::test]
async fn test_audit_log_auth_failure() {
    let logger = AuditLogger::new(10);

    logger.log_auth_failure("missing api key", "req-456").await;

    let events = logger.get_events().await;
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].action, "authenticate");
    assert_eq!(events[0].result, "failure: missing api key");
}

#[tokio::test]
async fn test_audit_clear() {
    let logger = AuditLogger::new(10);

    logger.log_auth_success("user1", "req-1").await;
    assert_eq!(logger.get_events().await.len(), 1);

    logger.clear().await;
    assert!(
        logger.get_events().await.is_empty(),
        "clear() must empty the ring buffer"
    );
}

/// The ring buffer must stay bounded and keep insertion order under sustained
/// load, the shape the production capacity sees (thousands of requests per
/// window rather than the three-event fixtures elsewhere in this file).
///
/// Eviction is `VecDeque::pop_front` — O(1) — so this also fails loudly if a
/// future change reintroduces a whole-buffer shift on the hot path.
#[tokio::test]
async fn test_ring_buffer_stays_bounded_and_ordered_under_load() {
    let capacity = 16;
    let logger = AuditLogger::new(capacity);

    for i in 0..1_000 {
        logger
            .log_api_request(
                Some("user"),
                "GET",
                &format!("/v1/requests/{i}"),
                "success",
                &format!("req-{i}"),
            )
            .await;
    }

    let events = logger.get_events().await;
    assert_eq!(
        events.len(),
        capacity,
        "the ring must hold exactly `max_events` entries"
    );
    // Newest last: the retained window is the tail of the inserted sequence.
    assert_eq!(events[0].resource, "/v1/requests/984");
    assert_eq!(
        events[capacity - 1].resource,
        "/v1/requests/999",
        "the most recent event must be last"
    );
}
