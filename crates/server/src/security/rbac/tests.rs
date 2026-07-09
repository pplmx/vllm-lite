//! Unit + integration tests for the `rbac` middleware.
//!
//! Two surfaces are exercised:
//!
//! 1. **Pure unit (3)**: `RbacMiddleware::check_permission` per-role
//!    capability matrix (Admin → read/write/admin, Operator → read,
//!    User → read, Anonymous → nothing); `Role::from_str` parses
//!    case-insensitively and falls back to `Anonymous` on unknown
//!    input; `required_action_for_path` maps URL prefixes to the
//!    action string required (`/health` → no action, `/v1/models`
//!    → read, `/v1/chat/completions` → execute, `/admin/*` →
//!    manage_users).
//! 2. **Axum middleware integration (5)**: a tiny `axum::Router`
//!    with the `rbac_middleware` layer installed; each test sends
//!    one request via `tower::ServiceExt::oneshot` and asserts the
//!    status code (and the body for the 403 case, which must
//!    contain both `forbidden` and the required-action name).
use super::*;
use axum::body::{Body, to_bytes};
use axum::http::{Request as HttpRequest, StatusCode as AxStatusCode};
use tower::ServiceExt;

async fn ok_handler() -> &'static str {
    "ok"
}

fn app_with_rbac() -> axum::Router {
    axum::Router::new()
        .route("/v1/models", axum::routing::get(ok_handler))
        .route("/admin/users", axum::routing::get(ok_handler))
        .route("/health", axum::routing::get(ok_handler))
        .layer(axum::middleware::from_fn(rbac_middleware))
}

#[test]
fn test_role_permissions() {
    let rbac = RbacMiddleware::new(Role::Anonymous);

    assert!(rbac.check_permission(Role::Admin, "read"));
    assert!(rbac.check_permission(Role::Admin, "write"));
    assert!(rbac.check_permission(Role::Admin, "admin"));

    assert!(rbac.check_permission(Role::Operator, "read"));
    assert!(!rbac.check_permission(Role::Operator, "write"));

    assert!(rbac.check_permission(Role::User, "read"));
    assert!(!rbac.check_permission(Role::User, "write"));

    assert!(!rbac.check_permission(Role::Anonymous, "read"));
}

#[test]
fn test_role_from_str() {
    assert_eq!(Role::from_str("admin"), Role::Admin);
    assert_eq!(Role::from_str("ADMIN"), Role::Admin);
    assert_eq!(Role::from_str("operator"), Role::Operator);
    assert_eq!(Role::from_str("user"), Role::User);
    assert_eq!(Role::from_str("unknown"), Role::Anonymous);
}

#[test]
fn test_required_action_for_path() {
    assert_eq!(RbacMiddleware::required_action_for_path("/health"), None);
    assert_eq!(
        RbacMiddleware::required_action_for_path("/v1/models"),
        Some("read")
    );
    assert_eq!(
        RbacMiddleware::required_action_for_path("/v1/chat/completions"),
        Some("execute")
    );
    assert_eq!(
        RbacMiddleware::required_action_for_path("/admin/users"),
        Some("manage_users")
    );
    assert_eq!(
        RbacMiddleware::required_action_for_path("/admin"),
        Some("manage_users")
    );
}

#[tokio::test]
async fn test_rbac_allows_admin_on_protected() {
    let app = app_with_rbac();
    let req = HttpRequest::builder()
        .method("GET")
        .uri("/v1/models")
        .header("X-User-Role", "admin")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), AxStatusCode::OK);
}

#[tokio::test]
async fn test_rbac_allows_user_on_read() {
    let app = app_with_rbac();
    let req = HttpRequest::builder()
        .method("GET")
        .uri("/v1/models")
        .header("X-User-Role", "user")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), AxStatusCode::OK);
}

#[tokio::test]
async fn test_rbac_denies_anonymous_on_admin() {
    let app = app_with_rbac();
    let req = HttpRequest::builder()
        .method("GET")
        .uri("/admin/users")
        // no X-User-Role → defaults to Anonymous
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), AxStatusCode::FORBIDDEN);
    let body_bytes = to_bytes(resp.into_body(), 4096).await.unwrap();
    let body_str = std::str::from_utf8(&body_bytes).unwrap();
    assert!(body_str.contains("forbidden"));
    assert!(body_str.contains("manage_users"));
}

#[tokio::test]
async fn test_rbac_denies_user_on_admin() {
    let app = app_with_rbac();
    let req = HttpRequest::builder()
        .method("GET")
        .uri("/admin/users")
        .header("X-User-Role", "user")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), AxStatusCode::FORBIDDEN);
}

#[tokio::test]
async fn test_rbac_allows_anonymous_on_health() {
    let app = app_with_rbac();
    let req = HttpRequest::builder()
        .method("GET")
        .uri("/health")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), AxStatusCode::OK);
}
