//! Unit + integration tests for the `rbac` middleware.
//!
//! Two surfaces are exercised:
//!
//! 1. **Pure unit (4)**: `RbacMiddleware::check_permission` per-role
//!    capability matrix (Admin → read/write/admin, Operator → read,
//!    User → read, Anonymous → nothing); `Role::from_str` parses
//!    case-insensitively and falls back to `Anonymous` on unknown
//!    input; `required_action_for_path` maps URL prefixes to the
//!    action string required (`/health` → no action, `/v1/models`
//!    → read, `/v1/chat/completions` → execute, `/admin/*` →
//!    `manage_users`); `resolve_role` ignores extension absence and
//!    returns the default role (SEC-01).
//! 2. **Axum middleware integration (7)**: a tiny `axum::Router`
//!    with the `rbac_middleware` layer installed; each test sends
//!    one request via `tower::ServiceExt::oneshot` and asserts the
//!    status code (and the body for the 403 case, which must contain
//!    both `forbidden` and the required-action name). Tests that
//!    need an elevated role insert [`AuthenticatedRole`] via request
//!    extensions — they never forge `X-User-Role`, because the v31.0
//!    P4 SEC-01 fix removes that codepath entirely.
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

/// Build a request with the given path and (optionally) a pre-installed
/// [`AuthenticatedRole`]. Tests that want to exercise the elevated-role
/// path pass `Some(role)`; tests that exercise the anonymous / forgery
/// path pass `None` and may set arbitrary `X-User-Role` headers to
/// prove the middleware ignores them.
fn build_request(
    method: &str,
    path: &str,
    role: Option<Role>,
    forged_header: Option<&str>,
) -> HttpRequest<Body> {
    let mut builder = HttpRequest::builder().method(method).uri(path);
    if let Some(forged) = forged_header {
        builder = builder.header("X-User-Role", forged);
    }
    let mut req = builder.body(Body::empty()).unwrap();
    if let Some(r) = role {
        req.extensions_mut().insert(AuthenticatedRole(r));
    }
    req
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

#[test]
fn test_resolve_role_defaults_to_anonymous_without_extension() {
    // SEC-01: a request that bypassed auth has no `AuthenticatedRole`
    // extension installed; the middleware must default to Anonymous.
    let rbac = RbacMiddleware::new(Role::Anonymous);
    assert_eq!(rbac.resolve_role(None), Role::Anonymous);
}

#[test]
fn test_resolve_role_honours_extension() {
    // Only server-side middleware (e.g. JWT claim path) can install
    // an `AuthenticatedRole`; whatever value it sets wins.
    let rbac = RbacMiddleware::new(Role::Anonymous);
    let ext = AuthenticatedRole(Role::Admin);
    assert_eq!(rbac.resolve_role(Some(&ext)), Role::Admin);
}

#[tokio::test]
async fn test_rbac_allows_admin_on_protected() {
    // Admin role granted via the legitimate extension path (not a
    // forged header).
    let app = app_with_rbac();
    let req = build_request("GET", "/v1/models", Some(Role::Admin), None);
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), AxStatusCode::OK);
}

#[tokio::test]
async fn test_rbac_allows_user_on_read() {
    let app = app_with_rbac();
    let req = build_request("GET", "/v1/models", Some(Role::User), None);
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), AxStatusCode::OK);
}

#[tokio::test]
async fn test_rbac_denies_anonymous_on_admin() {
    let app = app_with_rbac();
    let req = build_request("GET", "/admin/users", None, None);
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
    let req = build_request("GET", "/admin/users", Some(Role::User), None);
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), AxStatusCode::FORBIDDEN);
}

/// **SEC-01 regression test**: a forged `X-User-Role: admin` header
/// MUST NOT grant admin access. Pre-fix, the middleware trusted the
/// header and let any client become Admin. Post-fix, the header is
/// ignored entirely.
#[tokio::test]
async fn test_rbac_ignores_forged_admin_header() {
    let app = app_with_rbac();
    // Forged header set; no extension installed. Must be denied.
    let req = build_request("GET", "/admin/users", None, Some("admin"));
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        AxStatusCode::FORBIDDEN,
        "forged X-User-Role header must not grant admin (SEC-01)"
    );
}

/// **SEC-01 regression test**: a forged `X-User-Role` header on a
/// *public* path is still allowed (the path doesn't require any role),
/// but it must NOT escalate the caller to admin when they reach a
/// protected path two requests later. This is a defence-in-depth check
/// that the middleware never persists or remembers header values.
#[tokio::test]
async fn test_rbac_header_does_not_persist_across_requests() {
    let app = app_with_rbac();

    // 1. Forged header on /health (public) — passes, but the role is
    //    Anonymous for any future request.
    let req1 = build_request("GET", "/health", None, Some("admin"));
    let resp1 = app.clone().oneshot(req1).await.unwrap();
    assert_eq!(resp1.status(), AxStatusCode::OK);

    // 2. Second request to /admin/users with the SAME forged header —
    //    no extension installed. Must still be denied.
    let req2 = build_request("GET", "/admin/users", None, Some("admin"));
    let resp2 = app.oneshot(req2).await.unwrap();
    assert_eq!(
        resp2.status(),
        AxStatusCode::FORBIDDEN,
        "forged header must not carry across requests"
    );
}

#[tokio::test]
async fn test_rbac_allows_anonymous_on_health() {
    let app = app_with_rbac();
    let req = build_request("GET", "/health", None, None);
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), AxStatusCode::OK);
}
