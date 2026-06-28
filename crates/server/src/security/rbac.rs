use axum::{
    Json,
    extract::Request,
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use serde_json::json;
use std::sync::Arc;

/// Role: role enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Admin,
    Operator,
    User,
    Anonymous,
}

impl Role {
    #[allow(clippy::should_implement_trait)]
    #[must_use]
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "admin" => Self::Admin,
            "operator" => Self::Operator,
            "user" => Self::User,
            _ => Self::Anonymous,
        }
    }

    #[must_use]
    pub const fn can_read_models(&self) -> bool {
        !matches!(self, Self::Anonymous)
    }

    #[must_use]
    pub const fn can_write_models(&self) -> bool {
        matches!(self, Self::Admin)
    }

    #[must_use]
    pub const fn can_manage_users(&self) -> bool {
        matches!(self, Self::Admin)
    }

    #[must_use]
    pub const fn can_view_metrics(&self) -> bool {
        matches!(self, Self::Admin | Self::Operator)
    }

    #[must_use]
    pub const fn can_access_admin(&self) -> bool {
        matches!(self, Self::Admin)
    }
}

/// `RbacMiddleware`: rbac middleware.
pub struct RbacMiddleware {
    default_role: Role,
    role_permissions: Arc<Vec<(Role, Vec<&'static str>)>>,
}

impl RbacMiddleware {
    #[must_use]
    pub fn new(default_role: Role) -> Self {
        let role_permissions = vec![
            (Role::Admin, vec!["*"]),
            (Role::Operator, vec!["read", "execute"]),
            (Role::User, vec!["read", "execute"]),
            (Role::Anonymous, vec![]),
        ];

        Self {
            default_role,
            role_permissions: Arc::new(role_permissions),
        }
    }

    #[must_use]
    pub fn check_permission(&self, role: Role, action: &str) -> bool {
        for (r, actions) in self.role_permissions.iter() {
            if *r == role {
                return actions.iter().any(|a| *a == "*" || *a == action);
            }
        }
        false
    }

    pub fn extract_role_from_headers(&self, headers: &HeaderMap) -> Role {
        headers
            .get("X-User-Role")
            .and_then(|v| v.to_str().ok())
            .map_or(self.default_role, Role::from_str)
    }

    ///
    /// Static path → action mapping. Used by `rbac_middleware` to
    /// decide whether the requesting role has the required permission.
    /// Returns `None` for paths that have no RBAC requirement (i.e.
    /// public endpoints like `/health`).
    #[must_use]
    pub fn required_action_for_path(path: &str) -> Option<&'static str> {
        // Strip trailing slash for matching.
        let p = path.trim_end_matches('/');
        match p {
            "/health" | "/ready" => None,
            "/v1/models" => Some("read"),
            "/v1/chat/completions" => Some("execute"),
            "/v1/completions" => Some("execute"),
            "/v1/embeddings" => Some("execute"),
            "/metrics" => Some("view_metrics"),
            p if p.starts_with("/admin") => Some("manage_users"),
            // Default: unknown paths require `read` (least-privilege
            // default; admin wildcard still grants access).
            _ => Some("read"),
        }
    }
}

///
/// Enforces role-based access control. Extracts the role from either
/// the JWT-claims-style `X-User-Role` header (set upstream by the
/// auth middleware) or the configured default role, then denies the
/// request with HTTP 403 + a structured JSON error if the role lacks
/// the required permission for the requested path.
pub async fn rbac_middleware(request: Request, next: Next) -> Response {
    let rbac = RbacMiddleware::new(Role::Anonymous);
    let path = request.uri().path().to_string();
    let required = match RbacMiddleware::required_action_for_path(&path) {
        Some(a) => a,
        None => return next.run(request).await,
    };

    let role = rbac.extract_role_from_headers(request.headers());
    if rbac.check_permission(role, required) {
        next.run(request).await
    } else {
        let body = Json(json!({
            "error": "forbidden",
            "message": format!("Role {:?} lacks required permission '{}' for {}", role, required, path),
            "required_action": required,
        }));
        (StatusCode::FORBIDDEN, body).into_response()
    }
}

#[cfg(test)]
mod tests {
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
}
