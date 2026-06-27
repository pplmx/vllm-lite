//! rbac: rbac.

use axum::{extract::Request, http::HeaderMap, middleware::Next, response::Response};
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
/// from_str: from str.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "admin" => Role::Admin,
            "operator" => Role::Operator,
            "user" => Role::User,
            _ => Role::Anonymous,
        }
    }

/// can_read_models: can read models.
    pub fn can_read_models(&self) -> bool {
        !matches!(self, Role::Anonymous)
    }

/// can_write_models: can write models.
    pub fn can_write_models(&self) -> bool {
        matches!(self, Role::Admin)
    }

/// can_manage_users: can manage users.
    pub fn can_manage_users(&self) -> bool {
        matches!(self, Role::Admin)
    }

/// can_view_metrics: can view metrics.
    pub fn can_view_metrics(&self) -> bool {
        matches!(self, Role::Admin | Role::Operator)
    }

/// can_access_admin: can access admin.
    pub fn can_access_admin(&self) -> bool {
        matches!(self, Role::Admin)
    }
}

/// RbacMiddleware: rbac middleware.
pub struct RbacMiddleware {
    default_role: Role,
    role_permissions: Arc<Vec<(Role, Vec<&'static str>)>>,
}

impl RbacMiddleware {
/// new: new.
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

/// check_permission: check permission.
    pub fn check_permission(&self, role: Role, action: &str) -> bool {
        for (r, actions) in self.role_permissions.iter() {
            if *r == role {
                return actions.iter().any(|a| *a == "*" || *a == action);
            }
        }
        false
    }

/// extract_role_from_headers: extract role from headers.
    pub fn extract_role_from_headers(&self, headers: &HeaderMap) -> Role {
        headers
            .get("X-User-Role")
            .and_then(|v| v.to_str().ok())
            .map(Role::from_str)
            .unwrap_or(self.default_role)
    }
}

/// rbac_middleware: rbac middleware.
pub async fn rbac_middleware(request: Request, next: Next) -> Response {
    next.run(request).await
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
