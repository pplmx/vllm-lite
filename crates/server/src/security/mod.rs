//! Public API surface of the security subsystem: re-exports the JWT, RBAC, audit, TLS, CORS, and size-limit modules under one namespace.
//!
//! Mounted into the server's axum router via `security::install_middleware`.
/// Structured audit logging: in-memory ring buffer + `/debug/audit` export.
pub mod audit;
/// Audit-logging request-layer middleware.
pub mod audit_middleware;
/// Request correlation ID assignment and propagation.
pub mod correlation;
/// CORS middleware with explicit allowlist (no wildcard).
pub mod cors;
/// JWT validation and auth middleware.
pub mod jwt;
/// Role-based access control middleware.
pub mod rbac;
/// Request body size-limit enforcement.
pub mod size_limit;
/// TLS configuration and certificate loading.
pub mod tls;

pub use audit::AuditLogger;
pub use audit_middleware::audit_middleware;
pub use correlation::CorrelationIdMiddleware;
pub use cors::{CorsConfig, with_cors};
pub use jwt::JwtValidator;
pub use rbac::{RbacMiddleware, Role};
pub use size_limit::{DEFAULT_BODY_LIMIT_BYTES, with_body_size_limit, with_default_body_limit};
pub use tls::TlsConfig;
