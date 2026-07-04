//! Public API surface of the security subsystem: re-exports the JWT, RBAC, audit, TLS, and size-limit modules under one namespace.
//!
//! Mounted into the server's axum router via `security::install_middleware`.
pub mod audit;
pub mod correlation;
pub mod jwt;
pub mod rbac;
pub mod size_limit;
pub mod tls;

pub use audit::AuditLogger;
pub use correlation::CorrelationIdMiddleware;
pub use jwt::JwtValidator;
pub use rbac::{RbacMiddleware, Role};
pub use size_limit::{DEFAULT_BODY_LIMIT_BYTES, with_body_size_limit, with_default_body_limit};
pub use tls::{TlsConfig, TlsListener};
