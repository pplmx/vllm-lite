//! vLLM server crate - HTTP API server for LLM inference

use std::sync::Arc;

use crate::api::EngineHandle;
use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_model::config::Architecture;
use vllm_model::tokenizer::Tokenizer;

pub use crate::auth::AuthMiddleware;
pub use config::AuthConfig;
pub use health::{HealthChecker, HealthStatus};
pub use openai::batch::{BatchEndpoint, BatchManager, BatchResponse};
pub use security::audit::AuditEvent;

/// HTTP route handlers and engine channel bridge.
pub mod api;
/// JWT and API-key authentication middleware.
pub mod auth;
/// Token-bucket admission control for overload protection.
pub mod backpressure;
/// CLI argument parsing and server bootstrap.
pub mod cli;
/// Server configuration (auth, TLS, model paths).
pub mod config;
/// Debug and diagnostic endpoints.
pub mod debug;
/// Draft-model loading for speculative decoding.
pub mod draft_loader;
/// Liveness and readiness health probes.
pub mod health;
/// Structured logging setup (console + JSON file).
pub mod logging;
/// OpenAI-compatible chat, completions, and batch APIs.
pub mod openai;
/// Rate limiting, audit logging, and security middleware.
pub mod security;
/// Shared HTTP and serialization utilities.
pub mod util;

/// HTTP handlers for liveness/readiness/metrics probes.
///
/// Extracted from `main.rs` so integration tests can mount the
/// routes against a real `axum::Router` with a controlled
/// `ApiState` and assert mailbox-saturation behaviour.
pub mod health_handlers;

/// Hidden test helpers for unit/integration tests.
#[doc(hidden)]
pub mod test_fixtures;

/// Shared state for all API handlers
#[derive(Debug, Clone)]
pub struct ApiState {
    /// Channel to send messages to the inference engine
    pub engine_tx: EngineHandle,
    /// Tokenizer for encoding/decoding text
    pub tokenizer: Arc<Tokenizer>,
    /// Loaded model architecture (drives chat prompt formatting)
    pub architecture: Architecture,
    /// Batch manager for handling batch API requests
    pub batch_manager: Arc<BatchManager>,
    /// Authentication middleware (None if disabled)
    pub auth: Option<Arc<AuthMiddleware>>,
    /// Health checker for liveness/readiness probes
    pub health: Arc<std::sync::RwLock<HealthChecker>>,
    /// Enhanced metrics collector
    pub metrics: Arc<EnhancedMetricsCollector>,
}
