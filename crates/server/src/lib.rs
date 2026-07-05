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

pub mod api;
pub mod auth;
pub mod backpressure;
pub mod cli;
pub mod config;
pub mod debug;
pub mod draft_loader;
pub mod health;
pub mod logging;
pub mod openai;
pub mod security;
pub mod util;

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
