//! server: crate root.

// crates/server/src/lib.rs
//! vLLM server crate - HTTP API server for LLM inference

use std::sync::Arc;

use crate::api::EngineHandle;
use crate::auth::AuthMiddleware;
use crate::openai::batch::manager::BatchManager;
use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_model::config::Architecture;
use vllm_model::tokenizer::Tokenizer;

/// api: api module.
pub mod api;
/// auth: auth module.
pub mod auth;
/// backpressure: backpressure module.
pub mod backpressure;
/// cli: cli module.
pub mod cli;
/// config: config module.
pub mod config;
/// debug: debug module.
pub mod debug;
/// draft_loader: draft loader module.
pub mod draft_loader;
/// health: health module.
pub mod health;
/// logging: logging module.
pub mod logging;
/// openai: openai module.
pub mod openai;
/// security: security module.
pub mod security;

/// Hidden test helpers for unit/integration tests.
#[doc(hidden)]
pub mod test_fixtures;

pub use health::{HealthChecker, HealthStatus};

/// Shared state for all API handlers
#[derive(Clone)]
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
