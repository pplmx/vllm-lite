// crates/server/src/lib.rs
//! vLLM server crate - HTTP API server for LLM inference

use std::sync::Arc;

use crate::api::EngineHandle;
use crate::auth::AuthMiddleware;
use crate::openai::batch::manager::BatchManager;
use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_model::tokenizer::Tokenizer;

pub mod api;
pub mod auth;
pub mod cli;
pub mod config;
pub mod health;
pub mod logging;
pub mod openai;

pub use health::{HealthChecker, HealthStatus};

/// Shared state for all API handlers
#[derive(Clone)]
pub struct ApiState {
    /// Channel to send messages to the inference engine
    pub engine_tx: EngineHandle,
    /// Tokenizer for encoding/decoding text
    pub tokenizer: Arc<Tokenizer>,
    /// Batch manager for handling batch API requests
    pub batch_manager: Arc<BatchManager>,
    /// Authentication middleware (None if disabled)
    pub auth: Option<Arc<AuthMiddleware>>,
    /// Health checker for liveness/readiness probes
    pub health: Arc<std::sync::RwLock<HealthChecker>>,
    /// Enhanced metrics collector
    pub metrics: Arc<EnhancedMetricsCollector>,
}
