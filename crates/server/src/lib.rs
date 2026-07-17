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
    /// Audit logger: records every authenticated request for the
    /// in-memory ring buffer (exportable via `/debug/audit`) and
    /// the structured `tracing` log stream. Bounded at 10 000
    /// events by default. Mounted into the router via
    /// the [`mod@security::audit_middleware`] module.
    pub audit: Arc<crate::security::audit::AuditLogger>,
    /// Health checker for liveness/readiness probes
    pub health: Arc<std::sync::RwLock<HealthChecker>>,
    /// Enhanced metrics collector
    pub metrics: Arc<EnhancedMetricsCollector>,
    /// Maximum context length in tokens, read from the loaded
    /// model's `max_position_embeddings` config field. `None`
    /// when the model did not declare one (stub models, GGUF
    /// without the field, or any config that omits the key) —
    /// in that case context-length validation is skipped.
    ///
    /// Production-readiness recommendation §4: tokenization
    /// after the fact, the chat/completions handlers compare
    /// `prompt_tokens + max_tokens` against this value and
    /// return `400 context_length_exceeded` (OpenAI-compatible
    /// error code) when the request would exceed the limit.
    /// Without this, a 10× oversize prompt can exhaust KV
    /// blocks before any application-level validation runs.
    pub max_model_len: Option<usize>,
    /// Capability flags for the loaded architecture. `None`
    /// when the architecture could not be detected (unknown
    /// checkpoint, missing `model_type` field, etc.) —
    /// capability-gated endpoints (e.g. `/v1/embeddings`)
    /// treat `None` as "refuse with 501" rather than guessing.
    ///
    /// Production-readiness recommendation §10: not every
    /// causal LM checkpoint can produce quality-usable,
    /// normalised, dimension-stable embeddings. The
    /// embeddings handler reads [`Self::arch_capabilities`]
    /// to refuse with `501 Not Implemented` when the loaded
    /// model is a stub (which returns all-zero embeddings,
    /// i.e. meaningless noise) instead of silently shipping
    /// a degenerate response.
    pub arch_capabilities: Option<vllm_model::arch::ArchCapabilities>,
}
