//! vLLM-lite Core Engine
//!
//! A lightweight LLM inference engine implementing vLLM innovations:
//! - Continuous batching
//! - `PagedAttention` KV cache management
//! - Prefix caching
//! - Speculative decoding (production-ready since v18.0)
//!
//! # Core Components
//!
//! - [`Engine`] - Main inference engine
//! - [`SchedulerEngine`] - Request scheduling and batch building
//! - [`EnhancedMetricsCollector`] - Performance metrics

/// Beam-search sequence state and expansion helpers.
pub mod beam;
pub(crate) mod circuit_breaker;
/// Main inference engine and batch execution loop.
pub mod engine;
/// Typed `EngineError` and cross-crate `Result` alias.
pub mod error;
pub(crate) mod ha;
/// Lock-free and enhanced metrics collectors.
pub mod metrics;
pub(crate) mod routing;
/// Token sampling strategies (greedy, top-k, top-p, temperature).
pub mod sampling;
/// Continuous-batching scheduler, KV allocator, and prefix cache.
pub mod scheduler;
/// Tracing-subscriber bootstrap with optional OpenTelemetry bridge (gated
/// behind the `opentelemetry` feature).
#[cfg(feature = "opentelemetry")]
pub mod tracing_init;
/// Speculative decoding registry, draft models, and verification.
pub mod speculative;
pub(crate) mod sync;
/// Scheduler request types, priorities, and configuration.
pub mod types;

pub use beam::BeamSequence;
pub use engine::{Engine, EngineBuilder};
pub use error::{EngineError, Result};
pub use metrics::{DraftResolutionKind, EnhancedMetricsCollector, MetricsSnapshot};
pub use scheduler::SchedulerEngine;
pub use speculative::{AdaptiveSpeculativeDecoder, DraftModelRegistry, DraftResolver, DraftSpec};
pub use types::{Priority, Request, SamplingParams, SchedulerConfig};
