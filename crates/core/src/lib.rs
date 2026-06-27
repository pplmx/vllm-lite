//! vLLM-lite Core Engine
//!
//! A lightweight LLM inference engine implementing vLLM innovations:
//! - Continuous batching
//! - PagedAttention KV cache management
//! - Prefix caching
//! - Speculative decoding (in progress)
//!
//! # Core Components
//!
//! - [`Engine`] - Main inference engine
//! - [`SchedulerEngine`] - Request scheduling and batch building
//! - [`EnhancedMetricsCollector`] - Performance metrics
//! - [`kv_cache`] - Block allocation and prefix cache helpers

/// beam: beam module.
pub mod beam;
/// circuit_breaker: circuit breaker module.
pub mod circuit_breaker;
/// engine: engine module.
pub mod engine;
/// error: error module.
pub mod error;
/// ha: ha module.
pub mod ha;
/// kv_cache: kv cache module.
pub mod kv_cache;
/// metrics: metrics module.
pub mod metrics;
/// routing: routing module.
pub mod routing;
/// sampling: sampling module.
pub mod sampling;
/// scheduler: scheduler module.
pub mod scheduler;
/// speculative: speculative module.
pub mod speculative;
/// sync: sync module.
pub mod sync;
/// types: types module.
pub mod types;

pub use beam::BeamSequence;
pub use engine::Engine;
pub use ha::{FailoverManager, LeaderElection, LeadershipState};
pub use metrics::{EnhancedMetricsCollector, MetricsSnapshot};
pub use routing::HashRouter;
pub use scheduler::SchedulerEngine;
pub use types::Priority;
