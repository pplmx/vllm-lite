//! vLLM-lite Core Engine
//!
//! A lightweight LLM inference engine implementing vLLM innovations:
//! - Continuous batching
//! - PagedAttention KV cache management
//! - Prefix caching
//! - Speculative decoding (production-ready since v18.0)
//!
//! # Core Components
//!
//! - [`Engine`] - Main inference engine
//! - [`SchedulerEngine`] - Request scheduling and batch building
//! - [`EnhancedMetricsCollector`] - Performance metrics
//! - [`kv_cache`] - Block allocation and prefix cache helpers

pub mod beam;
pub mod circuit_breaker;
pub mod engine;
pub mod error;
pub mod ha;
pub mod kv_cache;
pub mod metrics;
pub mod routing;
pub mod sampling;
pub mod scheduler;
pub mod speculative;
pub mod sync;
pub mod types;

pub use beam::BeamSequence;
pub use engine::Engine;
pub use ha::{FailoverManager, LeaderElection, LeadershipState};
pub use metrics::{EnhancedMetricsCollector, MetricsSnapshot};
pub use routing::HashRouter;
pub use scheduler::SchedulerEngine;
pub use types::Priority;
