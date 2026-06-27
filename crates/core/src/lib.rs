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

pub mod beam;
pub(crate) mod circuit_breaker;
pub mod engine;
pub mod error;
pub(crate) mod ha;
pub mod metrics;
pub(crate) mod routing;
pub mod sampling;
pub mod scheduler;
pub mod speculative;
pub(crate) mod sync;
pub mod types;

pub use beam::BeamSequence;
pub use engine::Engine;
pub use metrics::{EnhancedMetricsCollector, MetricsSnapshot};
pub use scheduler::SchedulerEngine;
pub use types::Priority;
