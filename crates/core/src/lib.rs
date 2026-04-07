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
//! - [`Scheduler`] - Request scheduling and batch building
//! - [`MetricsCollector`] - Performance metrics
//! - [`kv_cache`] - Block allocation and prefix cache

pub mod beam;
pub mod engine;
pub mod error;
pub mod kv_cache;
pub mod metrics;
pub mod sampling;
pub mod scheduler;
pub mod types;

pub use beam::BeamSequence;
pub use engine::Engine;
pub use metrics::{MetricsCollector, MetricsSnapshot};
pub use scheduler::SchedulerEngine;
pub use types::Priority;
