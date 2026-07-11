//! Engine construction: the basic `Engine::new_boxed`, `with_config_boxed`,
//! `new`, `with_config` entry points.
//!
//! Other constructor variants live in sibling modules:
//! - [`drafts`] — `with_drafts_*`, `with_budget_*`, `install_default_resolver`
//! - [`builder`] — `EngineBuilder` struct + impl
//!
//! The `Engine` struct itself lives in the parent `engine::mod`.
//!
//! **CUDA Graph construction** (Phase 18 ARCH-06): when the `cuda-graph`
//! feature is enabled, this is the *only* file in `vllm-core` that imports
//! the concrete `vllm_model::kernels::BatchCudaGraphExecutor`. It builds the
//! value and immediately boxes it into
//! `Box<dyn vllm_traits::CudaGraphExecutor + Send>`. Every other engine
//! call site talks to the trait, never the concrete type.

mod builder;
mod drafts;

pub use builder::EngineBuilder;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::engine::Engine;
use crate::engine::SleepPolicy;
use crate::metrics::EnhancedMetricsCollector;
use crate::scheduler::engine::SchedulerEngine;
use crate::speculative::registry::DraftModelRegistry;
use crate::types::SchedulerConfig;
use vllm_traits::ModelBackend;

#[cfg(feature = "cuda-graph")]
use vllm_model::kernels::BatchCudaGraphExecutor;

#[cfg(feature = "cuda-graph")]
use vllm_traits::kernels::{CudaGraphConfig, CudaGraphExecutor};

impl Engine {
    /// Construct an Engine with default scheduler configuration and a fixed
    /// `max_draft_tokens=4` and `num_kv_blocks=1024`.
    ///
    /// Use [`Engine::with_config_boxed`] when you need to tune those values.
    /// The `boxed` suffix means the caller supplies trait-object models
    /// (already boxed); the generic [`Engine::new`] is the more ergonomic
    /// entry point for statically-known model types.
    #[must_use]
    pub fn new_boxed(
        target_model: Box<dyn ModelBackend>,
        draft_model: Option<Box<dyn ModelBackend>>,
    ) -> Self {
        Self::with_config_boxed(
            target_model,
            draft_model,
            SchedulerConfig::default(),
            4,
            1024,
        )
    }

    /// Construct an Engine with full control over scheduler and speculative
    /// parameters.
    ///
    /// - `config` — batch sizing, block size, preemption/eviction policies, etc.
    /// - `max_draft_tokens` — upper bound on draft length per speculative step.
    /// - `num_kv_blocks` — total number of paged KV-cache blocks the allocator
    ///   may hand out across all live sequences.
    #[must_use]
    pub fn with_config_boxed(
        target_model: Box<dyn ModelBackend>,
        draft_model: Option<Box<dyn ModelBackend>>,
        config: SchedulerConfig,
        max_draft_tokens: usize,
        num_kv_blocks: usize,
    ) -> Self {
        let max_seqs = config.max_num_seqs;
        #[cfg(feature = "cuda-graph")]
        let cuda_graph: Option<Box<dyn CudaGraphExecutor + Send>> = if config.cuda_graph.enabled {
            let graph_config = CudaGraphConfig {
                enabled: true,
                batch_sizes: config.cuda_graph.batch_sizes.clone(),
                ..Default::default()
            };
            match BatchCudaGraphExecutor::new(graph_config) {
                Ok(executor) => Some(Box::new(executor)),
                Err(e) => {
                    tracing::warn!("Failed to initialize CUDA Graph: {}", e);
                    None
                }
            }
        } else {
            None
        };
        let enhanced_metrics = Arc::new(EnhancedMetricsCollector::new());
        let draft = draft_model.map(|m| Arc::new(Mutex::new(m)));
        Self {
            scheduler: SchedulerEngine::new(config, num_kv_blocks, enhanced_metrics),
            target_model: Arc::new(Mutex::new(target_model)),
            draft_model: draft,
            max_draft_tokens,
            speculative_mode: false,
            error_count: 0,
            last_error: None,
            response_txs: HashMap::with_capacity(max_seqs),
            sleep_policy: SleepPolicy::default(),
            #[cfg(feature = "cuda-graph")]
            cuda_graph,
            #[cfg(feature = "multi-node")]
            distributed_kv: None,
            adaptive_decoder: None,
            draft_registry: Arc::new(DraftModelRegistry::new()),
            draft_resolver: None,
        }
    }

    /// Creates a new Engine with default configuration.
    ///
    /// Equivalent to `Engine::with_config(target, draft, SchedulerConfig::default(), 4, 1024)`.
    /// See [`Engine::with_config`] for parameter descriptions.
    pub fn new<M: ModelBackend + 'static>(target_model: M, draft_model: Option<M>) -> Self {
        Self::with_config(
            target_model,
            draft_model,
            SchedulerConfig::default(),
            4,
            1024,
        )
    }

    /// Generic constructor that boxes the model arguments and delegates to
    /// [`Engine::with_config_boxed`]. Prefer this over `with_config_boxed` when
    /// the concrete model type is known at the call site (it avoids one
    /// `Box::new` round-trip in user code).
    pub fn with_config<M: ModelBackend + 'static>(
        target_model: M,
        draft_model: Option<M>,
        config: SchedulerConfig,
        max_draft_tokens: usize,
        num_kv_blocks: usize,
    ) -> Self {
        Self::with_config_boxed(
            Box::new(target_model),
            draft_model.map(|m| Box::new(m) as Box<dyn ModelBackend>),
            config,
            max_draft_tokens,
            num_kv_blocks,
        )
    }
}
