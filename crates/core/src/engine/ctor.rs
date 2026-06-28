// Sub-module for constructors and EngineBuilder on Engine.
// See mod.rs for the Engine struct definition.

use crate::engine::Engine;
use crate::engine::SleepPolicy;
use crate::metrics::EnhancedMetricsCollector;
use crate::scheduler::engine::SchedulerEngine;
use crate::speculative::AdaptiveSpeculativeDecoder;
use crate::speculative::draft_resolver::{DraftLoader, DraftResolver, NoopLoader};
use crate::speculative::memory_budget::MemoryBudget;
use crate::speculative::registry::{DraftModelRegistry, DraftSpec};
use crate::types::SchedulerConfig;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use vllm_traits::ModelBackend;

#[cfg(feature = "cuda-graph")]
use vllm_model::kernels::BatchCudaGraphExecutor;

#[cfg(feature = "cuda-graph")]
use vllm_traits::kernels::CudaGraphConfig;

impl Engine {
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

    pub fn with_config_boxed(
        target_model: Box<dyn ModelBackend>,
        draft_model: Option<Box<dyn ModelBackend>>,
        config: SchedulerConfig,
        max_draft_tokens: usize,
        num_kv_blocks: usize,
    ) -> Self {
        let max_seqs = config.max_num_seqs;
        #[cfg(feature = "cuda-graph")]
        let cuda_graph = if config.cuda_graph.enabled {
            let graph_config = CudaGraphConfig {
                enabled: true,
                batch_sizes: config.cuda_graph.batch_sizes.clone(),
                ..Default::default()
            };
            match BatchCudaGraphExecutor::new(graph_config) {
                Ok(executor) => Some(executor),
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
            adaptive_decoder: None,
            draft_registry: Arc::new(DraftModelRegistry::new()),
            draft_resolver: None,
        }
    }

    /// Creates a new Engine with default configuration.
    pub fn new<M: ModelBackend + 'static>(target_model: M, draft_model: Option<M>) -> Self {
        Self::with_config(
            target_model,
            draft_model,
            SchedulerConfig::default(),
            4,
            1024,
        )
    }

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

    /// Construct an Engine pre-loaded with a set of draft specs.
    ///
    /// All specs are registered in the [`DraftModelRegistry`] as `Unloaded`.
    /// They will be loaded lazily on first use (or eagerly by the caller —
    /// see `DraftModelRegistry::attach_loaded`) — this method does NOT trigger
    /// any I/O. The Engine's `draft_resolver` is wired with a `NoopLoader`:
    /// any attempt to lazy-load falls back to self-spec via FALL-01. The
    /// server should construct a real `DraftLoader` via
    ///
    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    /// [`Self::set_draft_loader`] before serving requests that name drafts.
    #[must_use]
    pub fn with_drafts_boxed(
        target_model: Box<dyn ModelBackend>,
        draft_model: Option<Box<dyn ModelBackend>>,
        draft_specs: Vec<DraftSpec>,
        config: SchedulerConfig,
        max_draft_tokens: usize,
        num_kv_blocks: usize,
    ) -> Self {
        let mut engine = Self::with_config_boxed(
            target_model,
            draft_model,
            config,
            max_draft_tokens,
            num_kv_blocks,
        );
        for spec in draft_specs {
            // Duplicate ids in the spec list are a programmer error — surface them.
            engine
                .draft_registry
                // invariant: caller (`with_drafts_boxed`) supplies a deduplicated spec list;
                // duplicates are a programmer error.
                .register(spec)
                .expect("with_drafts_boxed: duplicate draft id in spec list");
        }
        engine.install_default_resolver();
        engine
    }

    /// Construct an Engine pre-loaded with a set of draft specs (generic form).
    pub fn with_drafts<M: ModelBackend + 'static>(
        target_model: M,
        draft_model: Option<M>,
        draft_specs: Vec<DraftSpec>,
        config: SchedulerConfig,
        max_draft_tokens: usize,
        num_kv_blocks: usize,
    ) -> Self {
        Self::with_drafts_boxed(
            Box::new(target_model),
            draft_model.map(|m| Box::new(m) as Box<dyn ModelBackend>),
            draft_specs,
            config,
            max_draft_tokens,
            num_kv_blocks,
        )
    }

    /// Construct an Engine with a custom memory budget. The same budget is
    /// shared with the draft registry. `draft_resolver` is wired with a
    ///
    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    /// `NoopLoader` (callers may replace via [`Self::set_draft_loader`]).
    pub fn with_budget_boxed(
        target_model: Box<dyn ModelBackend>,
        draft_model: Option<Box<dyn ModelBackend>>,
        draft_specs: Vec<DraftSpec>,
        budget: Arc<MemoryBudget>,
        config: SchedulerConfig,
        max_draft_tokens: usize,
        num_kv_blocks: usize,
    ) -> Self {
        let mut engine = Self::with_config_boxed(
            target_model,
            draft_model,
            config,
            max_draft_tokens,
            num_kv_blocks,
        );
        // Replace the default unlimited-budget registry with one bound to the
        // caller's budget.
        engine.draft_registry = Arc::new(DraftModelRegistry::with_budget(budget));
        for spec in draft_specs {
            engine
                .draft_registry
                // invariant: caller (`with_budget_boxed`) supplies a deduplicated spec list;
                // duplicates are a programmer error.
                .register(spec)
                .expect("with_budget_boxed: duplicate draft id in spec list");
        }
        engine.install_default_resolver();
        engine
    }

    /// Install a `DraftResolver` with a `NoopLoader` on this Engine. Called by
    /// `with_drafts_boxed` / `with_budget_boxed`. Idempotent — replaces any
    /// existing resolver. The resolver shares the same `Arc<DraftModelRegistry>`
    /// as `self.draft_registry`, so register/unload operations on the Engine
    /// are immediately visible to the resolver.
    fn install_default_resolver(&mut self) {
        let registry = self.draft_registry.clone();
        let metrics = self.scheduler.metrics.clone();
        let self_spec: Option<Arc<Mutex<Box<dyn ModelBackend>>>> = self.draft_model.clone();
        let loader: Arc<dyn DraftLoader> = Arc::new(NoopLoader);
        let resolver = Arc::new(DraftResolver::new(registry, self_spec, loader, metrics));
        self.draft_resolver = Some(resolver);
    }
}

/// Builder for [`Engine`] with named methods for all optional fields.
///
/// # Example
///
/// ```ignore
/// use vllm_core::EngineBuilder;
/// use vllm_traits::ModelBackend;
///
/// let target: Box<dyn ModelBackend> = Box::new(/* ... */);
/// let engine = EngineBuilder::new(target)
///     .with_num_kv_blocks(1024)
///     .with_max_draft_tokens(5)
///     .build();
/// ```
pub struct EngineBuilder {
    target_model: Box<dyn ModelBackend>,
    draft_model: Option<Box<dyn ModelBackend>>,
    config: SchedulerConfig,
    max_draft_tokens: usize,
    num_kv_blocks: usize,
    adaptive_decoder: Option<AdaptiveSpeculativeDecoder>,
    draft_resolver: Option<Arc<DraftResolver>>,
    sleep_policy: SleepPolicy,
}

impl std::fmt::Debug for EngineBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineBuilder")
            .field("target_model", &"<dyn ModelBackend>")
            .field(
                "draft_model",
                &self.draft_model.as_ref().map(|_| "<dyn ModelBackend>"),
            )
            .field("config", &self.config)
            .field("max_draft_tokens", &self.max_draft_tokens)
            .field("num_kv_blocks", &self.num_kv_blocks)
            .field("adaptive_decoder", &self.adaptive_decoder)
            .field(
                "draft_resolver",
                &self.draft_resolver.as_ref().map(Arc::strong_count),
            )
            .field("sleep_policy", &self.sleep_policy)
            .finish()
    }
}

impl EngineBuilder {
    /// Create a new builder with a target model. Other fields use defaults:
    /// - `draft_model = None`
    /// - `config = SchedulerConfig::default()`
    /// - `max_draft_tokens = 4`
    /// - `num_kv_blocks = 1024`
    /// - `adaptive_decoder = None`
    /// - `draft_resolver = None`
    /// - `sleep_policy = SleepPolicy::default()`
    #[must_use]
    pub fn new(target_model: Box<dyn ModelBackend>) -> Self {
        Self {
            target_model,
            draft_model: None,
            config: SchedulerConfig::default(),
            max_draft_tokens: 4,
            num_kv_blocks: 1024,
            adaptive_decoder: None,
            draft_resolver: None,
            sleep_policy: SleepPolicy::default(),
        }
    }

    /// Set the optional draft model.
    #[must_use]
    pub fn with_draft_model(mut self, draft_model: Box<dyn ModelBackend>) -> Self {
        self.draft_model = Some(draft_model);
        self
    }

    /// Override the scheduler config.
    #[must_use]
    pub fn with_config(mut self, config: SchedulerConfig) -> Self {
        self.config = config;
        self
    }

    /// Override the max draft tokens per step.
    #[must_use]
    pub const fn with_max_draft_tokens(mut self, n: usize) -> Self {
        self.max_draft_tokens = n;
        self
    }

    /// Override the number of KV-cache blocks.
    #[must_use]
    pub const fn with_num_kv_blocks(mut self, n: usize) -> Self {
        self.num_kv_blocks = n;
        self
    }

    /// Set an optional adaptive speculative decoder.
    #[must_use]
    pub fn with_adaptive_decoder(mut self, decoder: AdaptiveSpeculativeDecoder) -> Self {
        self.adaptive_decoder = Some(decoder);
        self
    }

    /// Set an optional per-request draft resolver (v18+).
    #[must_use]
    pub fn with_draft_resolver(mut self, resolver: Arc<DraftResolver>) -> Self {
        self.draft_resolver = Some(resolver);
        self
    }

    /// Override the sleep policy.
    #[must_use]
    pub const fn with_sleep_policy(mut self, policy: SleepPolicy) -> Self {
        self.sleep_policy = policy;
        self
    }

    /// Build the [`Engine`]. Equivalent to calling `Engine::with_config_boxed(...)`
    /// then setting the optional fields directly.
    #[must_use]
    pub fn build(self) -> Engine {
        let mut engine = Engine::with_config_boxed(
            self.target_model,
            self.draft_model,
            self.config,
            self.max_draft_tokens,
            self.num_kv_blocks,
        );
        engine.adaptive_decoder = self.adaptive_decoder;
        engine.draft_resolver = self.draft_resolver;
        engine.sleep_policy = self.sleep_policy;
        engine
    }
}
