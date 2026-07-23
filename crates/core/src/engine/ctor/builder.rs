//! `EngineBuilder` — named-method builder for [`Engine`]. Equivalent to
//! calling `Engine::with_config_boxed(...)` then setting the optional
//! fields directly.

use std::sync::Arc;

use crate::engine::Engine;
use crate::engine::SleepPolicy;
use crate::speculative::AdaptiveSpeculativeDecoder;
use crate::speculative::draft_resolver::DraftResolver;
use crate::types::SchedulerConfig;
use vllm_traits::ModelBackend;

#[cfg(feature = "cuda-graph")]
use vllm_traits::CudaGraphExecutor;

#[cfg(feature = "multi-node")]
use parking_lot::Mutex;

#[cfg(feature = "multi-node")]
use vllm_dist::DistributedKVCache;

#[cfg(feature = "multi-node")]
use vllm_model::paged_tensor::PagedKvCache;

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
    /// Caller-supplied CUDA-Graph executor. Overrides anything the engine
    /// constructor would build from `config.cuda_graph.enabled`. Lets
    /// callers plug in their own `Box<dyn CudaGraphExecutor>` without
    /// `vllm-core` knowing the concrete type.
    #[cfg(feature = "cuda-graph")]
    cuda_graph_executor: Option<Box<dyn CudaGraphExecutor + Send>>,
    /// Caller-supplied distributed KV-cache. The cache is owned by the
    /// engine and reachable via [`Engine::distributed_kv_enabled`] and
    /// [`Engine::distributed_kv_stats`]; the [`MemoryManager`] performs
    /// allocate/free round-trips through it.
    ///
    /// [`MemoryManager`]: crate::scheduler::memory::MemoryManager
    #[cfg(feature = "multi-node")]
    distributed_kv: Option<Arc<DistributedKVCache>>,
    /// Caller-supplied `PagedKvCache` for multi-node KV block byte
    /// transfer (Phase 41 OPS-32a second-half). When set, the builder
    /// constructs a `PagedKvCacheWrapper` and threads it through
    /// `Engine::set_paged_kv_cache` → `SchedulerEngine::set_block_data_source`
    /// → `MemoryManager::block_data_source` at build time so subsequent
    /// gRPC `TransferKVBlock` calls resolve to the wrapper.
    #[cfg(feature = "multi-node")]
    paged_kv_cache: Option<Arc<Mutex<PagedKvCache>>>,
}

impl std::fmt::Debug for EngineBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dbg = f.debug_struct("EngineBuilder");
        dbg.field("target_model", &"<dyn ModelBackend>")
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
            .field("sleep_policy", &self.sleep_policy);
        #[cfg(feature = "cuda-graph")]
        dbg.field(
            "cuda_graph_executor",
            &self
                .cuda_graph_executor
                .as_ref()
                .map(|_| "<dyn CudaGraphExecutor>"),
        );
        #[cfg(feature = "multi-node")]
        dbg.field(
            "distributed_kv",
            &self.distributed_kv.as_ref().map(Arc::strong_count),
        );
        #[cfg(feature = "multi-node")]
        dbg.field(
            "paged_kv_cache",
            &self.paged_kv_cache.as_ref().map(Arc::strong_count),
        );
        dbg.finish()
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
            #[cfg(feature = "cuda-graph")]
            cuda_graph_executor: None,
            #[cfg(feature = "multi-node")]
            distributed_kv: None,
            #[cfg(feature = "multi-node")]
            paged_kv_cache: None,
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

    /// Override the sleep policy.
    #[must_use]
    pub const fn with_sleep_policy(mut self, policy: SleepPolicy) -> Self {
        self.sleep_policy = policy;
        self
    }

    /// Plug in a pre-built CUDA-Graph executor.
    ///
    /// When set, this overrides whatever `Engine::with_config_boxed` would
    /// have constructed from `config.cuda_graph`. Use this when the
    /// concrete executor type lives outside `vllm-core` — the
    /// [`vllm_traits::CudaGraphExecutor`] trait object is what makes
    /// that possible.
    #[must_use]
    #[cfg(feature = "cuda-graph")]
    pub fn with_cuda_graph_executor(mut self, executor: Box<dyn CudaGraphExecutor + Send>) -> Self {
        self.cuda_graph_executor = Some(executor);
        self
    }

    /// Plug in a pre-built distributed KV-cache.
    ///
    /// The cache is owned by the engine and reachable via
    /// [`Engine::distributed_kv_enabled`] and [`Engine::distributed_kv_stats`].
    /// See [`MemoryManager`] for how block allocate / free round-trips
    /// through the cache.
    ///
    /// [`MemoryManager`]: crate::scheduler::memory::MemoryManager
    #[must_use]
    #[cfg(feature = "multi-node")]
    pub fn with_distributed_kv(mut self, cache: Arc<DistributedKVCache>) -> Self {
        self.distributed_kv = Some(cache);
        self
    }

    /// Wire a `PagedKvCache` into the engine for multi-node KV block
    /// replication (Phase 41 OPS-32a second-half).
    ///
    /// Takes the cache pre-wrapped in `Arc<Mutex<PagedKvCache>>` (the shape
    /// yielded by `Architecture::create_model` via `ModelLoader::paged_kv_cache_clone`)
    /// and threads it — along with a `PagedKvCacheWrapper` constructed from it —
    /// through `Engine::set_paged_kv_cache` at `.build()` time so every
    /// subsequent gRPC `TransferKVBlock` call resolves to the wrapper.
    /// Mirrors [`Self::with_distributed_kv`] (which wires the metadata
    /// cache).
    #[must_use]
    #[cfg(feature = "multi-node")]
    pub fn with_paged_kv_cache(mut self, cache: Arc<Mutex<PagedKvCache>>) -> Self {
        self.paged_kv_cache = Some(cache);
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
        #[cfg(feature = "cuda-graph")]
        {
            if let Some(executor) = self.cuda_graph_executor {
                engine.set_cuda_graph_executor(executor);
            }
        }
        #[cfg(feature = "multi-node")]
        {
            if let Some(cache) = self.distributed_kv {
                engine.set_distributed_kv(cache);
            }
            if let Some(cache) = self.paged_kv_cache {
                engine.set_paged_kv_cache(cache);
            }
        }
        engine
    }
}

#[cfg(test)]
impl EngineBuilder {
    /// Set an optional adaptive speculative decoder.
    #[must_use]
    pub(crate) fn with_adaptive_decoder(mut self, decoder: AdaptiveSpeculativeDecoder) -> Self {
        self.adaptive_decoder = Some(decoder);
        self
    }

    /// Set an optional per-request draft resolver (v18+).
    #[must_use]
    pub(crate) fn with_draft_resolver(mut self, resolver: Arc<DraftResolver>) -> Self {
        self.draft_resolver = Some(resolver);
        self
    }
}
