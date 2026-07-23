#![allow(clippy::module_name_repetitions)]
//! Engine module — see sub-modules for specific method groups.

mod ctor;
mod cuda_graph;
mod distributed_kv;
mod draft_management;
mod graph_step;
mod lifecycle;
#[cfg(feature = "multi-node")]
mod paged_kv_cache;
mod run;
mod spec_dispatch;

pub use ctor::EngineBuilder;

use crate::scheduler::engine::SchedulerEngine;
use crate::speculative::AdaptiveSpeculativeDecoder;
use crate::speculative::draft_resolver::DraftResolver;
use crate::speculative::registry::DraftModelRegistry;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};
use vllm_traits::{FinishReason, ModelBackend, SampledToken, SeqId};

#[cfg(feature = "cuda-graph")]
use vllm_traits::CudaGraphExecutor;

#[cfg(feature = "multi-node")]
use vllm_dist::DistributedKVCache;

/// Core inference engine managing requests, scheduling, and model execution.
///
/// The Engine orchestrates the entire inference pipeline:
/// - Receives requests via `add_request`
/// - Schedules batches via the Scheduler
/// - Executes model forward passes
/// - Streams generated tokens back to clients
///
/// # Thread Safety
///
/// The Engine runs on its own dedicated thread (via `run`), using `RefCell`
/// for interior mutability of model references. All external communication
/// happens through mpsc channels (actor pattern).
pub struct Engine {
    /// Token-batch scheduler. Owns the queue, preemption policy, and
    /// prefix-cache lookups for every in-flight sequence.
    pub scheduler: SchedulerEngine,
    /// The verified (target) language model used for the canonical
    /// forward pass on each step. Wrapped in a `Mutex<Box<dyn>>` so the
    /// engine can swap implementations (e.g. paged vs. fused) at runtime.
    pub target_model: Arc<Mutex<Box<dyn ModelBackend>>>,
    /// Optional draft model used by the legacy single-draft speculative
    /// path. When [`Self::draft_resolver`] is `Some`, this is unused and
    /// `None` is preferred for new code paths.
    pub draft_model: Option<Arc<Mutex<Box<dyn ModelBackend>>>>,
    /// Maximum number of draft tokens generated per step on the legacy
    /// single-draft path. Ignored when [`Self::draft_resolver`] is set.
    pub max_draft_tokens: usize,
    /// When `true`, the step loop interleaves draft-token generation and
    /// verification. When `false`, each step runs the target model
    /// directly with no speculative decoding.
    pub speculative_mode: bool,
    /// Monotonically increasing count of recoverable errors observed
    /// by the engine. Used for health-check thresholds and metrics
    /// export; tests and integration suites inspect this field.
    pub error_count: usize,
    /// Most recent recoverable error message (string form). Kept for
    /// log/diagnostics; the structured error chain is on the originating
    /// request, this is just a convenience field for quick inspection.
    pub last_error: Option<String>,
    /// Per-sequence mpsc senders for streaming generated tokens back
    /// to requesters. Map keyed by [`SeqId`]; entries are removed
    /// when the receiver is dropped. Visible to integration tests.
    ///
    /// **P36 v0.3 wire-type follow-up engine wire-through:** the
    /// sender carries [`SampledToken`] (token + logprob +
    /// top_logprobs) instead of a bare `TokenId` so the HTTP layer
    /// can render OpenAI's `choices[].logprobs` shape without
    /// re-running the softmax.
    pub response_txs: HashMap<SeqId, mpsc::Sender<SampledToken>>,
    /// Per-sequence oneshot senders for delivering the [`FinishReason`]
    /// that describes why the sequence stopped (`Length`, `Cancelled`,
    /// …). Populated by [`crate::engine::Engine::add_request`] (defined
    /// in the `lifecycle` sub-module) when the caller supplies one (the
    /// HTTP streaming handlers do); drained by the same paths that drop
    /// the matching `response_txs` entry so the handler learns the
    /// reason **before** the channel close. See the v31.0 P4 follow-up
    /// batch and `docs/technical-due-diligence/architecture-performance.md`
    /// §5.1.2 — pre-fix this reason was lost and the HTTP layer hardcoded
    /// `finish_reason = "stop"` for every response.
    pub finish_reason_txs: HashMap<SeqId, oneshot::Sender<FinishReason>>,
    sleep_policy: SleepPolicy,
    /// Optional CUDA-Graph executor behind a trait object. The concrete
    /// type (`vllm_model::kernels::BatchCudaGraphExecutor`) is built by the
    /// engine constructor and immediately boxed — every other engine call
    /// site talks to the [`vllm_traits::CudaGraphExecutor`] trait, not the
    /// concrete type.
    #[cfg(feature = "cuda-graph")]
    cuda_graph: Option<Box<dyn CudaGraphExecutor + Send>>,
    /// Optional distributed KV-cache for cross-node cache coherence.
    /// Surfaces the [`vllm_dist::DistributedKVCache`] seam to engine
    /// callers; the allocator-level hooks that wire block allocate/free
    /// into the cache are wired in
    /// [`crate::scheduler::memory::MemoryManager`]. The field exists so
    /// callers can construct a multi-node engine, query its status, and
    /// own the cache for the engine's lifetime.
    #[cfg(feature = "multi-node")]
    distributed_kv: Option<Arc<DistributedKVCache>>,
    /// Optional `PagedKvCache` for multi-node KV block byte transfer
    /// (Phase 41 OPS-32a second-half; P42 receiver-side). Set via
    /// [`crate::engine::EngineBuilder::with_paged_kv_cache`] which
    /// also constructs the wrapper and threads it to
    /// [`crate::scheduler::memory::MemoryManager::block_data_source`].
    /// The engine stores both the cache (so callers can introspect or
    /// feed it) and the wrapper (so the server bootstrap can hand it
    /// to the gRPC server without re-constructing).
    ///
    /// Wrapped in `Arc<Mutex<>>` (P42) so the wrapper can write
    /// (`write_block`) while the engine still holds its own Arc for
    /// diagnostics / future read paths. The two Arcs share the same
    /// Mutex so a write through one is visible to reads through the
    /// other.
    #[cfg(feature = "multi-node")]
    paged_kv_cache: Option<Arc<parking_lot::Mutex<vllm_model::paged_tensor::PagedKvCache>>>,
    /// Cached `BlockDataSource` wrapper produced by
    /// [`crate::engine::Engine::set_paged_kv_cache`] so
    /// `crates/server/src/bootstrap/engine.rs` can hand it to the gRPC
    /// server's `start_grpc_server_with_listener` without re-constructing.
    #[cfg(feature = "multi-node")]
    paged_kv_cache_wrapper: Option<Arc<dyn vllm_dist::BlockDataSource + Send + Sync>>,
    /// Optional adaptive speculative decoder that tunes the draft-token
    /// budget based on observed acceptance rates.
    pub adaptive_decoder: Option<AdaptiveSpeculativeDecoder>,
    /// Shared draft-model registry. Holds every loaded draft spec keyed
    /// by name; the resolver queries this map during each step.
    pub draft_registry: Arc<DraftModelRegistry>,
    /// v18.0 per-request draft resolver. When `Some`, the step loop dispatches
    /// each request to its named draft via `resolver.resolve()`. When `None`,
    /// the legacy single-`draft_model` path is used (v17 behavior, backward
    /// compatible with `Engine::new_boxed`).
    pub draft_resolver: Option<Arc<DraftResolver>>,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dbg = f.debug_struct("Engine");
        dbg.field("scheduler", &self.scheduler)
            .field("target_model", &"<dyn ModelBackend>")
            .field(
                "draft_model",
                &self.draft_model.as_ref().map(|_| "<dyn ModelBackend>"),
            )
            .field("max_draft_tokens", &self.max_draft_tokens)
            .field("speculative_mode", &self.speculative_mode)
            .field("error_count", &self.error_count)
            .field("last_error", &self.last_error)
            .field("response_txs_count", &self.response_txs.len())
            .field("finish_reason_txs_count", &self.finish_reason_txs.len())
            .field("sleep_policy", &self.sleep_policy)
            .field("adaptive_decoder", &self.adaptive_decoder)
            .field("draft_registry", &self.draft_registry)
            .field(
                "draft_resolver",
                &self.draft_resolver.as_ref().map(Arc::strong_count),
            );
        #[cfg(feature = "cuda-graph")]
        dbg.field(
            "cuda_graph",
            &self.cuda_graph.as_ref().map(|_| "<dyn CudaGraphExecutor>"),
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
        #[cfg(feature = "multi-node")]
        dbg.field(
            "paged_kv_cache_wrapper",
            &self.paged_kv_cache_wrapper.as_ref().map(Arc::strong_count),
        );
        dbg.finish_non_exhaustive()
    }
}

/// Adaptive sleep policy for the engine idle loop.
///
/// Tracks consecutive idle ticks and grows the sleep interval
/// geometrically up to [`Self::max_interval`], then snaps back to
/// [`Self::base_interval`] when work resumes. Used by the step loop to
/// avoid busy-spinning when the scheduler has no work.
#[derive(Debug)]
pub struct SleepPolicy {
    /// Minimum sleep interval (ms) used when the engine just became
    /// idle or has been busy recently.
    pub base_interval: u64,
    /// Upper bound (ms) on the sleep interval — the geometric growth
    /// caps here so the engine remains responsive to late-arriving
    /// requests even after long idle periods.
    pub max_interval: u64,
    /// Multiplicative growth factor applied each consecutive idle tick.
    /// A value of 1.5 grows from `base` (1ms) to `max` (50ms) in roughly
    /// 10 ticks.
    pub backoff_factor: f64,
    /// Number of consecutive idle ticks observed so far. Reset to 0
    /// by [`Self::next_interval`] whenever `has_work` is `true`.
    pub consecutive_idle: u32,
}

impl Default for SleepPolicy {
    fn default() -> Self {
        Self {
            base_interval: 1,
            max_interval: 50,
            backoff_factor: 1.5,
            consecutive_idle: 0,
        }
    }
}

impl SleepPolicy {
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap
    )]
    /// Return the wait duration (ms) before the next scheduler tick, extending backoff when idle.
    pub fn next_interval(&mut self, has_work: bool) -> u64 {
        if has_work {
            self.consecutive_idle = 0;
            return self.base_interval;
        }

        self.consecutive_idle += 1;

        if self.consecutive_idle == 1 {
            return self.base_interval;
        }

        ((self.base_interval as f64) * self.backoff_factor.powi(self.consecutive_idle as i32 - 1))
            .min(self.max_interval as f64) as u64
    }
}

// Unit tests are extracted to `tests.rs` to keep this file under the
// 800-line soft cap. See `tests.rs` for the test surface (Engine /
// SleepPolicy / EngineBuilder / draft-registry / memory-budget).
#[cfg(test)]
mod tests;
