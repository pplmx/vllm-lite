//! `SchedulerEngine::add_request` — enqueue a new request, check the
//! prefix cache for prompt reuse, and dispatch the
//! `RequestArrived` observer event.

use std::sync::Arc;
use std::time::Instant;

use vllm_traits::SeqId;

use super::SchedulerEngine;
use crate::scheduler::observer::ObserverEvent;
use crate::scheduler::policy::SchedulingContext;
use crate::types::{Request, Sequence, Status};

impl SchedulerEngine {
    /// Add a new request to the scheduler
    ///
    /// Checks the prefix cache for matching prompts and creates a sequence.
    /// Returns the assigned sequence ID.
    pub fn add_request(&mut self, req: Request) -> SeqId {
        self.add_request_inner(req, true)
    }

    /// Add a request WITHOUT consulting the prefix cache (RIL ISS-038).
    ///
    /// Stateful backends whose recurrent hidden state is not stored in the
    /// KV cache (Qwen3.5 hybrid GDN layers) cannot safely resume from a
    /// prefix hit — a full hit would admit the sequence into Decode with
    /// fresh `None` states. The engine routes such requests here so they
    /// always run the full prefill (rebuilding the recurrent state).
    pub fn add_request_without_prefix_cache(&mut self, req: Request) -> SeqId {
        self.add_request_inner(req, false)
    }

    fn add_request_inner(&mut self, mut req: Request, use_prefix_cache: bool) -> SeqId {
        let _span = tracing::info_span!(
            "scheduler.add_request",
            request_id = req.id,
            prompt_len = req.prompt.len(),
            max_tokens = req.max_tokens
        )
        .entered();

        // Record metrics: request received
        self.metrics.record_request();

        if req.id == 0 {
            req.id = self.next_seq_id;
            self.next_seq_id += 1;
        }

        // Check prefix cache for prompt reuse (skipped for stateful models
        // that cannot resume from cached KV — see
        // `add_request_without_prefix_cache`).
        let (tokens, kv_blocks, num_computed) = if use_prefix_cache {
            self.resolve_prefix_tokens(&req)
        } else {
            (req.prompt.clone(), Arc::new(vec![]), 0)
        };

        // Distributed prefix-cache lookup (OPS-05b3): even when the
        // local `RadixTree` misses, some peer node (post OPS-05c)
        // may hold KV for this prompt's prefix. The result is
        // informational today — actual block reuse requires the
        // gRPC transfer protocol. We dispatch an observer event so
        // metrics collectors / tracing can report cross-node hit
        // rates.
        let distributed_matched_tokens = if use_prefix_cache {
            self.lookup_distributed_matched_tokens(&req)
        } else {
            0
        };
        if distributed_matched_tokens > 0 {
            tracing::trace!(
                request_id = req.id,
                matched_tokens = distributed_matched_tokens,
                "Distributed prefix hit"
            );
        }

        let seq = Sequence {
            id: req.id,
            tokens,
            kv_blocks,
            num_computed_tokens: num_computed,
            prompt_len: req.prompt.len(),
            // Full prefix-cache hit: the prompt's KV is already
            // computed, so the sequence goes straight to decode. The
            // decode composer feeds the last prompt token exactly as
            // it does for a freshly prefilled sequence. Leaving a
            // full hit in `Waiting` stalls it forever — the prefill
            // composer skips sequences with no new tokens, and decode
            // batches only admit `Decoding` sequences (RIL ISS-005).
            // Empty prompts never hit the cache and keep the normal
            // prefill path (a `Decoding` sequence with no tokens
            // would decode from a phantom token).
            status: if num_computed >= req.prompt.len() && !req.prompt.is_empty() {
                Status::Decoding
            } else {
                Status::Prefilling
            },
            max_tokens: req.max_tokens,
            sampling_params: req.sampling_params,
            consecutive_decode_rounds: 0,
            priority: req.priority,
            degraded_draft: false,
            draft_model_id: req.draft_model_id.clone(),
        };

        let ctx = SchedulingContext {
            current_time: Instant::now(),
            queue_length: self.request_queue.len(),
            running_count: self.running.len(),
            memory_pressure: self.get_memory_pressure(),
        };

        self.request_queue.enqueue(seq, self.policy.as_ref(), &ctx);

        // Update metrics: queue depth
        self.metrics
            .set_queue_depth(self.request_queue.len() as u64);

        // Dispatch observer event
        self.observers.dispatch(&ObserverEvent::RequestArrived {
            seq_id: req.id,
            prompt_len: req.prompt.len(),
        });
        // Distributed prefix-cache result (OPS-05b3). Dispatched
        // unconditionally — the no-op observer just drops it; the
        // default `NoopSchedulerObserver` is silent.
        self.observers
            .dispatch(&ObserverEvent::DistributedPrefixMatched {
                seq_id: req.id,
                matched_tokens: distributed_matched_tokens,
            });

        tracing::info!(
            request_id = req.id,
            queue_depth = self.request_queue.len(),
            "Request added"
        );
        req.id
    }

    /// Resolve prompt tokens, KV blocks, and computed-token count from
    /// the prefix cache.
    ///
    /// On a cache hit, returns the full prompt tokens, the matched KV
    /// blocks (refcounted), and the number of matched tokens. On a miss,
    /// returns the prompt, empty blocks, and 0.
    ///
    /// ARCH-01: on a hit, the blocks are refcounted so this sequence
    /// owns them and can release them on cancel/finish; the prefix
    /// cache retains its own reference via the `RadixTree` node.
    fn resolve_prefix_tokens(
        &mut self,
        req: &Request,
    ) -> (Vec<u32>, Arc<Vec<vllm_traits::BlockId>>, usize) {
        // RIL ISS-006: the hit-rate counters were wired to the metrics
        // collector but never recorded, so `prefix_cache_hit_rate()`
        // always reported 0. Count every lookup, then every hit.
        self.metrics.record_prefix_cache_request();
        if let Some(result) = self.prefix_cache.longest_prefix_match(&req.prompt) {
            self.metrics.record_prefix_cache_hit();
            tracing::trace!(
                request_id = req.id,
                matched_tokens = result.matched_tokens,
                "Prefix cache hit"
            );
            self.memory.record_blocks(result.blocks.as_ref());
            (
                req.prompt.clone(),
                result.blocks.clone(),
                result.matched_tokens,
            )
        } else {
            tracing::trace!(request_id = req.id, "Prefix cache miss");
            (req.prompt.clone(), Arc::new(vec![]), 0)
        }
    }

    /// Distributed prefix-cache lookup (multi-node only).
    ///
    /// Returns the number of matched tokens from peer nodes, or 0 in
    /// single-node builds. The result is informational — actual block
    /// reuse requires the gRPC transfer protocol.
    #[cfg_attr(
        not(feature = "multi-node"),
        allow(clippy::missing_const_for_fn, clippy::unused_self)
    )]
    fn lookup_distributed_matched_tokens(&self, req: &Request) -> usize {
        #[cfg(feature = "multi-node")]
        {
            self.lookup_distributed_prefix(&req.prompt)
                .map_or(0, |m| m.matched_tokens)
        }
        #[cfg(not(feature = "multi-node"))]
        {
            let _ = req;
            0
        }
    }
}
