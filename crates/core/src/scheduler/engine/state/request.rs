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
    pub fn add_request(&mut self, mut req: Request) -> SeqId {
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

        // Check prefix cache for prompt reuse
        let (tokens, kv_blocks, num_computed) =
            if let Some(result) = self.prefix_cache.longest_prefix_match(&req.prompt) {
                tracing::trace!(
                    request_id = req.id,
                    matched_tokens = result.matched_tokens,
                    "Prefix cache hit"
                );
                (
                    req.prompt.clone(),
                    result.blocks.clone(),
                    result.matched_tokens,
                )
            } else {
                tracing::trace!(request_id = req.id, "Prefix cache miss");
                (req.prompt.clone(), Arc::new(vec![]), 0)
            };

        // Distributed prefix-cache lookup (OPS-05b3): even when the
        // local `RadixTree` misses, some peer node (post OPS-05c)
        // may hold KV for this prompt's prefix. The result is
        // informational today — actual block reuse requires the
        // gRPC transfer protocol. We dispatch an observer event so
        // metrics collectors / tracing can report cross-node hit
        // rates.
        #[cfg(feature = "multi-node")]
        let distributed_matched_tokens = self
            .lookup_distributed_prefix(&req.prompt)
            .map_or(0, |m| m.matched_tokens);
        #[cfg(not(feature = "multi-node"))]
        let distributed_matched_tokens: usize = 0;
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
            status: if num_computed >= req.prompt.len() {
                Status::Waiting
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
}
