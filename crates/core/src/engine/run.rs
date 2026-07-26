//! Engine main run loop: tick → schedule → forward → update, plus the `has_pending` accessor.
//!
//! The loop runs on a dedicated OS thread; cancellation is
//! cooperatively checked at the top of each tick.

// Sub-module for the main run loop and has_pending accessor on Engine.
// See mod.rs for the Engine struct definition.

use crate::engine::Engine;
use crate::sync::lock_mutex;
use crate::types::EngineMessage;
use tokio::sync::mpsc;
use tracing::error;

impl Engine {
    /// Run the engine's main loop, draining `msg_rx` and stepping the
    /// scheduler until a `Shutdown` message is received.
    ///
    /// The loop is single-threaded and non-async: incoming messages are
    /// drained with `try_recv`, then one model step is executed if any
    /// sequence is pending, then the thread sleeps for the duration
    /// produced by the current [`crate::engine::SleepPolicy`]. This pattern gives
    /// back-pressure-friendly batching without an async runtime.
    ///
    /// This call blocks the current thread and never returns except on
    /// `EngineMessage::Shutdown`. Spawn it on a dedicated worker thread
    /// (the `vllm-server` crate does this for you).
    pub fn run(&mut self, mut msg_rx: mpsc::Receiver<EngineMessage>) {
        let mut step_count = 0u64;
        loop {
            while let Ok(msg) = msg_rx.try_recv() {
                match msg {
                    EngineMessage::AddRequest {
                        request,
                        response_tx,
                        seq_id_tx,
                        finish_reason_tx,
                        request_id,
                    } => self.handle_add_request(
                        *request,
                        response_tx,
                        seq_id_tx,
                        finish_reason_tx,
                        request_id.as_ref(),
                    ),
                    EngineMessage::CancelRequest { seq_id } => {
                        // Production-readiness recommendation: when an
                        // HTTP client disconnects mid-stream, the
                        // handler sends CancelRequest to release the
                        // sequence's KV blocks and response channel.
                        // Unknown seq_id is a no-op (race with natural
                        // completion) — `cancel_request` returns
                        // false and we drop the message silently.
                        let _ = self.cancel_request(seq_id);
                    }
                    EngineMessage::GetMetrics { response_tx } => {
                        let (used, total) = self.scheduler.get_kv_cache_usage();
                        self.scheduler.metrics.record_kv_cache_usage(used, total);
                        let _ = response_tx.send(self.scheduler.metrics.snapshot());
                    }
                    EngineMessage::GetEmbeddings {
                        input_tokens,
                        response_tx,
                    } => self.handle_get_embeddings(&input_tokens, &response_tx),
                    EngineMessage::Shutdown => return,
                }
            }

            if self.scheduler.has_pending() {
                step_count += 1;
                let result = if self.cuda_graph_enabled() && !self.speculative_mode {
                    self.step_with_graph()
                } else {
                    self.step()
                };
                if let Err(e) = result {
                    self.error_count += 1;
                    self.last_error = Some(e.to_string());
                    error!(step = step_count, error = %e, "Engine step error");
                }
            }

            let interval = self
                .sleep_policy
                .next_interval(self.scheduler.has_pending());
            std::thread::sleep(std::time::Duration::from_millis(interval));
        }
    }

    /// Handle an `AddRequest` message: enter the request-id tracing span,
    /// admit the request, and reply on the `seq_id` / `finish_reason` channels.
    fn handle_add_request(
        &mut self,
        request: crate::types::Request,
        response_tx: tokio::sync::mpsc::Sender<vllm_traits::SampledToken>,
        seq_id_tx: Option<tokio::sync::oneshot::Sender<vllm_traits::SeqId>>,
        finish_reason_tx: Option<tokio::sync::oneshot::Sender<vllm_traits::FinishReason>>,
        request_id: Option<&String>,
    ) {
        // Production-readiness §6 (日志与追踪): when an HTTP handler
        // forwards a `request_id`, enter a tracing::info_span! so every
        // engine-side log line for this HTTP request carries the same
        // correlation id. When `request_id` is `None` (test fixtures,
        // non-HTTP callers) the span still enters, rendering as `null`.
        let _request_id_span = tracing::info_span!("engine.add_request", request_id = request_id);

        // P38: deref Box<Request> → Request (add_request public API
        // still takes Request by value). The caller treats seq_id 0 as
        // "do not bother cancelling" (rejection e.g. empty prompt).
        let seq_id = self.add_request(request, response_tx);
        if let Some(tx) = finish_reason_tx {
            self.finish_reason_txs.insert(seq_id, tx);
        }
        if let Some(tx) = seq_id_tx {
            let _ = tx.send(seq_id);
        }
    }

    /// Handle a `GetEmbeddings` message: call `model.embed` and send the
    /// result (or log the error).
    fn handle_get_embeddings(
        &self,
        input_tokens: &[Vec<vllm_traits::TokenId>],
        response_tx: &tokio::sync::mpsc::UnboundedSender<Vec<Vec<f32>>>,
    ) {
        let positions: Vec<Vec<usize>> = input_tokens
            .iter()
            .map(|tokens| (0..tokens.len()).collect())
            .collect();
        match lock_mutex(&self.target_model)
            .and_then(|mut model| model.embed(input_tokens, &positions).map_err(Into::into))
        {
            Ok(embeddings) => {
                let _ = response_tx.send(embeddings);
            }
            Err(e) => {
                error!(error = %e, "Embeddings error");
            }
        }
    }

    /// Returns `true` if the scheduler currently has at least one waiting or
    /// running sequence. Useful for tests and external monitors that want to
    /// know whether calling [`Engine::step`] would do meaningful work.
    pub fn has_pending(&self) -> bool {
        self.scheduler.has_pending()
    }
}
