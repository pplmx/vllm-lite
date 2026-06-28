// Sub-module for the main run loop and has_pending accessor on Engine.
// See mod.rs for the Engine struct definition.

use crate::engine::Engine;
use crate::sync::lock_mutex;
use crate::types::EngineMessage;
use tokio::sync::mpsc;
use tracing::error;

impl Engine {
    pub fn run(&mut self, mut msg_rx: mpsc::UnboundedReceiver<EngineMessage>) {
        let mut step_count = 0u64;
        loop {
            while let Ok(msg) = msg_rx.try_recv() {
                match msg {
                    EngineMessage::AddRequest {
                        request,
                        response_tx,
                    } => {
                        self.add_request(request, response_tx);
                    }
                    EngineMessage::GetMetrics { response_tx } => {
                        let (used, total) = self.scheduler.get_kv_cache_usage();
                        self.scheduler.metrics.record_kv_cache_usage(used, total);
                        let snapshot = self.scheduler.metrics.snapshot();
                        let _ = response_tx.send(snapshot);
                    }
                    EngineMessage::GetEmbeddings {
                        input_tokens,
                        response_tx,
                    } => {
                        let positions: Vec<Vec<usize>> = input_tokens
                            .iter()
                            .map(|tokens| (0..tokens.len()).collect())
                            .collect();
                        match lock_mutex(&self.target_model).and_then(|mut model| {
                            model.embed(&input_tokens, &positions).map_err(Into::into)
                        }) {
                            Ok(embeddings) => {
                                let _ = response_tx.send(embeddings);
                            }
                            Err(e) => {
                                error!(error = %e, "Embeddings error");
                            }
                        }
                    }
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

            let has_pending = self.scheduler.has_pending();
            let interval = self.sleep_policy.next_interval(has_pending);
            std::thread::sleep(std::time::Duration::from_millis(interval));
        }
    }

    pub fn has_pending(&self) -> bool {
        self.scheduler.has_pending()
    }
}
