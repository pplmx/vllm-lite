// Sub-module for graph-based step execution on Engine.
// See mod.rs for the Engine struct definition.
//
// Note: `step_with_graph` has both a `cfg(feature = "cuda-graph")` and a
// `cfg(not(feature = "cuda-graph"))` variant — the non-cuda-graph path falls
// back to the regular `step()` so callers compile unchanged.

use crate::engine::Engine;
use crate::error::Result;
use vllm_traits::{SeqId, TokenId};

#[cfg(feature = "cuda-graph")]
use tracing::trace;

#[cfg(feature = "cuda-graph")]
use vllm_traits::{BatchOutput, BatchPhase};

impl Engine {
    #[cfg(feature = "cuda-graph")]
    /// Run a single scheduling + model-forward step using a captured CUDA
    /// Graph when one is available for the current batch size; falls back to
    /// the regular forward path otherwise. The decision is made by
    /// [`crate::scheduler::engine::SchedulerEngine::build_batch_with_graph`],
    /// which produces either a `Graph(prepared)` variant or a `Regular(batch)`
    /// variant — this method just dispatches.
    ///
    /// Returns the list of `(seq_id, token_id)` pairs produced this step.
    /// An empty vector means there was no pending work.
    ///
    /// # Errors
    ///
    /// Propagates any error from the underlying forward pass or graph
    /// executor (after first attempting the regular-step fallback when
    /// graph execution fails).
    pub fn step_with_graph(&mut self) -> Result<Vec<(SeqId, TokenId)>> {
        let start = std::time::Instant::now();
        let graph_batch = self.scheduler.build_batch_with_graph();
        if graph_batch.batch_size() == 0 {
            return Ok(vec![]);
        }

        let (output, batch) = match graph_batch {
            crate::scheduler::GraphBatch::Graph(prepared) => {
                let batch = prepared.batch;
                let input_counts: Vec<usize> =
                    batch.input_tokens.iter().map(std::vec::Vec::len).collect();
                let output = if let Some(ref executor) = self.cuda_graph {
                    match executor.execute(&batch) {
                        Ok(output) => output,
                        Err(e) => {
                            tracing::warn!("CUDA Graph execution failed: {}, falling back", e);
                            self.execute_regular(&batch)?
                        }
                    }
                } else {
                    self.execute_regular(&batch)?
                };
                (output, input_counts)
            }
            crate::scheduler::GraphBatch::Regular(batch) => {
                let input_counts: Vec<usize> =
                    batch.input_tokens.iter().map(std::vec::Vec::len).collect();
                let output = self.execute_regular(&batch)?;
                (output, input_counts)
            }
        };

        self.process_output(output, batch, start)
    }

    #[cfg(not(feature = "cuda-graph"))]
    /// Fallback for non-`cuda-graph` builds: delegates straight to
    /// [`Engine::step`]. Always available so call sites compile unchanged
    /// regardless of feature flags.
    pub fn step_with_graph(&mut self) -> Result<Vec<(SeqId, TokenId)>> {
        tracing::warn!("CUDA Graph support not enabled, using regular step");
        self.step()
    }

    /// Execute regular forward pass (used by CUDA Graph fallback path).
    #[cfg(feature = "cuda-graph")]
    fn execute_regular(&mut self, batch: &vllm_traits::Batch) -> Result<BatchOutput> {
        use crate::sync::lock_mutex;

        let total_tokens: usize = batch.input_tokens.iter().map(std::vec::Vec::len).sum();
        tracing::debug!(
            batch_size = batch.seq_ids.len(),
            total_tokens = total_tokens,
            is_prefill = matches!(batch.phase, BatchPhase::Prefill),
            "Model forward started"
        );

        let start = std::time::Instant::now();
        let result = {
            let mut model = lock_mutex(&self.target_model)?;
            model.forward(
                &batch.seq_ids,
                &batch.input_tokens,
                &batch.positions,
                &batch.kv_block_ids,
                &batch.num_computed_tokens,
                &batch.is_prefill,
            )
        };
        let elapsed = start.elapsed().as_millis() as u64;

        match result {
            Ok(output) => {
                tracing::debug!(
                    elapsed_ms = elapsed,
                    tokens = output.next_tokens.len(),
                    "Model forward completed"
                );
                Ok(output)
            }
            Err(e) => {
                tracing::error!(error = %e, "Model forward failed");
                Err(crate::error::EngineError::from(e))
            }
        }
    }

    /// Process model output and update state (CUDA Graph path).
    #[cfg(feature = "cuda-graph")]
    fn process_output(
        &mut self,
        output: BatchOutput,
        input_counts: Vec<usize>,
        start: std::time::Instant,
    ) -> Result<Vec<(SeqId, TokenId)>> {
        tracing::debug!(
            seq_ids = ?output.seq_ids,
            tokens = ?output.next_tokens,
            "process_output: received model output"
        );

        let mut results = Vec::new();
        for (seq_id, token) in output.seq_ids.iter().zip(&output.next_tokens) {
            trace!(
                seq_id = %seq_id,
                token_id = %token,
                "Token generated"
            );
            tracing::debug!(seq_id = %seq_id, token = %token, "Sending token via channel");
            if let Some(tx) = self.response_txs.get(seq_id) {
                let _ = tx.try_send(*token);
            }
            results.push((*seq_id, *token));
        }

        let seq_ids: Vec<_> = results.iter().map(|(id, _)| *id).collect();
        let tokens: Vec<_> = results.iter().map(|(_, t)| *t).collect();
        self.scheduler.update(&seq_ids, &tokens, &input_counts);

        let finished = self.scheduler.finished_sequences();
        for seq in &finished {
            if let Some(tx) = self.response_txs.remove(&seq.id) {
                drop(tx);
            }
        }
        self.scheduler.clear_finished();

        // Record metrics
        if !results.is_empty() {
            self.scheduler.metrics.record_tokens(results.len() as u64);
            self.scheduler.metrics.record_batch_size(results.len());
            let elapsed = start.elapsed().as_millis() as f64;
            if elapsed > 0.0 {
                self.scheduler.metrics.record_latency(elapsed);
            }
        }

        Ok(results)
    }
}
