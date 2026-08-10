//! Regression (RIL ISS-034): the engine must notify the target model when a
//! sequence finishes (max_tokens / stop / cancel) so stateful backends (e.g.
//! the Qwen3.5 hybrid GDN recurrent-state map) can release per-sequence
//! state. Pre-fix, `gdn_states` grew one entry per finished request for the
//! engine's lifetime (SeqIds are monotonic) — an unbounded memory leak.

use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use vllm_core::engine::Engine;
use vllm_core::types::Request;
use vllm_testing::StubModel;
use vllm_traits::{BatchOutput, ModelBackend, ModelError, SampledToken, SeqId, TokenId};

/// Delegating backend that records every `on_sequence_finished` call.
#[derive(Clone)]
struct RecordingModel {
    inner: StubModel,
    finished: Arc<Mutex<Vec<SeqId>>>,
}

impl RecordingModel {
    fn new() -> Self {
        Self {
            inner: StubModel::returning(7),
            finished: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn finished_ids(&self) -> Vec<SeqId> {
        self.finished.lock().unwrap().clone()
    }
}

impl ModelBackend for RecordingModel {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> Result<BatchOutput, ModelError> {
        self.inner.forward(
            seq_ids,
            input_tokens,
            positions,
            kv_block_ids,
            num_computed_tokens,
            is_prefill,
        )
    }

    fn forward_logits(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        self.inner.forward_logits(
            seq_ids,
            input_tokens,
            positions,
            kv_block_ids,
            num_computed_tokens,
            is_prefill,
        )
    }

    fn embed(
        &mut self,
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        self.inner.embed(input_tokens, positions)
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }

    fn num_heads(&self) -> usize {
        self.inner.num_heads()
    }

    fn on_sequence_finished(&mut self, seq_id: SeqId) {
        self.finished.lock().unwrap().push(seq_id);
    }
}

#[test]
fn engine_notifies_model_when_sequence_finishes_by_max_tokens() {
    let model = RecordingModel::new();
    let finished = model.finished.clone();
    let mut engine = Engine::new_boxed(Box::new(model), None::<Box<dyn ModelBackend>>);
    let (tx, _rx) = mpsc::channel(64);
    let seq_id = engine.add_request(Request::new(1, vec![1, 2, 3], 2), tx);

    // Run enough steps for prefill + max_tokens=2 decode tokens.
    for _ in 0..10 {
        engine.step().unwrap();
        if finished.lock().unwrap().contains(&seq_id) {
            break;
        }
    }

    assert!(
        finished.lock().unwrap().contains(&seq_id),
        "engine must notify the model when the sequence finishes (RIL ISS-034)"
    );
}

#[test]
fn engine_notifies_model_when_sequence_is_cancelled() {
    let model = RecordingModel::new();
    let finished = model.finished.clone();
    let mut engine = Engine::new_boxed(Box::new(model), None::<Box<dyn ModelBackend>>);
    let (tx, _rx) = mpsc::channel(64);
    let seq_id = engine.add_request(Request::new(1, vec![1, 2, 3], 50), tx);

    engine.step().unwrap(); // prefill -> running
    assert!(
        engine.cancel_request(seq_id),
        "cancel_request must find the sequence"
    );
    assert!(
        finished.lock().unwrap().contains(&seq_id),
        "engine must notify the model on cancel (RIL ISS-034)"
    );
}
