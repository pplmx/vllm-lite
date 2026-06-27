//! Tests for the speculative decoding dispatch path.

use super::super::Engine;
use crate::types::{Request, SchedulerConfig};
use tokio::sync::mpsc as tokio_mpsc;
use vllm_traits::{BatchOutput, ModelBackend, Result as ModelResult, SeqId, TokenId};

/// A fake model that returns fixed tokens for both forward and forward_logits.
#[derive(Clone)]
struct FakeModel {
    token_to_return: TokenId,
    vocab_size: usize,
}

impl FakeModel {
    fn new(token: TokenId) -> Self {
        Self {
            token_to_return: token,
            vocab_size: 100,
        }
    }

    fn logits_for_token(&self, token: TokenId) -> Vec<f32> {
        let mut logits = vec![-10.0; self.vocab_size];
        if (token as usize) < self.vocab_size {
            logits[token as usize] = 10.0;
        }
        logits
    }
}

impl ModelBackend for FakeModel {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> ModelResult<BatchOutput> {
        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens: seq_ids.iter().map(|_| self.token_to_return).collect(),
        })
    }

    fn forward_logits(
        &mut self,
        _seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> ModelResult<Vec<Vec<f32>>> {
        Ok(input_tokens
            .iter()
            .map(|tokens| {
                tokens
                    .iter()
                    .flat_map(|_| self.logits_for_token(self.token_to_return))
                    .collect()
            })
            .collect())
    }

    fn embed(
        &mut self,
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> ModelResult<Vec<Vec<f32>>> {
        Ok(input_tokens
            .iter()
            .map(|tokens| vec![0.0; tokens.len()])
            .collect())
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn num_layers(&self) -> usize {
        1
    }

    fn num_heads(&self) -> usize {
        1
    }
}

/// Wrapper around FakeModel that counts forward/forward_logits invocations.
/// Used to verify warmup_draft_kv calls draft model per sequence.
/// `Arc<AtomicUsize>` + Clone enable inspecting call count after the model
/// has been moved into the engine (the engine clones the Arc internally).
#[derive(Clone)]
struct CounterModel {
    inner: FakeModel,
    forward_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl CounterModel {
    fn new(token: TokenId) -> Self {
        Self {
            inner: FakeModel::new(token),
            forward_count: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }
    fn forward_count(&self) -> usize {
        self.forward_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl ModelBackend for CounterModel {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> ModelResult<BatchOutput> {
        self.forward_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
    ) -> ModelResult<Vec<Vec<f32>>> {
        self.forward_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
    ) -> ModelResult<Vec<Vec<f32>>> {
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
}

/// Test Plan 17.4-A: warmup_draft_kv invokes draft model once per sequence.
/// Fast unit test (no #[ignore]): directly constructs a Prefill batch and
/// calls warmup_draft_kv to verify the contract independently of step().
#[test]
fn test_warmup_draft_kv_invokes_draft_per_sequence() {
    let target = FakeModel::new(42);
    let draft = CounterModel::new(42);
    let draft_count_before = draft.forward_count();
    let mut engine = Engine::new_boxed(Box::new(target), Some(Box::new(draft.clone())));
    engine.enable_speculative();

    let batch = vllm_traits::types::Batch {
        seq_ids: vec![1, 2, 3],
        input_tokens: vec![vec![10, 20], vec![30], vec![40, 50, 60]],
        positions: vec![vec![0, 1], vec![0], vec![0, 1, 2]],
        kv_block_ids: vec![vec![0], vec![0], vec![0]],
        num_computed_tokens: vec![0, 0, 0],
        is_prefill: vec![true, true, true],
        phase: vllm_traits::BatchPhase::Prefill,
        total_tokens: 6,
        max_seq_len: 3,
    };

    engine
        .warmup_draft_kv(&batch)
        .expect("warmup_draft_kv should succeed");

    let calls = draft.forward_count() - draft_count_before;
    assert_eq!(
        calls, 3,
        "warmup_draft_kv should invoke draft.forward() exactly once per seq_id (got {})",
        calls
    );
}

/// Test Plan 17.1-A: Unified step() dispatches correctly
#[test]
#[ignore]
fn test_step_unified_dispatch() {
    let target = FakeModel::new(42);
    let draft = FakeModel::new(42);
    let mut engine = Engine::new_boxed(Box::new(target), Some(Box::new(draft)));
    engine.enable_speculative();
    let (tx, _rx) = tokio_mpsc::channel(64);
    engine.add_request(Request::new(1, vec![10, 20], 5), tx);

    let result = engine.step().unwrap();
    assert!(!result.is_empty());

    engine.scheduler = super::super::super::scheduler::engine::SchedulerEngine::new(
        SchedulerConfig::default(),
        1024,
        std::sync::Arc::new(crate::metrics::EnhancedMetricsCollector::new()),
    );
    let (tx2, _rx2) = tokio_mpsc::channel(64);
    engine.add_request(Request::new(2, vec![10, 20], 5), tx2);
    engine.enable_speculative();
    let result = engine.step().unwrap();
    assert!(!result.is_empty());
}

/// Test Plan 17.1-B: Batched draft generation produces expected output shape
#[test]
#[ignore]
fn test_batched_draft_generation() {
    let target = FakeModel::new(42);
    let draft = FakeModel::new(42);
    let mut engine = Engine::new_boxed(Box::new(target), Some(Box::new(draft)));
    engine.max_draft_tokens = 4;
    engine.enable_speculative();

    let (tx, _rx) = tokio_mpsc::channel(64);
    engine.add_request(Request::new(1, vec![10, 20], 10), tx);
    let result = engine.step().unwrap();
    assert!(!result.is_empty());
}

/// Test Plan 17.1-C: Greedy-mode exact match via argmax verification
#[test]
#[ignore]
fn test_logit_verification_exact_match() {
    let target = FakeModel::new(42);
    let draft = FakeModel::new(42);
    let mut engine = Engine::new_boxed(Box::new(target), Some(Box::new(draft)));
    engine.max_draft_tokens = 3;
    engine.enable_speculative();

    let (tx, mut rx) = tokio_mpsc::channel(64);
    engine.add_request(Request::new(1, vec![10, 20], 10), tx);
    let result = engine.step().unwrap();
    assert!(!result.is_empty());
    assert_eq!(result[0].1, 42);
    let _ = rx.try_recv().ok();
}

/// Test Plan 17.1-D: KV cache rollback for rejected drafts
#[test]
#[ignore]
fn test_kv_rollback_rejected_drafts() {
    let target = FakeModel::new(42);
    let draft = FakeModel::new(99);
    let mut engine = Engine::new_boxed(Box::new(target), Some(Box::new(draft)));
    engine.max_draft_tokens = 3;
    engine.enable_speculative();

    let (tx, _rx) = tokio_mpsc::channel(64);
    engine.add_request(Request::new(1, vec![10, 20], 5), tx);
    let result = engine.step().unwrap();
    assert!(!result.is_empty());
    assert_eq!(result[0].1, 42);
}

/// Test Plan 17.1-E: Multi-token input_count is accepted by scheduler
#[test]
fn test_scheduler_multi_token_update() {
    use std::sync::Arc;
    let mut scheduler = super::super::super::scheduler::engine::SchedulerEngine::new(
        SchedulerConfig::default(),
        1024,
        Arc::new(crate::metrics::EnhancedMetricsCollector::new()),
    );
    let id = scheduler.add_request(Request::new(1, vec![10, 20], 10));
    let _batch = scheduler.build_batch();

    scheduler.update(&[id], &[100], &[3]);
    assert_eq!(scheduler.running_count(), 1);

    scheduler.update(&[id], &[101], &[0]);
    assert_eq!(scheduler.running_count(), 1);
}

/// Test Plan 17.1-F: Speculative fallback on draft error
#[test]
#[ignore]
fn test_draft_model_error_fallback() {
    let target = FakeModel::new(42);
    let mut engine = Engine::new_boxed(Box::new(target), None::<Box<dyn ModelBackend>>);
    engine.speculative_mode = true;

    let (tx, _rx) = tokio_mpsc::channel(64);
    engine.add_request(Request::new(1, vec![10, 20], 5), tx);
    let result = engine.step();
    assert!(result.is_ok());
}

/// Integration test: speculative step produces output
#[test]
#[ignore]
fn test_speculative_step_produces_output() {
    let target = FakeModel::new(42);
    let draft = FakeModel::new(42);
    let mut engine = Engine::new_boxed(Box::new(target), Some(Box::new(draft)));
    engine.max_draft_tokens = 4;
    engine.enable_speculative();

    let (tx, mut rx) = tokio_mpsc::channel(64);
    engine.add_request(Request::new(1, vec![10, 20], 10), tx);
    let result = engine.step().unwrap();
    assert!(!result.is_empty());
    assert_eq!(result[0].1, 42);

    let received = rx.try_recv().ok();
    assert_eq!(received, Some(42));
}

/// Integration test: speculative vs non-speculative equivalence
#[test]
#[ignore]
fn test_speculative_vs_non_speculative_equivalence() {
    let target = FakeModel::new(42);
    let draft = FakeModel::new(42);

    let mut engine_ns = Engine::new_boxed(Box::new(target.clone()), None);
    let (tx1, _rx1) = tokio_mpsc::channel(64);
    engine_ns.add_request(Request::new(1, vec![10, 20], 5), tx1);
    let result_ns = engine_ns.step().unwrap();

    let mut engine_sp = Engine::new_boxed(Box::new(target), Some(Box::new(draft)));
    engine_sp.enable_speculative();
    engine_sp.max_draft_tokens = 3;
    let (tx2, _rx2) = tokio_mpsc::channel(64);
    engine_sp.add_request(Request::new(2, vec![10, 20], 5), tx2);
    let result_sp = engine_sp.step().unwrap();

    assert!(!result_ns.is_empty());
    assert!(!result_sp.is_empty());
    assert_eq!(result_ns[0].1, result_sp[0].1);
}
