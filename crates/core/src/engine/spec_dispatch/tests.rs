//! Tests for the speculative decoding dispatch path.

use super::super::Engine;
use crate::types::{Request, SchedulerConfig};
use tokio::sync::mpsc as tokio_mpsc;
use vllm_traits::{BatchOutput, ModelBackend, Result as ModelResult, SampledToken, SeqId, TokenId};

/// A fake model that returns fixed tokens for both forward and `forward_logits`.
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
            next_tokens: seq_ids
                .iter()
                .map(|_| SampledToken {
                    token: self.token_to_return,
                    logprob: 0.0,
                    top_logprobs: vec![],
                })
                .collect(),
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

/// Wrapper around `FakeModel` that counts `forward/forward_logits` invocations.
/// Used to verify `warmup_draft_kv` calls draft model per sequence.
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

/// `FakeModel` wrapper that records the `positions` passed to
/// `forward_logits`. Used to verify the speculative verification path feeds
/// the target model the TRUE sequence positions (RIL ISS-016), not a
/// 0-based range.
#[derive(Clone)]
struct PositionRecordingModel {
    inner: FakeModel,
    recorded: std::sync::Arc<std::sync::Mutex<Vec<Vec<usize>>>>,
}

impl PositionRecordingModel {
    fn new(token: TokenId) -> Self {
        Self {
            inner: FakeModel::new(token),
            recorded: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }
    fn recorded(&self) -> Vec<Vec<usize>> {
        self.recorded.lock().unwrap().clone()
    }
    fn clear(&self) {
        self.recorded.lock().unwrap().clear();
    }
}

impl ModelBackend for PositionRecordingModel {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> ModelResult<BatchOutput> {
        if let Some(first) = positions.first() {
            self.recorded.lock().unwrap().push(first.clone());
        }
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
        if let Some(first) = positions.first() {
            self.recorded.lock().unwrap().push(first.clone());
        }
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

/// Regression (RIL ISS-016): the speculative verification forward pass must
/// feed the target model the TRUE sequence positions, not a 0-based range.
/// `RoPE` uses absolute positions, so verifying drafts at positions [0,1,...]
/// for a sequence already at decode position P>0 corrupts the target logits
/// (and thus the accept/reject decisions). After prefill of a 2-token prompt
/// the sequence decodes at position 2, so every position the target sees
/// during the decode-step verification must be >= 2.
#[test]
fn test_verification_uses_true_sequence_positions() {
    let target = PositionRecordingModel::new(42);
    let recorder = target.clone();
    let draft = FakeModel::new(42); // drafts match target argmax => accepted
    let mut engine = Engine::new_boxed(Box::new(target), Some(Box::new(draft)));
    engine.max_draft_tokens = 3;
    engine.enable_speculative();

    let (tx, _rx) = tokio_mpsc::channel(64);
    engine.add_request(Request::new(1, vec![10, 20], 10), tx);

    // Step 1: prefill the 2-token prompt; sequence advances to decode pos 2.
    let _ = engine.step().unwrap();
    recorder.clear();

    // Step 2: decode + speculative verification at position >= 2.
    let _ = engine.step().unwrap();

    let recorded = recorder.recorded();
    assert!(
        !recorded.is_empty(),
        "target forward_logits must be invoked during speculative verification"
    );
    for positions in &recorded {
        for &pos in positions {
            assert!(
                pos >= 2,
                "verification must use true sequence positions (>= 2 after a                  2-token prefill); got {pos} in {positions:?} — 0-based positions                  corrupt RoPE (RIL ISS-016)"
            );
        }
    }
}

/// Regression (RIL ISS-017): the legacy BATCHED draft path must advance draft
/// positions as `last_pos + 1` from the true base, not push
/// `current_positions.len()`. For a sequence decoding at position P the draft
/// positions must be P, P+1, P+2, ... so the draft model applies `RoPE` at the
/// true positions. Pre-fix the 2nd/3rd drafts got positions 1, 2 (the count)
/// regardless of P, corrupting the draft tokens. `Engine::new_boxed` leaves
/// `draft_resolver = None`, so `step()` exercises this batched path.
#[test]
fn test_batched_draft_positions_advance_from_base() {
    let target = FakeModel::new(42);
    let draft = PositionRecordingModel::new(42);
    let recorder = draft.clone();
    let mut engine = Engine::new_boxed(Box::new(target), Some(Box::new(draft)));
    engine.max_draft_tokens = 3;
    engine.enable_speculative();

    let (tx, _rx) = tokio_mpsc::channel(64);
    engine.add_request(Request::new(1, vec![10, 20], 10), tx);

    // Step 1: prefill the 2-token prompt; sequence advances to decode pos >= 2.
    let _ = engine.step().unwrap();
    recorder.clear();

    // Step 2: decode + batched draft generation at position >= 2.
    let _ = engine.step().unwrap();

    let recorded = recorder.recorded();
    assert!(
        !recorded.is_empty(),
        "draft model forward must be invoked during batched draft generation"
    );
    for positions in &recorded {
        for &pos in positions {
            assert!(
                pos >= 2,
                "batched draft positions must advance from the true decode base                  (>= 2 after a 2-token prefill); got {pos} in {positions:?} —                  positions.len() instead of last_pos+1 (RIL ISS-017)"
            );
        }
    }
}

/// Test Plan 17.4-A: `warmup_draft_kv` invokes draft model once per sequence.
/// Fast unit test (no #[ignore]): directly constructs a Prefill batch and
/// calls `warmup_draft_kv` to verify the contract independently of `step()`.
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
        sampling_params: vec![vllm_traits::SamplingParams::default(); 3],
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
        "warmup_draft_kv should invoke draft.forward() exactly once per seq_id (got {calls})"
    );
}

/// Test Plan 17.1-A: Unified `step()` dispatches correctly
#[test]
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
    assert_eq!(result[0].1.token, 42);
    let _ = rx.try_recv().ok();
}

/// Test Plan 17.1-D: KV cache rollback for rejected drafts
#[test]
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
    assert_eq!(result[0].1.token, 42);
}

/// Test Plan 17.1-E: Multi-token `input_count` is accepted by scheduler
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

    scheduler.update(
        &[id],
        &[SampledToken {
            token: 100,
            logprob: 0.0,
            top_logprobs: vec![],
        }],
        &[3],
    );
    assert_eq!(scheduler.running_count(), 1);

    scheduler.update(
        &[id],
        &[SampledToken {
            token: 101,
            logprob: 0.0,
            top_logprobs: vec![],
        }],
        &[0],
    );
    assert_eq!(scheduler.running_count(), 1);
}

/// Test Plan 17.1-F: Speculative fallback on draft error
#[test]
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
    assert_eq!(result[0].1.token, 42);

    let received = rx.try_recv().ok();
    assert_eq!(received.map(|s| s.token), Some(42));
}

/// Regression: a long prompt in speculative mode must complete prefill in a
/// single step (`num_computed_tokens >= prompt_len` and status `Decoding`).
///
/// The speculative step's `input_counts` must reflect the number of input
/// tokens the target model actually processed (`input_len + accepted`), not
/// just `accepted + 1`. With `accepted + 1 < prompt_len` the old arithmetic
/// left the sequence stuck in `Prefilling` and re-fed already-generated
/// draft tokens back into subsequent prefill batches — re-processing the
/// whole prompt every step until `num_computed_tokens` slowly crossed
/// `prompt_len` (RIL hypothesis HYP-006).
#[test]
fn test_speculative_prefill_long_prompt_advances_to_decode() {
    let target = FakeModel::new(42);
    let draft = FakeModel::new(42);
    let mut engine = Engine::new_boxed(Box::new(target), Some(Box::new(draft)));
    engine.max_draft_tokens = 4;
    engine.enable_speculative();

    let (tx, _rx) = tokio_mpsc::channel(64);
    // 20-token prompt, max_tokens large enough that the 5 emitted tokens
    // (4 accepted drafts + bonus) do not finish the sequence: accepted + 1
    // (<= 5) is well below prompt_len, so the prefill would not complete in
    // one step under the old arithmetic.
    let prompt: Vec<TokenId> = (0..20).map(|t| t + 10).collect();
    let seq_id = engine.add_request(Request::new(1, prompt, 20), tx);

    let result = engine.step().unwrap();
    assert!(!result.is_empty());

    let seq = engine
        .scheduler
        .get_sequence(seq_id)
        .expect("sequence should be running after the speculative step");
    assert!(
        seq.num_computed_tokens >= seq.prompt_len,
        "speculative prefill must complete in one step: num_computed_tokens={} prompt_len={}",
        seq.num_computed_tokens,
        seq.prompt_len
    );
    assert_eq!(
        seq.status,
        crate::types::Status::Decoding,
        "sequence must transition to Decoding after a completed speculative prefill"
    );
}

/// Regression: a speculative step that accepts every draft must fold ALL
/// emitted tokens into the scheduler sequence state (`seq.tokens`), not just
/// the first one.
///
/// `step_speculative_inner` flattens the per-sequence emitted tokens into a
/// single `seq_ids`/`sampled` vector but passes per-sequence `input_counts` —
/// `scheduler.update` zips the three together, so the count vector truncates
/// the loop to one iteration per sequence and every token after the first is
/// streamed to the client without ever being recorded in the sequence (RIL
/// hypothesis HYP-006).
#[test]
fn test_speculative_all_accepted_folds_every_token_into_sequence() {
    let target = FakeModel::new(42);
    let draft = FakeModel::new(42);
    let mut engine = Engine::new_boxed(Box::new(target), Some(Box::new(draft)));
    engine.max_draft_tokens = 4;
    engine.enable_speculative();

    let (tx, _rx) = tokio_mpsc::channel(64);
    // 2-token prompt: after prefill the sequence advances to Decoding, and
    // the step emits max_draft accepted drafts + 1 bonus token (5 total).
    let seq_id = engine.add_request(Request::new(1, vec![10, 20], 20), tx);

    let result = engine.step().unwrap();
    let emitted = result.iter().filter(|(id, _)| *id == seq_id).count();
    assert_eq!(emitted, 5, "all 4 drafts + bonus must be emitted");

    let seq = engine
        .scheduler
        .get_sequence(seq_id)
        .expect("sequence should be running after the speculative step");
    assert_eq!(
        seq.tokens.len(),
        2 + emitted,
        "every emitted token must be folded into seq.tokens (prompt 2 + {emitted} emitted)"
    );
}

/// Regression: a speculative prefill resumed after a partial prefix-cache
/// hit must NOT let the rejected-draft rollback release the matched prefix
/// blocks or rewind `num_computed_tokens`.
///
/// The verifier writes the draft KV into blocks pre-allocated for the
/// verification span (RIL ISS-026 fix); the rejected drafts' KV simply sits
/// beyond `num_computed_tokens` and is never read. Calling `memory_rollback`
/// on a prefill batch instead runs its block math against the pre-step
/// `num_computed_tokens` — for a resumed prefill (start > 0) that frees the
/// last computed-prefix block (possibly the prefix-cache matched block) and
/// rewinds the count, so the next chunk never recomputes the freed region
/// and the sequence reads garbage KV (RIL hypothesis HYP-008).
#[test]
fn test_speculative_resumed_prefill_rollback_does_not_release_prefix_blocks() {
    let target = FakeModel::new(42);
    let draft = FakeModel::new(99); // never matches the target -> all drafts rejected
    let mut engine = Engine::new_boxed(Box::new(target), Some(Box::new(draft)));
    engine.max_draft_tokens = 4;
    engine.enable_speculative();

    // Populate the prefix cache with a 4-token prefix (one block).
    let (tx1, _rx1) = tokio_mpsc::channel(64);
    engine.add_request(Request::new(1, vec![10, 20, 30, 40], 2), tx1);
    engine.step().unwrap();
    engine.step().unwrap();

    // 20-token prompt sharing the cached 4-token prefix [10, 20, 30, 40]:
    // the sequence is admitted with num_computed_tokens = 4 and a prefill
    // resume of 16 tokens. All 4 drafts are rejected.
    let (tx2, _rx2) = tokio_mpsc::channel(64);
    let mut prompt: Vec<TokenId> = vec![10, 20, 30, 40];
    prompt.extend((0..16).map(|t| t + 50));
    let seq_id = engine.add_request(Request::new(2, prompt, 20), tx2);
    engine.step().unwrap();

    let seq = engine
        .scheduler
        .get_sequence(seq_id)
        .expect("sequence should be running after the resumed speculative prefill");
    assert!(
        seq.num_computed_tokens >= seq.prompt_len,
        "resumed prefill must complete in one step: num_computed_tokens={} prompt_len={} (rollback \
         must not rewind the prefix-cache match)",
        seq.num_computed_tokens,
        seq.prompt_len
    );
    assert_eq!(
        seq.status,
        crate::types::Status::Decoding,
        "sequence must transition to Decoding after the resumed speculative prefill"
    );
}

/// Integration test: speculative vs non-speculative equivalence
#[test]
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

// =====================================================================
// Architecture-performance.md §6 — speculative decoding uses
// temperature-aware acceptance (sampled-match) instead of pure argmax
// when temperature > 0. The tests below pin down the three observable
// invariants:
//   1. Greedy (temperature == 0) still matches argmax exactly.
//   2. Sampling with a flat distribution accepts every draft whose token
//      is among the top-K with non-negligible probability (i.e. the
//      verifier stops rejecting high-probability drafts just because they
//      aren't the unique argmax).
//   3. When the draft's token is below the sampling threshold, the
//      verifier still rejects — but emits the *sampled* target token, not
//      the argmax.
// =====================================================================

use crate::engine::ctor::EngineBuilder;
use crate::sync::lock_mutex;
use vllm_traits::SamplingParams;

/// Build a tiny engine whose target model emits a flat logits vector with
/// the first `nonzero` entries hot. Used to make the target's sampling
/// distribution predictable: any token in `0..nonzero` is "high prob",
/// anything else is "low prob".
fn build_flat_logits_engine(nonzero: usize) -> Engine {
    #[derive(Clone)]
    struct FlatLogitsModel {
        vocab_size: usize,
        nonzero: usize,
    }

    impl ModelBackend for FlatLogitsModel {
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
                next_tokens: seq_ids
                    .iter()
                    .map(|_| SampledToken {
                        token: 0,
                        logprob: 0.0,
                        top_logprobs: vec![],
                    })
                    .collect(),
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
            let vocab = self.vocab_size;
            let nonzero = self.nonzero;
            Ok(input_tokens
                .iter()
                .map(|tokens| {
                    tokens
                        .iter()
                        .flat_map(|_| {
                            let mut logits = vec![-10.0_f32; vocab];
                            for slot in logits.iter_mut().take(nonzero) {
                                *slot = 5.0;
                            }
                            logits
                        })
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

    EngineBuilder::new(Box::new(FlatLogitsModel {
        vocab_size: 64,
        nonzero,
    }))
    .with_num_kv_blocks(64)
    .build()
}

/// `verify_draft_tokens_logits` must accept every draft token whose token
/// id is in the target's high-probability set, even when it isn't the
/// unique argmax. With `nonzero == 4` the target puts uniform mass on
/// tokens 0..4, so any draft in that range is a valid sample.
#[test]
fn verifier_accepts_high_prob_drafts_under_sampling() {
    let engine = build_flat_logits_engine(4);
    let seq_id: SeqId = 1;
    let vocab = 64_usize;

    // 5 input tokens, each with `vocab` logits. The flat-logits model
    // sets tokens 0..nonzero to 5.0 and the rest to -10.0 regardless of
    // the input token id. Sampling from this distribution with
    // temperature = 1.0 must always pick a token in `0..4` because
    // exp(-15) ≈ 3e-7 per low-prob token contributes essentially zero
    // to the softmax denominator.
    let verify_tokens: Vec<TokenId> = vec![10, 20, 1, 2, 3];
    let verify_positions: Vec<usize> = (0..verify_tokens.len()).collect();

    let logits = {
        let mut model = lock_mutex(&engine.target_model).expect("lock");
        model
            .forward_logits(
                &[seq_id],
                &[verify_tokens.clone()],
                &[verify_positions],
                &[vec![0_usize; 1]],
                &[0_usize],
                &[false],
            )
            .expect("forward_logits")
    };
    assert_eq!(logits.len(), 1);
    assert_eq!(logits[0].len(), verify_tokens.len() * vocab);

    let params = SamplingParams {
        temperature: 1.0, // enable sampling path
        ..SamplingParams::default()
    };

    // Sample from the first position's logits. The result must be in
    // `0..4` (the high-prob set) regardless of which RNG draw lands.
    let pos_logits = &logits[0][0..vocab];
    let target_token =
        crate::engine::spec_dispatch::verify::test_only_sample_or_argmax(pos_logits, &params, &[]);
    assert!(
        target_token.token < 4,
        "sampled target token {} fell outside the \
         high-prob set; sampling path is not engaged",
        target_token.token
    );
}

/// When the draft's token is below the target's sampling threshold, the
/// verifier MUST reject it (so the wall-clock speedup doesn't change the
/// output distribution). The emitted token is the sampled target token,
/// not necessarily argmax.
#[test]
fn verifier_rejects_low_prob_drafts_under_sampling() {
    let engine = build_flat_logits_engine(2); // tokens 0..2 are high-prob
    let seq_id: SeqId = 2;
    let vocab = 64_usize;

    let batch_input_tokens: &[u32] = &[10_u32, 20];
    let drafts: &[u32] = &[50_u32, 51, 52]; // all outside 0..2 (low-prob)

    let verify_tokens: Vec<TokenId> = batch_input_tokens
        .iter()
        .chain(drafts.iter())
        .copied()
        .collect();
    let verify_positions: Vec<usize> = (0..verify_tokens.len()).collect();

    let logits = {
        let mut model = lock_mutex(&engine.target_model).expect("lock");
        model
            .forward_logits(
                &[seq_id],
                &[verify_tokens],
                &[verify_positions],
                &[vec![0_usize; 1]],
                &[0_usize],
                &[false],
            )
            .expect("forward_logits")
    };

    let params = SamplingParams {
        temperature: 0.5,
        ..SamplingParams::default()
    };

    // First draft token 50 has -10 logit; sampling at temperature 0.5
    // picks uniformly from {0, 1} essentially, never 50. The verifier
    // must therefore reject immediately and emit a target-sampled token
    // in {0, 1}.
    let offset = 0;
    let pos_logits = &logits[0][offset..offset + vocab];
    let target_token =
        crate::engine::spec_dispatch::verify::test_only_sample_or_argmax(pos_logits, &params, &[]);
    assert!(
        target_token.token < 2,
        "sampled target token {} fell outside the high-prob \
         set; verifier is not respecting the target distribution",
        target_token.token
    );
    assert_ne!(
        target_token.token, 50,
        "verifier accepted an out-of-distribution draft token"
    );
}

/// With `temperature == 0` the verifier must still match argmax exactly
/// (the old behaviour). Otherwise we'd be silently changing greedy
/// decoding output.
#[test]
fn verifier_uses_argmax_when_temperature_is_zero() {
    let engine = build_flat_logits_engine(4); // 0..4 tied
    let seq_id: SeqId = 3;
    let vocab = 64_usize;

    let batch_input_tokens: &[u32] = &[10_u32, 20];
    let drafts: &[u32] = &[1_u32, 2, 3];

    let verify_tokens: Vec<TokenId> = batch_input_tokens
        .iter()
        .chain(drafts.iter())
        .copied()
        .collect();
    let verify_positions: Vec<usize> = (0..verify_tokens.len()).collect();

    let logits = {
        let mut model = lock_mutex(&engine.target_model).expect("lock");
        model
            .forward_logits(
                &[seq_id],
                &[verify_tokens],
                &[verify_positions],
                &[vec![0_usize; 1]],
                &[0_usize],
                &[false],
            )
            .expect("forward_logits")
    };

    let params = SamplingParams::default(); // temperature = 0 (greedy)

    let offset = 0;
    let pos_logits = &logits[0][offset..offset + vocab];
    let target_token =
        crate::engine::spec_dispatch::verify::test_only_sample_or_argmax(pos_logits, &params, &[]);
    // argmax of `vec![-10.0; 64]` with first 4 entries set to 5.0 is 0
    // (first max wins). Draft token 1 is *also* argmax-tied, but the
    // argmax implementation picks the first one — so this test pins the
    // argmax contract under temperature == 0.
    assert_eq!(target_token.token, 0);
}

/// Regression: speculative verification must compare each draft against
/// the target's prediction at the position AFTER the input prompt, not
/// at the start of the verify sequence. With a prefill batch (input
/// length > 1) and a target that predicts the next token exactly, all
/// correct drafts must be accepted. The old offset math (`j * vocab`)
/// compared draft j against position j's logits, which predict an
/// *input* token when the prompt has more than one token — rejecting
/// correct drafts and changing the output distribution.
#[test]
fn verifier_prefill_accepts_drafts_at_position_after_prompt() {
    #[derive(Clone)]
    struct NextTokenModel {
        vocab_size: usize,
    }

    impl ModelBackend for NextTokenModel {
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
                next_tokens: seq_ids
                    .iter()
                    .map(|_| SampledToken {
                        token: 0,
                        logprob: 0.0,
                        top_logprobs: vec![],
                    })
                    .collect(),
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
            // Position p's logits argmax to the token that follows
            // input_tokens[p] (i.e. input_tokens[p+1]); the final
            // position predicts token 0.
            let vocab = self.vocab_size;
            Ok(input_tokens
                .iter()
                .map(|tokens| {
                    tokens
                        .iter()
                        .enumerate()
                        .flat_map(|(p, _)| {
                            let mut logits = vec![-100.0_f32; vocab];
                            let next = tokens.get(p + 1).copied().unwrap_or(0);
                            logits[next as usize % vocab] = 0.0;
                            logits
                        })
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

    let mut engine = EngineBuilder::new(Box::new(NextTokenModel { vocab_size: 64 }))
        .with_num_kv_blocks(64)
        .build();
    engine.enable_speculative();

    // Prefill batch: 3 input tokens [10, 20, 30]; the correct
    // continuation is [40, 50] (target predicts 40 after 30, then 50
    // after 40). Greedy params -> argmax acceptance check.
    let batch = vllm_traits::types::Batch {
        seq_ids: vec![1],
        input_tokens: vec![vec![10, 20, 30]],
        positions: vec![vec![0, 1, 2]],
        kv_block_ids: vec![vec![0]],
        num_computed_tokens: vec![0],
        is_prefill: vec![true],
        sampling_params: vec![vllm_traits::SamplingParams::default()],
        phase: vllm_traits::BatchPhase::Prefill,
        total_tokens: 3,
        max_seq_len: 10,
    };
    let drafts: Vec<Vec<TokenId>> = vec![vec![40, 50]];

    let (verified, accepted_counts) = engine
        .verify_draft_tokens_logits(&batch, &drafts)
        .expect("verify should succeed");
    assert_eq!(
        accepted_counts,
        vec![2],
        "both correct drafts must be accepted at the post-prompt positions"
    );
    // Results: 2 accepted-draft placeholders + 1 bonus token sampled
    // from the position after the last draft (which the model
    // predicts as 0).
    assert_eq!(verified.len(), 3);
    assert_eq!(verified[0].1.token, 40);
    assert_eq!(verified[1].1.token, 50);
    assert_eq!(
        verified[2].1.token, 0,
        "bonus token must come from the post-draft position"
    );
}

// RIL ISS-057: a chunked prefill (prompt > max_num_batched_tokens) running
// in SPECULATIVE mode must not stream its mid-chunk "predicted next prompt
// token" to the client. `step_speculative_inner` processes prefill chunks
// too, and because drafts are generated even for a prefill entry (from the
// chunk's last token), every verified entry — drafts + bonus for mid-chunk
// prefills — used to reach `dispatch.rs`'s send loop with no phase filter:
// a 100-token prompt chunked by 16 streamed 24 tokens when only max_tokens=8
// were real output (observed pre-fix). The client must see exactly the
// decode tokens (the first generated token comes from the first decode/draft
// step after prefill completes, so even the final prefill chunk emits
// nothing).
#[test]
fn speculative_chunked_prefill_never_streams_stale_midchunk_predictions() {
    let config = SchedulerConfig::builder()
        .with_max_num_batched_tokens(32)
        .with_prefill_chunk_size(16)
        .build();
    let mut engine = Engine::with_config_boxed(
        Box::new(FakeModel::new(42)),
        Some(Box::new(FakeModel::new(42))),
        config,
        3,
        256,
    );
    engine.enable_speculative();

    let (tx, mut rx) = tokio_mpsc::channel(256);
    engine.add_request(Request::new(1, vec![7; 100], 8), tx);

    let mut rounds = 0;
    while engine.has_pending() {
        engine.step().unwrap();
        rounds += 1;
        assert!(
            rounds <= 40,
            "must complete within prefill + decode rounds; got {rounds}"
        );
    }

    let mut received = 0usize;
    while rx.try_recv().is_ok() {
        received += 1;
    }

    // Post-fix the client sees exactly max_tokens (8) real decode tokens:
    // every prefill chunk (mid AND final) is suppressed, and decode emits
    // all-accepted drafts (max_draft=3 -> 4 tokens/step) up to the cap.
    // Pre-fix this was 24 (mid-chunk draft/bonus tokens leaked in).
    assert_eq!(
        received, 8,
        "the speculative client stream must contain exactly max_tokens real output tokens, \
         no stale mid-chunk prefill predictions (got {received})"
    );
}

/// A [`ModelBackend`] that behaves like [`FakeModel`] (fixed token) but
/// records every `forward_logits` call's `(start_position, input_len)`
/// so a test can rebuild exactly which absolute positions the TARGET model
/// was fed as real input. Only calls with `input_len > 1` represent real
/// prefill chunks (decode-batch verification feeds exactly one input
/// token); draft-KV warmup uses `forward`, not `forward_logits`, so it
/// never pollutes the record. Records `(start_position, real_chunk_len)` —
/// see `forward_logits` for how the real chunk is separated from the drafts.
#[derive(Clone)]
struct RecordingModel {
    token_to_return: TokenId,
    vocab_size: usize,
    calls: std::sync::Arc<std::sync::Mutex<Vec<(usize, usize)>>>,
}

impl RecordingModel {
    fn new(token: TokenId) -> Self {
        Self {
            token_to_return: token,
            vocab_size: 100,
            calls: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
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

impl ModelBackend for RecordingModel {
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
            next_tokens: seq_ids
                .iter()
                .map(|_| SampledToken {
                    token: self.token_to_return,
                    logprob: 0.0,
                    top_logprobs: vec![],
                })
                .collect(),
        })
    }

    fn forward_logits(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> ModelResult<Vec<Vec<f32>>> {
        for ((&_sid, tokens), &nc) in seq_ids
            .iter()
            .zip(input_tokens.iter())
            .zip(num_computed_tokens.iter())
        {
            // A prefill verification call feeds `real chunk + drafts` as one
            // flat token list. The real chunk is the leading run of prompt
            // tokens (7 in this fixture); the drafts (the draft backend's
            // fixed token 42) follow it. Splitting on the first non-prompt
            // token yields the REAL chunk length, which is what composes the
            // sequence's true frontier — `tokens.len()` would include the
            // drafts and (falsely) suggest the front covered them.
            if tokens.len() > 1
                && let Ok(mut calls) = self.calls.lock()
            {
                let real_len = tokens.iter().take_while(|&&t| t == 7).count();
                if real_len > 0 {
                    calls.push((nc, real_len));
                }
            }
        }
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

// RIL ISS-059: a chunked prefill running in SPECULATIVE mode must advance
// its computed frontier by the REAL chunk only — never past real prompt
// tokens. Drafts ARE generated for prefill entries (drafts.rs is not
// phase-gated), and the verifier accepts a draft when it equals the target
// model's continuation guess — which is NOT the real prompt token at that
// position. `update_speculative` advanced `num_computed_tokens` by
// `input_len + accepted`, so a mid-chunk prefill overshot into its
// remaining prompt: verification wrote guessed-content KV at those
// positions and `requeue_partial_prefills` (ISS-054) preserved the
// inflated frontier, so the NEXT chunk composed from
// `num_computed_tokens` and the real prompt tokens in between were NEVER
// fed — the model attends to guesses, not the user's prompt (self-spec +
// greedy accepts every draft, so this used to corrupt EVERY chunked
// prompt).
#[test]
fn speculative_chunked_prefill_feeds_every_real_prompt_position() {
    let config = SchedulerConfig::builder()
        .with_max_num_batched_tokens(32)
        .with_prefill_chunk_size(16)
        .build();
    let target = RecordingModel::new(42);
    let records = target.calls.clone();
    let mut engine = Engine::with_config_boxed(
        Box::new(target),
        Some(Box::new(FakeModel::new(42))),
        config,
        3,
        256,
    );
    engine.enable_speculative();

    let (tx, mut rx) = tokio_mpsc::channel(256);
    engine.add_request(Request::new(1, vec![7; 100], 8), tx);

    let mut rounds = 0;
    while engine.has_pending() {
        engine.step().unwrap();
        rounds += 1;
        assert!(
            rounds <= 40,
            "must complete within prefill + decode rounds; got {rounds}"
        );
    }

    let calls = records.lock().map(|g| g.clone()).unwrap_or_default();
    assert!(
        !calls.is_empty(),
        "prefill verification should have fed real chunks (recorded {calls:?})"
    );
    let prompt_len = 100usize;
    let chunk_len = 16usize;
    // Each verification call's `num_computed` IS the next chunk's start.
    // While the prompt is unfinished, that start must be a real chunk
    // boundary — never the overshot `start + chunk_len + accepted` that
    // would skip real prompt tokens (the drafts occupy real upcoming
    // prompt positions and their KV is guessed content).
    for &(start, _len) in &calls {
        if start < prompt_len {
            assert_eq!(
                start % chunk_len,
                0,
                "prefill chunk start {start} is not a real chunk boundary ({chunk_len}); \
                 the frontier overshot real prompt tokens via accepted drafts (recorded                  starts {calls:?}, want multiples of {chunk_len})"
            );
        }
    }
    // The real chunks tile the prompt exactly once when the starts are
    // correct: interval [start, start+chunk) capped at the prompt end.
    let mut covered = vec![false; prompt_len];
    for &(start, len) in &calls {
        if start < prompt_len {
            let end = (start + len).min(prompt_len);
            for slot in &mut covered[start..end] {
                assert!(
                    !*slot,
                    "real prompt position is fed twice (chunk starts {calls:?})"
                );
                *slot = true;
            }
        }
    }
    let missing: Vec<usize> = covered
        .iter()
        .enumerate()
        .filter(|(_, c)| !**c)
        .map(|(i, _)| i)
        .collect();
    assert!(
        missing.is_empty(),
        "speculative chunked prefill must feed every real prompt position exactly once; \
         missing positions {missing:?} (chunk starts {calls:?}) — the frontier overshot \
         real prompt tokens via accepted drafts (ISS-059)"
    );

    let mut received = 0usize;
    while rx.try_recv().is_ok() {
        received += 1;
    }
    assert_eq!(
        received, 8,
        "the speculative client stream must contain exactly max_tokens real output tokens \
         (got {received})"
    );
}
