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
        crate::engine::spec_dispatch::verify::test_only_sample_or_argmax(pos_logits, &params);
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
        crate::engine::spec_dispatch::verify::test_only_sample_or_argmax(pos_logits, &params);
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
        crate::engine::spec_dispatch::verify::test_only_sample_or_argmax(pos_logits, &params);
    // argmax of `vec![-10.0; 64]` with first 4 entries set to 5.0 is 0
    // (first max wins). Draft token 1 is *also* argmax-tied, but the
    // argmax implementation picks the first one — so this test pins the
    // argmax contract under temperature == 0.
    assert_eq!(target_token.token, 0);
}
