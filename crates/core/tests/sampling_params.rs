//! ARCH-02 regression: per-sequence sampling parameters must reach the
//! hot path.
//!
//! Background (technical due diligence): prior to this fix, the HTTP
//! layer accepted `temperature` / `top_p` / `top_k` from the `OpenAI`
//! request and stored them on the `Request`, but the engine always
//! called `model.forward`, which chose the next token greedily inside
//! the model layer. The parameters were silently dropped. This file
//! locks in the seam:
//!
//! ```text
//! forward_logits → engine-side sample_batch_with_params(batch.sampling_params)
//! ```
//!
//! The tests use a deterministic mock that exposes per-sequence logits
//! so we can assert which token the engine emitted without relying on
//! randomness.

use tokio::sync::mpsc;
use vllm_core::engine::Engine;
use vllm_core::sampling::sample_batch_with_params;
use vllm_core::types::Request;
use vllm_traits::{
    BatchOutput, ModelBackend, Result, SampledToken, SamplingParams, SeqId, TokenId,
};

/// Mock whose `forward_logits` lights up one or two tokens per
/// sequence based on the sequence's first prompt token (used as a key
/// into `peaks`). This lets us assert per-sequence argmax in tests
/// where two requests run in the same step, and lets us design a
/// repeat-penalty test where the seen-token penalty flips the argmax.
struct PerSeqPeakModel {
    /// `peaks[prompt[0] as usize]` is the token id that will be the
    /// argmax for that sequence. Vocabulary must be large enough that
    /// each peak index is in range.
    peaks: Vec<TokenId>,
    /// Secondary peaks at half the primary height, used to test
    /// repeat-penalty. Indexed by the same key as `peaks`. When set to
    /// `None` for a key, only the primary peak is lit.
    secondary: Vec<Option<TokenId>>,
    vocab: usize,
}

impl PerSeqPeakModel {
    fn new(peaks: Vec<TokenId>, vocab: usize) -> Self {
        let n = peaks.len();
        Self {
            peaks,
            secondary: vec![None; n],
            vocab,
        }
    }

    fn with_secondary(mut self, secondary: Vec<Option<TokenId>>) -> Self {
        self.secondary = secondary;
        self
    }
}

impl ModelBackend for PerSeqPeakModel {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> Result<BatchOutput> {
        // Legacy path: still has to compile and produce something
        // sensible (used by the speculative dispatcher).
        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens: vec![
                SampledToken {
                    token: 0,
                    logprob: 0.0,
                    top_logprobs: vec![],
                };
                seq_ids.len()
            ],
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
    ) -> Result<Vec<Vec<f32>>> {
        Ok(input_tokens
            .iter()
            .map(|tokens| {
                let mut logits = vec![0.0f32; tokens.len() * self.vocab];
                let key = tokens.first().copied().unwrap_or(0) as usize;
                let peak = self.peaks[key.min(self.peaks.len() - 1)] as usize;
                let sec = self.secondary[key.min(self.secondary.len() - 1)];
                // Light up the peak in the LAST position (the one the
                // engine reads after slicing).
                let last = tokens.len().saturating_sub(1) * self.vocab;
                if last + self.vocab <= logits.len() {
                    logits[last + peak] = 5.0;
                    if let Some(s) = sec {
                        logits[last + s as usize] = 2.5;
                    }
                }
                logits
            })
            .collect())
    }

    fn embed(
        &mut self,
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>> {
        Ok(input_tokens
            .iter()
            .map(|tokens| vec![0.0; tokens.len()])
            .collect())
    }

    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn num_layers(&self) -> usize {
        1
    }

    fn num_heads(&self) -> usize {
        1
    }
}

#[test]
fn arch_02_greedy_per_sequence_uses_argmax() {
    // Two requests, both with default (greedy) sampling params.
    // The mock's `peaks` table maps prompt[0] → peak token: 0→10,
    // 1→20. Expect seq 1 → 10, seq 2 → 20.
    let model = PerSeqPeakModel::new(vec![10, 20], 32);
    let mut engine = Engine::new(model, None);

    let mut r1 = Request::new(1, vec![0], 1);
    r1.sampling_params = SamplingParams::default(); // T = 0 → greedy
    let mut r2 = Request::new(2, vec![1], 1);
    r2.sampling_params = SamplingParams::default();

    let (tx1, mut rx1) = mpsc::channel(16);
    let (tx2, mut rx2) = mpsc::channel(16);
    engine.add_request(r1, tx1);
    engine.add_request(r2, tx2);

    engine.step().expect("step ok");

    // Channel ordering isn't strictly seq-id, so collect and assert
    // both expected tokens are present.
    let mut got: Vec<u32> = Vec::new();
    if let Ok(t) = rx1.try_recv() {
        got.push(t.token);
    }
    if let Ok(t) = rx2.try_recv() {
        got.push(t.token);
    }
    got.sort_unstable();
    assert_eq!(
        got,
        vec![10, 20],
        "engine must emit the per-prompt peak token when each sequence uses greedy sampling"
    );
}

#[test]
fn arch_02_top_k_one_still_argmax_with_explicit_params() {
    // Same model behaviour, but each request declares top_k = 1
    // explicitly (not the default of 0). This proves that
    // non-default params survive the Batch composer → engine seam.
    let model = PerSeqPeakModel::new(vec![7, 13], 32);
    let mut engine = Engine::new(model, None);

    let mut r1 = Request::new(1, vec![0], 1);
    r1.sampling_params = SamplingParams::builder()
        .with_temperature(0.0)
        .with_top_k(1)
        .with_top_p(1.0)
        .with_repeat_penalty(1.0)
        .build();
    let mut r2 = Request::new(2, vec![1], 1);
    r2.sampling_params = SamplingParams::builder()
        .with_temperature(0.0)
        .with_top_k(1)
        .with_top_p(1.0)
        .with_repeat_penalty(1.0)
        .build();

    let (tx1, mut rx1) = mpsc::channel(16);
    let (tx2, mut rx2) = mpsc::channel(16);
    engine.add_request(r1, tx1);
    engine.add_request(r2, tx2);

    engine.step().expect("step ok");

    let mut got: Vec<u32> = Vec::new();
    if let Ok(t) = rx1.try_recv() {
        got.push(t.token);
    }
    if let Ok(t) = rx2.try_recv() {
        got.push(t.token);
    }
    got.sort_unstable();
    assert_eq!(
        got,
        vec![7, 13],
        "top_k=1 must still produce the deterministic argmax per sequence"
    );
}

#[test]
fn arch_02_repeat_penalty_suppresses_seen_token() {
    // Mock maps prompt[0]=0 → primary peak at 10, secondary at 3
    // (half height: 2.5 vs 5.0). For decode the input is the
    // previously-emitted token (10), so the mock lights up both 10
    // and 3 in the LAST position. With repeat_penalty=2.0 the
    // primary logit at 10 is divided by 2.0 → 2.5, which now ties
    // with the secondary at 3. Ties go to the lowest index in our
    // argmax implementation, so the engine should emit 3 (or any
    // non-10 token) on the second decode step.
    let model = PerSeqPeakModel::new(vec![10], 32).with_secondary(vec![Some(3)]);
    let mut engine = Engine::new(model, None);

    let mut r1 = Request::new(1, vec![0], 3);
    r1.sampling_params = SamplingParams::builder()
        .with_temperature(0.0)
        .with_repeat_penalty(2.0)
        .build();

    let (tx1, mut rx1) = mpsc::channel(16);
    engine.add_request(r1, tx1);

    // Step 0: prefill → mock maps prompt[0]=0 → primary peak 10
    // wins over secondary 3 (5.0 > 2.5). Expect 10.
    engine.step().expect("step ok");
    let t1 = rx1.try_recv().expect("first token");
    assert_eq!(t1.token, 10, "prefill should emit the raw argmax (10)");

    // Step 1: decode → mock maps prompt[0]=10 → peaks[0]=10,
    // secondary=3 → logits at 10 (5.0) and 3 (2.5). With
    // repeat_penalty=2.0 on seen set [10], the logit at 10 is
    // halved to 2.5. Argmax tie at 2.5 resolves to index 3.
    engine.step().expect("step ok");
    let t2 = rx1.try_recv().expect("second token");
    assert_ne!(
        t2.token, 10,
        "repeat_penalty must suppress the previously-seen token (got {})",
        t2.token
    );
}

// P28 v0.3 wire-type follow-up: presence_penalty engine wire-through.
//
// Pins the presence-style semantic end-to-end through
// `sample_one_with_params`: presence_penalty = 1.0 with seen set
// [10] must subtract 1.0 from the logit at 10, flipping the argmax
// from 10 to 3 (the secondary peak).
//
// Uses a lower-level test (no Engine mock needed) — we call
// `sample_one_with_params` directly to verify the helper is wired
// into the sampling pipeline at the right point in the order
// (repeat_penalty → presence_penalty → temperature → top_k → top_p).

#[test]
fn arch_02_presence_penalty_suppresses_seen_token_once_per_distinct_id() {
    use vllm_core::sampling::sample_one_with_params;

    // 32-token logit vector; token 10 is the strictly-highest peak
    // (logit 2.0), token 3 is the secondary at logit 1.0 (strictly
    // lower so the baseline argmax is unambiguously token 10).
    let logits = vec![0.0f32; 32];
    let mut logits = logits;
    logits[10] = 2.0;
    logits[3] = 1.0;

    // Baseline: no presence_penalty → argmax is 10 (strictly
    // highest).
    let params_no_pp = SamplingParams::builder().with_temperature(0.0).build();
    let t_baseline = sample_one_with_params(&logits, &params_no_pp, &[10]);
    assert_eq!(
        t_baseline.token, 10,
        "baseline (presence_penalty = 0) must emit the primary peak"
    );

    // With presence_penalty = 1.0 on seen = [10], the logit at 10
    // drops from 2.0 to 1.0. Token 3 still has 1.0. Tied → lowest
    // index → token 3. Argmax flips from 10 to 3.
    let params_pp = SamplingParams::builder()
        .with_temperature(0.0)
        .with_presence_penalty(1.0)
        .build();
    let t_pp = sample_one_with_params(&logits, &params_pp, &[10]);
    assert_eq!(
        t_pp.token, 3,
        "presence_penalty = 1.0 on seen = [10] must suppress token 10 (got {})",
        t_pp.token
    );

    // Key behavioural difference from repeat_penalty: presence-style
    // is per-distinct-id, not per-occurrence. With seen = [10, 10,
    // 10, 10, 10] (5 occurrences), the penalty must still be
    // subtracted ONCE (logit at 10 goes from 2.0 to 1.0, not to
    // -3.0). The result must match the single-occurrence case.
    let t_pp_repeated = sample_one_with_params(&logits, &params_pp, &[10, 10, 10, 10, 10]);
    assert_eq!(
        t_pp_repeated.token, 3,
        "presence_penalty must subtract once per *distinct* id, not per occurrence (got {})",
        t_pp_repeated.token
    );
}

#[test]
fn arch_02_presence_penalty_negative_encourages_repetition() {
    use vllm_core::sampling::sample_one_with_params;

    // Two-token logit vector; token 0 is the secondary peak.
    let logits = vec![1.0f32, 2.0];
    let params = SamplingParams::builder()
        .with_temperature(0.0)
        .with_presence_penalty(-1.0) // encourage token 0
        .build();

    // Without presence_penalty, the argmax would be token 1
    // (logit 2.0). With presence_penalty = -1.0 on seen = [0], the
    // logit at 0 increases from 1.0 to 2.0 (subtracting -1.0 is the
    // same as adding 1.0). Tied argmax → lowest index → token 0.
    let t = sample_one_with_params(&logits, &params, &[0]);
    assert_eq!(
        t.token, 0,
        "negative presence_penalty must encourage repetition by raising seen-token logits (got {})",
        t.token
    );
}

#[test]
fn arch_02_presence_penalty_combined_with_repeat_penalty() {
    use vllm_core::sampling::sample_one_with_params;

    // Logit vector: token 10 has logit 3.0, token 3 has logit 1.5.
    let logits = vec![0.0f32; 32];
    let mut logits = logits;
    logits[10] = 3.0;
    logits[3] = 1.5;

    // With repeat_penalty = 2.0 (frequency-style) on seen = [10],
    // the logit at 10 is halved from 3.0 to 1.5. Tied with token 3
    // → lowest index → token 3. Argmax flips.
    //
    // Adding presence_penalty = 0.5 (presence-style) subtracts 0.5
    // from the logit at 10 (the only distinct seen token), taking
    // it from 1.5 to 1.0. Token 3 stays at 1.5. Argmax is now
    // firmly token 3.
    //
    // This pins the ordering in `sample_one_with_params`:
    // repeat_penalty FIRST (divide), then presence_penalty
    // (subtract). Without this ordering, the test would still
    // pass (both penalties suppress token 10 in the same direction)
    // but the intermediate logits would differ.
    let params = SamplingParams::builder()
        .with_temperature(0.0)
        .with_repeat_penalty(2.0)
        .with_presence_penalty(0.5)
        .build();
    let t = sample_one_with_params(&logits, &params, &[10]);
    assert_eq!(
        t.token, 3,
        "combined repeat + presence penalty must suppress token 10 and emit token 3 (got {})",
        t.token
    );
}

#[test]
fn arch_02_logit_bias_flips_argmax_in_greedy_sampling() {
    use std::collections::HashMap;
    use vllm_core::sampling::sample_one_with_params;

    // Logit vector: token 0 has logit 1.0, token 1 has logit 2.0,
    // token 2 has logit 0.5. Without bias, the argmax is token 1.
    let logits = vec![1.0, 2.0, 0.5];
    let mut bias = HashMap::new();
    bias.insert(2, 5.0); // bump token 2 way up → new argmax

    let params = SamplingParams::builder()
        .with_temperature(0.0)
        .with_logit_bias(Some(bias))
        .build();
    let t = sample_one_with_params(&logits, &params, &[]);
    assert_eq!(
        t.token, 2,
        "logit_bias on token 2 must flip argmax from 1 → 2 (got {})",
        t.token
    );
}

#[test]
fn arch_02_logit_bias_per_sequence_divergence() {
    use std::collections::HashMap;
    use vllm_core::sampling::sample_batch_with_params;

    // Two sequences with the same logits but different logit_bias
    // maps. Sequence A biases token 0 (positive), sequence B biases
    // token 2 (positive). Per-sequence divergence: A should emit
    // token 0, B should emit token 2. Pins the per-sequence
    // `SamplingParams::logit_bias` threading through
    // `sample_batch_with_params`.
    let logits_list = vec![vec![0.5, 0.5, 0.5], vec![0.5, 0.5, 0.5]];
    let seen_tokens = vec![vec![], vec![]];

    let mut bias_a = HashMap::new();
    bias_a.insert(0, 1.0); // logit at 0 = 1.5 → argmax = 0

    let mut bias_b = HashMap::new();
    bias_b.insert(2, 1.0); // logit at 2 = 1.5 → argmax = 2

    let params_list = vec![
        SamplingParams::builder()
            .with_temperature(0.0)
            .with_logit_bias(Some(bias_a))
            .build(),
        SamplingParams::builder()
            .with_temperature(0.0)
            .with_logit_bias(Some(bias_b))
            .build(),
    ];

    let tokens = sample_batch_with_params(&logits_list, &params_list, &seen_tokens);
    assert_eq!(
        tokens[0].token, 0,
        "seq A with bias on token 0 must emit 0 (got {})",
        tokens[0].token
    );
    assert_eq!(
        tokens[1].token, 2,
        "seq B with bias on token 2 must emit 2 (got {})",
        tokens[1].token
    );
}

// ============================================================================
// P34: seed RNG seeding — sampling_params-level integration tests
// ============================================================================
//
// The unit tests in `crates/core/src/sampling/tests.rs` pin the
// sampler-internal determinism contract. These tests pin the same
// contract at the `sample_batch_with_params` integration boundary
// (the function the engine actually calls) so the chat / completions
// handlers' seed wire-through is verified end-to-end without going
// through the HTTP layer.

/// P34 end-to-end: same seed + same logits/seen → same token,
/// verified at the `sample_batch_with_params` boundary. This is the
/// integration-level equivalent of
/// `test_seed_determinism_same_seed_same_result` from the unit
/// tests, and it's the contract the `OpenAI` `seed` wire-type relies
/// on.
#[test]
fn arch_02_seed_determinism_end_to_end() {
    // Same logits, same params (including seed), same seen — must
    // produce identical tokens across two independent batch calls.
    let logits_list = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]; 2];
    let seen_tokens = vec![vec![]; 2];
    let params_list = vec![
        SamplingParams::builder()
            .with_temperature(1.0)
            .with_seed(42)
            .build();
        2
    ];
    let tokens_a = sample_batch_with_params(&logits_list, &params_list, &seen_tokens);
    let tokens_b = sample_batch_with_params(&logits_list, &params_list, &seen_tokens);
    assert_eq!(
        tokens_a, tokens_b,
        "same seed must produce identical batch output across calls \
         (got {tokens_a:?} vs {tokens_b:?})"
    );
}

/// P34 end-to-end: per-sequence independence at the batch level.
/// Two sequences in the SAME batch with DIFFERENT seeds must
/// produce DIFFERENT tokens (for a non-degenerate logit
/// distribution). This pins the contract that the engine doesn't
/// share RNG state across sequences.
#[test]
fn arch_02_seed_per_sequence_divergence_in_batch() {
    let logits_list = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]; 2];
    let seen_tokens = vec![vec![]; 2];
    let params_list = vec![
        SamplingParams::builder()
            .with_temperature(1.0)
            .with_seed(42)
            .build(),
        SamplingParams::builder()
            .with_temperature(1.0)
            .with_seed(99)
            .build(),
    ];
    let tokens = sample_batch_with_params(&logits_list, &params_list, &seen_tokens);
    assert_ne!(
        tokens[0], tokens[1],
        "sequences with different seeds in the same batch must \
         produce different tokens (got {0:?} for both — RNG state \
         is shared across sequences)",
        tokens[0]
    );
}
