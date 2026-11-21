//! A matched `stop` sequence must not reach the client.
//!
//! OpenAI's contract for `stop`: "Up to 4 sequences where the API will stop
//! generating further tokens. **The returned text will not contain the stop
//! sequence.**" (platform docs + openai-python `CompletionCreateParams`).
//!
//! The engine suffix-matches the stop and finalizes the sequence, but the
//! matched tokens are sent to the response channel before that check runs,
//! so `stop: ["END"]` delivers `...END` to the client.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::mpsc;
use vllm_core::engine::Engine;
use vllm_core::types::Request;
use vllm_traits::{BatchOutput, ModelBackend, Result, SampledToken, SeqId, TokenId};

const VOCAB: usize = 4096;

/// Emits `tokens` in order: `forward_logits` puts all its mass on the next
/// token so greedy decoding selects exactly that id.
struct SequenceModel {
    tokens: Vec<TokenId>,
    next: Arc<AtomicUsize>,
}

impl SequenceModel {
    fn desired(&self) -> TokenId {
        let idx = self.next.fetch_add(1, Ordering::SeqCst);
        self.tokens.get(idx).copied().unwrap_or(0)
    }
}

impl ModelBackend for SequenceModel {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> Result<BatchOutput> {
        let token = self.desired();
        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens: seq_ids
                .iter()
                .map(|_| SampledToken {
                    token,
                    logprob: 0.0,
                    top_logprobs: Vec::new(),
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
    ) -> Result<Vec<Vec<f32>>> {
        let token = usize::try_from(self.desired()).unwrap_or(0).min(VOCAB - 1);
        Ok(input_tokens
            .iter()
            .map(|tokens| {
                let mut logits = Vec::with_capacity(tokens.len() * VOCAB);
                for _ in tokens {
                    let mut pos = vec![-10.0_f32; VOCAB];
                    pos[token] = 10.0;
                    logits.extend(pos);
                }
                logits
            })
            .collect())
    }

    fn embed(
        &mut self,
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>> {
        Ok(Vec::new())
    }

    fn vocab_size(&self) -> usize {
        VOCAB
    }

    fn num_layers(&self) -> usize {
        1
    }

    fn num_heads(&self) -> usize {
        1
    }
}

fn streamed_tokens(tokens: Vec<TokenId>, stop: Vec<TokenId>) -> Vec<TokenId> {
    let model = SequenceModel {
        tokens,
        next: Arc::new(AtomicUsize::new(0)),
    };
    let mut engine = Engine::new_boxed(Box::new(model), None::<Box<dyn ModelBackend>>);
    let (tx, mut rx) = mpsc::channel(64);

    let mut request = Request::new(1, vec![1, 2], 8);
    request.sampling_params.stop_token_sequences = Some(vec![stop]);
    engine.add_request(request, tx);

    for _ in 0..10 {
        engine.step().unwrap();
    }
    let mut out = Vec::new();
    while let Ok(sampled) = rx.try_recv() {
        out.push(sampled.token);
    }
    out
}

#[test]
fn matched_stop_sequence_is_not_streamed_to_the_client() {
    let streamed = streamed_tokens(vec![5, 20, 21], vec![20, 21]);
    assert_eq!(
        streamed,
        vec![5],
        "the stop sequence must not reach the client"
    );
}

#[test]
fn single_token_stop_is_not_streamed() {
    let streamed = streamed_tokens(vec![7, 9, 7], vec![9]);
    assert_eq!(streamed, vec![7], "a one-token stop must be withheld too");
}

#[test]
fn tokens_before_the_stop_still_stream() {
    let streamed = streamed_tokens(vec![3, 4, 5, 20, 21], vec![20, 21]);
    assert_eq!(streamed, vec![3, 4, 5], "content before the stop is kept");
}

/// The speculative path must withhold stop tokens too.
///
/// `emit_verified_tokens` sends before `fold_speculative_update` appends this
/// step's tokens to `seq.tokens`, so the stop suffix is not readable at send
/// time. A speculative request therefore still streams the tokens that
/// complete its `stop` sequence.
#[test]
fn speculative_path_also_withholds_stop_tokens() {
    use vllm_testing::FakeModel;

    // `FakeModel` emits the last prompt token's value every step, so a prompt
    // ending in 6 makes the generated stream 6, 6, 6, ...
    let target = FakeModel::new(1000);
    let draft = FakeModel::new(1000);
    let mut engine = Engine::new_boxed(Box::new(target), Some(Box::new(draft)));
    engine.max_draft_tokens = 2;
    engine.enable_speculative();

    let (tx, mut rx) = mpsc::channel(64);
    let mut request = Request::new(1, vec![5, 6], 12);
    request.sampling_params.stop_token_sequences = Some(vec![vec![6, 6]]);
    engine.add_request(request, tx);

    for _ in 0..8 {
        engine.step().unwrap();
    }
    // The sequence really stops (otherwise this test would prove nothing).
    assert!(!engine.has_pending(), "the stop sequence must have matched");

    let mut streamed = Vec::new();
    while let Ok(sampled) = rx.try_recv() {
        streamed.push(sampled.token);
    }
    assert!(
        streamed.is_empty(),
        "both generated tokens form the stop sequence, so neither may stream; got {streamed:?}"
    );
}

/// The CUDA-graph step path must withhold stop tokens as well.
///
/// `step_with_graph` has its own send loop (`graph_step.rs`) that was given
/// the same hold-back as the regular path; without a test here that wiring
/// could regress silently on the `cuda-graph` feature only.
#[test]
#[cfg(feature = "cuda-graph")]
fn cuda_graph_path_also_withholds_stop_tokens() {
    use vllm_core::scheduler::cuda_graph::SchedulerCudaGraphConfig;
    use vllm_core::types::SchedulerConfig;
    use vllm_testing::FakeModel;

    let config = SchedulerConfig {
        cuda_graph: SchedulerCudaGraphConfig {
            enabled: true,
            batch_sizes: vec![1, 2, 4],
        },
        ..Default::default()
    };
    // Same engine construction the CUDA-graph integration tests use.
    let mut engine = Engine::with_config(FakeModel::new(1000), None, config, 4, 1024);
    // `cuda_graph_enabled` reports readiness-to-replay, so the fixture must
    // actually capture before the graph path is taken.
    engine
        .capture_cuda_graphs()
        .expect("capturing the configured batch sizes must succeed");
    assert!(
        engine.cuda_graph_enabled(),
        "fixture must have captured graphs for the graph path"
    );

    let (tx, mut rx) = mpsc::channel(64);
    let mut request = Request::new(1, vec![5, 6], 12);
    request.sampling_params.stop_token_sequences = Some(vec![vec![6, 6]]);
    engine.add_request(request, tx);

    for _ in 0..8 {
        engine.step_with_graph().unwrap();
    }
    assert!(
        !engine.has_pending(),
        "the stop sequence must have matched on the graph path"
    );

    let mut streamed = Vec::new();
    while let Ok(sampled) = rx.try_recv() {
        streamed.push(sampled.token);
    }
    assert!(
        streamed.is_empty(),
        "the graph path must not stream stop tokens; got {streamed:?}"
    );
}

/// The hold-back buffer must not outlive its sequence.
///
/// `pending_stream` is keyed by `SeqId`, and ids are monotonic — so an entry
/// left behind is never reused or overwritten. Every request that configured
/// a multi-token stop allocates one, which makes the leak unbounded for a
/// long-lived server.
#[test]
fn pending_stream_is_cleared_when_a_sequence_finishes() {
    // Prompt [1, 2]; the model emits 5, then 20, 21 (which completes the stop
    // `[20, 21]`), so the buffer is created and then discarded mid-flight —
    // but the map entry itself must go when the sequence is finalized.
    let model = SequenceModel {
        tokens: vec![5, 20, 21],
        next: Arc::new(AtomicUsize::new(0)),
    };
    let mut engine = Engine::new_boxed(Box::new(model), None::<Box<dyn ModelBackend>>);
    let (tx, _rx) = mpsc::channel(64);

    let mut request = Request::new(1, vec![1, 2], 8);
    request.sampling_params.stop_token_sequences = Some(vec![vec![20, 21]]);
    engine.add_request(request, tx);

    for _ in 0..10 {
        engine.step().unwrap();
    }
    assert!(
        !engine.has_pending(),
        "the sequence must have finished for this test to mean anything"
    );
    assert!(
        engine.scheduler.pending_stream_len() == 0,
        "the hold-back buffer must be dropped when the sequence finishes"
    );
}

/// Same for a plain length-terminated sequence that never matched a stop, so
/// the buffer holds unreleased tokens at finish time.
#[test]
fn pending_stream_is_cleared_when_a_sequence_hits_max_tokens() {
    let model = SequenceModel {
        tokens: vec![7, 7, 7, 7],
        next: Arc::new(AtomicUsize::new(0)),
    };
    let mut engine = Engine::new_boxed(Box::new(model), None::<Box<dyn ModelBackend>>);
    let (tx, _rx) = mpsc::channel(64);

    // `max_tokens = 2` so the sequence ends by length with tokens held back.
    let mut request = Request::new(1, vec![1, 2], 2);
    request.sampling_params.stop_token_sequences = Some(vec![vec![9, 9, 9]]);
    engine.add_request(request, tx);

    for _ in 0..10 {
        engine.step().unwrap();
    }
    assert!(!engine.has_pending(), "the sequence must have finished");
    assert!(
        engine.scheduler.pending_stream_len() == 0,
        "a length-terminated sequence must not leave its buffer behind"
    );
}

/// Withholding must not shorten the output when the sequence ends by length.
///
/// A `stop` of length 3 makes the engine hold back 2 trailing tokens. If the
/// sequence then ends by `max_tokens` instead of matching, those tokens are
/// real output — dropping them silently truncates what the client asked for
/// (`max_tokens = 4` delivered only 2).
#[test]
fn length_terminated_sequence_still_streams_its_held_tokens() {
    let model = SequenceModel {
        tokens: vec![7, 7, 7, 7],
        next: Arc::new(AtomicUsize::new(0)),
    };
    let mut engine = Engine::new_boxed(Box::new(model), None::<Box<dyn ModelBackend>>);
    let (tx, mut rx) = mpsc::channel(64);

    // `max_tokens = 4`, and a stop that never matches so nothing is withheld
    // permanently — but 2 tokens are always in flight.
    let mut request = Request::new(1, vec![1, 2], 4);
    request.sampling_params.stop_token_sequences = Some(vec![vec![9, 9, 9]]);
    engine.add_request(request, tx);

    for _ in 0..12 {
        let _ = engine.step();
    }

    let mut streamed = Vec::new();
    while let Ok(sampled) = rx.try_recv() {
        streamed.push(sampled.token);
    }
    assert_eq!(
        streamed.len(),
        4,
        "every generated token must reach the client, got {streamed:?}"
    );
}

/// A stop landing exactly on the `max_tokens` boundary must still be withheld.
///
/// `check_completion` finishes a sequence as soon as it has `max_tokens`
/// output tokens, which removes it from `running` before
/// `finalize_stop_sequences` runs — so the match is never observed there and
/// the finalization reports `Length`. The held-back tail was then flushed as
/// ordinary output, delivering the stop sequence the contract says to
/// withhold.
#[test]
fn stop_on_the_max_tokens_boundary_is_still_withheld() {
    // Every emitted token is 7; `max_tokens = 2` lands exactly on `[7, 7]`.
    // `StubModel` returns a constant, so prefill and decode agree (a
    // sequence-driven mock would burn its list during prefill).
    let model = vllm_testing::StubModel::returning(7);
    let mut engine = Engine::new_boxed(Box::new(model), None::<Box<dyn ModelBackend>>);
    let (tx, mut rx) = mpsc::channel(64);

    let mut request = Request::new(1, vec![1, 2], 2);
    request.sampling_params.stop_token_sequences = Some(vec![vec![7, 7]]);
    engine.add_request(request, tx);

    for _ in 0..10 {
        let _ = engine.step();
    }
    assert!(!engine.has_pending(), "the sequence must have finished");

    let mut streamed = Vec::new();
    while let Ok(sampled) = rx.try_recv() {
        streamed.push(sampled.token);
    }
    assert!(
        streamed.is_empty(),
        "the whole output is the stop sequence, so nothing may stream; got {streamed:?}"
    );
}

/// A longer stop on the boundary withholds too (`max_tokens = 4`, stop of 4).
#[test]
fn longer_stop_on_the_max_tokens_boundary_is_withheld() {
    let model = vllm_testing::StubModel::returning(7);
    let mut engine = Engine::new_boxed(Box::new(model), None::<Box<dyn ModelBackend>>);
    let (tx, mut rx) = mpsc::channel(64);

    let mut request = Request::new(1, vec![1, 2], 4);
    request.sampling_params.stop_token_sequences = Some(vec![vec![7, 7, 7, 7]]);
    engine.add_request(request, tx);

    for _ in 0..12 {
        let _ = engine.step();
    }
    assert!(!engine.has_pending(), "the sequence must have finished");

    let mut streamed = Vec::new();
    while let Ok(sampled) = rx.try_recv() {
        streamed.push(sampled.token);
    }
    assert!(
        streamed.is_empty(),
        "the whole output is the stop sequence; got {streamed:?}"
    );
}

/// The boundary record must not outlive its sequence either.
#[test]
fn stop_ended_record_is_released_at_finalization() {
    let model = vllm_testing::StubModel::returning(7);
    let mut engine = Engine::new_boxed(Box::new(model), None::<Box<dyn ModelBackend>>);
    let (tx, _rx) = mpsc::channel(64);

    let mut request = Request::new(1, vec![1, 2], 2);
    request.sampling_params.stop_token_sequences = Some(vec![vec![7, 7]]);
    engine.add_request(request, tx);

    for _ in 0..10 {
        let _ = engine.step();
    }
    assert!(!engine.has_pending(), "the sequence must have finished");
    assert_eq!(
        engine.scheduler.stop_ended_len(),
        0,
        "the boundary record must be cleared when the sequence finalizes"
    );
}

/// The speculative path must withhold a stop on the boundary too.
///
/// `emit_verified_tokens` has its own send loop; it needs the same
/// `stop_ended` check the regular path gained, or a speculative request
/// whose stop lands exactly on `max_tokens` still streams it.
#[test]
fn speculative_path_withholds_a_stop_on_the_boundary() {
    use vllm_testing::FakeModel;

    // `FakeModel` emits the last prompt token every step: a prompt ending in
    // 6 produces 6, 6, 6, ...  `max_tokens = 2` lands on `[6, 6]`.
    let target = FakeModel::new(1000);
    let draft = FakeModel::new(1000);
    let mut engine = Engine::new_boxed(Box::new(target), Some(Box::new(draft)));
    engine.max_draft_tokens = 2;
    engine.enable_speculative();

    let (tx, mut rx) = mpsc::channel(64);
    let mut request = Request::new(1, vec![5, 6], 2);
    request.sampling_params.stop_token_sequences = Some(vec![vec![6, 6]]);
    engine.add_request(request, tx);

    for _ in 0..8 {
        let _ = engine.step();
    }
    assert!(!engine.has_pending(), "the sequence must have finished");

    let mut streamed = Vec::new();
    while let Ok(sampled) = rx.try_recv() {
        streamed.push(sampled.token);
    }
    assert!(
        streamed.is_empty(),
        "the boundary stop must be withheld on the speculative path; got {streamed:?}"
    );
}

/// The CUDA-graph path must withhold a boundary stop as well.
#[test]
#[cfg(feature = "cuda-graph")]
fn cuda_graph_path_withholds_a_stop_on_the_boundary() {
    use vllm_core::scheduler::cuda_graph::SchedulerCudaGraphConfig;
    use vllm_core::types::SchedulerConfig;
    use vllm_testing::StubModel;

    let config = SchedulerConfig {
        cuda_graph: SchedulerCudaGraphConfig {
            enabled: true,
            batch_sizes: vec![1, 2, 4],
        },
        ..Default::default()
    };
    let mut engine = Engine::with_config(StubModel::returning(7), None, config, 4, 1024);
    // `cuda_graph_enabled` reports readiness-to-replay, so the fixture must
    // actually capture before the graph path is taken.
    engine
        .capture_cuda_graphs()
        .expect("capturing the configured batch sizes must succeed");
    assert!(
        engine.cuda_graph_enabled(),
        "fixture must have captured graphs for the graph path"
    );

    let (tx, mut rx) = mpsc::channel(64);
    let mut request = Request::new(1, vec![1, 2], 2);
    request.sampling_params.stop_token_sequences = Some(vec![vec![7, 7]]);
    engine.add_request(request, tx);

    for _ in 0..10 {
        let _ = engine.step_with_graph();
    }
    assert!(!engine.has_pending(), "the sequence must have finished");

    let mut streamed = Vec::new();
    while let Ok(sampled) = rx.try_recv() {
        streamed.push(sampled.token);
    }
    assert!(
        streamed.is_empty(),
        "the boundary stop must be withheld on the graph path; got {streamed:?}"
    );
}

/// A stop that completes INSIDE a speculative burst must be withheld too.
///
/// The speculative path folds the whole burst into `seq.tokens` before
/// emitting, so the stop is no longer at the tail when the send loop runs:
/// `token_completes_stop_sequence` (a tail check) and
/// `finalize_stop_sequences` (a suffix check on the folded sequence) both
/// look at the LAST token only. With `stop: [[8, 9]]` against a stream of
/// 7, 8, 9, 10, ... the client used to receive the whole stream while the
/// sequence kept generating.
///
/// The fold now cuts the burst at the match point and the step ends the
/// sequence there, so the client sees only the tokens generated before it.
#[test]
fn speculative_mid_burst_stop_is_withheld() {
    use vllm_testing::ConsistentModel;

    let mut engine = Engine::new_boxed(
        Box::new(ConsistentModel::new(1000)),
        Some(Box::new(ConsistentModel::new(1000))),
    );
    engine.max_draft_tokens = 4;
    engine.enable_speculative();

    let (tx, mut rx) = mpsc::channel(64);
    // ConsistentModel emits 7, 8, 9, 10, ...; the stop completes at tokens
    // 2-3, i.e. inside the first burst.
    let mut request = Request::new(1, vec![5, 6], 20);
    request.sampling_params.stop_token_sequences = Some(vec![vec![8, 9]]);
    let seq_id = engine.add_request(request, tx);
    let (finish_tx, mut finish_rx) = tokio::sync::oneshot::channel();
    engine.finish_reason_txs.insert(seq_id, finish_tx);

    for _ in 0..3 {
        let _ = engine.step();
    }

    let mut streamed = Vec::new();
    while let Ok(sampled) = rx.try_recv() {
        streamed.push(sampled.token);
    }
    assert!(
        !streamed.iter().any(|&t| t == 8 || t == 9),
        "a stop completing mid-burst must be withheld, got {streamed:?}"
    );
    assert_eq!(
        streamed,
        vec![7],
        "only the token before the stop may stream; got {streamed:?}"
    );
    assert!(
        !engine.has_pending(),
        "the sequence must end at the stop, not keep generating past it"
    );
    assert_eq!(
        engine.scheduler.running_count(),
        0,
        "the stopped sequence must leave the running set"
    );
    assert_eq!(
        finish_rx
            .try_recv()
            .expect("the handler must learn why it ended"),
        vllm_traits::FinishReason::Stop,
        "a mid-burst stop must report Stop, not Length"
    );
}

/// A stop whose opening token was generated by an EARLIER burst must be
/// withheld across the burst boundary.
///
/// With no drafts (`max_draft_tokens = 0`) each decode step emits exactly one
/// token, so `stop: [[8, 9]]` against 7, 8, 9, ... always completes in a
/// step after the one that produced `8`. `8` is in the send loop's hold-back
/// buffer at that point (held because it may start the stop — and here it
/// does), and the fold sees a match that began before the burst. The buffer
/// must be discarded, not flushed as ordinary output.
#[test]
fn speculative_stop_spanning_two_bursts_is_withheld() {
    use vllm_testing::ConsistentModel;

    let mut engine = Engine::new_boxed(
        Box::new(ConsistentModel::new(1000)),
        Some(Box::new(ConsistentModel::new(1000))),
    );
    engine.max_draft_tokens = 0;
    engine.enable_speculative();

    let (tx, mut rx) = mpsc::channel(64);
    let mut request = Request::new(1, vec![5, 6], 20);
    request.sampling_params.stop_token_sequences = Some(vec![vec![8, 9]]);
    let seq_id = engine.add_request(request, tx);
    let (finish_tx, mut finish_rx) = tokio::sync::oneshot::channel();
    engine.finish_reason_txs.insert(seq_id, finish_tx);

    for _ in 0..12 {
        let _ = engine.step();
        if !engine.has_pending() {
            break;
        }
    }

    let mut streamed = Vec::new();
    while let Ok(sampled) = rx.try_recv() {
        streamed.push(sampled.token);
    }
    assert_eq!(
        streamed,
        vec![6, 7],
        "only the tokens before the stop may stream, even when the stop starts in \
         the previous burst; got {streamed:?}"
    );
    assert!(
        !engine.has_pending(),
        "the sequence must end at the stop, not keep generating past it"
    );
    assert_eq!(
        finish_rx
            .try_recv()
            .expect("the handler must learn why it ended"),
        vllm_traits::FinishReason::Stop,
        "a spanning stop must report Stop, not Length"
    );
}
