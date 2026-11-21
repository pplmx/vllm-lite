// crates/core/tests/chunked_prefill_no_stale_stream.rs
//
// Regression: a mid-chunk prefill's "predicted
// next prompt token" must never reach the client's response channel.
//
// During chunked prefill each non-final chunk runs a forward that predicts
// the model's guess for the NEXT prompt token. That prediction is not
// generated output — the real next prompt token is re-fed on the next
// chunk — but `send_and_collect_results` used to forward every sampled
// token to `response_txs` regardless of phase, so a 100-token prompt
// chunked by 16 streamed six spurious "generated" tokens before the real
// generation began (polluting OpenAI `content`, `logprobs`, and the
// streaming SSE body).
//
// This test pins the contract: the response channel carries exactly
// `max_tokens` real output tokens — the final chunk's first generated
// token plus the decode tokens — and no stale mid-chunk prediction.

use tokio::sync::mpsc;
use vllm_core::Engine;
use vllm_core::types::{Request, SchedulerConfig};
use vllm_traits::{ModelBackend, Result as ModelResult, SampledToken, SeqId, TokenId};

/// Distinguishing mock: prefill chunks logits peak at `PREFILL_MARK` (111),
/// decode steps peak at `DECODE_MARK` (222). Because a mid-chunk prefill and
/// the final prefill chunk both run `is_prefill = true`, the mark alone
/// cannot tell them apart — that is the point: the scheduler must suppress
/// the mid-chunk (stale) marks and keep exactly the final chunk's mark.
const PREFILL_MARK: TokenId = 111;
const DECODE_MARK: TokenId = 222;
const VOCAB_SIZE: usize = 512;

/// Mock backend whose argmax is a distinctive token per phase.
#[derive(Clone)]
struct PhaseMarkModel;

impl ModelBackend for PhaseMarkModel {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> ModelResult<vllm_traits::BatchOutput> {
        Ok(vllm_traits::BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens: seq_ids
                .iter()
                .map(|_| SampledToken {
                    token: PREFILL_MARK,
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
        is_prefill: &[bool],
    ) -> ModelResult<Vec<Vec<f32>>> {
        Ok(input_tokens
            .iter()
            .zip(is_prefill)
            .map(|(tokens, prefill)| {
                let mark = if *prefill { PREFILL_MARK } else { DECODE_MARK };
                // One vocab-sized vector per input token; the engine samples
                // the LAST position, so the mark repeats for every position.
                tokens
                    .iter()
                    .flat_map(|_| {
                        let mut logits = vec![-10.0; VOCAB_SIZE];
                        logits[mark as usize] = 10.0;
                        logits
                    })
                    .collect()
            })
            .collect())
    }

    fn embed(
        &mut self,
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
    ) -> ModelResult<Vec<Vec<f32>>> {
        Ok(input_tokens
            .iter()
            .zip(positions)
            .map(|(tokens, _pos)| vec![0.0; tokens.len()])
            .collect())
    }

    fn vocab_size(&self) -> usize {
        VOCAB_SIZE
    }

    fn num_layers(&self) -> usize {
        1
    }

    fn num_heads(&self) -> usize {
        1
    }
}

/// Drain a response channel until empty, returning token ids in order.
fn drain_tokens(rx: &mut mpsc::Receiver<SampledToken>) -> Vec<TokenId> {
    let mut out = Vec::new();
    while let Ok(sampled) = rx.try_recv() {
        out.push(sampled.token);
    }
    out
}

// chunked prefill must not stream stale mid-chunk
// predictions. A 100-token prompt with budget 32 / chunk 16 runs six
// mid-chunk rounds (each predicting the NEXT prompt token: mark 111), then
// one final chunk (whose single real first output is also mark 111 because
// the model cannot distinguish it from a mid-chunk), then seven decode
// rounds (mark 222 each, max_tokens = 8). The client must see exactly
// [111, 222 x 7] — eight tokens — never the six stale 111s. Fails pre-fix.
#[test]
fn chunked_prefill_never_streams_stale_midchunk_predictions() {
    // Budget 32 forces the 100-token prompt to be chunked; chunk 16 caps
    // each round. 64 KV blocks comfortably holds 7 prompt blocks + decode.
    let config = SchedulerConfig::builder()
        .with_max_num_batched_tokens(32)
        .with_prefill_chunk_size(16)
        .build();
    let mut engine = Engine::with_config_boxed(Box::new(PhaseMarkModel), None, config, 0, 64);

    let (tx, mut rx) = mpsc::channel(64);
    engine.add_request(Request::new(7, vec![7; 100], 8), tx);

    // Drive rounds until the queue is fully drained (prefill + all decode).
    let mut rounds = 0;
    while engine.has_pending() {
        engine.step().expect("every chunk/decode step must succeed");
        rounds += 1;
        assert!(
            rounds <= 40,
            "the sequence must complete within 7 prefill chunks + max_tokens decode rounds"
        );
    }

    let received = drain_tokens(&mut rx);
    // Exactly the real output stream: one first-token from the final chunk
    // (mark 111) followed by the 7 decode tokens (mark 222). If a stale
    // mid-chunk prediction leaked to the channel, `received.len()` >= 14.
    assert_eq!(
        received,
        {
            let mut expect = vec![PREFILL_MARK];
            expect.extend(std::iter::repeat_n(DECODE_MARK, 7));
            expect
        },
        "the client stream must contain exactly max_tokens real output tokens \
         (final-chunk first token + 7 decodes), no stale mid-chunk predictions\n  got {received:?}"
    );
}

/// The CUDA-graph step path must apply the SAME mid-chunk suppression.
///
/// `Engine::process_output` (graph path) computes its own `stale_mask` with
/// `batch.num_computed_tokens[i] + batch.input_tokens[i].len() < prompt_len`.
/// `cargo mutants` flipped that `<` to `>` and the whole suite stayed green:
/// the CPU `chunked_prefill_never_streams_stale_midchunk_predictions` above
/// drives `engine.step()`, never `step_with_graph()`, so the graph copy of the
/// predicate was unverified. Inverted, the mask marks the WRONG rounds — the
/// stale mid-chunk predictions get through and the real final-chunk token is
/// withheld.
#[test]
#[cfg(feature = "cuda-graph")]
fn cuda_graph_step_never_streams_stale_midchunk_predictions() {
    use vllm_core::scheduler::cuda_graph::SchedulerCudaGraphConfig;

    let config = SchedulerConfig::builder()
        .with_max_num_batched_tokens(32)
        .with_prefill_chunk_size(16)
        .build();
    let config = SchedulerConfig {
        cuda_graph: SchedulerCudaGraphConfig {
            enabled: true,
            batch_sizes: vec![1, 2, 4],
        },
        ..config
    };
    let mut engine = Engine::with_config_boxed(Box::new(PhaseMarkModel), None, config, 0, 64);
    engine
        .capture_cuda_graphs()
        .expect("capturing the configured batch sizes must succeed");
    assert!(
        engine.cuda_graph_enabled(),
        "fixture must have captured graphs for the graph path"
    );

    let (tx, mut rx) = mpsc::channel(64);
    engine.add_request(Request::new(7, vec![7; 100], 8), tx);

    let mut rounds = 0;
    while engine.has_pending() {
        engine
            .step_with_graph()
            .expect("every graph chunk/decode step must succeed");
        rounds += 1;
        assert!(
            rounds <= 40,
            "must finish within 7 chunks + 8 decode rounds"
        );
    }

    let received = drain_tokens(&mut rx);
    assert_eq!(
        received,
        {
            let mut expect = vec![PREFILL_MARK];
            expect.extend(std::iter::repeat_n(DECODE_MARK, 7));
            expect
        },
        "the graph path must stream exactly max_tokens real tokens (final-chunk \
         first token + 7 decodes) and suppress every stale mid-chunk prediction\n  got {received:?}"
    );
}

/// The CUDA-graph step path must record its per-step metrics.
///
/// `record_batch_metrics` books the token count, the prefill/decode phase
/// split, the batch size and the step latency. `cargo mutants` replaced the
/// whole function with `()` and every test still passed — the graph path's
/// metrics were entirely unverified, so an engine running graphs would have
/// reported zero throughput and zero batch size to `/metrics` while serving
/// normally. This pins the token count and the batch size, which are the two
/// observable regardless of how fast the step runs.
#[test]
#[cfg(feature = "cuda-graph")]
fn cuda_graph_step_records_batch_metrics() {
    use vllm_core::scheduler::cuda_graph::SchedulerCudaGraphConfig;
    use vllm_testing::FakeModel;

    let config = SchedulerConfig {
        cuda_graph: SchedulerCudaGraphConfig {
            enabled: true,
            batch_sizes: vec![1, 2, 4],
        },
        ..Default::default()
    };
    let mut engine = Engine::with_config(FakeModel::new(1000), None, config, 4, 1024);
    engine
        .capture_cuda_graphs()
        .expect("capturing the configured batch sizes must succeed");

    let before = engine.scheduler.metrics.snapshot();
    assert_eq!(
        before.tokens_total, 0,
        "setup: a fresh engine has no tokens"
    );

    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(Request::new(1, vec![5, 6], 4), tx);
    for _ in 0..6 {
        if !engine.has_pending() {
            break;
        }
        engine.step_with_graph().expect("graph step must succeed");
    }

    let after = engine.scheduler.metrics.snapshot();
    assert_eq!(
        after.tokens_total, 4,
        "the graph path must count every emitted token (max_tokens = 4)"
    );
    assert_eq!(
        after.avg_batch_size, 1.0,
        "the graph path must record the batch size it ran"
    );
}

/// Without the `cuda-graph` feature, `step_with_graph` must still run the
/// regular step.
///
/// The fallback definition warns and delegates to `step()`. `cargo mutants`
/// replaced its body with `Ok(vec![])` and nothing failed: the only two tests
/// that call `step_with_graph` are both `#[cfg(feature = "cuda-graph")]`, so
/// the plain build's copy was never executed. A caller that always routed
/// through `step_with_graph` would then get an empty result forever while the
/// engine silently stopped producing tokens.
#[test]
fn fallback_step_with_graph_runs_the_regular_step() {
    use vllm_testing::FakeModel;

    let mut engine = Engine::with_config(
        FakeModel::new(1000),
        None,
        SchedulerConfig::default(),
        0,
        64,
    );
    assert!(
        !engine.cuda_graph_enabled(),
        "fixture must take the non-graph fallback"
    );

    let (tx, mut rx) = mpsc::channel(64);
    engine.add_request(Request::new(1, vec![5, 6], 4), tx);

    let mut rounds = 0;
    while engine.has_pending() && rounds < 10 {
        let produced = engine
            .step_with_graph()
            .expect("the fallback must propagate a successful step");
        assert_eq!(
            produced.len(),
            1,
            "each fallback step must report the token it generated"
        );
        rounds += 1;
    }

    assert!(!engine.has_pending(), "the request must have completed");
    assert_eq!(
        drain_tokens(&mut rx).len(),
        4,
        "the fallback must stream every generated token (max_tokens = 4)"
    );
}
