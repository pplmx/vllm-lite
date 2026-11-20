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
