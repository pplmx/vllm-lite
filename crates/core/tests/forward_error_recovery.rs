//! Forward-error recovery (RIL TASK-049, ISS-045).
//!
//! `build_batch` moves freshly-admitted prefills into `running` as
//! `Status::Prefilling` *before* the model forward runs. If the forward
//! errors, the error propagates with `?` before `update()` ever runs, so
//! those sequences stay stranded in `running` as `Prefilling` — which
//! `build_batch` never re-includes (only `Decoding` is) — pinning their KV
//! blocks, keeping `has_pending()` true forever (engine busy-spins), and
//! never firing the client's finish channel (request hangs).
//!
//! `Engine::step` now rolls back stranded `Prefilling` sequences on error
//! (`scheduler.requeue_stuck_prefills`), so a transient forward failure
//! re-queues the request for retry instead of permanently wedging the
//! engine.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use vllm_core::Engine;
use vllm_core::types::{Request, SchedulerConfig};
use vllm_traits::{
    BatchOutput, ModelBackend, ModelError, Result as ModelResult, SampledToken, SeqId, TokenId,
};

/// Target backend whose `forward_logits` always fails — models a
/// permanently-broken forward (e.g. a poisoned/malfunctioning GPU kernel).
struct AlwaysFailingBackend;

impl ModelBackend for AlwaysFailingBackend {
    fn forward(
        &mut self,
        _seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> ModelResult<BatchOutput> {
        Err(ModelError::new("boom"))
    }

    fn forward_logits(
        &mut self,
        _seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> ModelResult<Vec<Vec<f32>>> {
        Err(ModelError::new("boom"))
    }

    fn embed(
        &mut self,
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> ModelResult<Vec<Vec<f32>>> {
        Ok(vec![])
    }

    fn vocab_size(&self) -> usize {
        32
    }
    fn num_layers(&self) -> usize {
        1
    }
    fn num_heads(&self) -> usize {
        1
    }
}

/// Target backend that fails the FIRST `forward_logits`, then succeeds —
/// models a transient failure (e.g. a one-shot OOM or driver hiccup).
struct FlakyBackend {
    armed: Arc<AtomicBool>,
}

impl ModelBackend for FlakyBackend {
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
    ) -> ModelResult<Vec<Vec<f32>>> {
        if self.armed.swap(false, Ordering::SeqCst) {
            return Err(ModelError::new("transient boom"));
        }
        // One well-formed `vocab_size` logit per token (mirrors the FALL-02
        // ErrorBackend shape) so the engine can extract a next token.
        Ok(input_tokens
            .iter()
            .map(|tokens| vec![-10.0_f32; tokens.len() * self.vocab_size()])
            .collect())
    }

    fn embed(
        &mut self,
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> ModelResult<Vec<Vec<f32>>> {
        Ok(vec![])
    }

    fn vocab_size(&self) -> usize {
        32
    }
    fn num_layers(&self) -> usize {
        1
    }
    fn num_heads(&self) -> usize {
        1
    }
}

fn config() -> SchedulerConfig {
    SchedulerConfig::builder()
        .with_max_num_batched_tokens(4096)
        .build()
}

#[test]
fn persistent_forward_error_recovers_orphaned_prefill() {
    let mut engine =
        Engine::with_config_boxed(Box::new(AlwaysFailingBackend), None, config(), 4, 128);

    let (tx, _rx) = tokio::sync::mpsc::channel(64);
    engine.add_request(Request::new(7, vec![1, 2, 3], 5), tx);

    let result = engine.step();
    assert!(result.is_err(), "a failing forward must surface an error");

    // RIL ISS-045: the admitted prefill must NOT be stranded in `running`
    // as Prefilling (which would pin its KV block and keep has_pending
    // spinning forever); it is rolled back to the waiting queue.
    assert_eq!(
        engine.scheduler.running_count(),
        0,
        "no sequence may stay stranded in running after a forward error"
    );
    assert_eq!(
        engine.scheduler.waiting_count(),
        1,
        "the request must be re-queued for retry, not lost"
    );
    assert!(
        engine.has_pending(),
        "the recovered request is still pending for a later retry"
    );

    // A second step behaves the same way (still failing, still recovering)
    // rather than wedging the scheduler.
    let result = engine.step();
    assert!(result.is_err());
    assert_eq!(engine.scheduler.running_count(), 0);
    assert_eq!(engine.scheduler.waiting_count(), 1);
}

#[test]
fn transient_forward_error_retries_and_succeeds() {
    let mut engine = Engine::with_config_boxed(
        Box::new(FlakyBackend {
            armed: Arc::new(AtomicBool::new(true)),
        }),
        None,
        config(),
        4,
        128,
    );

    let (tx, _rx) = tokio::sync::mpsc::channel(64);
    let seq_id = engine.add_request(Request::new(7, vec![1, 2, 3], 5), tx);

    // First step hits the transient failure and recovers the request.
    let result = engine.step();
    assert!(
        result.is_err(),
        "first step must surface the transient error"
    );
    assert_eq!(engine.scheduler.waiting_count(), 1, "request rolled back");
    assert_eq!(engine.scheduler.running_count(), 0);

    // Second step retries the SAME request and succeeds: it is now admitted
    // into running as Decoding (the prefill completed) instead of being stuck.
    let result = engine.step();
    assert!(
        result.is_ok(),
        "second step must retry the recovered request successfully"
    );
    let seq = engine
        .scheduler
        .get_sequence(seq_id)
        .expect("the retried sequence must be served this round");
    assert_eq!(
        seq.num_computed_tokens, 3,
        "the prefill must have completed on the retry"
    );
    assert_eq!(
        vllm_core::types::Status::Decoding,
        seq.status,
        "a completed prefill must transition to Decoding"
    );
}
