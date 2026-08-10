//! Engine-loop panic guard (RIL TASK-050, ISS-046).
//!
//! The engine runs on a single dedicated thread. A panic inside the target
//! model's forward (a GPU kernel fault, an `assert!` in an architecture
//! impl, an aborted candle op) previously unwound `run()` entirely — killing
//! the whole server. The speculative *draft* path already wrapped its
//! forwards in `catch_unwind`; the canonical target-forward step was
//! unprotected, and `[profile.release]` set `panic = "abort"` so even the
//! draft guard was a no-op in production.
//!
//! `Engine::run` now wraps the step in `catch_unwind` (converting a panic to
//! a step error: `error_count`++, ISS-045 orphan recovery), and the release
//! profile unwinds so the guard is effective in production.
//!
//! This test drives the real `run()` loop on a thread with a backend that
//! panics on every forward call and asserts the loop survives (keeps
//! retrying), then stops cleanly on `Shutdown`.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use vllm_core::Engine;
use vllm_core::types::{EngineMessage, Request, SchedulerConfig};
use vllm_traits::{BatchOutput, ModelBackend, Result as ModelResult, SeqId, TokenId};

/// Backend whose every `forward_logits` call panics — models a permanently
/// crashing GPU kernel. `attempts` counts how many times the loop retried.
struct PanicBackend {
    attempts: Arc<AtomicU64>,
}

impl ModelBackend for PanicBackend {
    fn forward(
        &mut self,
        _seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> ModelResult<BatchOutput> {
        panic!("model kernel fault")
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
        self.attempts.fetch_add(1, Ordering::Relaxed);
        panic!("model kernel fault")
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

#[test]
fn model_forward_panic_does_not_kill_the_engine_loop() {
    let attempts = Arc::new(AtomicU64::new(0));
    let engine = Engine::with_config_boxed(
        Box::new(PanicBackend {
            attempts: attempts.clone(),
        }),
        None,
        SchedulerConfig::default(),
        4,
        128,
    );

    let (tx, rx) = tokio::sync::mpsc::channel(16);
    let handle = std::thread::spawn(move || {
        let mut engine = engine;
        engine.run(rx);
    });

    // Add a generation request; the loop drains it and steps (panic →
    // caught → ISS-045 requeue → retry).
    let (response_tx, _response_rx) = tokio::sync::mpsc::channel(64);
    tx.blocking_send(EngineMessage::AddRequest {
        request: Box::new(Request::new(7, vec![1, 2, 3], 5)),
        response_tx,
        seq_id_tx: None,
        finish_reason_tx: None,
        request_id: None,
    })
    .expect("engine mailbox open");

    // Give the loop time to hit the forward + catch the panic. If the panic
    // guard is missing, the unwind kills the spawned thread on the first
    // attempt (attempts == 1, thread finished) and join() below fails.
    // Note the forward may only be reached once: the panic poisons the
    // model's std::sync::Mutex, so subsequent steps surface LockPoisoned
    // (correctly — we don't silently trust a mutated model) while the loop
    // stays alive with backoff.
    std::thread::sleep(Duration::from_millis(250));
    let attempts_seen = attempts.load(Ordering::Relaxed);
    assert!(
        attempts_seen >= 1,
        "the engine must have reached the panicking forward (got {attempts_seen})"
    );
    assert!(
        !handle.is_finished(),
        "run() must not have died from the model panic"
    );

    // The loop is still responsive and can be stopped cleanly — it did not
    // unwind on the panic.
    tx.blocking_send(EngineMessage::Shutdown)
        .expect("mailbox open");
    handle
        .join()
        .expect("run() must return cleanly after Shutdown (panic guard held)");
}
