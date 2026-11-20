// Engine-loop panic guard.
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
// a step error: `error_count`++, orphan recovery), and the release
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
    // caught → requeue → retry).
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

/// A panic inside `model.embed` must not kill the engine loop.
///
/// Message handling runs outside the step `catch_unwind` guard, so a backend
/// unwinding in `embed` — foreign code, exactly the case the step guard
/// exists for — would unwind `run()` and terminate the engine thread,
/// closing every in-flight channel and stopping all further work.
///
/// The loop now survives the unwind. Note what it does NOT do: the model
/// lives behind a `std::sync::Mutex`, so the panic poisons it and later
/// calls fail with `LockPoisoned` rather than succeeding. Recovering a
/// poisoned model is a separate decision (the backend's internal state is
/// suspect after an unwind), so this test pins only the survival contract:
/// the loop thread stays alive and honours `Shutdown`.
#[test]
fn engine_loop_survives_a_panicking_embed() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::mpsc;
    use vllm_core::engine::Engine;
    use vllm_core::types::EngineMessage;
    use vllm_testing::StubModel;
    use vllm_traits::{ModelBackend, Result, SeqId, TokenId};

    /// Panics the first time `embed` is called.
    #[derive(Clone)]
    struct PanickingEmbed {
        inner: StubModel,
        calls: Arc<AtomicUsize>,
    }

    impl ModelBackend for PanickingEmbed {
        fn forward(
            &mut self,
            seq_ids: &[SeqId],
            input_tokens: &[Vec<TokenId>],
            positions: &[Vec<usize>],
            kv_block_ids: &[Vec<usize>],
            num_computed_tokens: &[usize],
            is_prefill: &[bool],
        ) -> Result<vllm_traits::BatchOutput> {
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
        ) -> Result<Vec<Vec<f32>>> {
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
            _input_tokens: &[Vec<TokenId>],
            _positions: &[Vec<usize>],
        ) -> Result<Vec<Vec<f32>>> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                panic!("backend embed exploded");
            }
            Ok(vec![vec![0.5_f32, 0.5]])
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

    let model = PanickingEmbed {
        inner: StubModel::returning(7),
        calls: Arc::new(AtomicUsize::new(0)),
    };
    let mut engine = Engine::new_boxed(Box::new(model), None::<Box<dyn ModelBackend>>);

    let (msg_tx, msg_rx) = mpsc::channel::<EngineMessage>(16);
    let loop_handle = std::thread::spawn(move || engine.run(msg_rx));
    std::thread::sleep(std::time::Duration::from_millis(50));

    // This request panics inside the backend.
    let (tx1, mut rx1) = mpsc::unbounded_channel::<Vec<Vec<f32>>>();
    msg_tx
        .blocking_send(EngineMessage::GetEmbeddings {
            input_tokens: vec![vec![1, 2]],
            response_tx: tx1,
        })
        .unwrap();
    // The request fails (dropped channel), which is the correct outcome.
    std::thread::sleep(std::time::Duration::from_millis(100));
    assert!(
        matches!(
            rx1.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected)
        ),
        "the panicking request must not produce a reply"
    );

    // The loop thread must still be running after the unwind.
    std::thread::sleep(std::time::Duration::from_millis(200));
    assert!(
        !loop_handle.is_finished(),
        "a backend panic must not terminate the engine loop"
    );

    // And it must still honour Shutdown (i.e. it is looping, not wedged).
    msg_tx.blocking_send(EngineMessage::Shutdown).unwrap();
    loop_handle
        .join()
        .expect("the engine loop thread must exit cleanly on Shutdown");
}

/// `GetEmbeddings` must actually deliver the vectors it computed.
///
/// The handler computes embeddings and sends them on the caller's channel. A
/// previous change that removed a debug print also removed the `send`, leaving
/// an empty `Ok(embeddings) => {}` arm — the RPC then silently produced
/// nothing and the HTTP layer answered 503 for every embeddings request. No
/// test covered the delivery, so it shipped.
#[test]
fn get_embeddings_delivers_the_vectors() {
    use tokio::sync::mpsc;
    use vllm_core::engine::Engine;
    use vllm_core::types::EngineMessage;
    use vllm_testing::StubModel;
    use vllm_traits::ModelBackend;

    let mut engine = Engine::new_boxed(
        Box::new(StubModel::returning(7)),
        None::<Box<dyn ModelBackend>>,
    );
    let (msg_tx, msg_rx) = mpsc::channel::<EngineMessage>(16);
    let loop_handle = std::thread::spawn(move || engine.run(msg_rx));
    std::thread::sleep(std::time::Duration::from_millis(50));

    let (tx, mut rx) = mpsc::unbounded_channel::<Vec<Vec<f32>>>();
    msg_tx
        .blocking_send(EngineMessage::GetEmbeddings {
            input_tokens: vec![vec![1, 2]],
            response_tx: tx,
        })
        .expect("engine mailbox");

    let delivered = rx
        .blocking_recv()
        .expect("the engine must send the computed embeddings back");
    assert_eq!(delivered.len(), 1, "one vector per input");

    msg_tx.blocking_send(EngineMessage::Shutdown).ok();
    loop_handle.join().expect("engine loop exits on Shutdown");
}
