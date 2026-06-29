// crates/server/tests/tutorial_e2e.rs
//
// Mirrors docs/tutorial/03-inference.md — minimal engine lifecycle test.
// This test uses `StubModelBackend` from `vllm-traits` (always-on dev-friendly
// stub) instead of a real model, so it runs without any checkpoint files and
// without GPU. The Engine is driven via its public actor API:
// `Engine::run(mpsc::Receiver<EngineMessage>)`.

use tokio::sync::mpsc;
use vllm_core::engine::Engine;
use vllm_core::types::EngineMessage;
use vllm_core::types::Request;
use vllm_traits::StubModelBackend;

#[test]
fn test_tutorial_request_construction() {
    // Confirm `Request::new` accepts the (id, prompt, max_tokens) shape
    // shown in Tutorial 3 ("Add request" step).
    let req = Request::new(1, vec![1, 2, 3], 5);
    assert_eq!(req.id, 1);
    assert_eq!(req.prompt, vec![1, 2, 3]);
    assert_eq!(req.max_tokens, 5);
}

#[test]
fn test_tutorial_engine_lifecycle() {
    // Drive the Engine through its public actor API: spawn a thread, send
    // an `AddRequest` message, drain tokens from the per-request channel,
    // then send `Shutdown`. This is what the vllm-server main loop does.
    let mut engine = Engine::new(StubModelBackend, None::<StubModelBackend>);
    let (msg_tx, msg_rx) = mpsc::unbounded_channel::<EngineMessage>();
    let (token_tx, mut token_rx) = mpsc::channel::<vllm_traits::TokenId>(64);

    let handle = std::thread::spawn(move || {
        engine.run(msg_rx);
    });

    let req = Request::new(1, vec![1, 2, 3], 3);
    msg_tx
        .send(EngineMessage::AddRequest {
            request: req,
            response_tx: token_tx,
        })
        .expect("engine mailbox should be open");

    // Collect at least one token; the StubModelBackend returns token 0 per step.
    let token = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime")
        .block_on(async { token_rx.recv().await })
        .expect("expected at least one token from StubModelBackend");
    assert_eq!(token, 0);

    msg_tx.send(EngineMessage::Shutdown).expect("send Shutdown");
    handle.join().expect("engine thread should exit cleanly");
}
