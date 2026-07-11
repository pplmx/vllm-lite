# Tutorial 3: Run Inference

This tutorial runs inference through the engine **actor API** — the same
pattern used by `vllm-server` and validated in
`crates/server/tests/tutorial_e2e.rs`.

## The Request Lifecycle

```text
AddRequest → Engine::run loop → stream tokens → Shutdown
```

1. Spawn `Engine::run(msg_rx)` on a dedicated worker thread
2. Send `EngineMessage::AddRequest { request, response_tx }`
3. Receive generated tokens from `response_tx`
4. Send `EngineMessage::Shutdown` when done

The engine owns scheduling, batching, and model forward internally — callers
do not call `build_batch()` or `model.forward()` directly.

## Minimal Inference Loop

```rust,no_run
use tokio::sync::mpsc;
use vllm_core::engine::Engine;
use vllm_core::types::{EngineMessage, Request};
use vllm_traits::{StubModelBackend, TokenId};

# fn doc() -> Result<(), Box<dyn std::error::Error>> {
// 1. Build engine (StubModelBackend needs no checkpoint files)
let mut engine = Engine::new(StubModelBackend, None::<StubModelBackend>);
let (msg_tx, msg_rx) = mpsc::unbounded_channel::<EngineMessage>();
let (token_tx, mut token_rx) = mpsc::channel::<TokenId>(64);

// 2. Start the engine actor on a worker thread
let handle = std::thread::spawn(move || {
    engine.run(msg_rx);
});

// 3. Submit a request
let req = Request::new(1, vec![1, 2, 3], /*max_tokens=*/ 10);
msg_tx.send(EngineMessage::AddRequest {
    request: req,
    response_tx: token_tx,
})?;

// 4. Drain streamed tokens
let rt = tokio::runtime::Builder::new_current_thread()
    .enable_all()
    .build()?;
while let Some(token) = rt.block_on(token_rx.recv()) {
    println!("token: {token}");
}

// 5. Shut down cleanly
msg_tx.send(EngineMessage::Shutdown)?;
handle.join().expect("engine thread");
# Ok(())
# }
```

## With a Real Model

Replace `StubModelBackend` with a loaded `Box<dyn ModelBackend>`:

```rust,no_run
use candle_core::Device;
use vllm_model::loader::ModelLoader;

# fn doc() -> Result<(), Box<dyn std::error::Error>> {
let model = ModelLoader::builder(Device::Cpu)
    .with_model_dir("/path/to/checkpoint".into())
    .with_kv_blocks(1024)
    .build()?
    .load_model()?;

let mut engine = Engine::new_boxed(model, None);
# Ok(())
# }
```

## Phases (Internal)

Inside the engine step loop, requests move through:

- **Prefill** — process the prompt (possibly chunked for long contexts)
- **Decode** — generate one token per step
- **Finished** — EOS or `max_tokens` reached

You observe phases via `RUST_LOG=debug` scheduler logs, not via a public
`build_batch()` call.

## Concurrency

Submit multiple `AddRequest` messages before draining tokens. The scheduler
continuously batches prefill and decode steps — no manual batch management
required.

## Next Steps

→ [Tutorial 4: Customize](04-customize.md)
