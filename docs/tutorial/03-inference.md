# Tutorial 3: Run Inference

This tutorial runs a single inference pass through the engine.

## The Request Lifecycle

```text
add_request → build_batch → model.forward → update_state → repeat
```

1. **`engine.add_request(req)`** — enqueue a generation request
2. **`engine.build_batch()`** — group requests into a batch (decode or prefill)
3. **`model.forward(&batch)`** — run the model on the batch, get output tokens
4. **`engine.update(&seq_ids, &tokens, &input_counts)`** — advance request state

## Minimal Inference Loop

```rust,no_run
use vllm_core::engine::Engine;
use vllm_core::types::Request;
use vllm_model::llama::LlamaModel;
use std::sync::Arc;

# async fn doc() -> Result<(), Box<dyn std::error::Error>> {
// 1. Load model (Tutorial 2)
let device = candle_core::Device::Cpu;
let model = LlamaModel::from_path("path/to/model", &device)?;
let model = Arc::new(model);

// 2. Build engine
let engine = Engine::new(model, None);

// 3. Add request
let req = Request::new(/*seq_id=*/ 1, /*tokens=*/ vec![1, 2, 3], /*max_tokens=*/ 10);
engine.add_request(req);

// 4. Inference loop
for step in 0..10 {
    let batch = engine.build_batch();
    if batch.is_empty() { break; }
    let output = engine.model().forward(&batch).await?;
    engine.update(&batch.seq_ids, &output.tokens, &output.input_counts);
}

// 5. Inspect output
for (seq_id, request) in engine.active_requests() {
    println!("seq {}: {:?}", seq_id, request.tokens());
}
# Ok(())
# }
```

## Phases

Requests transition through phases:

- **Prefill** — initial prompt processing (long, parallel)
- **Decode** — token-by-token generation (short, sequential)
- **Finished** — request complete (EOS reached or max_tokens)

`engine.build_batch()` returns a `Batch` with a phase indicator.

## Concurrency

vllm-lite supports concurrent requests via continuous batching: while one
request is in prefill, others can be in decode. The scheduler (`SchedulerEngine`)
handles this automatically — you just keep calling `build_batch()` in a loop.

## Next Steps

→ [Tutorial 4: Customize](04-customize.md)
