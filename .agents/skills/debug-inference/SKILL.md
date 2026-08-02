---
name: debug-inference
description: >
  Systematic debugging workflow for vllm-lite inference issues using the
  structured logging system and metrics. Use when the user reports incorrect
  model output, slow inference, scheduling bugs, KV cache issues, request
  hangs, speculative decoding problems, or asks to "debug", "diagnose", or
  "trace" an inference problem. Covers log-level selection, symptom-to-component
  mapping, metrics analysis, and the reproduce-isolate-fix loop.
---

# Debug Inference Issues

## 1. Reproduce with logging

```bash
# Debug: scheduling, batching, engine flow
RUST_LOG=debug cargo run -p vllm-server

# Trace: token generation, KV cache, attention internals
RUST_LOG=trace cargo run -p vllm-server

# JSON file output for post-analysis
RUST_LOG=trace cargo run -p vllm-server -- --log-dir ./logs

# Filter to specific component
RUST_LOG=vllm_core::scheduler=trace,vllm_model=debug cargo run -p vllm-server
```

Post-analyze JSON logs:

```bash
# Find all events for a request
jq 'select(.fields.request_id == "req-42")' logs/*.json

# Slowest operations
jq -s 'sort_by(.fields.duration_ms) | reverse | .[:10]' logs/*.json
```

## 2. Symptom → component map

| Symptom                   | Component                   | Log target                                        | Level       |
| ------------------------- | --------------------------- | ------------------------------------------------- | ----------- |
| Request never completes   | Scheduler queue             | `vllm_core::scheduler`                            | debug       |
| Wrong tokens generated    | Sampling                    | `vllm_core::sampling`                             | trace       |
| KV cache corruption       | Paged tensor / prefix cache | `vllm_model::paged_tensor`, `vllm_core::kv_cache` | trace       |
| Slow first-token latency  | Prefill / attention         | `vllm_model::components::attention`               | trace       |
| Slow per-token throughput | Batch composition           | `vllm_core::scheduler::batch_composer`            | debug       |
| Model load failure        | Loader / checkpoint         | `vllm_model::loader`                              | debug       |
| CUDA Graph fallback       | Kernel layer                | `vllm_model::kernels::cuda_graph`                 | warn        |
| OOM / block exhaustion    | Block allocator             | `vllm_core::scheduler::memory`                    | debug/trace |
| Draft tokens rejected     | Speculative decoding        | `vllm_core::speculative`                          | debug       |
| High draft overhead       | Adaptive draft tuning       | `vllm_core::speculative::adaptive`                | debug       |
| Circuit breaker trips     | Backend health              | `vllm_core::circuit_breaker`                      | warn        |
| HTTP 429 / auth errors    | Server middleware           | `vllm_server::security`, `vllm_server::auth`      | debug       |

## 3. Key log fields

Standard fields for correlation across all components:

- `request_id` — track a single request end-to-end
- `seq_id` — sequence ID within the scheduler
- `batch_size` — number of sequences in the batch
- `phase` — Prefill or Decode
- `duration_ms` — operation timing
- `prompt_tokens` / `output_tokens` — token counts

## 4. Metrics (Prometheus)

The server exposes metrics via `EnhancedMetricsCollector`:

```bash
# Scrape metrics endpoint (default port)
curl http://localhost:8000/metrics | grep vllm
```

Key metrics: request latency histogram, tokens/sec, queue depth,
cache hit rate, draft acceptance rate, block utilization.

## 5. Isolate with targeted tests

```bash
cargo test -p vllm-core -- scheduler --nocapture
cargo test -p vllm-core -- kv_cache --nocapture
cargo test -p vllm-core -- sampling --nocapture
cargo test -p vllm-core -- engine --nocapture
cargo test -p vllm-core -- speculative --nocapture
cargo test -p vllm-model -- paged_tensor --nocapture
cargo test -p vllm-model -- attention --nocapture
```

## 6. Common issues

### Request stuck in waiting queue

- `RUST_LOG=vllm_core::scheduler::memory=trace`
- Look for "no free blocks"; verify `max_num_seqs` and `num_kv_blocks` config

### Incorrect output tokens

- `RUST_LOG=vllm_core::sampling=trace`
- Check temperature / top_p / top_k; compare logits with `--nocapture`

### Prefix cache miss (slow repeated prompts)

- `RUST_LOG=vllm_core::kv_cache::prefix_cache=trace`
- Look for hit/miss entries; verify block hash consistency

### Speculative decoding low acceptance

- `RUST_LOG=vllm_core::speculative=debug`
- Check draft model quality; adaptive tuner may reduce draft length
- Verify `SpeculationConfig::acceptance_threshold`

### CUDA Graph disabled

- Look for `warn!("CUDA Graph disabled, falling back")`
- Check GPU memory; verify batch size within graph capture range

## 7. Minimal reproduction test

```rust
#[test]
fn test_<component>_<bug_description>() {
    // Arrange
    let config = SchedulerConfig::default();
    let mut engine = SchedulerEngine::default();

    // Act
    let id = engine.add_request(Request::new(0, vec![1, 2, 3], 10));
    let batch = engine.build_batch();

    // Assert
    assert_eq!(batch.seq_ids.len(), 1);
}
```

For integration-level reproduction, use `vllm_testing::TestHarness`:

```rust
use vllm_testing::{TestHarness, TestHarnessConfig};

let harness = TestHarness::new(TestHarnessConfig::default());
let seq_id = harness.add_request(vec![1, 2, 3], 10);
let output = harness.run_to_completion(seq_id);
```
