# vLLM-lite (Rust) — Design Specification

## 1. Overview

This project implements a minimal yet principled LLM inference runtime in Rust, focusing on:

* Continuous batching
* KV cache reuse
* Scheduling under dynamic workloads

Non-goals:

* Training
* Distributed inference (multi-node)
* Full transformer reimplementation

---

## 2. System Architecture

```text
            +------------------+
            |     API Layer     |
            +--------+---------+
                     |
                     v
            +------------------+
            |    Scheduler      |
            +--------+---------+
                     |
      +--------------+--------------+
      v                             v
+------------+               +-------------+
| KV Cache   |               | Model Runner|
+------------+               +-------------+
      |                             |
      +--------------+--------------+
                     v
               +------------+
               |  Backend   |
               | (CPU/GPU)  |
               +------------+
```

---

## 3. Core Abstractions

### 3.1 Request / Sequence

```rust
type TokenId = u32;

struct Request {
    id: u64,
    prompt: Vec<TokenId>,
    max_tokens: usize,
}

struct Sequence {
    id: u64,
    tokens: Vec<TokenId>,
    kv_blocks: Vec<BlockId>,
    status: Status,
}

enum Status {
    Prefill,
    Decoding,
    Finished,
}
```

---

## 4. Execution Model

### 4.1 Main Loop

```rust
loop {
    intake_requests();

    let batch = scheduler.build_batch();

    let output = model.forward(batch);

    scheduler.update(output);

    scheduler.evict_finished();
}
```

---

## 5. Scheduler Design

### 5.1 Goals

* Maximize GPU utilization
* Minimize latency
* Avoid starvation

---

### 5.2 Batch Composition

```text
Batch = Prefill Sequences + Decode Sequences
```

Trade-off:

* Prefill → compute-bound
* Decode → memory-bound

---

### 5.3 Policy (Initial)

```text
- Always prioritize decode (low latency)
- Fill remaining capacity with prefill
```

---

### 5.4 Future Policies

* Token-level scheduling
* Deadline-aware (EDF)
* Throughput-optimized batching

---

## 6. KV Cache Design

### 6.1 Problem

Naive layout leads to:

* Memory fragmentation
* Poor reuse
* OOM under concurrency

---

### 6.2 Paged KV Cache

```rust
const BLOCK_SIZE: usize = 16;

struct KVCache {
    blocks: Vec<Block>,
    free_list: Vec<BlockId>,
}

struct Block {
    key: Tensor,
    value: Tensor,
}

type BlockId = usize;
```

---

### 6.3 Sequence Mapping

```text
Sequence KV:

[block_1][block_7][block_3]
```

---

### 6.4 Allocation Strategy

* Fixed-size blocks
* Free list reuse
* No compaction (initial)

---

## 7. Model Interface

```rust
trait Model {
    fn forward(&self, batch: Batch) -> BatchOutput;
}
```

---

### 7.1 Batch

```rust
struct Batch {
    input_ids: Tensor,
    positions: Tensor,
    kv_cache_map: Vec<Vec<BlockId>>,
}
```

---

### 7.2 Output

```rust
struct BatchOutput {
    logits: Tensor,
}
```

---

## 8. Attention (Simplified)

Core computation:

Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V

MVP:

* Gather KV blocks
* Concatenate
* Run attention

Future:

* Block-wise (paged) attention

---

## 9. Memory Management

### 9.1 Requirements

* Fast allocation
* Low fragmentation
* Predictable latency

---

### 9.2 Initial Design

```rust
struct BlockAllocator {
    free: Vec<BlockId>,
}
```

---

### 9.3 Future Improvements

* Slab allocator
* Buddy allocator
* GPU memory pooling

---

## 10. Backend Integration

### 10.1 Options

* ONNX Runtime (recommended)
* Candle (pure Rust)
* GGUF / llama.cpp bindings

---

### 10.2 Key Constraint

KV cache must be:

```text
Managed by runtime, not backend
```

---

## 11. Concurrency Model

### 11.1 Architecture

```text
API thread → channel → scheduler → worker loop
```

---

### 11.2 Implementation

* tokio runtime
* mpsc channels
* single GPU worker thread

---

### 11.3 Rationale

* Avoid GPU contention
* Deterministic scheduling

---

## 12. Token Generation Pipeline

### 12.1 Prefill

```text
prompt → full forward → KV cache fill
```

---

### 12.2 Decode

```text
loop:
  last_token → forward
  → append KV
  → sample next token
```

---

## 13. Sampling

Initial:

* greedy
* top-k

Future:

* top-p
* temperature

---

## 14. Observability

### 14.1 Metrics

* tokens/sec
* latency (p50, p99)
* KV cache usage
* batch size

---

### 14.2 Debug Signals

* prefill vs decode time
* cache hit ratio
* allocation failures

---

## 15. Roadmap

### Phase 1

* single request
* no batching
* contiguous KV

---

### Phase 2

* multi-request
* basic batching

---

### Phase 3

* continuous batching

---

### Phase 4

* paged KV cache

---

### Phase 5

* paged attention
* speculative decoding

---

## 16. Risks & Failure Modes

### 16.1 Scheduler Complexity

* starvation
* unfairness

---

### 16.2 KV Cache Bugs

* silent corruption
* wrong attention results

---

### 16.3 Memory Issues

* fragmentation
* OOM under burst traffic

---

## 17. Success Criteria

Functional:

* produces valid text output

System-level:

* decode faster than prefill
* throughput increases with batching

Architectural:

* KV reuse works
* scheduler stabilizes under load

---

## 18. Future Extensions

* multi-GPU
* distributed inference
* prefix caching
* speculative decoding
* quantization support
