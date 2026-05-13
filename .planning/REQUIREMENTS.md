# Requirements: vllm-lite

**Defined:** 2026-05-13
**Core Value:** Fast, memory-efficient LLM inference with continuous batching, paged KV cache, and tensor parallelism

## v17.0 Requirements

Requirements for milestone v17.0: Production Speculative Decoding.

### Engine Integration

- [ ] **ENG-01**: Engine executes `step_speculative` as unified entry point for speculative decode
- [ ] **ENG-02**: Draft tokens are generated via batched per-position forward passes across all sequences
- [ ] **ENG-03**: Token verification uses logit-based rejection (not exact match)
- [ ] **ENG-04**: Rejected draft tokens' KV cache entries are rolled back via MemoryManager
- [ ] **ENG-05**: Speculative and non-speculative paths fall back gracefully on error
- [ ] **ENG-06**: Scheduler correctly tracks input token counts for multi-token draft acceptance

### Self-Speculation

- [ ] **SELF-01**: `SelfSpeculativeModel` implements greedy (argmax) draft generation via layer-truncated forward pass
- [ ] **SELF-02**: Draft model uses 1/8 target model layers with weight sharing (zero-copy references)
- [ ] **SELF-03**: Draft and target maintain separate KV cache isolation to prevent state corruption

### Adaptive Depth

- [ ] **ADPT-01**: `AdaptiveSpeculativeDecoder` is wired into the speculative decode loop
- [ ] **ADPT-02**: Draft depth adjusts dynamically based on real-time acceptance rates
- [ ] **ADPT-03**: Acceptance rate monitoring uses EWMA smoothing with deadband hysteresis

### Benchmarks

- [ ] **BENCH-01**: Throughput and latency benchmarks compare speculative vs non-speculative paths
- [ ] **BENCH-02**: Metrics include P50/P95/P99 latency and tokens/sec throughput
- [ ] **BENCH-03**: Benchmark methodology includes proper warmup and multi-sequence workloads
- [ ] **BENCH-04**: Results are reported for at least one target model architecture (e.g., Llama)

### Speculative Warmup

- [ ] **WARM-01**: Draft model KV cache is populated during/after target prefill
- [ ] **WARM-02**: Warmup ensures first speculative decode step has valid draft KV cache

### Metrics

- [ ] **MTRC-01**: Acceptance rate tracked per-request and aggregated across the batch
- [ ] **MTRC-02**: Speculative efficiency (draft tokens / total tokens) reported
- [ ] **MTRC-03**: Throughput speedup ratio vs non-speculative baseline reported in metrics

## v18.0 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Multi-Model

- **MULTI-01**: External draft model support (smaller model as drafter)
- **MULTI-02**: Draft model lifecycle management (load/unload/swap)
- **MULTI-03**: GPU memory budgeting for concurrent target + draft models

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature                             | Reason                                                                 |
| ----------------------------------- | ---------------------------------------------------------------------- |
| Tree-based speculation (draft tree) | Sigmoidally more complex, linear draft is sufficient                   |
| Medusa-style multiple heads         | Requires custom model training, incompatible with off-the-shelf models |
| Speculative decoding for prefill    | Prefill is compute-bound, speculative decode only                      |
| Dynamic model switching mid-request | Complex state management, low ROI                                      |
| External draft model (multi-model)  | Deferred to v18.0                                                      |

## Traceability

| Requirement | Phase      | Status  |
| ----------- | ---------- | ------- |
| ENG-01      | Phase 17.1 | Pending |
| ENG-02      | Phase 17.1 | Pending |
| ENG-03      | Phase 17.1 | Pending |
| ENG-04      | Phase 17.1 | Pending |
| ENG-05      | Phase 17.1 | Pending |
| ENG-06      | Phase 17.1 | Pending |
| SELF-01     | Phase 17.2 | Pending |
| SELF-02     | Phase 17.2 | Pending |
| SELF-03     | Phase 17.2 | Pending |
| ADPT-01     | Phase 17.3 | Pending |
| ADPT-02     | Phase 17.3 | Pending |
| ADPT-03     | Phase 17.3 | Pending |
| BENCH-01    | Phase 17.3 | Pending |
| BENCH-02    | Phase 17.3 | Pending |
| BENCH-03    | Phase 17.3 | Pending |
| BENCH-04    | Phase 17.3 | Pending |
| WARM-01     | Phase 17.4 | Pending |
| WARM-02     | Phase 17.4 | Pending |
| MTRC-01     | Phase 17.4 | Pending |
| MTRC-02     | Phase 17.4 | Pending |
| MTRC-03     | Phase 17.4 | Pending |

**Coverage:**

- v17.0 requirements: 21 total
- Mapped to phases: 21
- Unmapped: 0

---

*Requirements defined: 2026-05-13*
*Last updated: 2026-05-13 after initial definition*
