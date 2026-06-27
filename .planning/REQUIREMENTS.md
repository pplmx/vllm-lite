# Requirements: vllm-lite

**Defined:** 2026-05-13
**Core Value:** Fast, memory-efficient LLM inference with continuous batching, paged KV cache, and tensor parallelism

## v18.0 Requirements (Active)

Requirements for milestone v18.0: Multi-Model Speculative Decoding.
Carries forward v17 deferred MULTI-01/02/03 plus new RTE/FALL items.

### Multi-Model Loading (MMLT)

- [ ] **MMLT-01**: Engine can load a separate draft model instance (different architecture and/or size from target)
- [ ] **MMLT-02**: External draft uses its own `ModelBackend` instance with independent KV cache block IDs (no state leak with target)
- [ ] **MMLT-03**: Draft weights loaded lazily — deferred until first request selects this draft model

### Lifecycle Management (LIFE)

- [ ] **LIFE-01**: `DraftModelRegistry` provides register / load / unload operations for draft models at runtime
- [ ] **LIFE-02**: Unloading a draft model frees its KV cache blocks via `MemoryManager` (no orphan blocks)
- [ ] **LIFE-03**: Registry tracks active drafts with reference counts; auto-unload when refcount hits zero

### Memory Budget (MEM)

- [ ] **MEM-01**: Engine enforces total VRAM budget = target weights + target KV cache + N concurrent drafts
- [ ] **MEM-02**: Load-time weight-size estimation (from model loader metadata); runtime KV cache growth tracking
- [ ] **MEM-03**: Engine refuses to load a draft if estimated VRAM would exceed budget, with structured error

### Request Routing (RTE)

- [ ] **RTE-01**: Request can specify `draft_model_id` via `SamplingParams` or `Request` struct
- [ ] **RTE-02**: Scheduler routes request to correct draft model instance during batch composition
- [ ] **RTE-03**: Multiple drafts can coexist in the same batch (mixed draft routing across requests)

### Fallback Semantics (FALL)

- [ ] **FALL-01**: External draft load failure → automatic fallback to self-spec path (v17 baseline)
- [ ] **FALL-02**: Runtime draft inference error → graceful degradation to non-speculative decode for that request

## v17.0 Validated

Shipped in v17.0 (2026-06-26). All requirements complete; reference for traceability.

### Engine Integration

- ✓ **ENG-01**: Engine executes `step_speculative` as unified entry point for speculative decode
- ✓ **ENG-02**: Draft tokens generated via batched per-position forward passes across all sequences
- ✓ **ENG-03**: Token verification uses logit-based rejection (not exact match)
- ✓ **ENG-04**: Rejected draft tokens' KV cache entries rolled back via MemoryManager
- ✓ **ENG-05**: Speculative and non-speculative paths fall back gracefully on error
- ✓ **ENG-06**: Scheduler correctly tracks input token counts for multi-token draft acceptance

### Self-Speculation

- ✓ **SELF-01**: `SelfSpeculativeModel` implements greedy (argmax) draft generation via layer-truncated forward pass
- ✓ **SELF-02**: Draft model uses 1/8 target model layers with weight sharing (zero-copy references)
- ✓ **SELF-03**: Draft and target maintain separate KV cache isolation

### Adaptive Depth

- ✓ **ADPT-01**: `AdaptiveSpeculativeDecoder` wired into the speculative decode loop
- ✓ **ADPT-02**: Draft depth adjusts dynamically based on real-time acceptance rates
- ✓ **ADPT-03**: Acceptance rate monitoring uses EWMA smoothing with deadband hysteresis

### Benchmarks

- ✓ **BENCH-01**: Throughput and latency benchmarks compare speculative vs non-speculative paths
- ✓ **BENCH-02**: Metrics include P50/P95/P99 latency and tokens/sec throughput
- ✓ **BENCH-03**: Benchmark methodology includes proper warmup and multi-sequence workloads
- ✓ **BENCH-04**: Results reported for at least one target model architecture (e.g., Llama)

### Speculative Warmup

- ✓ **WARM-01**: Draft model KV cache populated during/after target prefill
- ✓ **WARM-02**: Warmup ensures first speculative decode step has valid draft KV cache

### Metrics

- ✓ **MTRC-01**: Acceptance rate tracked per-request and aggregated across the batch
- ✓ **MTRC-02**: Speculative efficiency (draft tokens / total tokens) reported
- ✓ **MTRC-03**: Throughput speedup ratio vs non-speculative baseline reported

## v19.0+ Requirements (Deferred)

Tracked but not in current roadmap. Promotion requires roadmap update.

### Multi-Model Extensions

- **MULTI-04**: Hot-swap of draft model during long-running request (state migration)
- **MULTI-05**: Draft model training / fine-tuning hooks (currently out of engine scope)
- **MULTI-06**: Cross-GPU draft model placement (draft on GPU 0, target on GPU 1)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature                             | Reason                                                                 |
| ----------------------------------- | ---------------------------------------------------------------------- |
| Tree-based speculation (draft tree) | Sigmoidally more complex, linear draft is sufficient                   |
| Medusa-style multiple heads         | Requires custom model training, incompatible with off-the-shelf models |
| Speculative decoding for prefill    | Prefill is compute-bound, speculative decode only                      |
| Dynamic model switching mid-request | Complex state management, low ROI                                      |
| Draft model retraining              | Engine-only scope; training belongs to a separate training service     |

## Traceability

Will be populated by `gsd-roadmapper` after roadmap approval.

| Requirement | Phase | Status  |
| ----------- | ----- | ------- |
| MMLT-01     | TBD   | Pending |
| MMLT-02     | TBD   | Pending |
| MMLT-03     | TBD   | Pending |
| LIFE-01     | TBD   | Pending |
| LIFE-02     | TBD   | Pending |
| LIFE-03     | TBD   | Pending |
| MEM-01      | TBD   | Pending |
| MEM-02      | TBD   | Pending |
| MEM-03      | TBD   | Pending |
| RTE-01      | TBD   | Pending |
| RTE-02      | TBD   | Pending |
| RTE-03      | TBD   | Pending |
| FALL-01     | TBD   | Pending |
| FALL-02     | TBD   | Pending |

**Coverage:**

- v18.0 requirements: 14 total
- Mapped to phases: 0 (TBD — pending roadmap creation)
- Unmapped: 14 ⚠️

---

*Requirements defined: 2026-05-13*
*Last updated: 2026-06-27 — v18.0 active milestone; v17.0 requirements moved to Validated*
