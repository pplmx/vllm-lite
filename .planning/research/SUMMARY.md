# Project Research Summary

**Project:** vLLM-lite — v17.0 Production Speculative Decoding
**Domain:** LLM Inference Engine — Speculative Decoding
**Researched:** 2026-05-13
**Confidence:** HIGH

## Executive Summary

vLLM-lite is a Rust-based LLM inference engine using Candle for GPU-accelerated model execution. The v17.0 milestone completes the speculative decoding pipeline that was architecturally laid out in v16.0 (DraftVerifier, SelfSpeculativeModel, adaptive tracking) but never integrated into the engine's main inference loop. Speculative decoding accelerates token generation by using a lightweight "draft" model to propose multiple tokens per step, then verifying them against the full "target" model — producing 2-3x throughput improvement on GPU-bound workloads without compromising output quality.

The research is clear: the top priority is fixing and wiring `step_speculative()` into the engine. The scaffold exists but has critical bugs: per-sequence draft generation (not batched), scheduler state desynchronization, shared KV cache between draft and target (causing silent corruption), and missing correct `is_prefill` mode for the verification pass. The `AdaptiveSpeculativeDecoder` is one of the few correct components — it just needs to be connected. The `SelfSpeculativeModel.generate_draft()` is a stub that needs its actual layer-truncated forward pass. No new dependencies are required; everything stays in the existing Rust + Candle stack.

The key risk is silent correctness bugs from shared KV cache references between draft and target model passes. If the draft forward modifies KV blocks that the verification pass reads, output corrupts without any error or crash. The mitigation is strict separation of KV block IDs for draft vs target during the same step. A secondary risk is benchmark contamination — cold-start speculative decode looks artificially slow unless proper warmup phases are run. The recommended approach is phased: (1) fix engine integration with batched draft generation, (2) implement the actual self-speculation forward pass, (3) wire adaptive depth and run benchmarks, (4) add multi-model support, (5) production hardening.

## Key Findings

### Recommended Stack

No new dependencies required. All speculative decoding additions use the existing Rust + Candle stack. The project already has an `AdaptiveSpeculativeDecoder`, `DraftVerifier`, `SelfSpeculativeModel`, `BenchmarkSuite`, `MetricsCollector`, and `PercentileStats` infrastructure — all from v16.0. The work is purely integration and implementation.

**Core technologies (existing, to be extended):**

- **Rust + Candle**: Core inference engine — model forward passes for both draft and target use the same `ModelBackend::forward()` interface
- **`criterion` (bench)**: Already in `benches/` Cargo.toml — end-to-end benchmark suite uses `ThroughputBenchmark` / `LatencyBenchmark` (not microbenchmarks)
- **`metrics` / `metrics-exporter-prometheus`**: Add three spec decode counters: draft_count, accepted_count, rejection_rate
- **`serde` / `serde_json`**: Config serialization for speculative decode parameters (max_draft_tokens, adaptive_depth, rejection_strategy)
- **`tracing`**: Speculative step tracing at DEBUG/TRACE levels — add structured log fields for draft_count/accepted/rejected
- **`tokio`**: Async runtime for benchmark runners and the HTTP server path

**Key design decisions (from research):**

- **Self-speculation over separate draft model**: Zero additional GPU memory via weight sharing (1/8 layer trunk). Separate draft model (multi-model) is deferred to v18.0.
- **EWMA + deadband over PID controller**: A full PID is overkill for a single-variable system with noisy measurements. EWMA smoothing + ±5% deadband + 10-step cooldown has fewer tuning parameters.
- **Prometheus counters over OpenTelemetry**: Prometheus is already integrated. Three additional counters are trivially addable.

**See:** [STACK.md](./STACK.md) for full details.

### Expected Features

**Must have (table stakes) — Phase A engine integration:**

- **Batched draft generation** — the #1 bottleneck fix. Current per-sequence loop does O(seqs × max_draft) sequential forward passes. Must batch across ALL sequences so each iteration generates 1 draft token per sequence. This is the highest-complexity item.
- **Logit-based token verification** — must call `forward_logits()` not just `forward()` for probability comparison. Exact token matching is too aggressive for non-greedy decoding.
- **Correct `is_prefill` for verification** — the concatenated [input + draft] sequence must use decode mode with logits from all positions. Using prefill mode is O(n²) incorrect.
- **KV cache rollback for rejected tokens** — `rollback_blocks()` in MemoryManager to prevent rejected draft KV entries from leaking.
- **Unified speculative step method** — merge `step_speculative()` and `step_adaptive_speculative()` into a single `step_speculative(max_draft)` with a parameter. Currently three code paths that will diverge.
- **Graceful fallback to non-speculative** — if draft model forward fails, engine must keep working with standard decode.
- **Acceptance rate metrics** — `DraftAccuracyTracker` already exists, just needs to be wired into the metrics pipeline.

**Should have (differentiators) — Phase B/C:**

- **Adaptive draft depth** — `AdaptiveSpeculativeDecoder` is already fully implemented! Sliding window tracking, threshold adjustment. Just needs to be wired into the step loop.
- **Self-speculation with weight sharing** — implemented but `generate_draft()` is a stub. Needs actual layer-truncated forward pass (1/8 layers, weight-shared).
- **Speculative warmup** — prefill draft KV cache during target prefill to avoid cold-start garbage on first step.
- **Token-level + block-level rejection strategies** — already implemented in `RejectionStrategy` enum.
- **Benchmark suite for spec vs non-spec** — A/B comparison with P50/P95/P99 latency and throughput.

**Defer (v18.0+):**

- **Multi-model speculation** — separate small draft model. High ROI but doubles GPU memory, needs lifecycle management, tokenizer validation, speculative warmup. Only valuable if self-speculation underperforms.
- **Production hardening** — graceful fallback is nice but the engine already has error handling.
- **Tree-based speculation (draft tree)** — sigmoidally more complex. Linear draft is good enough.

**See:** [FEATURES.md](./FEATURES.md) for full dependency graph and anti-features.

### Architecture Approach

The target architecture unifies all decode paths through a single `step()` entry point. `step()` checks `speculative_mode` and either dispatches to `step_speculative_inner()` (batch-aware, with draft generation + verification + metrics) or the standard decode path. This prevents code path divergence — the cardinal sin of speculative decode engine integration.

**Major components:**

1. **Engine** (`engine.rs`) — Main inference loop. Dispatches to spec or non-spec paths. Manages speculative warmup, metrics recording, and fallback. Target: single unified `step()` method.
2. **SchedulerEngine** — Batch construction, position tracking, KV block allocation, prefix caching. Must support `update_with_accepted()` to handle variable-length token acceptance (not fixed 1 token/step).
3. **DraftVerifier** — Verifies draft tokens against target output. Implements TokenLevel rejection (exact match for greedy, probability-based for sampling). Already built.
4. **SelfSpeculativeModel** — Wraps target model with 1/8 layer subset. `generate_draft()` currently a stub — needs actual forward pass through truncated layers. Must NOT share KV block IDs with target (separate scratch blocks).
5. **AdaptiveSpeculativeDecoder** — Tracks acceptance rate via sliding window. Adjusts `max_draft_tokens` with EWMA smoothing + deadband. Fully implemented, needs wiring.
6. **SpecDecodeMetrics** — Prometheus counters for draft/accepted/rejected metrics. Three counters, wired into existing `MetricsCollector`.
7. **BenchmarkSuite** (benchmarks) — Orchestrates A/B comparison. Must include separate spec warmup phase before measurement.
8. **DraftModelLifecycle** (multi-model, Phase D) — Load/unload/validate external draft models. Memory estimation at load. Strict `vocab_size` validation.

**Key data flow:** `build_batch()` → `generate_draft_batched()` (batch-aware, not per-seq loop) → `verify_draft_batched()` (one target forward on extended inputs) → `scheduler.update_with_accepted()` → `adaptive_decoder.record_verification()` → metrics recording.

**Scalability insight from research:** Speculative decoding helps MOST when GPU is underutilized (low batch sizes). At >32 concurrent requests, GPU is saturated and speculation adds overhead. Consider auto-disabling at high batch sizes.

**See:** [ARCHITECTURE.md](./ARCHITECTURE.md) for full data flow diagrams and anti-patterns.

### Critical Pitfalls

1. **Engine Loop Divergence** — Three code paths (`step()`, `step_speculative()`, `step_adaptive_speculative()`) that rot independently. Mitigation: refactor to a single unified `step()` with speculative decoding as a modifier, not a replacement. Detection: run same request through both paths with `speculative_mode = false` and assert output tokens match exactly. **This is a CI gate requirement.**

2. **Draft KV Cache Corruption Through Shared References** — Self-speculation shares weights. If KV block IDs are shared between draft and target passes, the draft forward corrupts blocks the verification pass reads. **Silent wrong output — no crash.** Mitigation: allocate separate scratch KV blocks for draft output that are NOT shared with target. Detection: compare KV cache block contents before and after draft forward.

3. **Adaptive Depth Oscillation** — The current `maybe_adjust()` uses simple threshold ±0.1 with single cooldown. Overcorrects: high drafts → lower acceptance → decrease drafts → higher acceptance → increase. Mitigation: EWMA smoothing for acceptance rate, PID-style proportional control, deadband with hysteresis. Log adjustments less frequently.

4. **Benchmark Contamination from Warmup Effects** — First spec decode step is always slower (cold KV cache). Non-spec vs spec comparison is invalid without separate spec warmup phase. Mitigation: require explicit "speculative warmup" before measurement (run N tokens, discard). Run spec and non-spec in the same process. Offer `--spec-warmup-tokens` flag. Detection: run twice with different warmup — if results differ, warmup is insufficient.

5. **Token Matching Rejection Is Too Aggressive** — Current `verify_draft_tokens()` uses exact token matching. This is deterministic (token must match) but the theoretical spec decode algorithm uses probability-based rejection sampling. With temperature > 0, valid alternative tokens are incorrectly classified as rejected. Mitigation: extend `ModelBackend::forward()` to return per-token logits/probabilities. Implement true speculative rejection `r ~ Uniform(0,1)`, accept if `r < min(1, target_prob / draft_prob)`.

**See:** [PITFALLS.md](./PITFALLS.md) for all 11 pitfalls with phase-specific warnings.

## Implications for Roadmap

Based on combined research, the milestone should be structured in 5 phases:

### Phase A: Engine Integration (Fix `step_speculative`)

**Rationale:** This comes first because everything else depends on it. The scaffold exists but is broken with critical bugs: per-sequence draft loop, scheduler state desync, shared KV cache, wrong `is_prefill` mode, no KV rollback. Without this phase, nothing speculative works correctly.

**Delivers:** A functional, correct speculative decode pipeline routed through the engine's main loop. Batched draft generation (not O(seqs × draft) sequential), correct verification with logit comparison, KV block rollback for rejected drafts, and scheduler state management that handles variable-length token acceptance.

**Addresses (from FEATURES.md):** Batched draft generation, logit-based verification, correct `is_prefill`, KV cache rollback, unified step method, graceful fallback.

**Avoids (from PITFALLS.md):** Pitfall 1 (code path divergence — single unified `step()`), Pitfall 2 (KV cache corruption — separate block IDs), Pitfall 5 (token matching over-aggression — logit comparison), Pitfall 8 (batch mismatch — batched draft).

**Corresponds to PROJECT.md:** SPEC-ENG-01, SPEC-ENG-02, and the core of SPEC-BENCH-01 (infrastructure).

**Needs research?** No — well-documented patterns. The architecture is defined in vLLM's `SpecDecodeBaseProposer.propose()`, TensorRT-LLM docs, and the current (broken) code.

### Phase B: Self-Speculation Forward Pass

**Rationale:** `SelfSpeculativeModel.generate_draft()` is a stub. It needs the actual layer-truncated forward pass (1/8 layers, weight-shared). This is the main deliverable of the milestone — zero-additional-memory speculation. Phase A must be complete first because the self-spec model's output feeds into the verification pipeline.

**Delivers:** Actual self-speculative decoding. `SelfSpeculativeModel` runs a forward pass through the first 1/8 of layers (weight-shared with target), generates draft tokens. No additional GPU memory beyond the target model. Separates KV block allocation for draft output (no corruption).

**Addresses (from FEATURES.md):** Self-speculation with weight sharing. This is the "differentiator" feature.

**Avoids (from PITFALLS.md):** Pitfall 2 (KV cache corruption — separate scratch blocks), Pitfall 6 (GPU memory leak — weight sharing means no extra memory).

**Corresponds to PROJECT.md:** Remaining work for v16.0 self-spec infrastructure that was stubbed.

**Needs research?** Yes — **Phase B needs `/gsd-research-phase`** during planning. The layer-truncated forward pass needs careful implementation: how to skip the remaining 7/8 of layers, how to wire the truncated model into the `ModelBackend` trait without breaking existing users, and how to handle cross-layer dependencies (residual connections, normalization that span the truncation boundary).

### Phase C: Adaptive Depth + Benchmarks

**Rationale:** The adaptive decoder is already fully implemented — it just needs to be wired into the step loop. Benchmarks must come after Phase A and B are operational (you can't benchmark something that doesn't work). Adaptive depth should be validated through benchmarks.

**Delivers:** Self-tuning draft depth (EWMA-smoothed acceptance rate, deadband hysteresis, no oscillation). Comprehensive A/B benchmark suite comparing speculative vs non-speculative with proper warmup methodology and P50/P95/P99 metrics.

**Addresses (from FEATURES.md):** Adaptive draft depth, acceptance rate monitoring, benchmark metrics (spec vs non-spec).

**Avoids (from PITFALLS.md):** Pitfall 3 (adaptive oscillation — EWMA + deadband), Pitfall 4 (benchmark contamination — separate warmup phase), Pitfall 9 (metrics overload — aggregated counters, not per-token).

**Corresponds to PROJECT.md:** SPEC-BENCH-01, SPEC-BENCH-02, SPEC-ADAPT-01, SPEC-ADAPT-02.

**Needs research?** No — standard patterns. Benchmarks follow existing `BenchmarkSuite` / `ThroughputBenchmark` patterns. Adaptive control is a simple EWMA loop.

### Phase D: Speculative Warmup + Streaming

**Rationale:** Warmup is required (not optional) for correct first-step behavior. Without it, the first speculative step always falls back to non-speculative (cold KV cache → garbage drafts → all rejected). Streaming needs buffer management for accepted tokens. Both depend on Phase A working engine integration.

**Delivers:** Speculative warmup (prefill draft KV cache during target prefill, no cold-start garbage). Deployable spec decode with streaming clients (accepted tokens buffered and sent one-by-one).

**Addresses (from FEATURES.md):** Speculative warmup.

**Avoids (from PITFALLS.md):** Pitfall 10 (uninitialized draft KV cache — warmup), Pitfall 11 (stream token dropping — buffering with backpressure).

**Corresponds to PROJECT.md:** SPEC-WARM-01.

**Needs research?** No — well-understood patterns. Warmup = run non-speculative prefill, copy KV cache to draft blocks. Streaming = buffer of verified tokens, send on consumer demand.

### Phase E: Multi-Model Speculation (Deferred to v18.0)

**Rationale:** Multi-model speculation (separate small draft model) doubles GPU memory, requires lifecycle management (load/unload/swap), tokenizer validation, and speculative warmup. The research recommends deferring because:

- Self-speculation (Phase B) may already provide sufficient speedup
- GPU memory is the primary constraint for most deployments
- Multi-model adds significant complexity (DraftModelLifecycle controller, separate CUDA memory pools, unload/cleanup)

**Delivers:** Optional external draft model support. Users can load a dedicated small model as drafter instead of using self-speculation.

**Addresses (from FEATURES.md):** External draft model support, draft model lifecycle.

**Avoids (from PITFALLS.md):** Pitfall 6 (GPU memory leak — memory estimation at load), Pitfall 7 (tokenizer mismatch — strict vocab validation), Pitfall 6's lifecycle management.

**Corresponds to PROJECT.md:** SPEC-MULTI-01, SPEC-MULTI-02.

**Needs research?** Yes — **Phase E needs `/gsd-research-phase`** during v18.0 planning. Multi-model lifecycle, separate CUDA memory pool management, and tokenizer validation for arbitrary model pairs are non-trivial.

### Phase Ordering Rationale

- **Phase A first** because it's foundational. All other phases depend on a working `step_speculative()` pipeline. The current code has multiple critical bugs that make it non-functional.
- **Phase B second** because self-speculation is the primary differentiator. The `generate_draft()` stub must be replaced with a real forward pass. Phase A provides the pipeline it feeds into.
- **Phase C third** because adaptive depth needs the spec pipeline operational, and benchmarks need both Phase A (comparison baseline) and Phase B (spec decode to measure).
- **Phase D fourth** because warmup and streaming are polish features that depend on the core pipeline. The code works without them, just less efficiently.
- **Phase E deferred** because it's the highest risk (GPU memory, lifecycle complexity) and may not be needed if self-speculation provides sufficient speedup.

### Research Flags

Phases likely needing deeper research during planning:

- **Phase B (Self-Spec Forward Pass):** The layer-truncated forward pass across the architecture registry boundary needs careful design. How does the truncated model interact with `ModelBackend`? How do residual connections and normalization layers that span the truncation boundary behave? Current `SelfSpeculativeModel` needs a concrete implementation plan.
- **Phase E (Multi-Model):** If pursued, needs research on CUDA memory pool separation, lifecycle management patterns, and tokenizer validation across arbitrary model pairs.

Phases with standard patterns (skip research-phase):

- **Phase A (Engine Integration):** Well-documented in vLLM's `SpecDecodeBaseProposer`, TensorRT-LLM docs, and the existing (broken) code structure. Follow vLLM's batched draft generation pattern and the current engine's scheduler contract.
- **Phase C (Adaptive + Benchmarks):** EWMA control loop is a textbook pattern. Benchmarks follow existing `BenchmarkSuite` infrastructure. Add separate warmup phase before measurement.
- **Phase D (Warmup + Streaming):** Warmup = copy KV cache from prefill. Streaming = buffer-and-send. Both are straightforward extensions of existing patterns.

## Confidence Assessment

| Area         | Confidence | Notes                                                                                                                                                                                                            |
| ------------ | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Stack        | HIGH       | All existing code analysis. No new dependencies. `Cargo.toml` and `AGENTS.md` verified.                                                                                                                          |
| Features     | HIGH       | Derived from direct codebase analysis of all speculative decoding files + vLLM competitive reference + Leviathan et al. (2023) academic paper. Feature dependency graph is cross-validated against architecture. |
| Architecture | HIGH       | Current engine.rs code, vLLM v1 engine source, TensorRT-LLM docs all agree on the architecture pattern. Data flow diagrams validated against existing infrastructure.                                            |
| Pitfalls     | HIGH       | 11 pitfalls derived from current code analysis + vLLM production experience + TensorRT-LLM limitations documentation. Each pitfall has detection strategy.                                                       |

**Overall confidence:** HIGH

### Gaps to Address

- **Phase B implementation details:** The layer-truncated forward pass through `SelfSpeculativeModel` needs a concrete design. How to truncate the layer loop in the model's `forward()` without duplicating the entire forward pass? This is the #1 technical gap from the research.
- **Multi-model memory footprint:** No exact measurement of how much GPU memory a separate draft model consumes for different architecture pairs (e.g., Llama-7B draft for Llama-70B target). The 1/50th estimate is a rough guess and needs validation before Phase E planning.
- **Real hardware benchmark numbers:** No empirical data on actual speedup from self-speculation on vLLM-lite's architecture. The 2-3x range from the literature may not transfer directly. Phase A/B/C should validate with real measurements before making commitments.
- **`ModelBackend` extension for logits:** The `forward()` method currently returns `Vec<TokenId>` without logits/probabilities. Phase A needs a design for `forward_with_logits()` or an optional logit return path. The impact on existing model implementations (Llama, Mistral, Qwen, etc.) needs assessment.
- **Streaming behavior under speculative decode:** The current `try_send` pattern can silently drop tokens if the channel is full. Research identified this as a minor pitfall but the exact buffering mechanism for accepted tokens needs design in Phase D.

## Sources

### Primary (HIGH confidence)

- Current vLLM-lite codebase: `engine.rs`, `engine/speculative.rs`, `speculative/adaptive.rs`, `speculative/self_spec.rs`, `speculative/verifier.rs` — direct code analysis
- vLLM v1 engine source: `vllm/v1/engine/core.py`, `vllm/v1/spec_decode/llm_base_proposer.py`, `vllm/v1/spec_decode/draft_model.py` — reference architecture
- TensorRT-LLM Speculative Decoding documentation — production architecture patterns
- Leviathan et al. (2023) "Fast Inference from Transformers via Speculative Decoding" — fundamental algorithm
- vLLM-lite benchmark infrastructure: `benches/src/e2e.rs`, `benches/src/speculative_benchmark.rs` — benchmark patterns
- `Cargo.toml` (workspace), `benches/Cargo.toml` — dependency verification

### Secondary (MEDIUM confidence)

- SpecInfer (2023) "Accelerating Generative LLM Serving with Speculative Inference" — tree-based speculation patterns (not used, confirms linear draft is sufficient)
- vLLM blog on speculative decoding throughput characteristics — concurrency scaling insight (batch > 32, disable speculation)

### Tertiary (LOW confidence)

- Multi-model memory estimates — 1/50th ratio is approximate, needs empirical validation
- Streaming behavior under speculative decode — identified as untested, needs design validation

---

*Research completed: 2026-05-13*
*Ready for roadmap: yes*
