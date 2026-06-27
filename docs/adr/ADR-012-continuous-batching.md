# ADR-012: Continuous Batching over Static Batching

**Date:** 2026-06-27
**Status:** Accepted
**Context version:** v1.0

## Context

LLM inference has two natural batching regimes:

- **Static batching**: assemble N requests into a batch, run them to completion (or to max_tokens), then return all results. The batch is "done" only when the longest sequence finishes.
- **Continuous batching** (also: in-flight batching, iteration-level scheduling): the scheduler maintains a rolling set of active sequences. After every decode step, finished sequences are evicted and waiting sequences are admitted into the freed slots.

Static batching is conceptually simpler — the scheduler's job is "assemble a batch, run it, return". But it has a well-known utilisation problem: in a batch of N sequences, the GPU is only as busy as the *longest* sequence. If 9 of 10 sequences finish after 50 tokens but the 10th runs to 500 tokens, the GPU spent 90% of its time running a single sequence with 9 empty slots.

Continuous batching closes this gap. As soon as a sequence finishes (or is cancelled, or hits its stop token), the scheduler admits a new request into the freed slot. Empirically, this yields **2–10× throughput improvement** vs static batching at the same concurrency level, depending on sequence-length distribution.

The trade-off is complexity:

- The scheduler must track per-sequence state across decode steps (KV cache block ownership, position counters, sampling state).
- Finished-sequence detection must run every step, not just at batch boundaries.
- Memory management is harder — freed slots release KV blocks back to the allocator in the middle of a step.
- Cancellation is naturally supported (just remove the sequence from the rolling set) but introduces new failure modes.

## Decision

vllm-lite uses **continuous batching** as the sole production scheduling strategy. The implementation lives in `crates/core/src/scheduler/engine.rs` (`//! engine: continuous-batching engine.`) and supporting modules under `crates/core/src/scheduler/`.

The scheduler structure:

- **Waiting queue** (`request_queue.rs`) — requests that haven't been admitted yet, ordered by arrival (FCFS) or policy.
- **Running set** — sequences currently holding KV blocks and consuming decode steps.
- **Free pool** — KV blocks released by finished sequences, available for re-allocation.
- **Per-step loop**: build batch → model.forward → sample → check for finished sequences → free their blocks → admit waiting requests into freed blocks → repeat.

Key implementation files:

- `crates/core/src/scheduler/engine.rs` — the main scheduler engine.
- `crates/core/src/scheduler/batch.rs` / `batch_planner.rs` / `batch_composer.rs` — batch construction from the running set.
- `crates/core/src/scheduler/preemption.rs` — preemption policies when a high-priority request needs blocks a low-priority request is holding.
- `crates/core/src/scheduler/phase_scheduler.rs` — Prefill/Decode/Mixed phase discrimination.
- `crates/core/src/scheduler/predictive_batching.rs` — heuristic pre-admission based on predicted sequence length.
- `crates/core/src/scheduler/policy/` — FCFS, SJF, Priority scheduling policies.
- `crates/core/src/scheduler/memory/` — block allocator + eviction.

The scheduler exposes `add_request`, `build_batch`, and the engine wires those into the main `step` loop. Finished sequences are detected after every decode step by checking stop tokens, max_tokens, and cancellation flags.

Static batching is **explicitly rejected** for production use. It may appear in tests for simplicity but the production scheduler is continuous.

## Rationale

1. **Throughput** — 2–10× higher than static batching at the same concurrency, depending on sequence-length distribution. This is the dominant factor in serving economics.
2. **Tail latency** — the longest-sequence bottleneck is broken. P99 latency is bounded by per-step admission fairness, not by the longest sequence in a static batch.
3. **Cancellation is free** — removing a sequence from the running set is a no-op; static batching has to either wait for the batch to complete or implement partial cancellation.
4. **Memory efficiency** — KV blocks freed at step boundaries can be re-allocated immediately, not after the longest sequence finishes. This is the foundation for high concurrency with bounded VRAM.
5. **Composes with all other vllm-lite features** — paged KV cache (ADR-013), prefix caching, speculative decoding (ADR-006), chunked prefill — all assume continuous batching.

Alternatives considered:

- **Static batching** — rejected; the throughput penalty is severe and only acceptable for offline batch jobs.
- **Continuous batching with pre-emption only** — considered; pure continuous batching without preemption can deadlock when a request needs more blocks than are available and no sequence can be safely evicted. vllm-lite added preemption (`scheduler/preemption.rs`) to handle this.
- **Continuous batching with adaptive batching** — considered; predictive admission (`predictive_batching.rs`) uses length predictions to pre-admit requests whose blocks are likely to free up soon. This is an *enhancement* to continuous batching, not a replacement.
- **Disaggregated prefill/decode** — orthogonal; vllm-lite has `enable_pd_separation` as a config flag, but it's an enhancement *on top of* continuous batching, not a replacement.

## Consequences

**Positive:**

- 2–10× throughput vs static batching at the same concurrency.
- Bounded P99 latency — long sequences don't block short ones.
- Free cancellation — clients can abort without waiting for batch completion.
- KV block utilisation stays high — freed blocks re-enter the pool immediately.
- Speculative decoding (ADR-006) and paged KV cache (ADR-013) compose naturally; both assume a rolling set of active sequences.

**Negative:**

- **Complexity** — the scheduler has ~10 supporting modules, each non-trivial. New contributors need weeks to become productive.
- **Preemption is hard** — when a high-priority request needs blocks held by a low-priority one, the low-priority sequence's KV blocks must be recomputed when it's resumed. The implementation (`preemption.rs`) must preserve correctness across the recompute.
- **Debugging is harder** — request-level traces must capture per-step state changes; static batching's "batch starts here, ends here" semantics don't apply.
- **Memory fragmentation risk** — if finished sequences don't release blocks promptly (e.g. due to a bug), the allocator can deadlock. Block release is on the hot path and must be tested extensively.
- **Sequence-length outliers can starve the scheduler** — a single 1M-token request can hold blocks indefinitely if no preemption policy fires.

**Mitigations / migration paths:**

- Preemption policies (`scheduler/preemption.rs`) handle block-exhaustion deadlocks.
- The scheduler emits DEBUG-level logs at every step (`running = N, waiting = M, free_blocks = K`) so operators can diagnose starvation.
- `predictive_batching.rs` provides length-prediction-based admission as an opt-in enhancement.
- For users who really want static batching (e.g. offline eval), the engine exposes a `step_until_done()` helper that drains the current batch before admitting new requests — but this is **not** the default.
- The scheduler's per-step state is serialisable for replay/debugging (`scheduler/stats.rs`).
