# Architecture Patterns: Production Speculative Decoding

**Domain:** LLM Inference Engine — Speculative Decoding
**Researched:** 2026-05-13

## Recommended Architecture

### Current State (v16.0)

```
Main Loop (engine.rs)
├── step() ────────────────────── non-speculative decode
├── step_speculative() ────────── generate_draft → verify → scheduler.update
│   ├── generate_draft_tokens() ─ per-seq loop over draft model
│   └── verify_draft_tokens() ─── target model forward on [input + drafts]
├── step_adaptive_speculative() ─ same + adaptive depth adjustment
│   └── verify_and_track() ────── verify + report to AdaptiveSpeculativeDecoder
└── step_with_graph() ─────────── CUDA graph accelerated (no spec)
```

### Target Architecture (v17.0)

```
Main Loop (engine.rs)
└── step() ─────────────────────────────── unified decode entry point
    ├── [speculative_mode = true]?
    │   ├── step_speculative_inner() ────── batch-aware spec decode
    │   │   ├── speculative_warmup() ────── warm draft KV cache during prefill
    │   │   ├── generate_draft_batched() ── batched draft generation (not per-seq loop)
    │   │   ├── verify_draft_batched() ──── batched verification (one forward pass)
    │   │   ├── scheduler.update_with_accepted() ── update with accepted tokens
    │   │   └── adaptive_decoder.maybe_adjust() ─── dynamic depth (if enabled)
    │   └── [all drafts rejected]?
    │       └── fallback to single-token decode ─── transparent
    └── [speculative_mode = false]?
        └── standard decode path ────────── unchanged
```

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| `Engine` (engine.rs) | Main inference loop, dispatch to spec or non-spec path | SchedulerEngine, ModelBackend (target + draft) |
| `SchedulerEngine` | Batch construction, position tracking, KV block allocation, prefix caching | Engine (via build_batch/update) |
| `DraftVerifier` | Verifies draft tokens against target output, applies rejection strategy | Engine (verify_and_track) |
| `SelfSpeculativeModel` | Wraps target model with 1/8 layer subset for draft generation | Engine (forward for draft tokens) |
| `AdaptiveSpeculativeDecoder` | Tracks acceptance rate, adjusts max_draft_tokens | Engine (record_verification) |
| `SpecDecodeMetrics` | Prometheus counters for draft/accepted/rejected metrics | MetricsCollector, Prometheus exporter |
| `BenchmarkSuite` (benchmarks) | Orchestrates A/B comparison of spec vs non-spec | E2E engine instance |
| `DraftModelLifecycle` (multi-model) | Load/unload/validate external draft models | ModelLoader, Engine |


### Data Flow (Speculative Decode Step)

```
1. SchedulerEngine.build_batch()
   └─> Batch { seq_ids, input_tokens, positions, kv_block_ids, ... }
   
2. Engine.step_speculative_inner(batch)
   │
   ├─ 2a. [Warmup] speculative_warmup(&batch)
   │       └─ Run draft model prefill on input tokens → populate draft KV cache
   │
   ├─ 2b. generate_draft_batched(&batch, max_draft)
   │       └─ For each seq: run draft forward iteratively
   │       └─ Output: Vec<Vec<TokenId>> (per-seq draft tokens)
   │
   ├─ 2c. verify_draft_batched(&batch, &draft_outputs)
   │       ├─ Concatenate [input_tokens | draft_tokens] per seq
   │       ├─ Run target model forward on extended inputs
   │       └─ Compare target_tokens vs draft_tokens (exact match for greedy)
   │       └─ Output: Vec<(SeqId, TokenId)> (accepted + bonus token)
   │
   ├─ 2d. [Adaptive] adaptive_decoder.record_verification(draft_count, accepted_count)
   │       └─ Maybe adjust current_max_draft_tokens
   │
   └─ 2e. scheduler.update(seq_ids, tokens, input_counts)
           └─ Advance sequence positions in scheduler

3. Metrics recording
   ├─ total_tokens produced (accepted + bonus)
   ├─ latency measurement
   └─ Prometheus counters: draft_count, accepted_count, rejection_rate
```

## Patterns to Follow

### Pattern 1: Unified Entry Point (Prevent Path Divergence)
**What:** All decode paths go through a single `step()` method. Speculative decoding is a *modifier* on the standard step, not a replacement.

**When:** When `speculative_mode = true` or `adaptive_decoder.is_some()`, the `step()` method calls `step_speculative_inner()` instead of the standard forward. This keeps the outer loop unified (metrics, error handling, response streaming are the same).

**Why:** Prevents Pitfall 1 (code path divergence). New features added to the outer `step()` automatically benefit speculative decode.

**Current Code:**
```rust
// engine.rs line ~385 — already partially unified
let result = if self.adaptive_decoder.is_some() {
    self.step_adaptive_speculative()
} else if self.speculative_mode {
    self.step_speculative()
} else if self.cuda_graph_enabled() {
    self.step_with_graph()
} else {
    self.step()
};
```

**Ideal:**
```rust
// Everything goes through step()
let result = self.step();
// step() internally checks speculative_mode and dispatches
```

### Pattern 2: Batched Draft Generation (Not Per-Sequence Loop)
**What:** Generate draft tokens for all sequences in parallel, not one-at-a-time in a loop.

**When:** Always, when the model backend supports batched forward passes (which it does — `ModelBackend::forward()` accepts `&[SeqId]`).

**Why:** The current loop:
```rust
for (i, seq_id) in batch.seq_ids.iter().enumerate() {
    for _ in 0..max_draft {
        let output = draft_model.forward(&[*seq_id], ...)?;
        // ...
    }
}
```
This is O(seqs × max_draft) sequential forward passes. For a batch of 8 sequences with 5 draft tokens each, that's 40 forward passes. A batched approach would do at most 5 forward passes (one per draft position).

**vLLM Reference:** `SpecDecodeBaseProposer.propose()` handles ALL draft tokens for ALL sequences in a single batched call to `self.model(...)`.

### Pattern 3: Accept+Bonus Token Visibility
**What:** The verification step always returns either:
- `accepted_draft_count` accepted tokens, OR (if all rejected) 0 accepted tokens
- PLUS exactly 1 "bonus" token (the next token from the target model's distribution over the true prefix)

**When:** This is the standard spec decode protocol from Leviathan et al. (2023). Every spec decode step produces at least 1 token (the bonus).

**Why:** Ensures speculative decoding is never slower than non-speculative (worst case: 1 token per step, same as standard). The current `verify_draft_tokens()` already implements this.

```rust
// Accept consecutive matching drafts
for (j, &draft_token) in drafts.iter().enumerate() {
    if target_tokens[j] == draft_token {
        results.push((*seq_id, draft_token));  // accepted draft
        accepted_count += 1;
    } else { break; }
}
// Always add one bonus token
if target_idx < target_tokens.len() {
    results.push((*seq_id, target_tokens[target_idx]));  // bonus token
}
```

### Pattern 4: Logit-Based Rejection for Non-Greedy Decoding
**What:** For temperature > 0, use probability-based acceptance instead of exact token matching.

**When:** When `sampling_params.temperature > 0` or non-greedy sampling mode.

**How:** 
```python
# Standard speculative rejection (Leviathan et al. 2023)
r ~ Uniform(0, 1)
accept if r < min(1, target_prob(draft) / draft_prob(draft))
```

**Current Limitation:** `ModelBackend::forward()` returns `next_tokens: Vec<TokenId>` but does NOT return per-token probabilities. This needs: (a) extend `ModelBackend` to optionally return logits/probs, or (b) add a `forward_with_logits()` method.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Global Draft State
**What:** Keeping draft depth, acceptance rate, or other speculative state as global engine fields that apply uniformly to all requests.

**Why bad:** Different requests have different acceptance characteristics. Code generation → high acceptance. Creative writing → low acceptance. A single global `max_draft_tokens` penalizes both types.

**Instead:** Maintain per-request (or per-sequence) acceptance tracking. vLLM's `SpecDecodeMetadata` tracks per-request `num_draft_tokens`. The scheduler can maintain per-sequence draft budgets.

### Anti-Pattern 2: Sequential Draft Generation
**What:** The current per-sequence loop for draft generation.

**Why bad:** Fails to utilize GPU parallelism. On a batch of 8 sequences with 5 drafts each, the current code does 40 sequential forward calls through the draft model. Even with 1/8 layer count, the overhead of 40 kernel launches dominates.

**Instead:** Restructure to generate all draft tokens for all sequences at each position together. Each iteration generates 1 draft token per sequence (batched). For 5 drafts: 5 batched forward passes instead of 40 single-sequence passes.

### Anti-Pattern 3: Spec Decode Metrics as an Afterthought
**What:** Adding metrics counters as a final step, wired into the spec decode path but disconnected from the alerting/monitoring pipeline.

**Why bad:** Without proper metrics, you can't tell if speculative decoding is working. Users enable it, see no difference, and disable it. Debugging acceptance rate problems requires per-draft-position telemetry.

**Instead:** Add all spec decode counters as part of the Engine Integration phase (SPEC-BENCH-01). The metrics pipeline must include: draft count, accepted count, per-position acceptance rate, throughput (spec vs non-spec), latency (spec vs non-spec).

### Anti-Pattern 4: Reusing Target KV Block IDs for Draft Forward
**What:** Passing the same `kv_block_ids` to both target and draft model forward passes in self-speculation.

**Why bad:** The draft forward may write to KV cache blocks that the verification pass reads. This is raced/corrupted KV cache data. The effect is silently wrong output — no crash, no error.

**Instead:** Separate KV block allocation for draft output during spec decode steps, or use read-only KV cache for draft layers.

## Scalability Considerations

| Concern | At 1 request (dev) | At 32 concurrent | At 256 concurrent |
|---------|-------------------|-------------------|-------------------|
| Draft generation cost | ~1/8 of target (negligible) | 32× batched draft = 32x faster vs sequential | 256× batched draft = same 1 forward pass (GPU saturates) |
| Verification cost | 1x target forward | 1x target forward on extended batch | 1x target forward (GPU-bound) |
| Adaptive depth | Per-request irrelevant | Global adjustment works fine | Need per-request budgeting |
| Metrics overhead | Trivial | ~N counters/step == fine | Aggregate counters only (avoid per-request histograms on hot path) |
| Multi-model memory | Draft = 1/50th of target | Same (2 models total) | 2 models × N GPU = O(N) memory (expensive) |

**Key insight:** Speculative decoding helps MOST when GPU is underutilized (low batch sizes, small models). At high concurrency (256 concurrent), the GPU is already saturated and speculative decoding adds overhead. vLLM's blog confirms: "speculation for high-throughput serving generally requires large speculation budgets to see benefit." At batch sizes > 32, consider disabling speculation.

## Sources

- vLLM v1 spec decode engine integration: `vllm/v1/engine/core.py` (EngineCore.step, use_spec_decode, post_step) — **HIGH confidence**
- vLLM v1 spec decode proposer: `vllm/v1/spec_decode/llm_base_proposer.py` (propose, set_inputs_first_pass) — **HIGH confidence**
- Leviathan et al. (2023) "Fast Inference from Transformers via Speculative Decoding" — **HIGH confidence**
- TensorRT-LLM Speculative Decoding docs — **HIGH confidence**
- Current vLLM-lite engine code: `engine.rs`, `engine/speculative.rs` — **HIGH confidence**

---
*Architecture research for: Production Speculative Decoding in LLM Inference Engine*
*Researched: 2026-05-13*
