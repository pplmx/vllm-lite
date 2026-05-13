# Domain Pitfalls: Production Speculative Decoding

**Domain:** LLM Inference Engine — Speculative Decoding
**Researched:** 2026-05-13

## Critical Pitfalls

### Pitfall 1: Engine Loop Divergence — Two Code Paths That Rot

**What goes wrong:** The speculative decode path (`step_speculative`) and the standard decode path (`step`) are implemented as entirely separate methods. Over time, they diverge as features are added to only one path. This creates a "shadow engine" where speculative users miss bug fixes and optimizations added to the standard path.

**Why it happens:** It's natural to add a feature to `step()` and forget `step_speculative()` needs the same change. The current architecture has THREE paths: `step()`, `step_speculative()`, `step_adaptive_speculative()`. Currently 904-line `engine.rs` with separate speculative file at `engine/speculative.rs`.

**Consequences:** Silent correctness bugs in the speculative path. Performance optimizations (CUDA Graph, chunked prefill) apply to one path but not the other. Users who enable speculative decoding get different behavior.

**Prevention:** Either (a) refactor so speculative decoding is a *modifier* on the standard step, not a replacement, or (b) create a `step_inner()` that both paths call, with spec decode adding draft generation + verification as hooks. vLLM v1 solution: always calls `step()` in `EngineCore`, speculative decoding is just a model executor feature (proposer loaded as part of model).

**Detection:** Run same request through both paths with `speculative_mode = false` and assert output tokens match exactly. This should be a CI gate.

### Pitfall 2: Draft KV Cache Corruption Through Shared References

**What goes wrong:** In self-speculation (weight sharing), the draft model uses a subset of target layers. If KV cache block IDs are shared between draft and target passes, the draft forward can modify KV cache entries that the target pass expects to be stable. This produces silently wrong tokens (no crash, just incorrect output).

**Why it happens:** The current `SelfSpeculativeModel` shares weights and KV block IDs. The draft forward at layer L modifies KV cache blocks. When the target later re-runs verification on the same block IDs, it loads corrupted data.

**Consequences:** Non-deterministic output corruption. Only shows up under load (contention). Impossible to debug without KV cache snapshot comparison.

**Prevention:** Use separate KV block IDs for draft vs target during the same step, or use copy-on-write semantics. vLLM's EAGLE approach avoids this entirely by using the target's KV cache directly (eagle layers don't write to KV cache — they read-only). For self-speculation: allocate a scratch KV block for draft output that is NOT shared with target.

**Detection:** Compare KV cache block contents before and after draft forward. Any modification to target blocks = corruption.

### Pitfall 3: Adaptive Depth Oscillation

**What goes wrong:** The adaptive draft depth continuously oscillates between high and low values, never stabilizing. Acceptance rate is inherently noisy (varies by prompt content, generation phase, etc.). Simple threshold-based control (current implementation: ±1 step, cooldown) is prone to oscillation.

**Why it happens:** Current `AdaptiveSpeculativeDecoder.maybe_adjust()` uses:

- `rate > target + 0.1` → increase
- `rate < target - 0.1` → decrease

With a single cooldown, the system overcorrects. High drafts → lower acceptance → decrease drafts → higher acceptance → increase drafts → cycle repeats.

**Consequences:** Users see constant "Adjusted max_draft_tokens" log spam. Effective throughput oscillates. The feature feels broken.

**Prevention:** Use (a) EWMA (exponential weighted moving average) for smoothing acceptance rate, (b) PID-style control with proportional term only (no integral to avoid windup), (c) deadband with hysteresis. Log adjustments at INFO but reduce frequency. Consider per-gen-phase adaptation: drafts matter more during stable decode than during prefill.

**Detection:** Parse logs for "Adjusted max_draft_tokens.* -> " and count adjustments over a 100-step window. >10 adjustments = oscillation.

### Pitfall 4: Benchmark Contamination — Warmup Effects

**What goes wrong:** First speculative decode step is always slower because KV caches are cold. Benchmark results conflate warmup effects with true spec decode performance. Non-speculative vs speculative comparison is invalid because the non-speculative baseline benefits from the same warmup.

**Why it happens:** The current benchmark infrastructure has `warmup_iterations` (default 3), but speculative warmup is separate from benchmark warmup. The first spec decode step runs draft layers on empty KV cache → bad performance → pollutes latency P99 metrics.

**Consequences:** Benchmarks show speculative decoding as *slower* than baseline. Incorrect conclusions about whether to enable the feature.

**Prevention:** (a) Require a separate "speculative warmup phase" before any benchmark measurement — run N tokens through spec decode, discard results. (b) In benchmark reports, explicitly state "Warmup tokens: N (discarded)". (c) Offer a `--spec-warmup-tokens` flag. (d) Run spec and non-spec benchmarks in the *same* process to ensure identical conditions.

**Detection:** Run the benchmark twice: once with default warmup, once with 10x warmup. If results differ significantly, warmup is insufficient.

### Pitfall 5: Token Matching Rejection Is Too Aggressive

**What goes wrong:** The current `verify_draft_tokens()` uses exact token matching:

```rust
if target_tokens[j] == draft_token {
    // accept
} else {
    break;  // reject ALL remaining drafts
}
```

This is a *deterministic* acceptance rule (token must match). The theoretical spec decode rejection uses *probability-based* acceptance (sample from max(0, target_prob - draft_prob) / target_prob). The exact-token approach matches the greedy decoding case but is incorrect for non-greedy (sampling) modes.

**Why it happens:** Simpler to implement. The current code doesn't track logits/probabilities — it only tracks token IDs stolen from `ModelBackend::forward().next_tokens`.

**Consequences:** With temperature > 0 or top-k sampling, the rejection strategy incorrectly classifies valid alternative tokens as "rejected." Speculative throughput is lower than theoretically possible.

**Prevention:** Extend `ModelBackend::forward()` to return both token IDs and their logits/probabilities. Implement true speculative rejection: sample r ~ Uniform(0,1), accept draft if `r < min(1, target_prob(draft) / draft_prob(draft))`. This requires the model to return per-token probabilities.

**Detection:** Compare acceptance rates between greedy (temp=0) and non-greedy (temp=0.7) modes. If non-greedy rates are significantly lower, the rejection strategy is too aggressive.

## Moderate Pitfalls

### Pitfall 6: Draft Model Lifecycle Leaks GPU Memory

**What goes wrong:** Loading a separate draft model doubles GPU memory usage. If loading fails mid-way, partial weights orphan GPU memory. Swapping draft models without proper cleanup causes memory fragmentation. Currently: `server/main.rs` loads draft model via `loader.load_model()` — same loader, blocks same GPU memory pool.

**Prevention:** (a) Estimate draft model memory before loading. (b) Fail fast with clear memory requirements if insufficient VRAM. (c) Use separate CUDA memory pools for draft vs target. (d) Implement `unload_draft_model()` that properly frees all GPU allocations.

### Pitfall 7: Tokenizer Mismatch Between Draft and Target

**What goes wrong:** Self-speculation shares the tokenizer (same model, fewer layers). Multi-model speculation requires the draft model to use the EXACT same tokenizer. If vocabularies differ even by one token, draft token IDs are meaningless to the target model exacerbating into silent garbage output.

**Prevention:** (a) Validate `vocab_size` matches at model load time (vLLM's `verify_equal_vocab_size_if_draft_model()`). (b) Validate tokenizer configs match (same `tokenizer_config.json`). (c) Reject mismatched pairs with clear error message.

### Pitfall 8: Batch Size Mismatch in Draft vs Target Forward

**What goes wrong:** The draft model forward is called per-sequence in a loop (`generate_draft_tokens()` iterates `for (i, seq_id) in batch.seq_ids.iter().enumerate()`). This means draft generation is NOT batched — it's O(seqs * max_draft) sequential forward passes. This can be slower than the target model's batched forward for large batches.

**Prevention:** Batch the draft model forward vLLM-style: process all draft tokens in parallel where possible. For self-speculation with 1/8 layers, the per-draft-token cost is small, but iterating in a loop loses batch parallelism. See: `SpecDecodeBaseProposer.propose()` in vLLM which handles all draft tokens in a single batched call.

### Pitfall 9: Metrics Overload from Per-Token Tracing

**What goes wrong:** Enabling TRACE-level speculative logging (`trace!(seq_id = %seq_id, token = %token, "Token generated")` for each draft token) generates O(draft_tokens * batch_size * steps_per_second) log entries per second. On a busy server (e.g., 256 drafts/step, 100 steps/s), that's 25,600 log lines/second.

**Prevention:** Keep per-token logs at TRACE level (not DEBUG). Use aggregated metrics for production: count of drafts/accepted/rejected, not per-token events. Implement rate-limited logging for unexpected conditions (e.g., "all drafts rejected" logged at WARN but at most once per N seconds).

## Minor Pitfalls

### Pitfall 10: Uninitialized Draft KV Cache on First Step

**What goes wrong:** On the very first speculative step after startup (or after a long idle period), the draft model's KV cache blocks are uninitialized (zero-filled). The first draft forward produces garbage tokens → all rejected → first speculative step falls back to non-speculative → user perceives spec decoding as "not working."

**Prevention:** Run one warmup step (non-speculative) before enabling speculation, or initialize draft KV cache with the prefill output. This is SPEC-WARM-01.

### Pitfall 11: Speculative Decoding with Streaming

**What goes wrong:** Streaming sends tokens to the client one at a time. Speculative decoding decodes multiple tokens in one step. The streaming layer needs to buffer accepted tokens and send them one-by-one at the correct timing. Current code uses `try_send` which can silently drop tokens if the channel is full.

**Prevention:** Use async streaming with backpressure for speculative tokens. Maintain a per-request buffer of verified tokens. Send at the correct rate matching the client's consumption.

## Phase-Specific Warnings

| Phase Topic        | Likely Pitfall                                                                | Mitigation                                                          |
| ------------------ | ----------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Engine Integration | Pitfall 1: Code path divergence + Pitfall 10: Cold start                      | Refactor into a single path with hooks. Add warmup pre-step.        |
| Benchmarks         | Pitfall 4: Warmup contamination + Pitfall 5: Incorrect rejection for sampling | Measure warmup adequacy. Always note sampling config.               |
| Adaptive Depth     | Pitfall 3: Oscillation                                                        | Use EWMA + PID-style control. Add deadband hysteresis.              |
| Speculative Warmup | Pitfall 10: Cold KV cache                                                     | Warm draft KV cache from prefill output. Validate block allocation. |
| Multi-model        | Pitfall 6: Memory leak + Pitfall 7: Tokenizer mismatch                        | Memory estimation at load. Strict vocab validation.                 |

## Sources

- vLLM v1 spec decode source: `vllm/v1/spec_decode/llm_base_proposer.py`, `draft_model.py`, `vllm/v1/engine/core.py` — **HIGH confidence** (actual source code)
- TensorRT-LLM spec decode docs and limitations — **HIGH confidence**
- Leviathan et al. (2023) speculative decoding paper — **HIGH confidence**
- Speculative decoding theory: rejection sampling requires logit/probability access, not just token ID matching — **HIGH confidence**
- Current vLLM-lite engine code: `engine.rs`, `engine/speculative.rs`, `speculative/adaptive.rs` — **HIGH confidence**
- vLLM-lite benchmark infrastructure: `benches/src/e2e.rs`, `benches/src/speculative_benchmark.rs` — **HIGH confidence**

---

*Pitfalls research for: Production Speculative Decoding in LLM Inference Engine*
*Researched: 2026-05-13*
