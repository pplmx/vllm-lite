# Feature Landscape: Production Speculative Decoding

**Domain:** LLM speculative decoding engine integration
**Researched:** 2026-05-13

## Table Stakes

Features that any production speculative decoding system must implement. Missing these means the feature is non-functional.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Batched draft generation | Single-threaded per-sequence draft kills throughput | High | This is the #1 bottleneck fix — draft steps must batch across ALL sequences |
| Logit-based token verification | Exact matching gives poor acceptance rates | High | Must call `forward_logits()` not just `forward()` for probability comparison |
| Correct is_prefill for verification | Using prefill mode for decode is O(n²) incorrect | Medium | Concatenated sequence needs decode mode with logits from all positions |
| KV cache rollback for rejected tokens | Without rollback, rejected draft KV entries leak | Medium | Need `rollback_blocks()` in MemoryManager |
| Unified speculative step method | Two code paths (spec + adaptive-spec) is a maintenance nightmare | Low | Merge into `step_speculative(max_draft)` with a parameter |
| Graceful fallback to non-spec | Single point of failure: if draft model fails, engine must keep working | Medium | Already partially handled by mode checking in run() |
| Acceptance rate metrics | Needed to evaluate spec quality and tune parameters | Low | DraftAccuracyTracker already exists |

## Differentiators

Features that set the vllm-lite implementation apart from basic speculative decoding.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Adaptive draft depth | No manual tuning needed — self-optimizes for different prompts/models | Medium | Already fully implemented! Just needs integration |
| Self-speculation with weight sharing | Zero additional GPU memory for draft model | High | Implemented but blank — needs actual forward pass truncation |
| External draft model support | Can use dedicated small model (e.g., 7B draft for 70B target) | High | Multi-model lifecycle manager needed |
| Speculative warmup | Draft KV cache populated during prefill, avoiding cold-start garbage | Medium | Only needed for external draft (self-spec shares KV) |
| Token-level + block-level strategies | Multiple rejection strategies (configurable per deployment) | Low | Already implemented in `RejectionStrategy` |
| Automatic draft depth adjustment | Reacts to real-time acceptance rates, no operator intervention | Low | AdaptiveSpeculativeDecoder done — just needs to be wired |
| Fine-grained per-model metrics | Track draft accuracy per model/draft combination | Low | Metrics infrastructure already exists |

## Anti-Features

Features to explicitly NOT build.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Tree-based speculation (draft tree) | Sigmoidally more complex: tree attention, verification, scoring | Linear draft sequence — well-established, simpler, good enough |
| Medusa-style multiple heads | Needs custom model training, incompatible with off-the-shelf models | Self-speculation with layer truncation — no training needed |
| Hardware-specific draft models | FPGA/ASIC draft models are niche and non-portable | Standard Candle model — portable across CUDA/CPU |
| Dynamic model switching mid-request | Extremely complex state management, low ROI | Per-request draft model selection (simpler, future) |
| Speculative decoding for prefill | Theoretically possible but practically useless — prefill is compute-bound anyway | Only speculative decode (post-prefill) — standard approach |

## Feature Dependencies

```
Basic Spec Path (Phase A)
  ├── Batched draft generation ──────────────────────────── No dependency
  ├── Logit-based verification ──────────────────────────── Needs batched draft output
  ├── Correct is_prefill for verify ─────────────────────── Depends on verify pipeline design
  ├── KV cache rollback ─────────────────────────────────── Depends on MemoryManager extension
  └── Unified step_speculative(max_draft) ───────────────── Depends on all above

Self-Speculation (Phase B)
  └── SelfSpeculativeModel implementation ──────────────── Depends on Phase A pipeline

Adaptive Depth (Phase C)
  ├── AdaptiveDecoder wiring ───────────────────────────── Depends on Phase A
  └── Benchmarks ──────────────────────────────────────── Depends on Phase A + B

Multi-Model (Phase D)
  ├── DraftModelManager ───────────────────────────────── Depends on Phase A
  ├── Speculative warmup ──────────────────────────────── Depends on Phase A
  └── Memory management ───────────────────────────────── Depends on Phase A

Production Hardening (Phase E)
  └── Everything above ────────────────────────────────── Depends on all phases
```

## MVP Recommendation

Prioritize in order:

1. **Fix step_speculative (Phase A)** — batched draft generation + logit verification + correct decode phase + KV rollback. This is the foundational block. Without it, nothing speculative works correctly.

2. **Implement SelfSpeculativeModel (Phase B)** — the actual layer-truncated forward pass. This gives speculative decoding with zero extra memory (weight sharing). This is the main deliverable.

3. **Wire adaptive decoder + benchmarks (Phase C)** — the adaptive depth layer makes the system self-tuning competitiveness. Benchmarks validate correctness and quantify throughput gain.

Defer:
- **Multi-model support (Phase D):** Only valuable if self-speculation underperforms on specific model architectures. Post-MVP.
- **Production hardening (Phase E):** Graceful fallback is nice but the engine already has error handling. Post-MVP.

## Sources

- Direct codebase analysis of all speculative decoding files
- Competitive context: vLLM (Python) reference architecture for speculative decoding
- Academic: Leviathan et al. "Fast Inference from Transformers via Speculative Decoding" (2023)
- Academic: SpecInfer "Accelerating Generative LLM Serving with Speculative Inference" (2023)
