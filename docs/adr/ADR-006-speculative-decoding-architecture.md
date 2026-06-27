# ADR-006: Speculative Decoding Architecture Overview

**Date:** 2026-06-27
**Status:** Accepted
**Context version:** v16.0

## Context

vllm-lite adopted speculative decoding in v16.0 to cut per-token latency. The classic speculative-decoding algorithm is:

1. A cheap **draft model** proposes K candidate tokens autoregressively.
2. A **target model** evaluates all K candidates in a single forward pass.
3. A **verifier** compares the draft's predicted probabilities against the target's for each position, accepting a prefix and sampling a correction token.

The architecture must satisfy several constraints:

- The target model must remain the source of truth — its logit distribution is canonical; the draft is purely a speedup.
- The draft model may be derived from the target (self-speculation, ADR-003), or an entirely separate model (external draft, ADR-007).
- Verification must be deterministic and safe — accepting a draft token must be statistically equivalent to target-only sampling (the Leviathan et al. / Chen et al. guarantee).
- The mechanism must compose with continuous batching (ADR-012), paged KV cache (ADR-013), and prefix caching — speculative decoding is not a separate mode, it's an accelerator on top of the normal decode loop.

Naive designs couple the draft and target via shared mutable state (the draft's KV cache is the target's KV cache), which breaks the ability to swap in external drafts. Designs that fully decouple draft and target force double allocation of every cache. The middle ground is: draft and target share *weight references* but own *KV cache state* independently.

## Decision

vllm-lite's speculative decoding has three architectural components forming a (target, draft, verifier) triplet:

```text
crates/core/src/speculative/
├── mod.rs                     # Re-exports
├── config.rs                  # SpeculationConfig (draft_layers, max_draft_tokens, ...)
├── self_spec.rs               # SelfSpeculativeModel — target as its own draft
├── verifier.rs                # DraftVerifier trait + VerificationResult
├── strategy.rs                # SpeculativeStrategy dispatch
├── adaptive.rs                # AdaptiveSpeculativeDecoder (EWMA + deadband)
├── draft_registry.rs          # DraftModelRegistry (external drafts)
├── draft_resolver.rs          # DraftResolver — per-request routing
└── memory_budget.rs           # VRAM budget enforcement for drafts

crates/core/src/engine.rs      # Engine::step_speculative_inner — orchestrates the triplet
```

**Target model** — the production model loaded by the user. Lives in `Engine` and is the source of truth for logit distributions.

**Draft model** — any `Box<dyn ModelBackend>` registered with the engine. Two flavours:

- *Self-spec*: `SelfSpeculativeModel<M>` wraps the target `M` and forwards to `M::forward_to_layer(..., draft_layer_count)` (ADR-003).
- *External draft*: a separate model loaded from disk, managed by `DraftModelRegistry`, with its own VRAM budget (`memory_budget.rs`).

**Verifier** — implemented at the engine level in `Engine::verify_draft_tokens_logits` (`crates/core/src/engine.rs`). The verifier calls `ModelBackend::forward_logits()` for the K candidate tokens in a single forward, then compares per-position argmax (or, in probabilistic mode, per-position acceptance sampling). The `DraftVerifier` trait in `verifier.rs` exists as a seam for testing but is bypassed in production — the engine's logit-based path is the implementation of record.

The three components communicate via three types:

- `SpeculationConfig` — static config (draft layer count, max draft tokens, fallback policy).
- `DraftVerifier::generate_draft` — draft → `(SeqId, Vec<TokenId>)` per sequence.
- `VerificationResult` — verifier → `(accepted_count, next_token)` per sequence.

## Rationale

1. **Triplet separation** — target, draft, verifier each have one job. The target runs full forward passes; the draft runs cheap proposals; the verifier is a pure comparison function. None of them needs to know about the others' internals.
2. **Target in core** — the target model is owned by the engine, which lives in `vllm-core`. Placing it there means scheduling, batching, and KV-cache concerns all live in one crate.
3. **Draft model in core/src/speculative/** — drafts are speculative; they belong with the rest of the speculative machinery (config, verifier, registry). They share `crates/core` with the target because they share types (`Batch`, `SeqId`, `BlockId` from `vllm-traits`).
4. **Verifier at engine level** — the verifier needs both the draft's tokens and the target's full logits; the engine is the only place that has both. A trait-based verifier adds an indirection that buys nothing in production.
5. **Trait seam for testing** — `DraftVerifier` exists so unit tests can swap in stub verifiers without standing up a full engine; production code bypasses it via the engine's logit path.
6. **Config object** — `SpeculationConfig` captures all knobs in one place, enabling safe `Clone`/`Debug` for logging and `Default` for ergonomic defaults.

Alternatives considered:

- **Single `SpeculativeModel` struct** with embedded verifier — rejected; conflates three concerns and prevents alternative verifier implementations.
- **Draft and target in same crate, verifier in separate crate** — rejected; the verifier needs engine state, creating a circular dependency.
- **Trait-based everything** — rejected; the verifier trait would need to return logit comparisons, which leaks the target's internals into the trait surface.
- **No verifier abstraction** — rejected; tests would have to instantiate full engines to verify draft logic.

## Consequences

**Positive:**

- Clear separation of concerns — adding a new draft strategy touches `speculative/` only; changing the verifier touches `engine.rs` only.
- Self-spec and external drafts share the same verifier and engine path; no duplication.
- `SpeculationConfig` is the single source of truth for all knobs; UI surfaces (CLI, YAML, HTTP) all bind to it.
- The trait seam lets unit tests stub the verifier cheaply.

**Negative:**

- Three components to coordinate means three things can fail independently; failure modes must be tested at the integration level.
- The trait seam (`DraftVerifier`) is a stub in production — slight risk of it drifting from the engine's real verifier path.
- Self-spec and external drafts have slightly different code paths in `Engine::step_speculative_inner` (e.g. self-spec shares KV state, external draft owns its own).
- Adding a third draft strategy (e.g. Medusa heads) requires touching both `speculative/` and the engine dispatch.

**Mitigations / migration paths:**

- The engine's speculative loop has explicit fallback paths (`step_speculative_inner` → non-spec decode on any verifier failure) so partial breakage is non-fatal.
- All three components log to the same `EnhancedMetricsCollector` namespace, making cross-component correlation straightforward.
- The `SpeculativeStrategy` enum in `strategy.rs` is the dispatch seam — adding new strategies is a localised change.
