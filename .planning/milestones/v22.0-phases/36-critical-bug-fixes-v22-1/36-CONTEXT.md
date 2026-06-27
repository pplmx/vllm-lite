# Phase 36: Critical Bug Fixes (v22.1) - Context

**Gathered:** 2026-06-27
**Status:** Ready for planning
**Mode:** Smart discuss auto-accepted (infrastructure phase)

<domain>

## Phase Boundary

Eliminate pre-existing P0/P1 bugs blocking production deployment. Three concrete fixes:

1. **`Engine::step()` speculative-mode hang** — 2 `#[ignore]`'d Phase 19 e2e tests (`crates/core/tests/engine_v18_wiring.rs`, `crates/core/tests/draft_resolver_integration.rs`) reveal an actual determinism bug when the engine is in speculative mode. The hang must be diagnosed, root-caused, and a regression test added before the `#[ignore]` markers can be removed.
2. **`cargo doc --workspace --no-deps` broken-link warnings** — 10 warnings across `vllm-model` (3), `vllm-core` (6), `vllm-testing` (1). All are intra-doc-link resolution failures in module-level doc comments.
3. **GGUF parser placeholder TODO** — `crates/model/src/quantize/gguf.rs:10` has an actionable TODO comment for a parser placeholder. Either implement the parser (preferred — see ADR-009 follow-up) or remove with documented rationale.

FINAL-01: All 1146+ existing tests must remain green throughout (hardening scope, no expected test growth).

</domain>

<decisions>

## Implementation Decisions

### the agent's Discretion

All implementation choices are at the agent's discretion — pure infrastructure / bug-fix phase. Use the codebase scout below to drive decisions. Specific guidance:

- **Hang diagnosis**: Use `cargo test --test engine_v18_wiring -- --ignored --nocapture` and `cargo test --test draft_resolver_integration -- --ignored --nocapture` to reproduce. Look at scheduler → engine → spec_dispatch → draft_resolver call paths. The bug was deferred from Phase 19 (v18.0) and is rooted in either draft model state, token ID wrapping, or speculative flag propagation.
- **Doc warnings**: 10 specific warnings to fix, all in intra-doc-link syntax (`[name]`) in module-level doc comments. Strategy: prefer path-qualified links (`[util](crate::components::attention::util)`), backtick escapes (`[\`name\`]`) for literal text, or remove broken links entirely.
- **GGUF TODO**: Decision between implementing the parser or removing the TODO. Since `GgufLoader::load` (feature-gated) is the only caller and currently falls back to empty map, removing the TODO + documenting the parser as future work is acceptable. Implementing a minimal GGUF reader would require significant scope.
- **Regression test**: One new test minimum covering the speculative-mode step completion.

</decisions>

<code_context>

## Existing Code Insights

### Reusable Assets

- **`SchedulerEngine::step()`** at `crates/core/src/engine.rs` — the function exhibiting the hang. Receives `SchedulerOutput` and feeds through `draft_resolver` + `spec_dispatch`.
- **`Engine::step()`** orchestration in same file — calls `scheduler.step()`, then routes to `spec_dispatch` if speculative mode is enabled.
- **`DraftModelRegistry`** (`crates/core/src/speculative/draft_registry.rs`) — heterogeneous draft management, has fallback paths via FALL-01.
- **`DraftResolver`** (`crates/core/src/speculative/draft_resolver.rs`) — resolves draft IDs to backends; known location of Phase 19 issues.
- **Test harness** (`crates/testing/`) — `BatchBuilder`, `RequestFactory`, `FakeModel`, `StubModel` provide deterministic mocks.

### Established Patterns

- **Verification gates**: `just clippy`, `just fmt-check`, `just nextest` (skips `#[ignore]`), `just nextest-all` (includes `#[ignore]`).
- **Doc comments**: Project uses brief `/// name: description.` style with intra-doc links like `[\`TypeName\`]` for cross-references.
- **Test patterns**: Integration tests in `crates/*/tests/*.rs`; unit tests in `#[cfg(test)] mod tests {}` blocks.
- **TODO convention**: TODOs use `TODO(vXX.X+):` format with milestone target.

### Integration Points

- **Doc warnings locations** (file:line):
  - `crates/model/src/components/attention/mod.rs:7` — `[`util`]` link to non-resolving item
  - `crates/model/src/components/block.rs:3` — `[super::decoder_block::PagedDecoderBlock]` (decoder_block is sibling module, rustdoc needs different syntax)
  - `crates/model/src/components/decoder_block/mod.rs:4` — `[super::block::TransformerBlock]` (block is sibling, same issue)
  - `crates/core/src/engine.rs:159` — `[Self::preload_drafts]` (private method, not linkable)
  - `crates/core/src/speculative/registry/mod.rs:30-33` — `[types]`, `[errors]`, `[loader]`, `[lifecycle]` (markdown bullets, not code refs)
  - `crates/core/src/speculative/registry/lifecycle.rs:148` — unclosed `<dyn>` HTML tag
  - `crates/testing/src/lib.rs:79` — `#[ignore]` syntax in markdown
- **GGUF TODO location**: `crates/model/src/quantize/gguf.rs:10`
- **Hang test files**: `crates/core/tests/engine_v18_wiring.rs`, `crates/core/tests/draft_resolver_integration.rs` (both have `#[ignore]` markers)

</code_context>

<specifics>

## Specific Ideas

- **Doc warning fix strategy**: For module-level bullet lists referencing submodules, replace `[name]` with plain backticks: `\`types\`` or `[`types`](crate::speculative::registry::types)`. For sibling-module references like `[super::block::X]`, use `[X](super::block::X)` syntax. For private method refs like `[Self::preload_drafts]`, use code span `\`Self::preload_drafts\``. For unclosed HTML `<dyn>` tags, escape as `\<dyn\>` or use code span.
- **Hang fix strategy**: Reproduce with the two ignored tests under `--nocapture`. Look at whether `step()` returns when there's no draft but speculative mode is enabled, or whether it loops on draft resolver returning `None` without self-spec fallback. The fix likely adds either an early-return condition or a fallback counter.
- **GGUF parser decision**: Project ADR-009 (orphan-module decision) treated gguf.rs as intentionally minimal. Removing the TODO with a "future parser" doc comment is consistent with that prior decision and avoids scope creep. Implementing a parser is out of v22.0 scope.

</specifics>

<deferred>

## Deferred Ideas

- **Full GGUF parser implementation** — would require reading the GGUF format spec (Q4_K_M, Q5_K, Q8_0 quantization types), parsing tensors and metadata, and integrating with `StorageTensor`. Out of v22.0 scope; tracked as future work (consistent with ADR-009).
- **Documentation coverage push to 99%+** — Already RFU-06 deferred from v21.0. v22.0 closes the broken-link warnings (10 → 0) but does not push coverage higher.

</deferred>
