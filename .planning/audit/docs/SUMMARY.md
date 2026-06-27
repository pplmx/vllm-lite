# Documentation Audit Summary

**Generated:** 2026-06-27
**Phase:** 22 — Comments + Documentation Audit (v19.0 milestone)
**Detailed findings:** [`REPORT.md`](./REPORT.md)
**Source files scanned:** 232 (under `crates/*/src/`)
**Workspace `.rs` files scanned:** 286 (incl. `tests/` and `benches/`)
**Total `pub` items surveyed:** 840 (in `src/`)
**Workspace total `pub` items:** 868 (incl. `tests/`)
**Existing ADRs found:** 2 (`docs/adr/ADR-001`, `docs/adr/ADR-002`)

---

## Total findings: 24

- **P0:** 0
- **P1:** 20
- **P2:** 4

### Distribution by dimension

| Dimension | Findings | P0 | P1 | P2 |
|-----------|---------:|---:|---:|---:|
| DOCS-01 Doc-comment coverage | 6  | 0 | 6 | 0 |
| DOCS-02 Module-level docs    | 1  | 0 | 1 | 0 |
| DOCS-03 Stale comments       | 4  | 0 | 3 | 1 |
| DOCS-04 External doc accuracy| 8  | 0 | 6 | 2 |
| DOCS-05 ADRs                 | 5  | 0 | 4 | 1 |

---

## Prioritized Findings

| ID | Dim | Description | Severity | Source | Effort |
|----|-----|-------------|----------|--------|--------|
| **DOCS-F-01** | DOCS-01 | **Workspace doc-coverage is 7.6%** — 776 of 840 `pub` items lack `///`. **No crate meets the 80% target** (range: 0.0% in `traits` to 12.9% in `testing`). | **P1** | all crates | 20-40h (full backfill) |
| **DOCS-F-02** | DOCS-04 | **README code example is broken** — `SchedulerEngine::new(config, 1024)` won't compile; actual signature requires 3 args (`config`, `num_kv_blocks`, `metrics`). README at `L448-459` misleads users. | **P1** | `README.md:448-459` | 0.5h |
| **DOCS-F-03** | DOCS-04 | **README/AGENTS.md list 5-6 architectures but registry registers 10.** Missing from docs: gemma3, llama4, mistral_small, phi4 (and qwen3_5 in README only). | **P1** | `README.md:251-268`, `AGENTS.md:317-324` | 1h |
| **DOCS-F-04** | DOCS-04 | **Workspace crate count is wrong** — README `L467` and AGENTS.md `L57` both claim 7 crates. `Cargo.toml:2` lists 6 (`core, model, server, traits, dist, testing`). The 7th (likely `benches`) does not exist. | **P1** | `README.md:467`, `AGENTS.md:57`, `Cargo.toml:2` | 0.5h |
| **DOCS-F-05** | DOCS-04 | **README missing debug endpoints** — `/v1/models`, `/debug/metrics`, `/debug/kv-cache`, `/debug/trace` are wired in `server/src/main.rs:292,309-311` but not documented. | **P1** | `README.md:365-375`, `crates/server/src/main.rs:292-311` | 0.5h |
| **DOCS-F-06** | DOCS-04 | **Test count claims outdated** — README `L11,177,178,237,239` cite "900+ unit tests" and "30+ E2E tests"; Phase 19 reported "287+ tests". Either update or remove specific numbers. | **P1** | `README.md:11,177-178,237,239` | 0.5h |
| **DOCS-F-07** | DOCS-05 | **No ADRs for major architectural decisions** — only 2 ADRs exist for a 19-milestone project. Missing: self-spec 1/8 layer ratio, FP8 E4M3 format, KV cache split, speculative architecture, draft routing, FP8 quantizer orphan. | **P1** | `docs/adr/` (only 2 files) | 8-12h (write ~6 ADRs) |
| **DOCS-F-08** | DOCS-05 | **KV cache concept split across 3 locations is undocumented** — `core/kv_cache/`, `model/kv_cache.rs`, `model/components/kv_cache_fp8.rs` (orphan per Phase 21). Split rationale should be in an ADR. | **P1** | `crates/core/src/kv_cache/`, `crates/model/src/kv_cache.rs`, `crates/model/src/components/kv_cache_fp8.rs` | 1h (ADR) + decision on whether to consolidate |
| **DOCS-F-09** | DOCS-04 | **README `L466-486` project structure tree is severely incomplete** — shows ~50% of `crates/model/src/` (missing `arch, causal_lm, config, gemma3, llama4, loader, mistral_small, paged_tensor, phi4, quantize` and several `.rs` files). | **P1** | `README.md:466-486` | 1h |
| **DOCS-F-10** | DOCS-04 | **Feature flags table in README is partial** — only lists `cuda`, `gguf`, `full` from model crate. Missing: `prometheus`, `opentelemetry`, `cuda-graph` (core), `candle`, `kernels` (traits), `cuda` (testing). | **P1** | `README.md:510-514`, all `crates/*/Cargo.toml` | 1h |
| **DOCS-F-11** | DOCS-04 | **AGENTS.md links outdated** — `core/kv_cache/prefix_cache.rs` is correct path; `server/openai/chat.rs` is correct; some other entries may need path validation. | **P1** | `AGENTS.md:533-545` | 0.5h |
| **DOCS-F-12** | DOCS-03 | **`quantize/gguf.rs:7` has `// Placeholder: return empty for now`** — looks like dead code or a regression. Verify it's intentional; if so, document why. | **P1** | `crates/model/src/quantize/gguf.rs:7` | 0.5h (verify + decide) |
| **DOCS-F-13** | DOCS-03 | **Two forward-looking comments reference non-existent "Phase 18.3"** — `draft_registry.rs:434` and `engine.rs:327`. v18.0 is shipped; "Phase 18.3" doesn't appear in PROJECT.md or REQUIREMENTS.md. Either this was deferred (rephrase) or completed (remove stale reference). | **P1** | `crates/core/src/speculative/draft_registry.rs:434`, `crates/core/src/engine.rs:327` | 0.5h |
| **DOCS-F-14** | DOCS-01 | **`traits` crate has 0% doc coverage** — 14 substantive `pub` items undocumented (CudaGraphConfig, ModelGraphConfig, GraphExecutionError, etc.). | **P1** | `crates/traits/src/kernels.rs`, `crates/traits/src/model.rs`, `crates/traits/src/types.rs` | 2h |
| **DOCS-F-15** | DOCS-01 | **`dist` crate has 2.7% doc coverage** — 36 substantive `pub` items undocumented. `tensor_parallel/parallel_linear.rs`, `pipeline/`, `distributed_kv/` all lack docs. | **P1** | `crates/dist/src/` | 3h |
| **DOCS-F-16** | DOCS-01 | **`server` crate has 4.9% doc coverage** — 65 substantive `pub` items undocumented. The OpenAI API surface (`openai/chat.rs`, `openai/completions.rs`, `openai/embeddings.rs`), auth (`auth.rs`), config (`config.rs`), security (`security/*`) are all missing docs. | **P1** | `crates/server/src/` | 4h |
| **DOCS-F-17** | DOCS-01 | **`model` crate has 8.5% doc coverage** — 170 substantive `pub` items undocumented. Architecture implementations (`llama`, `qwen3`, `qwen3_5`, `gemma3`, `gemma4`, `mistral`, `mistral_small`, `llama4`, `phi4`, `mixtral`), shared components (`attention/gqa.rs`, `attention/mla.rs`, `kv_cache_fp8.rs`), and the `loader/` API are all undocumented. | **P1** | `crates/model/src/` | 8h |
| **DOCS-F-18** | DOCS-01 | **`core` crate has 9.0% doc coverage** — 99 substantive `pub` items undocumented. Scheduler submodules (`scheduler/{batch,batch_composer,engine,memory,phase_scheduler,policy,preemption,radix_cache,request_queue}.rs`), KV cache (`prefix_cache.rs`), and `circuit_breaker/`, `ha/`, `routing/`, `error/`, `metrics/` all lack docs. | **P1** | `crates/core/src/` | 6h |
| **DOCS-F-19** | DOCS-01 | **`testing` crate has 12.9% doc coverage** — 12 substantive `pub` items undocumented (RequestBuilder, BatchBuilder, TestFixtures, TestHarnessConfig, etc.). | **P1** | `crates/testing/src/` | 1h |
| **DOCS-F-20** | DOCS-02 | **121 of 232 source files lack `//!` module doc** (52% of files). Critical modules lacking overview: `core/{engine.rs, error/, sampling.rs, scheduler/{batch,batch_composer,engine,memory,phase_scheduler,policy,preemption,radix_cache,request_queue}.rs, types.rs}`, `model/{components/{attention/{gqa,mla}, kv_cache_fp8}, loader/builder.rs, paged_tensor/tensor_store.rs}`, `server/{auth.rs, config.rs, openai/chat.rs}`. | **P1** | all crates | 12-16h (backfill) |
| DOCS-F-21 | DOCS-04 | **`.planning/REQUIREMENTS.md:53` lists "DeepSeek" but no `deepseek/` directory exists in `crates/model/src/`.** Either docs reference a deprecated path or a directory was renamed without updating requirements. | P2 | `.planning/REQUIREMENTS.md:53` | 0.5h |
| DOCS-F-22 | DOCS-05 | **`vllm-dist` is underused and may be candidate for deprecation.** Only `pipeline/` and `tensor_parallel/` have substantive code. `distributed_kv/` is 3 small files. ADR or explicit decision needed: keep investing, or deprecate. | P2 | `crates/dist/src/` | 2h (ADR + decision) |
| DOCS-F-23 | DOCS-03 | **`qwen3_5/speculative_tests.rs:1` references "Phase 5 Wave 4"** — old phase number. Could be reframed as "Qwen3.5 hybrid speculative tests". | P2 | `crates/model/src/qwen3_5/speculative_tests.rs:1` | 0.25h |
| DOCS-F-24 | DOCS-04 | **`.planning/PROJECT.md:229-234` lists "Key Decisions" in the doc but they aren't ADRs** — speculative strategy, FP8 format, etc. are tribal knowledge. Once ADRs are written (DOCS-F-07), they should be cross-linked. | P2 | `.planning/PROJECT.md:219-235` | 0.5h |

---

## Top 3 Action Items

### 1. **Fix README broken code example + update architecture tables** (DOCS-F-02 + DOCS-F-03 + DOCS-F-04 + DOCS-F-09)

The README is the user-facing entry point and currently has three concrete defects that will mislead users:

1. **L448-459** — code example that doesn't compile (3-arg signature required)
2. **L251-268** — lists 5 architectures, missing 5 more (gemma3, llama4, mistral_small, phi4, qwen3_5)
3. **L467, 466-486** — claims 7 crates and shows ~50% of model tree

Fix: update README architecture table, project structure tree, and code example. Cross-reference AGENTS.md. **Effort: 2-3 hours.**

### 2. **Write 4-6 ADRs for major architectural decisions** (DOCS-F-07 + DOCS-F-08 + DOCS-F-22)

For a 19-milestone, 6-crate project, only 2 ADRs is dramatically insufficient. Write ADRs for:

1. **ADR-003** — Self-speculation 1/8 layer ratio (why 1/8, what does it give up/gain, when to deviate)
2. **ADR-004** — FP8 E4M3 format for KV cache compression (why E4M3 over E5M2/INT8)
3. **ADR-005** — KV cache split across 3 locations (logical/physical/quantizer)
4. **ADR-006** — `vllm-dist` strategy (continue investing or deprecate; current state unclear)
5. **ADR-007** — Speculative decoding architecture (registry → resolver → budget → adaptive)
6. **ADR-008** — `cuda` vs `cuda-graph` feature split (why two features)

This makes tribal knowledge durable and gives future contributors (human or AI) documented rationale. **Effort: 8-12 hours.**

### 3. **Back-fill doc comments on public API** (DOCS-F-01)

This is the largest single body of work: 776 `pub` items across 6 crates lack `///`. The 80% target is unreachable in a single phase; prioritize by impact:

1. **`server/` first** — user-facing API surface (chat, completions, embeddings, config). Users hit these directly. **Effort: 4h.**
2. **`model/loader/`** — ModelLoader is the public entry point for loading models (documented in AGENTS.md examples). **Effort: 2h.**
3. **`traits/`** — only 14 items, foundational types. **Effort: 2h.**
4. **`core/{scheduler,error,types,sampling}.rs`** — engine-internal but high-traffic. **Effort: 4h.**
5. **`model/` attention, MLP, norm, positional components** — shared components, well-bounded surface. **Effort: 3h.**

Total: ~15 hours for the highest-impact subset. Full backfill to 80% would require ~30-40 additional hours.

---

## Suggested v20.0+ Phase

**Phase 25 — Documentation Pass** (proposed, advisory)

Combine these into a single documentation-focused phase:

### Phase 25a: External doc accuracy (4-6h)

1. Fix README broken code example (`SchedulerEngine::new` 3-arg signature) — DOCS-F-02
2. Update README architecture table to list all 10 architectures — DOCS-F-03
3. Update README + AGENTS.md to claim 6 crates (not 7) — DOCS-F-04
4. Update README project structure tree — DOCS-F-09
5. Update README test count claims — DOCS-F-06
6. Add `/v1/models` and `/debug/*` endpoints to README — DOCS-F-05
7. Expand README feature flag table — DOCS-F-10
8. Update `.planning/REQUIREMENTS.md:53` to remove DeepSeek or add it back — DOCS-F-21

### Phase 25b: ADR writing (8-12h)

Write ADRs for the 6 decisions listed in Top 3 #2. Use the template from ADR-001/ADR-002.

### Phase 25c: Doc-comment backfill (15-30h, prioritized)

Back-fill `///` on public API in priority order: server → model/loader → traits → core → model. Stop at 80% coverage per crate, OR at time budget — whichever comes first.

### Phase 25d: Module-level docs (12-16h)

Add `//!` module doc to the 121 source files currently missing it. Start with critical-path modules (`engine.rs`, `error/`, `scheduler/`, `kv_cache/`, `loader/`).

### Phase 25e: Stale comment cleanup (1-2h)

- Resolve `quantize/gguf.rs:7` placeholder
- Remove or rephrase "Phase 18.3 will drive this" forward-looking comments
- Reframe `qwen3_5/speculative_tests.rs:1` "Phase 5 Wave 4" reference
- Document cross-references in `.planning/PROJECT.md` "Key Decisions" → ADRs

### Effort estimate

- **Phase 25a**: 4-6h
- **Phase 25b**: 8-12h
- **Phase 25c**: 15-30h (depends on target depth)
- **Phase 25d**: 12-16h
- **Phase 25e**: 1-2h
- **Total**: 40-66 hours

### Risk assessment

- **Low risk**: External doc fixes (25a), ADR writing (25b), stale comment cleanup (25e). These are pure additions/edits to markdown and `///` blocks; no semantic code change.
- **Medium risk**: Doc-comment backfill (25c) and module-level docs (25d). Examples in `///` blocks could go stale if they reference APIs that change. Mitigation: don't add examples, just describe; or add `#[cfg(doc)]`-gated doctests.
- **Combined with**: Phase 23 (API + error handling audit) findings — likely overlap on error type definitions.

---

## Cross-references

- **Phase 20 (Architecture Audit)** — DOCS-F-04 (workspace crate count) and DOCS-F-22 (vllm-dist usage) are likely also flagged in Phase 20 ARCH findings.
- **Phase 21 (Naming Audit)** — DOCS-F-08 (KV cache split) overlaps with NAME-F-21 (KV cache concept split across 3 locations). DOCS-F-12 (`quantize/gguf.rs` placeholder) overlaps with NAME-F-01 (`kv_cache_fp8.rs` orphan module). Both phases touch the same files.
- **Phase 23 (API + Error Handling Audit)** — likely flags undocumented `EngineError` variants (DOCS-F-18 partial overlap). The `SchedulerEngine::new` 3-arg signature (DOCS-F-02) may also be an API-01 consistency concern.
- **Phase 24 (Synthesis)** — should correlate this audit's findings with architecture/naming/API audits to produce a unified remediation backlog. Especially: ADR-007 (speculative architecture) should connect to Phase 23 API surface findings on speculative decoding.

---

## Notable Positive Findings

These are **NOT problems** — they show where the codebase is doing things right and should serve as templates:

- **Excellent TODO/FIXME hygiene** — 0 occurrences across all crates (vs. typical Rust codebases which have dozens). Strong discipline.
- **`crates/core/src/scheduler/mod.rs`** — 133 lines of `//!` module doc with architecture diagram. Excellent template for module-level docs.
- **`crates/core/src/lib.rs`** — clear engine overview with module map. Excellent template for crate-level docs.
- **`crates/core/src/speculative/{draft_registry,draft_resolver,memory_budget}.rs`** — well-documented speculative subsystem. Excellent template for documenting a single feature cluster.
- **`crates/model/src/arch/{mod,registry}.rs`** — well-documented Architecture trait and registry. Excellent template for trait + registry patterns.
- **`crates/testing/src/lib.rs`** — extensive test docs. Excellent template for testing crates.
- **`ADR-001` and `ADR-002`** — both follow the standard Context / Decision / Rationale / Consequences format. Excellent template for future ADRs.
- **No broken intra-doc links detected** in spot-check of v18.0-era comments (DraftResolver, DraftModelRegistry, etc. all resolve).

---

*End of SUMMARY.md*
