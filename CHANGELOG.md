# 📋 Changelog

<p align="center">
  <img src="https://img.shields.io/badge/Keep%20a%20Changelog-1.0.0-blue.svg?style=flat-square" alt="Keep a Changelog">
  <img src="https://img.shields.io/badge/Semantic%20Versioning-2.0.0-green.svg?style=flat-square" alt="Semantic Versioning">
</p>

> All notable changes to **vLLM-lite** will be documented in this file.

---

## 📊 Release Statistics

|     版本     |    日期    |          测试          | 覆盖率 (raw / real) |
| :----------: | :--------: | :--------------------: | :-----------------: |
| [Unreleased] |     -      | 1235 (post-Phase-5 module-splitting) | 55.0% / 49.9% (Phase N baseline, `--real` excludes test/hidden/derive) |
|   [v22.0]    | 2026-06-27 |         1179+          | ~50% real (97.8% figure was placeholder-based) |
|   [v21.0]    | 2026-06-27 |         1146+          | ~50% real |
|   [v20.0]    | 2026-06-27 |         1144+          | ~50% real |
|   [v19.0]    | 2026-06-27 |         1139+          | ~50% real |
|   [v18.0]    | 2026-06-27 | 277 (vllm-core) + 654+ |  90%+  |

---

## 🚀 [Unreleased]

### Added

- **Tutorial & Onboarding (v30.0 Phase P)** — guided path from clone to production:
    - `docs/tutorial/01-setup.md` — clone, build, verify (Rust 1.85+, `cargo build --workspace`, `just nextest`)
    - `docs/tutorial/02-load-model.md` — `ModelLoader::builder()` usage, supported formats (safetensors, GGUF Q4_K_M)
    - `docs/tutorial/03-inference.md` — request lifecycle (`add_request` → `build_batch` → `forward` → `update`), prefill/decode phases, continuous batching
    - `docs/tutorial/04-customize.md` — custom sampling + scheduling strategies, property-based testing pattern
    - `docs/tutorial/05-production.md` — docker-compose, Kubernetes (`k8s/`), Prometheus/Grafana, security checklist, rollback strategy
    - `crates/server/tests/tutorial_e2e.rs` — end-to-end test using `StubModelBackend` + the public actor API (`Engine::run` + `EngineMessage::AddRequest`/`Shutdown`); 2 tests passing
    - `CONTRIBUTING.md` — new `## Tutorials` section linking all 5 lessons
    - `README.md` — tutorial pointer added in Quick Start callout and 文档 list
    - Total commits: 8 (P-1.1 through P-2.4 + CHANGELOG)
    - **Honest scope notes**: tutorial code examples use `no_run` and present the *conceptual* request lifecycle pattern; some specific API calls in tutorial 3/4 (e.g. `engine.build_batch()`, `engine.update()`) describe the conceptual flow but do not match the current public API verbatim. The integration test uses the real public API (`Engine::run` + `EngineMessage`) and serves as a living, executable example. Future work (Phase Q?) should reconcile tutorial code with the actual API surface.

- **Doc Coverage Push (v30.0 Phase N)** — partial progress with honest baseline:
    - `scripts/doc_coverage.sh --real` flag added (backward compatible)
        - New columns `RealTot`, `RealDoc`, `Real%` exclude `#[cfg(test)]` mod
          blocks, `#[doc(hidden)]` items, and `#[derive(...)]`-generated items
          from both the total and the documented count
        - Implementation: `scripts/_blank_for_real.py` (Python helper that
          preserves line numbers so file:line attribution is retained)
        - Default (raw) mode unchanged
    - `///` docs added to ~88 high-value pub items across `vllm-core` and
      `vllm-traits` (prioritized user-facing API):
        - Engine lifecycle (`is_healthy`, `get_last_error`, `cancel_request`,
          `add_request`, `run`, `has_pending`)
        - Engine construction (`new`, `new_boxed`, `with_config`, `EngineBuilder`)
        - Engine CUDA Graph paths (`capture_cuda_graphs`, `cuda_graph_enabled`,
          `step_with_graph`)
        - Sampling (`top_k_sample`, `sample_batch`, `apply_repeat_penalty`,
          full `SamplingParams` + `SamplingParamsBuilder` + `Request` API)
        - Beam search (`BeamSequence`, `SchedulerConfig` builder)
        - Scheduler (`RequestQueue`, `SchedulerStats`, `PhaseScheduler`,
          `GraphPreparedBatch`, builders for `PhaseSwitchPolicy` and
          `SchedulerCudaGraphConfig`)
        - Metrics (`LockFreeMetrics` record_* methods + snapshot, `MetricValue`)
        - Speculative (`DraftSpec` builder methods, `AdaptiveSpeculativeDecoder`)
        - Server (`chat_completions`, `completions`, `embeddings` handlers,
          `health_details`, `shutdown`, `get_prometheus`, `HealthStatus::http_status`,
          `AppConfig::load`/`validate`, `AuthConfig::resolve_api_keys`)
        - Types (`EngineMessage` variants, `Priority`, error module-level docs)
    - **Coverage numbers (before → after)**:
        - Raw: 49.8% → 55.0% (+89 items documented, 1708 → 939 undocumented)
        - Real (`--real`): 44.0% → 49.9% (+88 items, after filtering)
        - Module docs: 54.0% → 54.7%
    - **Honest historical context**: the v23.0 audit CHANGELOG entry claimed
      "97.8% doc coverage", but this was based on placeholder `/// Doc.`
      comments that were counted as documented. v23.0 Phase 42 removed 1062
      of those placeholders, dropping real coverage dramatically. The 99%
      target in the Phase N plan was based on the stale 97.8% metric and is
      not achievable in one session — reaching 90%+ requires documenting
      hundreds of additional items across `vllm-model`, `vllm-dist`, and the
      remaining `vllm-core`/`vllm-server` surface.
    - Total commits: 6 (N-1 metric, N-2..N-5 docs across 4 batches,
      N-6 CHANGELOG)

- **Test Coverage Expansion (v30.0 Phase M)** — 4 new fuzz targets + 3 new proptest modules:
    - Fuzz targets (7 total now, was 3 in v29.0):
        - `tokenizer_decode`: fuzz `tiktoken::CoreBpe::decode` / `decode_to_string` with arbitrary u32 token IDs (cl100k vocabulary). Validates decoder does not panic on out-of-range IDs.
        - `gguf_header`: fuzz GGUF magic check + version field read with arbitrary bytes. Catches panics in the header slice/compare path.
        - `openai_http_request`: fuzz `serde_json::from_slice::<ChatRequest>` with bounded (1MB) arbitrary bytes. Catches deserialization panics in the OpenAI HTTP endpoint.
        - `batch_json_input`: fuzz `serde_json::from_slice::<SimpleBatchRequest>` with bounded (10MB) arbitrary bytes. Catches deserialization panics in the batch API.
    - proptest modules (7 total now, was 4 in v28.0):
        - `SamplingStrategy` (`crates/core/src/sampling.rs::prop_tests`): 4 properties — `sample_batch` length preservation, `greedy_sample` index-in-bounds, `sample_batch` greedy matches per-row `greedy_sample`, `apply_repeat_penalty(1.0)` is a no-op
        - `EvictionPolicy` (`crates/core/src/scheduler/memory/eviction.rs::prop_tests`): 3 properties — refcount conservation across record/release cycles, `select_victims` length bound + empty-input invariant, cache-hit path on identical inputs
        - `PriorityPolicy` (`crates/core/src/scheduler/policy/priority.rs::prop_tests`): 3 properties — higher user priority → lower score, aging reduces score for older sequences, PriorityScore is bounded for arbitrary u8/u64 inputs
    - `fuzz/Cargo.toml` updated with `tiktoken = "3"` dependency for `tokenizer_decode`
    - All new targets build successfully under nightly + ASAN; all proptests pass at PROPTEST_CASES=100
    - All 4 new proptests exercise real production code paths (sample_batch, greedy_sample, apply_repeat_penalty, EvictionPolicy::{record_blocks, release_blocks, select_victims, get_block_ref_count, stats}, PriorityPolicy::compute_priority)
    - Total commits: 8 (M-1.1, M-1.2, M-1.3, M-1.4, M-2.1, M-2.2, M-2.3, M-3.2 this entry)

- **Fuzz CI Integration (v30.0 Phase L)** — fuzz-smoke + nightly long-run workflows:
    - `.github/workflows/fuzz.yml` — PR-triggered, 30s × 3 targets, corpus cached via `actions/cache`, crash artifacts auto-uploaded
    - `.github/workflows/fuzz-nightly.yml` — cron + manual dispatch, 5min × 3 targets, separate corpus cache, grown-corpus artifact upload
    - `just fuzz-repro TARGET CRASH` — local crash artifact replay
    - `docs/fuzz.md` — methodology + CI workflow + corpus management + crash handling
    - CI budget: PR workflow ~3-5 min/target, nightly ~15 min total — within GitHub free tier
    - Total commits: 6 (L-1.1, L-1.2, L-2.1, L-2.2, L-3.1, L-3.2)

- **Mutation Testing (v30.0 Phase K)** — cargo-mutants infrastructure + 1 real bug fixed:
    - `cargo-mutants v27.1.0` installed as standalone tool
    - justfile targets: `mutants MODULE`, `mutants-report`, `mutants-clean`, `mutants-score`, `mutants-ci MODULE BASELINE`
    - `scripts/check_mutation_score.sh` regression checker
    - Baseline scans across 4 modules: 907 mutants total, 100% mutation score strict, 0 missed
    - **Real bug found & fixed**: `Engine::cuda_graph_enabled` mutation not caught in non-cuda-graph build → added cfg-gated test `test_cuda_graph_disabled_when_feature_off`
    - Baseline reports: `docs/testing/mutation-{scheduler,sampling,speculative,engine}-baseline.md`
    - Methodology: `docs/testing/mutation-testing.md`
    - CI integration deferred to v31 (scan time + `--baseline skip` workaround)
    - Total commits: 9 (K-1.1 through K-3.2)

- **Fuzz Testing (v29.0)** — cargo-fuzz infrastructure + 3 fuzz targets:
    - `cargo-fuzz 0.13.2` scaffolded at `fuzz/` directory; nightly Rust toolchain required for sanitizer flags
    - `app_config_yaml`: fuzz `serde_saphyr::from_str::<AppConfig>` with arbitrary UTF-8 bytes
    - `safetensors_header`: fuzz `SafeTensors::deserialize` with arbitrary bytes
    - `qwen3_config_json`: fuzz `serde_json::from_slice::<Qwen3Config>` with arbitrary bytes
    - **Bugs found**: 0 across ~17.6M executions (3 targets × 60s each: 751k + 8.77M + 8.13M)
    - `justfile` targets: `fuzz-build`, `fuzz-smoke`, `fuzz TARGET`, `fuzz-list`
    - Test count: 1212 passed (fuzz targets run on-demand, not in `cargo test`)
    - Total commits: 5 (J-1 to J-5)

- **Property-Based Testing (v28.0)** — proptest infrastructure + invariants:
    - `proptest 1.11` added as workspace dev-dep
    - 4 components covered with 18 properties total:
        - RadixTree (3 props): insert+lookup round-trip, longest-prefix bound, insert+clear
        - BlockAllocator (3 props): allocation uniqueness, LIFO reuse, capacity bounding
        - RequestQueue (4 props): enqueue+remove round-trip, get-after-enqueue, FIFO order, phase index consistency
        - BatchComposer (7 props): batch size bound, token budget, parallel-vec consistency, decode token count, prefill total_tokens, deterministic compose, seq_id uniqueness
    - **Bug fix found by property tests**: `compose_decode_batch` panicked on empty-token sequences due to `tokens_len - 1` underflow (position computation + `num_computed_tokens`); fixed via `saturating_sub(1)`. Regression test added.
    - All properties pass at PROPTEST_CASES=100 (100 cases × 18 properties = 1800 generated test cases per run)
    - All existing tests still pass (1194+)
    - Total commits: 5 (I-1, I-2, I-3, I-4, I-5) + 1 (CHANGELOG)

- **MambaBlock Weight Loading**
    - Added `MambaBlock::from_weights` method to load SSM layer weights
    - Implemented full weight loading for Qwen3.5 Mamba models
    - Supports fallback for embed_tokens and lm_head weight names
    - Supports tied embeddings (tie_word_embeddings)

### Changed

- **Performance Optimization (v27.0)** — profile-driven speedups across attention + cache + scheduler:
    - **Measurement infrastructure**: 4 new model-layer criterion benches (GQA, MLA, FlashAttn, PagedKV); runtime CUDA detection so benches run real qwen3-7B dimensions on GPU and tiny smoke test on CPU + eprintln warning. `just bench-model` / `just bench-model-one BENCH` for invocation.
    - **Profiling**: pprof dev-dep + profiling guide; static analysis reports for 6 components identifying 39 hotspots total.
    - **H-11 GQA**: affine scale tensor (-2.5% CPU), redundant `.contiguous()` after softmax removal; `expand_kv` materialization skip deferred (requires custom fused GQA matmul kernel).
    - **H-12 FlashAttn + MLA**: affine scale in 5 FlashAttn sites + 1 MLA site (-3 to -7.5% CPU); redundant `.contiguous()` after MLA softmax.
    - **H-13 PagedKV + BatchComposer**: Tensor::cat → slice_assign for layer-rebuild (+17.8% CPU expected; GPU should win from eliminating 1024 kernel launches); BatchComposer prefill Vec::with_capacity, sort_unstable_by_key (-16% scheduler_build_batch); bug fix: chunked_prefill `num_computed_tokens` was non-`mut`.
    - **Correctness hardening**: GQA + MLA forward() `# Caution: No causal masking` doc blocks (forward() is intentionally unmasked low-level primitive; production routes through forward_prefill/forward_decode which apply causal). Regression test for determinism.
    - **Bug fix**: `engine.step()` infinite loop in `speculative_vs_baseline` + `optimization_benchmarks/throughput` — added step cap (MAX_STEPS_PER_ITER = 10_000).
    - **Bench infrastructure fixes**: 4 previously-orphaned core benches (`scheduler`, `scheduler_benchmarks`, `prefix_cache_benchmarks`, `optimization_benchmarks`) wired into Cargo.toml with `harness = false`.
    - **Deferred**: paste RUSTSEC-2024-0436 accepted (candle-core 0.11.0 still depends on `gemm → paste`; INFO severity; suppressed via `just audit` `--ignore`).
    - **GPU baseline captured**: `gqa_forward/standard/512` 937µs, `mla_forward/512` 1ms, `flash_attention/b1_h14_s2048_d64` 29.7ms, `paged_kv_cache/blocks1024` 1.8ms. Future A/B comparison possible.
    - **Test count**: 1194 passed (was 1189 before H-13 bug fix); 41 skipped (was 39); 0 failed; `just ci` clean.
    - Total commits: 16 (H-1 through H-15 + correctness investigation + doc hardening)
    - **Deferred optimizations** (separate specs needed): expand_kv fused kernel, FlashAttn tiled output buffer, BatchComposer kv_blocks Arc clone (cross-crate API), PagedKV host round-trip elimination.

- **Security & Dependency Updates (v26.0)** — addressed 6 GitHub Dependabot vulnerabilities + fixed CI:
    - **H-1 `rustls-pemfile` RUSTSEC-2025-0134 (high)** — `tls.rs` migrated to `rustls::pki_types::PemObject` (built-in since rustls 0.23); deprecated crate removed
    - **M-2 `tower-http` outdated** — workspace-unified to 0.7 (`0.5` dist + `0.6` server → `0.7` all); forced axum 0.8 upgrade as chain reaction
    - **M-3 `serde_yaml` deprecated** — migrated to pure-Rust `serde-saphyr = 0.0.27` (panic-free, Miri-tested, no `unsafe` code, no libyaml C dependency); 3 call sites updated (`config.rs:260/271`, `bin/vllm.rs:83`); supersedes the `serde_norway` choice (which still uses libyaml via `unsafe-libyaml-norway`); drop-in API compat with `serde_yaml::from_str`
    - **M-4 `tokio-rustls` outdated** — audit assumed 0.27 was available but registry only has 0.26.x; deferred until upstream releases 0.27
    - **M-5 `aws-lc-rs` outdated** — bumped 1.16.3 → 1.17.0 (transitive via tokio-rustls)
    - **Patch sweep (F-1)** — `cargo update` minor bumps for 50 deps (most already current from v22/v23); net Cargo.lock change is 56 package re-locks + 16 stale transitive removals
    - **Minor security bumps (F-2)** — `tiktoken 3.1.4 → 3.5.1` (model crate); `hyper 1.9.0 → 1.10.1` (transitive via tonic)
    - **CI workflow fix (F-4)** — removed broken `--all-features` from default `cargo clippy` job (no CUDA in default GitHub runners); switched to per-group denies matching local `just clippy`; added follow-up const fix for `Qwen3Fixture::with_kv_blocks` with targeted allow
    - **Deferred to v27.0+ (F-3d)**: `paste` (RUSTSEC-2024-0436) unmaintained — INFO severity only (no vuln, no patch available); verified `candle-core 0.11.0` (latest) still depends on `gemm → paste`, so upgrade does NOT resolve. Disposition: accepted risk; `just audit` uses `--ignore RUSTSEC-2024-0436`, documented in SECURITY.md
    - `cargo audit` warnings: 2 → 0
    - `tower` workspace skew resolved: `0.4` workspace + `0.5` server → `0.5` workspace
    - All 1191 tests pass (39 skipped, 1 slow)

- **Pedantic Cleanup (v25.0 Phase E-3)** — manual refactors + selective deny promotion:
    - 109 `use_self` candidates: most were already resolved by E-1/E-2; only 1 residual doc-markdown fix in `chat_template.rs`
    - 220 `module_name_repetitions` warnings: 117 files received `#![allow(clippy::module_name_repetitions)]` for legitimate patterns (`KvCache` in `kv_cache`, `MetricsExporter` in `metrics/exporter`, etc.)
    - 7 `return_self_not_must_use` builders tagged `#[must_use]`: `MetricsSnapshot::with`, `DraftSpec::with_arch_hint`, `Request::with_draft_model`, `CausalLM::with_embed_through_layers`, `JwtConfig::with_issuer`/`with_audience`, `TlsConfig::with_ca_cert`
    - 2 `missing_const_for_fn` warnings in tonic-generated gRPC code suppressed via `mod generated_proto` allow list in `crates/dist/src/grpc.rs`
    - Promoted 7 lints from `warn` → `deny` in `[workspace.lints.clippy]`: `module_name_repetitions`, `missing_errors_doc`, `missing_panics_doc`, `uninlined_format_args`, `must_use_candidate`, `return_self_not_must_use`, `missing_const_for_fn`
    - Wholesale `pedantic`/`nursery` promotion deferred: ~500 pedantic/nursery warnings remain (mostly `float_cmp` in tests, `unreadable_literal` for tokenizer vocab IDs, `significant_drop_tightening`) and would require per-file allows to deny-enforce
    - Pedantic warning count: 1210 → 982 (`-W pedantic`); default `just clippy` warning count: 509 → 500
    - `just ci` passes (1191 tests, 0 failures)

- **Workspace Lint Policy (v24.0 Phase A)** — established three-tier clippy lint
  configuration in root `Cargo.toml` `[workspace.lints.clippy]`:
    - **deny tier**: `correctness`, `suspicious`, `perf` (breaks CI)
    - **warn tier**: `pedantic`, `nursery`, `missing_errors_doc`, `must_use_candidate`, etc. (visible but not blocking)
    - **allow tier**: `cast_precision_loss`, `similar_names`, `too_many_lines`, `too_many_arguments`, etc. (with rationale)
    - All 6 crates inherit via `[lints] workspace = true`.
    - `just clippy` switched from `-D warnings` to explicit per-group denies so pedantic stays visible without breaking CI.
    - New `just clippy-pedantic` recipe for local pedantic+nursery view.
    - `AGENTS.md` gained a "Lint Policy" section documenting the tier system, local commands, and the rationale for each allow-list entry.

- **Unwrap Cleanup (v24.0 Phase B)** — fixed real bug risk and improved error reporting:
    - **B-1**: `cuda_graph/executor.rs:222` race condition unwrap → typed `GraphNotFound` error via new `lookup_graph` helper
    - **B-2**: 5 production unwraps in `engine.rs`, `main.rs`, `handler.rs` → typed error variants / cleaner panic messages
    - **B-3**: 48 production `// invariant:` comments added across 25+ files documenting legitimate invariants (RwLock/Mutex poison, SystemTime, Tensor allocation, signal handlers, etc.)
    - Baseline audit: spec originally claimed 787 production unwraps; actual was 60 (the 787 figure included inline `#[cfg(test)] mod tests` blocks). Spec target `≤160` was already met; Phase B reduced production unwraps to ~54.
    - New `EngineError::EmptyBeamList` variant added
    - `AGENTS.md` gained an "Invariant comments" section documenting the convention

- **API Ergonomics (v24.0 Phase C-1)** — added builder pattern for `Engine` and crate-root re-exports:
    - New `vllm_core::EngineBuilder` allows named-method construction of `Engine` with all optional fields (`with_draft_model`, `with_config`, `with_max_draft_tokens`, `with_num_kv_blocks`, `with_adaptive_decoder`, `with_draft_resolver`, `with_sleep_policy`)
    - Existing `Engine::new_boxed()` and `Engine::with_config_boxed()` remain unchanged (non-breaking)
    - `vllm-core` re-exports commonly-used types at crate root: `EngineBuilder`, `EngineError`, `Result`, `Request`, `SchedulerConfig`, `AdaptiveSpeculativeDecoder`, `DraftModelRegistry`, `DraftResolver`, `DraftSpec`, `SamplingParams`, etc.
    - `vllm-model` re-exports: `Architecture`, `ModelConfig`, `ModelLoader`, `ModelLoaderBuilder`, `Tokenizer`
    - `vllm-server` re-exports: `AuthConfig`, `AuthMiddleware`, `BatchManager`, `BatchResponse`, `AuditEvent`, `HealthChecker`, `HealthStatus` (intentionally excludes OpenAI types to avoid root namespace collision)

- **Stringly-typed Enums (v24.0 Phase C-2)** — replaced 3 string-typed public APIs with typed enums:
    - `DraftResolutionKind` enum (`External`, `SelfSpec`, `None`) replaces `&str` in `EnhancedMetricsCollector::inc_draft_resolution` (actual values: `"external"`, `"self_spec"`, `"none"`)
    - `RopeType` enum (`Default`, `Linear`, `Dynamic`, `Yarn`, `Su`, `Other` — serde `lowercase`) replaces `Option<String>` in `RopeScaling::rope_type` and `RopeParameters::rope_type`
    - `BatchEndpoint` enum (`Chat`, `Completion`) replaces `String` in batch request/response/job endpoint fields, with custom serde serializer/deserializer to preserve JSON wire compatibility
    - All conversions provide `parse(&str) -> Option<Self>`, `as_str() -> &'static str`, and `Display` impl
    - Affected call sites updated atomically (13 + 1 + 6 across the three enums)

- **Object-safe Trait Default Constructors (v24.0 Phase C-3)** — added `default_arc()` methods for 11 object-safe public traits:
    - **4 high-ROI**: `DraftVerifier` (`StubDraftVerifier`), `ModelBackend` (`StubModelBackend`), `SchedulerObserver` (`NoopSchedulerObserver`), `MetricsExporter` (`InMemoryMetricsExporter`)
    - **7 medium-ROI**: `SchedulingPolicy` (reused `FcfsPolicy`), `DraftLoader` (reused `NoopLoader`), `CudaGraphTensor` (`NullCudaGraphTensor`), `CudaGraphNode` (`NullCudaGraphNode`), `AllReduce` (`NoopAllReduce`), `PipelineStage` (`NoopPipelineStage`), `Architecture` (`UnknownArchitecture`)
    - Each trait gains `<dyn Trait>::default_arc() -> Arc<Self>` (Rust orphan rule prevents `impl Default for Arc<dyn Trait>`; the inherent-method pattern is the standard workaround)
    - Callers use `Arc::<dyn Trait>::default_arc()` (via type inference) or explicit `<dyn Trait>::default_arc()`
    - 6 low-ROI traits deferred (never used as `Arc<dyn Trait>` in current code)

- **Module Boundaries (v24.0 Phase D-1)** — split `crates/core/src/engine.rs` (866 non-test LOC) into 7 focused sub-modules under `engine/`:
    - `mod.rs` — facade: `Engine` struct, `SleepPolicy`, tests (437 LOC)
    - `ctor.rs` — constructors (`new_boxed`, `with_config_boxed`, `with_drafts_*`, `with_budget_boxed`) + `EngineBuilder` (318 LOC)
    - `draft_management.rs` — draft registry, resolver, speculative-mode toggles (124 LOC)
    - `cuda_graph.rs` — `capture_cuda_graphs` + `cuda_graph_enabled` (cfg-gated pairs) (35 LOC)
    - `lifecycle.rs` — `is_healthy`, `get_last_error`, `cancel_request`, `add_request` (38 LOC)
    - `run.rs` — `run` main loop + `has_pending` (74 LOC)
    - `beam.rs` — `step_beam` + `beam_search` + `get_top_k` (111 LOC)
    - `graph_step.rs` — `step_with_graph` (cfg-gated pair) + `execute_regular` + `process_output` (155 LOC)
    - Public API of `Engine` unchanged — single `crate::engine::Engine` type, all methods accessible
    - All 1191 tests pass (`just ci` clean)
    - Three `#[cfg(feature = "cuda-graph")]` / `#[cfg(not(feature = "cuda-graph"))]` duplicate method pairs (`capture_cuda_graphs`, `cuda_graph_enabled`, `step_with_graph`) preserved as intentional feature-gated pairs
    - Largest sub-module: `ctor.rs` at 318 LOC (was 866 LOC monolith); all sub-modules < 500 LOC

- **Module Boundaries (v24.0 Phase D-2)** — split `crates/core/src/scheduler/engine.rs` (654 non-test LOC) into 4 focused sub-modules under `scheduler/engine/`:
    - `mod.rs` — facade: sub-module declarations, `pub use` re-exports, the 8 unit tests (172 LOC)
    - `state.rs` — `SchedulerEngine` struct + `Default` impl + 17 methods: `new`, `set_policy`, `add_request`, `build_batch`, `schedule`, plus the 10 read-only / minor-mutating accessors (`has_pending`, `running_count`, `waiting_count`, `prefix_cache_hit_rate`, `running`, `get_sequence`, `get_sequence_mut`, `finished_sequences`, `clear_finished`, `register_observer`) (404 LOC)
    - `graph.rs` — CUDA Graph helpers: `build_batch_with_graph` + 2 private helpers (`get_scheduler_state`, `select_sequences_for_phase`) (81 LOC)
    - `update.rs` — post-step state update: `update` (121 LOC)
    - `memory.rs` — preemption + pressure: `execute_preemption`, `get_memory_pressure`, `memory_rollback`, `cancel_request`, `get_kv_cache_usage`, `prefix_cache` (114 LOC)
    - Public API of `SchedulerEngine` unchanged — single `crate::scheduler::engine::SchedulerEngine` type, all methods accessible via flat namespace
    - All 1191 tests pass (`cargo test --workspace` clean)
    - Largest sub-module: `state.rs` at 404 LOC (was 654 LOC monolith); the struct + 6 large lifecycle methods concentrate here. `graph.rs`, `update.rs`, and `memory.rs` are all ≤ 121 LOC.

- **Module Boundaries (v24.0 Phase D-3a)** — split the two remaining hard-target files > 500 LOC:
    - `crates/core/src/types.rs` (538 non-test LOC) → 7 sub-modules under `types/`:
        - `mod.rs` — facade: re-exports `vllm_traits::{Batch, BatchOutput, BlockId, SeqId, TokenId}` and `DraftId` (21 LOC)
        - `adaptive_draft.rs` — `AdaptiveDraftConfig` + `AdaptiveDraftConfigBuilder` (90 LOC)
        - `request.rs` — `Priority` + `Request` (55 LOC)
        - `sampling.rs` — `SamplingParams` + `SamplingParamsBuilder` (76 LOC)
        - `sequence.rs` — `Sequence` + `Status` + `Phase` (47 LOC)
        - `sequence_packing.rs` — `SequencePackingConfig` + builder + `from_env` (90 LOC)
        - `scheduler_config.rs` — `SchedulerConfig` + `SchedulerConfigBuilder` (175 LOC)
        - `messages.rs` — `EngineMessage` enum (23 LOC)
    - `crates/model/src/components/ssm.rs` (568 LOC) → 5 sub-modules under `components/ssm/`:
        - `mod.rs` — facade: re-exports + 7 unit tests (82 LOC)
        - `config.rs` — `SSMConfig` (48 LOC)
        - `layer.rs` — `softplus` helper + `SSMLayer` (137 LOC)
        - `mamba.rs` — `MambaBlock` (138 LOC)
        - `harmonic.rs` — `SSMHarmonicSSMLayer` (185 LOC)
        - `error.rs` — `SSMError` + `From<Infallible>` impl (16 LOC)
    - Public APIs unchanged: `crate::types::{Priority, Request, SamplingParams, SchedulerConfig, ...}` and `crate::components::ssm::{SSMConfig, SSMLayer, MambaBlock, SSMHarmonicSSMLayer, SSMError, softplus, ...}` still work via flat re-exports
    - All 1191 tests pass (`cargo test --workspace` clean)
    - Largest sub-module: `ssm/harmonic.rs` at 185 LOC (was 568 LOC monolith); all sub-modules ≤ 185 LOC

- **Module Boundaries (v24.0 Phase D-3b)** — split 7 soft-target files (224-907 LOC band) into focused sub-modules:
    - `crates/server/src/cli.rs` (548 LOC) → 2 sub-modules under `cli/`:
        - `mod.rs` — facade with re-exports (3 LOC)
        - `args.rs` — `CliArgs`, `ModelArgs`, validation helpers, `LogLevel` (548 LOC)
    - `crates/core/src/metrics/collector.rs` (521 LOC) → 3 sub-modules under `metrics/collector/`:
        - `mod.rs` — facade (11 LOC)
        - `metrics.rs` — `DraftResolutionKind`, `DraftMetricsSnapshot` (74 LOC)
        - `sampler.rs` — `EnhancedMetricsCollector` struct + impl + tests (470 LOC)
    - `crates/model/src/components/gated_delta/mod.rs` (581 LOC) → 3 sub-modules under `components/gated_delta/`:
        - `mod.rs` — facade (7 LOC)
        - `state.rs` — `GatedDeltaConfig` + `GatedDeltaState` (67 LOC)
        - `rule.rs` — `GatedDeltaNet` + helpers + tests (529 LOC)
    - `crates/model/src/qwen3/config.rs` (631 LOC) → 3 sub-modules under `qwen3/config/`:
        - `mod.rs` — facade (7 LOC)
        - `rope.rs` — `RopeType` + `RopeScaling` + `RopeParameters` (176 LOC)
        - `model.rs` — `TextConfig` + `Qwen3Config` + `AttentionType` (470 LOC)
    - `crates/core/src/scheduler/batch_composer.rs` (672 LOC) → 3 sub-modules under `scheduler/batch_composer/`:
        - `mod.rs` — facade (14 LOC)
        - `validate.rs` — `BatchCompositionConfig` + `ChunkedPrefillConfig` + builders (150 LOC)
        - `compose.rs` — `BatchComposer` + impl + tests (532 LOC)
    - `crates/model/src/paged_tensor/tensor_store.rs` (828 LOC) → 4 sub-modules under `paged_tensor/tensor_store/`:
        - `mod.rs` — facade with `PagedKvCache` struct + `new()` (68 LOC)
        - `buffer.rs` — `write_kv` / `read_kv` / `write_kv_batch` + tests (666 LOC)
        - `layout.rs` — hash + scale + `block_size` accessors (58 LOC)
        - `pool.rs` — `CacheBlock` + `KvCachePool` (71 LOC)
    - `crates/model/src/kernels/flash_attention.rs` (907 LOC) → 4 sub-modules under `kernels/flash_attention/`:
        - `mod.rs` — facade (15 LOC)
        - `config.rs` — `AttentionVariant` + `FlashAttentionConfig` + tile-size helpers (81 LOC)
        - `util.rs` — `AttentionStats` + `softmax_last_dim` (35 LOC)
        - `kernel.rs` — `FlashAttention` trait + `ScaledDotProductAttention` + `FlashAttentionV2` + `FlashAttentionKernel` + tests (809 LOC)
    - Public APIs unchanged across all 7 splits: external callers continue to import via flat namespace (e.g. `crate::cli::CliArgs`, `crate::components::gated_delta::GatedDeltaNet`, `crate::paged_tensor::tensor_store::PagedKvCache`)
    - All 1191 tests pass (`cargo test --workspace` clean)
    - Largest remaining sub-module: `flash_attention/kernel.rs` at 809 LOC (was 907 LOC monolith); all other sub-modules ≤ 666 LOC. The `kernel.rs` file is large because the `FlashAttentionV2` causal-mask + standard forward paths and the SDPA tiled/sliding-window paths are all in one place; further decomposition would require trait extraction beyond the scope of this phase.

- **Module Boundaries (v24.0 Phase D-3c)** — visibility tightening and re-export cleanup:
    - ~101 `pub` items → `pub(crate)` across crates (model: ~59, core: ~25, server: ~13, testing/dist: ~4)
    - 6 `pub mod` → `pub(crate) mod` (architecture modules in `model/` only consumed via the `Architecture` trait: `gemma3`, `gemma4`, `llama4`, `mistral_small`, `phi4`, `mixtral`). The other 4 architecture modules (`llama`, `mistral`, `qwen3`, `qwen3_5`) remain `pub` because integration tests use direct path access
    - 5 deep re-export chains flattened (added flat re-exports in intermediate modules; collapsed 2 thin re-export shims in `qwen3_5`)
    - 7 glob re-exports → explicit lists (`model/components/mod`, `model/components/ssm/mod`, `core/types/mod`, `core/scheduler/engine/mod`, `core/speculative/draft_registry`, plus consolidated `dist/lib.rs` `tensor_parallel` group)
    - 2 duplicate re-exports deleted (redundant `qwen3_5::hybrid` shim; `PipelineStageTrait` alias)
    - 4 additional types tightened via method visibility reduction: `GraphStats`, `GdnLinearConfig`, `AttentionConfigBuilder`, plus internal `validator::validate_chat_request` and `chat_template::build_prompt` in `server`
    - Conservative scope: items in public method signatures, axum handler parameter/return types, OpenAI DTOs, tonic-generated proto types, and crate-root re-exports remain `pub`
    - All 1191 tests pass (`cargo test --workspace` clean); build and `cargo clippy --workspace --all-features` clean

- **Pedantic Cleanup (v25.0 Phase E-1)** — mechanical fixes:
    - `cargo clippy --fix` applied (uninlined_format_args, redundant_closure, redundant_pub_crate, redundant_field_names, missing_const_for_fn, etc.) across 261 files
    - `#[must_use]` added to ~490 candidates (cargo clippy --fix auto-applied all of them)
    - `#[derive(Debug)]` added to ~124 types — zero `missing_debug_implementations` warnings remain
    - For types with `dyn Trait` fields (ModelBackend, SchedulingPolicy, etc.), manual `impl Debug` was added that displays a placeholder string instead of attempting to format the trait object
    - Pedantic warning count: 3605 → 1496 (-59%)
    - All 1191 tests pass (`just ci` clean)
    - Top remaining lints: `missing_errors_doc` (249), `module_name_repetitions` (225), `cast_possible_truncation` (128), `cast_precision_loss` (121), `unreadable_literal` (96), `significant_drop_tightening` (61), `float_cmp` (60) — most are in the "manual" or "already-allow" categories from the Phase E audit and are deferred to later sub-phases

- **Pedantic Cleanup (v25.0 Phase E-2)** — doc comment cleanup:
    - 4 `doc_markdown` backtick warnings fixed (2 in `chat_template.rs`, 2 in gRPC-generated code suppressed via module-level `#[allow]`)
    - 244 `# Errors` sections added across `vllm-core`, `vllm-model`, `vllm-server`, `vllm-dist`, and `vllm-traits`
    - 36 `# Panics` sections added across `vllm-core`, `vllm-model`, `vllm-server`, and `vllm-dist`
    - Generated proto code in `vllm-dist/src/grpc.rs` wrapped in a `generated_proto` module with `#[allow(clippy::doc_markdown, clippy::missing_errors_doc, clippy::missing_panics_doc)]` since the source is regenerated by `tonic_build` and not under our control
    - All `# Errors` / `# Panics` sections describe the specific failure conditions (e.g. "Returns `Err` if any tensor operation fails (shape mismatch, out-of-memory, dtype incompatibility, or kernel error)" rather than boilerplate)
    - Pedantic warning count: 1496 → 1210 (-19% for this phase; the remaining top lints are `must_use_candidate`, `module_name_repetitions`, `cast_*`, `unreadable_literal`, `significant_drop_tightening` which are deferred to later sub-phases)
    - All 1191 tests pass (`just nextest` clean)
    - `just clippy`, `just fmt-check`, and `just doc-check` all pass

- **Comprehensive Refactor (Phase 1 + Phase 2)** — production-grade hardening
  across CI, error handling, observability, and the public API surface.
  Follows the four parallel audits (architecture, error handling, testing,
  CI/CD) that ran in this session. 7 atomic commits, 1230+ tests passing.

  **Phase 1: Infrastructure (CI/CD + Engineering Practices)**
    - **CI audit hardening** (audit findings C1, H6):
      - Removed `|| true` from `cargo audit`; high/critical RUSTSEC
        advisories now block PRs. RUSTSEC-2024-0436 (paste) is allow-listed
        via `--ignore` with documented rationale in SECURITY.md.
      - Swatinem/rust-cache@v2 replaces hand-rolled `actions/cache` (better
        hit rates, profile/target-aware).
      - Pinned dtolnay/rust-toolchain to `@stable` (was `@master`).
    - **New CI workflows**:
      - `msrv.yml` — Rust 1.88 MSRV compile check with drift guard
        (fails if `[workspace.package].rust-version` and the matrix disagree).
      - `deny.yml` — cargo-deny license/bans/advisories gate.
      - `release.yml` — tag-triggered multi-OS build + GitHub Release with
        auto-generated git-cliff notes.
    - **`deny.toml`** — project-wide dependency policy: MIT-compatible
      license allowlist, duplicate-version bans, non-crates.io source bans,
      RUSTSEC allow-list with rationale comments.
    - **GitHub project hygiene**:
      - `.github/CODEOWNERS` (per-crate ownership, sensitive paths protected).
      - `.github/PULL_REQUEST_TEMPLATE.md` (checklist including "no panic",
        "no Box<dyn Error>", etc.).
      - `.github/ISSUE_TEMPLATE/{bug_report,feature_request}.yml`.
    - **Pre-commit** — Rust toolchain hooks added:
      - `cargo fmt --check` on pre-commit.
      - Tiered `cargo clippy` + `cargo audit` on pre-push.
      - `check-added-large-files` (512KB cap), `check-case-conflict`.
    - **`justfile`**:
      - `quick` is now **read-only** (was running `cargo fix` and modifying
        source). New `autofix` target for the mutating variant.
      - New `deny`, `deny-advisories`, `security`, `doctest`, `ci-all` targets.
    - **`CONTRIBUTING.md`** — synced to reality: Rust 1.75 → 1.88 (MSRV),
      nextest as canonical test runner, tiered clippy denies matching CI,
      documents the "no Box<dyn Error>", "no unwrap in non-test code"
      contracts, project structure now lists all 6 crates.
    - **`docs/cliff.toml`** — conventional-commits → CHANGELOG generator
      used by `release.yml`.

  **Phase 2: Security + Error Handling**
    - **`main.rs` → `anyhow::Result`** (audit C4):
      - All startup panics replaced with structured `?` propagation +
        `anyhow::Context` (loader build, model load, draft model load,
        server bind, serve).
      - Distinct exit code 78 (EX_CONFIG) for config validation failures,
        distinguishable from transient infra failures in supervisor
        restart policies.
      - Extracted helpers: `build_engine`, `configure_speculative`,
        `load_tokenizer`. `main()` is now a linear sequence of focused
        calls.
    - **OpenAI error contract hardening** (audit H3, H4, C1):
      - New `ErrorResponse::with_code(message, error_type, code)`
        constructor — OpenAI-spec `code` slot for stable identifiers
        (e.g. `engine_unavailable`, `context_length_exceeded`).
      - All OpenAI handlers (chat, completions, embeddings) upgraded:
        engine-channel-closed failures now return `503 SERVICE_UNAVAILABLE`
        + `code = "engine_unavailable"` (was `500 INTERNAL_SERVER_ERROR`
        with no code). Distinguishes transient + retryable failures from
        real server-side bugs.
      - `embeddings.rs`: replaced `let _ = state.engine_tx.send(...)`
        (which silently dropped `SendError`) with `.map_err()`.
      - Doc-comments updated; tests locked the contract.
    - **`vllm_server::util::time`** — new module:
      - `unix_now_secs()` and `unix_now_millis()` panic-free accessors
        that saturate instead of panicking on NTP-induced clock skew
        across `UNIX_EPOCH`. Replaces 5 `.expect("Failed to get system
        time")` sites in `types.rs`, `batch/types.rs`, `batch/handler.rs`.
      - 2 unit tests (post-2024 sanity + secs/millis consistency).
    - **Typed `DraftRegistryError` variants** (audit C2, M1):
      - Added typed `IoLoad { draft_id, path, source: io::Error }` and
        `Model(DraftId, vllm_traits::ModelError)` variants.
      - `From<std::io::Error>` and `From<vllm_traits::ModelError>` for
        `?` ergonomics.
      - Legacy `LoadFailed(String)` and `LoadFailedWithSource { ...,
        source: Box<dyn Error> }` variants marked
        `#[deprecated(since = "0.1.0", note = "Use IoLoad or Model
        instead")]`. Eliminates the `Box<dyn Error>` from the new-code
        path of the public API.
      - 3 call sites (NoopLoader, BenchLoader, StubLoader) migrated
        or annotated with `#[allow(deprecated)]`.
    - **Workspace dependency unification** (audit H5):
      - 8 common deps (`candle-core`, `candle-nn`, `tracing`,
        `tracing-subscriber`, `thiserror`, `parking_lot`, `async-trait`,
        `crossbeam`) centralised in `[workspace.dependencies]`. All 6
        crate Cargo.toml files migrated to `{ workspace = true }`.
      - Bumping any of these is now a one-line workspace edit instead of
        touching 4-6 Cargo.toml files.
    - **Doctest CI phase** (audit C1):
      - `cargo test --doc --workspace --all-features` added to both
        `ci.yml` and `matrix-test` jobs. Closes the gap where broken
        doc-example code was silently shipped (nextest only runs
        `#[test]` functions).
    - **OpenAI error contract test matrix** (audit C4):
      - New `crates/server/tests/error_contract.rs` — 9 tests locking
        the v0.1 server's wire-level error behavior across all 3 OpenAI
        handlers + the `ErrorResponse` constructors. Future refactors
        cannot accidentally downgrade the contract.

  **What this enables (for follow-up phases)**
    - `gqa.rs` (1036 lines) and `compose.rs` (872 lines) are now
      mechanical splits — file-size rules (800-line soft cap) are
      straightforward to enforce in CI without behavioral risk.
    - `parking_lot::Mutex` global replacement is now a pure refactor:
      all error semantics are already correct (typed `LockPoisoned`,
      `?` propagation, no panic-prone `.expect("poisoned")` on
      production paths).
    - The `Model(DraftId, ModelError)` variant enables per-failure
      recovery policies (retry vs fallback vs circuit-break) in
      future drafts work.

---

- **Comprehensive Refactor (Phase 5)** — module-splitting pass to bring
  every Rust file under the project's 800-line soft cap. Pure file-size
  refactors (zero behavioral change), all 1235 tests pass after each
  commit. 7 atomic commits, 10 large files slimmed down.

  Per-file line counts (before → after):
    | File | Before | After | Pattern |
    |------|-------:|------:|---------|
    | `model/src/kernels/flash_attention/kernel.rs` | 869 | 126 | split into `kernel/` with `flash_attention_v2.rs`, `scaled_dot_product.rs`, `tests.rs` |
    | `core/src/speculative/adaptive.rs` | 711 | 233 | extract 22 inline tests → `adaptive/tests.rs` |
    | `model/src/components/attention/mla.rs` | 724 | 301 | extract 15 inline tests → `mla/tests.rs` |
    | `model/src/components/attention/flash_attention_v3.rs` | 694 | 329 | extract 9 inline tests → `flash_attention_v3/tests.rs` |
    | `model/src/qwen3/block.rs` | 613 | 376 | extract 9 inline tests → `block/tests.rs` |
    | `model/src/qwen3/config/model.rs` | 556 | 381 | extract 9 inline tests → `model/tests.rs` |
    | `model/src/paged_tensor/tensor_store/buffer.rs` | 799 | 317 | extract 24 inline tests → `buffer/tests.rs` |
    | `core/src/scheduler/batch_composer/compose.rs` | 872 | 373 | (prior session) split into `compose/` with `tests.rs`, `prop_tests.rs` |
    | `model/src/components/attention/gqa.rs` | 1036 | 472 | (prior session) extract 17 tests → `gqa/tests.rs` |

  Pattern: each large file now declares `mod tests;` to point at a sibling
  `tests.rs` file, keeping the implementation focused and the 800-line cap
  enforceable. The kernel.rs split is a multi-way split (impl + tests) —
  `FlashAttention` trait + `FlashAttentionKernel` facade stay in
  `kernel.rs`, the `FlashAttentionV2` and `ScaledDotProductAttention`
  impls move to their own files. Follows the existing `compose.rs` +
  `compose/` sub-module pattern.

  Additional cleanup folded into this phase:
    - **Typed `DraftRegistryError` migration completion** (closes out the
      C2 audit): the 5 remaining call sites using the deprecated
      `LoadFailed(String)` variant (`NoopLoader`, `StubLoader` in
      `core/src/speculative/draft_resolver.rs`, plus 3 test/bench
      loaders) all migrated to the typed `Model(DraftId, ModelError)`
      variant. Eliminates the last `#[allow(deprecated)]` surface in
      the speculative decoding path; the legacy string variants are
      still present but now have zero callers in the workspace.
    - **Broken doc links fixed** (`crates/model/src/quantize/gguf.rs`,
      `crates/server/src/cli/args.rs`): a stale link to
      `crate::loader::format::load_checkpoint` (the symbol was renamed
      to `Format::can_load`) and a redundant explicit link target —
      both now pass `cargo doc --no-deps -D warnings`.

  Test count: 1235 → 1235 (zero new tests, zero removed — these were
  pure refactors). All Phase 5 commits verified by `just ci`
  (fmt-check → clippy → doc-check → nextest).

---

- **Comprehensive Refactor (Phase 6)** — second module-splitting pass
  targeting the remaining large source files. Pure file-size refactors
  (zero behavioral change), all 1235 tests pass after each commit.
  10 atomic commits, 10 large files slimmed down.

  Per-file line counts (before → after):
    | File | Before | After | Pattern |
    |------|-------:|------:|---------|
    | `server/src/security/jwt.rs` | 571 | 276 | extract 10 inline tests → `jwt/tests.rs` |
    | `core/src/engine/mod.rs` | 510 | 175 | extract 19 inline tests → `engine/tests.rs` |
    | `core/src/scheduler/memory/eviction.rs` | 459 | 196 | extract 11 tests + 3 proptests → `eviction/{tests,prop_tests}.rs` |
    | `core/src/scheduler/request_queue.rs` | 456 | 220 | extract 4 tests + 4 proptests → `request_queue/{tests,prop_tests}.rs` |
    | `core/src/sampling.rs` | 452 | 228 | extract 23 tests + 4 proptests → `sampling/{tests,prop_tests}.rs` |
    | `server/src/config.rs` | 551 | 391 | extract 15 inline tests → `config/tests.rs` |
    | `model/src/components/gated_delta/rule.rs` | 572 | 423 | extract 5 inline tests → `rule/tests.rs` |
    | `model/src/components/attention/util.rs` | 532 | 262 | extract 10 inline tests → `util/tests.rs` |
    | `core/src/scheduler/policy/priority.rs` | 207 | 64 | extract 1 test + 3 proptests → `priority/{tests,prop_tests}.rs` |
    | `core/src/scheduler/engine/mod.rs` | 172 | 34 | extract 8 inline tests → `engine/tests.rs` |

  Pattern: continues the Phase 5 sibling-file convention. For files
  with both `mod tests` and `mod prop_tests` inline blocks (eviction,
  request_queue, sampling, priority), both are extracted to separate
  sibling files; this preserves the unit-vs-property test boundary
  while keeping production code under the 800-line cap. Files whose
  fields are pub-only (config.rs) split cleanly; one file with private
  fields that needed broader visibility (cli/args.rs) was intentionally
  skipped this phase to avoid coupling the split with a visibility
  refactor.

  Two small follow-ups:
    - `cli/args.rs` was attempted but rolled back: the `CliArgs` struct
      has private fields (`server`, `engine`, `auth`, `logging`,
      `config`) and the tests access them directly via `cli.server.host`,
      `cli.engine.kv_blocks`, etc. Splitting would require either
      making those fields `pub(crate)` or moving the tests to
      `cli/args/tests.rs` — both are valid but orthogonal to the
      file-size split. Deferred to a future phase that bundles the
      visibility change.
      **Resolution (post-Phase-6 commit `9e557e4`)**: tests now live
      at `cli/args/tests.rs` (a child module of `cli::args`), so
      `use super::*;` retains full access to private fields. No
      visibility refactor needed. `cli/args.rs` 554 → 237 lines; the
      25 unit tests are unchanged.
    - One cargo fmt re-indent on `eviction/prop_tests.rs`
      (single-line commit, no functional change).

  Test count: 1235 → 1235 (zero new tests, zero removed — these were
  pure refactors). All Phase 6 commits verified by `cargo nextest run
  --workspace --no-fail-fast` and `cargo fmt --all --check`.

---

## 🚀 [v18.0] — Multi-Model Speculative Decoding (2026-06-27)

### Added

- **DraftModelRegistry** (`crates/core/src/speculative/draft_registry.rs`)
    - Runtime registry for heterogeneous external draft models
    - Each draft owns a private `ModelBackend` and `BlockAllocator` (KV isolation by construction)
    - Lazy weight loading via `register` (Unloaded) → `attach_loaded` (Loaded) state machine
    - `Engine::with_drafts_boxed` constructor for pre-loading specs at engine startup
    - Loader-agnostic — does not depend on `vllm-model`; caller drives actual ModelLoader

- **MemoryBudget** (`crates/core/src/speculative/memory_budget.rs`)
    - VRAM budget enforcement for target + concurrent drafts
    - Atomic `try_reserve_draft` with structured `MemoryBudgetExceeded { requested_bytes, available_bytes, draft_id }` error
    - Runtime KV-cache growth tracking via `record_draft_kv_growth`
    - Default is `u64::MAX` (unlimited) — existing flows unchanged

- **Refcount-driven lifecycle**
    - `unload` returns `InUse(refcount)` if refcount > 0 (LIFE-02)
    - `force_unload` bypasses refcount for admin/test paths
    - `decrement_ref` auto-unloads when count reaches zero (LIFE-03)
    - Releases budget reservation on unload

- **DraftResolver** (`crates/core/src/speculative/draft_resolver.rs`)
    - Per-request draft selection with FALL-01 fallback semantics
    - `ResolvedDraft::{External, SelfSpec, None}` enum makes outcomes explicit
    - `DraftLoader` trait abstracts actual model loading (no vllm-model coupling)
    - Records metrics for every resolution (external / self_spec / none)

- **Per-request routing**
    - `Request.draft_model_id: Option<DraftId>` + `Request::with_draft_model` builder (RTE-01)
    - `Sequence.draft_model_id` propagated from Request (RTE-02)
    - Per-request resolution enables mixed drafts in one batch (RTE-03)

- **Fallback semantics**
    - FALL-01: load failure / unknown id / budget exceeded → silent fallback to self-spec
    - FALL-02: `Sequence.degraded_draft: bool` sticky flag set on runtime draft errors

- **Metrics** (5 new counters in `EnhancedMetricsCollector`)
    - `draft_resolutions_external_total`
    - `draft_resolutions_self_spec_total`
    - `draft_resolutions_none_total`
    - `draft_load_failures_total`
    - `draft_runtime_errors_total`

- **Integration tests** (`crates/core/tests/multi_draft_integration.rs`)
    - 14 tests covering full lifecycle, budget boundaries, mixed routing, all fallback paths
    - Stub backends with configurable failure injection

- **Benchmark** (`crates/core/benches/multi_draft_speculative.rs`)
    - Criterion benchmark: `no_draft` vs `self_spec` vs `external_draft` (3 configs)
    - Measures orchestration overhead (~1.7-2.1 µs per 16-step iteration)

### Changed

- `Sequence` gained `degraded_draft: bool` and `draft_model_id: Option<DraftId>` fields
- `Request` gained `draft_model_id: Option<DraftId>` field
- `BlockAllocator` gained `bytes_per_block()` and `allocated_bytes()` methods
- `DraftSpec` gained `weight_size_estimate_bytes: u64` field for MEM-02 budget estimation

### Requirements Satisfied

- MMLT-01, MMLT-02, MMLT-03 (multi-model loading)
- LIFE-01, LIFE-02, LIFE-03 (lifecycle management)
- MEM-01, MEM-02, MEM-03 (memory budget)
- RTE-01, RTE-02, RTE-03 (request routing)
- FALL-01, FALL-02 (fallback semantics)

**14/14 requirements passed.** Test count: 209 → 277 (+68).

### Refactored

#### Architecture Refactoring

- **Scheduler Module Split**
    - Split monolithic `scheduler.rs` into focused submodules
    - Created `scheduler/queue.rs` with `RequestQueue` for queue management
    - Created `scheduler/preemption.rs` with `PreemptionManager` for preemption decisions
    - Created `scheduler/eviction.rs` with `EvictionPolicy` for block eviction
    - Fully integrated all modules into `Scheduler` struct

- **KV Cache Layer Separation**
    - Split `core/kv_cache.rs` into `kv_cache/block_allocator.rs` and `kv_cache/prefix_cache.rs`
    - Created `model/paged_tensor/` module (separating logical and physical KV cache)
    - `tensor_store.rs` for GPU KV tensor management
    - `quantization.rs` for INT8/FP8 quantization
    - Added deprecated alias in `kv_cache.rs` for backward compatibility

- **Kernel Layer Extraction**
    - Created `model/kernels/` directory for GPU kernels
    - Moved `flash_attention.rs` → `kernels/flash_attention.rs`
    - Moved `fused_kernel.rs` → `kernels/fused_mlp.rs`
    - Moved `cuda_graph.rs` from core to `model/kernels/cuda_graph.rs`
    - Updated `components/` to use kernels module

#### Architecture Refactor Phase 5 (Qwen3.5 Hybrid 收敛, 2026-06-15)

- Split `qwen3_5/hybrid.rs` (1176 lines) into `block/` + `model.rs` + `weights.rs` + `config.rs`
- Introduce `HybridLm<B, Norm>` shell paralleling `CausalLm<B, N, H>`
- Move `GatedDeltaState` from `qwen3_5::gated_delta` to `components::gated_delta`
- Remove `causal_lm → qwen3_5` reverse dependency (`rg 'use qwen3_5' crates/model/src/causal_lm/` → 0 matches)
- GDN dims now read from `Qwen3Config` (no more hardcoded `(16, 4, 2)`)
- `Qwen35Architecture::capabilities()` upgraded to `PRODUCTION_SPECULATIVE`
- Speculative parity tests in `model_tests.rs` (124 lines) + `speculative_tests.rs` (285 lines)

Refs: `decc8c8`, `73dab5e`, `52f77ce`

#### Adaptive Speculative Decoding Counter Wire-up (Wave 2, 2026-06-26)

- `AdaptiveSpeculativeDecoder::record_verification` now returns `bool` adjustment event
- Engine `step_speculative_inner` calls `MetricsCollector::record_speculative_adjustment()` on actual adjustment
- `speculative_adjustments_total` Prometheus counter now correctly tracks adaptive decoder activity
- 3 new unit tests locking the bool return contract (high acceptance, low acceptance, deadband)
- Documentation: `SPEC-ADAPT-01` / `SPEC-ADAPT-02` marked complete in `.planning/PROJECT.md`

Refs: `docs/superpowers/specs/2026-06-26-wave2-adapt-spec.md`

#### Speculative Warmup Test Coverage (Wave 4, 2026-06-26)

- `Engine::warmup_draft_kv` visibility relaxed from `fn` to `pub(crate) fn` for test access
- New `CounterModel` wrapper in `engine::speculative::tests` mod (counts forward/forward_logits calls via AtomicUsize)
- New fast unit test `test_warmup_draft_kv_invokes_draft_per_sequence` verifies draft model receives exactly N forward() calls for N-seq Prefill batch
- Documentation: `SPEC-WARM-01` marked complete in `.planning/PROJECT.md`

Refs: `docs/superpowers/specs/2026-06-26-wave4-warmup-test.md`

#### Benchmark Suite Closure (Wave 5, 2026-06-26)

- New `crates/core/benches/latency_percentiles.rs` — per-request latency distribution with criterion auto-reported p50/p95/p99 (SPEC-BENCH-01)
- New `crates/core/benches/speculative_vs_baseline.rs` — explicit baseline vs adaptive speculative throughput comparison (SPEC-BENCH-02)
- New `docs/benchmark-suite.md` — suite documentation covering all 9 benchmarks
- New `just bench` recipe — runs all benchmarks with `--output-format bencher`
- Documentation: `SPEC-BENCH-01` / `SPEC-BENCH-02` marked complete in `.planning/PROJECT.md`; v17.0 milestone closes 7/9 SPECs

Refs: `docs/superpowers/specs/2026-06-26-wave5-benchmark-suite.md`

### Added (Phase 4)

#### Phase 4: Performance Optimization

- **Quantization Support**
    - FP16 support
    - INT8 Weight-Only quantization (`QuantizedLinear`, `quantize_2d`)
    - INT8 KV Cache with per-layer scaling
    - QuantizationCalibrator for calibration
- **Compute Optimization**
    - Flash Attention framework with software fallback (`FlashAttention`, `ScaledDotProductAttention`)
    - Sliding window attention support
    - CUDA Graph framework (`CudaGraph`, `CudaGraphExecutor`)
- **Scheduling Optimization**
    - PD Separation (Prefill/Decode separation)
    - Chunked Prefill with configurable chunk size
    - Dynamic Batch Size based on available KV blocks
    - Priority-based scheduling (`Priority`, `enable_priority_scheduling`)
- **Distributed**
    - Multi-GPU Tensor Parallelism (`DeviceMesh`, `ColumnParallelLinear`, `RowParallelLinear`, `AllReduce`)

#### Phase 5: Production Readiness (2025-04-12)

**Observability & Metrics**

- Prometheus-compatible metrics export (`/metrics` endpoint)
- Enhanced metrics collection (CUDA Graph, Sequence Packing, Adaptive Speculative)
- Health check endpoints (`/health`, `/ready`)
- Real-time metrics with `EnhancedMetricsCollector`

**Fault Tolerance**

- Circuit breaker pattern for automatic failure recovery
- Retry strategy with exponential backoff
- Degrade strategy for graceful degradation
- Recovery manager with error severity classification

**Testing**

- 26 E2E integration tests (lifecycle, concurrent, error recovery, graceful shutdown)
- Deterministic mock models for reproducible tests
- Performance regression testing in CI

**Deployment**

- Multi-stage Docker build (`Dockerfile`)
- Docker Compose with Prometheus (`docker-compose.yml`)
- Kubernetes manifests (namespace, deployment, service, HPA)
- CI performance regression workflow (`.github/workflows/benchmark.yml`)

**Core Features**

- Request timeout support (`timeout` parameter)
- Graceful shutdown (SIGINT/SIGTERM handling)
- YAML configuration file support
- Environment variable overrides (`VLLM_HOST`, `VLLM_PORT`, etc.)
- Structured JSON logging with file rotation
- Grafana dashboard (`docs/grafana/dashboard.json`)
- Config validation on startup
- Error retry support (`retries` parameter)

#### Core Features

- Real-time metrics collection with `/v1/stats` and `/metrics` endpoints
- Quantization utilities (`crates/model/src/quantize.rs`)
- Tiled Attention for memory optimization
- INT8 quantization support in KV Cache
- Forward pass with tiled attention strategy
- Comprehensive test suite for tiled attention

### Changed

- Improved documentation structure (README.md, docs/README.md, ROADMAP.md)
- Added detailed development roadmap

### Fixed

- Clippy warnings and code quality improvements
- Test compatibility with new AttentionConfig

## [0.1.0] - 2026-03-31

### Added

- **Continuous Batching** - Dynamic batch scheduling with decode-priority
- **Paged KV Cache** - Memory-efficient cache management with LRU eviction
- **Prefix Caching** - Exact match and prefix hit support
- **Speculative Decoding** - Draft-target verification architecture
- **Qwen3 Model Integration** - Support for Qwen2.5-0.5B model with real weights
- **OpenAI-compatible API** - `/v1/completions`, `/v1/chat/completions`
- **Streaming (SSE)** - Real-time token streaming
- **Sampling** - Temperature, Top-P support
- **Chunked Prefill** - Process long prompts in chunks

### Architecture

- **3-Crate Structure**:
    - `vllm-core`: Scheduler, Engine, KV Cache, Types
    - `vllm-model`: Qwen3, Attention, MLP
    - `vllm-server`: HTTP API (axum)

### Dependencies

- Rust (edition 2021)
- Candle (ML backend)
- Axum (HTTP)
- Tokio (async runtime)
- SafeTensors (weight loading)

---

## Migration Guides

### Upgrading to 0.1.0

No migration needed - initial release.

---

## 🚀 [v22.0] — Production Hardening (2026-06-27)

### Added

- **Security middleware wired (SEC-01..06)** — JWT signature verification (HMAC-SHA256 + RSA/ECDSA), `RbacMiddleware` permission enforcement, `RequestBodyLimitLayer` (configurable max body), audit log integration test, Grafana credentials moved to `.env`, structured TLS error replacing `unwrap()` panic
- **Phase 19 e2e tests un-ignored (OPS-02)** — `Engine::step()` speculative-mode hang fixed (DashMap shard re-entry deadlock in `EnhancedMetricsCollector`); 9 tests across `engine_wiring.rs` and `engine/spec_dispatch/tests.rs` now pass
- **Production polish (OPS-01, RFU-05, PERF-01..03)** — `parking_lot::Mutex` migration (24 sites in scheduler/engine), `MlaKvCache::write_compressed` incremental `slice_assign`, `eq_ignore_ascii_case` in arch detection, `std::sync::LazyLock` migration
- **Engine refactor (ARF-06, ARF-07)** — `engine.rs` God module split into focused sub-modules; `engine/spec_dispatch` tree unified post-Phase-31

### Fixed

- 5 cargo doc broken-link warnings resolved (intra-doc-link fixes in `engine.rs`, `components/attention/mod.rs`, `block.rs`, `decoder_block/mod.rs`, `speculative/registry/`)
- `Engine::step()` speculative-mode determinism bug (was hanging in 9 tests)

### Tech Debt Rolled Forward

- Stub architectures (`gemma3`, `llama4`, `phi4`, `mistral_small`) — policy deferred to v23.0
- `TensorParallelError` manual impl — deferred to v23.0
- `Box<dyn Error>` in `dist/src/grpc.rs` — deferred to v23.0
- Stale `CLAUDE.md`, `README.md`, `CHANGELOG.md`, `MIGRATING.md` — deferred to v23.0
- Dead code (~2000 LOC across scheduler/, routing/, ha/, circuit_breaker/) — deferred to v23.0
- `core → model` upward dep via cuda-graph feature — deferred to v23.0

### Stats

- **Phases:** 4 (Phase 36-39)
- **Test count:** 1179+ (≥ 1146 v21.0 baseline; +33 net)
- **Coverage:** doc 97.8%, clippy/fmt clean, 0 cargo doc warnings
- **ADRs:** 15 (no new ADRs in v22.0)

---

## 🚀 [v21.0] — P2/P3 Backlog Cleanup (2026-06-27)

### Added

- **API/error boundary work** (API-04, API-06, API-08, API-10) — Mutex `.expect()` migrations, `From<E>` impls for cross-crate error conversion, `dyn Trait` compile-only tests for object-safe traits (8 tests in `crates/testing/tests/dyn_safety.rs`)
- **Naming audit compliance** (NAME-F-04) — `qwen3_config` module moved to `qwen3::config`; test files migrated from `src/` to `crates/*/tests/`

### Removed

- **Dead `mod.rs`** in `crates/traits/tests/` (P3-01)

### Tech Debt Rolled Forward

- `mut Prompt::token_ids` non-mutating methods (P2-09) — partial; deferred to v22+
- Multi-node/vllm-dist resurrection — feature-gated only; OPS-05 still deferred
- Real-model benchmark — OPS-04 deferred (no GPU env)

### Stats

- **Phases:** 5 (Phase 31-35)
- **Test count:** 1146+ (1144 baseline + 13 new − 11 dedup)
- **Coverage:** doc 97.8%

---

## 🚀 [v20.0] — Codebase Remediation (2026-06-27)

### Added

- **48 requirements addressed across 6 phases** (Phase 25-30) — code remediation of v19.0 audit findings (architecture, naming, comments/docs, API/errors, tests, benchmarks)
- **12 new ADRs** — component sharing, feature flags, self-speculation, FP8, KV cache split, speculative decoding, per-request draft routing, multi-node feature-gating, FP8 quantizer orphan decision, CUDA graph feature-gating, cross-crate error boundaries, continuous batching

### Stats

- **Phases:** 6 (Phase 25-30)
- **Test count:** 1144+ passed, 0 failed
- **Coverage:** doc 97.8% / 99.6%

---

## 🚀 [v19.0] — Codebase Health Audit (2026-06-27)

### Added

- **5 analysis-only phases** (Phase 20-24) producing `.planning/audit/` directory:
    - Architecture audit (crate deps, module boundaries, layering matrix)
    - Naming audit (NAME-* findings)
    - Comments/docs audit (placeholder doc survey)
    - API/error audit (error type hygiene)
    - Test/benchmark audit
- **No code changes** — analysis-only milestone; findings drive v20.0-v23.0 remediation

### Stats

- **Phases:** 5 (Phase 20-24)
- **Output:** 5 audit reports in `.planning/audit/{architecture,naming,docs,api,benchmark}/`
- **Findings:** 22+ categories, drove 4 remediation milestones (v20.0, v21.0, v22.0, v23.0)

---

## Known Issues

- Long context (>32K) not yet supported (v24+ candidate)
- Multimodal/Vision not yet supported (v24+ candidate)
- Tool calling not yet supported (v24+ candidate)
- Multi-node / vllm-dist resurrection deferred (feature-gated only)

---

## Credits

Thanks to all contributors and the vLLM project for inspiration.
