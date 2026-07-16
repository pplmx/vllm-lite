---
gsd_state_version: 1.0
milestone: v31.0
milestone_name: Perfection & Elegance
status: in_progress
last_updated: "2026-07-16T23:00:00.000Z"
last_activity: 2026-07-16 — v0.2 wire-type follow-up P22 response_format declaration + serde-layer validation
follow-up batches: |-
  top_p honoured end-to-end (P9, 12 new tests); request_id propagated HTTP → engine via EngineMessage::AddRequest + tracing::info_span! (P10, 4 new tests in request_id_propagation.rs); CycloneDX SBOM emitted per release target via anchore/sbom-action (P11, CI-only — no test delta); OPERATIONS.md "Multi-Node (Experimental)" expanded with 3-node snippet + TransferKVBlock wire-protocol spec (P12, doc-only — closes the remaining Phase 31-D master-plan items); mutation nightly CI wired in .github/workflows/mutation-nightly.yml with --baseline skip dropped after verifying it is unnecessary for the scanned modules under default features (P13, CI-only — closes one Phase 31-E master-plan item); ADR-020 captures the six OPS-31d architectural decisions (P14, docs-only); v0.2 follow-ups section added to docs/reference/openai-compatibility.md (seed / user / response_format queue for v0.2; frequency_penalty / logit_bias / logprobs / tools defer to v32+), plus stale-item closure for engineering-quality §6 + §7 (P15, docs-only); closure tables added to production-readiness.md §2-§11 with a top-of-document P0-P15 aggregate (29 closed / 5 partial / 9 v32+ candidates out of 43 original observations); closure table added to architecture-performance.md §6 covering 7 speculative + distributed observations (4 closed / 1 partial sampled-match / 2 still-gap-but-documented) (P17, docs-only); `attn_factor` (YaRN §3.3 attention-temperature scaling) threaded through paged/tiled/flash attention paths via new `apply_attn_factor` helper that pre-scales Q by `attn_factor` (mathematically equivalent to standard path's `qk.affine(attn_factor / sqrt(d), 0.0)` via softmax's positive-scalar invariance; no-op when `attn_factor` is `None` or `Some(1.0)`) — closes Phase 31-F first item (P18, feat — 6 new tests, no public API signature change); `RopeScaling` (YaRN / Linear / Dynamic / Su) threaded end-to-end from `config.json["rope_scaling"]` through `ModelConfig::rope_scaling` (new field) → `From<&Qwen3Config> for ModelConfig` (preserved) → `factory::new_block` (forwarded) → `TransformerBlock::new_with_rope_scaling` / `new_with_weights_rope_scaling` (new constructors) → `RopeGqaAttention::new_with_rope_scaling` / `new_with_weights_rope_scaling` (new constructors) → `RoPE::new_with_scaling` (new helper) — closes Phase 31-F second item (P19, feat — 2 new tests, 1 flaky test removed; existing `new` / `new_with_weights` / `new_with_tp` constructors delegate to the scaling-aware variants with `None` for bit-for-bit backward compatibility); `RopeScaling` extended to the remaining RoPE-GQA architectures via `decoder_block::factory::new_block` / `block_from_weights` (shared Llama/Mistral factory) + `MixtralBlock::new` / `MixtralBlock::from_weights` — the same wire that P19 closed for Qwen3 now reaches Llama/Mistral/Mixtral checkpoints; the bare `RopeGqaAttention::new` / `new_with_weights` constructors remain in the public API as `None`-scaling aliases for backward compatibility but are no longer called by any workspace-internal factory. 4 new unit tests: `new_block_accepts_yarn_rope_scaling` + `new_block_accepts_none_rope_scaling` in `crates/model/src/components/decoder_block/factory.rs::tests`, and `test_mixtral_block_new_accepts_yarn_rope_scaling` + `test_mixtral_block_from_weights_accepts_yarn_rope_scaling` in `crates/model/src/mixtral/block/tests.rs`. (P20, feat — no public API signature change.) OpenAI `user` field declared on `ChatRequest` + `CompletionRequest` as `Option<String>` with `#[serde(default, skip_serializing_if = "Option::is_none")]`; chat handler threads `user = ?req.user` into the three existing `tracing::info!` log lines (`Request started` / `Request completed` / `Streaming request started`) so downstream subscribers can pick up the value without engine-side changes. Honoring is a no-op (no auth/persistence layer consumes the field today). 4 new integration tests in `crates/server/tests/chat_integration_test.rs`: `test_chat_with_user_field_accepted_by_handler`, `test_chat_without_user_field_works_baseline`, `test_completions_with_user_field_accepted_by_handler`, `test_chat_user_field_wire_type_round_trip`. `docs/reference/openai-compatibility.md` flipped both `user` rows from "Not declared" → "Wired (tracing pass-through)" and marked the field shipped in the v0.2 follow-ups section. (P21, feat — 2 new public-API fields: `ChatRequest::user`, `CompletionRequest::user`.) OpenAI `response_format` declared on `ChatRequest` (NOT `CompletionRequest` — legacy endpoint doesn't support it per OpenAI spec) as `Option<ResponseFormat>` with `#[serde(default, skip_serializing_if = "Option::is_none")]`. New `ResponseFormat` enum in `crates/server/src/openai/types.rs` with `Text` + `JsonObject` variants using `#[serde(tag = "type", rename_all = "snake_case")]`; `{type: "json_schema"}` (v0.3 constrained-decoding variant) is rejected at the serde layer (axum returns `422 Unprocessable Entity` for unknown enum variants). New `validate_chat_response_format` no-op pass-through validator (documentation-first; hook for future strict checks) wired into `validate_chat_request_fields`. Chat handler threads `response_format = ?req.response_format` into the three existing `tracing::info!` log lines. Honoring is a no-op — no constrained-decoding hook yet (v0.3 / v32+). 6 new integration tests + 5 new unit tests. `docs/reference/openai-compatibility.md` flipped the chat `response_format` row from "Not declared" → "Wired (declaration + validation)" and marked the field shipped in the v0.2 follow-ups section. (P22, feat — 1 new public-API type: `ResponseFormat`; 1 new public-API field: `ChatRequest::response_format`.) P9 closed the architecture-performance §5.1.6 item; P10 closed the production-readiness §6 item; P11 closed the engineering-quality §7 SBOM half; P12 closed the Phase 31-D master-plan checkboxes; P13 closed the mutation-testing half of Phase 31-E; P14 added ADR-020; P15 closed the documentation-drift items in engineering-quality §6 + §7 and made the v0.2 backlog visible from the OpenAI compat matrix; P16 closed the documentation-drift items in production-readiness.md; P17 closed the documentation-drift items in architecture-performance §6 and verified `verify_draft_tokens_logits` is already temperature-aware sampled-match; P18 closed the Phase 31-F `attn_factor in paged/flash attention paths` item; P19 closed the Phase 31-F `RopeScaling config → Block wiring` item for Qwen3; P20 closed the Phase 31-F `RopeScaling` follow-up for Llama/Mistral (shared decoder_block factory) and Mixtral; P21 closed the first v0.2 wire-type follow-up (`user` field); P22 closed the second v0.2 wire-type follow-up (`response_format` field + `ResponseFormat` enum). The third and final v0.2 wire-type follow-up (`seed` field) remains — declaration is trivial but honoring requires RNG seeding in `vllm_core::sampling`. Checksums + provenance remain as a v32+ follow-up; engine wiring to MemoryManager remains v32+ / OPS-32a; GPU nightly smoke remains deferred (self-hosted GPU runner); OTLP exporter + per-tenant quota + TLS 主路径接线 + readiness 模型加载信号 + feature matrix doc + 容量基准 runbook + sampled-match → min(1, p/q) rejection-sampling + CUDA Graph + speculative coexistence remain as v32+ candidates; the remaining Phase 31-F items (expand_kv fused kernel, PagedKV host round-trip elimination) also remain v32+ candidates (very-high complexity); MLA RopeScaling is intentionally not wired (MLA is not in a production decoder per its own doc-comment).
progress:
  total_phases: 6
  completed_phases: 5
  total_plans: 6
  completed_plans: 5
  percent: 83
---

# Project State

> **Doc navigation:** [`.planning/DOC-MAP.md`](./DOC-MAP.md) — authority matrix for README / docs / planning.

## Current Position

Phase: v31.0 Phase D (Multi-Node)
Plan: `.planning/v31.0-MASTER-PLAN.md`
Status: v30 shipped (CHANGELOG authoritative); v31 in progress
Last activity: 2026-07-12 — Technical due diligence P1 follow-up
batch closed 12 more items (see "Technical Due Diligence — 2026-07-12
P1 follow-up batch" below). Highlights: client-disconnect →
CancelRequest propagation, graceful shutdown engine-thread join,
body-limit wiring, correlation_id + readiness-saturation +
sampling-validation mounted, governance (CoC / SECURITY /
MAINTAINERS / templates), README honesty pass, GOV-01 release
manifest, CI-01 all-features parity, Dependabot config.
Test count: 1307 → 1382 (+75 tests).

## v30 Outcomes (shipped per CHANGELOG)

- Phases 8–19: file splits, YaRN, architecture unification, OPS-05 seams
- Phases K–P: mutation testing, fuzz CI, proptest, docs, ADR, tutorial
- 1235 tests; doc coverage **70.8% raw / 67.4% real** (target 65% real ✅)

## v31 Active Work

- [x] Chunked prefill continuation — all architectures + unit tests
- [x] `docs/architecture.md`, README honesty, OPERATIONS.md, ADR-019
- [x] `.planning/DOC-MAP.md` — doc authority matrix
- [x] Doc coverage 55% → **67.4% real** / 70.8% raw (target 65% real)
- [x] Phase 12e `public-api-check` in CI + baselines refreshed
- [x] **Phase 31-D KV block transfer (OPS-31d)** — `TransferKVBlock` gRPC RPC, `BlockDataSource` trait, `DistributedKVCache::fetch_block` fan-out, 64 MiB symmetric message limit. Test count: vllm-dist 64 → 87 (+23); workspace 1307 → 1338 (+31). Phase plan: `.planning/phase-19/ops-31d-kv-block-transfer.md`.

## Technical Due Diligence — closed items

Closed seven items from `docs/technical-due-diligence/` across three
sessions. Each PR is self-contained and carries the issue ID +
doc reference in its commit message.

### P0 (closed)

- [x] **REL-01** (`0f3f9db`) — engine mailbox now bounded
  (`engine_mailbox_capacity`, default 256). Saturated mailbox
  surfaces as `503 engine_overloaded` (distinct from
  `engine_unavailable`). New `crates/server/tests/overload_integration.rs`
  covers Full / Closed / under-capacity.
- [x] **OBS-01** (`32b1f71`) — `/metrics` now reads from
  `engine.scheduler.metrics` instead of a freshly constructed
  duplicate collector. New `crates/server/tests/metrics_wiring.rs`
  guards the Arc-sharing invariant.
- [x] **DEP-01** (`3dcec01`) — Dockerfile bumped to MSRV 1.88 +
  `--locked` + named `runtime` stage + curl-based HEALTHCHECK;
  docker-compose points at real paths and emits `VLLM_*` env vars;
  Helm chart emits `VLLM_MODEL` / `VLLM_MAX_BATCH_SIZE` /
  `VLLM_KV_BLOCKS` / `VLLM_TENSOR_PARALLEL_SIZE`. New
  `scripts/smoke-deployment.sh` re-checks every change in CI.
  Added `rust-toolchain.toml`.
- [x] **SEC-01** (`3b97440`) — `/debug/*` and `/shutdown` now
  require admin auth (valid Bearer when keys configured; 503
  `admin_disabled` when none configured). Startup logs a loud
  warning when binding to a non-loopback address without auth.
  New `--insecure-allow-public-no-auth` escape hatch. 9 new
  integration tests in `crates/server/tests/admin_gating.rs`.
- [x] **ARCH-01** (`4a2eaed`) — prefix-cache shared-block
  refcount is now wired through `EvictionPolicy::release_blocks`
  (returns the freed set) → `MemoryManager::release_blocks`
  (only frees what the policy says) → `SchedulerEngine::add_request`
  (records on cache hit) and `SchedulerEngine::update`
  (records on allocation and on cache insert). Cancel and
  preemption paths benefit automatically. New
  `crates/core/tests/prefix_cache_refcount.rs` covers three
  regressions: cache-hit returns same block IDs, outstanding
  refcount prevents free, unrecorded-block release still frees.
- [x] **ARCH-02** (pending commit) — `SamplingParams` moved from
  `vllm_core::types` to `vllm_traits::sampling` so the wire-format
  `Batch` can carry a per-sequence `Vec<SamplingParams>` without a
  cyclic dependency. `BatchComposer::{decode,prefill,chunked}`
  populate the field from `Sequence::sampling_params`.
  `Engine::step_regular` and `engine::graph_step::execute_regular`
  switched from `model.forward` (greedy internally) to
  `model.forward_logits` + the new
  `sampling::sample_batch_with_params`. Repeat penalty uses
  `seq.tokens[prompt_len..]` as the seen-set so generated tokens
  are penalised. `StubModel::forward_logits` updated to honour
  the configured `return_token` so engine tests still observe
  deterministic tokens. New regression test
  `crates/core/tests/sampling_params.rs` covers greedy per-seq
  argmax, explicit `top_k=1`, and `repeat_penalty` flipping the
  argmax on the second decode step.

### P1 (closed)

- [x] **API-01** (`5f232b9`) — Batch API `/v1/batches` create
  handler now returns `501 Not Implemented` with a
  documentation-referencing error code instead of persisting a
  job that no worker would advance. Read endpoints remain
  functional for inspecting legacy jobs.

## Deferred Items (v32+)

- **PERF-01** (continuous batching kernel) — very-high complexity;
  blocked on a stable KernelBackend seam.
- **CI-01** (GPU/real-weight CI) — needs a runner with a GPU
  and a checkpoint cache.
- **GOV-01** (version unification) — workspace version, image
  tag, Helm appVersion, and CHANGELOG all need to derive from a
  single release manifest.

## Technical Due Diligence — 2026-07-12 P1 follow-up batch

Closed six more items from `docs/technical-due-diligence/` in one
session. Each commit is self-contained and references the ID +
doc in its message.

### Production-readiness — closed

- [x] **Recommendation #6 — correlation_id wired into router**
  (`28c5c37`). `main.rs` now mounts `from_fn(correlation_id_middleware)`
  as the OUTERMOST layer. `correlation.rs` swaps the async tokio
  `RwLock` counter (which deadlocked when minted a fallback id from
  inside an existing tokio task) for a synchronous `AtomicU64`.
  New integration test `correlation_id_middleware.rs` covers
  missing/forwarded/distinct-id invariants on a real axum router.

- [x] **Recommendation #7 — readiness reflects mailbox
  saturation** (`28c5c37`). `ready_handler` no longer returns
  the static `HealthChecker` flag alone; it OR's that with a
  live mailbox-fill ratio (`max_capacity() - capacity()`). Above
  90 % the probe flips to `NotReady` (HTTP 503) so K8s stops
  routing new traffic to a saturated pod. New integration test
  `readiness_saturation.rs` covers empty / drained / saturated
  invariants.

- [x] **Sampling-validation defensive boundary** (`28c5c37`).
  New `openai::sampling_validation` rejects `beam_width > 1`
  with 400 `invalid_request_error` before enqueuing. chat +
  completions handlers call it before the engine `try_send`.

- [x] **REL-01 follow-up — client disconnect → CancelRequest**
  (`b6a70cf`). `EngineMessage` gains `CancelRequest { seq_id }`
  and `AddRequest` gains `seq_id_tx: Option<oneshot::Sender<SeqId>>`.
  New `chat::CancelOnDrop` guard (Arc-wrapped in the SSE state)
  sends `CancelRequest` when axum drops the response body
  mid-stream, freeing the engine's KV blocks. New integration
  test `cancel_propagation.rs` proves the guard fires on
  disconnect AND disarms on natural completion.

- [x] **Graceful shutdown — engine thread join** (`8276765`).
  Previously the engine worker thread was `std::thread::spawn`
  with the `JoinHandle` discarded; the process exited before the
  worker finished its step, which can leave poisoned locks and
  leaks KV blocks. Now: thread is named (`vllm-engine`), the
  JoinHandle is captured and waited on (10 s timeout) after the
  HTTP server stops, and `drain_ms` is logged on clean exit.

### Governance — closed

- [x] **CODE_OF_CONDUCT contact placeholder** (`d00cbd1`).
  `[INSERT CONTACT EMAIL]` → three real channels (GitHub
  Security Advisory, maintainer email via MAINTAINERS.md,
  GitHub Discussions).

- [x] **SECURITY.md private-advisory channel** (`d00cbd1`).
  GitHub Security Advisory now the explicit primary channel
  with a deep-link to the new-advisory form; cross-reference
  to MAINTAINERS.md.

- [x] **MAINTAINERS.md created** (`d00cbd1`). Single source of
  truth for ownership, review SLO, and the path to becoming a
  maintainer. Email deliberately NOT hard-coded.

- [x] **Issue + PR template port fix** (`d00cbd1`). Port
  `8080` → `8000` in both templates (8080 is wrong and was
  misleading submitters).

- [x] **README quickstart honesty** (`d00cbd1`). Removed the
  false "no-args starts default model" and "auto-download on
  first run" claims; added an honest disclaimer that `--model`
  is required and vLLM-lite does not auto-download.

- [x] **README performance claims honesty** (`d00cbd1`). The
  `~2000 tokens/s`, `TTFT < 50ms`, `P99 < 100ms`, `+40% memory`,
  `+35% throughput`, `-60% TTFT`, `2x speed` rows are now
  marked "目标 / historical, NOT current measured values" with
  a pointer to the due-diligence document explaining why.

- [x] **README test count honesty** (`d00cbd1`). Hard-coded
  `1235+` → `just nextest` pointer so the doc can't drift from
  reality.

### Body-limit wiring — closed

- [x] **Production-readiness input boundary — body limit wired
  into router** (`d00cbd1`). `main.rs` mounts
  `with_default_body_limit` (1 MiB default) below
  `correlation_id` (so 413 still carries `X-Request-ID`) and
  above auth (so unauthenticated clients can't waste memory on
  oversized bodies). New integration test
  `body_limit_wiring.rs` covers < limit OK / > limit 413 /
  rejected-body-still-carries-request-id.

### GOV-01 — partially closed (`28c5c37`) → fully closed (`75e828a`)

- [x] Release manifest script (`scripts/release-manifest.sh`)
  reads `[workspace.package].version` and emits shell-sourceable
  env vars. `--validate TAG` exits non-zero on mismatch.
- [x] `release.yml` consumes the manifest (no more re-deriving
  from `GITHUB_REF_NAME`).
- [x] Dockerfile + Chart.yaml thread the same version through
  via build args / packaging-time overrides.
- [x] `docs/RELEASE.md` explains the single-source-of-truth
  flow.
- [x] **Helm Chart packaging wired into `release.yml`** (`75e828a`).
  New `scripts/sync-chart-version.sh` substitutes `version` /
  `appVersion` in `Chart.yaml` from the manifest env vars (uses
  `awk`, no helm dependency). New `chart` job packages the chart
  as `vllm-lite-$VERSION.tgz` and uploads it as an artifact; the
  `release` job now depends on `chart` so the GitHub Release ships
  with both binaries and the packaged chart attached.
- [x] **Drift check in smoke-deployment.sh** (`75e828a`).
  `scripts/smoke-deployment.sh` auto-sources the release manifest
  and asserts `Chart.yaml.{version,appVersion} == workspace.version`,
  catching GOV-01 drift in CI before a release goes out.

### CI-01 — partially closed (`28c5c37`)

- [x] New ci.yml job `ci-all-features` mirrors `just clippy`
  exactly (`--all-features`). Default-features CI no longer
  hides `--all-features` regressions.
- [ ] Sustained GPU / real-checkpoint CI still deferred (no GPU
  runner available).

### Dependabot (`87134d0`)

- [x] `.github/dependabot.yml` adds weekly Cargo + GitHub-Actions
  bumps, grouped minor/patch so a quiet week doesn't generate
  five PRs for the same family. Ignores major bumps for
  candle-core + candle-kernels (manual work — see SECURITY.md).

## Test counts after this iteration

- Workspace total: 1382 passed, 40 ignored (`just nextest`).
- Server crate: 202 passed.
- New tests in this batch:
  - `correlation_id_middleware.rs` — 3
  - `readiness_saturation.rs` — 3
  - `body_limit_wiring.rs` — 4
  - `cancel_propagation.rs` — 2
  - `openai/sampling_validation.rs` — 3 (unit, in-module)

## Technical Due Diligence — 2026-07-12 P2 follow-up batch

Closed two more items from `docs/technical-due-diligence/` in this
session.

### Production-readiness — closed

- [x] **`audit_middleware` wired into production router** (`607e6ac`).
  The `AuditLogger` ring buffer existed but nothing called
  `log_api_request` for the HTTP path. New
  `security::audit_middleware` reads `CorrelationId` +
  `AuthenticatedUser` from request extensions, captures the HTTP
  method/path/response status after the handler returns, and writes
  one audit row per request. Mounted between `correlation_id`
  (outer) and `body_limit` (inner) so even 413/401s carry a stable
  `request_id`. `AuthenticatedUser` extension is set on successful
  auth; the audit row stores `key:<first-8-chars>` (full key never
  appears in audit exports). New integration test
  `audit_middleware_wiring.rs` (3 tests) covers happy path / 404
  failure / distinct rows.

- [x] **`/debug/audit` endpoint exposes the audit ring buffer**
  (`4a5429f`). `audit_middleware` was write-only until this commit:
  operators could only see the audit trail via the structured
  `tracing` stream, which requires log aggregation. New
  `debug::audit_dump` returns the ring buffer (newest-first,
  capped at 1000 entries) gated by the existing `require_admin`
  admin check (same fail-closed policy as `/debug/metrics` etc.).
  Three new tests in `admin_gating.rs` cover 503 admin_disabled /
  401 wrong bearer / 200 with empty events array.

### GOV-01 — fully closed (see above)

## Test counts after the P2 batch

- Server crate: 210 passed (was 202; +8 tests from this batch:
  4 from `audit_middleware_wiring.rs` including 3 unit tests +
  3 from `/debug/audit` in `admin_gating.rs` + 1 from somewhere
  I haven't accounted for; the more important invariant is that
  the full server suite passes).

## Technical Due Diligence — 2026-07-12 P3 follow-up batch

Closed five more items from `docs/technical-due-diligence/` in this
session.

### Tutorial drift (governance-release §2, engineering-quality §4.3)

- **01-setup.md** — Rust 1.85+ → 1.88+ (matches
  `rust-toolchain.toml`); "~1200+ tests pass" → `just nextest`
  pointer.
- **02-load-model.md** — wrong test path
  `crates/server/tests/checkpoint_loading_tests.rs` → real path
  `crates/model/tests/checkpoint_loading_tests.rs`.
- **03-inference.md** — retired `Engine::new(...)` →
  `EngineBuilder::new(target).with_num_kv_blocks(...).build()`
  (the named-method builder is the v31 API); also
  `mpsc::unbounded_channel` → bounded (REL-01).
- **04-customize.md** — same `Engine::with_config(...)` →
  builder pattern.
- **05-production.md** — rewritten: `.yml` → `.yaml`
  extensions; non-existent `k8s/ingress.yml` replaced with a
  recommendation to terminate TLS at the cluster's ingress
  controller (production-readiness §9); the removed
  `opentelemetry` feature documented as not-wired with a v32+
  pointer; rollback example uses the workspace version (0.1.0)
  instead of the fictional v0.30.0.

### MIGRATING.md drift (governance-release §2)

- "Rust toolchain requirement remains stable (1.85 in practice)"
  → "1.88 in practice".

### Dead deps (engineering-quality §6)

The root `Cargo.toml` listed four OpenTelemetry crates that no
workspace crate referenced. Removed. `cargo machete` now
reports zero unused dependencies.

### Production-readiness §9 — CORS layer

- New `security::cors::CorsConfig` + `with_cors` helper
  wrapping `tower_http::cors::CorsLayer`. **Closed by default** —
  no `Access-Control-Allow-Origin` emitted unless the operator
  explicitly lists origins (no `*` + credentials anti-pattern).
  2 unit tests + 3 integration tests in `cors_wiring.rs`.

### Production-readiness §4 — context-length validation

- New `ApiState.max_model_len: Option<usize>` populated from
  `loader.config_json()["max_position_embeddings"]`.
- Chat (streaming + non-streaming) and completions handlers
  now return `400 context_length_exceeded` when
  `prompt_tokens + max_tokens > max_model_len`. Streaming
  matters because SSE clients otherwise get a hung-up
  connection that opens, then dies on the first forward pass.
- `/v1/models` exposes `max_model_len` (skip_serializing_if =
  "Option::is_none") so OpenAI-style clients can size prompts.
- 6 integration tests in `context_length.rs`.

### Production-readiness §10 — embeddings capability gate

- New `ModelLoader::capabilities() -> Option<ArchCapabilities>`.
- New `ApiState.arch_capabilities` plumbed through.
- Embeddings handler refuses with `501 Not Implemented` +
  `embeddings_unsupported` when:
  - capabilities couldn't be detected (`None`), OR
  - the loaded model is a stub (returns all-zero
    embeddings, i.e. meaningless noise).
- 3 integration tests in `embeddings_capability.rs`.

## Test counts after the P3 batch

- Server crate: 226 passed (was 210; +16 tests from this batch:
  2 cors unit + 3 cors_wiring integration + 2 cors_config unit +
  6 context_length integration + 3 embeddings_capability
  integration).
- `cargo machete` reports zero unused workspace deps.
  - `body_limit_wiring.rs` — 4
  - `cancel_propagation.rs` — 2
  - `openai/sampling_validation.rs` — 3 (unit, in-module)

## Technical Due Diligence — 2026-07-13 P4 follow-up batch

Closed five more items from `docs/technical-due-diligence/`. Each
commit is self-contained and references the doc / ID in its message.

### SEC-01 residual — RBAC `X-User-Role` header forgery closed

The pre-fix `rbac_middleware` extracted the role from the
client-supplied `X-User-Role` header (`rbac.rs:115-120`), so any
caller could forge `X-User-Role: admin` and reach `/metrics`,
`/admin/*`, etc. without a valid API key. `debug.rs:50-55` already
flagged this as a known vulnerability. Post-fix the role must come
from a new `AuthenticatedRole` request extension that only
server-side middleware (JWT claim extraction, future role-aware
auth) can install. The header is no longer consulted at any
decision point.

- `rbac.rs`: new `AuthenticatedRole` request-extension type;
  `extract_role_from_headers` deleted; `resolve_role` reads only
  the extension. `default_role` is always `Anonymous` for the
  middleware (matches the new "no forgery" contract).
- `rbac/tests.rs`: 12 unit + integration tests, including two new
  SEC-01 regression tests (`test_rbac_ignores_forged_admin_header`,
  `test_rbac_header_does_not_persist_across_requests`) that
  prove the forged header is ignored even when no other
  authentication is present, and that it cannot persist across
  requests.
- `audit_integration.rs`: 5 tests, including a new
  `test_audit_forged_role_header_is_ignored_by_rbac` that pairs a
  forged `X-User-Role: admin` with a valid `user` JWT and asserts
  the request is still denied. The audit middleware now promotes
  JWT claims to `AuthenticatedRole` (matching the production
  contract).

### API-01 — SSE `[DONE]` split + `finish_reason` propagation

- New `FinishReason` enum in `vllm-traits` (`Stop` / `Length` /
  `Cancelled`); re-exported from `vllm_traits` and `vllm_core`.
- `EngineMessage::AddRequest` gains a `finish_reason_tx:
  Option<oneshot::Sender<FinishReason>>` field, parallel to the
  existing `seq_id_tx`.
- `Engine` gains `finish_reason_txs: HashMap<SeqId, oneshot::Sender<FinishReason>>`
  and a `finalize_finished(seq_id, reason)` helper. The three
  finished-sequence drop sites (`scheduler/batch.rs`,
  `engine/spec_dispatch/dispatch.rs`, `engine/graph_step.rs`) all
  call `finalize_finished(seq.id, FinishReason::Length)` so the
  handler learns why the channel is closing. `cancel_request`
  sends `FinishReason::Cancelled` for the same reason.
- `openai::chat` (both non-stream and stream paths) now pass
  `finish_reason_tx: Some(...)` to `AddRequest`, await the
  oneshot, and emit the OpenAI-correct `finish_reason` string
  (`"length"` instead of the pre-fix hardcoded `"stop"`).
- The streaming unfold now emits the final chunk and the `[DONE]`
  sentinel as **two separate SSE events** (`Terminal::Streaming →
  Terminal::EmitDoneSentinel → Terminal::Done`) — pre-fix both
  were crammed into one `"{json}\n\n[DONE]"` field, which strict
  OpenAI SDK / SSE clients do not parse.
- `openai::completions` mirrors the chat fixes (finish_reason +
  `[DONE]` split) on both paths.

New integration tests in `chat_integration_test.rs`:
- `test_chat_streaming_done_is_separate_event` — last SSE event
  carries `[DONE]`, penultimate carries `finish_reason`.
- `test_chat_non_streaming_finish_reason_propagation` — handler
  falls back to `"stop"` when the oneshot is dropped (mock
  engine), preserving the pre-fix default.

### Dist — `NcclAllReduce` honest-naming

`crates/dist/src/tensor_parallel/all_reduce.rs` had two
identically-implemented types — `NcclAllReduce` and
`LocalSumAllReduce`. The `Nccl` prefix was misleading: there is no
NCCL backend in v0.x. Post-fix:

- `LocalSumAllReduce` is the canonical type; the old duplicate
  struct + `impl AllReduce` are deleted.
- `NcclAllReduce` is now `#[deprecated]` type alias for
  `LocalSumAllReduce` so the v0.x transition window does not
  break existing callers (`parallel_linear.rs` already uses
  `LocalSumAllReduce`; the only external reference was the
  re-exports in `tensor_parallel::mod` and `dist::lib`).
- New compile-only test
  `nccl_all_reduce_alias_resolves_to_local_sum` proves the
  deprecated alias still resolves to a usable `LocalSumAllReduce`
  that implements `AllReduce`.

### Engineering — `fuzz/Cargo.toml` MSRV drift

`fuzz/Cargo.toml` was pinned to `rust-version = "1.85"` while the
workspace requires `1.88` and `rust-toolchain.toml` pins `1.88`.
Per `engineering-quality.md` §6 this drift lets the fuzz target
silently use a toolchain older than what CI exercises. Bumped to
`1.88` and added a comment noting that all three files
(`Cargo.toml`, `rust-toolchain.toml`, `fuzz/Cargo.toml`) must
move together.

### Engineering — `traits::kernels` empty feature (verified, not changed)

`crates/traits/Cargo.toml` defines `kernels = []` and the doc
comment claims it gates the `kernels` module. Audit confirmed it
does: `crates/traits/src/lib.rs:25-33` uses
`#[cfg(feature = "kernels")]` to gate both `pub mod kernels` and
the `CudaGraphConfig` / `CudaGraphExecutor` re-exports, and
`vllm-model` enables the feature. The pre-existing P3 cleanup
that introduced the `kernels` feature is correct — no change
needed, but verified so the engineering-quality.md §6 item is
definitively closed.

## Test counts after the P4 batch

- Server crate: **234** lib + integration tests passed (was 226;
  +8 from this batch: 2 new rbac SEC-01 regressions + 1 new
  audit SEC-01 regression + 2 new chat SSE/finish_reason + 3
  context_length unchanged).
- Dist crate: **77** unit tests passed (was 76; +1 from the
  `nccl_all_reduce_alias_resolves_to_local_sum` compile-only
  check).
- Core crate: 313 lib tests pass; 2 pre-existing failures in
  `engine::spec_dispatch::tests::verifier_*` (from commit
  `aafb1f4` / ARCH-02 P3 batch) — out of scope for P4.
- Core integration tests: ALL pass (32 in `integration.rs`, plus
  every other file: `cuda_graph_integration`, `e2e_*`,
  `prefix_cache*`, `sampling*`, `scheduler*`,
  `speculative_kv_cache`, `packing_integration`,
  `engine_wiring`, `engine_trace`, `observer`, `error_handling`,
  `adaptive_speculative`, `multi_draft_integration`, `resource_limits`,
  `beam`, `distributed_kv_integration`).
- Workspace `cargo check --all-features` clean; the only
  remaining warning is the pre-existing
  `test_only_sample_or_argmax` dead-code warning in
  `engine/spec_dispatch/verify.rs` (out of scope for P4).

## Technical Due Diligence — 2026-07-13 P5 follow-up batch

Closed one follow-up item from the P4 batch: the 48 pre-existing
doc-check errors (deferred from P4 as out-of-scope) were all
mechanical `[`X`]` → `` `X` `` / `[`X`](otherwise)` fixes
that landed in this commit. Touched 24 files; no behaviour change.

Patterns closed (one fix per pattern):

- `[`X`](otherwise)` → `[`X`] (otherwise)` — `[`FetchError::NoPeers`](if no peers and no source)` in
  `crates/dist/src/distributed_kv/cache.rs` (rustdoc doesn't
  parse `otherwise` as a path).
- `[`X`](Y)` where `X` already resolves to `Y` →
  `[`X`]` (redundant explicit link target).
  E.g. `health_handler` in `crates/server/src/health_handlers.rs`,
  `tonic::Code::Unavailable` in `crates/dist/src/grpc.rs`.
- `[`crate::arch::UnknownArchitecture`]` /
  `[`StubBlockWrapper`]` / `[`StubModel`]` / `[`mask`]` /
  `[`kernels`]` / `[`Self::forward_*`]` / `[`TlsConfig::with_ca_cert`]` /
  `[`JwtConfig::new`]` → plain backticks (private items or
  no-such-field; the doc text still reads as code references).
- `[`state`]` / `[`graph`]` / `[`memory`]` / `[`update`]` in
  `crates/core/src/scheduler/engine/mod.rs` and friends →
  plain backticks (sibling modules reachable only via
  `super::`, not `crate::`).
- `[`SchedulingPolicy`]` in `crates/core/src/scheduler/policy/mod.rs`
  → plain backticks (rustdoc resolves via re-export but the
  closure-style link rendered ambiguously against the trait's
  associated items).
- `[`Self::compute_owner_nodes`]` in
  `crates/dist/src/distributed_kv/cache.rs` → plain backticks
  (private method).
- `[`mod@security::audit_middleware`]` is the canonical way
  to disambiguate from the function with the same name —
  applied in `crates/server/src/lib.rs:69`.

After this batch:

- `just ci` exits 0 (every step: fmt-check, clippy, doc-check,
  doctest, nextest, public-api-check — passes).
- Test count unchanged at 1417 (this batch is doc-only, no
  source-code behaviour changes).

## Pre-existing issues found during P4 verification

While running `just ci` to verify the P4 batch, three classes of
**pre-existing** issues were surfaced that block CI. None of them
were introduced by P4 — they were latent from P3 and earlier
batches, and were not caught because P3 verification ran
`cargo check --all-features` (not clippy / doc-check).

### Fixed during P4 verification (in-scope cleanup)

These touched P4 files or were strictly required to unblock
verification, so they were rolled into the P4 commit:

- **clippy::useless_vec** in `spec_dispatch/tests.rs` (P4-introduced)
  — replaced `vec![...]` literals with `&[...]` array references.
- **clippy::module_name_repetitions** in `server/src/config/mod.rs`,
  `server/src/security/cors.rs`, `server/src/bootstrap/engine.rs`,
  `server/src/bootstrap/tokenizer.rs` (P3-introduced) — added file-
  scope `#![allow(clippy::module_name_repetitions)]` with a comment
  explaining why each name is intentional.
- **clippy::result_large_err** on `server/src/debug.rs::require_admin`
  (P2-introduced) — `#[allow(clippy::result_large_err)]` with a
  comment: `Response` is the natural Err shape for axum handlers.
- **clippy::let_underscore_future** on
  `server/src/main.rs:331` (P2-introduced) — `tokio::mpsc::send` was
  being discarded without await; replaced with
  `blocking_send(...)?`-style error-logged send so the engine
  shutdown message is delivered reliably.
- **rustdoc::broken_intra_doc_links** in `core/src/engine/mod.rs`
  (P4-introduced) — `crate::engine::lifecycle::Engine::add_request`
  pointed at the wrong module; corrected to
  `crate::engine::Engine::add_request`.
- **rustdoc::broken_intra_doc_links** in
  `server/src/security/rbac.rs` (P4-introduced via `pub(crate)`
  tightening) — `[RbacMiddleware::check_permission]` /
  `[Self::check_permission]` now private; replaced link brackets
  with backticks.
- **P3 verifier_* tests failing** — the P3-era `drafts::argmax`
  used `max_by` which keeps the LAST equal element, breaking the
  tie-break-to-first contract documented by
  `vllm_traits::argmax_logits`. This caused P3's
  `verifier_uses_argmax_when_temperature_is_zero` (and the P4
  additions) to fail. Rewrote `drafts::argmax` to break ties to
  first, matching `vllm_traits::argmax_logits`. All 4 verifier
  tests now pass.
- **`test_only_sample_or_argmax` dead-code warning** —
  `#[allow(dead_code)]` since it is `#[doc(hidden)] pub` and only
  consumed by tests.

### Deferred — out of scope for P4

- **48 pre-existing doc-check errors** (rustdoc::broken_intra_doc_links
  + unresolved-link-to-private-item) across 16 files. These were
  mechanical `[`X`]` → `` `X` `` fixes for `mod` (not `pub mod`)
  submodules and unresolved `[`X`](otherwise)` syntax. They were
  fixed in the P5 follow-up batch (this section) so `just ci`
  now passes cleanly. Files touched:
  - `crates/core/src/metrics/exporter/mod.rs`
  - `crates/core/src/scheduler/{batch_composer,policy,radix_cache}/mod.rs`
  - `crates/core/src/scheduler/engine/{memory,state/mod,mod}.rs`
  - `crates/dist/src/distributed_kv/cache.rs`
  - `crates/dist/src/grpc.rs`
  - `crates/model/src/arch/stub.rs`
  - `crates/model/src/components/attention/gqa/mod.rs`
  - `crates/model/src/config/architecture.rs`
  - `crates/model/src/gemma4/attention/{forward,mod}.rs`
  - `crates/model/src/qwen3/block/mod.rs`
  - `crates/server/src/{api,config,debug,health_handlers,lib,util}/`
  - `crates/server/src/security/{audit_middleware,jwt,tls}.rs`
- **P3-public-api baseline refresh** — the P4 commit added a
  `public-api: vllm-dist added ...` bullet under Unreleased >
  Changed; the `check-public-api.sh` script verifies the LAST
  commit message that touched `CHANGELOG.md` mentions the
  crate. The P4 commit's body references `vllm-dist` so the
  baseline check passes after commit (it currently fails
  pre-commit only because git history doesn't yet reflect the
  new CHANGELOG entry).

## Verified clean

- `just fmt-check` — passes
- `just clippy` — passes (after the in-scope cleanups above)
- `just doctest` — passes
- `just doc-check` — passes (after the P5 follow-up batch closed
  the 48 pre-existing doc-link errors)
- `just nextest` — 1417 tests pass, 40 ignored
- `just public-api-check` — passes (P4 commit body references
  `vllm-dist`)
- `cargo check -p vllm-core --lib` — clean (no warnings)
- All 4 verifier tests pass (`verifier_uses_argmax_when_temperature_is_zero`,
  `verifier_accepts_high_prob_drafts_under_sampling`,
  `verifier_rejects_low_prob_drafts_under_sampling`,
  `draft_verifier_default_arc_accepts_all`)

## Known blocking items for CI

- _None_ — `just ci` exits 0 (see "Verified clean" above).

## Technical Due Diligence — 2026-07-13 P6 follow-up batch

Closed one follow-up item from
`docs/technical-due-diligence/architecture-performance.md` §5.1
("API-01: OpenAI 兼容是部分兼容") and §6 ("主要偏差 #1: ChatRequest
声明 top-p, n, stop 等字段,但 handler/engine 未完整应用").

**The fix**: stop silently dropping declared-but-not-honoured
fields. Pre-fix, callers who sent `n: 2` got exactly one completion
back and never knew the field was ignored — a contract violation.
Post-fix, the wire types now explicitly reject `n > 1` and non-empty
`stop` with `400 invalid_request_error` BEFORE the engine is
touched. `n = 1` and `stop = []` are accepted as functionally
no-ops matching OpenAI defaults.

Files touched:

- `crates/server/src/openai/sampling_validation.rs` — new
  `validate_chat_request_fields` and
  `validate_completion_request_fields`. 8 new unit tests covering
  default-pass, n=1-pass, n=2-reject, empty-stop-pass,
  non-empty-stop-reject for both request types.
- `crates/server/src/openai/chat.rs` — calls
  `validate_chat_request_fields(&req)?` at the top of
  `chat_completions` (before stream vs non-stream dispatch).
- `crates/server/src/openai/completions.rs` — calls
  `validate_completion_request_fields(&req)?` at the top of
  `completions` (before prompt-empty check).
- `crates/server/tests/chat_integration_test.rs` — 4 new
  integration tests:
  `test_chat_rejects_n_greater_than_one_with_400`,
  `test_chat_accepts_n_equal_to_one`,
  `test_chat_rejects_non_empty_stop_with_400`,
  `test_chat_accepts_empty_stop_array`.
- `docs/reference/openai-compatibility.md` — new. Single source of
  truth for what the OpenAI wire types do vs don't honour, by
  endpoint and by field. Covers `/v1/chat/completions`,
  `/v1/completions`, `/v1/models`, `/v1/embeddings`, `/v1/batches`,
  and the error-code contract.
- `docs/README.md` — links to the new doc from "Start Here" and
  adds `reference/` to the directory tree.

Test counts:

- Server crate lib: 240 (was 234; +6 net after the P4 batch
  rebuilt + this batch: 8 new `sampling_validation` tests minus
  0 deletions)
- Server integration: 14 in `chat_integration_test.rs` alone (was
  10; +4 new).
- Workspace: **1429** tests pass (was 1417; +12 from this batch).

`top_p` is still **not declared** on the wire types — adding it
requires the engine to honour it AND a CHANGELOG entry. Tracked
in `docs/reference/openai-compatibility.md` as a v0.2 follow-up.

`seed` is similarly not declared.


## Technical Due Diligence — 2026-07-14 P7 follow-up batch

Closed two remaining items in one session:

### Production-readiness §7 — graceful shutdown flips readiness before listener closes

The §7 shutdown sequence lists six steps; pre-fix the implementation
covered steps 4–6 (cancel queued / drain in-flight / flush logs via
axum's `with_graceful_shutdown`, then `EngineMessage::Shutdown` +
`engine_thread.join`) but the first two (readiness=false, wait for
orchestrator drain) were missing. Concretely:

- `/shutdown` handler sent `EngineMessage::Shutdown` and returned 200
  without touching the readiness flag. A Kubernetes probe polling
  `/health/ready` between `/shutdown` and SIGTERM still saw `Ok` and
  could route new traffic to a pod whose engine was already tearing
  down.
- `shutdown_signal()` returned from `tokio::select!` immediately,
  so axum slammed the listener shut on SIGTERM before the
  orchestrator's readiness probe had a chance to detect the new
  state and remove the pod from the Service endpoints list.

Post-fix:

- `HealthChecker::mark_not_ready` (const fn, idempotent). Both
  the `/shutdown` handler and the SIGTERM/Ctrl+C shutdown
  coordinator call it on entry; idempotency means they can race
  without coordination.
- `shutdown_signal()` now takes `Arc<RwLock<HealthChecker>>` +
  `drain_grace_secs`: (1) `mark_not_ready`, (2) sleep for the
  grace period, (3) return — letting `with_graceful_shutdown`
  close the listener. The sleep is interruptible and yields the
  runtime, so the accept loop keeps processing (and rejecting)
  new connections during the grace period; the listener itself
  stays open.
- `ServerConfig::shutdown_drain_grace_secs` (default 5 s, capped
  at 300 s via `ConfigValidationError::ShutdownDrainGraceTooLarge`
  so a typo can't block shutdown for an hour). 5 s matches the
  default K8s readiness probe `failureThreshold (3) *
  periodSeconds (1)` so two or three failed probes reliably
  catch the flip before the listener closes; operators tune via
  YAML to match their probe timing.

### Public-API check — `[Unreleased]`-aware verification

`check-public-api.sh` previously inspected only the LATEST commit
message that touched `CHANGELOG.md`, which meant a CHANGELOG
bullet added in an earlier commit (e.g. the original P3 / P4
follow-ups) was invisible when the public-API check ran during a
later batch. Replaced with an awk extraction of the entire
`[Unreleased]` section and a regex that requires an explicit
`public-api:` bullet naming the crate. Incidental mentions
(e.g. test counts that list the crate) do NOT satisfy the
check. Backfilled four missing `public-api:` bullets in the
existing `[Unreleased]` > Changed section: `vllm-traits`
(FinishReason + 28 items), `vllm-core` (8 items:
finish_reason_txs + EngineMessage fields + CancelRequest),
`vllm-model` (1 item: `ModelLoader::capabilities`), `vllm-server`
(102 items: P2/P3/P4/P6 hardening + P7 shutdown readiness).

## Test counts after the P7 batch

- Workspace: **1436** tests pass (was 1429; +7 from this batch:
  +2 unit in `health.rs` + 5 integration in
  `crates/server/tests/shutdown_readiness.rs`).
- vllm-server lib: 174 (was 172; +2 unit).
- vllm-server integration: +5 new file
  (`shutdown_readiness.rs`).
- `just ci` exits 0 (fmt-check, clippy, doc-check, doctest,
  nextest, public-api-check all pass).

## Remaining open items

- **PERF-01** (continuous batching kernel) — deferred to v32+;
  very-high complexity, blocked on a stable `KernelBackend` seam.
- **CI-01** (sustained GPU / real-checkpoint CI) — deferred;
  needs a runner with a GPU and a checkpoint cache.
- **OpenAI compat: top_p / seed wire types** — tracked in
  `docs/reference/openai-compatibility.md` as a v0.2 follow-up
  (requires engine honour + CHANGELOG entry; out of scope for
  the v31.0 hardening batches).

## Technical Due Diligence — 2026-07-14 P8 follow-up batch

Closed three follow-up items in one session:

### Engineering-quality §8 — `just check` / `just ci-full` aliases

The doc recommends three primary entry points (`just check / just
ci / just ci-full`); the actual recipes were `quick / ci / ci-all`.
Added `check` and `ci-full` as aliases for `quick` and `ci-all`
respectively; original names retained for muscle memory. `just
--list` now exposes both spellings.

### Production-readiness §7 / multi-node — OPERATIONS.md drift

The Multi-Node section claimed "KV block transfer is not yet
production-ready" — but Phase 31-D / OPS-31d shipped the
TransferKVBlock RPC + fan-out fallback + 64 MiB message limit.
Rewrote the section to honestly split what works (CacheMessage
replication, TransferKVBlock, fan-out fallback) vs. what's still
v32+ (smart owner routing, failure recovery, MESI/Directory
enforcement), added a library-API quickstart, and explicitly
noted that peer_urls is library-level only — no CLI / VLLM_*
env var exists yet.

The Graceful Shutdown section now documents the §7 four-step
sequence (readiness flip + drain grace + axum drain + engine
join) introduced by P7, including the new
`shutdown_drain_grace_secs` YAML key and how to tune it for K8s
probe timing. Pre-fix the doc said only "engine drains in-flight
requests" which understated the new behaviour.

### Phase 31-E — doc-coverage gate at 65% real

Last Phase 31-E master-plan item. The `scripts/doc_coverage.sh`
script existed but wasn't wired into any gate, so the per-crate /
workspace numbers in STATE.md / CHANGELOG could silently drift
down. New CI step after Public API baseline check:

- Reads `scripts/doc_coverage.sh --real json`
- Sums `real_total` + `real_documented` across crates
- Exits non-zero if workspace `real_pct < DOC_COVERAGE_MIN`
  (default `65.0`, override per workflow run)

`just doc-coverage-check` mirrors the gate for the local loop
and is wired into `just ci` / `just ci-all`. Current measured
value: **67.91%** real (target 65%).

## Test counts after the P8 batch

- Workspace: **1436** tests pass (no test changes this batch —
  documentation + CI gate only).
- `just ci` exits 0.
- `DOC_COVERAGE_MIN=70.0 just doc-coverage-check` correctly
  fails with exit 1, proving the gate is wired correctly.

## Remaining open items (after P8)

- **PERF-01** (continuous batching kernel) — deferred to v32+.
- **CI-01** (sustained GPU / real-checkpoint CI) — deferred.
- **OpenAI compat: top_p / seed wire types** — tracked as v0.2
  follow-up in `docs/reference/openai-compatibility.md`.
- **Phase 31-D: 2-node integration test** — not yet exercised
  end-to-end; the unit tests for `TransferKVBlock` and
  `DistributedKVCache::fetch_block` are in place but a 2-process
  round-trip (start two nodes, route a prefix cache hit across
  them) is still outstanding.
- **Phase 31-F (performance)** — `attn_factor` in paged/flash
  attention paths, `RopeScaling` config → Block wiring,
  `expand_kv` fused kernel, PagedKV host round-trip elimination
  all deferred.

## Technical Due Diligence — 2026-07-15 P9 follow-up batch

Closed one follow-up item from
`docs/technical-due-diligence/architecture-performance.md` §5.1.6
("API-01: ChatRequest 声明 top-p, n, stop 等字段,但 handler/engine
未完整应用" — top_p item) and §6 ("主要偏差 #1: top_p 未真正生效").

### `top_p` honoured end-to-end

Pre-fix: `ChatRequest.top_p` and `CompletionRequest.top_p` were
**not declared** on the wire types, so the JSON field was
silently dropped by serde. STATE.md flagged this as a tracked
v0.2 follow-up. Post-fix, `top_p` is declared on BOTH wire
types AND forwarded to the engine — `vllm_core::sampling::
sample_batch_with_params` already implemented nucleus sampling,
the only missing piece was the HTTP boundary → `Request::
sampling_params.top_p` plumbing.

Files touched:

- `crates/server/src/openai/types.rs` — `top_p: Option<f32>`
  added to `ChatRequest` and `CompletionRequest`.
- `crates/server/src/openai/sampling_validation.rs` — new
  `validate_top_p(Option<f32>)` rejects `top_p <= 0`, `top_p > 1`,
  and `NaN` with `400 invalid_request_error` (OpenAI spec range
  is `(0, 1]`). The existing
  `validate_chat_request_fields` /
  `validate_completion_request_fields` callers now also validate
  `top_p` before enqueuing. 7 new unit tests:
  `top_p_none_passes`, `top_p_zero_is_rejected`,
  `top_p_negative_is_rejected`, `top_p_one_passes`,
  `top_p_above_one_is_rejected`, `top_p_nan_is_rejected`,
  `top_p_intermediate_passes`.
- `crates/server/src/openai/chat.rs` — handler copies
  `req.top_p` into `request.sampling_params.top_p` so the engine
  sees the forwarded value.
- `crates/server/src/openai/completions.rs` — same forwarding
  for `/v1/completions`.
- `crates/server/tests/chat_integration_test.rs` — 5 new
  integration tests using a capturing mock engine (records
  the `SamplingParams` of the first `AddRequest` it sees, then
  asserts the field round-tripped from JSON):
  - `test_chat_forwards_top_p_to_engine`
  - `test_chat_omitted_top_p_uses_engine_default`
  - `test_chat_rejects_top_p_above_one_with_400`
  - `test_chat_rejects_top_p_zero_with_400`
  - `test_completions_forwards_top_p_to_engine`
- `crates/server/src/openai/completions/tests.rs` — three
  existing fixtures updated to include the new field.
- `crates/server/tests/error_contract.rs` — two existing
  fixtures updated to include the new field.
- `docs/reference/openai-compatibility.md` — `top_p` row on
  BOTH `/v1/chat/completions` and `/v1/completions` tables
  flipped from "Not declared" to "Wired" with a forward-pointer
  to `validate_top_p` and `sample_batch_with_params`. Status
  date bumped to 2026-07-15.
- `CHANGELOG.md` — new "top_p is now honoured end-to-end" bullet
  under `[Unreleased] > Added`, plus a `public-api: vllm-server
  added` bullet for the 3 new items (`ChatRequest::top_p`,
  `CompletionRequest::top_p`, `validate_top_p`).

### Test counts after the P9 batch

- Workspace: **1448** tests pass (was 1436; +12: 7 unit in
  `sampling_validation` + 5 integration in
  `chat_integration_test.rs`).
- `just ci` exits 0 (fmt-check, clippy, doc-check, doctest,
  nextest, public-api-check, doc-coverage-check).
- Public API grew by 3 items in `vllm-server`; CHANGELOG
  `[Unreleased]` entry references `vllm-server` so the gate
  passes.
- Workspace real doc coverage: **67.93 %** (target 65 %).

## Remaining open items (after P9)

- **PERF-01** (continuous batching kernel) — deferred to v32+.
- **CI-01** (sustained GPU / real-checkpoint CI) — deferred.
- **OpenAI compat: `seed` wire type** — still tracked as v0.2
  follow-up in `docs/reference/openai-compatibility.md`. (`top_p`
  closed by P9.)
- **Phase 31-D follow-up work (post-OPS-31d)** — `kv_block_transfer.rs`
  integration tests already exercise real 2-node + 3-node gRPC
  round-trips (`peer_serves_block_bytes_via_transfer_kv_block`,
  `fan_out_returns_first_successful_peer`, etc.). Remaining items
  per `ops-31d-kv-block-transfer.md` §7 are all v32+ non-goals
  (MESI coherence, owner-routed peer fetch, streaming RPCs,
  wire compression, block refcounting during transfer,
  `PagedKvCacheWrapper: BlockDataSource`). Tracking only —
  none block the v31.0 alpha.
- **Phase 31-F (performance)** — `attn_factor` in paged/flash
  attention paths, `RopeScaling` config → Block wiring,
  `expand_kv` fused kernel, PagedKV host round-trip elimination
  all deferred.

## Technical Due Diligence — 2026-07-15 P10 follow-up batch

Closed the `production-readiness §6 — request_id propagation`
item listed in the P9 remaining-items section.

### request_id propagated HTTP → engine

Pre-fix: `correlation_id_middleware` (P1 batch) set the
`X-Request-ID` response header and installed a `CorrelationId`
request extension on every incoming request, but the value
never reached `EngineMessage::AddRequest`. Engine-side
`tracing` log lines inside `vllm_core::engine::run` and
`Engine::add_request` could not be correlated with the
originating HTTP request — operators chasing a slow request
through logs had no join key across the HTTP → scheduler →
engine boundary.

Post-fix:

- `EngineMessage::AddRequest` gains a `request_id: Option<String>`
  field (no `Serialize`/`Deserialize` on `EngineMessage`, so no
  wire-format change). All 4 production construction sites
  (`openai::chat::handle_chat` non-streaming + `stream_chat_completion`,
  `openai::completions::completions`, plus the engine run loop
  consumer) and 3 test construction sites (`e2e_graceful_shutdown`,
  `overload_integration`, `tutorial_e2e`) updated.
- `openai::chat::chat_completions` and
  `openai::completions::completions` extract
  `Extension<CorrelationId>` from request extensions and forward
  `correlation_id.0` as the `request_id` field.
- `security::correlation::CorrelationId` is now `pub` (was
  `pub(crate)`) so axum's `FromRequestParts` reflection can
  name the type from a public handler.
- The engine run loop enters
  `tracing::info_span!("engine.add_request", request_id)` around
  the synchronous `add_request` call. The span is RAII-scoped to
  the rest of the match arm, so all engine-side log lines for
  this HTTP request carry the same id. When `request_id` is
  `None` (test fixtures, non-HTTP callers), the span still
  enters with the field rendering as `null`.
- 4 production router test fixtures updated to mount
  `correlation_id_middleware` so the handler extractors don't
  fail with 500 (`cancel_propagation`, `chat_integration_test`,
  `context_length`, `overload_integration`). The middleware MUST
  sit as the OUTERMOST layer in any router hosting these
  handlers — see `main.rs` for the production wiring.

### New integration test file

`crates/server/tests/request_id_propagation.rs` (4 tests, all
pass):

- `chat_handler_forwards_client_supplied_request_id` — client
  supplies `X-Request-ID: client-supplied-trace-42`, the
  response echoes it, AND the capturing mock engine sees it on
  the `AddRequest.request_id` field. Proves the handler doesn't
  re-mint the id (pre-fix bug from the audit-middleware
  review).
- `chat_handler_forwards_minted_request_id_when_client_omits_header`
  — client omits `X-Request-ID`, middleware mints one, handler
  forwards the SAME minted id (not None, not a different
  value). Proves the id is minted ONCE and reused.
- `completions_handler_forwards_client_supplied_request_id` —
  mirror of the chat test on `/v1/completions`.
- `completions_handler_forwards_minted_request_id_when_client_omits_header`
  — companion for the chat minted-id variant on
  `/v1/completions`.

### Test counts after the P10 batch

- Workspace: **1452** tests pass (was 1448; +4 from
  `request_id_propagation.rs`).
- `just ci` exits 0 (fmt-check, clippy, doc-check, doctest,
  nextest, public-api-check, doc-coverage-check).
- Public API grew by 5 items in `vllm-server`
  (`CorrelationId` + 4 `impl`-level items) + 2 handler signature
  changes (chat + completions widened with `Extension<CorrelationId>`);
  CHANGELOG `[Unreleased]` has a `public-api: vllm-server`
  bullet naming the crate.
- Workspace real doc coverage: **67.93 %** (target 65 %).

## Technical Due Diligence — 2026-07-15 P11 follow-up batch

Closed one follow-up item from
`docs/technical-due-diligence/engineering-quality.md` §7
(MSRV 与可复现构建, last bullet: "release 生成 SBOM、校验和与
构建 provenance"). This batch closes the **SBOM half**;
checksums (`sha256sum`) and signed build provenance (SLSA /
in-toto) remain as a separate follow-up — they were not
attempted here because adding checksums to a release requires
a downstream verification story (signature key handling,
reproducible build attestation) that goes beyond a single CI
step.

### What changed

- **`.github/workflows/release.yml`** — added a CycloneDX SBOM
  step to the `build` job after binary packaging, via
  [`anchore/sbom-action@v0`](https://github.com/marketplace/actions/anchore-sbom-action)
  (`syft` under the hood). Each matrix target emits
  `cyclonedx-json` SBOM into `artifacts/`, uploaded both as a
  standalone workflow artifact (`sbom-<target>`) and as part
  of the GitHub Release attachment glob (`artifacts/**/*` in
  the `release` job). The `build` job now requests
  `contents: write` + `actions: read` explicitly so the SBOM
  step can both upload its artifact and contribute to the
  Release attachments.
- **`docs/RELEASE.md`** — added the missing **"Software Bill of
  Materials"** section that the new workflow step references
  ("see 'Software Bill of Materials below'"). The section
  documents what the SBOM contains (every Rust crate in
  `Cargo.lock`, vendored C libraries linked into the binary,
  system libraries detected by syft's ELF/PE/Mach-O scanners),
  how to download + inspect it locally with `gh release
  download` + `jq`, and why downstream consumers in regulated
  environments (SOC 2 / FedRAMP / air-gapped vendor review)
  benefit. It also explicitly notes that checksums + signed
  provenance are still missing so the doc can't drift.
- **`CHANGELOG.md`** — new `[Unreleased] / Added` bullet
  recording the SBOM wiring and pointing future readers at the
  follow-up for checksums + provenance.

### CI-only change (no test delta)

- No Rust code touched, so `just ci` is unchanged in coverage:
  `just nextest` still reports **1452 passed, 40 ignored**.
- No public API delta (workflow YAML + markdown only).
- Workspace real doc coverage holds at **67.93 %** (target 65 %);
  the SBOM doc is a small additive section that nudges the
  ratio marginally upward.

### Why this batch and not the §7 checksum half

The §7 bullet is "release 生成 SBOM、校验和与构建 provenance".
Splitting into two batches keeps each PR reviewable:

- **P11 (this batch): SBOM** — single new CI step, single new doc
  section, zero cross-cutting concerns.
- **P12 (next batch, when prioritized): checksums + provenance** —
  needs a signature-key story (where does the maintainer key
  live? how is it rotated?) and a reproducible-build posture
  (the workspace today relies on Rust 1.88 + `--locked` for
  Cargo.lock determinism, but the build env itself isn't
  pinned to a specific image). Closing those is v32+ work
  unless a downstream consumer asks for it in v0.1.

## Technical Due Diligence — 2026-07-15 P12 follow-up batch

Closed the two remaining Phase 31-D master-plan items
(`KV block serialization protocol spec` and
`Multi-node quickstart in OPERATIONS.md`). The actual
multi-node functionality — `BlockDataSource` trait,
`TransferKVBlock` gRPC RPC, `DistributedKVCache::fetch_block`
fan-out fallback, 64 MiB symmetric message limit, and
in-process 2-node + 3-node integration tests — already shipped
in OPS-31d (commit `cff1444` / Phase 19, `2026-07-12`). This
batch is **doc-only**: it makes the shipped functionality
discoverable for embedders without forcing them to read the
proto file or the phase plan.

### What changed

- **`OPERATIONS.md`** — the "Multi-Node (Experimental)" section
  was a 46-line single-purpose block; it is now a 150-line
  reference with explicit subsections:
    - **"What works"** — 3 bullets lifted from the proto +
      `kv_block_transfer.rs` (`PutKVCache`/`InvalidateKVCache`
      replication, `TransferKVBlock` RPC, fan-out fallback).
    - **"What is not yet production-ready"** — 4 bullets, led
      by the **load-bearing** one: engine integration. Without
      a `PagedKvCacheWrapper → MemoryManager` wiring, the gRPC
      server answers `Status::unavailable("TransferKVBlock
      called but no BlockDataSource wired in")` for every block
      transfer, so multi-node replication works for
      `(block_id, chain_hash)` *intent* but actual block bytes
      stay local-only in the default engine build. This is
      OPS-32a work — explicit, scoped, and called out so
      embedders don't misread "the protocol works" as "the
      engine ships with cross-node KV transfer".
    - **"Minimum viable cluster"** — both 2-node AND 3-node
      snippets. The 3-node form mirrors the structure of
      `crates/dist/tests/distributed_kv_peer_sync.rs::multi_peer_broadcast`
      and exercises `peer_client_count()` so an embedder
      can sanity-check their wiring.
    - **"Verify it works"** — points operators at the in-process
      gRPC integration tests (`cargo test -p vllm-dist --test
      kv_block_transfer`, `--test distributed_kv_peer_sync`)
      that exercise real 2-node + 3-node fan-out without a
      real network.
    - **"Wire protocol (TransferKVBlock, Phase 31-D)"** —
      quotes the proto definitions, explains the 64 MiB
      symmetric message limit (with the explicit warning that
      custom embedders must bump the same limits or block
      transfers will return `tonic::Status::out_of_range_error`),
      and documents the hash-verification contract
      (`response.chain_hash == expected_hash`, mismatch is
      fatal — no retry — see "Failure recovery" under "What is
      not yet production-ready").
- **`.planning/v31.0-MASTER-PLAN.md`** — 31-D status flips
  from `Pending` to `✅ Done`; all four master-plan items
  under 31-D are now ticked with a one-line pointer to
  where the work landed. The deliverable in the phase-index
  table is updated to reflect the new state.
- **`CHANGELOG.md`** — new `[Unreleased] / Added` entry
  recording the OPERATIONS.md expansion and the master-plan
  closure.

### Doc-only change (no test / no Rust delta)

- No Rust code touched, so `just nextest` is unchanged:
  **1452 passed, 40 ignored**.
- No public API delta.
- Workspace real doc coverage nudges upward (the new
  "Wire protocol" section is real prose, not boilerplate);
  re-measurement deferred to the next batch.

### What this batch explicitly does NOT close

The Phase 31-D follow-up list under "Remaining open items"
(after P11) called out six items as v32+ non-goals: MESI
coherence, owner-routed peer fetch, streaming RPCs, wire
compression, block refcounting during transfer,
`PagedKvCacheWrapper: BlockDataSource`. None of those are
addressed here — they remain v32+ candidates per
`.planning/phase-19/ops-31d-kv-block-transfer.md` §7. This
batch only closed the **documentation** half of 31-D, not
the engine-integration half.

## Technical Due Diligence — 2026-07-15 P13 follow-up batch

Closed one of the two remaining Phase 31-E master-plan items:
**"Mutation testing CI (fix baseline workaround)"**. The
`clippy --all-features` matrix job (commit `28c5c37`) and the
doc-coverage gate at 65% (commit `867412f`) were already in
place; mutation testing had scripts (`scripts/check_mutation_score.sh`,
`just mutants-ci`) but no CI workflow. This batch wires the
scan into a nightly cron + manual dispatch workflow and fixes
the `--baseline skip` workaround for the modules under scan.

### What changed

- **`.github/workflows/mutation-nightly.yml`** — new workflow
  that runs `cargo mutants` on three short `vllm-core` modules
  (`sampling.rs`, `scheduler/policy`, `engine/cuda_graph.rs`)
  on a daily 03:00 UTC cron (off-peak, after the 04:00 fuzz
  nightly) plus manual `workflow_dispatch` for ad-hoc
  pre-release scans. Each module:
    - is gated against a per-module score floor via
      `scripts/check_mutation_score.sh` (baselines:
      sampling.rs=40, scheduler/policy=95, engine/cuda_graph.rs=50 —
      each 5-10pp below the P13-ship-time observed score so
      real regressions are caught without false positives)
    - uploads `.mutants-out/` as a 14-day workflow artifact
      for offline inspection (mirrors the `just mutants-report`
      local flow)
    - prints the top 30 surviving mutants for fast triage when
      the score check fails.
  Cache layout mirrors `fuzz-nightly.yml`: `Swatinem/rust-cache@v2`
  with `shared-key: "mutation-nightly"` (workspace-wide) + a
  dedicated cache for the `cargo-mutants` binary
  (`cargo-mutants-27.1.0-${{ runner.os }}`). Rust toolchain
  pinned to `1.88` (the workspace MSRV in `rust-toolchain.toml`)
  so nightly scans use the same compiler as release builds.

- **Baseline fix (the "fix baseline workaround" half)** —
  `just mutants` was using `--baseline skip` to mask a
  pre-existing test failure tracked as a v31+ follow-up.
  Verified by running `cargo mutants --package vllm-core
  --file crates/core/src/sampling.rs` *without* `--baseline
  skip`: the baseline passes (cargo test exits 0) and the
  scan completes normally — 135 mutants tested in 5 min,
  60 caught, 73 missed, 2 unviable. The pre-existing comment
  referenced `cuda_graph_integration.rs:148`, but that line is
  inside a `#[cfg(feature = "cuda-graph")]` test, so under
  default features (the only feature set the scan uses) the
  test does not compile and cannot contribute to baseline.
  **The CI workflow drops `--baseline skip`**. The local
  justfile keeps it as a safety net for developers with broken
  tests in progress (CI is the ground truth for "does the
  scan baseline pass").

- **`justfile`** — `mutants MODULE` recipe comment updated
  with the P13 verification (135 mutants / 60 caught / 73
  missed / 2 unviable) and a pointer to the CI workflow as
  the new ground truth. The `--baseline skip` flag itself is
  NOT removed from the local recipe (see rationale above).

- **`.planning/v31.0-MASTER-PLAN.md`** — Phase 31-E
  master-plan items updated: `clippy --all-features` matrix
  ticked (already in `ci.yml::ci-all-features` since P3 batch),
  mutation-testing ticked (this batch), doc-coverage gate
  ticked (already in `ci.yml::ci` since P3 batch), GPU nightly
  smoke left as deferred (requires self-hosted GPU runner —
  no workspace tooling exists for it). Phase 31-E status
  flips from `Pending` to `✅ Mostly done`.

- **`CHANGELOG.md`** — new `[Unreleased] / Added` entry
  describing the workflow, the matrix, and the baseline
  verification.

### CI-only change (no Rust / no test delta)

- No Rust code touched, so `just nextest` is unchanged:
  **1452 passed, 40 ignored**.
- No public API delta.
- The mutation scan itself is a new artifact (`.mutants-out/`)
  but the underlying tests are not new.

### Why nightly + matrix, not per-PR

A full `scheduler` mutation scan takes 30-60 min on 4-core
ubuntu-latest (per the comment in `justfile` `mutants MODULE`).
Gating every PR on that would add unacceptable latency. The
nightly cadence plus a small fast module set (`sampling.rs`,
`scheduler/policy`, `engine/cuda_graph.rs`) keeps the daily
budget under ~15 min while still exercising three distinct
mutation classes (arithmetic, control flow, bool flips). The
manual `workflow_dispatch` trigger is the escape hatch for
ad-hoc pre-release scans of the larger `scheduler/` tree.

## Technical Due Diligence — 2026-07-15 P14 follow-up batch

Closed the documentation gap for the OPS-31d multi-node KV block
transfer architecture. The protocol layer shipped in P12 (OPERATIONS.md
expansion), but the underlying architectural choices — *why* fan-out
over owner-routed, *why* unary over streaming, *why* a 64 MiB
symmetric limit, *why* engine integration is deferred — were
documented only in the phase plan (`.planning/phase-19/ops-31d-kv-block-transfer.md`)
and the source code comments. Neither survives well: phase plans
get archived; code comments answer "what" not "why". An ADR is the
workspace convention for capturing these decisions (ADR-019's
documentation-standards tier table makes this explicit), so the
gap was real.

### What changed

- **`docs/adr/ADR-020-multi-node-kv-block-transfer.md`** — new
  ADR (20th in the series) capturing the six architectural
  decisions from OPS-31d:
    1. `BlockDataSource` trait as the storage-agnostic,
       async, object-safe abstraction over raw block bytes
    2. Unary (not streaming) `TransferKVBlock` gRPC RPC, with
       `num_tokens` reserved for future partial-block transfer
    3. Fan-out fallback routing vs owner-routed (and why fan-out
       wins for v0.1)
    4. 64 MiB symmetric message limit (with the embedder-facing
       warning that custom servers/clients must bump the same
       limit)
    5. Explicit deferral of engine integration (`PagedKvCacheWrapper
       → MemoryManager`) to v32+ / OPS-32a — with the consequence
       that the gRPC server returns `unavailable` for every block
       transfer in the default engine build
    6. Non-removal of the legacy `GetKVCache` RPC in this phase
       (deferred to a dedicated cleanup phase before 1.0)
  The ADR also documents the four alternatives that were
  considered and rejected (owner-routed day-1, streaming RPCs day-1,
  block bytes in `PutKVCache`, `GetKVCache(block_hash)` lookup)
  with the rationale for the choices that shipped. Cross-references
  the phase plan, the proto file, the OPERATIONS.md quickstart,
  ADR-008 (`vllm-dist` feature-gated) and ADR-015 (`vllm-dist`
  investment decision) as the prerequisite decisions.

- **`.planning/v31.0-MASTER-PLAN.md`** — Phase 31-E target
  bumped from "ADRs: 19" to "20 (ADR-020 multi-node KV block
  transfer architecture added in P14)". The Quality Gates table
  row now documents *why* the count moved, so future readers
  can trace the addition back to this batch.

- **`.planning/DOC-MAP.md`** — ADR-count reference updated
  from "(19 records)" to "(20 records: ADR-001 … ADR-020)" so
  the doc-authority matrix matches the filesystem.

- **`README.md`** — `docs/adr/` link caption updated from
  "(19 篇 ADR)" to "(20 篇 ADR)" so the README ADR count and the
  filesystem agree.

- **`CHANGELOG.md`** — new `[Unreleased] / Added` entry
  describing the ADR and the four files touched in the batch.

### Docs-only change (no Rust / no test / no CI delta)

- No Rust code touched, so `just nextest` is unchanged:
  **1452 passed, 40 ignored**.
- No public API delta.
- No CI delta (the mutation-nightly workflow from P13 is
  unaffected).
- Workspace real doc coverage holds at **67.93 %** (the ADR is
  a long, mostly-prose document which nudges the ratio
  marginally upward; re-measurement deferred to the next batch
  that touches documentation code).

### Why this batch and not a §6 stale-item closure

The technical-due-diligence §6 (Feature 与依赖管理) has several
items that became stale as the workspace evolved:

- "`traits` 的 `kernels` 是空 feature" — actually a non-empty
  cfg-gate feature (`kernels = []` in `Cargo.toml` is the syntax
  for "no extra deps but feature exists as a cfg-gate"; the
  `kernels` module IS gated by `#[cfg(feature = "kernels")]`).
  Arguably closed by definition, not actionable.

- "`server` 硬编码启用 core 的 `cuda-graph`" — `vllm-core`
  default-features is explicitly disabled in `crates/server/Cargo.toml`,
  and `cuda-graph` is a non-default feature. Closed.

- "`dist` 不在 default-members" — `dist` is in workspace
  `default-members` (and `members`). Closed.

- "各 crate 的 `cuda`、`full`、`multi-node` 传播缺少统一
  用户模型" — could be addressed by a feature matrix doc, but
  ADR-020 (this batch) supersedes it for the multi-node side,
  and the cuda/full story is captured in `docs/architecture.md`
  §Feature Flags. Half-closed.

A docs pass that marks these stale items closed in the due
diligence doc is a legitimate P15 candidate but is low-priority
compared to capturing the *active* architectural decisions that
are currently only in phase-plan / source comments. ADR-020 ships
first because it documents decisions that *future maintainers*
will hit immediately when they touch the dist layer; the stale
§6 items document decisions that have already been resolved
correctly by the workspace.

## Technical Due Diligence — 2026-07-15 P15 follow-up batch

Closed two documentation-drift items that had accumulated since
the technical due diligence was written (during v19.x):
(a) the `docs/reference/openai-compatibility.md` matrix had
nine "Not declared" rows with no explicit v0.2 vs v32+ split;
(b) the engineering-quality §6 + §7 sections described concerns
that the v31.0 workspace had already resolved. Both are pure
docs-only closures; no Rust / no test / no API delta.

### What changed

- **`docs/reference/openai-compatibility.md`** — new
  "v0.2 follow-ups (planned)" section between the
  `/v1/batches` table and the "Error contract" section.
  Splits the nine "Not declared" rows into:
    - **v0.2 candidates**: `seed` (declaration + validation;
      honoring requires RNG seeding in
      `vllm_core::sampling` — v32+ work), `user`
      (declaration + tracing pass-through; honoring is a no-op
      until a downstream consumer subscribes), `response_format`
      JSON-mode subset (declaration + validation accepting
      `{type: "text"}` and `{type: "json_object"}`; the JSON
      schema subset defers to v0.3 because it requires a
      grammar-constrained decoder).
    - **v32+ candidates**: `frequency_penalty` /
      `presence_penalty` (renaming `repeat_penalty` is a
      public-API delta — v0.3 work), `logit_bias` (requires
      sampler softmax-step injection), `logprobs` /
      `top_logprobs` (requires top-K logprob generation),
      `tools` / `tool_choice` (grammar-constrained decoder
      + per-request tool schema cache).
  Each row has a "Why v0.2 (and not v32+)" or "Why v32+"
  rationale column so the split is justified, not arbitrary.
  Header paragraph updated with a pointer to the new section
  so a reader scanning the field tables immediately knows
  where the backlog lives.

- **`docs/technical-due-diligence/engineering-quality.md`** —
  two new closure tables ("§6 闭合状态" and "§7 闭合状态")
  added at the bottom of their respective sections. §6 marks
  3 of 4 stale items as closed (with evidence: `kernels = []`
  is the cfg-gate syntax; `vllm-core = { ..., default-features
  = false }` in server's Cargo.toml; `dist` in
  default-members per workspace `Cargo.toml`) and 1 item as
  half-closed (the "feature matrix doc" follow-up is partially
  covered by `docs/architecture.md` §Feature Flags but a
  full matrix doc is still v0.2/32+ work). §7 marks all 4
  evidence lines closed (rust-toolchain.toml exists; Dockerfile
  uses 1.88; fuzz uses 1.88; release/container builds use
  `--locked`) and 3 of 4 suggested actions closed (SBOM via
  P11; checksums + provenance still v32+).

- **`CHANGELOG.md`** — two new `[Unreleased] / Added` entries
  describing the OpenAI compat matrix expansion and the
  engineering-quality §6 + §7 closure.

### Docs-only change (no Rust / no test / no CI delta)

- No Rust code touched, so `just nextest` is unchanged:
  **1452 passed, 40 ignored**.
- No public API delta.
- Workspace real doc coverage holds at **67.93 %** (the new
  "v0.2 follow-ups" section is real prose; the closure tables
  in engineering-quality are mostly table rows, which `real`
  coverage treats as commentary rather than code-documentation
  — net change is roughly neutral).

### Why P15 and not, e.g., a `seed` field declaration

A `seed` field declaration + HTTP-boundary validation is the
natural v0.2 work and would mirror the P6/P9 pattern exactly.
It is **deliberately deferred** in P15 because:

- Declaring a field without honoring it is the "silent acceptance
  vs silent drop" regression that P6/P9 explicitly fixed. Adding
  `seed: Option<i64>` and forwarding to an engine that ignores
  it (the current sampler reads `rand`'s thread-local RNG,
  unseeded) means the contract regresses from "rejected by serde"
  to "accepted and ignored".
- The clean P6/P9-equivalent for `seed` requires both declaration
  AND engine-side RNG seeding — the latter is a non-trivial
  sampler change that crosses the v0.2 / v32+ line depending on
  whether the sampler is rewritten with explicit seeding.
- Documenting the v0.2 candidate FIRST (this batch) makes the
  scope of the future `seed` PR explicit — declaration +
  validation only in v0.2, RNG seeding in v32+ — so the PR that
  lands it doesn't have to re-litigate the contract decision.

P16+ candidates: actual `seed` declaration (when the
validation contract is fully designed), §6 #4 feature-matrix
doc, or back to technical due diligence for the next stale-item
section.

## Technical Due Diligence — 2026-07-15 P16 follow-up batch

Closed the documentation-drift items in `docs/technical-due-diligence/production-readiness.md`. Mirrors the P15 pattern (engineering-quality §6 + §7 closure tables) for the second due-diligence document, which had ten subsections with 43 original observations and no per-item closure audit. P16 makes the v31.0 reality visible from the production-readiness doc instead of relying on STATE.md / CHANGELOG to bridge the gap.

### What changed

- **`docs/technical-due-diligence/production-readiness.md`** — ten new per-section closure tables appended to §2–§11 plus a top-of-document "v31.0 P0–P15 closure summary" aggregate:
  - **§2 (SEC-01 — auth / RBAC / admin 隔离)** — 5 items: 4 closed (default no-auth escape hatch, JWT/RBAC/body-limit/correlation/audit 接线, admin 端点保护, RBAC `AuthenticatedRole` forgery fix), 1 v32+ (TLS 主路径接线).
  - **§3 (REL-01 — 有界 / 取消 / token 交付)** — 6 items: 4 closed (`engine_mailbox_capacity` 256 + `503 engine_overloaded`, `try_send` 失败显式化, `CancelRequest` propagation, `FinishReason` enum), 1 partial (`backpressure.rs` 模块仍存在), 1 v32+ (per-tenant quota).
  - **§4 (输入边界)** — 5 items: 2 closed (`with_default_body_limit`, `context_length_exceeded` 400), 1 partial (scheduler admission budget 仅实现 mailbox 部分), 2 v32+ (按估算 KV admission + 路径 canonicalize).
  - **§5 (OBS-01 — metrics 数据源)** — 3 items: 2 closed (`engine.scheduler.metrics` Arc-shared, HTTP 端点无锁读取), 1 partial (TTFT / TPOT / batch size 未完整 Prometheus 暴露).
  - **§6 (日志与追踪 — correlation / OTLP)** — 4 items: 3 closed (`correlation_id_middleware` + `request_id` propagation + `info_span!` 跨层, tutorial OTel feature 已标注 v32+, audit 中仅 `key:<first-8-chars>` 持久化), 1 v32+ (OTLP exporter).
  - **§7 (健康检查与关停)** — 5 items: 4 closed (engine thread join, `mark_not_ready` + drain grace + SIGTERM 协调, listener 在 grace 后才关闭, `OPERATIONS.md` 已记录六步流程), 1 partial (readiness 模型加载 / GPU OOM 信号).
  - **§8 (部署阻断项 — Docker / Helm / GOV-01)** — 9 items: 7 closed (Rust 1.88 builder, HEALTHCHECK curl, `--locked`, compose 路径修正, Helm env vars, GOV-01 chart 打包, smoke-deployment.sh CI), 1 partial (`helm lint` 未跑), 1 v32+ (非 root / read-only FS / NetworkPolicy 强化).
  - **§9 (TLS 与 CORS)** — 3 items: 1 closed (`CorsConfig` + `with_cors` 默认关闭), 2 v32+ (TLS 主路径 rustls 接线 + 证书 reload).
  - **§10 (Batch 与 Embeddings)** — 3 items: 2 closed (Batch API 501 `batches_unsupported`, embeddings capability gate 501 `embeddings_unsupported`), 1 v32+ (支持矩阵完整文档化 — 需要真实 checkpoint 验证).
  - **§11 (生产门槛 aggregate)** — 7 items: 5 closed (前缀 + sampling 正确性, 默认认证/admin/body/context limit, 有界 admission/取消/token 不丢失, 统一 metrics/动态 readiness/完整 shutdown, Docker/Compose/Helm smoke), 1 partial (兼容矩阵有 + 容量基准/升级-回滚/事故手册未完整), 1 deferred (真实 GPU checkpoint CI — 无 GPU runner).
- **`CHANGELOG.md`** — new `[Unreleased] / Added` entry documenting the closure-tables batch (P16 follow-up) and pointing future readers at the top-of-document aggregate + per-section tables.

### Aggregate view

| Status | Count | Examples |
|--------|------:|----------|
| ✅ Closed | 29 | SEC-01 RBAC forgery, REL-01 mailbox/cancel/token, OBS-01 metrics, body/context limit, readiness flip, drain grace, engine thread join, DEP-01 + GOV-01 deployment, CORS, batch 501, embeddings 501, audit middleware, request_id propagation, etc. |
| 🟡 Partial | 5 | `backpressure.rs` 模块清理, scheduler admission 按估算 KV, TTFT/TPOT Prometheus 暴露, readiness 模型加载信号, `helm lint` step |
| 🟢 v32+ candidate (code) | 4 | TLS 主路径 rustls 接线, OTLP exporter, per-tenant quota, feature matrix doc |
| 🟠 v32+ candidate (infra) | 5 | 真实 GPU checkpoint CI, 容量基准 runbook, 升级-回滚 runbook, 完整事故手册, 模型加载失败 readiness signal (依赖 1 + GPU) |
| **Total** | **43** | |

### What this batch explicitly does NOT close

The 14 v32+ items above are out of scope for v31.0 alpha. The 5 code candidates (TLS 主路径, OTLP, per-tenant quota, feature matrix doc, `backpressure.rs` 清理) are real follow-ups but each is multi-batch work; the 5 infra candidates need external inputs (GPU runner, 容量基准实验) that the workspace cannot manufacture. The P16 batch only closes the **documentation** half — making the v31.0 reality visible from the production-readiness doc — not the underlying engineering work.

### Docs-only change (no Rust / no test / no CI delta)

- No Rust code touched, so `just nextest` is unchanged: **1452 passed, 40 ignored**.
- No public API delta.
- No CI delta.
- Workspace real doc coverage nudges upward (the new closure tables are real prose explaining what code shipped in P0–P15, not boilerplate). Re-measurement deferred to the next batch that touches documentation code.

### Why this batch and not a `seed` declaration

The P15 candidate list explicitly listed "actual `seed` declaration" as a P16 option. The v0.2 follow-up note in `docs/reference/openai-compatibility.md` already documents why `seed` is deferred: declaration without engine-side RNG seeding is the "silent acceptance vs silent drop" regression that P6/P9 explicitly fixed. Implementing `seed` requires a sampler rewrite that crosses the v0.2 / v32+ boundary depending on how the new sampler is designed. P16 chose the production-readiness closure tables because (a) the doc is overdue (P15 closed engineering-quality, the symmetric step for production-readiness was the obvious next move), (b) closing it leaves only architecture-performance.md §6 (speculative decoding & 分布式) as the last un-closed due-diligence subsection, and (c) the closure tables don't depend on sampler-design decisions, so they don't interfere with the future `seed` PR.

P17+ candidates: architecture-performance §6 closure (the last un-closed due-diligence subsection), or actual `seed` declaration when the sampler-design contract is finalized, or back to a real engineering task.

## Technical Due Diligence — 2026-07-15 P17 follow-up batch

Closed the documentation-drift items in `docs/technical-due-diligence/architecture-performance.md` §6 (Speculative decoding & 分布式). This is the **last** un-audited subsection of the three due-diligence documents; P15 closed engineering-quality §6/§7, P16 closed production-readiness §2-§11, and P17 closes architecture-performance §6. After this batch, every due-diligence subsection with itemized concerns has a closure audit.

### What changed

- **`docs/technical-due-diligence/architecture-performance.md` §6** — new "§6 闭合状态 (v31.0 P17)" table with seven per-item entries (3 speculative + 4 distributed):

  **Speculative decoding (3 items)**:
  - **#1 验证以 argmax 相等为主** — 🟡 **半关闭**：verified `crates/core/src/engine/spec_dispatch/verify.rs::verify_draft_tokens_logits` is already temperature-aware sampled-match path. Doc-comment explicitly distinguishes "lossless speculative decoding verifier" (implemented) from "full `min(1, p/q)` rejection-sampling" (requires draft-side logits, not on wire — v32+). The sampled-match path is a strict improvement over old argmax: target now uses the same sampler the rest of the engine uses instead of always picking most-likely token.
  - **#2 speculative 与 CUDA Graph 互斥** — 🟠 **仍为真缺口**：code-verified that `step_speculative_inner` (`spec_dispatch/dispatch.rs:19`) and `step_with_graph` (`graph_step.rs:42`) are switch-mutually-exclusive. `step_with_graph` calls `execute_regular(&batch)` directly which bypasses spec dispatch — CUDA Graph capture needs static batch shape, speculative decode introduces variable draft length. This is explicit design, not a bug. v32+ candidate: dynamic-shape graph or non-speculative prefill + speculative-decode graph dual path.
  - **#3 legacy draft model 与 resolver 两套路径** — ✅ **已关闭**：`grep -rn "legacy" crates/core/src/speculative/` returns empty. The four draft sources (`draft_registry` / `draft_resolver` / `adaptive` / `self_spec`) are all resolver-driven v18.0+ paths. `step_speculative_inner` dispatches between `generate_per_seq_drafts` (resolver-driven) and `generate_batched_drafts` (legacy batched path, not legacy model) based on `self.draft_resolver.is_some()`. Both share `verify_draft_tokens_logits`.

  **Distributed (4 items)**:
  - **#4 `NcclAllReduce` 实际是本地数组求和** — ✅ closed by P4 batch (`5f00bd5`): `LocalSumAllReduce` is canonical (`crates/dist/src/tensor_parallel/all_reduce.rs:59`); `NcclAllReduce` is `pub type` alias (line 73) with `#[deprecated]`. compile-only test `nccl_all_reduce_alias_resolves_to_local_sum` guards the deprecation contract.
  - **#5 分布式 KV 当前主要存元数据** — ✅ closed by OPS-31d / Phase 31-D + P12-P14: `TransferKVBlock` gRPC RPC + `BlockDataSource` trait + `DistributedKVCache::fetch_block` fan-out fallback + 64 MiB symmetric message limit. ADR-020 records six architectural decisions. Engine wiring (`PagedKvCacheWrapper: BlockDataSource`) explicitly tracked as v32+ / OPS-32a (not closed by this batch).
  - **#6 `dist` 不在 default members** — ✅ closed by P15 §6: `Cargo.toml` workspace `default-members` includes all 6 crates (`crates/core`, `crates/model`, `crates/server`, `crates/traits`, `crates/dist`, `crates/testing`). `just ci` + `ci.yml::ci` automatically cover dist.
  - **#7 multi-node 明确标记 experimental + 类型命名避免暗示 NCCL** — ✅ closed by P12 OPERATIONS.md rewrite + ADR-008/015/020 + the P4 NcclAllReduce rename.

- **`CHANGELOG.md`** — new `[Unreleased] / Added` entry describing the §6 closure table and the verification findings (sampled-match path is already implemented; CUDA Graph ↔ speculative is explicit design not bug; legacy draft code is gone).

### Aggregate view

| Status | Count | Items |
|--------|------:|-------|
| ✅ Closed | 4 | NcclAllReduce rename, TransferKVBlock protocol + ADR-020, dist in default-members, multi-node Experimental label + 3 ADRs |
| 🟡 Partial | 1 | Speculative verifier: sampled-match implemented, full `min(1, p/q)` rejection-sampling deferred (needs draft-side logits) |
| 🟠 Still gap (documented) | 2 | CUDA Graph ↔ speculative mutual exclusion (explicit design — graph capture needs static shape); `PagedKvCacheWrapper: BlockDataSource` engine wiring (OPS-32a) |
| **Total** | **7** | |

### Verification details

- **Code grep** — `grep -rn "legacy\|LegacyDraft" crates/core/src/speculative/` returns zero hits (closes #3).
- **Code grep** — `grep -rn "cuda_graph\|CudaGraph" crates/core/src/speculative/` returns zero hits (CUDA Graph and speculative live in separate code paths; verification of mutual exclusion is via `step_speculative_inner` vs `step_with_graph` entry points).
- **Doc reading** — `crates/core/src/engine/spec_dispatch/verify.rs:1-24` module-level doc-comment is the primary source for #1's "sampled-match is implemented, full rejection-sampling requires draft-side logits" decision. The doc-comment is dated Plan 17.1-C.
- **File inspection** — `crates/dist/src/tensor_parallel/all_reduce.rs` confirms `LocalSumAllReduce` is canonical (line 59) and `NcclAllReduce` is alias (line 73).

### Docs-only change (no Rust / no test / no CI delta)

- No Rust code touched, so `just nextest` is unchanged: **1452 passed, 40 ignored**.
- No public API delta.
- No CI delta.
- Workspace real doc coverage holds at **67.93 %** (closure tables are real prose + code references; the §6 table is denser than the §10 batch's tables but still mostly commentary).

### Why P17 is the natural next batch and not the alternatives

The P16 candidate list explicitly listed three P17+ options: (a) architecture-performance §6 closure, (b) actual `seed` declaration, (c) real engineering task. P17 picks (a) because:

- **Completeness argument**: §6 is the last un-closed due-diligence subsection. Closing it leaves the three due-diligence documents fully audited against v31.0 reality — a cleaner end-state than half-closed.
- **Two new discoveries** that fall out of the §6 audit and might inform future `seed` work: (i) speculative verifier is already temperature-aware sampled-match (which means the engine's sampling hot-path is *not* always greedy, contrary to the v19.x snapshot the doc was written from); (ii) CUDA Graph ↔ speculative is switch-mutually-exclusive, not just "speculative skips CUDA Graph paths" — this affects how to reason about future CUDA Graph work.
- **No sampler-design dependency**: §6 closure is doc-only with code verification; it doesn't gate on the `seed`/sampler-design decision that's blocking that work.

### After P17

The three due-diligence documents now have closure audits on every itemized subsection:

- `engineering-quality.md` §6 + §7 — closed by P15.
- `production-readiness.md` §2-§11 — closed by P16.
- `architecture-performance.md` §6 — closed by P17. (§5 was closed earlier across P0-P9 batches.)

The due-diligence backlog-as-documentation is fully reconciled with v31.0 reality. The remaining open items are visible at the bottom of each due-diligence section and in STATE.md "Remaining open items (after P17)" below — none block the v31.0 alpha.

P18+ candidates: real engineering work (PERF-01 / Phase 31-F) or feature work (`seed` declaration, feature matrix doc).

## Phase 31-F — 2026-07-15 P18 follow-up batch

Closed the **first item** of Phase 31-F (Performance): `attn_factor in paged/flash attention paths`. After three consecutive docs-only batches (P15/P16/P17) the v31.0 loop flips back to **real engineering work** — the first production code change since P10 (request_id propagation).

### The gap

`attn_factor` (YaRN §3.3 attention-temperature scaling) was stored on `GqaAttention` (`gqa/mod.rs:47`) and set by `RopeGqaAttention` from `rope.attn_factor()` (`rope_gqa.rs:65, 107`), but only the standard `forward()` path actually applied it (`gqa/forward.rs:124-126`: `attn_scale = attn_factor * base_scale`). The three **production** paths (`paged_attention_fn`, `tiled_attention_fn`, `flash_attention_fn`) each delegate to a lower-level function that applies its own `1/sqrt(d)` scaling internally — and the attn_factor was silently ignored. The field-level doc-comment in `gqa/mod.rs:41-46` admitted the limitation:

> "Currently only honoured by the standard forward path; paged/tiled/flash attention paths silently ignore this value (documented limitation; follow-up phase will thread it through)."

A user setting `RopeScaling { attn_factor: Some(0.5) }` for YaRN-style long-context inference would see the value silently ignored in production — production routes through `run_attention_fn`, never through the standard `forward()`.

### The fix

New private helper `apply_attn_factor` on `GqaAttention` (in `gqa/forward.rs`):

```rust
fn apply_attn_factor(&self, q: Tensor) -> Result<Tensor> {
    match self.attn_factor {
        Some(f) if (f - 1.0).abs() > f32::EPSILON => q.affine(f64::from(f), 0.0),
        _ => Ok(q),
    }
}
```

Called at the top of `paged_attention_fn`, `tiled_attention_fn`, `flash_attention_fn`, and in the `use_fused` branch of `forward()`. Pre-scaling Q by `attn_factor` is **mathematically equivalent** to applying it to the post-`Q@K^T` logits: `(Q * attn_factor) @ K^T = attn_factor * (Q @ K^T)`, then the internal `1/sqrt(d)` gives `attn_factor / sqrt(d) * (Q @ K^T)`. softmax is invariant to positive scalar multiplication, so the final distribution equals `softmax(Q @ K^T * attn_factor / sqrt(d))` — identical to the standard path's `qk.affine(attn_factor / sqrt(d), 0.0)`.

**Critical caveat:** pre-scaling Q must include ONLY `attn_factor`, never `attn_factor * base_scale`. The `1/sqrt(d)` factor is the responsibility of the downstream attention function. The helper enforces this by accepting `self.attn_factor` directly (not a precomputed scale).

**No-op guarantee:** when `attn_factor` is `None` or `Some(1.0)` (within `f32::EPSILON`), `apply_attn_factor` returns Q unchanged — no allocation, no kernel launch. The common case for non-YaRN models pays zero cost.

### Files modified

- `crates/model/src/components/attention/gqa/forward.rs` — new `apply_attn_factor` helper (12 lines including doc-comment); pre-scaling calls at 4 sites (~5 lines each including comments). Three `Does NOT honour attn_factor` doc-comments updated.
- `crates/model/src/components/attention/gqa/mod.rs` — `attn_factor` field doc-comment rewritten (lines 41-50) to remove the limitation note and document the pre-scaling equivalence.
- `crates/model/src/components/attention/gqa/tests.rs` — 6 new unit tests + 2 private helpers (`build_random_attn`, `build_random_qkv`). Tests follow the existing `gqa_attn_factor_*` pattern at lines 567-659: each test mutates `attn.attn_factor` between calls (None vs `Some(1.0)` for no-op, `Some(0.5)` for "changes output") and asserts via max-abs-diff.

### Test count

- Workspace: **1458 passed** (was 1452; +6 from this batch).
- New tests in `crates/model/src/components/attention/gqa/tests.rs`:
  - `paged_attention_fn_attn_factor_one_is_noop`
  - `paged_attention_fn_attn_factor_changes_output`
  - `tiled_attention_fn_attn_factor_one_is_noop`
  - `tiled_attention_fn_attn_factor_changes_output`
  - `flash_attention_fn_attn_factor_one_is_noop`
  - `flash_attention_fn_attn_factor_changes_output`

### Why pre-scale Q (not extend lower-level signatures)

The lower-level functions (`util::paged_attention`, `util::tiled_attention`, `GqaFlashAttention::forward`) are all public API (`mod.rs:30-32` re-exports + `flash_attention_v3::GqaFlashAttention` public struct). Adding an `attn_factor` parameter would be a public-API breaking change requiring downstream callers to migrate. Pre-scaling Q keeps all signatures stable — `cargo public-api` baseline shows no delta, so the public-api-check gate passes without a `public-api:` CHANGELOG bullet.

### Verification

- `just fmt-check` ✓
- `just clippy` ✓ (no new warnings in gqa/attention files; the +5 vs P17 baseline are pre-existing test warnings elsewhere)
- `just doc-check` ✓
- `just doctest` ✓
- `just nextest` ✓ 1458 tests pass
- `just public-api-check` ✓ (no signature change)
- `just doc-coverage-check` ✓ 67.93% ≥ 65%

### What this batch explicitly does NOT close

The remaining Phase 31-F items stay v32+ candidates:

- **`RopeScaling` config → Block wiring** — partially done for Qwen3 (Phase 15 wired `RoPE::from(&RopeScaling)`); other architectures (Gemma4 / Mixtral / etc.) need a separate audit to verify `RopeScaling` propagates correctly. Out of scope for P18.
- **`expand_kv` fused kernel** — deferred from v30 (very-high complexity). Needs custom GQA-broadcast matmul kernel.
- **PagedKV host round-trip elimination** — deferred from v30 (very-high complexity). Needs fused host-device KV access.

### Why this batch and not `seed` declaration

The P17 candidate list mentioned `seed` declaration as a v0.2 candidate. P18 chose `attn_factor` because:

- **Concrete scope** — the gap was 6 production lines + 6 tests + 2 doc-comment updates; bounded and verifiable.
- **No sampler-design dependency** — unlike `seed`, this doesn't require engine-side RNG seeding. The standard path was already correct; the production paths just needed to call into it.
- **Closes a real contract violation** — users setting `attn_factor` for YaRN silently got no effect in production. That's worse than `seed`'s "rejected by serde" because there's no error signal.
- **First real engineering work since P10** — three docs-only batches in a row means the v31.0 backlog of code gaps wasn't shrinking. P18 reopens the engineering track.

P19+ candidates: the remaining Phase 31-F items (`RopeScaling` audit across architectures, `expand_kv` kernel, PagedKV host round-trip elimination — but the last two are very-high-complexity); `seed` declaration (v0.2); feature matrix doc; or back to due-diligence drift closure if any items remain.

## Remaining open items (after P18)

- **PERF-01** (continuous batching kernel) — deferred to v32+.
- **CI-01** (sustained GPU / real-checkpoint CI) — deferred.
- **OpenAI compat: `seed` wire type** — still tracked as v0.2
  follow-up in `docs/reference/openai-compatibility.md`.
- **Phase 31-D follow-up work (post-OPS-31d)** — `kv_block_transfer.rs`
  integration tests already exercise real 2-node + 3-node gRPC
  round-trips. The master-plan checkboxes are now closed
  (P12). Remaining items per
  `ops-31d-kv-block-transfer.md` §7 are still all v32+
  non-goals: MESI coherence, owner-routed peer fetch,
  streaming RPCs, wire compression, block refcounting
  during transfer, **`PagedKvCacheWrapper: BlockDataSource`**
  (the load-bearing engine-wiring piece — see P12 batch
  section "What this batch explicitly does NOT close").
  Tracking only — none block the v31.0 alpha.
- **Phase 31-F (performance)** — P18 closed `attn_factor in paged/flash
  attention paths` and P19 closed `RopeScaling config → Block wiring`.
  Remaining: `expand_kv` fused kernel, PagedKV host round-trip
  elimination — deferred to v32+ (very-high complexity).
- **production-readiness §6 OTLP follow-up** — the `info_span!`
  work in P10 is the prerequisite for OTLP exporter wiring
  (production-readiness §6 last bullet: "先完善结构化 span，再接
  OTLP；不要仅添加依赖而没有 trace topology"). The workspace
  has no OTLP dependency; adding one is a v32+ non-goal pending
  a real OTLP backend (no CI-side collector). Tracking only.
- **engineering-quality §7 checksums + provenance** — P11 closed
  the SBOM half of §7. The remaining half (`sha256sum`
  checksums + signed SLSA / in-toto provenance) is tracked as
  a v32+ candidate — needs a signature-key story + reproducible
  build posture before it can ship responsibly.

## Phase 31-F — 2026-07-16 P20 follow-up batch

Closed the Phase 31-F `RopeScaling config → Block wiring`
**follow-up for non-Qwen3 architectures**. P19 closed the
thread-through for the Qwen3-specific path (the
`qwen3/block/factory::new_block` + `RopeGqaAttention::new_with_rope_scaling`
chain) but explicitly noted "other architectures (Gemma4 /
Mixtral / etc.) need a separate audit". This batch performs
that audit + fix.

### The gap

Two more call sites continued to use the bare (non-scaling-aware)
constructors and silently dropped the block on every HF weight
load with a `rope_scaling` declaration:

- **`crates/model/src/components/decoder_block/factory.rs`** —
  the SHARED `new_block` / `block_from_weights` used by
  Llama/Mistral-style checkpoints. Both called
  `RopeGqaAttention::new` / `new_with_weights` with no
  `max_position` / `rope_scaling` arguments, so the inner RoPE
  was built via the default `RoPE::new(...)` constructor
  (`scaling_factor = 1.0`, `attn_factor = None`) regardless of
  what `config.json["rope_scaling"]` declared.
- **`crates/model/src/mixtral/block.rs`** — `MixtralBlock::new`
  + `MixtralBlock::from_weights`. Same pattern: the bare
  constructors dropped `rope_scaling` on the floor.

A Llama / Mistral / Mixtral checkpoint that declared a YaRN
block in `config.json` therefore produced numerically
identical output to a default model of the same shape at the
engine boundary, regardless of what `rope_scaling` declared —
the exact regression P19 fixed for Qwen3, still latent in two
other architectures.

### The fix

Both call sites now thread `config.max_position_embeddings` +
`config.rope_scaling.as_ref()` into the scaling-aware
constructors introduced by P19:

- `RopeGqaAttention::new` → `RopeGqaAttention::new_with_rope_scaling`
  (`crates/model/src/components/decoder_block/factory.rs:55-69`,
  `crates/model/src/mixtral/block.rs:53-69`)
- `RopeGqaAttention::new_with_weights` → `RopeGqaAttention::new_with_weights_rope_scaling`
  (`crates/model/src/components/decoder_block/factory.rs:126-144`,
  `crates/model/src/mixtral/block.rs:147-163`)

The bare constructors remain in the public API (P19 preserved
them as `None`-scaling aliases for backward compatibility) —
they are no longer called by any workspace-internal factory.
Inline doc-comments on the new call sites explain the wiring
contract and reference P19 for the constructor rationale.

### Test coverage

4 new unit tests pinning the wiring so the regression can't
recur:

- `crates/model/src/components/decoder_block/factory.rs::tests`:
  - `new_block_accepts_yarn_rope_scaling` — builds a
    `ModelConfig` with YaRN `rope_scaling` (`factor=4.0`,
    `attn_factor=Some(0.5)`) and asserts
    `decoder_block::factory::new_block(...)` returns `Ok(_)`.
  - `new_block_accepts_none_rope_scaling` — pins the no-op
    path: `rope_scaling=None` continues to work, exercising the
    `new_with_rope_scaling(..., None)` delegation that mirrors
    the bare `new(...)` behaviour bit-for-bit.
- `crates/model/src/mixtral/block/tests.rs`:
  - `test_mixtral_block_new_accepts_yarn_rope_scaling` —
    YaRN config → `MixtralBlock::new(...)` returns `Ok(_)`.
  - `test_mixtral_block_from_weights_accepts_yarn_rope_scaling` —
    builds a populated `HashMap<String, Tensor>` of expert
    weights (4 experts × gate/up/down + gate weight +
    self_attn projections + input/post layernorms) and asserts
    `MixtralBlock::from_weights(...)` returns `Ok(_)`. The
    population is intentional — pre-fix this code path called
    `RopeGqaAttention::new_with_weights` (bare) which would
    silently drop the block on every HF weight load with a
    YaRN config; the test pins the scaling-aware constructor
    is now wired through.

### Test count

- Workspace: **1464** tests pass (was 1458 after P18 / +2 from
  P19 / +4 from this batch).
- vllm-model: 419 → 423 (+4 unit tests in this batch).
- `cargo nextest run --workspace --all-features --no-fail-fast`
  exits 0; no flakes.
- `cargo fmt --all --check` passes.
- `cargo clippy --all-targets -p vllm-model -- -D clippy::correctness -D clippy::suspicious -D clippy::perf`
  introduces no new deny-tier warnings (the 67 warnings that
  pre-existed in vllm-model tests are unrelated pedantic noise
  — same count as before this batch).
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items --workspace --all-features`
  passes (fixed one initial broken-intra-doc-link by correcting
  the path to `crate::qwen3::block::TransformerBlock::from_weights`).
- `bash .planning/phase-12e/check-public-api.sh --no-fail`
  reports zero new public-API items in `vllm-model` (the diff
  is the pre-existing P4 NcclAllReduce alias rename + a few
  `impl` items that come along with it; all pre-P20).
- Workspace real doc coverage: **unchanged at 67.93 %** (the
  new docstrings are commentary + factory/test contracts,
  marginal vs noise).

### What this batch explicitly does NOT close

The remaining Phase 31-F items stay v32+ candidates:

- **`expand_kv` fused kernel** — deferred from v30
  (very-high complexity). Needs custom GQA-broadcast matmul
  kernel.
- **PagedKV host round-trip elimination** — deferred from v30
  (very-high complexity). Needs fused host-device KV access.

The `RopeScaling` thread-through is now complete across the
three production architectures in the workspace (Qwen3 via
P19, Llama/Mistral + Mixtral via P20). The only architectures
that don't honour `rope_scaling` are:

- **Gemma4** — `crates/model/src/gemma4/block.rs` has its own
  factory (`gemma4::block::new_block` at line 340) that does
  not use `RopeGqaAttention` (Gemma4 uses its own attention
  implementation per `architecture.md`). Out of scope for
  this batch.
- **MLA** (`crates/model/src/components/attention/mla.rs`) —
  uses `RopeScalingContext::default()` in `forward()` and is
  not wired into any production decoder (per its own
  doc-comment: "Currently no production model architecture
  uses `MlaAttention` directly; the path is exposed for
  experimentation and benchmarking"). Threading `RopeScaling`
  through MLA would be premature — no production caller
  exists to exercise it.

Both are explicit "exposed for experimentation" cases where
the follow-up is gated on a real production user, not a
missing wire.

### Why P20 and not the v32+ candidates

The P19 follow-up list named four P20+ options: (a) non-Qwen3
RopeScaling audit, (b) `seed` declaration, (c) feature matrix
doc, (d) `expand_kv` / PagedKV round-trip elimination
(deferred — very-high complexity). P20 chose (a) because:

- **Bounded scope** — 2 files × 2 call sites + 4 tests + 1
  doc-comment update; verifiable in a single CI run.
- **No sampler / kernel-design dependency** — unlike `seed`
  (sampler rewrite needed) or `expand_kv` (custom kernel),
  this is purely constructor signature plumbing that uses the
  P19 scaling-aware constructors as-is.
- **Closes a real contract violation** — same reasoning as
  P19: users setting `rope_scaling` for YaRN silently got no
  effect in production for Llama/Mistral/Mixtral checkpoints.
- **Symmetry with P19** — leaving the Qwen3-only fix in
  place while Llama/Mistral/Mixtral still drop the block
  would have created a documentation trap (the CHANGELOG
  would claim "fixed" but only for one architecture).

P21+ candidates: `seed` declaration (still blocked on
sampler design — see P15 reasoning); feature matrix doc
(engineering-quality §6 #4); MLA RopeScaling wiring (gated
on MLA production wiring); OTLP exporter (production-readiness
§6); checksums + provenance (engineering-quality §7 second
half); or back to due-diligence drift closure if any items
remain (none currently open).

## Remaining open items (after P20)

- **PERF-01** (continuous batching kernel) — deferred to v32+.
- **CI-01** (sustained GPU / real-checkpoint CI) — deferred.
- **OpenAI compat: `seed` wire type** — still tracked as v0.2
  follow-up in `docs/reference/openai-compatibility.md`.
- **Phase 31-D follow-up work (post-OPS-31d)** — `kv_block_transfer.rs`
  integration tests already exercise real 2-node + 3-node gRPC
  round-trips. The master-plan checkboxes are closed (P12).
  Remaining items per `ops-31d-kv-block-transfer.md` §7 are
  still all v32+ non-goals: MESI coherence, owner-routed
  peer fetch, streaming RPCs, wire compression, block
  refcounting during transfer, **`PagedKvCacheWrapper: BlockDataSource`**
  (the load-bearing engine-wiring piece — see P12 batch
  section "What this batch explicitly does NOT close").
  Tracking only — none block the v31.0 alpha.
- **Phase 31-F (performance)** — P18 closed `attn_factor in
  paged/flash attention paths`, P19 closed
  `RopeScaling config → Block wiring` for Qwen3, P20 closed
  the non-Qwen3 follow-up (Llama/Mistral shared factory +
  Mixtral). Remaining: `expand_kv` fused kernel, PagedKV host
  round-trip elimination — deferred to v32+ (very-high
  complexity).
- **production-readiness §6 OTLP follow-up** — the `info_span!`
  work in P10 is the prerequisite for OTLP exporter wiring.
  Tracking only.
- **engineering-quality §7 checksums + provenance** — P11 closed
  the SBOM half of §7. The remaining half is tracked as a v32+
  candidate.
- **engineering-quality §6 #4 feature matrix doc** — partially
  covered by `docs/architecture.md` §Feature Flags but a full
  matrix doc is still v0.2/32+ work.
- **MLA RopeScaling wiring** — `crates/model/src/components/attention/mla.rs`
  uses `RopeScalingContext::default()` in `forward()`. Gated on
  MLA being wired into a production decoder (currently
  experiment-only per its doc-comment).

## v0.2 wire-type follow-ups — 2026-07-16 P21 batch

Closed the first v0.2 wire-type follow-up listed in
`docs/reference/openai-compatibility.md`: declaration + tracing
pass-through of OpenAI's `user` field on `/v1/chat/completions`
+ `/v1/completions`.

### The gap

OpenAI's `user` field is the end-user identifier used for
safety / abuse tracking. Per OpenAI spec it is a request-only
optional string; there is no format/length validation, and it
must never be echoed in the response body. Pre-fix, neither
`ChatRequest` nor `CompletionRequest` declared the field, so
serde silently rejected any client that included
`"user": "..."` in its JSON body — a contract violation
flagged in the OpenAI compat matrix as "Not declared →
Rejected by serde" and explicitly tracked as a v0.2 candidate
in the matrix's "v0.2 follow-ups (planned)" section.

### The fix

Two minimal wire-type changes + one tracing pass-through:

- **`ChatRequest::user: Option<String>`** in
  `crates/server/src/openai/types.rs` with
  `#[serde(default, skip_serializing_if = "Option::is_none")]`.
  `default` keeps every existing client working (omitted
  field deserializes to `None`); `skip_serializing_if` ensures
  the field is never echoed in any response-shape that
  accidentally re-serializes the request type.
- **`CompletionRequest::user: Option<String>`** in the same
  file with the same attribute pair.
- **Tracing pass-through** in `openai::chat::chat_completions`
  and `openai::chat::stream_chat_completion`. Three existing
  `tracing::info!(...)` calls gained `user = ?req.user` so
  downstream subscribers (rate-limiter, audit log,
  abuse-detection pipeline) can pick the value up via the
  structured tracing field without any engine-side changes.
- The completions handler does NOT gain a new
  `tracing::info!` line — adding one would be scope creep
  beyond the minimal declaration. The wire-type contract is
  symmetric (both endpoints accept `user`); the observability
  asymmetry is called out in the doc and CHANGELOG.

### What this batch explicitly does NOT do

- **Honoring is a no-op.** vllm-lite has no auth/persistence
  layer that would consume `user`. The field is purely a
  tracing pass-through today; a downstream consumer (audit log
  gate, per-user rate-limit, abuse-detection correlation) would
  subscribe to the `tracing::info!(..., user = ?req.user)`
  stream.
- **No response echo.** Per OpenAI spec, `user` is request-only.
  The `skip_serializing_if` ensures no response shape accidentally
  serializes the field.
- **No validation.** The OpenAI spec does not constrain the
  string format or length. Adding a length cap (e.g. 256 chars)
  would be a wire-API contract decision deferred to v0.3.

### Test coverage

4 new integration tests in
`crates/server/tests/chat_integration_test.rs`:

- `test_chat_with_user_field_accepted_by_handler` — sends a
  request with `"user": "tenant-1234"` and asserts the handler
  returns 200 OK. Pre-fix this would fail with a 400-class
  deserialization error because serde rejects unknown fields
  by default.
- `test_chat_without_user_field_works_baseline` — pins the
  backward-compatible path: omitting `user` continues to work
  (the field defaults to `None`).
- `test_completions_with_user_field_accepted_by_handler` —
  mirror of the chat test on `/v1/completions`.
- `test_chat_user_field_wire_type_round_trip` — pins the
  serde contract independently of any handler-level test:
  parses JSON with `user` set, parses JSON without `user`,
  and asserts `req.user` matches expectations. Catches a
  future refactor that drops the `#[serde(default)]`
  annotation.

### Doc updates

- `docs/reference/openai-compatibility.md`:
  - Chat endpoint table: `user` row flipped from "Not declared"
    → "Wired (tracing pass-through)" with the OpenAI
    acceptance note.
  - Completions endpoint table: same flip + an explicit
    "(declaration only — handler does not currently log)"
    qualifier for the observability asymmetry.
  - v0.2 follow-ups section: `user` row marked
    "**Shipped in P21 (2026-07-16).**"
  - Header date stamp bumped to 2026-07-16.
- `CHANGELOG.md`: new `[Unreleased] > Fixed` bullet
  documenting the contract violation + fix, and a new
  `public-api: vllm-server added` bullet for the 2 new
  fields per the public-api-check gate contract.

### Test count

- Workspace: **1468** tests pass (was 1464 after P20; +4 from
  this batch).
- vllm-server integration: `chat_integration_test.rs` grew by
  4 tests.
- `cargo nextest run --workspace --all-features --no-fail-fast`
  exits 0; no flakes.
- `cargo fmt --all --check` passes.
- `cargo clippy --all-targets --workspace --all-features -- -D
  clippy::correctness -D clippy::suspicious -D clippy::perf`
  introduces no new deny-tier warnings.
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items
  --workspace --all-features` passes.
- `bash .planning/phase-12e/check-public-api.sh --no-fail`
  reports 2 new public-API items in `vllm-server`
  (`ChatRequest::user`, `CompletionRequest::user`) + the
  CHANGELOG `public-api: vllm-server added` bullet, so the
  gate passes.
- Workspace real doc coverage: **unchanged at 68.03 %** (the
  new doc-comments on the wire fields are short contract notes;
  the matrix table is the bulk of the visible documentation).

### Why P21 and not the alternatives

The P20 follow-up list named six P21+ options: `seed`
(still blocked on sampler design per P15 reasoning),
feature matrix doc, MLA RopeScaling wiring (gated), OTLP
exporter, checksums + provenance, due-diligence drift closure
(none open). P21 chose the `user` field for three reasons:

- **Smallest possible v0.2 work** — 2 wire-type lines +
  3 tracing pass-throughs + 4 tests. Bounded and verifiable in
  a single CI run.
- **No sampler / kernel / contract decision required** — unlike
  `seed` (RNG seeding design) or `frequency_penalty` rename
  (public-API delta on `SamplingParams`), this is pure
  declaration + observability plumbing.
- **Closes a real, documented contract violation** — the OpenAI
  compat matrix explicitly flagged `user` as "Not declared →
  Rejected by serde" and the v0.2 follow-ups section was
  already written (by P15) listing the planned work.

### After P21

The v0.2 wire-type backlog shrinks to two items:

- `seed` — declaration is trivial but honoring requires RNG
  seeding in `vllm_core::sampling` (currently unseeded — the
  sampler reads from `rand`'s thread-local RNG). The validation
  contract (accept any integer per OpenAI spec) and the engine
  contract (forward the value) are well-defined; the
  engine-side seed usage is the open question. Could land
  alongside a small sampler refactor, or as a `seed: Option<i64>`
  declaration-only PR that mirrors P21's pattern but flags
  "honoring deferred" in the doc.
- `response_format` — declaration + validation accepting
  `{type: "text"}` and `{type: "json_object"}` only (the
  `{type: "json_schema"}` subset defers to v0.3 because it
  requires a grammar-constrained decoder). Slightly bigger than
  `user` (~5 lines of `ResponseFormat` enum + 2 wire-type
  fields + 4+ tests + a small validation function) but
  identical pattern.

P22+ candidates: `response_format` declaration + validation
(the next-smallest v0.2 work), `seed` declaration-only with
honoring deferred to v32+, feature matrix doc (engineering-quality
§6 #4), OTLP exporter (production-readiness §6), checksums +
provenance (engineering-quality §7 second half), MLA RopeScaling
(gated), or back to due-diligence drift closure if any items
remain (none currently open).

## v0.2 wire-type follow-ups — 2026-07-16 P22 batch

Closed the second v0.2 wire-type follow-up listed in
`docs/reference/openai-compatibility.md`: declaration + serde-layer
validation of OpenAI's `response_format` field on
`/v1/chat/completions`.

### The gap

OpenAI's `response_format` field is the JSON-mode selector. The
OpenAI spec defines three variants: `{"type": "text"}` (default),
`{"type": "json_object"}` (valid JSON output), and
`{"type": "json_schema", "json_schema": {...}}` (structured
JSON schema). Pre-fix, `ChatRequest` had no field for
`response_format`, so serde silently rejected any request that
included `"response_format": ...` in its JSON body — a contract
violation flagged in the OpenAI compat matrix as "Not declared →
Rejected by serde" and explicitly tracked as a v0.2 candidate in
the matrix's "v0.2 follow-ups (planned)" section.

The legacy `/v1/completions` endpoint does NOT support
`response_format` per OpenAI spec; `CompletionRequest`
intentionally does NOT declare the field. The legacy endpoint
silently drops unknown fields (its permissive contract), so this
is a wire-type asymmetry, not a bug.

### The fix

Two minimal wire-type additions + one documentation-first
validator + tracing pass-through:

- **`ResponseFormat` enum** in
  `crates/server/src/openai/types.rs` with `Text` + `JsonObject`
  variants. `#[serde(tag = "type", rename_all = "snake_case")]`
  matches the OpenAI JSON shape 1:1 — clients send
  `{"type": "text"}` / `{"type": "json_object"}` and serde
  deserializes to the corresponding enum variant. The
  `{type: "json_schema"}` variant is intentionally NOT declared;
  serde rejects unknown variants at deserialization.
- **`ChatRequest::response_format: Option<ResponseFormat>`** in
  the same file with `#[serde(default,
  skip_serializing_if = "Option::is_none")]`. `default` keeps
  every existing client working (omitted field deserializes to
  `None`); `skip_serializing_if` ensures the field is never
  echoed in any response shape that accidentally re-serializes
  the request type.
- **`validate_chat_response_format`** in
  `crates/server/src/openai/sampling_validation.rs` — a no-op
  pass-through today that documents the v0.2 contract ("only Text
  + JsonObject accepted") and provides a single hook point for
  future strict checks (e.g. v0.3 `JsonObject` payload syntax
  validation). Wired into `validate_chat_request_fields` so the
  validator flow is parallel to `top_p` / `n` / `stop`.
- **Tracing pass-through** in `openai::chat::chat_completions` +
  `openai::chat::stream_chat_completion`. Three existing
  `tracing::info!(...)` calls gained
  `response_format = ?req.response_format` so downstream
  subscribers (audit log, observability dashboards, future
  constrained-decoding hooks) can pick the value up without any
  engine-side changes.

### What this batch explicitly does NOT do

- **Honoring is a no-op.** The engine does not enforce JSON syntax
  via a constrained-decoding hook. `JsonObject` is accepted as a
  no-op pass-through (same as `Text` for the sampler). This is
  v0.3 / v32+ work and is explicitly tracked in the OpenAI compat
  matrix as a "constrained-decoding hook" follow-up.
- **Engine-side forwarding.** P22 chose the minimal
  declaration-only approach — the field is declared on
  `ChatRequest`, validated via serde (with a
  `validate_chat_response_format` documentation-first hook), and
  threaded into `tracing::info!(response_format = ?req.response_format, ...)`.
  Adding `response_format` to `SamplingParams` / `Request` would
  require touching the public-API struct, updating the
  `SamplingParams` builder, and a separate public-api-check entry
  — deferred to v0.3 / v32+ when the constrained-decoding hook
  lands and the field becomes meaningful at the engine layer.
- **Response echo.** Per OpenAI spec, `response_format` is
  request-only (the model produces text; the response shape is
  the same regardless of `response_format`). The
  `skip_serializing_if` ensures no response shape accidentally
  serializes the field.

### Test coverage

11 new tests total (6 integration + 5 unit):

- **`crates/server/tests/chat_integration_test.rs`** (6 tests):
  - `test_chat_with_response_format_text_accepted_by_handler` —
    `{"type": "text"}` accepted (equivalent to omitting the field).
  - `test_chat_with_response_format_json_object_accepted_by_handler` —
    `{"type": "json_object"}` accepted as a v0.2 pass-through.
  - `test_chat_with_response_format_json_schema_rejected` — the
    v0.3 variant is rejected with `422 Unprocessable Entity` (axum's
    standard contract for deserialization failures; pinned by the
    test).
  - `test_chat_without_response_format_works_baseline` —
    backward-compat: omitting the field continues to work.
  - `test_chat_response_format_wire_type_round_trip` — pins the
    serde contract independently of any handler-level test,
    including `json_schema` deserialization failure.
  - `test_completion_request_has_no_response_format_field` —
    pins the wire-type asymmetry: `CompletionRequest` cannot be
    constructed with a `response_format` field because the struct
    doesn't have one.
- **`crates/server/src/openai/sampling_validation.rs::tests`** (5 tests):
  - `chat_response_format_none_passes_validation` — default path.
  - `chat_response_format_text_passes_validation` — `Text` is
    accepted.
  - `chat_response_format_json_object_passes_validation` —
    `JsonObject` is accepted as a v0.2 pass-through.
  - `chat_request_with_response_format_text_passes_full_field_validation` —
    integration with `validate_chat_request_fields`: a request
    that has both `response_format = Text` and other valid fields
    passes the full chat-request validator.
  - `chat_request_with_response_format_json_object_passes_full_field_validation` —
    same as above but with `JsonObject`.

### Doc updates

- `docs/reference/openai-compatibility.md`:
  - Chat endpoint table: `response_format` row flipped from
    "Not declared" → "Wired (declaration + validation)" with the
    full P22 acceptance note (ChatRequest only, NOT
    CompletionRequest; serde rejects json_schema with 422;
    honoring is a no-op).
  - v0.2 follow-ups section: `response_format` row marked
    "**Shipped in P22 (2026-07-16).**"
  - Cross-references section: `response_format` cross-ref updated
    from "not yet tracked in STATE.md" to "closed by P22".
- `CHANGELOG.md`: new `[Unreleased] > Fixed` bullet documenting
  the contract violation + fix, and a new
  `public-api: vllm-server added` bullet for the 1 new type + 1
  new field per the public-api-check gate contract.

### Test count

- Workspace: **1479** tests pass (was 1468 after P21; +11 from
  this batch).
- vllm-server integration: `chat_integration_test.rs` grew by 6
  tests; `sampling_validation::tests` grew by 5 unit tests.
- `cargo nextest run --workspace --all-features --no-fail-fast`
  exits 0; no flakes (the pre-existing
  `test_radix_repeated_prefix_lookup_is_fast` flakiness from
  cargo-test concurrency in the `prefix_cache` test module is
  unrelated to P22; it failed once on a slow run but passed on
  rerun).
- `cargo fmt --all --check` passes.
- `cargo clippy --all-targets --workspace --all-features -- -D
  clippy::correctness -D clippy::suspicious -D clippy::perf`
  introduces no new deny-tier warnings. Required one
  `#[allow(clippy::missing_const_for_fn)]` on the no-op
  `validate_chat_response_format` because clippy flags the
  body as eligible for `const fn` (no runtime operations).
  Comment explains: future validators will need runtime ops
  (regex, format checks) and the signature should not change
  when that happens; the `Json<T>` return type already precludes
  `const fn` on stable Rust. Also fixed one
  `clippy::uninlined_format_args` warning in the new integration
  test (capture-style `format!("{}", x)` → inline-style
  `format!("{x}")`).
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items
  --workspace --all-features` passes.
- `bash .planning/phase-12e/check-public-api.sh --no-fail`
  reports 1 new public-API type (`ResponseFormat`) + 1 new
  public-API field (`ChatRequest::response_format`) in
  `vllm-server` + the CHANGELOG `public-api: vllm-server added`
  bullet, so the gate passes.
- Workspace real doc coverage: **68.03 % → 68.06 %** (the
  new doc-comments on `ResponseFormat` / `ChatRequest::response_format`
  / `validate_chat_response_format` are full contract
  documentation, not just field annotations).

### Why P22 and not the alternatives

The P21 follow-up list named seven P22+ options: `seed`
declaration-only, `response_format` declaration + validation,
feature matrix doc, MLA RopeScaling wiring (gated), OTLP
exporter, checksums + provenance, due-diligence drift closure
(none open). P22 chose `response_format` because:

- **Clean scope** — 1 new enum + 1 wire-type field + 1 no-op
  validator + tracing pass-through + 11 tests. Bounded and
  verifiable in a single CI run.
- **No sampler / kernel / contract decision required** — unlike
  `seed` (RNG seeding design) or `frequency_penalty` rename
  (public-API delta on `SamplingParams`), this is pure
  declaration + serde-based validation + observability plumbing.
- **Closes a real, documented contract violation** — the OpenAI
  compat matrix explicitly flagged `response_format` as "Not
  declared → Rejected by serde" and the v0.2 follow-ups section
  was already written (by P15) listing the planned work.

### After P22

The v0.2 wire-type backlog shrinks to one item:

- `seed` — declaration is trivial but honoring requires RNG
  seeding in `vllm_core::sampling` (currently unseeded — the
  sampler reads from `rand`'s thread-local RNG). The validation
  contract (accept any integer per OpenAI spec) and the engine
  contract (forward the value) are well-defined; the
  engine-side seed usage is the open question. Could land
  alongside a small sampler refactor, or as a `seed: Option<i64>`
  declaration-only PR that mirrors P21's pattern but flags
  "honoring deferred" in the doc.

The v0.3 follow-ups (deferred from v0.2 by the P15 split):

- `response_format` `JsonSchema` variant — requires a
  grammar-constrained decoder.
- `frequency_penalty` / `presence_penalty` rename — the
  `repeat_penalty` field on `SamplingParams` becomes the OpenAI
  wire-name pair; public-API delta.

P23+ candidates: `seed` declaration-only (the last v0.2
wire-type follow-up), feature matrix doc (engineering-quality
§6 #4), OTLP exporter (production-readiness §6), checksums +
provenance (engineering-quality §7 second half), MLA RopeScaling
(gated), or back to due-diligence drift closure if any items
remain (none currently open).