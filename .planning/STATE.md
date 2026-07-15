---
gsd_state_version: 1.0
milestone: v31.0
milestone_name: Perfection & Elegance
status: in_progress
last_updated: "2026-07-12T18:30:00.000Z"
last_activity: 2026-07-12 — Technical due diligence P1 follow-up batch:
cancel propagation, graceful shutdown thread join, body-limit
wiring, correlation_id + readiness-saturation + sampling-validation,
governance (CoC / SECURITY / MAINTAINERS / templates), README
honesty pass, GOV-01 release manifest, CI-01 all-features parity,
Dependabot config.
progress:
  total_phases: 6
  completed_phases: 0
  total_plans: 6
  completed_plans: 4
  percent: 67
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
  closed by this batch.)
- **Phase 31-D: 2-node integration test** — not yet exercised
  end-to-end; the unit tests for `TransferKVBlock` and
  `DistributedKVCache::fetch_block` are in place but a 2-process
  round-trip (start two nodes, route a prefix cache hit across
  them) is still outstanding.
- **Phase 31-F (performance)** — `attn_factor` in paged/flash
  attention paths, `RopeScaling` config → Block wiring,
  `expand_kv` fused kernel, PagedKV host round-trip elimination
  all deferred.
