# Phase 43 — OTLP Metrics + Traces Exporter (production-readiness §6 follow-up)

**Date:** 2026-07-22
**Scope:** `vllm-core` (new `opentelemetry` feature + `OtlpExporter` + tracing-init helper), `vllm-server` (bootstrap wiring + YAML config + shutdown flush), `docs/OPERATIONS.md` (OTLP setup examples)
**Status:** Design
**Phase goal (long-term):** Close one of the five remaining production-readiness code follow-ups tracked in `.planning/STATE.md` P16 closure table — **"OTLP"** — by shipping a push-based metrics + traces exporter that streams engine telemetry to any OpenTelemetry-compatible collector (Jaeger / Tempo / Datadog / Honeycomb / etc.) behind a single feature flag.

---

## 1. Why this phase

The v31.0 multi-node story (P40-P42, OPS-32a + OPS-31d receiver-side closure) just landed end-to-end. With distributed KV replication now functional, operators running vllm-lite in production have a critical observability gap:

- **Today** the only way to read engine metrics is `curl http://pod:8000/metrics` — a Prometheus text-format snapshot. Works for `kube-prometheus` scrapers, but useless for organizations running OTel-native stacks (Tempo / Jaeger / Datadog / Honeycomb / New Relic), which is the majority of new SRE shops per the 2026 CNCF survey referenced in `docs/technical-due-diligence/production-readiness.md` §6.
- **Today** the only way to read trace spans is the structured `tracing` JSON log stream, which requires log aggregation (Loki / ELK / Vector) before spans can be visualized. There's no way to get a distributed trace across `HTTP request → axum middleware → EngineMessage → scheduler.add_request → KV allocate → prefix-cache lookup` — the JSON logs lose the span hierarchy.

The P16 production-readiness closure table explicitly lists **OTLP** as one of 5 v32+ code follow-ups: "5 code follow-ups (TLS 主路径接线 / OTLP / per-tenant quota / readiness 模型加载信号 / feature matrix doc)". This phase closes the OTLP item.

### 1.1 ROI

The team's own ranking (per `STATE.md` last_activity rationale) put OTLP at **MEDIUM-HIGH ROI**: the standard observability stack composes naturally with the existing tonic-based stack, the schema mapping is bounded (~15 metrics + a handful of `tracing` instrument points), and the resulting OTLP infrastructure becomes the foundation for future v0.2/v0.3 work (per-tenant quota, request_id propagation, distributed traces across nodes — all of which need OTel as the data plane).

### 1.2 What this phase explicitly does NOT close

- **Logs via OTLP** — the existing `tracing-subscriber` + JSON output is already log-aggregator friendly (Filebeat / Vector / Loki can ingest it). Adding OTel log export would duplicate this; defer to v32+.
- **HTTP/protobuf OTLP transport** — v43 ships `grpc-tonic` (composes with the existing `tonic` stack). HTTP/proto follows when a customer asks for it.
- **Per-tenant quota** — separate production-readiness §6 item, requires tenant-id plumbing first.
- **TLS to the collector** — operator config concern, not a runtime change. Documented in OPERATIONS.md as "use a sidecar / mTLS proxy in production".

---

## 2. Goals

1. **G1 — `OtlpExporter` push-based background task.** New `pub struct OtlpExporter` in `crates/core/src/metrics/exporter/otlp.rs` (gated `#[cfg(feature = "opentelemetry")]`). Implements `MetricsExporter` trait for parity with `PrometheusExporter`, but its primary mode is **push**: a `tokio::spawn`-ed background task calls `collector.snapshot()` every `metrics_export_interval_secs` and ships the result via `opentelemetry-otlp` to the configured collector endpoint.

2. **G2 — Tracing → OTel bridge.** New `init_tracing_with_otlp(env_filter, otlp_config) -> OtlpGuard` helper in `crates/core/src/tracing_init.rs` (new file) that:
   - Sets up the existing `tracing-subscriber` with the requested `env_filter`.
   - Adds a `tracing_opentelemetry::layer().with_tracer(otlp_tracer)` bridge so every `tracing::info_span!(...)` becomes an OTel span.
   - Returns an `OtlpGuard` whose `Drop` impl flushes pending spans (graceful shutdown).
   - Sampling ratio configurable via `OtlpConfig::trace_sampling_ratio`.

3. **G3 — YAML config + `AppConfig` integration.** New `app_config.observability.otlp` section in `ServerConfig` (deserialized via `serde` + `#[serde(default)]` for backward compat). Fields: `enabled` (bool, default `false`), `endpoint` (String, default `"http://localhost:4317"`), `service_name` (String, default `"vllm-lite"`), `service_version` (String, default workspace version from release manifest), `metrics_export_interval_secs` (u64, default `30`), `trace_sampling_ratio` (f64 in `[0.0, 1.0]`, default `1.0`), `protocol` (enum `Grpc | Http`, default `Grpc`).

4. **G4 — Server bootstrap wiring.** `crates/server/src/bootstrap/observability.rs` (new file) provides:
   - `spawn_otlp_exporter(state, config) -> Result<OtlpHandle, OtlpError>` — constructs the meter + tracer, returns a handle holding the spawned task's `JoinHandle` + the SDK `MeterProvider` (for graceful shutdown flush).
   - `init_tracing_with_otlp(env_filter, config) -> OtlpGuard` — thin re-export from `vllm_core` so the bootstrap can wire it without touching `vllm_core` directly.
   - `main.rs` calls both when `app_config.observability.otlp.enabled` is `true`; both are no-ops when `false`.

5. **G5 — Failure-tolerant runtime.** Collector unreachable → log `warn!` once per minute (rate-limited) + continue. Bootstrap-time failure (bad config) → log `warn!` + continue without OTLP. NEVER crash the engine or the HTTP server because OTLP misbehaves.

6. **G6 — Observability parity with existing metrics.** The OTLP exporter reports the **same** 15 metrics as `PrometheusExporter` (cuda_graph_hits_total, packing_efficiency, draft_*_total, throughput_speedup_ratio, etc.), so operators who switch from `/metrics` scraping to OTLP push see no regression. New metrics can be added later without touching this PR.

7. **G7 — Feature-gated, default-off.** New `opentelemetry` feature on `vllm-core` (pulls `opentelemetry`, `opentelemetry_sdk`, `opentelemetry-otlp`, `tracing-opentelemetry`, `opentelemetry-semantic-conventions`). `vllm-server` enables it via `vllm-core/opentelemetry`. Default features unchanged. `cargo build` without `--features opentelemetry` compiles zero OTLP code.

---

## 3. Non-goals (P43 explicitly defers)

- **OTLP logs export** — current `tracing-subscriber` JSON output is log-aggregator friendly. Defer to v32+ if a customer asks.
- **HTTP/protobuf transport** — only `grpc-tonic` ships. HTTP/proto follows when needed.
- **Prometheus exporter removal** — `PrometheusExporter` stays alongside `OtlpExporter`. Operators choose; both can coexist.
- **Adaptive sampling / tail-based sampling** — only static `trace_sampling_ratio`. Adaptive sampling is a v0.2+ concern (needs a sampler that knows the full trace, which requires an OTel collector sidecar).
- **Metrics for the multi-node gRPC server itself** — the gRPC server in `crates/dist` has no metrics today; that's a separate concern (P43 scoped to the inference engine + HTTP server, which is the hot path).
- **OpenTelemetry logs SDK** — same as above; defer.

---

## 4. Architecture

### 4.1 Component diagram

```
┌─────────────────────────────────────┐
│ EnhancedMetricsCollector            │  snapshot() every N secs
│ (in-memory metrics, existing)       │ ───────────────┐
└─────────────────────────────────────┘                 │
                                                        ▼
┌─────────────────────────────────────┐    ┌─────────────────────────────┐
│ tracing-subscriber spans            │    │ OtlpExporter                │
│ (request_id, engine.*,              │    │ - background tokio task     │
│  scheduler.*)                       │    │ - opentelemetry-otlp        │
└────────────────┬────────────────────┘    │   (grpc-tonic)              │
                 │ layer                   └──────────────┬──────────────┘
                 ▼                                        │
┌─────────────────────────────────────┐                   │
│ tracing-opentelemetry               │                   │
│ → OpenTelemetry Spans               │                   │
└────────────────┬────────────────────┘                   │
                 │                                        │
                 └────────────────┐                      │
                                  ▼                      ▼
                          ┌──────────────────────────────┐
                          │ OTel Collector (any impl)    │
                          │ Jaeger / Tempo / Datadog     │
                          │ Honeycomb / New Relic        │
                          └──────────────────────────────┘
```

### 4.2 Data flow

1. **Server bootstrap** (main.rs):
   - Reads `app_config.observability.otlp`.
   - If `enabled`, calls `init_tracing_with_otlp(env_filter, otlp_config)` which:
     - Sets up `tracing-subscriber` with `env_filter` + `tracing-opentelemetry` layer.
     - Creates `OtlpGuard` (holds `MeterProvider` + `TracerProvider` for shutdown flush).
   - Calls `spawn_otlp_exporter(state, otlp_config)` which:
     - Creates a `MeterProvider` with `PeriodicReader` exporting every `metrics_export_interval_secs`.
     - Spawns a `tokio` task that polls `state.metrics.snapshot()` and pushes via `opentelemetry-otlp`.
     - Returns `OtlpHandle { meter_provider, tracer_provider, task_handle }`.

2. **Steady state**:
   - `EnhancedMetricsCollector` records counters / gauges as the engine runs (existing code, unchanged).
   - Background task pushes the snapshot every 30s (default) via `opentelemetry-otlp`'s gRPC exporter.
   - Every `tracing::info_span!(...)` (already used pervasively for `request_id`, `engine.add_request`, `scheduler.batch`) is bridged to OTel spans in real time.

3. **Shutdown** (existing graceful shutdown sequence from P7):
   - Engine thread joins.
   - `OtlpHandle` dropped → flushes pending spans + metrics via the SDK's `shutdown()` method.
   - axum listener drains (with `shutdown_drain_grace_secs` grace period).

### 4.3 Schema mapping (Prometheus metrics → OTel instruments)

| Prometheus metric | OTel instrument | Type | Unit |
|-------------------|-----------------|------|------|
| `cuda_graph_hits_total` | `cuda.graph.hits` | Counter | `{hit}` |
| `cuda_graph_misses_total` | `cuda.graph.misses` | Counter | `{miss}` |
| `speculative_adjustments_total` | `speculative.adjustments` | Counter | `{adjustment}` |
| `requests_total` | `requests` | Counter | `{request}` |
| `draft_resolutions_external_total` | `draft.resolutions.external` | Counter | `{resolution}` |
| `draft_resolutions_self_spec_total` | `draft.resolutions.self_spec` | Counter | `{resolution}` |
| `draft_resolutions_none_total` | `draft.resolutions.none` | Counter | `{resolution}` |
| `draft_load_failures_total` | `draft.load.failures` | Counter | `{failure}` |
| `draft_runtime_errors_total` | `draft.runtime.errors` | Counter | `{error}` |
| `packing_efficiency` | `packing.efficiency` | Gauge | `{ratio}` |
| `speculative_acceptance_rate` | `speculative.acceptance_rate` | Gauge | `{ratio}` |
| `speculative_efficiency` | `speculative.efficiency` | Gauge | `{ratio}` |
| `throughput_speedup_ratio` | `throughput.speedup_ratio` | Gauge | `{ratio}` |
| `speculative_per_request_count` | `speculative.per_request_count` | Gauge | `{sequence}` |
| `request_queue_depth` | `request.queue_depth` | UpDownCounter | `{request}` |
| `active_sequences` | `active.sequences` | UpDownCounter | `{sequence}` |
| `gpu_memory_used_bytes` | `gpu.memory.used` | UpDownCounter | `By` |
| `gpu_memory_total_bytes` | `gpu.memory.total` | UpDownCounter | `By` |
| `is_leader` | `is_leader` | Gauge | `{bool}` |
| `inflight_requests` | `inflight.requests` | UpDownCounter | `{request}` |
| `scheduler_queue_size` | `scheduler.queue_size` | UpDownCounter | `{request}` |

Unit conventions follow the OTel semconv style (e.g. `By` for bytes, `{request}` for request counts).

### 4.4 OTel resource attributes

The OTel SDK is configured with a `Resource` carrying:

| Attribute | Source | Example |
|-----------|--------|---------|
| `service.name` | `otlp.service_name` config | `"vllm-lite"` |
| `service.version` | `otlp.service_version` config (synced from release manifest) | `"0.1.0"` |
| `service.instance.id` | auto-generated UUID v4 at startup | `"550e8400-e29b-41d4-a716-446655440000"` |
| `host.arch` | `std::env::consts::ARCH` | `"x86_64"` |

Follows [OTel semantic conventions](https://opentelemetry.io/docs/specs/semconv/resource/).

---

## 5. Module layout

```
crates/core/src/metrics/exporter/
├── mod.rs              # existing (MetricsExporter trait + InMemoryMetricsExporter + PrometheusExporter)
└── otlp.rs             # NEW — OtlpConfig + OtlpExporterBuilder + background task + MetricsError::Otlp

crates/core/src/
├── metrics.rs          # existing (module root)
└── tracing_init.rs     # NEW — init_tracing_with_otlp + OtlpGuard

crates/server/src/
├── bootstrap/
│   ├── mod.rs          # existing (re-export from new observability module)
│   └── observability.rs # NEW — spawn_otlp_exporter + bootstrap glue
├── config.rs (or new config/observability.rs) # extend AppConfig with observability.otlp section
└── main.rs             # wire init_tracing_with_otlp + spawn_otlp_exporter in startup; drop handle in shutdown
```

### 5.1 New types

- `vllm_core::metrics::exporter::otlp::OtlpConfig` (gated `#[cfg(feature = "opentelemetry")]`)
  - `enabled: bool`
  - `endpoint: String` (default `"http://localhost:4317"`)
  - `service_name: String` (default `"vllm-lite"`)
  - `service_version: String` (default workspace version)
  - `metrics_export_interval_secs: u64` (default `30`)
  - `trace_sampling_ratio: f64` (default `1.0`, range `[0.0, 1.0]`)
  - `protocol: OtlpProtocol` (enum with single `Grpc` variant in v43; the type is reserved so a future v32+ HTTP/proto transport is a non-breaking addition)
- `vllm_core::metrics::exporter::otlp::OtlpExporter` (gated)
  - `new(collector: Arc<EnhancedMetricsCollector>, config: OtlpConfig) -> Result<Self, OtlpError>`
  - `async fn run(self) -> Result<(), OtlpError>` — the background task body
- `vllm_core::metrics::exporter::otlp::OtlpError` (gated) — typed errors:
  - `Config(String)`
  - `Export(String)`
  - `CollectorUnreachable(String)`
  - `Builder(String)`
- `vllm_core::tracing_init::init_tracing_with_otlp(env_filter: EnvFilter, config: OtlpConfig) -> Result<OtlpGuard, OtlpError>` (gated)
- `vllm_core::tracing_init::OtlpGuard` (gated) — Drop-flush guard
- `vllm_server::bootstrap::observability::spawn_otlp_exporter(state, config) -> Result<OtlpHandle, OtlpError>`
- `vllm_server::bootstrap::observability::OtlpHandle` — Drop-flush handle for shutdown

### 5.2 Modified types

- `vllm_core::metrics::exporter::MetricsError` — add `Otlp(String)` variant
- `vllm_server::config::ServerConfig` (or wherever `app_config` lives) — add `observability: ObservabilityConfig` field with `#[serde(default)]`
- `vllm_server::config::ObservabilityConfig` (new) — wraps `otlp: OtlpConfig` (defined in `vllm_core` and re-exported, or mirrored)

### 5.3 Dependencies

Cargo.toml changes (versions are the workspace's best estimate as of 2026-07-22 — the implementation phase will verify against `cargo add` and may adjust):

- `crates/core/Cargo.toml`:
  ```toml
  [features]
  opentelemetry = [
      "dep:opentelemetry",
      "dep:opentelemetry_sdk",
      "dep:opentelemetry-otlp",
      "dep:opentelemetry-semantic-conventions",
      "dep:tracing-opentelemetry",
  ]
  
  [dependencies]
  opentelemetry = { version = "0.27", optional = true }
  opentelemetry_sdk = { version = "0.27", optional = true, features = ["rt-tokio"] }
  opentelemetry-otlp = { version = "0.27", optional = true, features = ["grpc-tonic", "metrics"] }
  opentelemetry-semantic-conventions = { version = "0.27", optional = true }
  tracing-opentelemetry = { version = "0.28", optional = true }
  ```

- `crates/server/Cargo.toml`:
  ```toml
  vllm-core = { path = "../core", features = ["opentelemetry"] }  # opt-in
  ```

Versions chosen to align with the `tracing` 0.1.x + `opentelemetry` 0.27.x + `tracing-opentelemetry` 0.28.x ecosystem as of mid-2026.

---

## 6. Configuration

```yaml
# config/server.yaml (full example)
server:
  bind: "0.0.0.0:8000"
  # ... existing fields ...

observability:
  otlp:
    enabled: false                              # default off (backward compat)
    endpoint: "http://localhost:4317"           # OTLP/gRPC default port
    service_name: "vllm-lite"
    service_version: "0.1.0"                    # synced from release manifest
    metrics_export_interval_secs: 30
    trace_sampling_ratio: 1.0                   # 0.0=disabled, 1.0=always sample
    protocol: "grpc"                            # grpc (only one supported in v43)
```

CLI override: `--otlp-endpoint http://collector:4317` (mutually exclusive with YAML; CLI wins if both set).

---

## 7. Error handling

| Failure mode | Behavior |
|--------------|----------|
| Bad config (negative sampling ratio, invalid endpoint URL) | `OtlpError::Config` returned at bootstrap; log `warn!` + continue without OTLP |
| Collector unreachable at startup | `OtlpError::CollectorUnreachable` returned; log `warn!` + continue; first periodic export will retry |
| Collector unreachable during runtime | Log `warn!` once per minute (rate-limited); next interval retries |
| OTel SDK internal panic | Process-wide panic (matches other deps); no defense-in-depth here |
| Tracer / meter double-init | `init_tracing_with_otlp` returns `OtlpError::AlreadyInitialized`; bootstrap logs + continues with the existing subscriber |
| Shutdown while flush pending | `OtlpGuard::drop` blocks until SDK reports flush complete (5s timeout, configurable) |

---

## 8. Testing strategy

### 8.1 Unit tests

In `crates/core/src/metrics/exporter/otlp.rs::tests`:
- `otlp_config_default_matches_spec` — all default fields match the spec
- `otlp_config_parses_yaml_minimal` — `{ otlp: { enabled: true, endpoint: "http://x:4317" } }` parses
- `otlp_config_rejects_negative_sampling_ratio` — out-of-range rejected at config-load time
- `otlp_config_rejects_zero_metrics_interval` — `metrics_export_interval_secs = 0` rejected
- `otlp_exporter_builder_constructs_meter_provider` — no I/O, just construction
- `metric_schema_mapping_covers_all_prometheus_metrics` — every Prometheus metric has an OTel counterpart (catches drift)
- `otlp_error_display_messages_are_actionable` — error variants render with field-naming messages

In `crates/core/src/tracing_init.rs::tests`:
- `init_tracing_with_otlp_disabled_is_noop` — when `enabled = false`, returns a guard that drops cleanly
- `otlp_guard_drop_flushes_pending_spans` — using a stub span exporter that records `force_flush()` calls

### 8.2 Integration tests

New file `crates/server/tests/otlp_exporter_integration.rs` (gated `--features opentelemetry`):

- **Stub collector**: a minimal in-process tonic server that implements the OTLP `MetricsService` and `TraceService` gRPC services, recording received requests in `Arc<Mutex<Vec<...>>>`.
- `otlp_metrics_arrive_with_correct_name_and_value` — start server with OTLP enabled + stub collector on port 0 → wait one export interval → assert received metrics match the Prometheus snapshot.
- `otlp_traces_arrive_with_parent_child_relationship` — fire a chat completion through the HTTP handler → wait for span flush → assert span tree includes `http.request → engine.add_request → scheduler.batch` with correct parent IDs.
- `otlp_collector_unreachable_does_not_crash_server` — point at `http://127.0.0.1:1` (always refused) → server starts, chat completion succeeds, OTLP logs warn.
- `otlp_sampling_ratio_zero_emits_no_spans` — `trace_sampling_ratio = 0.0` → span count == 0 after several requests.
- `otlp_sampling_ratio_one_emits_all_spans` — `trace_sampling_ratio = 1.0` → span count == request count.
- `otlp_disabled_default_does_not_initialize_sdk` — no `app_config.observability.otlp` section → no OTLP SDK calls, no I/O.

### 8.3 Live collector smoke test (manual, doc-only)

`docs/OPERATIONS.md` adds a "Verify it works" subsection with three commands:
- `docker run -d --name jaeger -p 4317:4317 -p 16686:16686 jaegertracing/all-in-one:latest`
- `cargo run --features opentelemetry -- --otlp-endpoint http://localhost:4317` (in another shell)
- `curl http://localhost:8000/v1/chat/completions -d '{"model": "...", "messages": [...]}'`
- Open `http://localhost:16686` → see traces.

---

## 9. CI integration

### 9.1 New CI job

`.github/workflows/ci.yml` gains a `ci-otlp` job (parallel to `ci`, `ci-all-features`, `mutation-nightly`):

```yaml
ci-otlp:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@1.88
    - run: cargo test -p vllm-core --features opentelemetry otlp
    - run: cargo test -p vllm-server --features vllm-core/opentelemetry --test otlp_exporter_integration
```

Runs on every PR + push to main. ~2-3 min budget (stub collector, no external I/O).

### 9.2 Doc coverage

`scripts/doc_coverage.sh --real json` continues to count only the default-features build (which excludes the OTLP module). The `#[cfg(feature = "opentelemetry")]` gating should keep the default-features doc coverage ratio unchanged at 68.0%+ — this will be re-verified during the implementation phase.

### 9.3 Feature matrix doc

`docs/reference/feature-matrix.md` gains a new `opentelemetry` row under `vllm-core` (always-off by default; opt-in for OTLP exporter + tracing bridge).

---

## 10. Documentation

### 10.1 New ADR

`docs/adr/ADR-021-otlp-exporter.md` — captures the architectural decisions:

- Why `grpc-tonic` over `http-proto` (composes with existing stack; transport parity with the multi-node gRPC server in P40-P42)
- Why push-based metrics (OTLP is push; Prometheus is pull — two different mental models, two different exporters)
- Why `tracing-opentelemetry` bridge (already have `tracing` spans everywhere; cheaper than re-instrumenting with `opentelemetry::trace` directly)
- Why `trace_sampling_ratio = 1.0` default (vllm-lite is not a high-volume tracing target; opt-in sampling saves future users from the "why am I missing spans" footgun)
- Why defer logs (existing `tracing-subscriber` JSON is log-aggregator friendly)
- Why defer HTTP/proto transport (no current customer request)
- Why keep `PrometheusExporter` (parity; operators choose)

Bumps ADR count from 20 → 21.

### 10.2 OPERATIONS.md additions

New top-level section "OTLP Observability" (under existing "Observability" → after the Prometheus subsection):

- **Setup with Jaeger all-in-one** (Docker one-liner).
- **Setup with Tempo** (Docker Compose snippet referencing Grafana).
- **Setup with Datadog Agent** (yaml config + env var).
- **Setup with Honeycomb** (env var pointing at the API endpoint).
- **Sampling strategies** — when to drop `trace_sampling_ratio` below 1.0; how to wire a tail-based sampler (link to OTel docs).
- **Troubleshooting** — common failure modes (wrong endpoint, no collector listening, sampling too low, firewall) + how to read the warning logs.

### 10.3 CHANGELOG.md

`[Unreleased] > Added` entry following the established P-batch format:

- New feature: `opentelemetry` opt-in for OTLP metrics + traces export
- New config: `app_config.observability.otlp.{enabled, endpoint, service_name, service_version, metrics_export_interval_secs, trace_sampling_ratio, protocol}`
- New public types (gated): `OtlpConfig`, `OtlpExporter`, `OtlpError`, `init_tracing_with_otlp`, `OtlpGuard`, `spawn_otlp_exporter`, `OtlpHandle`
- New ADR-021
- New CI job `ci-otlp`
- N new tests (target: ~7 unit tests in `otlp.rs::tests` + ~6 integration tests in `otlp_exporter_integration.rs` = ~13 new tests; final count recorded at land time)
- `public-api: vllm-core added` + `public-api: vllm-server added` bullets (per the public-api-check gate)

### 10.4 STATE.md update

After P43 lands, update `STATE.md` to record the closure of the OTLP v32+ item and bump test counts.

---

## 11. Risk + rollback

### 11.1 Risks

- **Dependency weight** — adding 5 OTel crates will grow the workspace compile time by ~15-25s (single build, after warm cache). Acceptable.
- **gRPC port collision** — OTLP defaults to 4317, which is unlikely to collide with the multi-node gRPC server (50051). Documented.
- **Sampling 0.0 footgun** — operators who set `trace_sampling_ratio = 0.0` will see no spans; documented as "no traces" in OPERATIONS.md.
- **OTel SDK upgrade cadence** — OTel 0.x → 1.0 transition is in progress (2026 H2); v43 pins to 0.27 for stability, with a follow-up phase for 1.0 migration tracked as v32+.

### 11.2 Rollback

- All P43 code is gated by `#[cfg(feature = "opentelemetry")]`. Default-features build is unchanged.
- `cargo build --no-default-features` skips the OTLP module entirely.
- Reverting the P43 commits restores the prior behavior with zero blast radius.

---

## 12. Out-of-scope follow-ups (v32+ candidates after P43)

1. **OTLP logs export** — when a customer asks.
2. **HTTP/protobuf transport** — when a customer asks.
3. **Adaptive / tail-based sampling** — needs an OTel collector sidecar.
4. **OpenTelemetry 1.0 migration** — once stable.
5. **gRPC server metrics** — `crates/dist` has no metrics today; separate concern.
6. **Per-tenant quota** — production-readiness §6 follow-up (separate from observability).

---

## 13. Acceptance criteria

P43 ships when ALL of the following are true:

- [ ] `cargo build --features opentelemetry` succeeds on the workspace.
- [ ] `cargo build` (default features) succeeds with no change vs main.
- [ ] `cargo test --features opentelemetry -p vllm-core` passes all unit tests.
- [ ] `cargo test --features vllm-core/opentelemetry -p vllm-server --test otlp_exporter_integration` passes all 6 integration tests.
- [ ] `just ci` exits 0 (fmt-check, clippy, doc-check, doctest, nextest, public-api-check, doc-coverage-check).
- [ ] `cargo doc --workspace --features opentelemetry` passes with no broken intra-doc links.
- [ ] The OTLP CI job (`ci-otlp`) passes on a PR.
- [ ] OPERATIONS.md "OTLP Observability" section is published with Jaeger + Tempo + Datadog examples.
- [ ] ADR-021 is committed under `docs/adr/`.
- [ ] CHANGELOG.md has the `[Unreleased] > Added` entry + `public-api:` bullets.
- [ ] STATE.md records the P43 closure and the updated test count.
- [ ] Live verification: operator can run `docker run -d jaegertracing/all-in-one` + `cargo run --features opentelemetry` + `curl /v1/chat/completions` and see traces in `http://localhost:16686`.
- [ ] CLI override `--otlp-endpoint <URL>` works (takes precedence over YAML when both are set).

---

## 14. Effort estimate

- T1: Cargo.toml + module skeleton + OtlpConfig type — 0.3 day
- T2: OtlpExporter background task + schema mapping — 0.7 day
- T3: tracing-init helper + OtlpGuard — 0.4 day
- T4: Server bootstrap wiring (config + main.rs + shutdown) — 0.4 day
- T5: Unit tests + integration tests + stub collector — 0.5 day
- T6: CI job + OPERATIONS.md + ADR-021 + CHANGELOG + STATE.md — 0.2 day
- **Total: ~2.5 days** (matches the AskUserQuestion estimate)
