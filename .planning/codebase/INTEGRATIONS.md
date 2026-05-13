# External Integrations

**Analysis Date:** 2026-05-13

## APIs & External Services

### OpenAI-Compatible HTTP API

The server exposes an OpenAI-compatible REST API via Axum. This is the primary external interface for clients consuming LLM inference. All endpoints are defined in `crates/server/src/main.rs:230-253`.

- **Chat Completions:** `POST /v1/chat/completions` — Streaming (SSE) and non-streaming chat completions
    - Implementation: `crates/server/src/openai/chat.rs`
- **Text Completions:** `POST /v1/completions` — Legacy completions endpoint with streaming support
    - Implementation: `crates/server/src/openai/completions.rs`
- **Embeddings:** `POST /v1/embeddings` — Text embedding generation
    - Implementation: `crates/server/src/openai/embeddings.rs`
- **Models:** `GET /v1/models` — List available models
    - Implementation: `crates/server/src/openai/models.rs`
- **Batches:** `POST /v1/batches`, `GET /v1/batches`, `GET /v1/batches/{id}`, `GET /v1/batches/{id}/results`
    - Implementation: `crates/server/src/openai/batch/`

**Note:** These endpoints implement the OpenAI API contract but route to the local inference engine — no external OpenAI API calls are made.

### gRPC Distributed Inference Service

Inter-node communication for multi-node distributed inference uses gRPC (tonic/protobuf):

- **Service:** `NodeService` defined in `crates/dist/proto/node.proto`
- **Endpoints:**
    - `Ping` — Health check between nodes
    - `AllReduce` — Tensor all-reduce operation (NCCL placeholder, not yet implemented)
    - `GetKVCache` — Distributed KV cache retrieval by block hash
    - `GetPeers` — Peer discovery
- **Codegen:** `crates/dist/build.rs` uses `tonic_build` to compile protobuf
- **Implementation:** `crates/dist/src/grpc.rs` — `NodeServiceImpl` and gRPC server bootstrap

### Third-Party SDKs/Imports

| SDK                                | Usage                                     | Crate                                            |
| ---------------------------------- | ----------------------------------------- | ------------------------------------------------ |
| `candle-core` / `candle-nn` 0.10.2 | ML tensor operations (HuggingFace Candle) | vllm-model, vllm-traits, vllm-core, vllm-testing |
| `tiktoken` 3.x                     | BPE tokenization (OpenAI tokenizer)       | vllm-model                                       |
| `tokenizers` 0.22                  | HuggingFace tokenizers library            | vllm-model                                       |
| `safetensors` 0.7.0                | SafeTensor model checkpoint loading       | vllm-model                                       |
| `gguf` 0.1                         | GGUF model format loading                 | vllm-model (optional)                            |
| `reqwest` 0.12                     | HTTP client for external requests         | vllm-server                                      |
| `k8s.io/client-go` v0.30.0         | Kubernetes API client                     | k8s/operator (Go)                                |

**No detected imports for:** Stripe, Supabase, AWS SDK, Anthropic SDK, Redis, PostgreSQL, MySQL, MongoDB, or any traditional database drivers.

## Data Storage

**Databases:**

- **None.** The project is stateless at the persistence layer. No relational or NoSQL databases are used.

**File Storage:**

- Model checkpoints: Loaded from local filesystem via `ModelLoader` (`crates/model/src/loader/`). Supports Safetensors (`.safetensors`, sharded) and GGUF (`.gguf`) formats.
- KV cache: In-memory, paged block allocation (`crates/core/src/kv_cache/`), distributed via gRPC (`crates/dist/src/distributed_kv/`)
- Log files: Optional JSON log output to a configured directory (`VLLM_LOG_DIR`) with daily rotation via `tracing-appender`
- Docker volume: `vllm-cache` volume for `.cache` directory in containers
- Models mounted as read-only volumes in Docker/K8s: `./models:/app/models:ro`

**Caching:**

- KV cache prefix sharing: `crates/core/src/kv_cache/prefix_cache.rs`
- CUDA Graph caching: Via `cuda-graph` feature in vllm-core
- Docker layer caching in `Dockerfile` (multi-stage, dependency-only layer)

## Authentication & Identity

**Auth Provider:**

- **Custom / Self-contained.** No external identity provider (no OAuth, OIDC, Auth0, etc.).
- **API Key Bearer Tokens:**
    - Configured via `VLLM_API_KEY` env var (repeatable) or `VLLM_API_KEYS_FILE` file path
    - OR via `AppConfig` YAML config (`auth.api_keys`, `auth.api_keys_env`, `auth.api_keys_file`)
    - Validated in `crates/server/src/auth.rs` with `AuthMiddleware::verify()`
    - Sends `401 Unauthorized` on invalid/missing keys
    - Middleware is conditionally applied; empty key list = auth disabled
- **Rate Limiting:**
    - In-memory, per-api-key tracking in `crates/server/src/auth.rs` `RateLimiter`
    - Default: 100 requests per 60-second window
    - Sends `429 Too Many Requests` on exceeded limits

**JWT Support (Planned/Stub):**

- `crates/server/src/security/jwt.rs` provides `JwtValidator` and `JwtAuthMiddleware`
- Implements HS256-style validation (secret key or public key)
- Claims: `sub`, `iss`, `aud`, `exp`, `iat`, `roles`, `scope`
- **Not currently wired into the main request pipeline** — this is a security module available for future integration

**RBAC (Planned/Stub):**

- `crates/server/src/security/rbac.rs` provides `RbacMiddleware` and `Role` enum
- Roles: Admin, Operator, User, Anonymous
- Permission model based on action strings (`read`, `write`, `execute`, `admin`)
- **Not currently wired into the main request pipeline**

**TLS/mTLS:**

- `crates/server/src/security/tls.rs` — `TlsConfig` and `TlsListener` using `tokio-rustls` / `rustls`
- Supports server-side TLS and mTLS (client certificate verification via CA cert)
- Configuration: certificate path, key path, optional CA cert path
- Dependencies: `tokio-rustls` 0.26, `rustls-pemfile` 2

**Audit Logging (Stub):**

- `crates/server/src/security/audit.rs` — `AuditLogger` struct
- `crates/server/src/security/correlation.rs` — `CorrelationIdMiddleware` for request tracing

## Monitoring & Observability

**Metrics Export:**

- **Prometheus:** Via `metrics-exporter-prometheus` 0.13 (enabled by default via `prometheus` feature)
    - Endpoint: `GET /metrics` returns Prometheus text format
    - Custom metrics: `vllm_tokens_total`, `vllm_requests_total`, `vllm_avg_latency_ms`, `vllm_p50_latency_ms`, `vllm_p90_latency_ms`, `vllm_p99_latency_ms`, `vllm_avg_batch_size`, `vllm_current_batch_size`, `vllm_requests_in_flight`, `vllm_kv_cache_usage_percent`, `vllm_prefix_cache_hit_rate`, `vllm_prefill_throughput`, `vllm_decode_throughput`, `vllm_avg_scheduler_wait_time_ms`
    - Implementation: `crates/server/src/api.rs:53-105` and `crates/server/src/main.rs:79-88`
- **OpenTelemetry:** Via `opentelemetry` / `opentelemetry-otlp` 0.21 (optional, `opentelemetry` feature in vllm-core)
    - Supports OTLP export to collectors (e.g., Grafana Cloud, Jaeger, Datadog)

**Logging:**

- **Framework:** `tracing` 0.1 ecosystem
- **Dual Output:** When `VLLM_LOG_DIR` is set, logs to both console (human-readable compact format) and file (JSON format with daily rotation)
- **Levels:** TRACE, DEBUG, INFO, WARN, ERROR
- **Configuration:** `VLLM_LOG_LEVEL` env var or `RUST_LOG` env-filter
- **File location:** `{VLLM_LOG_DIR}/vllm-lite.log` with `Rotation::DAILY`
- **Implementation:** `crates/server/src/logging.rs`

**Health Checks:**

- `GET /health` / `GET /health/live` — Liveness probe (200 OK or 503)
- `GET /ready` / `GET /health/ready` — Readiness probe (200 OK or 503)
- `GET /health/details` — Detailed health with GPU info and KV cache usage
- Implementation: `crates/server/src/health.rs`

**Debug Endpoints:**

- `GET /debug/metrics` — Metrics snapshot (counters, gauges, queue depth, CUDA graph hit rate)
- `GET /debug/kv-cache` — KV cache dump (blocks, usage, prefix cache stats)
- `GET /debug/trace` — Trace status (log level, active spans)
- Implementation: `crates/server/src/debug.rs`

**Dashboard Stack (Docker Compose):**

- **Prometheus:** `prom/prometheus:latest` on port 9090 (profile: `monitoring`)
    - Config: `config/prometheus.yml` (scrapes port 9090 every 15s)
- **Grafana:** `grafana/grafana:latest` on port 3000 (profile: `monitoring`)
    - Default credentials: `admin` / `vllm-admin`

**Error Tracking:**

- **None.** No Sentry, Bugsnag, or equivalent error tracking service configured.

## CI/CD & Deployment

**Hosting:**

- **Self-hosted.** No cloud-platform-specific deployment (no AWS, GCP, Azure SDK usage detected)
- **Containerized:** Docker via `Dockerfile` (multi-stage, non-root user)
- **Orchestration:** Kubernetes via Helm chart (`k8s/charts/vllm-lite/`) and custom controller (Go operator at `k8s/operator/`)
- **Load testing:** k6 via Docker Compose (`grafana/k6:latest`, profile: `loadtest`)

**CI Pipeline (GitHub Actions):**

- **Workflow files:**
    - `.github/workflows/ci.yml` — CI: format check, clippy, build, test (stable + beta matrix), docs, security audit
    - `.github/workflows/benchmark.yml` — PR benchmark regression check (10% threshold)
    - `.github/workflows/performance-regression.yml` — Detailed performance regression with PR comments
- **Rust installation:** `dtolnay/rust-toolchain`
- **Caching:** `actions/cache` for `.cargo/` and `target/`
- **Deploy docs:** `peaceiris/actions-gh-pages` on push to main
- **Security:** `cargo-audit` in the `security` job

**Kubernetes Resources:**

- `k8s/deployment.yaml` — Base deployment manifest
- `k8s/service.yaml` — Service definition
- `k8s/configmap.yaml` — Configuration
- `k8s/hpa.yaml` — Horizontal Pod Autoscaler
- `k8s/namespace.yaml` — Namespace
- `k8s/deploy.sh` — Deployment script
- `k8s/charts/vllm-lite/` — Helm chart (templates for deployment, service, hpa, configmap, rbac, pdb)
- `k8s/operator/` — Custom Kubernetes controller in Go
    - CRD: `vllmengine-crd.yaml` (`k8s/operator/config/crd/`)
    - Controller: `k8s/operator/cmd/controller/main.go`
    - Built with `controller-runtime` v0.18.0 and `controller-tools` v0.14.0

**Pre-commit Hooks:**

- `.pre-commit-config.yaml` — `commitizen` (commit message validation), `rumdl` (markdown lint), `end-of-file-fixer`, `trailing-whitespace`, `check-toml`, `check-yaml`, `check-merge-conflict`, `mixed-line-ending`

## Environment Configuration

**Required env vars for operation:**

- `VLLM_MODEL` — Path to the model directory (required via CLI `--model` / `-m`)
- Nothing else is strictly required — all other configs have defaults

**Secrets location:**

- API keys: CLI args (`--api-key`), env vars (`VLLM_API_KEY`), or file (`VLLM_API_KEYS_FILE`)
- TLS certificates: Filesystem paths configured via `TlsConfig` struct
- Docker Compose: Hardcoded `GF_SECURITY_ADMIN_PASSWORD=vllm-admin` in `docker-compose.yml:72`
- No `.env` files committed to the repository (`.env` and `.env.local` in `.gitignore`)
- No external secrets manager detected (no Vault, AWS Secrets Manager, etc.)

## Webhooks & Callbacks

**Incoming:**

- **None.** No webhook reception endpoints are configured.

**Outgoing:**

- **None.** No outbound webhooks are sent to external services.

---

*Integration audit: 2026-05-13*
