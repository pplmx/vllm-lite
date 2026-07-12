# Tutorial 5: Production Deployment

This tutorial covers deploying vllm-lite to production. It assumes
you've already worked through tutorials 1–4 and have a working
local build.

> ⚠️ **Honest disclosure**: vllm-lite's production-readiness
> work is documented in
> [`docs/technical-due-diligence/production-readiness.md`](../technical-due-diligence/production-readiness.md).
> Several items called out there are still open (GPU CI,
> continuous-batching kernel, TLS termination, CORS layer). Treat
> this tutorial as "the deployment surfaces we currently have" —
> not "vllm-lite is production-grade". See the
> [ROADMAP](../technical-due-diligence/roadmap.md) for what the
> project is working toward.

## Local Development (docker-compose)

A minimal local stack:

```bash
docker-compose up
```

This starts vllm-server on `localhost:8000` (OpenAI-compatible API) plus
Prometheus + Grafana for monitoring. See `docker-compose.yml` for the
full service list.

## Kubernetes Deployment

The committed manifests in `k8s/` use `.yaml` (not `.yml`) extensions:

- `k8s/deployment.yaml` — vllm-server deployment
- `k8s/service.yaml` — ClusterIP service
- `k8s/hpa.yaml` — horizontal pod autoscaler
- `k8s/configmap.yaml` — non-secret config (host, port, model path)
- `k8s/namespace.yaml` — `vllm` namespace
- `k8s/charts/vllm-lite/` — Helm chart (preferred for real deployments)

The Helm chart is what `release.yml` packages; see
[`docs/RELEASE.md`](../RELEASE.md) for the version-source-of-truth
flow.

> **No `ingress.yaml` is shipped.** The project's recommendation
> is to terminate TLS at the cluster's ingress controller (Envoy,
> nginx-ingress, Traefik, etc.) and let the chart + service expose
> plain HTTP inside the cluster. Bringing up the ingress is
> environment-specific and would drift from one provider to
> another. If you need a sample, copy `k8s/service.yaml` into a
> `Ingress` resource in your platform's preferred form.

Apply the raw manifests:

```bash
kubectl apply -f k8s/
```

Or install via Helm (preferred):

```bash
helm install vllm-lite k8s/charts/vllm-lite/ \
  --set model.path=/models/qwen2.5-0.5b \
  --set replicaCount=1
```

## Observability

vllm-lite exports metrics in Prometheus format on `/metrics`:

- `vllm:request_total` — total requests processed
- `vllm:request_duration_seconds` — request latency histogram
- `vllm:prefix_cache_hit_rate` — prefix cache effectiveness
- `vllm:kv_cache_utilization` — KV cache memory usage

Grafana dashboards in `docs/grafana/`. Import into your Grafana
instance.

> **OpenTelemetry is not currently wired.** The `opentelemetry`,
> `opentelemetry-otlp`, `opentelemetry_sdk`, and
> `tracing-opentelemetry` crates are listed in the root
> `Cargo.toml` workspace dependency table but no crate uses them
> (the v30.0 dependency-cleanup pass removed the corresponding
> `#[cfg(feature = "opentelemetry")]` gates). Adding a real OTel
> exporter is tracked as a v32+ item; see
> [ROADMAP §3](../technical-due-diligence/roadmap.md#3-短期演进1-2-个月).
> In the meantime, the structured `tracing` JSON log stream covers
> the same observability needs without the OTel infrastructure.

## Security Checklist

- [ ] API key auth enabled (`AuthConfig::resolve_api_keys`)
- [ ] TLS termination at your ingress controller
- [ ] Rate limiting at the API gateway (vllm-lite's per-key
      rate limiter is in-token only — see
      [`crates/server/src/auth.rs`](../../crates/server/src/auth.rs))
- [ ] Pod security context (non-root, read-only filesystem)
- [ ] Network policies restricting egress to model registry

## Performance Tuning

See [`docs/optimization_guide.md`](../optimization_guide.md) for
detailed tuning. Quick wins:

- Increase `kv_blocks` for higher concurrency
- Enable prefix caching for workloads with repeated prompts
- Use the `--features cuda-graph` build for GPU acceleration
  (requires CUDA 11.8+; only meaningful on the linux-amd64
  release build — see [ROADMAP §4](../technical-due-diligence/roadmap.md#4-中期架构演进3-6-个月))
- Set `--release` profile for production builds

## Rollback Strategy

Releases are tagged in git with `vX.Y.Z` matching the
`[workspace.package] version` in `Cargo.toml` (the single source
of truth — see [`docs/RELEASE.md`](../RELEASE.md)). To rollback:

```bash
# Current workspace version is 0.1.0; the tag is v0.1.0.
# Replace with whatever previous known-good version exists.
kubectl set image deployment/vllm-server \
  vllm=vllm-lite:v0.1.0
```

## See also

- `docker-compose.yml` — local stack definition
- `k8s/` — raw Kubernetes manifests
- `k8s/charts/vllm-lite/` — Helm chart
- `docs/grafana/` — monitoring dashboards
- `docs/optimization_guide.md` — performance tuning
- `SECURITY.md` — security policy
- `docs/RELEASE.md` — release / rollback process
