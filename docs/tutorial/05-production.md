# Tutorial 5: Production Deployment

This tutorial covers deploying vllm-lite to production.

## Local Development (docker-compose)

A minimal local stack:

```bash
docker-compose up
```

This starts vllm-server on `localhost:8000` (OpenAI-compatible API) plus
Prometheus + Grafana for monitoring. See `docker-compose.yml` for the
full service list.

## Kubernetes Deployment

Manifests in `k8s/`:

- `k8s/deployment.yml` — vllm-server deployment
- `k8s/service.yml` — ClusterIP service
- `k8s/ingress.yml` — external access (with TLS)
- `k8s/hpa.yml` — horizontal pod autoscaler

Apply:

```bash
kubectl apply -f k8s/
```

## Observability

vllm-lite exports metrics in Prometheus format:

- `vllm:request_total` — total requests processed
- `vllm:request_duration_seconds` — request latency histogram
- `vllm:prefix_cache_hit_rate` — prefix cache effectiveness
- `vllm:kv_cache_utilization` — KV cache memory usage

Grafana dashboards in `docs/grafana/`. Import into your Grafana instance.

Distributed tracing via OpenTelemetry (optional, `opentelemetry` feature):

```bash
cargo build --features opentelemetry
```

## Security Checklist

- [ ] API key auth enabled (`AuthConfig::resolve_api_keys`)
- [ ] TLS termination at ingress
- [ ] Rate limiting at API gateway
- [ ] Pod security context (non-root, read-only filesystem)
- [ ] Network policies restricting egress to model registry

## Performance Tuning

See `docs/optimization_guide.md` for detailed tuning. Quick wins:

- Increase `kv_blocks` for higher concurrency
- Enable prefix caching for workloads with repeated prompts
- Use `--features cuda` for GPU acceleration
- Set `--release` profile for production builds

## Rollback Strategy

All v30+ releases are tagged in git. To rollback:

```bash
kubectl set image deployment/vllm-server \
  vllm=vllm-lite:v0.30.0  # or previous known-good version
```

## See also

- `docker-compose.yml` — local stack definition
- `k8s/` — Kubernetes manifests
- `docs/grafana/` — monitoring dashboards
- `docs/optimization_guide.md` — performance tuning
- `SECURITY.md` — security policy
