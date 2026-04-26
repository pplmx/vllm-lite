# Architecture Patterns: vllm-lite v13.0 Host Deployment

**Domain:** LLM Inference Engine Production Deployment
**Researched:** 2026-04-27

## Recommended Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Kubernetes Cluster                        │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    vllm-lite Namespace                    │   │
│  │                                                            │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │   │
│  │  │  Pod-0      │  │  Pod-1      │  │  Pod-N      │       │   │
│  │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │       │   │
│  │  │ │ vllm    │ │  │ │ vllm    │ │  │ │ vllm    │ │       │   │
│  │  │ │ Engine  │ │  │ │ Engine  │ │  │ │ Engine  │ │       │   │
│  │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │       │   │
│  │  │     │       │  │     │       │  │     │       │       │   │
│  │  │  GPU:0     │  │  GPU:0     │  │  GPU:0     │       │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘       │   │
│  │        │                │                │               │   │
│  │        └────────────────┼────────────────┘               │   │
│  │                         │                                │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │         Headless Service (peer discovery)         │   │   │
│  │  │  vllm-lite-peer.vllm-lite.svc.cluster.local      │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  │                                                            │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │         LoadBalancer Service (API)                │   │   │
│  │  │  vllm-lite.vllm-lite.svc.cluster.local:8000      │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  │                                                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 ConfigMap & Secrets                       │   │
│  │   - config.yaml (engine settings)                        │   │
│  │   - api-keys (sealed secrets)                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| vllm-lite Engine | Token generation, KV cache | Pod network |
| ConfigMap | Engine configuration | Mounted as volume |
| Secret | API keys, credentials | Mounted as volume |
| Headless Service | Peer discovery | DNS-based |
| LoadBalancer Service | Client traffic | External clients |
| Lease (Coordination) | Leader election | K8s API server |

## Data Flow

### Request Flow (v13.0)
```
Client Request
     │
     ▼
LoadBalancer Service
     │
     ├──▶ Pod-0 (if selected)
     │
     └──▶ Pod-N (round-robin fallback)
              │
              ▼
         vllm Engine
              │
              ├──▶ Tokenizer
              ├──▶ Model Forward
              └──▶ KV Cache (local only)
```

### Multi-Node Flow (v13.1+)
```
Client Request
     │
     ▼
Router Service
     │
     ▼
Peer Discovery (DNS)
     │
     ▼
Request Router (consistent hashing)
     │
     ├──▶ Pod-0 (has KV cache for prompt)
     │
     └──▶ Pod-N (cache miss, compute)
              │
              ▼
         Distributed KV Cache
         (inter-node transport)
```

## Patterns to Follow

### Pattern 1: Stateless Inference Pods

**What:** Pods do not store persistent state. All state in KV cache is ephemeral.

**When:** v13.0 initial release

**Example:**
```yaml
# Deployment - no PVC, no persistent volumes
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-lite
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: vllm-lite
        # No volume mounts for model (loaded from ConfigMap or image)
```

### Pattern 2: ConfigMap-Driven Configuration

**What:** Engine config from ConfigMap, not baked into image.

**When:** Always

**Example:**
```yaml
# ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: vllm-lite-config
data:
  config.yaml: |
    engine:
      max_batch_size: 256
---
# Pod uses ConfigMap
volumeMounts:
- name: config
  mountPath: /etc/vllm
env:
- name: VLLM_CONFIG_PATH
  value: /etc/vllm/config.yaml
```

### Pattern 3: Health Probe Hierarchy

**What:** Liveness checks process, Readiness checks service availability.

**When:** Always

**Example:**
```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 30  # Wait for model load
  periodSeconds: 5
```

### Pattern 4: Graceful Degradation

**What:** Service returns 503 when not ready, allowing K8s to route elsewhere.

**When:** Always

**Example:**
```rust
async fn health_ready(State(state): State<AppState>) -> impl IntoResponse {
    if !state.engine.is_model_loaded() {
        return (StatusCode::SERVICE_UNAVAILABLE, "Model loading");
    }
    if state.engine.gpu_memory_pct() > 95 {
        return (StatusCode::SERVICE_UNAVAILABLE, "GPU OOM");
    }
    (StatusCode::OK, "Ready")
}
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Embedded Config in Image

**What:** Hardcoding config values in Dockerfile.

**Why bad:** Requires image rebuild for config changes. Violates 12-factor.

**Instead:** Use ConfigMap volume mount.

### Anti-Pattern 2: No Resource Limits

**What:** Missing GPU memory limits.

**Why bad:** OOM kills, noisy neighbor problems.

**Instead:** Set explicit GPU memory requests/limits.

### Anti-Pattern 3: Single Replica

**What:** No high availability.

**Why bad:** Updates cause downtime. Node failure causes outage.

**Instead:** Minimum 2 replicas with PodDisruptionBudget.

## Scalability Considerations

| Concern | At 10 users | At 1K users | At 100K users |
|---------|-------------|-------------|---------------|
| Replicas | 1-2 | 4-8 | 16-32 |
| KV Cache | Local only | Per-node | Distributed |
| Request Routing | LB only | LB + affinity | Consistent hashing |
| Metrics | Basic | Full Prometheus | Aggregated |

---

## Sources

- [Kubernetes Architecture](https://kubernetes.io/docs/concepts/architecture/)
- [Service Networking](https://kubernetes.io/docs/concepts/services-networking/)
- [Pod Lifecycle](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/)
