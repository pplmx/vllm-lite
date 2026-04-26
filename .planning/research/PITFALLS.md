# Domain Pitfalls: vllm-lite v13.0 Host Deployment

**Domain:** LLM Inference Engine Production Deployment
**Researched:** 2026-04-27

## Critical Pitfalls

Mistakes that cause rewrites or major issues.

### Pitfall 1: OOM on GPU Memory Exhaustion

**What goes wrong:** Pod gets OOMKilled, request fails, KV cache lost.

**Why it happens:** LLM inference uses variable memory based on:
- Batch size
- Sequence length
- KV cache block count

Without proper limits, memory grows unbounded until GPU OOM.

**Consequences:**
- Request failure mid-generation
- KV cache loss (regeneration required)
- Pod restart loop (crash → OOM → crash)

**Prevention:**
```yaml
resources:
  limits:
    nvidia.com/gpu: 1
    memory: 32Gi  # Conservative estimate
  requests:
    memory: 16Gi
```

Calculate memory requirements:
```
Model weights (FP16): ~2 bytes per parameter
KV cache per token: ~2 * 2 * layers * hidden_size bytes
Max memory = model_weights + (max_seq_len * kv_per_token * max_batch)
```

**Detection:**
```bash
# Watch GPU memory
nvidia-smi -l 1

# Check pod memory
kubectl top pod vllm-lite-xxx -n vllm-lite
```

---

### Pitfall 2: Model Load Time Blocks Readiness

**What goes wrong:** Pod stays in "NotReady" for minutes during model load. K8s doesn't route traffic, but probes may timeout.

**Why it happens:** Large models (7B+) take 30-60s to load from disk. Readiness probe waits for model load.

**Consequences:**
- Rolling update takes forever (new pods stuck in NotReady)
- Probe timeouts if load exceeds probe settings
- Service degraded during scale-up

**Prevention:**
```yaml
readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 60  # Wait for model load
  periodSeconds: 5
  failureThreshold: 60     # Allow 5 minutes of loading
```

**Alternative:** Split liveness from model readiness:
- Liveness: Always returns OK (process running)
- Readiness: Returns OK when model loaded

**Detection:**
```bash
# Check pod events
kubectl describe pod vllm-lite-xxx -n vllm-lite
# Look for: "Readiness probe failed"
```

---

### Pitfall 3: ConfigMap Not Propagating Changes

**What goes wrong:** ConfigMap updated, pods don't pick up changes.

**Why it happens:** ConfigMap updates don't automatically trigger pod restarts. Pods may use stale config.

**Consequences:**
- Config change has no effect
- Unexpected behavior (e.g., wrong batch size)
- Debugging difficulty

**Prevention:**
- Use `subPath` mounting (NOT recommended, causes restart issues)
- Use ConfigMap reload operator (recommended)
- Implement SIGHUP handling in application

**Recommended pattern:** Annotation-based reload
```rust
// On config change, check annotation and reload
async fn watch_config_map(client: Client) {
    let cm = client.config_maps().get("vllm-lite-config").await;
    let config_hash = cm.annotations().get("config.hash");
    if config_hash != current_hash {
        reload_config().await;
    }
}
```

**Detection:**
```bash
# Check config hash annotation
kubectl get configmap vllm-lite-config -n vllm-lite -o jsonpath='{.metadata.annotations}'
```

---

### Pitfall 4: Distributed Cache Coherence Complexity

**What goes wrong:** Attempting distributed KV cache without proper architecture leads to correctness bugs and performance regression.

**Why it happens:** 
- KV cache coherence requires network round-trips
- Invalidation protocols add latency
- Race conditions in multi-writer scenarios

**Consequences:**
- Incorrect outputs (stale KV data)
- Latency spikes (coherence overhead)
- Complexity explosion (debugging distributed state)

**Prevention:** Defer distributed KV cache to v13.1+. Use per-node caching only.

**Detection:** Monitor cache coherence overhead metrics.

---

## Moderate Pitfalls

### Pitfall 5: Health Probe Port Mismatch

**What goes wrong:** Probe configured on wrong port, always fails or always succeeds.

**Why it happens:** Default port differs from actual port (8000 vs 8080).

**Prevention:**
```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8000  # Must match server config
```

---

### Pitfall 6: Pod Disruption Budget Starvation

**What goes wrong:** PDB blocks all evictions, cluster upgrades stuck.

**Why it happens:** `minAvailable: 50%` prevents evicting pods when at minimum replicas.

**Prevention:**
```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: vllm-lite-pdb
spec:
  minAvailable: 1  # Allow all but 1 to be evicted
  # OR
  maxUnavailable: 1  # Allow 1 pod down at a time
```

---

### Pitfall 7: GPU Node Affinity Issues

**What goes wrong:** Pods unschedulable because GPU nodes have taints.

**Why it happens:** GPU nodes often have `NoSchedule` taints.

**Prevention:**
```yaml
tolerations:
- key: "nvidia.com/gpu"
  operator: "Exists"
  effect: "NoSchedule"
```

---

## Minor Pitfalls

### Pitfall 8: Missing ServiceAccount

**What goes wrong:** Pod can't access K8s APIs (for leader election, metrics, etc.)

**Prevention:**
```yaml
serviceAccountName: vllm-lite
```

### Pitfall 9: ImagePullBackOff

**What goes wrong:** Image not accessible from cluster.

**Prevention:**
- Use public registry or configure imagePullSecrets
- Test `docker pull` from cluster network

### Pitfall 10: Log Volume Pressure

**What goes wrong:** Excessive logging fills ephemeral storage.

**Prevention:**
- Configure log rotation (journald or logrotate)
- Set log level appropriately for production

---

## Phase-Specific Warnings

| Phase | Feature | Pitfall | Mitigation |
|-------|---------|---------|------------|
| v13.0 | K8s Deployment | OOM on GPU | Set resource limits |
| v13.0 | Health Checks | Probe timeout | Increase initialDelaySeconds |
| v13.0 | ConfigMap | Stale config | Implement reload mechanism |
| v13.1 | Multi-Node | Coherence bugs | Defer, use per-node only |
| v13.1 | HPA | Metric missing | Implement /metrics endpoint |
| v13.1 | HA | Failover delay | Pre-warm standby |

---

## Sources

- [Kubernetes Best Practices - Resource Management](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)
- [Pod Lifecycle](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/)
- [Disruption Budgets](https://kubernetes.io/docs/concepts/workloads/pods/disruptions/)
