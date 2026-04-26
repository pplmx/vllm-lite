# Technology Stack: vllm-lite v13.0 Host Deployment

**Project:** vllm-lite Host Deployment Infrastructure
**Researched:** 2026-04-27

## Deployment Stack

### Container Runtime
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Docker | 24+ | Container building | Industry standard, NVIDIA support |
| containerd | 1.7+ | Container runtime | K8s default, better performance than dockershim |

### Kubernetes
| Component | Version | Purpose | Why |
|-----------|---------|---------|-----|
| Kubernetes | 1.28+ | Container orchestration | Required for production deployment |
| NVIDIA GPU Operator | 23.9+ | GPU node management | Enables GPU scheduling in K8s |
| Metrics Server | 0.6+ | Resource metrics | Required for HPA |
| cert-manager | 1.14+ | TLS certificate management | Automated certificate rotation |

### Monitoring
| Technology | Version | Purpose | When to Use |
|------------|---------|---------|-------------|
| Prometheus | 2.48+ | Metrics collection | Always (required for HPA) |
| Grafana | 10+ | Visualization | Optional, for dashboards |
| kube-prometheus | 0.13+ | Full stack | Production deployment |

### Service Networking
| Technology | Purpose | When to Use |
|------------|---------|-------------|
| CoreDNS | Cluster DNS | Always |
| Metallb | Load balancer | Bare metal clusters |
| Cloud LB | Load balancer | Cloud providers (GKE, EKS, AKS) |

## Deployment Tools

### Helm Charts (Recommended)
| Option | Purpose | Notes |
|--------|---------|-------|
| Custom Helm chart | vllm-lite deployment | Build in v13.0 |
| Kustomize | Overlay configuration | Alternative to Helm |

### Configuration
```bash
# Deployment command
helm install vllm-lite ./charts/vllm-lite \
  --set model.path=/models/llama-7b \
  --set engine.max_batch_size=256 \
  --namespace vllm-lite

# Upgrade command
helm upgrade vllm-lite ./charts/vllm-lite \
  --set image.tag=v0.13.0
```

## Supporting Infrastructure

### Required Cluster Components
```yaml
# GPU node requirements
nodeSelector:
  node.kubernetes.io/gpu-type: nvidia
tolerations:
- key: "nvidia.com/gpu"
  operator: "Exists"
  effect: "NoSchedule"
```

### Optional Components
| Component | Purpose | Required for |
|-----------|---------|--------------|
| Istio/Linkerd | Service mesh | Advanced traffic management |
| Vault | Secret management | Enterprise security |
| ArgoCD/Flux | GitOps | Automated deployments |

## Existing Dependencies

vllm-lite already uses:
- `tokio` — Async runtime (K8s-compatible)
- `tracing` — Structured logging (prometheus-metrics compatible)
- `serde` — Serialization (K8s resource compatible)
- `axum` — HTTP server (K8s health probes)

## Installation

```bash
# Build container
docker build -t vllm-lite:v0.13.0 .

# Optional: Push to registry
docker push registry.example.com/vllm-lite:v0.13.0

# Deploy to K8s
kubectl apply -f k8s/manifests/

# Check status
kubectl get pods -n vllm-lite
```

## Sources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/)
- [Prometheus Operator](https://prometheus-operator.dev/)
- [Helm Charts Best Practices](https://helm.sh/docs/chart_best_practices/)
