# Production Readiness Execution Design

**Date:** 2025-04-12  
**Status:** Draft  
**Author:** AI Assistant  
**Scope:** Production deployment preparation for vLLM-lite

---

## Executive Summary

This document describes the phased execution plan to make vLLM-lite production-ready. The plan follows a "steady progress" approach suitable for single-developer teams, prioritizing observability, testing, fault tolerance, and deployment infrastructure over a 4-week timeline.

---

## Current State Analysis

### ✅ Existing Infrastructure
- **Core Features**: Adaptive Speculative Decoding, CUDA Graph, Sequence Packing
- **Testing**: 530 tests passing, clippy clean
- **CI/CD**: `just ci` with format, clippy, doc, test
- **Code Quality**: Follows Rust best practices, comprehensive error handling

### ❌ Missing Infrastructure
- **Observability**: No metrics export, no health checks, no tracing
- **Testing**: No E2E tests covering full request lifecycle
- **Fault Tolerance**: No automatic fallback, no circuit breaker
- **Deployment**: No Docker, no K8s configs

---

## Design Philosophy

### Approach: Steady Progress (稳扎稳打)

This approach prioritizes:
1. **Visibility first** - Know what's happening before trying to fix it
2. **Safety net second** - Ensure changes don't break existing functionality
3. **Resilience third** - Handle failures gracefully
4. **Deployment last** - Standardize environment after system is stable

### Principles

| Principle | Description |
|-----------|-------------|
| **Incremental** | Each phase delivers visible value, standalone |
| **Testable** | Every component includes tests |
| **Observable** | Every component exports metrics |
| **Fallback** | Every optimization has a safe fallback |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        vLLM-lite System                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐│
│  │   Engine    │  │  Scheduler  │  │    Model    │  │  Server ││
│  │             │  │             │  │             │  │         ││
│  │  Batching   │  │  Queue Mgmt │  │  Inference  │  │  HTTP   ││
│  │  KV Cache   │  │  Preemption │  │  Forward    │  │  API    ││
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────┬────┘│
│         │                │                │              │     │
│         └────────────────┴────────────────┴──────────────┘     │
│                              │                                 │
│                              ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              EnhancedMetricsCollector                    │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │   │
│  │  │  CUDA Graph │  │   Packing   │  │  Speculative    │  │   │
│  │  │   Metrics   │  │   Metrics   │  │    Metrics      │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                 │
│                              ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Metrics Exporters                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │   │
│  │  │  Prometheus │  │OpenTelemetry│  │     Stdout      │  │   │
│  │  │  /metrics   │  │    Traces   │  │   (dev mode)    │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                 │
│                              ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Circuit Breaker & Fallback                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │   │
│  │  │   Retry     │  │   Degrade   │  │    Fail Fast    │  │   │
│  │  │  Strategy   │  │   Strategy  │  │    Strategy     │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Enhanced Metrics and Tracing (Week 1)

### Goal
Enable visibility into system behavior. Export metrics to Prometheus/OpenTelemetry.

### Components

#### 1.1 EnhancedMetricsCollector

Centralized metrics collection for all optimization components.

```rust
pub struct EnhancedMetricsCollector {
    cuda_graph: CudaGraphMetrics,
    packing: PackingMetrics,
    speculative: SpeculativeMetrics,
    system: SystemMetrics,
}

impl EnhancedMetricsCollector {
    pub fn new() -> Self { /* ... */ }
    pub fn record_cuda_graph_hit(&self) { /* ... */ }
    pub fn record_packing_efficiency(&self, ratio: f64) { /* ... */ }
    pub fn record_speculative_acceptance(&self, accepted: usize, total: usize) { /* ... */ }
}
```

#### 1.2 Metrics Registry

Thread-safe metrics storage using `dashmap` for concurrent access.

```rust
use dashmap::DashMap;
use metrics::{Counter, Gauge, Histogram};

pub struct MetricsRegistry {
    counters: DashMap<String, Counter>,
    gauges: DashMap<String, Gauge>,
    histograms: DashMap<String, Histogram>,
}
```

#### 1.3 Exporters

**Prometheus Exporter:**
```rust
pub struct PrometheusExporter {
    registry: Registry,
    bind_addr: SocketAddr,
}

impl PrometheusExporter {
    pub async fn serve(&self) -> Result<()> {
        // HTTP server on port 9090
        // Serve /metrics endpoint
    }
}
```

**OpenTelemetry Exporter:**
```rust
pub struct OpenTelemetryExporter {
    tracer: Tracer,
    meter: Meter,
}

impl OpenTelemetryExporter {
    pub fn init_tracer(&self) -> Result<()> {
        // Initialize OTel tracer
        // Export to Jaeger/Zipkin
    }
}
```

### Metrics Definition

| Category | Metric Name | Type | Labels | Description |
|----------|-------------|------|--------|-------------|
| CUDA Graph | `cuda_graph_hits_total` | Counter | - | Number of cache hits |
| CUDA Graph | `cuda_graph_misses_total` | Counter | - | Number of cache misses |
| CUDA Graph | `cuda_graph_execution_duration_seconds` | Histogram | - | Execution time |
| Packing | `packing_waste_ratio` | Gauge | - | Waste ratio (0-1) |
| Packing | `packing_efficiency` | Gauge | - | Batch efficiency (0-1) |
| Packing | `packing_sequences_total` | Counter | - | Total sequences packed |
| Speculative | `speculative_acceptance_rate` | Gauge | - | Token acceptance rate |
| Speculative | `speculative_draft_count` | Gauge | - | Current draft tokens |
| Speculative | `speculative_adjustments_total` | Counter | - | Number of adjustments |
| System | `request_queue_depth` | Gauge | - | Pending requests |
| System | `inference_latency_seconds` | Histogram | quantile | P50/P99 latency |
| System | `active_sequences` | Gauge | - | Currently processing |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe |
| `/ready` | GET | Readiness probe |
| `/metrics` | GET | Prometheus metrics |

### Health Check Logic

```rust
pub struct HealthChecker {
    engine: Arc<Engine>,
    model: Arc<dyn ModelBackend>,
}

impl HealthChecker {
    pub fn check_liveness(&self) -> HealthStatus {
        // Process is running
        HealthStatus::Ok
    }
    
    pub fn check_readiness(&self) -> HealthStatus {
        // Engine initialized, model loaded
        if self.engine.is_ready() && self.model.is_ready() {
            HealthStatus::Ok
        } else {
            HealthStatus::NotReady
        }
    }
}
```

### Acceptance Criteria

- [ ] `/health` returns 200 when process is running
- [ ] `/ready` returns 200 only when model is loaded
- [ ] `/metrics` exports all defined metrics in Prometheus format
- [ ] Metrics update within 1 second of events
- [ ] Memory overhead < 5% of total memory
- [ ] Latency overhead < 1ms per request

---

## Phase 2: E2E Integration Tests (Week 1-2)

### Goal
Establish a safety net for refactoring. Test complete request lifecycle.

### Test Architecture

```
tests/e2e/
├── common/
│   ├── mod.rs           # Shared utilities
│   ├── mock_model.rs    # FakeModel wrapper
│   └── assertions.rs      # Custom assertions
├── lifecycle_test.rs    # Complete request lifecycle
├── concurrent_test.rs   # Concurrent request handling
├── error_recovery_test.rs # Error scenarios
└── graceful_shutdown_test.rs # Shutdown behavior
```

### Test Scenarios

#### 2.1 Complete Request Lifecycle

```rust
#[tokio::test]
async fn test_complete_request_lifecycle() {
    // Arrange
    let engine = setup_engine().await;
    let request = generate_test_request(tokens: 100);
    
    // Act
    let seq_id = engine.add_request(request).await.unwrap();
    let output = process_until_complete(&engine, seq_id).await;
    
    // Assert
    assert!(output.tokens.len() > 0);
    assert!(output.finish_reason.is_some());
    assert_eq!(engine.get_status(seq_id).await, Status::Completed);
}
```

#### 2.2 Concurrent Request Handling

```rust
#[tokio::test]
async fn test_concurrent_requests() {
    let engine = setup_engine().await;
    let requests: Vec<_> = (0..100)
        .map(|i| generate_test_request(tokens: 50 + i))
        .collect();
    
    // Spawn concurrent requests
    let handles: Vec<_> = requests
        .into_iter()
        .map(|req| {
            let eng = engine.clone();
            tokio::spawn(async move {
                let id = eng.add_request(req).await.unwrap();
                process_until_complete(&eng, id).await
            })
        })
        .collect();
    
    // All should complete successfully
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }
}
```

#### 2.3 Error Recovery

```rust
#[tokio::test]
async fn test_model_failure_recovery() {
    let engine = setup_engine_with_mock(
        MockModel::new()
            .with_failure_rate(0.1) // 10% failure rate
    ).await;
    
    // Should handle failures gracefully
    let results = process_n_requests(&engine, 100).await;
    let success_count = results.iter().filter(|r| r.is_ok()).count();
    
    // At least 80% should succeed (with retry)
    assert!(success_count >= 80);
}
```

#### 2.4 Graceful Shutdown

```rust
#[tokio::test]
async fn test_graceful_shutdown() {
    let (engine, shutdown_tx) = setup_engine().await;
    
    // Add in-progress requests
    let ids: Vec<_> = (0..10)
        .map(|_| engine.add_request(test_request()).await.unwrap())
        .collect();
    
    // Initiate shutdown
    shutdown_tx.send(()).await.unwrap();
    
    // Wait for completion
    tokio::time::timeout(Duration::from_secs(30), async {
        for id in ids {
            wait_for_completion(&engine, id).await;
        }
    }).await.unwrap();
}
```

### Mock Model Design

```rust
pub struct MockModel {
    inner: FakeModel,
    failure_rate: f64,
    latency_ms: u64,
    failure_sequence: Vec<bool>, // Deterministic failures
}

impl MockModel {
    pub fn builder() -> MockModelBuilder { /* ... */ }
    
    fn should_fail(&self, sequence_id: SeqId) -> bool {
        if let Some(&fail) = self.failure_sequence.get(sequence_id as usize) {
            fail
        } else {
            random() < self.failure_rate
        }
    }
}

impl ModelBackend for MockModel {
    fn forward(&self, batch: &Batch) -> Result<ForwardOutput> {
        // Simulate latency
        thread::sleep(Duration::from_millis(self.latency_ms));
        
        // Check failure
        if self.should_fail(batch.seq_ids[0]) {
            return Err("Simulated failure".into());
        }
        
        self.inner.forward(batch)
    }
}
```

### Acceptance Criteria

- [ ] `lifecycle_test` covers add → process → complete flow
- [ ] `concurrent_test` handles 100 concurrent requests
- [ ] `error_recovery_test` verifies retry and fallback
- [ ] `graceful_shutdown_test` completes in-flight requests
- [ ] All E2E tests run in < 60 seconds
- [ ] No flaky tests (deterministic outcomes)

---

## Phase 3: Error Handling and Fallback (Week 2)

### Goal
System remains operational during partial failures.

### Circuit Breaker Design

```rust
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: Arc<RwLock<CircuitState>>,
    failure_count: AtomicUsize,
    last_failure_time: AtomicInstant,
}

pub struct CircuitBreakerConfig {
    failure_threshold: usize,      // 5
    recovery_timeout: Duration,    // 30s
    half_open_max_calls: usize,   // 3
}

pub enum CircuitState {
    Closed,      // Normal operation
    Open,        // Failing, reject calls
    HalfOpen,    // Testing recovery
}

impl CircuitBreaker {
    pub async fn call<F, T>(&self, operation: F) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        match *self.state.read().await {
            CircuitState::Open => {
                if self.should_attempt_reset() {
                    self.transition_to_half_open().await;
                } else {
                    return Err(CircuitBreakerError::Open);
                }
            }
            _ => {}
        }
        
        match operation.await {
            Ok(result) => {
                self.on_success().await;
                Ok(result)
            }
            Err(e) => {
                self.on_failure().await;
                Err(e)
            }
        }
    }
}
```

### Fallback Strategies

```rust
pub trait FallbackStrategy {
    async fn execute<F, T>(&self, operation: F) -> Result<T>
    where
        F: Future<Output = Result<T>>;
}

pub struct RetryStrategy {
    max_attempts: usize,
    backoff: ExponentialBackoff,
}

impl FallbackStrategy for RetryStrategy {
    async fn execute<F, T>(&self, operation: F) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        let mut attempts = 0;
        loop {
            match operation.await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    attempts += 1;
                    if attempts >= self.max_attempts {
                        return Err(e);
                    }
                    tokio::time::sleep(self.backoff.next()).await;
                }
            }
        }
    }
}

pub struct DegradeStrategy {
    fallback_fn: Box<dyn Fn() -> Result<Output>>,
}

impl FallbackStrategy for DegradeStrategy {
    async fn execute<F, T>(&self, operation: F) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        match operation.await {
            Ok(result) => Ok(result),
            Err(_) => (self.fallback_fn)(),
        }
    }
}
```

### Component-Specific Fallbacks

| Component | Normal Mode | Fallback Mode | Trigger |
|-----------|-------------|---------------|---------|
| CUDA Graph | Graph-optimized execution | Standard execution | Graph capture/execution failure |
| Sequence Packing | Smart packing algorithm | FIFO queue | Packing panic |
| Adaptive Speculative | Dynamic draft count | Fixed 3 drafts | Acceptance rate calculation error |
| Model Inference | Full model | Stub model (for testing) | Model load failure |

### Error Classification

```rust
pub enum ErrorSeverity {
    /// Log and continue
    Warning,
    
    /// Attempt retry with backoff
    Retryable,
    
    /// Switch to fallback mode
    Degradable,
    
    /// Trip circuit breaker
    CircuitBreaker,
    
    /// Log and terminate process
    Fatal,
}

impl From<&EngineError> for ErrorSeverity {
    fn from(error: &EngineError) -> Self {
        match error {
            EngineError::ModelTimeout => ErrorSeverity::Retryable,
            EngineError::CudaGraphFailed => ErrorSeverity::Degradable,
            EngineError::ModelCrashed => ErrorSeverity::CircuitBreaker,
            EngineError::OutOfMemory => ErrorSeverity::Fatal,
            _ => ErrorSeverity::Warning,
        }
    }
}
```

### Recovery Mechanisms

```rust
pub struct RecoveryManager {
    strategies: HashMap<ErrorType, Box<dyn RecoveryStrategy>>,
}

impl RecoveryManager {
    pub async fn handle(&self, error: EngineError) -> RecoveryAction {
        let severity = ErrorSeverity::from(&error);
        
        match severity {
            ErrorSeverity::Retryable => {
                RecoveryAction::Retry { attempts: 3 }
            }
            ErrorSeverity::Degradable => {
                RecoveryAction::Degrade { component: error.component() }
            }
            ErrorSeverity::CircuitBreaker => {
                RecoveryAction::OpenCircuit { component: error.component() }
            }
            _ => RecoveryAction::Propagate,
        }
    }
}
```

### Acceptance Criteria

- [ ] Circuit breaker transitions through Closed → Open → HalfOpen → Closed
- [ ] CUDA Graph failure falls back to standard execution automatically
- [ ] Retry with exponential backoff (max 3 attempts)
- [ ] Degraded mode metrics exported to Prometheus
- [ ] Recovery from degraded mode when error rate drops
- [ ] No data loss during fallback transitions

---

## Phase 4: Docker Deployment (Week 3)

### Goal
Standardize runtime environment. Enable consistent deployment.

### Dockerfile

```dockerfile
# syntax=docker/dockerfile:1

# Stage 1: Build environment
FROM rust:1.75-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests
COPY Cargo.toml Cargo.lock ./
COPY crates/*/Cargo.toml ./crates/

# Build dependencies (cached layer)
RUN mkdir -p crates && \
    for f in crates/*/Cargo.toml; do \
        mkdir -p "$(dirname "$f")/src" && \
        echo "fn main() {}" > "$(dirname "$f")/src/main.rs"; \
    done
RUN cargo build --release --workspace

# Copy actual source
COPY . .

# Build application
RUN cargo build --release --workspace && \
    strip target/release/vllm-server

# Stage 2: Runtime environment
FROM debian:bookworm-slim AS runtime

LABEL maintainer="vLLM-lite Team"
LABEL description="Production vLLM-lite inference server"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -r -s /bin/false -m -d /app vllm

WORKDIR /app

# Copy binary and configs
COPY --from=builder /build/target/release/vllm-server ./
COPY --from=builder /build/config ./config

# Set ownership
RUN chown -R vllm:vllm /app

USER vllm

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 9090

# Run
ENTRYPOINT ["./vllm-server"]
CMD ["--config", "/app/config/production.yaml"]
```

### Docker Compose (Development)

```yaml
version: '3.8'

services:
  vllm:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"  # API
      - "9090:9090"  # Metrics
    volumes:
      - ./models:/models:ro
      - ./config:/app/config:ro
    environment:
      - RUST_LOG=info
      - RUST_BACKTRACE=1
      - CUDA_VISIBLE_DEVICES=0
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
    depends_on:
      - prometheus
      - jaeger
    networks:
      - vllm-net
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    networks:
      - vllm-net

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # UI
      - "4317:4317"    # OTLP
    networks:
      - vllm-net

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - ./config/grafana:/etc/grafana/provisioning:ro
      - grafana-data:/var/lib/grafana
    networks:
      - vllm-net

volumes:
  prometheus-data:
  grafana-data:

networks:
  vllm-net:
    driver: bridge
```

### Kubernetes Manifests (Production)

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: vllm
  labels:
    name: vllm

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: vllm-config
  namespace: vllm
data:
  production.yaml: |
    server:
      host: 0.0.0.0
      port: 8000
    scheduler:
      max_batch_size: 256
      max_tokens_per_batch: 4096
    metrics:
      enabled: true
      port: 9090
    optimizations:
      cuda_graph: true
      sequence_packing: true
      adaptive_speculative: true

---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
  namespace: vllm
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      containers:
        - name: vllm
          image: vllm-lite:latest
          ports:
            - containerPort: 8000
              name: http
            - containerPort: 9090
              name: metrics
          env:
            - name: RUST_LOG
              value: "info"
          volumeMounts:
            - name: config
              mountPath: /app/config
            - name: models
              mountPath: /models
          resources:
            requests:
              memory: "4Gi"
              cpu: "2"
            limits:
              memory: "8Gi"
              cpu: "4"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
      volumes:
        - name: config
          configMap:
            name: vllm-config
        - name: models
          persistentVolumeClaim:
            claimName: vllm-models

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
  namespace: vllm
spec:
  selector:
    app: vllm
  ports:
    - port: 80
      targetPort: 8000
      name: http
    - port: 9090
      targetPort: 9090
      name: metrics
  type: ClusterIP

---
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
  namespace: vllm
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-server
  minReplicas: 1
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: request_queue_depth
        target:
          type: AverageValue
          averageValue: "50"

---
# k8s/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: vllm-models
  namespace: vllm
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 50Gi
  storageClassName: standard
```

### Acceptance Criteria

- [ ] `docker build` completes in < 10 minutes
- [ ] Image size < 500MB
- [ ] `docker-compose up` starts all services
- [ ] Health checks pass within 30 seconds
- [ ] K8s manifests apply without errors
- [ ] Rolling updates complete without downtime
- [ ] Resource limits prevent OOM

---

## Phase 5: CI Performance Regression (Week 4)

### Goal
Automatically detect performance degradation in pull requests.

### Benchmark Suite

```rust
// benches/throughput.rs
use criterion::{criterion_group, criterion_main, Criterion, Throughput};

fn benchmark_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");
    
    for concurrency in [1, 10, 50, 100] {
        group.throughput(Throughput::Elements(concurrency as u64));
        group.bench_with_input(
            format!("concurrent_{}", concurrency),
            &concurrency,
            |b, &conc| {
                b.iter(|| {
                    runtime().block_on(async {
                        run_concurrent_requests(conc).await
                    })
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_throughput);
criterion_main!(benches);
```

### CI Workflow

```yaml
# .github/workflows/benchmark.yml
name: Performance Regression

on:
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install Rust
        uses: dtolnay/rust-action@stable

      - name: Cache cargo
        uses: actions/cache@v3
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target/
          key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}

      - name: Run Benchmarks
        run: |
          cargo bench -- --save-baseline pr

      - name: Checkout Main
        run: |
          git fetch origin main
          git checkout origin/main

      - name: Run Main Benchmarks
        run: |
          cargo bench -- --save-baseline main

      - name: Compare Results
        run: |
          cargo bench -- --baseline main --threshold 10
          echo "| Metric | Main | PR | Change |" >> $GITHUB_STEP_SUMMARY
          echo "|--------|------|-----|--------|" >> $GITHUB_STEP_SUMMARY
          cat target/criterion/report/index.html | grep -o '.*' >> $GITHUB_STEP_SUMMARY

      - name: Upload Results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: target/criterion/
```

### Regression Detection

```rust
pub struct RegressionDetector {
    threshold: f64, // 10%
}

impl RegressionDetector {
    pub fn detect(&self, baseline: &BenchmarkResult, current: &BenchmarkResult) -> Option<Regression> {
        let baseline_throughput = baseline.tokens_per_second();
        let current_throughput = current.tokens_per_second();
        
        let change = (current_throughput - baseline_throughput) / baseline_throughput;
        
        if change < -self.threshold {
            Some(Regression {
                metric: "throughput",
                baseline: baseline_throughput,
                current: current_throughput,
                change,
            })
        } else {
            None
        }
    }
}
```

### Baseline Storage

Store baseline results in GitHub Actions artifacts:

```bash
# Store on main branch builds
if [ "$GITHUB_REF" = "refs/heads/main" ]; then
    cargo bench -- --save-baseline main
    # Upload to artifact storage
fi

# Compare PR against baseline
cargo bench -- --baseline main --threshold 10
```

### Metrics to Track

| Metric | Baseline | Alert Threshold |
|--------|----------|-----------------|
| Throughput (1 concurrent) | >1000 tok/s | -10% |
| Throughput (50 concurrent) | >500 tok/s | -10% |
| P99 Latency | <100ms | +20% |
| Memory Usage | <4GB | +500MB |
| Compilation Time | <5min | +2min |

### Acceptance Criteria

- [ ] Benchmarks run on every PR
- [ ] Results compared to main branch baseline
- [ ] CI fails if throughput regression > 10%
- [ ] Results posted as PR comment
- [ ] Baseline updated on main branch merges
- [ ] Historical trends tracked

---

## Timeline Summary

| Week | Phase | Focus | Deliverables |
|------|-------|-------|--------------|
| 1 | B1 | Observability | /metrics, /health, dashboard |
| 1-2 | D1 | Testing | E2E test suite |
| 2 | B3 | Fault Tolerance | Circuit breaker, fallback |
| 3 | B4 | Deployment | Docker, K8s configs |
| 4 | D2 | CI/CD | Performance regression in CI |

---

## Resource Requirements

### Development
- **Time**: 4 weeks (single developer)
- **Compute**: GPU instance for testing (can use CPU for most work)
- **Storage**: 50GB for models and artifacts

### Production
- **Memory**: 4-8GB per instance
- **CPU**: 2-4 cores
- **GPU**: 1x for inference (optional for initial deployment)
- **Storage**: 50GB for models

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| OpenTelemetry overhead | Medium | Medium | Make optional, use sampling |
| Docker build slow | Medium | Low | Multi-stage, aggressive caching |
| E2E test flaky | High | Medium | Deterministic mocks, avoid time-based |
| Memory leak in metrics | Low | High | Monitor memory, bounded buffers |
| K8s complexity | Medium | Low | Start simple, add complexity later |

---

## Success Criteria

### Quantitative
- [ ] Test coverage > 90%
- [ ] P99 latency < 100ms
- [ ] Availability 99.9%
- [ ] Recovery time < 5s
- [ ] Memory overhead < 5%

### Qualitative
- [ ] Single developer can understand full system
- [ ] New feature takes < 1 week to implement safely
- [ ] Production issues are observable within 30 seconds
- [ ] Deployment requires < 5 commands

---

## Appendix

### A. Dependencies

```toml
[dependencies]
# Metrics
metrics = "0.22"
metrics-exporter-prometheus = "0.13"
opentelemetry = "0.21"
opentelemetry-otlp = "0.14"

# Testing
tokio-test = "0.4"
mockall = "0.12"
criterion = "0.5"

# Error Handling
thiserror = "1.0"
backoff = "0.4"

# Docker
# No runtime dependencies
```

### B. Configuration Schema

```yaml
# config/production.yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4

scheduler:
  max_batch_size: 256
  max_tokens_per_batch: 4096
  max_sequences: 1000

metrics:
  enabled: true
  port: 9090
  export_interval_ms: 1000
  
  prometheus:
    enabled: true
    endpoint: "/metrics"
  
  opentelemetry:
    enabled: false
    endpoint: "http://localhost:4317"
    sample_rate: 0.1

circuit_breaker:
  enabled: true
  failure_threshold: 5
  recovery_timeout_ms: 30000
  half_open_max_calls: 3

fallback:
  cuda_graph: "standard_execution"
  sequence_packing: "fifo"
  adaptive_speculative: "fixed_drafts"

optimizations:
  cuda_graph: true
  sequence_packing: true
  adaptive_speculative: true
```

### C. Monitoring Queries

```promql
# Request rate
rate(vllm_requests_total[5m])

# P99 latency
histogram_quantile(0.99, rate(vllm_inference_latency_seconds_bucket[5m]))

# Error rate
rate(vllm_errors_total[5m]) / rate(vllm_requests_total[5m])

# Queue depth
vllm_request_queue_depth

# Optimization efficiency
vllm_packing_efficiency
vllm_speculative_acceptance_rate
```

---

## Related Documents

- `docs/production-readiness-plan.md` - Original production readiness plan
- `docs/superpowers/plans/2025-04-11-adaptive-speculative-decoding-plan.md` - Optimization details
- `AGENTS.md` - Development guidelines
- `README.md` - Project overview

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2025-04-12 | AI Assistant | Initial version |

---

**End of Document**
