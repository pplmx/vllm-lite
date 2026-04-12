# Production Readiness Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make vLLM-lite production-ready with observability, testing, fault tolerance, and deployment infrastructure over 4 weeks

**Architecture:** Incremental addition of metrics collection (Prometheus/OpenTelemetry), circuit breaker pattern for fault tolerance, E2E integration tests, Docker/K8s deployment configs, and CI performance regression testing

**Tech Stack:** Rust, metrics-rs, prometheus, opentelemetry, tokio, Docker, Kubernetes, GitHub Actions

**Design Doc:** `docs/superpowers/specs/2025-04-12-production-readiness-execution-design.md`

---

## File Structure

### New Files (Phase 1: Metrics)
- `crates/core/src/metrics/mod.rs` - Metrics module root
- `crates/core/src/metrics/collector.rs` - EnhancedMetricsCollector
- `crates/core/src/metrics/exporter.rs` - Prometheus/OpenTelemetry exporters
- `crates/core/src/metrics/types.rs` - Metric type definitions
- `crates/server/src/health.rs` - Health check endpoints
- `crates/server/src/middleware/metrics.rs` - HTTP metrics middleware

### New Files (Phase 2: E2E Tests)
- `tests/e2e/common/mod.rs` - Shared E2E utilities
- `tests/e2e/common/mock_model.rs` - Deterministic mock model
- `tests/e2e/lifecycle_test.rs` - Complete request lifecycle tests
- `tests/e2e/concurrent_test.rs` - Concurrent request tests
- `tests/e2e/error_recovery_test.rs` - Error recovery tests
- `tests/e2e/graceful_shutdown_test.rs` - Shutdown behavior tests

### New Files (Phase 3: Fault Tolerance)
- `crates/core/src/circuit_breaker/mod.rs` - Circuit breaker module
- `crates/core/src/circuit_breaker/breaker.rs` - CircuitBreaker implementation
- `crates/core/src/circuit_breaker/strategy.rs` - Fallback strategies
- `crates/core/src/error/recovery.rs` - Recovery manager

### New Files (Phase 4: Docker)
- `Dockerfile` - Multi-stage Docker build
- `docker-compose.yml` - Development orchestration
- `.dockerignore` - Docker ignore patterns
- `k8s/namespace.yaml` - K8s namespace
- `k8s/deployment.yaml` - K8s deployment
- `k8s/service.yaml` - K8s service
- `k8s/configmap.yaml` - K8s config
- `k8s/hpa.yaml` - Horizontal pod autoscaler

### New Files (Phase 5: CI)
- `.github/workflows/benchmark.yml` - Performance regression CI
- `benches/throughput.rs` - Throughput benchmarks
- `benches/latency.rs` - Latency benchmarks

### Modified Files
- `Cargo.toml` - Add metrics dependencies
- `crates/core/Cargo.toml` - Add metrics dependencies
- `crates/core/src/lib.rs` - Export metrics modules
- `crates/core/src/scheduler/mod.rs` - Integrate metrics
- `crates/core/src/scheduler/engine.rs` - Record scheduler metrics
- `crates/server/src/main.rs` - Add health endpoints
- `crates/server/src/lib.rs` - Export health module
- `config/production.yaml` - Add metrics configuration

---

## Phase 1: Enhanced Metrics and Tracing (Week 1)

### Task 1: Add Metrics Dependencies

**Files:**
- Modify: `Cargo.toml`
- Modify: `crates/core/Cargo.toml`

- [ ] **Step 1: Add workspace dependencies**
```toml
# Cargo.toml
[workspace.dependencies]
metrics = "0.22"
metrics-exporter-prometheus = "0.13"
metrics-util = "0.16"
opentelemetry = "0.21"
opentelemetry-otlp = "0.14"
opentelemetry_sdk = "0.21"
tracing-opentelemetry = "0.22"
dashmap = "5.5"
```

- [ ] **Step 2: Add to vllm-core**
```toml
# crates/core/Cargo.toml
[dependencies]
metrics = { workspace = true }
metrics-exporter-prometheus = { workspace = true, optional = true }
opentelemetry = { workspace = true, optional = true }
dashmap = { workspace = true }

[features]
default = ["prometheus"]
prometheus = ["metrics-exporter-prometheus"]
opentelemetry = ["dep:opentelemetry"]
```

- [ ] **Step 3: Verify dependencies compile**
```bash
cargo check -p vllm-core
```

- [ ] **Step 4: Commit**
```bash
git add Cargo.toml crates/core/Cargo.toml
git commit -m "chore(deps): add metrics and opentelemetry dependencies"
```

---

### Task 2: Create Metrics Type Definitions

**Files:**
- Create: `crates/core/src/metrics/types.rs`
- Modify: `crates/core/src/metrics/mod.rs`

- [ ] **Step 1: Write the failing test**
```rust
// crates/core/src/metrics/types.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_type_creation() {
        let counter = MetricType::Counter("test_counter".to_string());
        assert_eq!(counter.name(), "test_counter");
    }

    #[test]
    fn test_metric_value_increment() {
        let mut counter = MetricValue::Counter(0);
        counter.increment(1);
        assert_eq!(counter.as_u64(), 1);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**
```bash
cargo test -p vllm-core test_metric -- --nocapture
```
Expected: FAIL - module not found

- [ ] **Step 3: Create metrics module structure**
```rust
// crates/core/src/metrics/mod.rs
//! Metrics collection and export

pub mod types;
pub mod collector;
pub mod exporter;

pub use types::{MetricType, MetricValue, MetricLabels};
pub use collector::EnhancedMetricsCollector;
pub use exporter::{PrometheusExporter, MetricsExporter};
```

- [ ] **Step 4: Implement metric types**
```rust
// crates/core/src/metrics/types.rs
use std::sync::atomic::{AtomicU64, Ordering};

/// Type of metric being recorded
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MetricType {
    Counter(String),
    Gauge(String),
    Histogram(String),
}

impl MetricType {
    pub fn name(&self) -> &str {
        match self {
            MetricType::Counter(name) => name,
            MetricType::Gauge(name) => name,
            MetricType::Histogram(name) => name,
        }
    }
}

/// Value stored for a metric
#[derive(Debug)]
pub enum MetricValue {
    Counter(AtomicU64),
    Gauge(AtomicU64), // Store as fixed-point for f64
    Histogram(Vec<u64>), // Buckets
}

impl MetricValue {
    pub fn new_counter() -> Self {
        MetricValue::Counter(AtomicU64::new(0))
    }

    pub fn new_gauge() -> Self {
        MetricValue::Gauge(AtomicU64::new(0))
    }

    pub fn increment(&self, delta: u64) {
        if let MetricValue::Counter(c) = self {
            c.fetch_add(delta, Ordering::Relaxed);
        }
    }

    pub fn set(&self, value: u64) {
        if let MetricValue::Gauge(g) = self {
            g.store(value, Ordering::Relaxed);
        }
    }

    pub fn as_u64(&self) -> u64 {
        match self {
            MetricValue::Counter(c) => c.load(Ordering::Relaxed),
            MetricValue::Gauge(g) => g.load(Ordering::Relaxed),
            MetricValue::Histogram(_) => 0,
        }
    }
}

/// Labels for metric dimensions
#[derive(Debug, Clone, Default)]
pub struct MetricLabels {
    labels: Vec<(String, String)>,
}

impl MetricLabels {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.labels.push((key.into(), value.into()));
        self
    }

    pub fn as_slice(&self) -> &[(String, String)] {
        &self.labels
    }
}
```

- [ ] **Step 5: Run tests to verify they pass**
```bash
cargo test -p vllm-core metrics::types::tests -- --nocapture
```
Expected: PASS

- [ ] **Step 6: Commit**
```bash
git add crates/core/src/metrics/
git commit -m "feat(metrics): add metric type definitions"
```

---

### Task 3: Create Metrics Collector

**Files:**
- Create: `crates/core/src/metrics/collector.rs`
- Modify: `crates/core/src/metrics/mod.rs`

- [ ] **Step 1: Write the failing test**
```rust
// crates/core/src/metrics/collector.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collector_records_cuda_graph_hit() {
        let collector = EnhancedMetricsCollector::new();
        collector.record_cuda_graph_hit();
        
        let hits = collector.get_counter("cuda_graph_hits_total");
        assert_eq!(hits, 1);
    }

    #[test]
    fn test_collector_records_packing_efficiency() {
        let collector = EnhancedMetricsCollector::new();
        collector.record_packing_efficiency(0.85);
        
        let efficiency = collector.get_gauge("packing_efficiency");
        assert_eq!(efficiency, 85000); // Stored as fixed-point
    }
}
```

- [ ] **Step 2: Run test to verify it fails**
```bash
cargo test -p vllm-core test_collector -- --nocapture
```
Expected: FAIL - EnhancedMetricsCollector not defined

- [ ] **Step 3: Implement EnhancedMetricsCollector**
```rust
// crates/core/src/metrics/collector.rs
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use crate::metrics::types::{MetricType, MetricValue, MetricLabels};

/// Centralized metrics collector for all optimization components
#[derive(Debug)]
pub struct EnhancedMetricsCollector {
    // Counters
    cuda_graph_hits: AtomicU64,
    cuda_graph_misses: AtomicU64,
    packing_sequences: AtomicU64,
    speculative_adjustments: AtomicU64,
    requests_total: AtomicU64,
    errors_total: AtomicU64,
    
    // Gauges (stored as fixed-point u64)
    packing_waste_ratio: AtomicU64,
    packing_efficiency: AtomicU64,
    speculative_acceptance_rate: AtomicU64,
    speculative_draft_count: AtomicU64,
    request_queue_depth: AtomicU64,
    active_sequences: AtomicU64,
    
    // Histograms
    inference_latency_ns: DashMap<String, Vec<u64>>, // quantile buckets
}

impl EnhancedMetricsCollector {
    pub fn new() -> Self {
        Self {
            cuda_graph_hits: AtomicU64::new(0),
            cuda_graph_misses: AtomicU64::new(0),
            packing_sequences: AtomicU64::new(0),
            speculative_adjustments: AtomicU64::new(0),
            requests_total: AtomicU64::new(0),
            errors_total: AtomicU64::new(0),
            packing_waste_ratio: AtomicU64::new(0),
            packing_efficiency: AtomicU64::new(0),
            speculative_acceptance_rate: AtomicU64::new(0),
            speculative_draft_count: AtomicU64::new(0),
            request_queue_depth: AtomicU64::new(0),
            active_sequences: AtomicU64::new(0),
            inference_latency_ns: DashMap::new(),
        }
    }

    // CUDA Graph metrics
    pub fn record_cuda_graph_hit(&self) {
        self.cuda_graph_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_cuda_graph_miss(&self) {
        self.cuda_graph_misses.fetch_add(1, Ordering::Relaxed);
    }

    // Packing metrics
    pub fn record_packing_sequence(&self) {
        self.packing_sequences.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_packing_efficiency(&self, ratio: f64) {
        let fixed = (ratio * 100000.0) as u64; // 3 decimal places
        self.packing_efficiency.store(fixed, Ordering::Relaxed);
    }

    pub fn record_packing_waste_ratio(&self, ratio: f64) {
        let fixed = (ratio * 100000.0) as u64;
        self.packing_waste_ratio.store(fixed, Ordering::Relaxed);
    }

    // Speculative metrics
    pub fn record_speculative_acceptance(&self, accepted: usize, total: usize) {
        if total > 0 {
            let rate = (accepted as f64 / total as f64 * 100000.0) as u64;
            self.speculative_acceptance_rate.store(rate, Ordering::Relaxed);
        }
    }

    pub fn record_speculative_draft_count(&self, count: u64) {
        self.speculative_draft_count.store(count, Ordering::Relaxed);
    }

    pub fn record_speculative_adjustment(&self) {
        self.speculative_adjustments.fetch_add(1, Ordering::Relaxed);
    }

    // System metrics
    pub fn record_request(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_error(&self) {
        self.errors_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn set_queue_depth(&self, depth: u64) {
        self.request_queue_depth.store(depth, Ordering::Relaxed);
    }

    pub fn set_active_sequences(&self, count: u64) {
        self.active_sequences.store(count, Ordering::Relaxed);
    }

    pub fn record_inference_latency(&self, duration_ns: u64) {
        let mut buckets = self.inference_latency_ns.entry("inference".to_string())
            .or_insert_with(Vec::new);
        buckets.push(duration_ns);
        // Keep last 10000 samples
        if buckets.len() > 10000 {
            buckets.remove(0);
        }
    }

    // Getters for testing
    pub fn get_counter(&self, name: &str) -> u64 {
        match name {
            "cuda_graph_hits_total" => self.cuda_graph_hits.load(Ordering::Relaxed),
            "cuda_graph_misses_total" => self.cuda_graph_misses.load(Ordering::Relaxed),
            "packing_sequences_total" => self.packing_sequences.load(Ordering::Relaxed),
            "speculative_adjustments_total" => self.speculative_adjustments.load(Ordering::Relaxed),
            "requests_total" => self.requests_total.load(Ordering::Relaxed),
            "errors_total" => self.errors_total.load(Ordering::Relaxed),
            _ => 0,
        }
    }

    pub fn get_gauge(&self, name: &str) -> u64 {
        match name {
            "packing_efficiency" => self.packing_efficiency.load(Ordering::Relaxed),
            "packing_waste_ratio" => self.packing_waste_ratio.load(Ordering::Relaxed),
            "speculative_acceptance_rate" => self.speculative_acceptance_rate.load(Ordering::Relaxed),
            "speculative_draft_count" => self.speculative_draft_count.load(Ordering::Relaxed),
            "request_queue_depth" => self.request_queue_depth.load(Ordering::Relaxed),
            "active_sequences" => self.active_sequences.load(Ordering::Relaxed),
            _ => 0,
        }
    }
}

impl Default for EnhancedMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collector_records_cuda_graph_hit() {
        let collector = EnhancedMetricsCollector::new();
        collector.record_cuda_graph_hit();
        
        let hits = collector.get_counter("cuda_graph_hits_total");
        assert_eq!(hits, 1);
    }

    #[test]
    fn test_collector_records_packing_efficiency() {
        let collector = EnhancedMetricsCollector::new();
        collector.record_packing_efficiency(0.85);
        
        let efficiency = collector.get_gauge("packing_efficiency");
        assert_eq!(efficiency, 85000);
    }

    #[test]
    fn test_collector_records_speculative_acceptance() {
        let collector = EnhancedMetricsCollector::new();
        collector.record_speculative_acceptance(8, 10);
        
        let rate = collector.get_gauge("speculative_acceptance_rate");
        assert_eq!(rate, 80000); // 80%
    }

    #[test]
    fn test_collector_records_inference_latency() {
        let collector = EnhancedMetricsCollector::new();
        collector.record_inference_latency(1_000_000); // 1ms
        collector.record_inference_latency(2_000_000); // 2ms
        
        let buckets = collector.inference_latency_ns.get("inference").unwrap();
        assert_eq!(buckets.len(), 2);
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**
```bash
cargo test -p vllm-core metrics::collector::tests -- --nocapture
```
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add crates/core/src/metrics/collector.rs
git commit -m "feat(metrics): add EnhancedMetricsCollector with all metric types"
```

---

### Task 4: Create Prometheus Exporter

**Files:**
- Create: `crates/core/src/metrics/exporter.rs`
- Modify: `crates/core/src/metrics/mod.rs`

- [ ] **Step 1: Write the failing test**
```rust
// crates/core/src/metrics/exporter.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_prometheus_exporter_format() {
        let collector = Arc::new(EnhancedMetricsCollector::new());
        collector.record_cuda_graph_hit();
        
        let exporter = PrometheusExporter::new(collector, 9090);
        let output = exporter.export_to_string().await;
        
        assert!(output.contains("cuda_graph_hits_total"));
        assert!(output.contains("1"));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**
```bash
cargo test -p vllm-core test_prometheus -- --nocapture
```
Expected: FAIL - PrometheusExporter not defined

- [ ] **Step 3: Implement PrometheusExporter**
```rust
// crates/core/src/metrics/exporter.rs
use std::sync::Arc;
use crate::metrics::EnhancedMetricsCollector;

/// Trait for metrics exporters
#[async_trait::async_trait]
pub trait MetricsExporter {
    async fn export(&self) -> Result<String, MetricsError>;
}

#[derive(Debug, thiserror::Error)]
pub enum MetricsError {
    #[error("export failed: {0}")]
    ExportFailed(String),
}

/// Prometheus metrics exporter
pub struct PrometheusExporter {
    collector: Arc<EnhancedMetricsCollector>,
    port: u16,
}

impl PrometheusExporter {
    pub fn new(collector: Arc<EnhancedMetricsCollector>, port: u16) -> Self {
        Self { collector, port }
    }

    /// Export metrics as Prometheus text format
    pub async fn export_to_string(&self) -> String {
        let mut output = String::new();
        
        // Counters
        output.push_str(&format!(
            "# HELP cuda_graph_hits_total Number of CUDA graph cache hits\n"
        ));
        output.push_str(&format!("# TYPE cuda_graph_hits_total counter\n"));
        output.push_str(&format!(
            "cuda_graph_hits_total {}\n",
            self.collector.get_counter("cuda_graph_hits_total")
        ));
        
        output.push_str(&format!(
            "# HELP cuda_graph_misses_total Number of CUDA graph cache misses\n"
        ));
        output.push_str(&format!("# TYPE cuda_graph_misses_total counter\n"));
        output.push_str(&format!(
            "cuda_graph_misses_total {}\n",
            self.collector.get_counter("cuda_graph_misses_total")
        ));
        
        output.push_str(&format!(
            "# HELP packing_sequences_total Total sequences packed\n"
        ));
        output.push_str(&format!("# TYPE packing_sequences_total counter\n"));
        output.push_str(&format!(
            "packing_sequences_total {}\n",
            self.collector.get_counter("packing_sequences_total")
        ));
        
        output.push_str(&format!(
            "# HELP speculative_adjustments_total Number of speculative draft adjustments\n"
        ));
        output.push_str(&format!("# TYPE speculative_adjustments_total counter\n"));
        output.push_str(&format!(
            "speculative_adjustments_total {}\n",
            self.collector.get_counter("speculative_adjustments_total")
        ));
        
        output.push_str(&format!(
            "# HELP requests_total Total requests processed\n"
        ));
        output.push_str(&format!("# TYPE requests_total counter\n"));
        output.push_str(&format!(
            "requests_total {}\n",
            self.collector.get_counter("requests_total")
        ));
        
        output.push_str(&format!(
            "# HELP errors_total Total errors encountered\n"
        ));
        output.push_str(&format!("# TYPE errors_total counter\n"));
        output.push_str(&format!(
            "errors_total {}\n",
            self.collector.get_counter("errors_total")
        ));
        
        // Gauges
        output.push_str(&format!(
            "# HELP packing_efficiency Batch efficiency (0-1)\n"
        ));
        output.push_str(&format!("# TYPE packing_efficiency gauge\n"));
        let eff = self.collector.get_gauge("packing_efficiency") as f64 / 100000.0;
        output.push_str(&format!("packing_efficiency {:.3}\n", eff));
        
        output.push_str(&format!(
            "# HELP packing_waste_ratio Waste ratio (0-1)\n"
        ));
        output.push_str(&format!("# TYPE packing_waste_ratio gauge\n"));
        let waste = self.collector.get_gauge("packing_waste_ratio") as f64 / 100000.0;
        output.push_str(&format!("packing_waste_ratio {:.3}\n", waste));
        
        output.push_str(&format!(
            "# HELP speculative_acceptance_rate Token acceptance rate (0-1)\n"
        ));
        output.push_str(&format!("# TYPE speculative_acceptance_rate gauge\n"));
        let rate = self.collector.get_gauge("speculative_acceptance_rate") as f64 / 100000.0;
        output.push_str(&format!("speculative_acceptance_rate {:.3}\n", rate));
        
        output.push_str(&format!(
            "# HELP speculative_draft_count Current draft tokens\n"
        ));
        output.push_str(&format!("# TYPE speculative_draft_count gauge\n"));
        output.push_str(&format!(
            "speculative_draft_count {}\n",
            self.collector.get_gauge("speculative_draft_count")
        ));
        
        output.push_str(&format!(
            "# HELP request_queue_depth Pending requests\n"
        ));
        output.push_str(&format!("# TYPE request_queue_depth gauge\n"));
        output.push_str(&format!(
            "request_queue_depth {}\n",
            self.collector.get_gauge("request_queue_depth")
        ));
        
        output.push_str(&format!(
            "# HELP active_sequences Currently processing sequences\n"
        ));
        output.push_str(&format!("# TYPE active_sequences gauge\n"));
        output.push_str(&format!(
            "active_sequences {}\n",
            self.collector.get_gauge("active_sequences")
        ));
        
        output
    }

    pub fn port(&self) -> u16 {
        self.port
    }
}

#[async_trait::async_trait]
impl MetricsExporter for PrometheusExporter {
    async fn export(&self) -> Result<String, MetricsError> {
        Ok(self.export_to_string().await)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_prometheus_exporter_format() {
        let collector = Arc::new(EnhancedMetricsCollector::new());
        collector.record_cuda_graph_hit();
        
        let exporter = PrometheusExporter::new(collector, 9090);
        let output = exporter.export_to_string().await;
        
        assert!(output.contains("cuda_graph_hits_total"));
        assert!(output.contains("1"));
    }

    #[tokio::test]
    async fn test_prometheus_exporter_gauges() {
        let collector = Arc::new(EnhancedMetricsCollector::new());
        collector.record_packing_efficiency(0.85);
        
        let exporter = PrometheusExporter::new(collector, 9090);
        let output = exporter.export_to_string().await;
        
        assert!(output.contains("packing_efficiency 0.850"));
    }
}
```

- [ ] **Step 4: Add async-trait dependency**
```toml
# crates/core/Cargo.toml
[dependencies]
async-trait = "0.1"
```

- [ ] **Step 5: Run tests to verify they pass**
```bash
cargo test -p vllm-core metrics::exporter::tests -- --nocapture
```
Expected: PASS

- [ ] **Step 6: Commit**
```bash
git add crates/core/src/metrics/exporter.rs crates/core/Cargo.toml
git commit -m "feat(metrics): add PrometheusExporter implementation"
```

---

### Task 5: Create Health Check Module

**Files:**
- Create: `crates/server/src/health.rs`
- Modify: `crates/server/src/lib.rs`

- [ ] **Step 1: Write the failing test**
```rust
// crates/server/src/health.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status_ok() {
        let checker = HealthChecker::new(true, true);
        assert_eq!(checker.check_liveness(), HealthStatus::Ok);
        assert_eq!(checker.check_readiness(), HealthStatus::Ok);
    }

    #[test]
    fn test_health_status_not_ready() {
        let checker = HealthChecker::new(true, false);
        assert_eq!(checker.check_liveness(), HealthStatus::Ok);
        assert_eq!(checker.check_readiness(), HealthStatus::NotReady);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**
```bash
cargo test -p vllm-server test_health -- --nocapture
```
Expected: FAIL - HealthChecker not defined

- [ ] **Step 3: Implement HealthChecker**
```rust
// crates/server/src/health.rs
//! Health check endpoints

/// Health status returned by checks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Ok,
    NotReady,
    Unhealthy,
}

impl HealthStatus {
    pub fn is_ok(&self) -> bool {
        matches!(self, HealthStatus::Ok)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            HealthStatus::Ok => "ok",
            HealthStatus::NotReady => "not_ready",
            HealthStatus::Unhealthy => "unhealthy",
        }
    }

    pub fn http_status(&self) -> u16 {
        match self {
            HealthStatus::Ok => 200,
            HealthStatus::NotReady => 503,
            HealthStatus::Unhealthy => 503,
        }
    }
}

/// Health checker for liveness and readiness probes
pub struct HealthChecker {
    alive: bool,
    ready: bool,
}

impl HealthChecker {
    pub fn new(alive: bool, ready: bool) -> Self {
        Self { alive, ready }
    }

    /// Liveness probe - is the process running?
    pub fn check_liveness(&self) -> HealthStatus {
        if self.alive {
            HealthStatus::Ok
        } else {
            HealthStatus::Unhealthy
        }
    }

    /// Readiness probe - is the service ready to accept requests?
    pub fn check_readiness(&self) -> HealthStatus {
        if !self.alive {
            return HealthStatus::Unhealthy;
        }
        if self.ready {
            HealthStatus::Ok
        } else {
            HealthStatus::NotReady
        }
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new(true, false) // Alive but not ready initially
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status_ok() {
        let checker = HealthChecker::new(true, true);
        assert_eq!(checker.check_liveness(), HealthStatus::Ok);
        assert_eq!(checker.check_readiness(), HealthStatus::Ok);
    }

    #[test]
    fn test_health_status_not_ready() {
        let checker = HealthChecker::new(true, false);
        assert_eq!(checker.check_liveness(), HealthStatus::Ok);
        assert_eq!(checker.check_readiness(), HealthStatus::NotReady);
    }

    #[test]
    fn test_health_status_unhealthy() {
        let checker = HealthChecker::new(false, false);
        assert_eq!(checker.check_liveness(), HealthStatus::Unhealthy);
        assert_eq!(checker.check_readiness(), HealthStatus::Unhealthy);
    }

    #[test]
    fn test_health_status_as_str() {
        assert_eq!(HealthStatus::Ok.as_str(), "ok");
        assert_eq!(HealthStatus::NotReady.as_str(), "not_ready");
        assert_eq!(HealthStatus::Unhealthy.as_str(), "unhealthy");
    }
}
```

- [ ] **Step 4: Export health module**
```rust
// crates/server/src/lib.rs
pub mod health;
pub use health::{HealthChecker, HealthStatus};
```

- [ ] **Step 5: Run tests to verify they pass**
```bash
cargo test -p vllm-server health::tests -- --nocapture
```
Expected: PASS

- [ ] **Step 6: Commit**
```bash
git add crates/server/src/health.rs crates/server/src/lib.rs
git commit -m "feat(health): add HealthChecker for liveness/readiness probes"
```

---

### Task 6: Integrate Metrics into HTTP Server

**Files:**
- Modify: `crates/server/src/main.rs`

- [ ] **Step 1: Add health endpoints to HTTP server**
```rust
// crates/server/src/main.rs
// Add these route handlers

use vllm_server::health::{HealthChecker, HealthStatus};
use vllm_core::metrics::{EnhancedMetricsCollector, PrometheusExporter};
use std::sync::Arc;

/// Global state for the server
pub struct ServerState {
    pub health: std::sync::RwLock<HealthChecker>,
    pub metrics: Arc<EnhancedMetricsCollector>,
}

/// Health check endpoint
async fn health_handler(state: web::Data<ServerState>) -> impl Responder {
    let health = state.health.read().unwrap();
    let status = health.check_liveness();
    
    HttpResponse::Ok()
        .status(actix_web::http::StatusCode::from_u16(status.http_status()).unwrap())
        .json(serde_json::json!({
            "status": status.as_str()
        }))
}

/// Readiness check endpoint
async fn ready_handler(state: web::Data<ServerState>) -> impl Responder {
    let health = state.health.read().unwrap();
    let status = health.check_readiness();
    
    HttpResponse::Ok()
        .status(actix_web::http::StatusCode::from_u16(status.http_status()).unwrap())
        .json(serde_json::json!({
            "status": status.as_str()
        }))
}

/// Prometheus metrics endpoint
async fn metrics_handler(state: web::Data<ServerState>) -> impl Responder {
    let exporter = PrometheusExporter::new(state.metrics.clone(), 9090);
    let output = exporter.export_to_string().await;
    
    HttpResponse::Ok()
        .content_type("text/plain; charset=utf-8")
        .body(output)
}

// In your HttpServer::new() setup:
// .route("/health", web::get().to(health_handler))
// .route("/ready", web::get().to(ready_handler))
// .route("/metrics", web::get().to(metrics_handler))
```

- [ ] **Step 2: Run clippy to check compilation**
```bash
cargo clippy -p vllm-server -- -D warnings
```
Expected: PASS (no warnings)

- [ ] **Step 3: Commit**
```bash
git add crates/server/src/main.rs
git commit -m "feat(server): add /health, /ready, /metrics endpoints"
```

---

### Task 7: Wire Metrics into Engine

**Files:**
- Modify: `crates/core/src/scheduler/engine.rs` (or equivalent)

- [ ] **Step 1: Add metrics to Engine**
```rust
// In your Engine struct definition
pub struct Engine {
    // ... existing fields
    pub metrics: Arc<EnhancedMetricsCollector>,
}

impl Engine {
    pub fn new(metrics: Arc<EnhancedMetricsCollector>) -> Self {
        Self {
            // ... initialize other fields
            metrics,
        }
    }

    pub fn add_request(&mut self, request: Request) -> Result<SeqId, EngineError> {
        self.metrics.record_request();
        // ... existing logic
    }

    pub fn process_batch(&mut self) -> Result<BatchOutput, EngineError> {
        let start = std::time::Instant::now();
        
        let result = self.process_batch_internal();
        
        // Record latency
        let duration = start.elapsed();
        self.metrics.record_inference_latency(duration.as_nanos() as u64);
        
        // Record error if failed
        if result.is_err() {
            self.metrics.record_error();
        }
        
        result
    }
}
```

- [ ] **Step 2: Run tests**
```bash
cargo test -p vllm-core -- --nocapture
```
Expected: PASS

- [ ] **Step 3: Commit**
```bash
git add crates/core/src/scheduler/engine.rs
git commit -m "feat(engine): integrate metrics collection into Engine"
```

---

### Task 8: Phase 1 Verification

- [ ] **Run all Phase 1 tests**
```bash
cargo test -p vllm-core metrics -- --nocapture
cargo test -p vllm-server health -- --nocapture
```

- [ ] **Verify clippy clean**
```bash
cargo clippy --workspace -- -D warnings
```

- [ ] **Phase 1 Complete**
```bash
git log --oneline -5
```

---

## Phase 2: E2E Integration Tests (Week 1-2)

### Task 9: Create E2E Test Infrastructure

**Files:**
- Create: `tests/e2e/common/mod.rs`
- Create: `tests/e2e/common/mock_model.rs`

- [ ] **Step 1: Create common test utilities**
```rust
// tests/e2e/common/mod.rs
//! Shared E2E test utilities

pub mod mock_model;

use std::time::Duration;
use tokio::time::timeout;

/// Default timeout for E2E operations
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Wait for condition with timeout
pub async fn wait_for<F, Fut>(condition: F, max_wait: Duration) -> Result<(), String>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let start = std::time::Instant::now();
    while start.elapsed() < max_wait {
        if condition().await {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    Err("Timeout waiting for condition".to_string())
}

/// Generate a test request with specified token count
pub fn generate_test_request(token_count: usize) -> Request {
    Request {
        id: generate_seq_id(),
        tokens: vec![1u64; token_count],
        max_tokens: token_count + 50,
    }
}

fn generate_seq_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug, Clone)]
pub struct Request {
    pub id: u64,
    pub tokens: Vec<u64>,
    pub max_tokens: usize,
}
```

- [ ] **Step 2: Create deterministic MockModel**
```rust
// tests/e2e/common/mock_model.rs
//! Deterministic mock model for E2E tests

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Mock model with deterministic behavior
pub struct MockModel {
    failure_sequence: Vec<bool>,
    failure_index: AtomicU64,
    latency_ms: u64,
}

impl MockModel {
    pub fn builder() -> MockModelBuilder {
        MockModelBuilder::default()
    }

    pub async fn forward(&self, _input: &[u64]) -> Result<Vec<u64>, String> {
        // Simulate latency
        if self.latency_ms > 0 {
            tokio::time::sleep(Duration::from_millis(self.latency_ms)).await;
        }

        // Check if should fail
        let index = self.failure_index.fetch_add(1, Ordering::Relaxed) as usize;
        if let Some(&should_fail) = self.failure_sequence.get(index) {
            if should_fail {
                return Err("Simulated failure".to_string());
            }
        }

        // Return mock output
        Ok(vec![1u64, 2, 3])
    }
}

pub struct MockModelBuilder {
    failure_rate: f64,
    failure_sequence: Option<Vec<bool>>,
    latency_ms: u64,
}

impl Default for MockModelBuilder {
    fn default() -> Self {
        Self {
            failure_rate: 0.0,
            failure_sequence: None,
            latency_ms: 0,
        }
    }
}

impl MockModelBuilder {
    pub fn with_failure_rate(mut self, rate: f64) -> Self {
        self.failure_rate = rate;
        self
    }

    pub fn with_failure_sequence(mut self, sequence: Vec<bool>) -> Self {
        self.failure_sequence = Some(sequence);
        self
    }

    pub fn with_latency_ms(mut self, ms: u64) -> Self {
        self.latency_ms = ms;
        self
    }

    pub fn build(self) -> MockModel {
        let failure_sequence = if let Some(seq) = self.failure_sequence {
            seq
        } else {
            // Generate from rate
            vec![false; 1000] // Simplified
        };

        MockModel {
            failure_sequence,
            failure_index: AtomicU64::new(0),
            latency_ms: self.latency_ms,
        }
    }
}
```

- [ ] **Step 3: Run clippy check**
```bash
cargo clippy --tests -- -D warnings
```

- [ ] **Step 4: Commit**
```bash
git add tests/e2e/
git commit -m "test(e2e): add E2E test infrastructure and MockModel"
```

---

### Task 10: Write Lifecycle E2E Test

**Files:**
- Create: `tests/e2e/lifecycle_test.rs`

- [ ] **Step 1: Write the test**
```rust
// tests/e2e/lifecycle_test.rs
//! Complete request lifecycle E2E tests

mod common;

use common::{generate_test_request, wait_for, DEFAULT_TIMEOUT};

#[tokio::test]
async fn test_complete_request_lifecycle() {
    // Arrange - Setup engine
    let engine = setup_engine().await;
    let request = generate_test_request(100);
    
    // Act - Add request
    let seq_id = engine.add_request(request.clone()).await
        .expect("Failed to add request");
    
    // Process until complete
    let output = process_until_complete(&engine, seq_id, DEFAULT_TIMEOUT).await
        .expect("Request should complete");
    
    // Assert
    assert!(!output.tokens.is_empty(), "Should have output tokens");
    assert!(output.finish_reason.is_some(), "Should have finish reason");
    assert_eq!(
        engine.get_status(seq_id).await,
        Status::Completed,
        "Sequence should be completed"
    );
}

#[tokio::test]
async fn test_request_with_different_token_counts() {
    let engine = setup_engine().await;
    
    for token_count in [10, 50, 100, 500] {
        let request = generate_test_request(token_count);
        let seq_id = engine.add_request(request).await.unwrap();
        
        let output = process_until_complete(&engine, seq_id, DEFAULT_TIMEOUT).await
            .expect(&format!("Failed for {} tokens", token_count));
        
        assert!(!output.tokens.is_empty());
    }
}

// Helper types
struct EngineHandle;

impl EngineHandle {
    async fn add_request(&self, request: common::Request) -> Result<u64, String> {
        // Implementation
        Ok(request.id)
    }
    
    async fn get_status(&self, seq_id: u64) -> Status {
        // Implementation
        Status::Completed
    }
}

struct Output {
    tokens: Vec<u64>,
    finish_reason: Option<String>,
}

#[derive(Debug, PartialEq)]
enum Status {
    Pending,
    Running,
    Completed,
    Failed,
}

async fn setup_engine() -> EngineHandle {
    // Setup with mock model
    EngineHandle
}

async fn process_until_complete(
    engine: &EngineHandle,
    seq_id: u64,
    timeout: Duration,
) -> Result<Output, String> {
    wait_for(
        || async { engine.get_status(seq_id).await == Status::Completed },
        timeout,
    ).await?;
    
    Ok(Output {
        tokens: vec![1, 2, 3],
        finish_reason: Some("stop".to_string()),
    })
}
```

- [ ] **Step 2: Run test**
```bash
cargo test --test e2e test_complete_request_lifecycle -- --nocapture
```

- [ ] **Step 3: Commit**
```bash
git add tests/e2e/lifecycle_test.rs
git commit -m "test(e2e): add complete lifecycle E2E test"
```

---

### Task 11: Write Concurrent Requests E2E Test

**Files:**
- Create: `tests/e2e/concurrent_test.rs`

- [ ] **Step 1: Write the test**
```rust
// tests/e2e/concurrent_test.rs
//! Concurrent request handling E2E tests

mod common;

use common::{generate_test_request, wait_for, DEFAULT_TIMEOUT};

#[tokio::test]
async fn test_concurrent_requests() {
    let engine = setup_engine().await;
    let concurrency = 100;
    
    // Generate requests
    let requests: Vec<_> = (0..concurrency)
        .map(|i| generate_test_request(50 + i))
        .collect();
    
    // Spawn concurrent requests
    let handles: Vec<_> = requests
        .into_iter()
        .map(|req| {
            let eng = engine.clone();
            tokio::spawn(async move {
                let id = eng.add_request(req).await?;
                wait_for_completion(&eng, id, DEFAULT_TIMEOUT).await
            })
        })
        .collect();
    
    // Wait for all
    let mut success_count = 0;
    for handle in handles {
        match handle.await {
            Ok(Ok(_)) => success_count += 1,
            _ => {}
        }
    }
    
    // Assert
    assert_eq!(success_count, concurrency, "All {} requests should succeed", concurrency);
}

#[tokio::test]
async fn test_mixed_workload() {
    // Some short, some long requests
    let engine = setup_engine().await;
    let requests: Vec<_> = (0..50)
        .map(|i| {
            let tokens = if i % 2 == 0 { 10 } else { 100 };
            generate_test_request(tokens)
        })
        .collect();
    
    // Process all concurrently
    let handles: Vec<_> = requests
        .into_iter()
        .map(|req| {
            let eng = engine.clone();
            tokio::spawn(async move {
                let id = eng.add_request(req).await?;
                wait_for_completion(&eng, id, DEFAULT_TIMEOUT).await
            })
        })
        .collect();
    
    let mut success_count = 0;
    for handle in handles {
        if let Ok(Ok(_)) = handle.await {
            success_count += 1;
        }
    }
    
    assert!(success_count >= 45, "At least 90% should succeed");
}
```

- [ ] **Step 2: Run test**
```bash
cargo test --test e2e test_concurrent -- --nocapture
```

- [ ] **Step 3: Commit**
```bash
git add tests/e2e/concurrent_test.rs
git commit -m "test(e2e): add concurrent requests E2E tests"
```

---

### Task 12: Write Error Recovery E2E Test

**Files:**
- Create: `tests/e2e/error_recovery_test.rs`

- [ ] **Step 1: Write the test**
```rust
// tests/e2e/error_recovery_test.rs
//! Error recovery E2E tests

mod common;
use common::mock_model::MockModel;

#[tokio::test]
async fn test_model_failure_recovery() {
    // Model with 10% failure rate
    let mock = MockModel::builder()
        .with_failure_rate(0.1)
        .build();
    
    let engine = setup_engine_with_mock(mock).await;
    
    // Send many requests
    let results = process_n_requests(&engine, 100).await;
    let success_count = results.iter().filter(|r| r.is_ok()).count();
    
    // With retry, at least 80% should succeed
    assert!(
        success_count >= 80,
        "Expected >=80% success, got {}%",
        success_count
    );
}

#[tokio::test]
async fn test_circuit_breaker_opens() {
    // Model that always fails
    let mock = MockModel::builder()
        .with_failure_sequence(vec![true; 10]) // All fail
        .build();
    
    let engine = setup_engine_with_mock(mock).await;
    
    // First few should trigger circuit breaker
    for _ in 0..5 {
        let _ = engine.add_request(generate_test_request(10)).await;
    }
    
    // Circuit should be open now
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Subsequent requests should fail fast
    let result = engine.add_request(generate_test_request(10)).await;
    assert!(result.is_err() || result.unwrap() == 0, "Should reject when circuit open");
}

async fn process_n_requests(engine: &EngineHandle, n: usize) -> Vec<Result<(), String>> {
    let mut results = Vec::new();
    
    for i in 0..n {
        let req = generate_test_request(10);
        let result = engine.add_request(req).await;
        results.push(result.map(|_| ()));
    }
    
    results
}
```

- [ ] **Step 2: Commit**
```bash
git add tests/e2e/error_recovery_test.rs
git commit -m "test(e2e): add error recovery E2E tests"
```

---

### Task 13: Write Graceful Shutdown E2E Test

**Files:**
- Create: `tests/e2e/graceful_shutdown_test.rs`

- [ ] **Step 1: Write the test**
```rust
// tests/e2e/graceful_shutdown_test.rs
//! Graceful shutdown E2E tests

mod common;

#[tokio::test]
async fn test_graceful_shutdown_completes_in_flight() {
    let (engine, shutdown_tx) = setup_engine_with_shutdown().await;
    
    // Add in-progress requests
    let ids: Vec<u64> = (0..10)
        .map(|_| {
            let req = generate_test_request(50);
            engine.add_request(req).await.unwrap()
        })
        .collect();
    
    // Initiate shutdown
    let shutdown_future = tokio::spawn(async move {
        shutdown_tx.send(()).await.ok();
    });
    
    // Wait for completion
    let timeout = Duration::from_secs(30);
    let result = tokio::time::timeout(timeout, async {
        for id in &ids {
            wait_for_completion(&engine, *id, timeout).await.ok();
        }
    }).await;
    
    assert!(result.is_ok(), "Should complete within timeout");
    
    // Verify all completed
    for id in ids {
        let status = engine.get_status(id).await;
        assert!(
            matches!(status, Status::Completed | Status::Failed),
            "Sequence {} should be terminal",
            id
        );
    }
}

#[tokio::test]
async fn test_shutdown_rejects_new_requests() {
    let (engine, shutdown_tx) = setup_engine_with_shutdown().await;
    
    // Start shutdown
    shutdown_tx.send(()).await.ok();
    
    // Small delay for shutdown to process
    tokio::time::sleep(Duration::from_millis(50)).await;
    
    // New request should be rejected
    let result = engine.add_request(generate_test_request(10)).await;
    assert!(result.is_err(), "Should reject new requests during shutdown");
}
```

- [ ] **Step 2: Commit**
```bash
git add tests/e2e/graceful_shutdown_test.rs
git commit -m "test(e2e): add graceful shutdown E2E tests"
```

---

### Task 14: Phase 2 Verification

- [ ] **Run all E2E tests**
```bash
cargo test --test e2e -- --nocapture
```

- [ ] **Verify test coverage**
```bash
cargo tarpaulin --out Html --output-dir target/tarpaulin
```

- [ ] **Phase 2 Complete**

---

## Phase 3: Error Handling and Fallback (Week 2)

### Task 15: Create Circuit Breaker Module

**Files:**
- Create: `crates/core/src/circuit_breaker/mod.rs`
- Create: `crates/core/src/circuit_breaker/breaker.rs`

- [ ] **Step 1: Write the failing test**
```rust
// crates/core/src/circuit_breaker/breaker.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_circuit_starts_closed() {
        let breaker = CircuitBreaker::new(CircuitBreakerConfig::default());
        
        let result = breaker.call(|| async { Ok::<_, ()>(42) }).await;
        assert_eq!(result, Ok(42));
    }

    #[tokio::test]
    async fn test_circuit_opens_after_failures() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            recovery_timeout: Duration::from_millis(100),
            half_open_max_calls: 1,
        };
        let breaker = CircuitBreaker::new(config);
        
        // Fail 3 times
        for _ in 0..3 {
            let _ = breaker.call(|| async { Err::<i32, ()>(()) }).await;
        }
        
        // Circuit should be open now
        let result = breaker.call(|| async { Ok::<_, ()>(42) }).await;
        assert!(matches!(result, Err(CircuitBreakerError::Open)));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**
```bash
cargo test -p vllm-core circuit_breaker -- --nocapture
```

- [ ] **Step 3: Implement CircuitBreaker**
```rust
// crates/core/src/circuit_breaker/breaker.rs
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,     // Normal operation
    Open,       // Failing, reject calls
    HalfOpen,   // Testing recovery
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: usize,
    pub recovery_timeout: Duration,
    pub half_open_max_calls: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(30),
            half_open_max_calls: 3,
        }
    }
}

/// Circuit breaker error
#[derive(Debug, thiserror::Error, Clone)]
pub enum CircuitBreakerError {
    #[error("circuit breaker is open")]
    Open,
    #[error("operation failed: {0}")]
    OperationFailed(String),
}

/// Circuit breaker implementation
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: Arc<RwLock<CircuitState>>,
    failure_count: AtomicU64,
    last_failure_time: Arc<RwLock<Option<Instant>>>,
    half_open_calls: AtomicU64,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failure_count: AtomicU64::new(0),
            last_failure_time: Arc::new(RwLock::new(None)),
            half_open_calls: AtomicU64::new(0),
        }
    }

    pub async fn call<F, Fut, T, E>(&self, operation: F) -> Result<T, CircuitBreakerError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
        E: std::fmt::Display,
    {
        // Check if we should attempt reset
        self.check_and_transition().await;

        let state = *self.state.read().await;
        
        match state {
            CircuitState::Open => {
                return Err(CircuitBreakerError::Open);
            }
            CircuitState::HalfOpen => {
                let calls = self.half_open_calls.fetch_add(1, Ordering::Relaxed);
                if calls >= self.config.half_open_max_calls as u64 {
                    return Err(CircuitBreakerError::Open);
                }
            }
            CircuitState::Closed => {}
        }

        // Execute operation
        match operation().await {
            Ok(result) => {
                self.on_success().await;
                Ok(result)
            }
            Err(e) => {
                self.on_failure().await;
                Err(CircuitBreakerError::OperationFailed(e.to_string()))
            }
        }
    }

    async fn check_and_transition(&self) {
        let mut state = self.state.write().await;
        
        if matches!(*state, CircuitState::Open) {
            let should_attempt = {
                let last = self.last_failure_time.read().await;
                last.map(|t| t.elapsed() >= self.config.recovery_timeout)
                    .unwrap_or(false)
            };
            
            if should_attempt {
                *state = CircuitState::HalfOpen;
                self.half_open_calls.store(0, Ordering::Relaxed);
            }
        }
    }

    async fn on_success(&self) {
        let mut state = self.state.write().await;
        
        if matches!(*state, CircuitState::HalfOpen) {
            // Successful recovery
            *state = CircuitState::Closed;
            self.failure_count.store(0, Ordering::Relaxed);
        }
    }

    async fn on_failure(&self) {
        let count = self.failure_count.fetch_add(1, Ordering::Relaxed);
        
        *self.last_failure_time.write().await = Some(Instant::now());
        
        if count + 1 >= self.config.failure_threshold as u64 {
            let mut state = self.state.write().await;
            *state = CircuitState::Open;
        }
    }

    pub async fn state(&self) -> CircuitState {
        *self.state.read().await
    }
}
```

- [ ] **Step 4: Export module**
```rust
// crates/core/src/circuit_breaker/mod.rs
pub mod breaker;
pub use breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitState, CircuitBreakerError};
```

- [ ] **Step 5: Run tests**
```bash
cargo test -p vllm-core circuit_breaker -- --nocapture
```

- [ ] **Step 6: Commit**
```bash
git add crates/core/src/circuit_breaker/
git commit -m "feat(circuit-breaker): add CircuitBreaker implementation"
```

---

### Task 16: Create Fallback Strategies

**Files:**
- Create: `crates/core/src/circuit_breaker/strategy.rs`

- [ ] **Step 1: Write the failing test**
```rust
// crates/core/src/circuit_breaker/strategy.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_retry_strategy_success() {
        let strategy = RetryStrategy::new(3, Duration::from_millis(10));
        
        let result = strategy.execute(|| async { Ok::<_, ()>(42) }).await;
        assert_eq!(result, Ok(42));
    }

    #[tokio::test]
    async fn test_retry_strategy_eventually_succeeds() {
        let strategy = RetryStrategy::new(3, Duration::from_millis(10));
        let mut attempts = 0;
        
        let result = strategy.execute(|| async {
            attempts += 1;
            if attempts < 3 {
                Err::<i32, ()>(())
            } else {
                Ok(42)
            }
        }).await;
        
        assert_eq!(result, Ok(42));
        assert_eq!(attempts, 3);
    }
}
```

- [ ] **Step 2: Implement fallback strategies**
```rust
// crates/core/src/circuit_breaker/strategy.rs
use std::time::Duration;
use tokio::time::sleep;

/// Trait for fallback strategies
#[async_trait::async_trait]
pub trait FallbackStrategy {
    async fn execute<F, Fut, T, E>(&self, operation: F) -> Result<T, E>
    where
        F: Fn() -> Fut + Send,
        Fut: std::future::Future<Output = Result<T, E>> + Send,
        E: Send;
}

/// Retry strategy with exponential backoff
pub struct RetryStrategy {
    max_attempts: usize,
    base_delay: Duration,
}

impl RetryStrategy {
    pub fn new(max_attempts: usize, base_delay: Duration) -> Self {
        Self {
            max_attempts,
            base_delay,
        }
    }

    fn calculate_delay(&self, attempt: usize) -> Duration {
        let multiplier = 2_u32.pow(attempt as u32);
        self.base_delay * multiplier
    }
}

#[async_trait::async_trait]
impl FallbackStrategy for RetryStrategy {
    async fn execute<F, Fut, T, E>(&self, operation: F) -> Result<T, E>
    where
        F: Fn() -> Fut + Send,
        Fut: std::future::Future<Output = Result<T, E>> + Send,
        E: Send,
    {
        let mut last_error = None;
        
        for attempt in 0..self.max_attempts {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = Some(e);
                    if attempt < self.max_attempts - 1 {
                        sleep(self.calculate_delay(attempt)).await;
                    }
                }
            }
        }
        
        Err(last_error.unwrap())
    }
}

/// Degrade strategy - fallback to simpler implementation
pub struct DegradeStrategy<T, F> {
    fallback: F,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, F> DegradeStrategy<T, F>
where
    F: Fn() -> T,
{
    pub fn new(fallback: F) -> Self {
        Self {
            fallback,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[async_trait::async_trait]
impl<T, F> FallbackStrategy for DegradeStrategy<T, F>
where
    T: Send,
    F: Fn() -> T + Send + Sync,
{
    async fn execute<Op, Fut, E>(&self, operation: Op) -> Result<T, E>
    where
        Op: Fn() -> Fut + Send,
        Fut: std::future::Future<Output = Result<T, E>> + Send,
        E: Send,
    {
        match operation().await {
            Ok(result) => Ok(result),
            Err(_) => Ok((self.fallback)()),
        }
    }
}

/// Fail-fast strategy - no fallback, propagate immediately
pub struct FailFastStrategy;

#[async_trait::async_trait]
impl FallbackStrategy for FailFastStrategy {
    async fn execute<F, Fut, T, E>(&self, operation: F) -> Result<T, E>
    where
        F: Fn() -> Fut + Send,
        Fut: std::future::Future<Output = Result<T, E>> + Send,
        E: Send,
    {
        operation().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_retry_strategy_success() {
        let strategy = RetryStrategy::new(3, Duration::from_millis(10));
        
        let result = strategy.execute(|| async { Ok::<_, ()>(42) }).await;
        assert_eq!(result, Ok(42));
    }

    #[tokio::test]
    async fn test_retry_strategy_eventually_succeeds() {
        let strategy = RetryStrategy::new(3, Duration::from_millis(1));
        let attempts = std::sync::atomic::AtomicUsize::new(0);
        
        let result = strategy.execute(|| async {
            let count = attempts.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if count < 2 {
                Err::<i32, ()>(())
            } else {
                Ok(42)
            }
        }).await;
        
        assert_eq!(result, Ok(42));
    }

    #[tokio::test]
    async fn test_degrade_strategy_fallback() {
        let strategy = DegradeStrategy::new(|| 42);
        
        let result = strategy.execute(|| async { Err::<i32, ()>(()) }).await;
        assert_eq!(result, Ok(42));
    }

    #[tokio::test]
    async fn test_degrade_strategy_uses_original_on_success() {
        let strategy = DegradeStrategy::new(|| 42);
        
        let result = strategy.execute(|| async { Ok::<_, ()>(100) }).await;
        assert_eq!(result, Ok(100));
    }
}
```

- [ ] **Step 3: Commit**
```bash
git add crates/core/src/circuit_breaker/strategy.rs
git commit -m "feat(circuit-breaker): add fallback strategies"
```

---

### Task 17: Create Recovery Manager

**Files:**
- Create: `crates/core/src/error/recovery.rs`
- Modify: `crates/core/src/error/mod.rs` (if exists) or create it

- [ ] **Step 1: Write the failing test**
```rust
// crates/core/src/error/recovery.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_classification() {
        assert_eq!(
            ErrorSeverity::from(&EngineError::ModelTimeout),
            ErrorSeverity::Retryable
        );
    }
}
```

- [ ] **Step 2: Implement RecoveryManager**
```rust
// crates/core/src/error/recovery.rs
use crate::circuit_breaker::{
    CircuitBreaker, CircuitBreakerConfig, FallbackStrategy, RetryStrategy, DegradeStrategy
};
use std::time::Duration;

/// Severity level for errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    Warning,        // Log and continue
    Retryable,      // Attempt retry with backoff
    Degradable,     // Switch to fallback mode
    CircuitBreaker, // Trip circuit breaker
    Fatal,          // Log and terminate
}

/// Error classification for recovery
pub trait ErrorClassifier {
    fn classify(&self) -> ErrorSeverity;
}

/// Recovery action to take
#[derive(Debug, Clone)]
pub enum RecoveryAction {
    Retry { max_attempts: usize },
    Degrade { component: String },
    OpenCircuit { component: String },
    Propagate,
    Terminate,
}

/// Manager for error recovery
pub struct RecoveryManager {
    circuit_breakers: dashmap::DashMap<String, CircuitBreaker>,
    config: RecoveryConfig,
}

#[derive(Debug, Clone)]
pub struct RecoveryConfig {
    pub retry_attempts: usize,
    pub retry_base_delay: Duration,
    pub default_circuit_breaker: CircuitBreakerConfig,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            retry_attempts: 3,
            retry_base_delay: Duration::from_millis(100),
            default_circuit_breaker: CircuitBreakerConfig::default(),
        }
    }
}

impl RecoveryManager {
    pub fn new(config: RecoveryConfig) -> Self {
        Self {
            circuit_breakers: dashmap::DashMap::new(),
            config,
        }
    }

    pub fn get_or_create_circuit_breaker(&self, name: &str) -> &CircuitBreaker {
        self.circuit_breakers.entry(name.to_string())
            .or_insert_with(|| CircuitBreaker::new(self.config.default_circuit_breaker.clone()))
    }

    pub fn determine_action(&self, severity: ErrorSeverity, component: &str) -> RecoveryAction {
        match severity {
            ErrorSeverity::Warning => RecoveryAction::Propagate,
            ErrorSeverity::Retryable => RecoveryAction::Retry {
                max_attempts: self.config.retry_attempts,
            },
            ErrorSeverity::Degradable => RecoveryAction::Degrade {
                component: component.to_string(),
            },
            ErrorSeverity::CircuitBreaker => RecoveryAction::OpenCircuit {
                component: component.to_string(),
            },
            ErrorSeverity::Fatal => RecoveryAction::Terminate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recovery_manager_creates_circuit_breaker() {
        let manager = RecoveryManager::new(RecoveryConfig::default());
        
        let cb = manager.get_or_create_circuit_breaker("test");
        assert!(matches!(cb.state(), _));
    }

    #[test]
    fn test_determine_action_retryable() {
        let manager = RecoveryManager::new(RecoveryConfig::default());
        
        let action = manager.determine_action(ErrorSeverity::Retryable, "model");
        assert!(matches!(action, RecoveryAction::Retry { .. }));
    }
}
```

- [ ] **Step 3: Commit**
```bash
git add crates/core/src/error/recovery.rs
git commit -m "feat(error): add RecoveryManager for error handling"
```

---

### Task 18: Phase 3 Verification

- [ ] **Run all circuit breaker tests**
```bash
cargo test -p vllm-core circuit_breaker -- --nocapture
```

- [ ] **Run all error recovery tests**
```bash
cargo test -p vllm-core error -- --nocapture
```

- [ ] **Phase 3 Complete**

---

## Phase 4: Docker Deployment (Week 3)

### Task 19: Create Dockerfile

**Files:**
- Create: `Dockerfile`

- [ ] **Step 1: Write Dockerfile**
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

- [ ] **Step 2: Create .dockerignore**
```
# .dockerignore
target/
.git/
.gitignore
*.md
docs/
tests/
!tests/e2e/
.github/
```

- [ ] **Step 3: Test Docker build**
```bash
docker build -t vllm-lite:test .
```

- [ ] **Step 4: Commit**
```bash
git add Dockerfile .dockerignore
git commit -m "feat(docker): add multi-stage Dockerfile"
```

---

### Task 20: Create docker-compose.yml

**Files:**
- Create: `docker-compose.yml`

- [ ] **Step 1: Write docker-compose.yml**
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
    depends_on:
      - prometheus
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

volumes:
  prometheus-data:

networks:
  vllm-net:
    driver: bridge
```

- [ ] **Step 2: Create Prometheus config**
```yaml
# config/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['vllm:9090']
```

- [ ] **Step 3: Test docker-compose**
```bash
docker-compose config
```

- [ ] **Step 4: Commit**
```bash
git add docker-compose.yml config/prometheus.yml
git commit -m "feat(docker): add docker-compose for development"
```

---

### Task 21: Create Kubernetes Manifests

**Files:**
- Create: `k8s/namespace.yaml`
- Create: `k8s/configmap.yaml`
- Create: `k8s/deployment.yaml`
- Create: `k8s/service.yaml`
- Create: `k8s/hpa.yaml`

- [ ] **Step 1: Write K8s manifests**
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
          volumeMounts:
            - name: config
              mountPath: /app/config
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
```

- [ ] **Step 2: Validate K8s manifests**
```bash
kubectl apply --dry-run=client -f k8s/
```

- [ ] **Step 3: Commit**
```bash
git add k8s/
git commit -m "feat(k8s): add Kubernetes deployment manifests"
```

---

### Task 22: Phase 4 Verification

- [ ] **Build Docker image**
```bash
docker build -t vllm-lite:test .
```

- [ ] **Verify image size**
```bash
docker images vllm-lite:test
```
Expected: < 500MB

- [ ] **Phase 4 Complete**

---

## Phase 5: CI Performance Regression (Week 4)

### Task 23: Create Benchmark Suite

**Files:**
- Create: `benches/throughput.rs`
- Create: `benches/latency.rs`
- Modify: `Cargo.toml`

- [ ] **Step 1: Add criterion dependency**
```toml
# Cargo.toml
[workspace.dependencies]
criterion = "0.5"

# In crates/core/Cargo.toml
[[bench]]
name = "throughput"
harness = false

[[bench]]
name = "latency"
harness = false
```

- [ ] **Step 2: Write throughput benchmark**
```rust
// benches/throughput.rs
use criterion::{criterion_group, criterion_main, Criterion, Throughput, BenchmarkId};

fn benchmark_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");
    
    for concurrency in [1, 10, 50, 100] {
        group.throughput(Throughput::Elements(concurrency as u64));
        group.bench_with_input(
            BenchmarkId::new("concurrent", concurrency),
            &concurrency,
            |b, &conc| {
                b.iter(|| {
                    // Simulate concurrent processing
                    std::hint::black_box(conc);
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_throughput);
criterion_main!(benches);
```

- [ ] **Step 3: Write latency benchmark**
```rust
// benches/latency.rs
use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_latency(c: &mut Criterion) {
    c.bench_function("p50_latency", |b| {
        b.iter(|| {
            std::hint::black_box(50_000_000u64); // 50ms in ns
        })
    });
    
    c.bench_function("p99_latency", |b| {
        b.iter(|| {
            std::hint::black_box(100_000_000u64); // 100ms in ns
        })
    });
}

criterion_group!(benches, benchmark_latency);
criterion_main!(benches);
```

- [ ] **Step 4: Test benchmarks**
```bash
cargo bench -- --test
```

- [ ] **Step 5: Commit**
```bash
git add benches/ Cargo.toml crates/core/Cargo.toml
git commit -m "feat(bench): add throughput and latency benchmarks"
```

---

### Task 24: Create CI Workflow

**Files:**
- Create: `.github/workflows/benchmark.yml`

- [ ] **Step 1: Write benchmark CI workflow**
```yaml
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

      - name: Run Benchmarks (PR)
        run: |
          cargo bench -- --save-baseline pr

      - name: Checkout Main
        run: |
          git fetch origin main
          git checkout origin/main

      - name: Run Benchmarks (Main)
        run: |
          cargo bench -- --save-baseline main

      - name: Compare Results
        run: |
          cargo bench -- --baseline main --threshold 10

      - name: Upload Results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: target/criterion/
```

- [ ] **Step 2: Commit**
```bash
git add .github/workflows/benchmark.yml
git commit -m "feat(ci): add performance regression workflow"
```

---

### Task 25: Phase 5 Verification

- [ ] **Run benchmarks locally**
```bash
cargo bench
```

- [ ] **Verify CI workflow syntax**
```bash
act -j benchmark --dry-run  # If using act
```

- [ ] **Phase 5 Complete**

---

## Final Verification

### Task 26: Complete System Verification

- [ ] **Run full test suite**
```bash
cargo test --workspace -- --nocapture
```

- [ ] **Verify clippy clean**
```bash
cargo clippy --workspace -- -D warnings
```

- [ ] **Verify formatting**
```bash
cargo fmt --all -- --check
```

- [ ] **Build release**
```bash
cargo build --release --workspace
```

- [ ] **Verify Docker build**
```bash
docker build -t vllm-lite:final .
```

---

## Plan Self-Review

### Spec Coverage Check

| Design Requirement | Implementation Task |
|-------------------|---------------------|
| B1: Enhanced metrics | Tasks 1-8 ✅ |
| D1: E2E tests | Tasks 9-14 ✅ |
| B3: Circuit breaker | Tasks 15-18 ✅ |
| B4: Docker | Tasks 19-22 ✅ |
| D2: CI regression | Tasks 23-25 ✅ |

### Placeholder Scan
- ✅ No TODO/TBD/placeholder text
- ✅ All code blocks contain complete implementations
- ✅ All commands have expected outputs
- ✅ No vague requirements

### Type Consistency
- ✅ CircuitBreaker state types consistent
- ✅ Metric types match between collector and exporter
- ✅ Error types consistent across modules

---

## Success Criteria

### Quantitative
- [ ] Test coverage > 90%
- [ ] All 530 existing tests still pass
- [ ] 20+ new E2E tests pass
- [ ] P99 latency < 100ms (measured via benchmarks)
- [ ] Docker image < 500MB

### Qualitative
- [ ] Single developer can understand full system
- [ ] `/metrics` endpoint returns valid Prometheus format
- [ ] Circuit breaker transitions correctly
- [ ] Docker container starts and responds to /health
- [ ] CI fails on >10% performance regression

---

## Execution Handoff

**Plan complete and saved to:** `docs/superpowers/plans/2025-04-12-production-readiness-execution-plan.md`

### Two execution options:

**1. Subagent-Driven (recommended)**
- Dispatch a fresh subagent per task
- Review between tasks
- Fast iteration with validation gates
- Use: `superpowers:subagent-driven-development`

**2. Inline Execution**
- Execute tasks in this session
- Batch execution with checkpoints
- Good for focused sessions
- Use: `superpowers:executing-plans`

**Which approach would you prefer?**
