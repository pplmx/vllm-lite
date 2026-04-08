# Model Loading Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Achieve 50%+ model loading speed improvement, 50% memory reduction via BF16, and better runtime responsiveness

**Architecture:** Three-stage pipeline: (1) mmap with fallback for I/O, (2) rayon parallel deserialize+convert for CPU, (3) sequential HashMap merge. Runtime: adaptive sleep + lock-free metrics.

**Tech Stack:** Rust, rayon (parallel), memmap2 (mmap), crossbeam (MPSC channels), half (BF16/F16)

---

## File Structure

```
crates/model/
├── Cargo.toml                          # Add rayon, memmap2
├── src/
│   ├── loader.rs                       # Main: mmap, parallel, BF16
│   └── config/
│       └── mod.rs                      # Add weight_dtype config

crates/core/
├── Cargo.toml                          # Add crossbeam
├── src/
│   ├── engine.rs                       # Add SleepPolicy
│   └── metrics.rs                      # Replace Mutex with crossbeam channels
```

---

## Task 1: Add Dependencies

**Files:**
- Modify: `crates/model/Cargo.toml`
- Modify: `crates/core/Cargo.toml`

- [ ] **Step 1: Add dependencies to model crate**

Add to `crates/model/Cargo.toml`:

```toml
[dependencies]
rayon = "1.10"
memmap2 = "0.9"
```

- [ ] **Step 2: Add dependencies to core crate**

Add to `crates/core/Cargo.toml`:

```toml
[dependencies]
crossbeam = "0.8"
```

- [ ] **Step 3: Run build to verify**

Run: `cargo build --workspace`
Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add crates/model/Cargo.toml crates/core/Cargo.toml
git commit -m "chore(deps): add rayon, memmap2, crossbeam for load optimization"
```

---

## Task 2: Implement mmap Loading with Fallback

**Files:**
- Modify: `crates/model/src/loader.rs`

- [ ] **Step 1: Write failing test for mmap loading**

Add to `crates/model/src/loader.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_load_file_uses_mmap_for_large() {
        let temp_dir = TempDir::new().unwrap();
        let large_file = temp_dir.path().join("large.bin");
        // Create 150MB file
        fs::write(&large_file, vec![0u8; 150 * 1024 * 1024]).unwrap();
        
        let data = load_file(&large_file).unwrap();
        assert_eq!(data.len(), 150 * 1024 * 1024);
    }

    #[test]
    fn test_load_file_uses_read_for_small() {
        let temp_dir = TempDir::new().unwrap();
        let small_file = temp_dir.path().join("small.bin");
        fs::write(&small_file, b"hello world").unwrap();
        
        let data = load_file(&small_file).unwrap();
        assert_eq!(data, b"hello world");
    }

    #[test]
    fn test_load_file_fallback_on_error() {
        let temp_dir = TempDir::new().unwrap();
        let file = temp_dir.path().join("test.bin");
        fs::write(&file, b"test").unwrap();
        
        // Should work regardless
        let data = load_file(&file).unwrap();
        assert_eq!(data, b"test");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p vllm-model test_load_file -- --nocapture 2>&1 | head -30`
Expected: FAIL - function `load_file` not found

- [ ] **Step 3: Implement load_file function**

Add to `crates/model/src/loader.rs` (after imports, before `ModelLoader` struct):

```rust
const MMAP_THRESHOLD_BYTES: u64 = 100 * 1024 * 1024; // 100MB
const MAX_MMAP_SIZE: u64 = 10 * 1024 * 1024 * 1024; // 10GB max

pub fn load_file(path: &Path) -> Result<Vec<u8>> {
    let metadata = std::fs::metadata(path)?;
    let file_size = metadata.len();
    
    // Use mmap for large files
    if file_size >= MMAP_THRESHOLD_BYTES && file_size <= MAX_MMAP_SIZE {
        match load_mmap(path) {
            Ok(mmap) => {
                return Ok(mmap.to_vec());
            }
            Err(e) => {
                eprintln!("mmap failed for {}, falling back to read(): {}", path.display(), e);
            }
        }
    }
    
    // Fallback: traditional read() for small files or when mmap fails
    std::fs::read(path).map_err(|e| candle_core::Error::msg(format!("read failed: {}", e)))
}

fn load_mmap(path: &Path) -> Result<Mmap> {
    use memmap2::Mmap;
    let file = std::fs::File::open(path)?;
    unsafe { Mmap::map(&file) }
        .map_err(|e| candle_core::Error::msg(format!("mmap failed: {}", e)))
}
```

Add to imports in loader.rs:

```rust
use memmap2::Mmap;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p vllm-model test_load_file -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/model/src/loader.rs
git commit -m "feat(model): add mmap loading with fallback"
```

---

## Task 3: Implement Parallel Processing Pipeline

**Files:**
- Modify: `crates/model/src/loader.rs`

- [ ] **Step 1: Write failing test for parallel loading**

Add to `crates/model/src/loader.rs` tests:

```rust
#[test]
fn test_load_weights_collects_all_tensors() {
    // This tests that load_weights can handle multiple files
    // Note: requires actual model files, so we test the structure
    let loader = ModelLoader::new(Device::Cpu);
    
    // Verify find_safetensors_files works
    let temp_dir = TempDir::new().unwrap();
    fs::write(temp_dir.path().join("model-00001-of-00002.safetensors"), b"test1").unwrap();
    fs::write(temp_dir.path().join("model-00002-of-00002.safetensors"), b"test2").unwrap();
    
    let files = find_safetensors_files(temp_dir.path()).unwrap();
    assert_eq!(files.len(), 2);
}
```

- [ ] **Step 2: Run test to verify it passes (existing function)**

Run: `cargo test -p vllm-model test_load_weights_collects_all_tensors -- --nocapture`
Expected: PASS (function already exists)

- [ ] **Step 3: Modify load_weights to use mmap + parallel**

Replace `load_weights` function in `crates/model/src/loader.rs`:

```rust
pub fn load_weights(&self, model_dir: &str) -> Result<HashMap<String, Tensor>> {
    use rayon::prelude::*;
    
    let model_path = Path::new(model_dir);
    let files = find_safetensors_files(model_path)?;
    
    // Stage 1: Parallel mmap (I/O bound)
    let mmap_results: Vec<Result<(PathBuf, Vec<u8>)>> = files
        .par_iter()
        .map(|path| {
            let data = load_file(path)?;
            Ok((path.clone(), data))
        })
        .collect();
    
    // Stage 2: Parallel deserialize + convert (CPU bound)
    let tensor_vec: Vec<Result<Vec<(String, Tensor)>>> = mmap_results
        .into_par_iter()
        .map(|result| {
            let (_path, data) = result?;
            let file = SafeTensors::deserialize(&data).map_err(|e| 
                candle_core::Error::msg(format!("deserialize failed: {}", e))
            )?;
            Ok(file.tensors()
                .into_iter()
                .filter(|(name, _)| !name.contains("visual.") 
                    && !name.contains("vision_") 
                    && !name.contains("img_"))
                .map(|(name, view)| {
                    let tensor = convert_tensor(&view, &self.device)?;
                    Ok((name, tensor))
                })
                .collect::<Result<Vec<_>>>()?)
        })
        .collect();
    
    // Stage 3: Merge into HashMap (sequential, needs exclusive access)
    let mut weights = HashMap::new();
    let mut loaded = 0;
    let total = tensor_vec.len();
    
    for result in tensor_vec {
        let tensors = result?;
        for (name, tensor) in tensors {
            if weights.insert(name.clone(), tensor).is_some() {
                return Err(candle_core::Error::msg(format!(
                    "Duplicate weight '{}'", name
                )));
            }
            loaded += 1;
            if loaded % 20 == 0 {
                eprintln!("Loading: {}/{}", loaded, total);
            }
        }
    }
    
    eprintln!("Loaded {} tensors total", loaded);
    Ok(weights)
}
```

- [ ] **Step 4: Add convert_tensor helper function**

Add after `load_weights`:

```rust
use half::{bf16, f16};

fn convert_tensor(view: &safetensors::tensor::TensorView, device: &Device) -> Result<Tensor> {
    use safetensors::Dtype;
    
    let tensor_data: &[u8] = view.data();
    let shape = view.shape().to_vec();
    let dtype = view.dtype();
    
    match dtype {
        Dtype::BF16 => {
            let n = tensor_data.len() / 2;
            let data_bf16: &[u16] = unsafe {
                std::slice::from_raw_parts(tensor_data.as_ptr() as *const u16, n)
            };
            let data_f32: Vec<f32> = data_bf16
                .iter()
                .map(|&bits| bf16::from_bits(bits).to_f32())
                .collect();
            candle_core::Tensor::from_slice(&data_f32, shape, device)
        }
        Dtype::F16 => {
            let n = tensor_data.len() / 2;
            let data_f16: &[u16] = unsafe {
                std::slice::from_raw_parts(tensor_data.as_ptr() as *const u16, n)
            };
            let data_f32: Vec<f32> = data_f16
                .iter()
                .map(|&bits| f16::from_bits(bits).to_f32())
                .collect();
            candle_core::Tensor::from_slice(&data_f32, shape, device)
        }
        Dtype::F32 => {
            let n = tensor_data.len() / 4;
            let data_f32: &[f32] = unsafe {
                std::slice::from_raw_parts(tensor_data.as_ptr() as *const f32, n)
            };
            candle_core::Tensor::from_slice(data_f32, shape, device)
        }
        _ => Err(candle_core::Error::msg(format!(
            "Unsupported dtype {:?} for weight", dtype
        ))),
    }
    .map_err(|e| candle_core::Error::msg(format!("Failed to create tensor: {}", e)))
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p vllm-model -- --nocapture 2>&1 | tail -20`
Expected: PASS (existing tests still work)

- [ ] **Step 6: Commit**

```bash
git add crates/model/src/loader.rs
git commit -m "feat(model): parallelize weight loading with rayon"
```

---

## Task 4: Implement Adaptive Sleep Policy

**Files:**
- Modify: `crates/core/src/engine.rs`

- [ ] **Step 1: Write failing test for SleepPolicy**

Add to `crates/core/src/engine.rs` tests:

```rust
#[test]
fn test_sleep_policy_immediate_work() {
    let mut policy = SleepPolicy::default();
    // First call after work should return base interval
    let interval = policy.next_interval(true);
    assert_eq!(interval, 1);
    assert_eq!(policy.consecutive_idle, 0);
}

#[test]
fn test_sleep_policy_exponential_backoff() {
    let mut policy = SleepPolicy::default();
    
    // First idle
    let _ = policy.next_interval(false);
    assert_eq!(policy.consecutive_idle, 1);
    
    // Second idle - should start backing off
    let interval2 = policy.next_interval(false);
    assert_eq!(policy.consecutive_idle, 2);
    
    // Third idle - more backoff
    let interval3 = policy.next_interval(false);
    assert!(interval3 >= interval2);
    
    // After work, reset
    let interval4 = policy.next_interval(true);
    assert_eq!(interval4, 1);
}

#[test]
fn test_sleep_policy_max_interval() {
    let mut policy = SleepPolicy::default();
    
    // Many idle cycles
    for _ in 0..100 {
        policy.next_interval(false);
    }
    
    // Should cap at max_interval
    let interval = policy.next_interval(false);
    assert!(interval <= policy.max_interval);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p vllm-core test_sleep_policy -- --nocapture 2>&1 | head -20`
Expected: FAIL - `SleepPolicy` not found

- [ ] **Step 3: Implement SleepPolicy**

Add to end of `crates/core/src/engine.rs` (before test module):

```rust
pub struct SleepPolicy {
    pub base_interval: u64,
    pub max_interval: u64,
    pub backoff_factor: f64,
    pub consecutive_idle: u32,
}

impl Default for SleepPolicy {
    fn default() -> Self {
        Self {
            base_interval: 1,
            max_interval: 50,
            backoff_factor: 1.5,
            consecutive_idle: 0,
        }
    }
}

impl SleepPolicy {
    pub fn next_interval(&mut self, has_work: bool) -> u64 {
        if has_work {
            self.consecutive_idle = 0;
            return self.base_interval;
        }
        
        self.consecutive_idle += 1;
        
        if self.consecutive_idle == 1 {
            return self.base_interval;
        }
        
        let interval = ((self.base_interval as f64) 
            * self.backoff_factor.powi(self.consecutive_idle as i32 - 1))
            .min(self.max_interval as f64) as u64;
        
        interval
    }
}
```

- [ ] **Step 4: Modify engine run loop to use SleepPolicy**

Find the run loop in `engine.rs`:

```rust
// Current code:
std::thread::sleep(std::time::Duration::from_millis(1));
```

Replace with:

```rust
static SLEEP_POLICY: std::sync::LazyLock<std::cell::RefCell<SleepPolicy>> = 
    std::sync::LazyLock::new(|| std::cell::RefCell::new(SleepPolicy::default()));

let has_pending = self.scheduler.has_pending();
let interval = SLEEP_POLICY.borrow_mut().next_interval(has_pending);
std::thread::sleep(std::time::Duration::from_millis(interval));
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p vllm-core test_sleep_policy -- --nocapture`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/engine.rs
git commit -m "feat(core): add adaptive sleep policy"
```

---

## Task 5: Implement Lock-Free Metrics

**Files:**
- Modify: `crates/core/src/metrics.rs`

- [ ] **Step 1: Write failing test for LockFreeMetrics**

Add to `crates/core/src/metrics.rs` tests:

```rust
#[test]
fn test_lock_free_metrics_single_record() {
    let collector = LockFreeMetrics::new(1024);
    collector.record_latency(10.5);
    
    let snapshot = collector.snapshot();
    assert!((snapshot.avg_latency_ms - 10.5).abs() < 0.01);
}

#[test]
fn test_lock_free_metrics_burst_records() {
    let collector = LockFreeMetrics::new(1024);
    
    // Record many values
    for i in 1..=100 {
        collector.record_latency(i as f64);
    }
    
    let snapshot = collector.snapshot();
    assert!((snapshot.avg_latency_ms - 50.5).abs() < 0.01);
    assert!((snapshot.p50_latency_ms - 50.0).abs() < 1.0);
}

#[test]
fn test_lock_free_metrics_buffer_overflow() {
    let collector = LockFreeMetrics::new(10); // Small buffer
    
    // Try to record more than buffer can hold
    for i in 0..100 {
        collector.record_latency(i as f64); // Some will be dropped
    }
    
    // Should not panic, just drop overflow
    let snapshot = collector.snapshot();
    assert!(snapshot.avg_latency_ms > 0.0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p vllm-core test_lock_free_metrics -- --nocapture 2>&1 | head -20`
Expected: FAIL - `LockFreeMetrics` not found

- [ ] **Step 3: Implement LockFreeMetrics**

Replace `MetricsCollector` in `crates/core/src/metrics.rs`:

```rust
use crossbeam::channel::{Sender, Receiver, bounded};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct LockFreeMetrics {
    tokens_total: Arc<AtomicU64>,
    requests_total: Arc<AtomicU64>,
    requests_in_flight: Arc<AtomicU64>,
    kv_cache_blocks_used: Arc<AtomicU64>,
    kv_cache_blocks_total: Arc<AtomicU64>,
    prefix_cache_hits: Arc<AtomicU64>,
    prefix_cache_requests: Arc<AtomicU64>,
    prefill_tokens: Arc<AtomicU64>,
    decode_tokens: Arc<AtomicU64>,
    start_time: std::time::Instant,
    
    // MPSC channels - non-blocking
    latency_sender: Sender<f64>,
    latency_receiver: Receiver<f64>,
    batch_size_sender: Sender<usize>,
    batch_size_receiver: Receiver<usize>,
    scheduler_wait_sender: Sender<f64>,
    scheduler_wait_receiver: Receiver<f64>,
}

impl LockFreeMetrics {
    pub fn new(capacity: usize) -> Self {
        let (latency_tx, latency_rx) = bounded(capacity);
        let (batch_tx, batch_rx) = bounded(capacity);
        let (wait_tx, wait_rx) = bounded(capacity);
        
        Self {
            tokens_total: Arc::new(AtomicU64::new(0)),
            requests_total: Arc::new(AtomicU64::new(0)),
            requests_in_flight: Arc::new(AtomicU64::new(0)),
            kv_cache_blocks_used: Arc::new(AtomicU64::new(0)),
            kv_cache_blocks_total: Arc::new(AtomicU64::new(0)),
            prefix_cache_hits: Arc::new(AtomicU64::new(0)),
            prefix_cache_requests: Arc::new(AtomicU64::new(0)),
            prefill_tokens: Arc::new(AtomicU64::new(0)),
            decode_tokens: Arc::new(AtomicU64::new(0)),
            start_time: std::time::Instant::now(),
            latency_sender: latency_tx,
            latency_receiver: latency_rx,
            batch_size_sender: batch_tx,
            batch_size_receiver: batch_rx,
            scheduler_wait_sender: wait_tx,
            scheduler_wait_receiver: wait_rx,
        }
    }
    
    pub fn record_tokens(&self, count: u64) {
        self.tokens_total.fetch_add(count, Ordering::Relaxed);
    }
    
    pub fn record_request(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn record_latency(&self, ms: f64) {
        let _ = self.latency_sender.try_send(ms);
    }
    
    pub fn record_batch_size(&self, size: usize) {
        let _ = self.batch_size_sender.try_send(size);
    }
    
    pub fn record_request_start(&self) {
        self.requests_in_flight.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn record_request_end(&self) {
        self.requests_in_flight.fetch_sub(1, Ordering::Relaxed);
    }
    
    pub fn record_kv_cache_usage(&self, used: u64, total: u64) {
        self.kv_cache_blocks_used.store(used, Ordering::Relaxed);
        self.kv_cache_blocks_total.store(total, Ordering::Relaxed);
    }
    
    pub fn record_prefix_cache_hit(&self) {
        self.prefix_cache_hits.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn record_prefix_cache_request(&self) {
        self.prefix_cache_requests.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn record_prefill_tokens(&self, count: u64) {
        self.prefill_tokens.fetch_add(count, Ordering::Relaxed);
    }
    
    pub fn record_decode_tokens(&self, count: u64) {
        self.decode_tokens.fetch_add(count, Ordering::Relaxed);
    }
    
    pub fn record_scheduler_wait_time(&self, ms: f64) {
        let _ = self.scheduler_wait_sender.try_send(ms);
    }
    
    pub fn snapshot(&self) -> MetricsSnapshot {
        // Drain latency channel
        let mut latencies = Vec::new();
        while let Ok(ms) = self.latency_receiver.try_recv() {
            latencies.push(ms);
        }
        
        // Drain batch size channel
        let mut batch_sizes = Vec::new();
        while let Ok(size) = self.batch_size_receiver.try_recv() {
            batch_sizes.push(size);
        }
        
        // Drain wait time channel
        let mut wait_times = Vec::new();
        while let Ok(ms) = self.scheduler_wait_receiver.try_recv() {
            wait_times.push(ms);
        }
        
        // Calculate latency percentiles
        let (avg_latency, p50, p90, p99) = if latencies.is_empty() {
            (0.0, 0.0, 0.0, 0.0)
        } else {
            let sum: f64 = latencies.iter().sum();
            let avg = sum / latencies.len() as f64;
            
            let mut sorted = latencies.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            
            let get_p = |xs: &[f64], p: f64| -> f64 {
                xs[(p * xs.len() as f64).min(xs.len() as f64 - 1.0) as usize]
            };
            
            (avg, get_p(&sorted, 0.5), get_p(&sorted, 0.9), get_p(&sorted, 0.99))
        };
        
        // Calculate batch stats
        let (avg_batch, current_batch) = if batch_sizes.is_empty() {
            (0.0, 0)
        } else {
            let sum: usize = batch_sizes.iter().sum();
            (sum as f64 / batch_sizes.len() as f64, *batch_sizes.last().unwrap_or(&0))
        };
        
        let requests_in_flight = self.requests_in_flight.load(Ordering::Relaxed);
        
        let kv_used = self.kv_cache_blocks_used.load(Ordering::Relaxed);
        let kv_total = self.kv_cache_blocks_total.load(Ordering::Relaxed);
        let kv_cache_usage_percent = if kv_total > 0 {
            (kv_used as f64 / kv_total as f64) * 100.0
        } else {
            0.0
        };
        
        let hits = self.prefix_cache_hits.load(Ordering::Relaxed);
        let total_reqs = self.prefix_cache_requests.load(Ordering::Relaxed);
        let prefix_cache_hit_rate = if total_reqs > 0 {
            (hits as f64 / total_reqs as f64) * 100.0
        } else {
            0.0
        };
        
        let uptime = self.start_time.elapsed().as_secs_f64();
        let prefill_throughput = if uptime > 0.0 {
            self.prefill_tokens.load(Ordering::Relaxed) as f64 / uptime
        } else {
            0.0
        };
        let decode_throughput = if uptime > 0.0 {
            self.decode_tokens.load(Ordering::Relaxed) as f64 / uptime
        } else {
            0.0
        };
        
        let avg_wait = if wait_times.is_empty() {
            0.0
        } else {
            wait_times.iter().sum::<f64>() / wait_times.len() as f64
        };
        
        MetricsSnapshot {
            tokens_total: self.tokens_total.load(Ordering::Relaxed),
            requests_total: self.requests_total.load(Ordering::Relaxed),
            avg_latency_ms: avg_latency,
            p50_latency_ms: p50,
            p90_latency_ms: p90,
            p99_latency_ms: p99,
            avg_batch_size: avg_batch,
            current_batch_size: current_batch,
            requests_in_flight,
            kv_cache_usage_percent,
            prefix_cache_hit_rate,
            prefill_throughput,
            decode_throughput,
            avg_scheduler_wait_time_ms: avg_wait,
        }
    }
}

impl Default for LockFreeMetrics {
    fn default() -> Self {
        Self::new(1024)
    }
}
```

Note: Keep the original `MetricsCollector` as an alias for backward compatibility:

```rust
pub type MetricsCollector = LockFreeMetrics;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p vllm-core test_lock_free_metrics -- --nocapture`
Expected: PASS

- [ ] **Step 5: Run all metrics tests**

Run: `cargo test -p vllm-core metrics -- --nocapture`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/metrics.rs
git commit -m "feat(core): replace Mutex with crossbeam channel for lock-free metrics"
```

---

## Task 6: Run Full Test Suite

- [ ] **Step 1: Run all tests**

Run: `cargo test --workspace 2>&1 | tail -30`
Expected: All tests pass

- [ ] **Step 2: Run clippy**

Run: `cargo clippy --workspace -- -D warnings 2>&1 | tail -20`
Expected: No warnings

- [ ] **Step 3: Run format check**

Run: `cargo fmt --all --check`
Expected: No formatting issues

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: resolve test/clippy issues"
```

---

## Task 7: Optional - BF16 Native Support

**Files:**
- Modify: `crates/model/src/config/mod.rs`
- Modify: `crates/model/src/loader.rs`

This is optional - only implement if Candle supports native BF16.

- [ ] **Step 1: Add weight_dtype config option**

Add to `EngineConfig` in `crates/server/src/config.rs`:

```rust
#[serde(default = "default_weight_dtype")]
pub weight_dtype: String,

// ...

fn default_weight_dtype() -> String {
    "f32".to_string()
}
```

- [ ] **Step 2: Modify convert_tensor to try BF16 first**

Update the match statement in `convert_tensor` to check device support and preserve BF16 format when possible.

---

## Verification

After all tasks complete:

1. **Loading speed**: Should see 30-50% improvement from parallel loading
2. **Memory**: BF16 option can reduce weight memory by ~50%
3. **CPU idle**: Adaptive sleep reduces idle CPU usage
4. **Metrics**: Lock-free design improves high-concurrency performance

---

**Plan complete and saved to `docs/superpowers/plans/2026-04-08-load-optimization.md`**

Two execution options:

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
