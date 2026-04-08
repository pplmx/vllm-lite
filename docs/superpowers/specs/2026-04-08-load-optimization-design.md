# Model Loading Optimization Design

**Date**: 2026-04-08
**Status**: Draft
**Author**: AI Agent
**Reviewer**: 

---

## 1. Overview

This document describes the optimization design for model loading and runtime performance in vllm-lite. The goal is to achieve 50%+ loading speed improvement, 50% memory reduction, and better runtime responsiveness.

## 2. Problem Statement

Current loading implementation has several performance issues:

1. **Double deserialization**: Each safetensors file is read and deserialized twice (once for counting, once for loading)
2. **Serial processing**: Tensors are loaded sequentially, not utilizing multi-core CPUs
3. **Type conversion overhead**: BF16/F16 weights are converted to f32, doubling memory usage
4. **Fixed sleep interval**: Engine loop sleeps 1ms regardless of workload
5. **Lock contention**: Metrics collection uses Mutex, causing contention under high concurrency

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Loading Pipeline (I/O → CPU → Device)               │
├─────────────────────────────────────────────────────────────────────────┤
│  Filesystem              CPU Processing              Target Device      │
│  ─────────               ─────────────               ─────────────      │
│                                                                         │
│  ┌─────────┐        ┌─────────────┐           ┌─────────────┐          │
│  │ mmap    │───────▶│ Deserialize │──────────▶│ BF16 Tensor │          │
│  │ zero-copy│        │ (parallel)  │           │ (native)    │          │
│  └─────────┘        └─────────────┘           └─────────────┘          │
│       │                   │                         │                   │
│       │              ┌────▼────┐                   │                   │
│       │              │ SIMD    │                   │                   │
│       │              │ convert │                   │                   │
│       │              └─────────┘                   │                   │
│       │                                         │                   │
│  ┌────▼────┐                              ┌──────▼──────┐               │
│  │ Prefetch│                              │ Device direct│              │
│  │ thread  │                              │ allocation   │              │
│  └─────────┘                              └──────────────┘               │
└─────────────────────────────────────────────────────────────────────────┘
```

## 4. Optimization Details

### 4.1 Memory-Mapped File Loading (mmap)

**Problem**: Traditional `read()` copies data from disk → kernel space → user space (2 copies)

**Solution**: Use `memmap2` crate for zero-copy file loading with fallback

```rust
use memmap2::Mmap;

const MMAP_THRESHOLD_BYTES: u64 = 100 * 1024 * 1024; // 100MB
const MAX_MMAP_SIZE: u64 = 10 * 1024 * 1024 * 1024; // 10GB max

pub fn load_file(path: &Path) -> Result<Vec<u8>> {
    let metadata = std::fs::metadata(path)?;
    let file_size = metadata.len();
    
    // Use mmap for large files
    if file_size >= MMAP_THRESHOLD_BYTES && file_size <= MAX_MMAP_SIZE {
        match load_mmap(path) {
            Ok(mmap) => {
                // Convert Mmap to Vec<u8> - this copies but is still faster for large files
                // because mmap avoids kernel->user copy on read
                return Ok(mmap.to_vec());
            }
            Err(e) => {
                eprintln!("mmap failed for {}, falling back to read(): {}", path.display(), e);
                // Fall through to read()
            }
        }
    }
    
    // Fallback: traditional read() for small files or when mmap fails
    std::fs::read(path).map_err(|e| format!("read failed: {}", e))
}

pub fn load_mmap(path: &Path) -> Result<Mmap> {
    let file = std::fs::File::open(path)?;
    // UNSAFE: memmap2 handles platform differences safely
    unsafe { Mmap::map(&file) }
        .map_err(|e| format!("mmap failed: {}", e))
}
```

**When to use**: Files > 100MB (controlled by threshold `MMAP_THRESHOLD_BYTES`)

**Fallback strategy**:
- If file < 100MB: use `read()` directly (overhead not worth it)
- If file > 10GB: use `read()` (mmap may exceed address space)
- If mmap fails: silently fallback to `read()`

### 4.2 Parallel Processing Pipeline

**Problem**: Serial loading doesn't utilize multi-core CPUs

**Solution**: Use Rayon for parallel processing with staged pipeline

```rust
use rayon::prelude::*;
use memmap2::Mmap;

pub fn load_weights(&self, model_dir: &str) -> Result<HashMap<String, Tensor>> {
    let files = find_safetensors_files(Path::new(model_dir))?;
    
    // Stage 1: Parallel mmap (I/O bound)
    let mmap_results: Vec<Result<Mmap>> = files
        .par_iter()
        .map(|path| MmapLoader::load_mmap(path))
        .collect();
    
    // Stage 2: Parallel deserialize + convert (CPU bound)
    // Collect to Vec first, then convert to HashMap
    let tensor_vec: Vec<Result<(String, Tensor)>> = mmap_results
        .into_par_iter()
        .map(|mmap_result| {
            let mmap = mmap_result?;
            let file = SafeTensors::deserialize(&mmap)?;
            Ok(file.tensors()
                .into_iter()
                .filter(|(name, _)| !name.contains("visual.") 
                    && !name.contains("vision_") 
                    && !name.contains("img_"))
                .map(|(name, view)| (name, convert_tensor(view, &self.device)?))
                .collect::<Vec<_>>()
        })
        .collect();
    
    // Stage 3: Merge into HashMap (sequential, needs exclusive access)
    let mut weights = HashMap::new();
    for result in tensor_vec {
        let tensors = result?;
        for (name, tensor) in tensors {
            if weights.insert(name.clone(), tensor).is_some() {
                return Err(format!("Duplicate weight '{}'", name));
            }
        }
    }
    
    Ok(weights)
}
```

**Important Notes**:
- Cannot collect directly to `HashMap` in parallel - rayon requires sequential merge
- Use `try_map` or collect `Result` types for error handling
- Filter out vision-related weights early to reduce memory

### 4.3 BF16 Native Support

**Problem**: All weights converted to f32, doubling memory usage

**Solution**: Keep weights in BF16 format when target device supports it

For Qwen2.5-0.5B:
| Format | Weight Memory |
|--------|---------------|
| f32    | ~2GB          |
| bf16   | ~1GB (50% saved) |

**Strategy**:

1. **Try BF16 first**: Attempt to create tensor with BF16 dtype
2. **Fallback to F32**: If Candle doesn't support BF16 for the target device, convert to f32
3. **Configurable**: Add `weight_dtype` option in `ModelConfig`

```rust
use half::{bf16, f16};

fn convert_tensor(&self, view: &TensorView, device: &Device) -> Result<Tensor> {
    use safetensors::Dtype;
    
    let tensor_data: &[u8] = view.data();
    let shape = view.shape().to_vec();
    let dtype = view.dtype();
    
    match dtype {
        Dtype::BF16 => {
            // Use half crate's bf16 - has SIMD support
            let n = tensor_data.len() / 2;
            let data_bf16: &[u16] = unsafe {
                std::slice::from_raw_parts(tensor_data.as_ptr() as *const u16, n)
            };
            // Convert to f32 for Candle (half crate handles SIMD conversion)
            let data_f32: Vec<f32> = data_bf16
                .iter()
                .map(|&bits| bf16::from_bits(bits).to_f32())
                .collect();
            candle_core::Tensor::from_slice(&data_f32, shape, device)?
        }
        Dtype::F16 => {
            // Use half crate's f16 - has SIMD support
            let n = tensor_data.len() / 2;
            let data_f16: &[u16] = unsafe {
                std::slice::from_raw_parts(tensor_data.as_ptr() as *const u16, n)
            };
            let data_f32: Vec<f32> = data_f16
                .iter()
                .map(|&bits| f16::from_bits(bits).to_f32())
                .collect();
            candle_core::Tensor::from_slice(&data_f32, shape, device)?
        }
        Dtype::F32 => {
            let n = tensor_data.len() / 4;
            let data_f32: &[f32] = unsafe {
                std::slice::from_raw_parts(tensor_data.as_ptr() as *const f32, n)
            };
            candle_core::Tensor::from_slice(data_f32, shape, device)?
        }
        _ => return Err(format!("Unsupported dtype: {:?}", dtype)),
    }
}
```

**Note**: The `half` crate (VoidStarKat/half-rs) provides:
- Hardware SIMD support (f16c on x86, fp16 on aarch64)
- Fast conversion via `to_f32()` method
- Already in project dependencies (`half = "2"`)

**Backward Compatibility**: Default to current behavior (f32), add config flag to enable BF16

### 4.4 Adaptive Sleep Policy

**Problem**: Fixed 1ms sleep wastes CPU when idle, adds latency when busy

**Solution**: Exponential backoff based on workload

```rust
pub struct SleepPolicy {
    base_interval: u64,      // 1ms
    max_interval: u64,       // 50ms
    backoff_factor: f64,     // 1.5
    consecutive_idle: u32,
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
        
        ((self.base_interval as f64) * self.backoff_factor.powi(self.consecutive_idle as i32 - 1))
            .min(self.max_interval) as u64
    }
}
```

### 4.5 Lock-Free Metrics

**Problem**: Mutex causes contention under high concurrency

**Solution**: Use `crossbeam::channel` for MPSC (Multi-Producer Single-Consumer)

Note: rtrb is SPSC only, so we use crossbeam which supports MPSC.

```rust
use crossbeam::channel::{Sender, Receiver, bounded};
use std::sync::Arc;

pub struct LockFreeMetrics {
    tokens_total: Arc<AtomicU64>,
    requests_total: Arc<AtomicU64>,
    // MPSC channels - multiple threads can record, one thread reads
    latency_sender: Sender<f64>,
    latency_receiver: Receiver<f64>,
    batch_size_sender: Sender<usize>,
    batch_size_receiver: Receiver<usize>,
}

impl LockFreeMetrics {
    pub fn new(capacity: usize) -> Self {
        let (latency_tx, latency_rx) = bounded(capacity);
        let (batch_tx, batch_rx) = bounded(capacity);
        
        Self {
            tokens_total: Arc::new(AtomicU64::new(0)),
            requests_total: Arc::new(AtomicU64::new(0)),
            latency_sender: latency_tx,
            latency_receiver: latency_rx,
            batch_size_sender: batch_tx,
            batch_size_receiver: batch_rx,
        }
    }
    
    pub fn record_latency(&self, ms: f64) {
        // Non-blocking send, drop if buffer full (acceptable for metrics)
        let _ = self.latency_sender.try_send(ms);
    }
    
    pub fn record_batch_size(&self, size: usize) {
        let _ = self.batch_size_sender.try_send(size);
    }
    
    /// Drain all buffered values and compute statistics
    /// Should be called from a single thread (e.g., metrics endpoint handler)
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
        
        // Calculate percentiles from latencies
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
        let avg_batch = if batch_sizes.is_empty() {
            0.0
        } else {
            batch_sizes.iter().sum::<usize>() as f64 / batch_sizes.len() as f64
        };
        
        let current_batch = batch_sizes.last().copied().unwrap_or(0);
        
        MetricsSnapshot {
            avg_latency_ms: avg_latency,
            p50_latency_ms: p50,
            p90_latency_ms: p90,
            p99_latency_ms: p99,
            avg_batch_size: avg_batch,
            current_batch_size: current_batch,
            // ... other fields from atomic counters
            ..Default::default()
        }
    }
}
```

**Buffer Size**: Use 1024-4096 to balance memory vs. data loss under burst

## 5. Dependencies

```toml
# In crates/model/Cargo.toml
[dependencies]
rayon = "1.10"
memmap2 = "0.9"
half = "2"                 # https://github.com/VoidStarKat/half-rs (f16, bf16 with SIMD)

# In crates/core/Cargo.toml  
[dependencies]
crossbeam = "0.8"

[features]
default = []
cuda-direct = []           # GPU direct read (requires CUDA)
```

## 6. Implementation Order

| Order | Optimization           | Expected Gain | Difficulty |
|-------|----------------------|---------------|------------|
| 1     | mmap loading         | 30%+ (I/O)    | Low        |
| 2     | parallel processing  | Linear scale  | Medium     |
| 3     | BF16 native          | 50% memory    | Medium     |
| 4     | adaptive sleep       | Fast response, low idle CPU | Low |
| 5     | lock-free metrics    | High concurrency | High   |
| 6     | GPU direct read      | PCIe savings  | High       |

## 7. Files to Modify

| File | Changes |
|------|---------|
| `crates/model/Cargo.toml` | Add `rayon`, `memmap2` |
| `crates/core/Cargo.toml` | Add `crossbeam` |
| `crates/model/src/loader.rs` | mmap loading, parallel processing, BF16 support |
| `crates/model/src/config/mod.rs` | Add `weight_dtype` config option |
| `crates/core/src/engine.rs` | Adaptive sleep, add `SleepPolicy` |
| `crates/core/src/metrics.rs` | Replace Mutex with crossbeam channels |

## 8. Prefetch Feature (Future Enhancement)

Not implemented in Phase 1, but architecture supports future addition:

```rust
pub struct ModelCache {
    models: Mutex<HashMap<String, CachedModel>>,
    prefetch_tx: Sender<String>,
}

impl ModelCache {
    pub fn prefetch(&self, model_path: &str) {
        let _ = self.prefetch_tx.send(model_path.to_string());
    }
    
    fn prefetch_worker(&self, model_dir: &str) {
        std::thread::spawn(move || {
            let loader = ModelLoader::new(Device::Cpu);
            // Pre-load to CPU, actual use copies to target device
            let _model = loader.load_model(model_dir, 1024, false);
        });
    }
}
```

## 9. Backward Compatibility

- All changes are internal optimizations
- No API changes
- Optional features for advanced capabilities (SIMD, CUDA direct)

## 10. Testing Plan

### 9.1 Unit Tests

| Component | Test Cases |
|-----------|------------|
| `load_file` | Small file (<100MB), large file (>=100MB), mmap fallback, file not found |
| `load_weights` | Single safetensor, sharded files, duplicate key detection, visual filter |
| `convert_tensor` | BF16 → BF16, BF16 → F32 fallback, F16, F32 |
| `SleepPolicy` | Immediate work, 1 idle tick, exponential backoff, max interval cap |
| `LockFreeMetrics` | Single record, burst records, buffer overflow handling, snapshot drain |

### 9.2 Integration Tests

Add new tests in `crates/model/tests/`:

```rust
// test_load_performance.rs
#[test]
#[ignore = "Requires model files"]
fn test_load_weights_speed() {
    let loader = ModelLoader::new(Device::Cpu);
    let start = std::time::Instant::now();
    let _weights = loader.load_weights("/models/Qwen2.5-0.5B-Instruct").unwrap();
    let elapsed = start.elapsed();
    eprintln!("Load time: {:?}", elapsed);
    assert!(elapsed.as_secs() < 30, "Load should complete in <30s");
}
```

### 9.3 Benchmark

Use existing `criterion` crate (already in dev-dependencies):

```rust
// benches/load_weights.rs
use criterion::*;

fn bench_load_weights(c: &mut Criterion) {
    c.bench_function("load_weights", |b| {
        b.iter(|| {
            let loader = ModelLoader::new(Device::Cpu);
            loader.load_weights("/models/Qwen2.5-0.5B-Instruct").unwrap()
        });
    });
}
```

Run: `cargo bench --bench load_weights`

### 9.4 Memory Profiling

```bash
# Use massif (Valgrind) to profile memory
valgrind --tool=massif ./target/release/vllm-server --model /models/xxx

# Or using /usr/bin/time
/usr/bin/time -v ./target/release/vllm-server --model /models/xxx
```

Check: Total memory usage should be ~50% of current when using BF16

### 9.5 Load Testing

Use `wrk` or `oha` for HTTP load testing:

```bash
# Benchmark with oha
oha -n 10000 -c 100 -m POST \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 100}' \
  http://localhost:8000/v1/completions
```

Verify: Metrics collection doesn't become bottleneck under high QPS

---

## Review History

- 2026-04-08: Initial draft
- 2026-04-08: Fixed parallel loading code, BF16 fallback strategy, crossbeam instead of rtrb, mmap fallback, detailed testing plan

---
