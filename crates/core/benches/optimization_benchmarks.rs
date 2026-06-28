use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::sync::Arc;
use tokio::sync::mpsc;
use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_core::scheduler::SchedulerEngine;
use vllm_core::types::{AdaptiveDraftConfig, Request, SchedulerConfig};
use vllm_testing::TestFixtures;

/// Upper bound on engine.step() calls per b.iter() to prevent infinite loops
/// when step() returns empty results (e.g., engine idle or paused).
const MAX_STEPS_PER_ITER: usize = 10_000;

/// Benchmark Sequence Packing vs FIFO
fn bench_sequence_packing(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequence_packing");
    let metrics = Arc::new(EnhancedMetricsCollector::new());

    for batch_size in &[4, 8, 16] {
        // FIFO baseline
        group.bench_with_input(
            BenchmarkId::new("fifo", batch_size),
            batch_size,
            |b, &batch_size| {
                let config = SchedulerConfig {
                    packing: vllm_core::types::SequencePackingConfig {
                        enabled: false,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                let mut scheduler = SchedulerEngine::new(config, 1024, metrics.clone());

                // Add requests with varying lengths
                for i in 0..batch_size {
                    let len = 100 + (i * 50); // 100, 150, 200, ...
                    scheduler.add_request(Request::new(i as u64, vec![1; len], 10));
                }

                b.iter(|| {
                    black_box(scheduler.build_batch());
                });
            },
        );

        // Packing optimized
        group.bench_with_input(
            BenchmarkId::new("packing", batch_size),
            batch_size,
            |b, &batch_size| {
                let config = SchedulerConfig::default(); // packing enabled by default
                let mut scheduler = SchedulerEngine::new(config, 1024, metrics.clone());

                // Add requests with varying lengths
                for i in 0..batch_size {
                    let len = 100 + (i * 50);
                    scheduler.add_request(Request::new(i as u64, vec![1; len], 10));
                }

                b.iter(|| {
                    black_box(scheduler.build_batch());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Adaptive Speculative Decoding
fn bench_adaptive_speculative(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_speculative");

    // Fixed draft tokens
    group.bench_function("fixed_draft", |b| {
        let config = SchedulerConfig::default();
        let mut engine = TestFixtures::increment_engine_with(config, 4, 1024);

        let (tx, _rx) = mpsc::channel(64);
        engine.add_request(Request::new(1, vec![10, 20], 50), tx);

        b.iter(|| {
            black_box(engine.step().unwrap());
        });
    });

    // Adaptive draft tokens
    group.bench_function("adaptive_draft", |b| {
        let config = SchedulerConfig::default();
        let mut engine = TestFixtures::increment_engine_with(config, 4, 1024);
        engine.enable_adaptive_speculative(AdaptiveDraftConfig::default());

        let (tx, _rx) = mpsc::channel(64);
        engine.add_request(Request::new(1, vec![10, 20], 50), tx);

        b.iter(|| {
            black_box(engine.step().unwrap());
        });
    });

    group.finish();
}

/// Benchmark end-to-end throughput
fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");
    group.sample_size(10);

    for num_requests in &[10, 50, 100] {
        // Baseline: No optimizations
        group.bench_with_input(
            BenchmarkId::new("baseline", num_requests),
            num_requests,
            |b, &num_requests| {
                let config = SchedulerConfig {
                    packing: vllm_core::types::SequencePackingConfig {
                        enabled: false,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                let mut engine = TestFixtures::increment_engine_with(config, 4, 1024);

                for i in 0..num_requests {
                    let (tx, _rx) = mpsc::channel(64);
                    engine.add_request(Request::new(i as u64, vec![10, 20], 20), tx);
                }

                b.iter(|| {
                    let mut completed = 0;
                    for _ in 0..MAX_STEPS_PER_ITER {
                        let results = black_box(engine.step().unwrap());
                        completed += results.len();
                        if completed >= num_requests {
                            break;
                        }
                    }
                    black_box(completed);
                });
            },
        );

        // All optimizations enabled
        group.bench_with_input(
            BenchmarkId::new("optimized", num_requests),
            num_requests,
            |b, &num_requests| {
                let config = SchedulerConfig::default();
                let mut engine = TestFixtures::increment_speculative_engine_with(config, 4, 1024);
                engine.enable_adaptive_speculative(AdaptiveDraftConfig::default());

                for i in 0..num_requests {
                    let (tx, _rx) = mpsc::channel(64);
                    engine.add_request(Request::new(i as u64, vec![10, 20], 20), tx);
                }

                b.iter(|| {
                    let mut completed = 0;
                    for _ in 0..MAX_STEPS_PER_ITER {
                        let results = black_box(engine.step().unwrap());
                        completed += results.len();
                        if completed >= num_requests {
                            break;
                        }
                    }
                    black_box(completed);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_sequence_packing,
    bench_adaptive_speculative,
    bench_throughput
);
criterion_main!(benches);
