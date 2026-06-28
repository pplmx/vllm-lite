use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Instant;
use tokio::sync::mpsc;
use vllm_core::types::{Request, SchedulerConfig};
use vllm_testing::TestFixtures;

/// Benchmark per-request end-to-end latency.
///
/// Each iteration of the inner closure drives `num_requests` requests to
/// completion through a freshly-constructed `Engine`, and returns the
/// per-request latency as a `Duration`. Criterion samples many such
/// iterations and reports the empirical p50/p95/p99 distribution
/// automatically.
fn bench_latency_percentiles(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency_percentiles");
    group.sample_size(20);

    for num_requests in &[10, 50] {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_requests),
            num_requests,
            |b, &num_requests| {
                b.iter(|| {
                    let config = SchedulerConfig::default();
                    let mut engine = TestFixtures::increment_engine_with(config, 4, 1024);

                    for i in 0..num_requests {
                        let (tx, _rx) = mpsc::channel(64);
                        engine.add_request(Request::new(i as u64, vec![10, 20], 20), tx);
                    }

                    let start = Instant::now();
                    let mut completed = 0;
                    while completed < num_requests {
                        let results = black_box(engine.step().unwrap());
                        completed += results.len();
                    }

                    start.elapsed() / num_requests as u32
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_latency_percentiles);
criterion_main!(benches);
