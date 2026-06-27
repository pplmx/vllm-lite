use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tokio::sync::mpsc;
use vllm_core::types::{AdaptiveDraftConfig, Request, SchedulerConfig};
use vllm_testing::TestFixtures;

/// SPEC-BENCH-02: Baseline comparison vs non-speculative inference.
/// Reports `baseline` (no spec decode) vs `speculative` (with adaptive spec decode).
fn bench_speculative_vs_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("speculative_vs_baseline");
    group.sample_size(10);

    for num_requests in [10, 50, 100].iter() {
        // Baseline: regular engine (no speculative)
        group.bench_with_input(
            BenchmarkId::new("baseline", num_requests),
            num_requests,
            |b, &num_requests| {
                let config = SchedulerConfig::default();
                let mut engine = TestFixtures::increment_engine_with(config, 4, 1024);
                for i in 0..num_requests {
                    let (tx, _rx) = mpsc::channel(64);
                    engine.add_request(Request::new(i as u64, vec![10, 20], 20), tx);
                }
                b.iter(|| {
                    let mut completed = 0;
                    while completed < num_requests {
                        let results = black_box(engine.step().unwrap());
                        completed += results.len();
                    }
                });
            },
        );

        // Speculative: with adaptive speculative
        group.bench_with_input(
            BenchmarkId::new("speculative", num_requests),
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
                    while completed < num_requests {
                        let results = black_box(engine.step().unwrap());
                        completed += results.len();
                    }
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_speculative_vs_baseline);
criterion_main!(benches);
