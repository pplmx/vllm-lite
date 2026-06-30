use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::sync::Arc;
use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_core::scheduler::SchedulerEngine;
use vllm_core::types::{Request, SchedulerConfig};

fn scheduler_add_request(c: &mut Criterion) {
    let config = SchedulerConfig {
        max_num_seqs: 256,
        max_num_batched_tokens: 2048,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
        ..Default::default()
    };
    let metrics = Arc::new(EnhancedMetricsCollector::new());

    c.bench_function("scheduler_new", |b| {
        b.iter(|| {
            let scheduler = SchedulerEngine::new(config.clone(), 1024, metrics.clone());
            black_box(scheduler)
        });
    });
}

fn scheduler_build_batch(c: &mut Criterion) {
    let config = SchedulerConfig::default();
    let metrics = Arc::new(EnhancedMetricsCollector::new());
    let mut scheduler = SchedulerEngine::new(config, 1024, metrics);

    for i in 0..100 {
        let tokens: Vec<u32> = (0..128)
            .map(|j| u32::try_from(i * 100 + j).expect("bounded bench index"))
            .collect();
        let id = u64::try_from(i).expect("bounded bench index");
        scheduler.add_request(Request::new(id, tokens, 256));
    }

    let mut group = c.benchmark_group("scheduler_build_batch");

    for num_seqs in [10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_seqs),
            &num_seqs,
            |b, &_n| {
                b.iter(|| {
                    let _batch = scheduler.build_batch();
                });
            },
        );
    }

    group.finish();
}

fn radix_prefix_match_benchmark(c: &mut Criterion) {
    use vllm_core::scheduler::RadixTree;

    let mut tree = RadixTree::new();
    for i in 0..256u32 {
        let tokens: Vec<u32> = (0..=i).collect();
        tree.insert(&tokens, vec![i as usize]);
    }
    let query: Vec<u32> = (0..200).collect();

    c.bench_function("radix_prefix_match_200", |b| {
        b.iter(|| tree.longest_prefix_match(black_box(&query)));
    });
}

criterion_group!(
    benches,
    scheduler_add_request,
    scheduler_build_batch,
    radix_prefix_match_benchmark
);
criterion_main!(benches);
