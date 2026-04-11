use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
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
    };

    c.bench_function("scheduler_new", |b| {
        b.iter(|| {
            let scheduler = SchedulerEngine::new(config.clone(), 1024);
            black_box(scheduler)
        });
    });
}

fn scheduler_build_batch(c: &mut Criterion) {
    let config = SchedulerConfig::default();
    let mut scheduler = SchedulerEngine::new(config, 1024);

    for i in 0..100 {
        let tokens: Vec<u32> = (0..128).map(|j| (i * 100 + j) as u32).collect();
        scheduler.add_request(Request::new(i as u64, tokens, 256));
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

fn hash_tokens_benchmark(c: &mut Criterion) {
    use vllm_core::kv_cache::hash_tokens;

    let tokens: Vec<u32> = (0..512).collect();

    c.bench_function("hash_tokens_512", |b| {
        b.iter(|| hash_tokens(black_box(&tokens)));
    });
}

criterion_group!(
    benches,
    scheduler_add_request,
    scheduler_build_batch,
    hash_tokens_benchmark
);
criterion_main!(benches);
