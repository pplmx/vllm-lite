use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;
use vllm_core::scheduler::policy::{FcfsPolicy, SchedulingContext, SchedulingPolicy, SjfPolicy};
use vllm_core::scheduler::{
    PhaseScheduler, PhaseSwitchPolicy, RequestQueue, SchedulerEngine, SchedulerState,
};
use vllm_core::types::{Priority, Request, SchedulerConfig, Sequence, Status};

/// Benchmark RequestQueue O(1) operations vs O(n)
fn bench_request_queue_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("request_queue");
    // Setup: Create queue with 1000 sequences
    let policy = FcfsPolicy::new();
    let ctx = SchedulingContext {
        current_time: Instant::now(),
        queue_length: 0,
        running_count: 0,
        memory_pressure: 0.0,
    };

    group.bench_function("get_o1", |b| {
        let mut queue = RequestQueue::new();
        for i in 0..1000 {
            let seq = Sequence {
                id: i,
                tokens: vec![1, 2, 3],
                kv_blocks: Arc::new(vec![]),
                num_computed_tokens: 0,
                prompt_len: 3,
                status: Status::Waiting,
                max_tokens: 10,
                sampling_params: vllm_core::types::SamplingParams::default(),
                consecutive_decode_rounds: 0,
                priority: Priority::default(),
            };
            queue.enqueue(seq, &policy, &ctx);
        }
        b.iter(|| black_box(queue.get(black_box(500))))
    });

    group.bench_function("remove_o1", |b| {
        // Reset queue for each iteration
        b.iter_with_setup(
            || {
                let mut q = RequestQueue::new();
                for i in 0..100 {
                    let seq = Sequence {
                        id: i,
                        tokens: vec![1, 2, 3],
                        kv_blocks: Arc::new(vec![]),
                        num_computed_tokens: 0,
                        prompt_len: 3,
                        status: Status::Waiting,
                        max_tokens: 10,
                        sampling_params: vllm_core::types::SamplingParams::default(),
                        consecutive_decode_rounds: 0,
                        priority: Priority::default(),
                    };
                    q.enqueue(seq, &policy, &ctx);
                }
                q
            },
            |mut q| black_box(q.remove(black_box(50))),
        )
    });
    group.finish();
}

/// Benchmark scheduling policies
fn bench_scheduling_policies(c: &mut Criterion) {
    let mut group = c.benchmark_group("scheduling_policies");
    let seq = Sequence {
        id: 1,
        tokens: vec![1; 100],
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: 0,
        prompt_len: 100,
        status: Status::Waiting,
        max_tokens: 200,
        sampling_params: vllm_core::types::SamplingParams::default(),
        consecutive_decode_rounds: 0,
        priority: Priority(50),
    };
    let ctx = SchedulingContext {
        current_time: Instant::now(),
        queue_length: 100,
        running_count: 50,
        memory_pressure: 0.5,
    };

    group.bench_function("fcfs", |b| {
        let policy = FcfsPolicy::new();
        b.iter(|| black_box(policy.compute_priority(&seq, &ctx)))
    });

    group.bench_function("sjf", |b| {
        let policy = SjfPolicy::default();
        b.iter(|| black_box(policy.compute_priority(&seq, &ctx)))
    });
    group.finish();
}

/// Benchmark batch building
fn bench_batch_building(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_building");
    let config = SchedulerConfig::default();

    group.bench_function("build_batch_10", |b| {
        b.iter_with_setup(
            || {
                let mut engine = SchedulerEngine::new(config.clone(), 1024);
                for i in 0..10 {
                    engine.add_request(Request::new(i, vec![i as u32; 50], 100));
                }
                engine
            },
            |mut engine| black_box(engine.build_batch()),
        )
    });

    group.bench_function("build_batch_100", |b| {
        b.iter_with_setup(
            || {
                let mut engine = SchedulerEngine::new(config.clone(), 1024);
                for i in 0..100 {
                    engine.add_request(Request::new(i, vec![i as u32; 50], 100));
                }
                engine
            },
            |mut engine| black_box(engine.build_batch()),
        )
    });
    group.finish();
}

/// Benchmark PhaseScheduler
fn bench_phase_scheduler(c: &mut Criterion) {
    let mut group = c.benchmark_group("phase_scheduler");
    let switch_policy = PhaseSwitchPolicy::default();
    let mut scheduler = PhaseScheduler::new(switch_policy);
    let state = SchedulerState {
        waiting_count: 100,
        running_count: 50,
        prefill_queue_len: 50,
        decode_queue_len: 50,
        available_memory: 500,
        consecutive_decode_rounds: 0,
    };

    group.bench_function("select_phase", |b| {
        b.iter(|| black_box(scheduler.select_phase(&state)))
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_request_queue_operations,
    bench_scheduling_policies,
    bench_batch_building,
    bench_phase_scheduler
);
criterion_main!(benches);
