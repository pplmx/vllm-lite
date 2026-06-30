use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use vllm_core::scheduler::RadixTree;

fn bench_radix_longest_prefix_match(c: &mut Criterion) {
    let mut tree = RadixTree::new();
    for i in 0..500usize {
        let tokens: Vec<u32> = (0..=u32::try_from(i).expect("bounded bench index")).collect();
        tree.insert(&tokens, vec![i]);
    }
    let search: Vec<u32> = (0..250).collect();

    c.bench_function("radix_longest_prefix_match_250", |b| {
        b.iter(|| tree.longest_prefix_match(black_box(&search)));
    });
}

fn bench_radix_insert(c: &mut Criterion) {
    c.bench_function("radix_insert_64_tokens", |b| {
        b.iter(|| {
            let mut tree = RadixTree::new();
            let tokens: Vec<u32> = (0..64).collect();
            tree.insert(black_box(&tokens), vec![1, 2, 3, 4]);
            tree
        });
    });
}

criterion_group!(
    benches,
    bench_radix_longest_prefix_match,
    bench_radix_insert
);
criterion_main!(benches);
