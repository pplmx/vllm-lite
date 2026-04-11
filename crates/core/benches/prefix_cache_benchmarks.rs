use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use vllm_core::scheduler::RadixTree;
use vllm_traits::TokenId;

/// Benchmark RadixTree prefix matching
fn bench_radix_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("radix_tree");
    // Setup: Create tree with various prefix patterns
    let mut tree = RadixTree::new();
    // Insert 1000 entries with shared prefixes
    for i in 0..1000 {
        let tokens: Vec<TokenId> = vec![
            i / 100,        // Shared first token
            (i % 100) / 10, // Shared second token
            i % 10,         // Unique
        ];
        tree.insert(&tokens, vec![i as usize]);
    }

    // Benchmark longest prefix match
    group.bench_function("longest_prefix_match_hit", |b| {
        let query = vec![5, 3, 7, 8, 9];
        b.iter(|| black_box(tree.longest_prefix_match(&query)))
    });

    group.bench_function("longest_prefix_match_miss", |b| {
        let query = vec![99, 99, 99];
        b.iter(|| black_box(tree.longest_prefix_match(&query)))
    });

    // Benchmark insert
    group.bench_function("insert", |b| {
        let mut t = RadixTree::new();
        let tokens: Vec<TokenId> = vec![1, 2, 3, 4, 5];
        b.iter(|| {
            t.insert(&tokens, vec![100]);
        })
    });
    group.finish();
}

criterion_group!(benches, bench_radix_tree);
criterion_main!(benches);
