# v27.0 Performance Baseline (2026-06-28)

**Hardware:** 128-core x86_64 (Linux product-1-23 5.15.0-179-generic)
**Rust:** rustc 1.96.0 (ac68faa20 2026-05-25)
**Profile:** release with `--all-features` (criterion default sample_size=100)
**Branch:** main (no worktree, per AGENTS.md)

## Core benches (existing)

These are pre-optimization baselines captured before any v27.0 work. They will be
compared against post-optimization numbers recorded in `docs/perf/v27-after.md`
(planned for H-14). All numbers from criterion `--output-format bencher` output.

### Quick bench (radix cache only)

Source: `just bench-quick` → `/tmp/v27_h1_baseline_quick.txt`

| Bench                     | Metric   | Baseline        |
| ------------------------- | -------- | --------------- |
| `radix_longest_prefix_match_250` | ns/op    | 4,654–4,671 ns (median 4,661 ns) |
| `radix_insert_64_tokens`         | ns/op    | 4,654–4,671 ns (median 4,661 ns) |

(`just bench-quick` uses `--sample-size 10` for fast feedback; full sample in next section.)

### Full bench (all criterion benches)

Source: `just bench` (with the 4 unregistered benches now wired via Cargo.toml
— see [Notes](#notes)) → `/tmp/v27_h1_baseline_full.txt`

| Bench                                       | Workload                              | Metric   | Baseline (median)   |
| ------------------------------------------- | ------------------------------------- | -------- | ------------------- |
| `latency_percentiles/10`                    | p50/p99 estimator, 10 events          | ns/iter  | 21,521 ns (±74)     |
| `latency_percentiles/50`                    | p50/p99 estimator, 50 events          | ns/iter  | 87,113 ns (±287)    |
| `multi_draft_iter_total/no_draft`           | speculative decode, no draft          | ns/iter  | 1,474 ns (±63)      |
| `multi_draft_iter_total/self_spec`          | speculative decode, self spec          | ns/iter  | 1,520 ns (±48)      |
| `multi_draft_iter_total/external_draft`     | speculative decode, external draft    | ns/iter  | 1,968 ns (±24)      |
| `radix_tree/longest_prefix_match_hit`       | prefix cache, 1000-entry tree, hit    | ns/iter  | 48 ns (±0)          |
| `radix_tree/longest_prefix_match_miss`      | prefix cache, 1000-entry tree, miss   | ns/iter  | 13 ns (±1)          |
| `radix_tree/insert`                         | prefix cache, single insert           | ns/iter  | 97 ns (±0)          |
| `radix_longest_prefix_match_250`            | radix cache, 250-token query          | ns/iter  | 4,683 ns (±43)      |
| `radix_insert_64_tokens`                    | radix cache, 64-token insert          | ns/iter  | 4,646 ns (±17)      |
| `scheduler_new`                             | SchedulerEngine::new construction     | ns/iter  | 667 ns (±4)         |
| `scheduler_build_batch/10`                  | build_batch with 10 sequences         | ns/iter  | 99 ns (±0)          |
| `scheduler_build_batch/50`                  | build_batch with 50 sequences         | ns/iter  | 100 ns (±0)         |
| `scheduler_build_batch/100`                 | build_batch with 100 sequences        | ns/iter  | 100 ns (±0)         |
| `radix_prefix_match_200`                    | radix cache, 200-token query          | ns/iter  | 4,267 ns (±26)      |
| `request_queue/get_o1`                      | RequestQueue O(1) lookup              | ns/iter  | 16 ns (±0)          |
| `request_queue/remove_o1`                   | RequestQueue O(1) removal             | ns/iter  | 4,333 ns (±24)      |
| `scheduling_policies/fcfs`                  | FCFS scheduling decision              | ns/iter  | 0 ns (±0)           |
| `scheduling_policies/sjf`                   | SJF scheduling decision               | ns/iter  | 3 ns (±0)           |
| `batch_building/build_batch_10`             | batch builder, 10 seqs                | ns/iter  | 5,095 ns (±21)      |
| `batch_building/build_batch_100`            | batch builder, 100 seqs               | ns/iter  | 38,684 ns (±264)    |
| `phase_scheduler/select_phase`              | prefill/decode phase switch           | ns/iter  | 2 ns (±0)           |
| `sequence_packing/fifo/4`                   | FIFO schedule, 4 seqs                 | ns/iter  | 56 ns (±0)          |
| `sequence_packing/packing/4`                | packed schedule, 4 seqs               | ns/iter  | 56 ns (±0)          |
| `sequence_packing/fifo/8`                   | FIFO schedule, 8 seqs                 | ns/iter  | 59 ns (±0)          |
| `sequence_packing/packing/8`                | packed schedule, 8 seqs               | ns/iter  | 58 ns (±0)          |
| `sequence_packing/fifo/16`                  | FIFO schedule, 16 seqs                | ns/iter  | 60 ns (±0)          |
| `sequence_packing/packing/16`               | packed schedule, 16 seqs              | ns/iter  | 60 ns (±0)          |
| `adaptive_speculative/fixed_draft`          | adaptive spec, fixed draft            | ns/iter  | 98 ns (±0)          |
| `adaptive_speculative/adaptive_draft`       | adaptive spec, adaptive draft         | ns/iter  | 98 ns (±1)          |

### Skipped benches (pre-existing hangs)

Two benches are skipped because they hang indefinitely under criterion's default
sample collection. Both share the same code pattern:

```rust
b.iter(|| {
    let mut completed = 0;
    while completed < num_requests {
        let results = black_box(engine.step().unwrap());
        completed += results.len();
    }
});
```

The `while completed < num_requests` loop relies on `engine.step()` returning
non-empty `results` to make progress. When results are empty (e.g., between
batches), the loop never terminates and criterion hangs. This is a **pre-existing
benchmark bug**, not a v27.0 regression.

| Bench                                  | Status     | Reason                                                                |
| -------------------------------------- | ---------- | --------------------------------------------------------------------- |
| `speculative_vs_baseline/*`            | DEFERRED   | engine.step() loop hang (3 inputs × 2 variants = 6 configs blocked)   |
| `optimization_benchmarks/throughput/*` | DEFERRED   | same engine.step() loop hang (3 inputs × 2 variants = 6 configs)      |

These should be fixed in a follow-up (either fix `engine.step()` to surface an
empty-result signal, or rewrite the bench with a timeout / step cap). They are
**not** in scope for H-1 and do not affect the baseline numbers above.

## Notes

- All numbers from criterion `--output-format bencher` output (median ns/iter).
- `just bench-quick` uses `--sample-size 10` for fast iteration. Use `just bench`
  for sample_size=100 authoritative numbers.
- 4 of 8 bench files (`scheduler`, `scheduler_benchmarks`,
  `prefix_cache_benchmarks`, `optimization_benchmarks`) were not registered in
  `crates/core/Cargo.toml` `[[bench]]` entries. Without `harness = false`, cargo
  built them with the libtest harness, which rejects `--output-format bencher`
  and silently aborts. **Fix applied**: added 4 `[[bench]]` entries with
  `harness = false`. See commit in this branch.
- Comparison baseline for H-14 (validation sub-phase).
- Model-layer benches (GQA/MLA/flash attention/paged KV) added in H-2~H-5.

## Files

- Quick bench output: `/tmp/v27_h1_baseline_quick.txt`
- Full bench output: `/tmp/v27_h1_baseline_full.txt`
- Plan reference: `docs/superpowers/plans/2026-06-28-v27-performance.md`
- Audit context: `/tmp/phase_g_audit/SUMMARY.md`

## Model benches (H-2 added 2026-06-28, restructured 2026-06-28)

**Strategy:** Runtime CUDA detection (per user direction).
- **GPU runners:** standard qwen3-7B dims (hidden_size=896, num_heads=14, num_kv_heads=2, head_dim=64, seq_len=[128,512,2048])
- **CPU-only runners (default):** minimal smoke test (hidden_size=64, num_heads=2, head_dim=32, seq_len=16); prints eprintln warning

**Rationale:** Standard implementation should look like production code. CPU-only environments cannot measure realistic perf for qwen3-7B dims. Smoke test verifies the code path executes; perf validation requires GPU runner.

### CPU-only environment (current dev/CI)

| Bench path | seq_len | ns/iter (median) |
|------------|---------|------------------|
| gqa_forward_smoke/cpu_smoke | 16 | 38,445 ns |

**Note:** This is a smoke test, not a perf baseline. Real GPU numbers will be recorded when a GPU runner is available.

### Standard dimensions (recorded when GPU available)

| Bench path | seq_len | ns/iter (median) |
|------------|---------|------------------|
| gqa_forward/standard | 128 | TBD |
| gqa_forward/standard | 512 | TBD |
| gqa_forward/standard | 2048 | TBD |
| gqa_forward/fused | 128 | TBD |
| gqa_forward/fused | 512 | TBD |
| gqa_forward/fused | 2048 | TBD |

## Model benches — MLA (H-3 added 2026-06-28)

**Strategy:** Runtime CUDA detection (same as H-2).
- **GPU:** qwen3-7B class MLA (hidden_size=896, num_heads=14, kv_lora_rank=64, head_dim=64, seq_len=[128,512,2048])
- **CPU:** smoke test (hidden_size=64, num_heads=2, kv_lora_rank=16, head_dim=16, seq_len=16) + eprintln warning

### CPU-only environment (current dev/CI)

| Bench path | seq_len | ns/iter (median) |
|------------|---------|------------------|
| mla_forward_smoke/cpu_smoke | 16 | 43,367 ns |

**Note:** This is a smoke test, not a perf baseline. Real GPU numbers will be recorded when a GPU runner is available.

### Standard dimensions (recorded when GPU available)

| Bench path | seq_len | ns/iter (median) |
|------------|---------|------------------|
| mla_forward | 128 | TBD |
| mla_forward | 512 | TBD |
| mla_forward | 2048 | TBD |

## Model benches — Flash Attention (H-4 added 2026-06-28)

**Strategy:** Runtime CUDA detection (same as H-2/H-3).
- **GPU:** qwen3-7B-class FlashV2 (num_heads=14, head_dim=64). Standard configs: (batch=1, seq=512), (1, 2048), (4, 512).
- **CPU:** smoke test (batch=1, heads=2, seq=16, head_dim=32) + eprintln warning. Uses `FlashAttentionConfig::new().with_flash_v2()`; internally takes the standard path for seq_len≤128.

**Note:** Flash attention speedups are primarily a GPU optimization (memory bandwidth wins from tile-based kernel fusion). The CPU smoke test only verifies correctness; real perf validation requires a GPU runner.

### CPU-only environment (current dev/CI)

| Bench path | Config | ns/iter (median) |
|------------|--------|------------------|
| flash_attention_smoke/cpu_smoke | b1_h2_s16_d32 | 11,285 ns |

### Standard dimensions (recorded when GPU available)

| Bench path | Config | ns/iter (median) |
|------------|--------|------------------|
| flash_attention/standard | b1_h14_s512_d64 | TBD |
| flash_attention/standard | b1_h14_s2048_d64 | TBD |
| flash_attention/standard | b4_h14_s512_d64 | TBD |
