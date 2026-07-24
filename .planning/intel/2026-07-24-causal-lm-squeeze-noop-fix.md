# RIL: causal_lm squeeze no-op fix — 2026-07-24

## Context

Iteration: v31.0 Phase D (Multi-Node)
Agent cycle: 1 (first autonomous engineering cycle)
Status: Complete — 2 commits produced, full CI green

## Problem

`crates/model/src/causal_lm/mod.rs` — `greedy_sample_token` and
`logits_to_vector` used chained `squeeze(dim)` calls followed by
`to_vec1()` to extract logits. Candle's `squeeze(dim)` is a no-op
on dimensions with size > 1, so if the incoming tensor doesn't
have the expected shape, the squeeze chain may not reduce the rank
sufficiently. `to_vec1()` then panics with "unexpected rank" at
runtime.

The latent bug was in the decode path where `squeeze(0).squeeze(0)`
could be a no-op on dim 0 (size > 1) depending on the tensor shape.

## Fix

- `greedy_sample_token`: `flatten_all()` before `to_vec1()`
- `logits_to_vector`: `flatten_all()` before `to_vec1()`
- Added 4 regression tests covering:
  - Decode path with rank-2 [batch, vocab] input (squeeze no-op)
  - Prefill path with rank-3 [1, seq, vocab] multi-position input
  - `greedy_sample_token` returns correct argmax token
  - `logits_to_vector` returns correct flattened values

## Verification

- 556/556 tests passing in vllm-model (up from 552)
- Full CI (`just ci`): 1837/1837 tests passing (up from 1833)
- No clippy warnings introduced
- `cargo fmt --check` clean
- Docs build clean (`RUSTDOCFLAGS="-D warnings" cargo doc`)
- No public API change

## Commits

1. `4d805848` — fix(model): harden greedy_sample_token/logits_to_vector against squeeze no-op + add regression tests
2. `924c23e5` — test(gpu): fix nextest build options in GPU integration test script

## Audit Results

- **squeeze + to_vec1 pattern**: Audited all 14 `squeeze()` call sites in `crates/model/src/`. The remaining instances are safe because they're either:
  - Followed by `reshape()` (not `to_vec1()`)
  - Preceded by `narrow(dim, _, 1)` which guarantees the dim has size 1
  - Inside conditional checks that verify rank before squeezing
- No other call sites need the `flatten_all()` fix
- **clippy warnings**: 209 total (43 in chat_integration_test.rs are `significant_drop_tightening` style warnings on Mutex guards in tests; 1 `cognitive_complexity` in vllm-dist; rest are pedantic/nursery)
- **No deny-tier clippy violations**

## Commit

`4d805848` — fix(model): harden greedy_sample_token/logits_to_vector against squeeze no-op + add regression tests

## Next Steps

- Review whether other call sites in the codebase have the same
  squeeze-without-flatten_all anti-pattern
- Consider adding a clippy lint or code review checklist item for
  "squeeze followed by to_vec1 without flatten_all"

## Key Learnings

- Candle's `squeeze(dim)` API differs from PyTorch's — PyTorch's
  `squeeze()` (without args) removes all size-1 dims, while Candle's
  `squeeze(dim)` only removes a specific dim and is a no-op if that
  dim has size > 1
- Always use `flatten_all()` or explicit reshaping before `to_vec1()`
  in Candle to avoid rank-related panics
