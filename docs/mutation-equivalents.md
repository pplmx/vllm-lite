# Mutation equivalents

`cargo mutants` reports a mutant as MISSED when no test fails after the edit.
That is *usually* untested code, but not always: some edits are **equivalent** —
the mutated program and the original have identical observable behaviour, so no
test can ever distinguish them. Chasing those with contrived tests wastes
effort and leaves misleading "coverage" behind.

This file records every equivalence this repository has established, with the
evidence. `.cargo/mutants.toml` excludes them from the nightly scan so the
score reflects real gaps.

Two rules for adding an entry:

1. **Measure, do not reason.** A candidate is only equivalent after you have
   either constructed a fixture that fails to separate the variants, or shown
   the reachable input set excludes the differing case. Guessing that a branch
   is unreachable is how a real gap gets suppressed.
2. **Never exclude a survivor you have not tried to kill.** "Hard to trigger"
   is not "equivalent".

Note that a reported MISSED is not proof of a gap either — `cargo mutants` has
mis-attributed several verdicts in this tree. `scripts/verify_missed_mutants.py`
re-applies a recorded span edit by hand so the two cases can be told apart.

## Registered equivalents

### `spec_dispatch/dispatch.rs:527` — `stop_len > matched` → `>=`

`trim_burst_at_stop` returns `keep = matched.saturating_sub(stop_len)` together
with `match_starts_before_burst = stop_len > matched`. At `stop_len == matched`
both variants differ only in that flag, and `keep == 0`: the entire burst is
dropped, so `pending_stream` never holds a token for this sequence and the flag
has nothing to suppress. Measured both variants on a stop that begins at the
burst's first token (`stop: [7,8]`, model emitting `7,8,9,…`): the client stream
is `[]` either way.

### `spec_dispatch/dispatch.rs:164` — `total_draft > 0` → `>=`

The guard is redundant. `record_speculative_acceptance` opens with its own
`if total > 0` and only then divides, so calling it with `total == 0` records
nothing. With `usize`, `>= 0` is always true, which makes the mutant equivalent
to deleting the outer guard entirely.

### `spec_dispatch/dispatch.rs:565` — `kv_block_ids[i].len() < blocks_needed` → `<=`

The body calls `ensure_blocks_for_tokens(seq, blocks_needed)`, whose growth loop
is `while kv_blocks.len() < min_blocks`. On the equality case the loop body
never runs, so the call is a no-op.

### `spec_dispatch/dispatch.rs:227` — `elapsed > 0.0` → `>=`

`elapsed` is `start.elapsed().as_millis() as f64`. The two variants differ only
at exactly `0`, where both skip the sample. (The `<` and `==` variants of this
same line are **not** equivalent — they drop every sample, and a slow-backend
test now kills them.)

### `engine/graph_step.rs:280` — `elapsed > 0.0` → `>=`

Same reasoning as `dispatch.rs:227`. Again `<` and `==` are genuinely wrong and
are covered by `cuda_graph_step_records_latency_for_a_slow_step`.

### `spec_dispatch/verify.rs:179` — `offset + vocab_size > logits.len()` → `>=`

The verifier walks a `input + drafts` window, so the largest offset is
`(input_len - 1 + drafts - 1) * vocab`; adding one more vocabulary block stays
at or below `logits.len()`. Instrumented the guard on the prefill fixture: the
`end` values observed max out at `logits_len - vocab`, so the comparison never
reaches equality and both variants agree.

### `batch_composer/compose/prefill.rs:168` — `tokens_to_process > chunk_cap` → `>=`

At `tokens_to_process == chunk_cap` the second branch computes
`tokens_to_process.min(chunk_cap)` and the fall-through returns
`tokens_to_process`; both are `chunk_cap`. `resumed_prefill_exactly_one_chunk_is_not_re_split`
pins the behaviour anyway, but no fixture can separate the variants.

### `scheduler/engine/memory.rs:107` — `s.status == Status::Prefilling` → `!=`

`execute_preemption` filters `running` for preemptable sequences. The `Prefilling`
arm cannot match: `advance_computed_tokens` keeps a sequence `Prefilling` only
while `num_computed_tokens < prompt_len`, and `requeue_partial_prefills` moves
exactly those sequences back to the waiting queue at the end of both update
paths (`update` line 69, `update_speculative` line 124). `build_batch` calls
`execute_preemption` before `running.extend(new_sequences)`, so a freshly
admitted prefill is not in `running` yet either. When the filter runs,
`running` holds only `Decoding` sequences, and `Decoding || != Prefilling` is
true for all of them.

### `scheduler/engine/memory.rs:251` — `delete !` in `cancel_request` (queued branch)

The queued branch removes from `request_queue`. A request that is still queued
has never been batched, so its `kv_blocks` is empty and `is_empty()` is always
true — inverting the guard skips a call that would have been a no-op. The
*running* branch of the same function is a different story: a simplified
mutation there leaks four blocks and `test_cancel_request_releases_kv_blocks`
catches it. Only the queued-branch mutant is equivalent.

### `scheduler/engine/memory.rs:62` — `available_blocks() < blocks_needed` → `<=`

Callers only reach `execute_preemption` after `additional > available_blocks()`
and pass `blocks_needed` as the sequence's WHOLE table, so on entry
`available < needed` almost always holds. The `<=` variant differs only at exact
equality, which the production call path does not produce. Recorded from earlier
instrumentation; this one has also shown up as a tool TIMEOUT, so it is excluded
on evidence gathered by hand rather than by the scanner.

### `sampling.rs:178` / `:179` — the `>=` relaxations

`renormalized_top_p_logprobs` guards its normalisation with
`if total > 0.0 { p / total } else { 0.0 }` and then `if p_norm > 0.0`.
Neither `>=` variant is reachable in a different state:

- `total` is the sum of `probs[..cutoff]`, and `cutoff >= 1` always, so
  `total >= probs[0] = exp(max - max) / sum = 1/sum > 0`. The `total == 0`
  branch the `>=` variant would newly take cannot occur while `max_val` is
  finite (which the earlier guard already enforces).
- At `p_norm == 0`, `>` sends control to the else arm (`-inf`) and `>=` takes
  `0.0_f32.ln()`, which is also `-inf`.

The `-` -> `+` variant on the same stabiliser is NOT equivalent, and the reason
is worth recording: arithmetically a constant shift cancels in the softmax, but
with `f32` the term `exp(v + max_val)` overflows to `inf` once `max_val` passes
about 88, the sum becomes `inf`, and every probability collapses to `NaN`.
`renormalized_top_p_logprobs_stays_finite_for_large_logits` pins that.

### `sampling.rs:372` — the temperature-scaling guard

`if temperature > 0.0 && (temperature - 1.0).abs() > EPSILON { *l /= temperature }`.

- Replacing `-` with `+` or `/` changes the predicate to
  `|T + 1| > EPSILON` / `|T / 1| > EPSILON`, both true at `T == 1.0`. The
  guard then performs `*l /= 1.0`, which is a no-op; for every other
  temperature the original guard was already true, so the work is identical.
- Relaxing `>` to `>=` differs only when `|T - 1|` is exactly `f32::EPSILON`.
  In that case the mutant divides by a temperature within one f32 ulp of 1.0,
  which cannot change the resulting sampling distribution.

The `temperature > 0.0` half of the same guard is NOT equivalent and is
covered: `>=` admits `T = 0`, and a logit of exactly `0.0` divides to
`0.0 / 0.0 = NaN`, which `greedy_sample` then runs on.

### `sampling.rs:368` — the repeat-penalty guard

`if (repeat_penalty - 1.0).abs() > EPSILON && !seen.is_empty() { apply_repeat_penalty(..) }`.

- Replacing `-` with `+` or `/` makes the predicate true at `penalty == 1.0`,
  where `apply_repeat_penalty` divides by 1.0; for every other penalty the
  original guard was already true, so the work performed is identical.
- `>` to `>=` differs only when `|penalty - 1|` is exactly `f32::EPSILON` — a
  penalty within one f32 ulp of 1.0.
- `&&` to `||` opens two paths the `&&` closes, and both are no-ops: an empty
  history makes `apply_repeat_penalty` iterate nothing, and a non-neutral
  penalty paired with `penalty == 1.0` divides by 1.0.

The `!seen.is_empty()` half is NOT equivalent — deleting the `!` inverts the
history test and is covered by
`sample_batch_repeat_penalty_requires_history_and_a_non_neutral_value`.

### `ha/failover.rs:82` and `ha/leader_election.rs:109` — timeout boundaries

- `now.duration_since(req.created_at) > request_timeout`: relaxing `>` to `>=`
  differs only when the age is EXACTLY the timeout. Both sides are `Duration`
  values derived from independent `Instant` reads, so equality needs a
  nanosecond-exact coincidence that a real clock does not produce.
- `while start.elapsed() < timeout`: relaxing to `<=` matters only for
  `timeout == 0`, where `elapsed()` is already non-zero. Measured: immediately
  after `Instant::now()` the value is 121ns, so both `121ns < 0` and
  `121ns <= 0` are false and a zero-length window never iterates. For every
  positive timeout the two forms differ only in a sub-nanosecond window after
  the deadline.

### `metrics/lock_free.rs:239:89` — the percentile saturation bound

The helper ends `... .min(xs.len() - 1)`. The index it caps is
`floor(p * (n - 1))`, which for `p` in `0..=1` never exceeds `n - 1`, so the
`min` never binds and both `n - 1` and `n + 1` return the index unchanged.

Note the column: this line contains TWO minus signs — column 50 inside
`(p * (xs.len() as f64 - 1.0))` and column 89 inside `min(xs.len() - 1)`. The
first governs the percentile RANK and is a genuine gap (killed by
`test_snapshot_percentiles_match_the_index_formula`); the second is the
clamp and is equivalent. A verification run that trusts a stale column
number will test the wrong one and report a false survivor.

### `metrics/collector/sampler/runtime.rs:46` and `:49` — zero-count recording

`record_batch_phase_tokens` guards with `if prefill > 0 { record_prefill_tokens(prefill) }`
and the same for `decode`. Relaxing `>` to `>=` puts a `record_*_tokens(0)` call
on the zero path, which adds zero to an `AtomicU64`. The counters, and therefore
the throughputs derived from them, are unchanged.

### `engine/state/request.rs:76` — the id counter parked after issuing

`self.next_seq_id += 1` advances past the id just handed out. Relaxing it to
`*= 1` leaves the counter ON that id. The next admission opens with
`while self.seq_id_is_taken(self.next_seq_id)`, which now sees the id as taken
and advances it, so the ids issued are identical — the mutant costs one extra
iteration of a bounded (max 256 running) loop and nothing else.

The neighbouring `+= 1` inside that skip loop is NOT equivalent: `-= 1` walks the
counter backwards and `*= 1` makes the loop unable to terminate. Both are killed
by `sequence_id_counter_skips_taken_ids_and_advances`.

### `engine/state/batch.rs:338` — growing a table that is already big enough

`admit_decode_sequences` guards growth with `if kv_blocks.len() < needed`. With
`<=` the call also happens at equality, but `grow_running_table` computes
`needed - have == 0`, builds an empty `additions` vector, and returns before
recording or cloning anything. The following `len() >= needed` check admits the
sequence in both versions.

### `scheduler/phase_scheduler.rs:36` — `PhaseSwitchPolicy::builder` -> `Default::default()`

`PhaseSwitchPolicyBuilder` derives `Default`, and its only field is
`inner: PhaseSwitchPolicy`, whose own `Default` is the written `impl Default`
(the documented 10 / 5 / 2). The real `builder()` returns
`PhaseSwitchPolicyBuilder::default()` verbatim, so the mutant is the same
expression: no fixture can separate them, whatever it asserts about the built
policy.

The other builder mutants in this file are NOT equivalent and are killed:
`build` replaced by `Default::default()` ignores every `with_*` override, and
the three `with_*` setters replaced by `Default::default()` drop their
respective field.

### `circuit_breaker/breaker.rs:49` and `strategy.rs:69` — `builder()` -> `Default::default()`

`CircuitBreakerConfig::builder()` and `RetryStrategy::builder()` both return
`XBuilder::default()` verbatim. `CircuitBreakerConfigBuilder` derives `Default`
over a single `inner: CircuitBreakerConfig` field whose own `Default` is the
documented 5 / 30s / 3, and `RetryStrategyBuilder`'s hand-written `Default`
sets the same 3 attempts / 100ms as `RetryStrategy::default()`. The mutant body
is therefore textually the original body, and no fixture can separate them.

The sibling builder mutants in both files are NOT equivalent and are killed:
replacing `build` with `Default::default()` ignores every `with_*` override,
and replacing a `with_*` setter drops its field.

### `batch_composer/compose/mod.rs:92` — the packing dispatch guard

`compose` matches `Phase::Prefill if self.packing_config.enabled && sequences.len() > 1`
to `compose_prefill_with_packing` and the fall-through `Phase::Prefill` arm to
`compose_prefill_batch`. Those two functions have the same body:

```rust
pub(super) fn compose_prefill_with_packing(&self, sequences: Vec<Sequence>) -> Batch {
    self.compose_prefill_batch(sequences)
}
```

The packing optimisation is deferred (the entry point exists so callers can opt
in without changing the dispatcher), so every guard variant — `false`, `true`,
`&&`→`||`, `>`→`<`/`==`/`>=` — only chooses between two calls that run the same
code on the same argument. Six mutants, all equivalent. When the packing
optimisation lands, the bodies diverge and these exclusions lapse (the patterns
are line-anchored, so a refactor also makes them fail loudly).

### `batch_composer/validate.rs:38`, `validate.rs:100`, `cuda_graph.rs:97` — `builder()`

Same class as the `phase_scheduler`/circuit-breaker entries above: each
`builder()` returns `XBuilder::default()` verbatim, and each builder derives
`Default` over a single `inner` field whose own `Default` is the written impl.

### `scheduler/batch.rs:404` — `elapsed > 0.0` → `>=`

Same reasoning as `spec_dispatch/dispatch.rs:227` and `engine/graph_step.rs:280`:
`elapsed` is `start.elapsed().as_millis() as f64`, so the two variants differ
only at exactly `0`, where both skip the sample. (The `<` and `==` variants of
this line are NOT equivalent — they drop every sample — and are killed by the
latency-recording test.)

### `scheduler/engine/state/batch.rs:405` — `num_computed_tokens > 0` → `>=`

`requeue_seq` branches on `seq.num_computed_tokens > 0`. At exactly `0` the two
variants take different branches, but the branches coincide: the reset branch
for a never-advanced sequence releases an empty block table, sets
`kv_blocks = vec![]` (already empty), `num_computed_tokens = 0` (already zero)
and `status = Waiting` (what the preserve branch also sets). Measured with the
fresh-sequence fixture used by the requeue tests: both variants leave the same
queue entry with the same fields.

The `<` and `==` variants of the same line are NOT equivalent — they send an
ADVANCED sequence down the reset branch, releasing its KV and zeroing its
frontier — and are killed by
`requeue_seq_preserves_an_advanced_prefills_frontier_and_blocks`.

### Unobservable rather than equivalent

`RequestQueue::cleanup_priority_queue` (replace with `()`) cannot be
distinguished from a no-op by any fixture: `dequeue` already skips stale heap
entries in its `while let`, and the heap's internal length is not exposed. The
memory it would reclaim is real, so this is a genuine (if minor) leak — just not
one the current interface can observe. It is **not** excluded, so it stays
visible as a known limitation.
