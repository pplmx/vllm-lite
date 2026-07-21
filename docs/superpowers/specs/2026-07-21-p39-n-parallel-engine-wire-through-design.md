# P39 — `n > 1` Engine Wire-Through on `/v1/chat/completions` + `/v1/completions`

**Date:** 2026-07-21
**Phase:** v31.0 / P39 (closes v0.x wire-type backlog)
**Author:** vllm-lite evolution iteration (post-P38 ROI analysis)
**Status:** Approved design (2026-07-21)

---

## 1. Motivation

After P38 closed the **last hard-400 v0.x wire-type engine-honoring carve-out** (`stop` sequences on chat + completions), exactly one v0.x wire-type carve-out remains at the **validation-only** layer:

- **`n`** — `ChatRequest::n` (declared in P22) and `CompletionRequest::n` (declared in P32) are both declared on the wire type, but the HTTP-layer validator currently **rejects `n > 1` with `400 invalid_request_error`** ("n > 1 is not supported…"). This is a broken OpenAI contract — clients that set `n > 1` for multi-candidate generation (a standard OpenAI capability used by LiteLLM, LangChain, comparison/ensemble workflows) cannot use vllm-lite at all.

P37 shipped the `best_of` engine-honoring pattern (`tokio::spawn` of N parallel `EngineMessage::AddRequest` + per-candidate `Vec<SampledToken>` collection + ranking). P36 shipped the `SampledToken::logprob` primitive that P37 built on. Both are direct enablers for `n > 1`: the only architectural difference is that `n > 1` returns **all N candidates** (no ranking), whereas `best_of` returns the **single best** candidate.

The work has **very high ROI**: closes the **last** v0.x wire-type gap (all other chat + completions OpenAI-spec fields are end-to-end after P38), reuses P37's helpers almost verbatim, and ships with **public-API delta = 0** (P22/P32 already declared the `n` field).

## 2. Goals

- **G1:** `n > 1` accepted and honored end-to-end on both `/v1/chat/completions` and `/v1/completions`.
- **G2:** `n > 1` returns N distinct choices (NOT ranked — distinct from `best_of`'s "single best" semantics).
- **G3:** `n = 1` / `n = None` path is **zero-overhead** — short-circuits to the existing P37/P38 single-shot handler.
- **G4:** Composes correctly with P38 (`stop`), P37 (`best_of` — cross-field reject), P36 (`logprobs` / `top_logprobs`), P35 (`echo` / `suffix` — NEW cross-field reject), P34 (`seed`).
- **G5:** Streaming SSE carries `choices: [{index: 0, ...}, {index: 1, ...}, ...]` per event (OpenAI convention).
- **G6:** Public-API surface increase = **0** (P22/P32 already declared the `n` field).
- **G7:** Closes the v31.0 "Perfection & Elegance" milestone (last remaining v0.x wire-type item; all other open items in the v32+ backlog are explicitly deferred).

## 3. Non-Goals

- **N1:** `tools` / `tool_choice` engine honoring — v32+ work (grammar-constrained decoder; declared in P33).
- **N2:** `response_format: json_object` constrained-decoder honoring — v32+ work (declared in P22).
- **N3:** `n` upper bound above 8 — see §6 (8 is the practical scheduler-safe cap; OpenAI docs nominally allow up to 128, but real-world LiteLLM/LangChain usage tops out at 4–8).
- **N4:** Streaming cross-candidate ordering guarantees beyond "lock-free arrival-order merge" — OpenAI spec does not require per-candidate interleaving (each choice's tokens stream independently).
- **N5:** Per-candidate seed override field — single `seed` parameter is the only API; per-candidate seed is **derived** deterministically (`seed.wrapping_add(i)`).

## 4. Design

### 4.1 Execution Model

The execution model mirrors P37's `best_of` pipeline almost verbatim — the architectural difference is "return all N" vs "rank + return 1":

```
ChatRequest { n: Some(2), seed: Some(42), stop: Some(["</s>"]), ... }
       │
       ▼
┌──────────────────────────────────────────────────┐
│ Private helper: run_n_parallel_chat              │
│   1. tokenize prompt                             │
│   2. for i in 0..n:                              │
│        per_candidate_seed = seed.map(|s| s.wrapping_add(i)) │
│        EngineMessage::AddRequest {               │
│          sampling_params: { seed: per_candidate_seed, ... }, │
│          response_tx: oneshot,                   │
│        }                                         │
│        tokio::spawn(async move { send + collect  │ })
│   3. join_all → Vec<Vec<SampledToken>>           │
│   4. assemble ChatResponse { choices: [...] }    │
└──────────────────────────────────────────────────┘
```

### 4.2 New Private Helpers (`crates/server/src/openai/{chat,completions}.rs`)

Reuse P37's `spawn_best_of_candidate` → rename to **`spawn_n_candidate`** (parameterised by `candidate_index: usize` and `n_total: usize`). Internal mechanism is identical: build a fresh `EngineMessage::AddRequest`, send through `state.engine_tx`, collect the per-candidate `Vec<SampledToken>` stream via `response_tx`.

New private helper **`collect_n_candidates`** wraps `join_all` over N `spawn_n_candidate` futures and returns `Vec<Vec<SampledToken>>` (outer index = candidate index, inner = sampled tokens).

New private helper **`per_candidate_seed(seed: Option<i64>, index: usize) -> Option<u64>`** computes `seed.map(|s| s.wrapping_add(index as u64))`. Used by both chat and completions handlers.

### 4.3 New Private Helper — Streaming Event Assembly

New private helper **`assemble_streaming_event(candidates: &[StreamingCandidateState]) -> serde_json::Value`** assembles one SSE event from the in-flight candidate states:

```rust
struct StreamingCandidateState {
    index: usize,
    next_token: Option<SampledToken>,    // None if candidate has not produced a new token since last event
    finished: bool,
    finish_reason: Option<FinishReason>,
}
```

The chat streaming loop holds `Vec<StreamingCandidateState>` (length = `n`) and calls `assemble_streaming_event` after every `select!` round across N `response_rx` channels. Events with all-`None` next_token + all-unfinished are skipped (avoids emitting empty SSE events).

The completions streaming loop uses the same helper but emits the legacy `text` field shape (one entry per candidate, indexed).

### 4.4 Non-Streaming Assembly

For `stream = false`:

- **`ChatResponse::choices`** = `Vec<ChatChoice>` of length N; each entry has `index: usize`, `message: ChatMessage`, `logprobs: Option<ChatChoiceLogprobs>`, `finish_reason: Option<String>`.
- **`CompletionResponse::choices`** = `Vec<CompletionChoice>` of length N; each entry has `index: usize`, `text: String`, `logprobs: Option<CompletionLogprobs>`, `finish_reason: Option<String>`.
- **`usage`** = `Usage { prompt_tokens, completion_tokens, total_tokens }` — completion_tokens is **summed across all N candidates** (matches OpenAI billing semantics).

### 4.5 Composes with Prior Wire-Throughs

| Field | Composes? | Notes |
|-------|-----------|-------|
| `stop` (P38) | ✅ | Each candidate honors its own stop set independently (tokenization is shared — same prompt + same stop list ⇒ identical token sequences per candidate) |
| `logprobs` / `top_logprobs` (P36) | ✅ | Each `ChatChoice` / `CompletionChoice` carries its own `logprobs` field |
| `seed` (P34) | ✅ | Per-candidate seed derived via `per_candidate_seed(seed, i)` — deterministic, repeatable |
| `temperature` / `top_p` / `top_k` / `repeat_penalty` / `presence_penalty` / `logit_bias` | ✅ | Identical `SamplingParams` forwarded to every candidate |
| `best_of` (P37) | ❌ | Existing cross-field rule (`n > 1` × `best_of > 1` rejected with 400) — unchanged |
| `echo` (P35) | ❌ | **NEW cross-field rule**: `n > 1` × `echo = true` rejected with 400 (OpenAI spec: echo only makes sense for single completion) |
| `suffix` (P35) | ❌ | **NEW cross-field rule**: `n > 1` × `suffix = Some(_)` rejected with 400 (same rationale as `echo`) |
| `max_tokens` | ✅ | Each candidate honors its own `max_tokens`; the per-candidate finish reason for the candidate that hit max first is `Length`, others continue |
| `tools` / `tool_choice` / `response_format` | ⚠️ | No interaction (still no-op at engine layer; v32+ work) |
| `stream` | ✅ | N candidate streams are interleaved into one SSE event stream; final `[DONE]` emitted after all N finalize |

### 4.6 Validator Tightening (`crates/server/src/openai/sampling_validation.rs`)

#### 4.6.1 New upper-bound check on `n`

```rust
const MAX_N: i64 = 8;

// In validate_chat_request_fields and validate_completion_request_fields:
if let Some(n) = req.n {
    if n < 1 { /* existing rejection */ }
    if n > MAX_N {
        return Err(ValidationError::new(format!(
            "n = {n} exceeds maximum allowed value of {MAX_N} (n = 1..=8)",
        )));
    }
}
```

**Justification for cap = 8:**

| Cap | Scheduler worst-case (256 seq × N) | LiteLLM/LangChain practice | Comparison to `best_of` (cap 20) |
|-----|------------------------------------|-----------------------------|----------------------------------|
| 8 | 2048 in-flight | ✅ Covers typical comparison/ensemble | N is "return all", so N × cost > best_of (1 × cost) |
| 20 | 5120 in-flight | ❌ Excessive | — |
| 128 (OpenAI nominal) | 32768 in-flight | ❌ Way beyond practice | — |

The `best_of` cap of 20 is higher than `n`'s cap of 8 because `best_of` returns **1** token stream (ranked), whereas `n > 1` returns **N** token streams (each paying full inference cost). 8 is the largest value where the scheduler budget stays predictable for typical LiteLLM/LangChain workloads (2–4 candidates is the common case; 8 is headroom for niche comparison workflows).

#### 4.6.2 New cross-field checks on `n > 1`

```rust
// In validate_chat_request_fields (no echo on chat — chat has no echo field per OpenAI spec)
// (no change needed)

// In validate_completion_request_fields:
if let Some(n) = req.n {
    if n > 1 {
        if req.echo == Some(true) {
            return Err(ValidationError::new(
                "echo = true is not compatible with n > 1 (echo applies to a single completion)".to_string(),
            ));
        }
        if req.suffix.is_some() {
            return Err(ValidationError::new(
                "suffix is not compatible with n > 1 (suffix applies to a single completion)".to_string(),
            ));
        }
    }
}
```

### 4.7 Type Layer (`crates/traits`)

**No changes.** `ChatRequest::n: Option<i64>` and `CompletionRequest::n: Option<i64>` already exist (P22 / P32). The validator's new upper-bound and cross-field checks are pure HTTP-layer logic.

### 4.8 Engine Layer (`crates/core`)

**No changes.** The engine already supports N parallel `EngineMessage::AddRequest` invocations (P37 `best_of` does exactly this). The new `run_n_parallel_chat` / `run_n_parallel_completions` helpers send N independent requests through the existing `EngineMessage` channel; the engine schedules each on its own sequence ID with its own `SamplingParams`.

### 4.9 Test Coverage

#### 4.9.1 Unit tests (`crates/server/src/openai/{chat,completions}/tests.rs`)

- `test_per_candidate_seed_none_propagates_none`
- `test_per_candidate_seed_zero_index_is_identity`
- `test_per_candidate_seed_wraps_on_overflow` (i64::MAX as u64 + 1)
- `test_per_candidate_seed_distinguishes_candidates` (seed=42, i=0 vs i=1)
- `test_spawn_n_candidate_uses_distinct_seq_id_per_index`
- `test_collect_n_candidates_returns_n_results_in_order` (even if completion order is non-deterministic)
- `test_assemble_streaming_event_skips_empty_candidates`
- `test_assemble_streaming_event_emits_finish_reason_per_index`
- `test_n_short_circuits_to_single_shot_path` (mock — verify `run_n_parallel_chat` is NOT called when `n = 1`)
- `test_n_one_zero_overhead_baseline` (no measurable regression vs pre-P39 path)

#### 4.9.2 Validator unit tests (`crates/server/src/openai/sampling_validation.rs::tests`)

- `test_chat_n_at_upper_bound_passes` (n = 8)
- `test_chat_n_above_upper_bound_is_rejected` (n = 9 → 400)
- `test_chat_n_well_above_upper_bound_is_rejected` (n = 1000 → 400)
- `test_chat_n_negative_still_rejected` (n = -1 → 400; existing rule unchanged)
- `test_chat_n_zero_still_rejected` (n = 0 → 400; existing rule unchanged)
- `test_completions_n_at_upper_bound_passes` (n = 8)
- `test_completions_n_above_upper_bound_is_rejected` (n = 9 → 400)
- `test_completions_n_with_echo_true_returns_400` (n = 2, echo = true)
- `test_completions_n_with_suffix_returns_400` (n = 2, suffix = Some(_))
- `test_completions_n_with_best_of_returns_400` (n = 2, best_of = 2; existing rule, unchanged)

#### 4.9.3 Integration tests (`crates/server/tests/chat_integration_test.rs`)

- `test_chat_n_one_is_noop_baseline` (single-choice response shape)
- `test_chat_n_above_one_returns_n_choices` (n = 2 → 2 choices)
- `test_chat_n_above_one_choices_have_distinct_indices` (index 0, index 1)
- `test_chat_n_above_one_choices_have_distinct_text` (same prompt + n = 2 → different completions due to non-greedy sampling)
- `test_chat_n_above_one_with_logprobs_returns_per_choice_logprobs`
- `test_chat_n_above_one_with_stop_returns_n_truncated_responses` (each choice honors stop independently)
- `test_chat_n_above_one_with_seed_produces_deterministic_choices`
- `test_chat_n_above_one_streaming_emits_n_choices_per_event` (SSE shape)
- `test_chat_n_above_eight_returns_400` (n = 9 → 400)
- `test_chat_n_with_best_of_returns_400` (cross-field)
- `test_completions_n_one_is_noop_baseline`
- `test_completions_n_above_one_returns_n_choices`
- `test_completions_n_above_one_with_logprobs_returns_per_choice_logprobs`
- `test_completions_n_above_one_with_stop_returns_n_truncated_responses`
- `test_completions_n_above_one_with_echo_returns_400`
- `test_completions_n_above_one_with_suffix_returns_400`
- `test_completions_n_above_one_with_best_of_returns_400`
- `test_completions_n_above_eight_returns_400`

#### 4.9.4 Wire-shape tests

- `test_chat_n_two_response_wire_shape` — pins the exact JSON layout (`choices[0]`, `choices[1]`, `usage.prompt_tokens`, `usage.completion_tokens = sum across N candidates`, `usage.total_tokens`)
- `test_chat_n_two_streaming_wire_shape` — pins the SSE event shape (one event per token round across N candidates, with `choices: [{index, delta, finish_reason}, ...]`)
- `test_completions_n_two_response_wire_shape` — pins the legacy-completions JSON layout (`choices[0]`, `choices[1]`)

### 4.10 Documentation Updates

#### 4.10.1 `docs/reference/openai-compatibility.md`

- Flip the `n` row in **both** the chat table and the completions table from **"Wired (validation)"** → **"Wired (declaration + validation + engine wire-through)"** with the `<= 8` upper bound + per-candidate seed derivation + streaming event shape documented in the row notes.
- Remove the `n` row from the **v32+ candidates table** (now closed).
- Add a new **v31.0 closed items** callout noting that P39 closes the v0.x wire-type backlog entirely.

#### 4.10.2 `CHANGELOG.md`

- New entry under "v31.0 / P39": `n > 1` engine wire-through summary (mirroring the P37 / P38 / P36 entry style — bullet points on helpers, cross-field rules, test count, public-API delta).

#### 4.10.3 `.planning/v31.0-MASTER-PLAN.md`

- Add P39 row to the 31-F (Performance — completion) or new 31-G (Engine Honoring — completion) sub-section.
- Mark "v0.x wire-type backlog FULLY CLOSED" prominently in the **v31.0 ship criteria** section.
- Move `n > 1` from the "Deferred to v32+" section → "Shipped in P39" section.

#### 4.10.4 `.planning/STATE.md`

- Append P39 entry to `last_activity`, summarising the helpers + tests + cross-field rules + the "v0.x wire-type backlog FULLY CLOSED" milestone.
- Update `status` from `in_progress` → `complete` (or `shipping` if follow-up ceremony is pending).

## 5. Architecture Rationale

### 5.1 Why N parallel `EngineMessage::AddRequest` (not a single batched sequence)?

**Decision:** N parallel `EngineMessage::AddRequest`, one per candidate.

**Rationale:**

- **Reuses P37 verbatim.** P37 already implemented this pattern with full test coverage (best_of + partial engine failure + logprobs + suffix + per-candidate independence). Code reuse = lower risk + faster delivery.
- **No engine refactor.** The engine already supports N independent `seq_id`s; the scheduler can pack them into a single step batch automatically (continuous batching).
- **Clean failure isolation.** If one candidate fails (e.g. partial engine failure on a long context), the other N-1 candidates continue. The P37 partial-failure test pattern (`test_completions_best_of_with_partial_engine_failure_returns_503`) is reused — promoted to a generic helper that returns N successful candidates + the failure status.
- **Composable with `seed`.** Per-candidate seed derivation requires distinct `SamplingParams::seed` per call, which is naturally expressed as N independent `AddRequest` messages.

**Rejected alternative:** Single `AddRequest` carrying `n` in `SamplingParams`, with the engine spawning N sub-sequences internally. This would require engine refactor (new `SamplingParams::n`, scheduler changes, batch reshape), is a wire-breaking change (touches `vllm_core`), and is harder to test in isolation. P37 already proved the parallel-spawn pattern works; no reason to deviate.

### 5.2 Why per-candidate seed derivation (`seed.wrapping_add(i)`)?

**Decision:** `per_candidate_seed(seed, i) = seed.map(|s| s.wrapping_add(i as u64))`.

**Rationale:**

- **Matches `sample_batch_with_params` semantics.** P34's seed wire-through guarantees per-sequence independence: sequences with the same seed re-seed independently, sequences with different seeds draw different thresholds. Extending this to `n > 1` means each candidate is a "sequence" with a deterministic derived seed — same property, same guarantee.
- **Avoids duplicate outputs.** If all N candidates shared the same seed, non-greedy sampling would produce identical outputs (since `StdRng::seed_from_u64` is deterministic). N identical candidates defeats the purpose of `n > 1`.
- **Deterministic + repeatable.** Given the same `seed`, the N candidate outputs are reproducible across runs (modulo scheduler non-determinism in token arrival order, which is irrelevant for correctness).
- **Compatible with `seed = None`.** When `seed = None`, each candidate falls back to the thread-local default RNG (P34's "independent per-sequence" contract) — N independent random draws.

**Rejected alternative:** All candidates share the same seed (results in N identical completions for non-greedy sampling — useless).

**Rejected alternative:** New `per_candidate_seeds: Vec<u64>` API field. OpenAI does not expose this; v0.x lite has no precedent. YAGNI.

### 5.3 Why cap `n` at 8 (not 20 like `best_of`)?

**Decision:** `n > 8` rejected with 400.

**Rationale:** See §4.6.1 table. The cap is lower than `best_of` because `n > 1` returns **all** N candidates (each paying full inference cost) whereas `best_of` returns **1** candidate (after ranking). 8 is the largest value where the scheduler budget stays predictable for typical LiteLLM/LangChain workloads (2–4 candidates is the common case; 8 is headroom).

### 5.4 Why reject `n > 1` × `echo` / `n > 1` × `suffix`?

**Decision:** 400 with descriptive error message.

**Rationale:**

- **OpenAI spec incompatibility.** OpenAI's `echo` and `suffix` apply to a **single** completion; they have no defined semantics when `n > 1` (which echo? which suffix?). Rejecting is the spec-correct behavior.
- **Avoids ambiguous rendering.** A silent "echo only the first candidate" or "suffix only the first candidate" behavior would violate the principle of least surprise.
- **Symmetric with existing `best_of` rule.** P32 already rejected `echo = true` × `best_of > 1`; this rule is the same pattern, applied to `n > 1`.

**Rejected alternative:** Silently echo the first candidate. Violates OpenAI spec + surprises users.

## 6. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| N parallel candidates oversubscribe the scheduler | Medium | Medium | Hard cap `n <= 8` (§4.6.1); integration test `test_chat_n_eight_max_load_stays_under_scheduler_budget` |
| Streaming event assembly drops tokens under high N | Low | High | Lock-free arrival-order merge via tokio mpsc; wire-shape test pins SSE event payload; per-candidate `response_rx` decouples candidate lifecycles |
| Per-candidate seed derivation collides (seed = i64::MIN wraps to 0) | Low | Low | `wrapping_add` is intentional; `i = 0` ⇒ identity is the desired behavior; `wrapping_add` is the documented contract |
| P37 partial-failure regression when N > 2 | Low | High | Reuse P37's `spawn_best_of_candidate` partial-failure handler verbatim; promote `test_completions_best_of_with_partial_engine_failure_returns_503` to apply for any N |
| Cross-field `n × echo / suffix` validator rule breaks existing clients | Low | Low | Both fields default to `None` / `false` (per OpenAI spec); only clients explicitly setting `n > 1` AND `echo = true` / `suffix = Some(_)` are affected; these clients have a spec-incompatible request anyway |
| `n = 1` regression (subtle change to single-shot path) | Low | High | Explicit `test_chat_n_one_is_noop_baseline` and `test_completions_n_one_is_noop_baseline` integration tests; `n = 1` short-circuits to existing P38 single-shot handler (no new code path) |
| `n > 8` cap surprises users who set `n = 16` expecting OpenAI behavior | Low | Low | Error message names the cap (`"n = 16 exceeds maximum allowed value of 8 (n = 1..=8)"`); `docs/reference/openai-compatibility.md` documents the cap prominently in the `n` row notes |

## 7. Success Criteria

- [ ] `n = 2` on `/v1/chat/completions` returns 2 distinct choices (verified by integration test + manual curl smoke)
- [ ] `n = 2` on `/v1/completions` returns 2 distinct choices
- [ ] `n = 2` + streaming on chat emits SSE events with `choices: [{index: 0, ...}, {index: 1, ...}, ...]`
- [ ] `n = 9` rejected with 400 on both endpoints
- [ ] `n = 2` + `echo = true` rejected with 400 on completions
- [ ] `n = 2` + `suffix = Some(_)` rejected with 400 on completions
- [ ] `n = 2` + `best_of = 2` rejected with 400 (existing rule, unchanged)
- [ ] `n = 2` + `logprobs = true` returns per-choice logprobs
- [ ] `n = 2` + `stop = Some([...])` returns per-choice truncated responses
- [ ] `n = 2` + `seed = 42` produces deterministic outputs across runs
- [ ] `n = 1` short-circuits to single-shot path with **zero measurable regression** vs pre-P39 baseline (validated by `test_chat_n_one_is_noop_baseline` + benchmark)
- [ ] `just ci` is green on workspace
- [ ] `docs/reference/openai-compatibility.md` flipped + v32+ candidates table updated
- [ ] `CHANGELOG.md` has P39 entry mirroring P37 / P38 style
- [ ] `.planning/v31.0-MASTER-PLAN.md` marks v0.x wire-type backlog as FULLY CLOSED
- [ ] Public-API surface increase = 0 (verified by `public-api` CI gate)

## 8. Out-of-Scope Follow-Ups (v32+ candidates)

These remain **explicitly deferred** to v32+ after P39 lands:

- **`tools` / `tool_choice` engine honoring** — grammar-constrained decoder (JSON schema → grammar); v32+ candidate per master plan.
- **`response_format: json_object` engine honoring** — constrained-decoder hook; v32+ candidate per master plan.
- **Multi-node engine wiring** — deferred from v31-D per OPS-31d §7.
- **Long context >32K end-to-end** — NMC-01.
- **Vision encoder** — NMC-02.
- **Real GPU benchmark suite** — OPS-04.
- **True NCCL AllReduce** — deferred from v31-D.

After P39, the **only** remaining OpenAI-compat engine-honoring work on the chat + completions endpoints is `tools` / `tool_choice` / `response_format`, all of which require grammar-constrained decoders — categorically different work (new dependency + new error model + new test surface), scoped to v32+.

## 9. Decision Log

| Decision | Rationale | Date |
|----------|-----------|------|
| N parallel `EngineMessage::AddRequest` (vs single batched sequence) | Reuses P37 verbatim; no engine refactor | 2026-07-21 |
| Per-candidate seed derivation (`seed.wrapping_add(i)`) | Matches P34 per-sequence independence; deterministic; avoids duplicate outputs | 2026-07-21 |
| Cap `n <= 8` (vs 20 / 128) | Protects scheduler; covers LiteLLM/LangChain practice | 2026-07-21 |
| Reject `n > 1` × `echo = true` | OpenAI spec incompatibility | 2026-07-21 |
| Reject `n > 1` × `suffix = Some(_)` | OpenAI spec incompatibility | 2026-07-21 |
| Streaming event shape = one event per token round, `choices: [{index, delta, finish_reason}, ...]` | OpenAI convention; matches existing chat streaming shape with `index` field added | 2026-07-21 |
| Public-API delta = 0 | `ChatRequest::n` and `CompletionRequest::n` already declared (P22 / P32); validator changes are HTTP-layer only | 2026-07-21 |
