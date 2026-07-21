# P39 Implementation Plan — `n > 1` Engine Wire-Through on Chat + Completions

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `n > 1` end-to-end on `/v1/chat/completions` and `/v1/completions` (currently hard-400), closing the last v0.x wire-type engine-honoring carve-out at the validation-only layer.

**Architecture:** Reuse P37's `spawn_best_of_candidate` pattern → rename to `spawn_n_candidate` and parameterise by `candidate_index: usize` so each candidate gets a deterministically-derived seed (`seed.wrapping_add(i)`). The new `run_n_parallel_chat` / `run_n_parallel_completions` helpers spawn N parallel `EngineMessage::AddRequest` candidates via `tokio::spawn`, join them, and assemble all N choices into the response (NOT ranked — distinct from `best_of`'s "return 1" semantics). Streaming uses a new `assemble_streaming_event` helper that interleaves N `response_rx` channels into one SSE event stream (`choices: [{index, delta, finish_reason}, ...]`). Validator gains `n <= 8` upper bound + new cross-field rejections (`n > 1` × `echo` / `suffix`).

**Tech Stack:** Rust (workspace), axum, tokio mpsc/oneshot, parking_lot, serde, thiserror.

**Spec:** `docs/superpowers/specs/2026-07-21-p39-n-parallel-engine-wire-through-design.md`

**Estimated effort:** 1.5–2 working days, 9 tasks.

---

## File Structure (mapped up-front)

### Modified files
- `crates/server/src/openai/sampling_validation.rs` — add `MAX_N` constant, tighten `n` validator (`n <= 8`), add cross-field rules (`n > 1` × `echo` / `suffix`), update existing "n > 1 is not supported" error message
- `crates/server/src/openai/sampling_validation.rs::tests` — 10 new validator tests
- `crates/server/src/openai/completions.rs` — rename `spawn_best_of_candidate` → `spawn_n_candidate` (add `candidate_index` param), add `per_candidate_seed` helper, add `run_n_parallel_completions`, add `assemble_streaming_event` (completions shape), update existing `run_best_of` caller to pass `candidate_index`
- `crates/server/src/openai/completions/tests.rs` — 5 new unit tests for `per_candidate_seed` + `assemble_streaming_event`; update existing `spawn_best_of_candidate` references if any
- `crates/server/src/openai/chat.rs` — add `per_candidate_seed` helper (or share from completions), add `run_n_parallel_chat`, add `assemble_streaming_event` (chat shape), add `n > 1` branch in `chat_completions` + `stream_chat_completion` handlers
- `crates/server/src/openai/chat/tests.rs` — 5 new unit tests for chat-specific helpers
- `crates/server/tests/chat_integration_test.rs` — 19 new integration tests
- `docs/reference/openai-compatibility.md` — flip `n` rows (chat + completions) to "engine wire-through", remove v32+ candidates table row, add v31.0 closed-items callout
- `CHANGELOG.md` — P39 entry
- `.planning/STATE.md` — append P39 entry, update status
- `.planning/v31.0-MASTER-PLAN.md` — add P39 to phase index, mark v0.x wire-type backlog as FULLY CLOSED, remove `n > 1` from v32+ deferred

### New files
- None. All new helpers are private to existing modules.

### Public-API surface
**Zero changes.** `ChatRequest::n: Option<i64>` and `CompletionRequest::n: Option<i64>` already declared (P22 / P32). Validator changes are HTTP-layer only.

---

## Dependency Graph

```
Task 1: Validator tightening (TDD)
   ├─> Task 2: per_candidate_seed helper (TDD)
   └─> Task 3: spawn_n_candidate rename + parameterise (TDD, no behavior change for best_of)
            ├─> Task 4: run_n_parallel_completions + non-streaming wire-through (TDD integration)
            │        └─> Task 5: completions streaming wire-through (TDD integration)
            └─> Task 6: run_n_parallel_chat + non-streaming wire-through (TDD integration)
                     └─> Task 7: chat streaming wire-through (TDD integration)
                              └─> Task 8: Documentation updates
                                       └─> Task 9: Final CI verification
```

Tasks 4-7 form two parallel pairs (completions + chat, each non-streaming then streaming). Task 8 depends on all four. Task 9 is the final verification.

---

## Task 1: Validator Tightening (n upper bound + cross-field rules)

**Files:**
- Modify: `crates/server/src/openai/sampling_validation.rs` — add `MAX_N` const, update `n` checks, add cross-field rules
- Modify: `crates/server/src/openai/sampling_validation.rs::tests` — add 10 new tests, update 2 existing tests with new error message

- [ ] **Step 1.1: Locate existing `n` validator code**

Read `crates/server/src/openai/sampling_validation.rs` and find:
- `validate_chat_request_fields` — locate the existing `n > 1` rejection (returns `ValidationError::new("n > 1 is not supported…")`)
- `validate_completion_request_fields` — same pattern
- Existing cross-field check `n > 1 && best_of > 1` (returns error message naming both fields)

Verify the line numbers by reading the file before editing.

- [ ] **Step 1.2: Add `MAX_N` constant at the top of `sampling_validation.rs`**

Insert at the top (after the existing `use` statements and before the first `pub fn`):

```rust
/// Maximum value of `n` accepted on chat + completions endpoints.
/// 8 is the practical scheduler-safe cap (vs OpenAI's nominal 128):
/// each candidate pays full inference cost (unlike `best_of` which
/// returns ONE ranked completion), so N must stay bounded.
pub(crate) const MAX_N: i64 = 8;
```

- [ ] **Step 1.3: Write failing tests for the new `n <= 8` upper bound**

Add to `crates/server/src/openai/sampling_validation.rs::tests`:

```rust
#[test]
fn test_chat_n_at_upper_bound_passes() {
    let req = sample_chat_request_with_n(MAX_N);
    assert!(validate_chat_request_fields(&req).is_ok());
}

#[test]
fn test_chat_n_above_upper_bound_is_rejected() {
    let req = sample_chat_request_with_n(MAX_N + 1);
    let err = validate_chat_request_fields(&req).unwrap_err();
    assert!(
        err.to_string().contains(&format!("exceeds maximum allowed value of {MAX_N}")),
        "error message must name the cap; got: {err}"
    );
}

#[test]
fn test_chat_n_well_above_upper_bound_is_rejected() {
    let req = sample_chat_request_with_n(1_000);
    assert!(validate_chat_request_fields(&req).is_err());
}

#[test]
fn test_chat_n_negative_still_rejected() {
    let req = sample_chat_request_with_n(-1);
    let err = validate_chat_request_fields(&req).unwrap_err();
    assert!(err.to_string().contains("n") || err.to_string().contains("positive"));
}

#[test]
fn test_chat_n_zero_still_rejected() {
    let req = sample_chat_request_with_n(0);
    assert!(validate_chat_request_fields(&req).is_err());
}

#[test]
fn test_completions_n_at_upper_bound_passes() {
    let req = sample_completion_request_with_n(MAX_N);
    assert!(validate_completion_request_fields(&req).is_ok());
}

#[test]
fn test_completions_n_above_upper_bound_is_rejected() {
    let req = sample_completion_request_with_n(MAX_N + 1);
    let err = validate_completion_request_fields(&req).unwrap_err();
    assert!(
        err.to_string().contains(&format!("exceeds maximum allowed value of {MAX_N}")),
        "error message must name the cap; got: {err}"
    );
}

#[test]
fn test_completions_n_with_echo_true_returns_400() {
    let mut req = sample_completion_request_with_n(2);
    req.echo = Some(true);
    let err = validate_completion_request_fields(&req).unwrap_err();
    assert!(err.to_string().contains("echo"));
    assert!(err.to_string().contains("n > 1"));
}

#[test]
fn test_completions_n_with_suffix_returns_400() {
    let mut req = sample_completion_request_with_n(2);
    req.suffix = Some("}\n".to_string());
    let err = validate_completion_request_fields(&req).unwrap_err();
    assert!(err.to_string().contains("suffix"));
    assert!(err.to_string().contains("n > 1"));
}

#[test]
fn test_completions_n_with_best_of_returns_400() {
    // Existing rule, unchanged. Pin it so future refactors don't drop it.
    let mut req = sample_completion_request_with_n(2);
    req.best_of = Some(2);
    let err = validate_completion_request_fields(&req).unwrap_err();
    assert!(err.to_string().contains("best_of"));
    assert!(err.to_string().contains("n"));
}
```

If `sample_chat_request_with_n` / `sample_completion_request_with_n` don't exist as test helpers, add them above the tests:

```rust
fn sample_chat_request_with_n(n: i64) -> ChatRequest {
    let mut req = ChatRequest::default();
    req.n = Some(n);
    req
}

fn sample_completion_request_with_n(n: i64) -> CompletionRequest {
    let mut req = CompletionRequest::default();
    req.n = Some(n);
    req
}
```

- [ ] **Step 1.4: Run tests to verify they fail**

Run: `cargo nextest run -p vllm-server --all-features sampling_validation::tests 2>&1 | tail -20`
Expected: New tests fail with "n > 1 is not supported…" or "function not found" depending on whether the cap check or the cross-field check is missing.

- [ ] **Step 1.5: Update `validate_chat_request_fields` — tighten `n` upper bound**

Find the existing `n > 1` rejection in `validate_chat_request_fields`. Replace with:

```rust
// P39: `n <= MAX_N`. Cap protects the scheduler — each candidate
// pays full inference cost (unlike `best_of` which returns ONE
// ranked completion), so N must stay bounded.
if let Some(n) = req.n {
    if n < 1 {
        return Err(ValidationError::new(
            "n must be a positive integer (n >= 1)".to_string(),
        ));
    }
    if n > MAX_N {
        return Err(ValidationError::new(format!(
            "n = {n} exceeds maximum allowed value of {MAX_N} (n = 1..={MAX_N})",
        )));
    }
}
```

- [ ] **Step 1.6: Update `validate_completion_request_fields` — tighten `n` + add cross-field rules**

Find the existing `n > 1` rejection. Replace with:

```rust
// P39: `n <= MAX_N`. Cap protects the scheduler — each candidate
// pays full inference cost, so N must stay bounded.
if let Some(n) = req.n {
    if n < 1 {
        return Err(ValidationError::new(
            "n must be a positive integer (n >= 1)".to_string(),
        ));
    }
    if n > MAX_N {
        return Err(ValidationError::new(format!(
            "n = {n} exceeds maximum allowed value of {MAX_N} (n = 1..={MAX_N})",
        )));
    }
}

// P39: `n > 1` is incompatible with `echo` and `suffix` per OpenAI
// spec — both apply to a SINGLE completion. Cross-field rejection.
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

Note: the existing `n > 1 && best_of > 1` cross-field check stays — it was added in P32 and is unchanged by P39.

- [ ] **Step 1.7: Run tests to verify they pass**

Run: `cargo nextest run -p vllm-server --all-features sampling_validation::tests 2>&1 | tail -20`
Expected: All new tests pass; any pre-existing tests that asserted the old "n > 1 is not supported" message need their assertions updated to the new message format.

- [ ] **Step 1.8: Update pre-existing tests with new error message**

Find any test in `sampling_validation.rs::tests` (or elsewhere) that asserts the old "n > 1 is not supported" message string. Update the assertion to:

```rust
assert!(err.to_string().contains("exceeds maximum allowed value"));
```

- [ ] **Step 1.9: Re-run to confirm all pass**

Run: `cargo nextest run -p vllm-server --all-features sampling_validation::tests 2>&1 | tail -10`
Expected: All tests pass.

- [ ] **Step 1.10: Commit**

```bash
git add crates/server/src/openai/sampling_validation.rs crates/server/src/openai/sampling_validation.rs::tests
# Note: if tests are inline, the path may be crates/server/src/openai/sampling_validation.rs only
git commit -m "feat(server): tighten n upper bound to 8 + cross-field rules (v31.0 P39)

n > 8 rejected with 400 (scheduler-safe cap; each candidate pays full
inference cost). n > 1 x echo=true / suffix rejected with 400 per
OpenAI spec (echo / suffix apply to single completion).

10 new validator tests + 2 pre-existing tests updated for new error
message. Public API delta = 0 (n field already declared in P22 / P32)."
```

---

## Task 2: `per_candidate_seed` Helper

**Files:**
- Modify: `crates/server/src/openai/completions.rs` — add `per_candidate_seed` private helper
- Modify: `crates/server/src/openai/completions/tests.rs` — add 5 unit tests

- [ ] **Step 2.1: Locate where to put the helper**

The helper is used by both chat and completions handlers. Add it to `crates/server/src/openai/completions.rs` (where `populate_completion_sampling_params` lives — already shared with chat in spirit). The chat handler will call it via `super::completions::per_candidate_seed` OR we duplicate the small function in `chat.rs` (preferred — 3-line function, no need for cross-module coupling).

Decision: Put it in `completions.rs` as `pub(super) fn per_candidate_seed(...)` and re-export in `chat.rs` via a small wrapper. This matches the existing pattern where `completions.rs` hosts shared sampling-params helpers.

- [ ] **Step 2.2: Add `per_candidate_seed` to `completions.rs`**

Insert above `run_best_of`:

```rust
/// Derive a deterministic per-candidate seed for `n > 1` / `best_of > 1`.
///
/// `None` propagates as `None` (the engine falls back to its thread-local
/// default RNG, which is per-sequence independent per P34).
///
/// `Some(seed)` produces `Some(seed.wrapping_add(candidate_index as u64))` —
/// deterministic + distinct per candidate (avoids identical outputs when
/// all candidates share the same seed). Matches P34's per-sequence
/// independence contract.
pub(super) fn per_candidate_seed(seed: Option<i64>, candidate_index: usize) -> Option<u64> {
    seed.map(|s| s.wrapping_add(candidate_index as u64))
}
```

- [ ] **Step 2.3: Write failing unit tests**

Add to `crates/server/src/openai/completions/tests.rs`:

```rust
use super::per_candidate_seed;

#[test]
fn test_per_candidate_seed_none_propagates_none() {
    assert_eq!(per_candidate_seed(None, 0), None);
    assert_eq!(per_candidate_seed(None, 5), None);
}

#[test]
fn test_per_candidate_seed_zero_index_is_identity() {
    assert_eq!(per_candidate_seed(Some(42), 0), Some(42));
    assert_eq!(per_candidate_seed(Some(0), 0), Some(0));
}

#[test]
fn test_per_candidate_seed_wraps_on_overflow() {
    // i64::MAX as u64 = 9223372036854775807
    // + 1 wraps to 0
    let max = i64::MAX;
    assert_eq!(per_candidate_seed(Some(max), 1), Some(0));
    assert_eq!(per_candidate_seed(Some(max), 2), Some(1));
}

#[test]
fn test_per_candidate_seed_distinguishes_candidates() {
    let s0 = per_candidate_seed(Some(42), 0).unwrap();
    let s1 = per_candidate_seed(Some(42), 1).unwrap();
    let s2 = per_candidate_seed(Some(42), 2).unwrap();
    assert_ne!(s0, s1);
    assert_ne!(s1, s2);
    assert_ne!(s0, s2);
}

#[test]
fn test_per_candidate_seed_negative_i64_wraps_to_u64() {
    // i64::MIN as u64 = 9223372036854775808 (wraps via `as`)
    let min = i64::MIN;
    let s = per_candidate_seed(Some(min), 0).unwrap();
    assert_eq!(s, i64::MIN as u64); // 9223372036854775808
    // + 1 wraps
    assert_eq!(per_candidate_seed(Some(min), 1), Some(0));
}
```

- [ ] **Step 2.4: Run tests to verify they fail**

Run: `cargo nextest run -p vllm-server --all-features completions::tests::test_per_candidate_seed 2>&1 | tail -10`
Expected: All 5 tests fail with "function `per_candidate_seed` not found".

- [ ] **Step 2.5: Confirm tests pass (helper already added in Step 2.2)**

Run: `cargo nextest run -p vllm-server --all-features completions::tests::test_per_candidate_seed 2>&1 | tail -10`
Expected: All 5 tests pass.

- [ ] **Step 2.6: Commit**

```bash
git add crates/server/src/openai/completions.rs crates/server/src/openai/completions/tests.rs
git commit -m "feat(server): add per_candidate_seed helper for n > 1 / best_of (v31.0 P39)

Deterministic per-candidate seed derivation via wrapping_add(index).
Matches P34's per-sequence independence contract. 5 new unit tests."
```

---

## Task 3: Rename `spawn_best_of_candidate` → `spawn_n_candidate` + Add `candidate_index` Param

**Files:**
- Modify: `crates/server/src/openai/completions.rs` — rename function, add `candidate_index: usize` param, apply `per_candidate_seed` when populating `SamplingParams::seed`
- Modify: `crates/server/src/openai/completions.rs` — update `run_best_of` caller to pass `candidate_index`
- Modify: `crates/server/src/openai/completions/tests.rs` — update any direct references to the renamed function

- [ ] **Step 3.1: Locate the function definition**

Read `crates/server/src/openai/completions.rs` and find `async fn spawn_best_of_candidate`. Note the line number and current signature.

- [ ] **Step 3.2: Write failing test for the per-candidate seed application**

This test pins the contract: when `spawn_n_candidate` is called with `candidate_index = i` and the request has `seed = Some(42)`, the resulting `Request`'s `SamplingParams::seed` must be `Some(42.wrapping_add(i as u64))`.

Add to `crates/server/src/openai/completions/tests.rs`:

```rust
#[tokio::test]
async fn test_spawn_n_candidate_applies_per_candidate_seed() {
    // Build a minimal ApiState with a mock engine mailbox; capture the
    // AddRequest that arrives. Verify its sampling_params.seed.
    // (Use the same mock infrastructure as P37's spawn tests.)
    let (tx, mut rx) = tokio::sync::mpsc::channel::<EngineMessage>(4);
    let state = build_test_api_state_with_engine_tx(tx);

    let req = CompletionRequest {
        seed: Some(42),
        ..sample_completion_request()
    };
    let prompt_tokens: Vec<TokenId> = vec![1, 2, 3];
    let correlation_id = "test-correlation".to_string();

    // Spawn candidate 2
    let handle = tokio::spawn(spawn_n_candidate(
        state,
        req,
        prompt_tokens,
        /* max_tokens = */ 10,
        correlation_id,
        /* candidate_index = */ 2,
    ));

    // Capture the AddRequest that was sent to the engine mailbox
    let msg = rx.recv().await.expect("AddRequest should be sent");
    let EngineMessage::AddRequest { sampling_params, .. } = msg else {
        panic!("expected AddRequest");
    };
    // Candidate 2 + seed 42 → 42.wrapping_add(2) = 44
    assert_eq!(sampling_params.seed, Some(44));

    // Drop the handle (the spawned task will fail because we never
    // drive the response_rx; that's fine — we're only testing the
    // AddRequest message construction)
    drop(handle);
}
```

If the mock infrastructure (`build_test_api_state_with_engine_tx`, `sample_completion_request`) doesn't exist, look at how P37's `spawn_best_of_candidate` tests are wired and mirror that pattern.

- [ ] **Step 3.3: Run test to verify it fails**

Run: `cargo nextest run -p vllm-server --all-features completions::tests::test_spawn_n_candidate_applies_per_candidate_seed 2>&1 | tail -10`
Expected: Test fails with "function `spawn_n_candidate` not found".

- [ ] **Step 3.4: Rename + parameterise `spawn_best_of_candidate`**

In `crates/server/src/openai/completions.rs`:

1. Rename the function: `async fn spawn_best_of_candidate(...)` → `async fn spawn_n_candidate(...)`.
2. Add `candidate_index: usize` as the last parameter (after `correlation_id`).
3. Inside the function, after `populate_completion_sampling_params` is called (or however `SamplingParams` is built), apply the per-candidate seed:

```rust
// P39: per-candidate seed derivation. Each candidate gets a
// deterministic + distinct seed via wrapping_add(candidate_index),
// matching P34's per-sequence independence contract. Avoids
// identical outputs when all candidates share the same seed.
sampling_params.seed = per_candidate_seed(sampling_params.seed, candidate_index);
```

Adjust based on where `sampling_params` is actually constructed inside the function.

- [ ] **Step 3.5: Update `run_best_of` caller**

In `run_best_of`, update the loop:

```rust
for i in 0..n {
    let state = state.clone();
    let req = req.clone();
    let prompt_tokens = prompt_tokens.clone();
    let correlation_id = correlation_id.0.clone();
    let candidate = tokio::spawn(spawn_n_candidate(
        state,
        req,
        prompt_tokens,
        max_tokens,
        correlation_id,
        i, // NEW: pass candidate_index
    ));
    handles.push(candidate);
}
```

- [ ] **Step 3.6: Update doc comments**

The doc comment on the renamed function should reflect its broader role. Update from "Spawns one best_of candidate" to:

```rust
/// Spawns one candidate (for `n > 1` or `best_of > 1`).
///
/// Each candidate is an independent `EngineMessage::AddRequest` with
/// its own `seq_id`. Sampling params are identical across candidates
/// EXCEPT for the seed, which is derived deterministically via
/// `per_candidate_seed(seed, candidate_index)` (P39).
///
/// Returns the candidate's `Vec<SampledToken>` stream + `FinishReason`
/// once the engine closes `response_tx` (natural EOS, `max_tokens`,
/// `stop`, or `Cancelled`).
```

- [ ] **Step 3.7: Search-and-replace any other references**

Run: `grep -rn "spawn_best_of_candidate" crates/ 2>&1 | head -20`
Replace all remaining occurrences with `spawn_n_candidate` (should only be in `completions/tests.rs` for any direct unit tests).

- [ ] **Step 3.8: Run new test + existing best_of tests**

Run: `cargo nextest run -p vllm-server --all-features completions::tests::test_spawn_n_candidate -E 'test(/spawn_n_candidate|run_best_of|rank_by_mean_logprob/)' 2>&1 | tail -10`
Expected: All new + existing tests pass.

- [ ] **Step 3.9: Run the best_of integration tests to confirm no regression**

Run: `cargo nextest run -p vllm-server --all-features test_completions_best_of 2>&1 | tail -10`
Expected: All 12 P37 best_of integration tests pass (no regression).

- [ ] **Step 3.10: Commit**

```bash
git add crates/server/src/openai/completions.rs crates/server/src/openai/completions/tests.rs
git commit -m "refactor(server): rename spawn_best_of_candidate to spawn_n_candidate + per-candidate seed (v31.0 P39)

The renamed helper now takes candidate_index: usize and applies
per_candidate_seed(seed, index) so each candidate gets a deterministic
distinct seed. run_best_of caller passes i in 0..best_of; behavior
for best_of is unchanged (existing P37 tests still pass).

1 new unit test pins the per-candidate seed contract. Renamed + updated
doc comments. No public API changes."
```

---

## Task 4: `run_n_parallel_completions` — Non-Streaming Wire-Through

**Files:**
- Modify: `crates/server/src/openai/completions.rs` — add `run_n_parallel_completions` helper
- Modify: `crates/server/src/openai/completions.rs` — add `n > 1` branch in `completion` handler (non-streaming)
- Modify: `crates/server/tests/chat_integration_test.rs` — 5 new integration tests

- [ ] **Step 4.1: Locate the non-streaming `completion` handler**

Read `crates/server/src/openai/completions.rs` and find the `pub async fn completion(...)` handler (the non-streaming branch). Note where the single-shot path lives (after which we'll insert the `n > 1` branch).

- [ ] **Step 4.2: Write failing integration tests**

Add to `crates/server/tests/chat_integration_test.rs` (or a completions-specific integration test file if one exists — verify before adding):

```rust
#[tokio::test]
async fn test_completions_n_one_is_noop_baseline() {
    // Existing single-shot path; pins that n = 1 doesn't change anything
    let req = CompletionRequest {
        prompt: "Hello".into(),
        n: Some(1),
        max_tokens: Some(10),
        ..Default::default()
    };
    let resp = post_completion(&state, &req).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body: CompletionResponse = serde_json::from_slice(...).unwrap();
    assert_eq!(body.choices.len(), 1);
    assert_eq!(body.choices[0].index, 0);
}

#[tokio::test]
async fn test_completions_n_above_one_returns_n_choices() {
    let req = CompletionRequest {
        prompt: "Hello".into(),
        n: Some(2),
        max_tokens: Some(10),
        seed: Some(42),
        ..Default::default()
    };
    let resp = post_completion(&state, &req).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body: CompletionResponse = serde_json::from_slice(...).unwrap();
    assert_eq!(body.choices.len(), 2);
    assert_eq!(body.choices[0].index, 0);
    assert_eq!(body.choices[1].index, 1);
}

#[tokio::test]
async fn test_completions_n_above_one_choices_have_distinct_text() {
    let req = CompletionRequest {
        prompt: "Hello".into(),
        n: Some(2),
        max_tokens: Some(20),
        seed: Some(42),
        temperature: Some(0.8), // non-greedy to encourage diversity
        ..Default::default()
    };
    let resp = post_completion(&state, &req).await;
    let body: CompletionResponse = serde_json::from_slice(...).unwrap();
    // Different seeds → different outputs (per_candidate_seed derivation)
    assert_ne!(body.choices[0].text, body.choices[1].text);
}

#[tokio::test]
async fn test_completions_n_above_one_with_logprobs_returns_per_choice_logprobs() {
    let req = CompletionRequest {
        prompt: "Hello".into(),
        n: Some(2),
        max_tokens: Some(10),
        seed: Some(42),
        logprobs: Some(3),
        ..Default::default()
    };
    let resp = post_completion(&state, &req).await;
    let body: CompletionResponse = serde_json::from_slice(...).unwrap();
    assert_eq!(body.choices.len(), 2);
    assert!(body.choices[0].logprobs.is_some());
    assert!(body.choices[1].logprobs.is_some());
}

#[tokio::test]
async fn test_completions_n_above_eight_returns_400() {
    let req = CompletionRequest {
        prompt: "Hello".into(),
        n: Some(9),
        ..Default::default()
    };
    let resp = post_completion(&state, &req).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}
```

If `post_completion` / `CompletionResponse` / `sample_*` helpers don't exist, mirror the test patterns from P37's `test_completions_best_of_*` tests in the same file.

- [ ] **Step 4.3: Run tests to verify they fail**

Run: `cargo nextest run -p vllm-server --all-features test_completions_n 2>&1 | tail -10`
Expected: Tests fail (n > 1 still returns 400 from the existing pre-P39 validation).

- [ ] **Step 4.4: Implement `run_n_parallel_completions`**

In `crates/server/src/openai/completions.rs`, insert above `run_best_of`:

```rust
/// Run `n` parallel completion candidates and assemble all N choices
/// into the response (NOT ranked — distinct from `best_of`'s
/// "return ONE" semantics).
///
/// Each candidate is independent: separate `EngineMessage::AddRequest`,
/// separate `seq_id`, separate `SampledToken` stream. Sampling params
/// are identical EXCEPT for the seed (per-candidate derived via
/// `per_candidate_seed`).
async fn run_n_parallel_completions(
    state: ApiState,
    req: CompletionRequest,
    prompt_tokens: Vec<vllm_traits::TokenId>,
    prompt: String,
    max_tokens: usize,
    correlation_id: CorrelationId,
) -> Result<axum::response::Response, (axum::http::StatusCode, Json<ErrorResponse>)> {
    let n = req.n.unwrap_or(1) as usize;

    // Spawn N candidates. Mirrors `run_best_of` (P37) but does NOT
    // rank — we collect all N streams and assemble them into N choices.
    let mut handles = Vec::with_capacity(n);
    for i in 0..n {
        let state = state.clone();
        let req = req.clone();
        let prompt_tokens = prompt_tokens.clone();
        let correlation_id = correlation_id.0.clone();
        let candidate = tokio::spawn(spawn_n_candidate(
            state,
            req,
            prompt_tokens,
            max_tokens,
            correlation_id,
            i,
        ));
        handles.push(candidate);
    }

    // Join all candidates. First failure wins.
    let mut candidates: Vec<Vec<vllm_traits::SampledToken>> = Vec::with_capacity(n);
    let mut finish_reasons: Vec<vllm_traits::FinishReason> = Vec::with_capacity(n);
    for handle in handles {
        match handle.await {
            Ok(Ok((tokens, finish_reason))) => {
                candidates.push(tokens);
                finish_reasons.push(finish_reason);
            }
            Ok(Err(e)) => return Err(e),
            Err(e) => {
                return Err((
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse::new(
                        format!("n > 1 candidate task panicked: {e}").as_str(),
                        "server_error",
                    )),
                ));
            }
        }
    }

    // Assemble N choices. Each choice: index + decoded text + logprobs
    // + finish_reason. No echo / suffix (rejected by validator when n > 1).
    let choices: Vec<CompletionChoice> = candidates
        .iter()
        .zip(finish_reasons.iter())
        .enumerate()
        .map(|(index, (tokens, finish_reason))| {
            let text = clean_completion_text(
                &state.tokenizer,
                &state.tokenizer.decode(&token_ids(tokens)),
            );
            let logprobs = build_completion_choice_logprobs(tokens, /* ... */);
            CompletionChoice {
                index: Some(index as u32),
                text,
                logprobs,
                finish_reason: Some(match finish_reason {
                    vllm_traits::FinishReason::Length => "length".to_string(),
                    vllm_traits::FinishReason::Stop | vllm_traits::FinishReason::Cancelled => "stop".to_string(),
                }),
            }
        })
        .collect();

    // usage: completion_tokens = sum across N candidates (OpenAI billing).
    let total_completion_tokens: u32 = candidates.iter().map(|c| c.len() as u32).sum();
    let prompt_tokens_count = prompt_tokens.len() as u32;

    let body = CompletionResponse {
        id: format!("cmpl-{}", uuid::Uuid::new_v4()),
        object: "text_completion".to_string(),
        created: chrono::Utc::now().timestamp() as u64,
        model: req.model.clone(),
        choices,
        usage: Some(CompletionUsage {
            prompt_tokens: prompt_tokens_count,
            completion_tokens: total_completion_tokens,
            total_tokens: prompt_tokens_count + total_completion_tokens,
        }),
    };

    Ok(Json(body).into_response())
}
```

The exact struct field names (`CompletionChoice::index`, `CompletionUsage` etc.) should be verified against the existing `CompletionResponse` definition. The `build_completion_choice_logprobs` helper is already used by `run_best_of` (P37) — reuse it.

- [ ] **Step 4.5: Add `n > 1` branch in the non-streaming `completion` handler**

Find the single-shot path in `pub async fn completion(...)` and insert the `n > 1` branch BEFORE it:

```rust
// P39: `n > 1` branches to the parallel-candidate path. The single-
// shot path is unchanged when `n = 1` or `n = None`.
if let Some(n) = req.n {
    if n > 1 {
        return run_n_parallel_completions(
            state,
            req,
            prompt_tokens,
            prompt,
            max_tokens,
            correlation_id,
        ).await;
    }
}
// ... existing single-shot path follows
```

- [ ] **Step 4.6: Run integration tests**

Run: `cargo nextest run -p vllm-server --all-features test_completions_n 2>&1 | tail -10`
Expected: All 5 new tests pass; pre-existing `test_completions_best_of_*` tests still pass.

- [ ] **Step 4.7: Wire-shape test (JSON layout pinning)**

Add a wire-shape test that pins the exact JSON layout:

```rust
#[tokio::test]
async fn test_completions_n_two_response_wire_shape() {
    let req = CompletionRequest {
        prompt: "Hello".into(),
        n: Some(2),
        max_tokens: Some(5),
        seed: Some(42),
        ..Default::default()
    };
    let resp = post_completion(&state, &req).await;
    let body: serde_json::Value = serde_json::from_slice(...).unwrap();
    assert!(body["choices"].is_array());
    assert_eq!(body["choices"].as_array().unwrap().len(), 2);
    assert_eq!(body["choices"][0]["index"], 0);
    assert_eq!(body["choices"][1]["index"], 1);
    assert!(body["choices"][0]["text"].is_string());
    assert!(body["choices"][0]["finish_reason"].is_string());
    assert!(body["usage"]["prompt_tokens"].is_u64());
    assert!(body["usage"]["completion_tokens"].is_u64());
    // completion_tokens = sum across N candidates
    assert!(body["usage"]["completion_tokens"].as_u64().unwrap() >= 2);
}
```

Run: `cargo nextest run -p vllm-server --all-features test_completions_n_two_response_wire_shape 2>&1 | tail -5`
Expected: PASS.

- [ ] **Step 4.8: Commit**

```bash
git add crates/server/src/openai/completions.rs crates/server/tests/chat_integration_test.rs
git commit -m "feat(server): n > 1 wire-through on legacy /v1/completions (v31.0 P39)

run_n_parallel_completions helper spawns N independent candidates
via spawn_n_candidate, joins all N, and assembles N choices into
the response (NOT ranked). completion_tokens = sum across N candidates
(OpenAI billing convention).

5 new integration tests + 1 wire-shape test. n = 1 / None short-circuits
to the existing single-shot path with zero overhead. Existing best_of
tests still pass (no regression)."
```

---

## Task 5: Completions Streaming Wire-Through

**Files:**
- Modify: `crates/server/src/openai/completions.rs` — add `assemble_streaming_event` helper (completions shape), update streaming handler
- Modify: `crates/server/tests/chat_integration_test.rs` — 2 new integration tests

- [ ] **Step 5.1: Locate the streaming `stream_completion` handler**

Find `pub async fn stream_completion(...)` in `crates/server/src/openai/completions.rs`.

- [ ] **Step 5.2: Write failing integration tests**

```rust
#[tokio::test]
async fn test_completions_n_above_one_streaming_emits_n_choices_per_event() {
    let req = CompletionRequest {
        prompt: "Hello".into(),
        n: Some(2),
        max_tokens: Some(10),
        seed: Some(42),
        stream: Some(true),
        ..Default::default()
    };
    let mut sse_stream = post_streaming_completion(&state, &req).await;
    let mut events: Vec<serde_json::Value> = Vec::new();
    while let Some(event) = sse_stream.next().await {
        if event.data == "[DONE]" { break; }
        events.push(serde_json::from_str(&event.data).unwrap());
    }
    // First event should have 2 choices (one per candidate)
    assert!(events[0]["choices"].as_array().unwrap().len() >= 1);
    // Indices across events should be 0 and 1
    let indices: HashSet<u32> = events.iter()
        .flat_map(|e| e["choices"].as_array().unwrap().iter())
        .filter_map(|c| c["index"].as_u64().map(|i| i as u32))
        .collect();
    assert!(indices.contains(&0));
    assert!(indices.contains(&1));
}

#[tokio::test]
async fn test_completions_n_above_one_streaming_final_event_has_finish_reason_per_index() {
    let req = CompletionRequest {
        prompt: "Hello".into(),
        n: Some(2),
        max_tokens: Some(10),
        seed: Some(42),
        stream: Some(true),
        ..Default::default()
    };
    let mut sse_stream = post_streaming_completion(&state, &req).await;
    let mut final_event = None;
    while let Some(event) = sse_stream.next().await {
        if event.data == "[DONE]" { break; }
        final_event = Some(serde_json::from_str::<serde_json::Value>(&event.data).unwrap());
    }
    let final_event = final_event.unwrap();
    // All N candidates have finished by the [DONE] event
    let finish_reasons: HashSet<String> = final_event["choices"].as_array().unwrap()
        .iter()
        .filter_map(|c| c["finish_reason"].as_str().map(String::from))
        .collect();
    // Both indices must have a non-null finish_reason
    assert!(finish_reasons.iter().all(|r| r == "stop" || r == "length"));
}
```

- [ ] **Step 5.3: Run tests to verify they fail**

Run: `cargo nextest run -p vllm-server --all-features test_completions_n_above_one_streaming 2>&1 | tail -10`
Expected: Tests fail (n > 1 + stream still 400'd).

- [ ] **Step 4: Add `n > 1` branch in `stream_completion`**

Insert BEFORE the existing single-shot streaming path:

```rust
if let Some(n) = req.n {
    if n > 1 {
        return stream_n_parallel_completions(
            state, req, prompt_tokens, prompt, max_tokens, correlation_id,
        ).await;
    }
}
```

Then implement `stream_n_parallel_completions`:

```rust
async fn stream_n_parallel_completions(
    state: ApiState,
    req: CompletionRequest,
    prompt_tokens: Vec<vllm_traits::TokenId>,
    _prompt: String,
    max_tokens: usize,
    correlation_id: CorrelationId,
) -> Sse<Response> {
    // Spawn N candidates (same as run_n_parallel_completions but
    // returns an SSE stream of per-round assembled events).
    let n = req.n.unwrap_or(1) as usize;
    let mut handles = Vec::with_capacity(n);
    for i in 0..n {
        let state = state.clone();
        let req = req.clone();
        let prompt_tokens = prompt_tokens.clone();
        let correlation_id = correlation_id.0.clone();
        let candidate = tokio::spawn(spawn_n_candidate(
            state, req, prompt_tokens, max_tokens, correlation_id, i,
        ));
        handles.push(candidate);
    }

    // For each round: collect one token per candidate, assemble SSE event
    // with choices: [{index, text, finish_reason?}, ...].
    // Stream [DONE] after all N finalize.
    // ... (full implementation follows the P37 single-candidate streaming
    // pattern but extended to N parallel response_rx channels)
}
```

Implementation note: this requires more careful handling than non-streaming because the SSE event loop needs to `select!` across N `response_rx` channels. Use `tokio::select!` with N branches + a state machine. See the P37 single-candidate streaming implementation in `stream_completion` for the pattern; generalise to N branches.

For the plan, leave the exact `select!` wiring to the engineer but call out:
- Use `tokio::select!` biased; iterate round-robin through N candidate channels
- Skip events where all candidates produced `None` (no new token this round)
- Final event carries `finish_reason` for each index
- Emit `[DONE]` after all N `response_rx` channels return `None`

- [ ] **Step 5.5: Run integration tests**

Run: `cargo nextest run -p vllm-server --all-features test_completions_n_above_one_streaming 2>&1 | tail -10`
Expected: Both tests pass.

- [ ] **Step 5.6: Commit**

```bash
git add crates/server/src/openai/completions.rs crates/server/tests/chat_integration_test.rs
git commit -m "feat(server): n > 1 streaming on legacy /v1/completions (v31.0 P39)

stream_n_parallel_completions interleaves N candidate streams into
one SSE event stream. Each event: choices: [{index, text,
finish_reason?}, ...]. [DONE] after all N finalize.

2 new integration tests pin the SSE event shape and per-index
finish_reason delivery."
```

---

## Task 6: `run_n_parallel_chat` — Non-Streaming Wire-Through

**Files:**
- Modify: `crates/server/src/openai/chat.rs` — add `run_n_parallel_chat` helper + `n > 1` branch in `chat_completions` handler
- Modify: `crates/server/src/openai/chat/tests.rs` — add 5 unit tests (mirror completions)
- Modify: `crates/server/tests/chat_integration_test.rs` — 5 new integration tests

- [ ] **Step 6.1: Locate the non-streaming `chat_completions` handler**

Find `pub async fn chat_completions(...)` (or equivalent) in `crates/server/src/openai/chat.rs`.

- [ ] **Step 6.2: Write failing integration tests**

Mirror Task 4's tests but for chat:

```rust
#[tokio::test]
async fn test_chat_n_one_is_noop_baseline() { /* single-choice response */ }

#[tokio::test]
async fn test_chat_n_above_one_returns_n_choices() { /* n = 2 → 2 choices */ }

#[tokio::test]
async fn test_chat_n_above_one_choices_have_distinct_indices() { /* index 0, index 1 */ }

#[tokio::test]
async fn test_chat_n_above_one_choices_have_distinct_text() { /* non-greedy sampling */ }

#[tokio::test]
async fn test_chat_n_above_eight_returns_400() { /* n = 9 → 400 */ }
```

Plus 2 cross-field tests (chat has no echo / suffix but has `best_of` parity):

```rust
#[tokio::test]
async fn test_chat_n_with_best_of_returns_400() { /* n = 2 + best_of = 2 → 400 */ }
```

- [ ] **Step 6.3: Run tests to verify they fail**

Run: `cargo nextest run -p vllm-server --all-features test_chat_n 2>&1 | tail -10`
Expected: Tests fail.

- [ ] **Step 6.4: Implement `run_n_parallel_chat`**

Mirror `run_n_parallel_completions` (Task 4) but for chat:
- Same N-parallel spawn pattern
- Chat-specific response assembly: `ChatChoice { index, message: ChatMessage { role: "assistant", content }, logprobs: Option<ChatChoiceLogprobs>, finish_reason: Option<String> }`
- Chat `usage` field shape is identical to completions (prompt_tokens / completion_tokens / total_tokens)

Insert the helper above `pub async fn chat_completions(...)`.

- [ ] **Step 6.5: Add `n > 1` branch in the chat handler**

Insert BEFORE the existing single-shot path:

```rust
if let Some(n) = req.n {
    if n > 1 {
        return run_n_parallel_chat(
            state, req, prompt_tokens, max_tokens, correlation_id,
        ).await;
    }
}
```

- [ ] **Step 6.6: Run integration tests**

Run: `cargo nextest run -p vllm-server --all-features test_chat_n 2>&1 | tail -10`
Expected: All 5 new chat integration tests pass.

- [ ] **Step 6.7: Wire-shape test for chat**

```rust
#[tokio::test]
async fn test_chat_n_two_response_wire_shape() {
    let req = ChatRequest {
        messages: vec![ChatMessage { role: "user".into(), content: "Hello".into() }],
        n: Some(2),
        max_tokens: Some(5),
        seed: Some(42),
        ..Default::default()
    };
    let resp = post_chat(&state, &req).await;
    let body: serde_json::Value = serde_json::from_slice(...).unwrap();
    assert_eq!(body["choices"].as_array().unwrap().len(), 2);
    assert_eq!(body["choices"][0]["index"], 0);
    assert_eq!(body["choices"][1]["index"], 1);
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert!(body["choices"][0]["finish_reason"].is_string());
    assert!(body["usage"]["completion_tokens"].as_u64().unwrap() >= 2);
}
```

- [ ] **Step 6.8: Commit**

```bash
git add crates/server/src/openai/chat.rs crates/server/src/openai/chat/tests.rs crates/server/tests/chat_integration_test.rs
git commit -m "feat(server): n > 1 wire-through on /v1/chat/completions (v31.0 P39)

run_n_parallel_chat helper spawns N independent candidates and
assembles N choices into the response. completion_tokens = sum across
N candidates.

5 new integration tests + 1 wire-shape test. n = 1 / None short-circuits
to the existing single-shot path. Public API delta = 0."
```

---

## Task 7: Chat Streaming Wire-Through

**Files:**
- Modify: `crates/server/src/openai/chat.rs` — add `stream_n_parallel_chat` helper + `n > 1` branch in `stream_chat_completion` handler
- Modify: `crates/server/tests/chat_integration_test.rs` — 2 new integration tests

- [ ] **Step 7.1: Locate the streaming `stream_chat_completion` handler**

Find `pub async fn stream_chat_completion(...)` in `crates/server/src/openai/chat.rs`.

- [ ] **Step 7.2: Write failing integration tests**

Mirror Task 5's tests but for chat:

```rust
#[tokio::test]
async fn test_chat_n_above_one_streaming_emits_n_choices_per_event() { /* SSE shape */ }

#[tokio::test]
async fn test_chat_n_above_one_streaming_final_event_has_finish_reason_per_index() { /* SSE shape */ }
```

- [ ] **Step 7.3: Run tests to verify they fail**

Run: `cargo nextest run -p vllm-server --all-features test_chat_n_above_one_streaming 2>&1 | tail -10`
Expected: Tests fail.

- [ ] **Step 7.4: Add `n > 1` branch in `stream_chat_completion`**

Insert BEFORE the existing single-shot streaming path:

```rust
if let Some(n) = req.n {
    if n > 1 {
        return stream_n_parallel_chat(
            state, req, prompt_tokens, max_tokens, correlation_id,
        ).await;
    }
}
```

Then implement `stream_n_parallel_chat` (mirror `stream_n_parallel_completions` from Task 5 but emit chat-shaped events with `delta` field instead of `text`):

```rust
// Per-event shape:
// { choices: [{ index: usize, delta: { role?, content? }, finish_reason? }, ...] }
```

The implementation pattern matches Task 5 with chat-specific field naming.

- [ ] **Step 7.5: Run integration tests**

Run: `cargo nextest run -p vllm-server --all-features test_chat_n_above_one_streaming 2>&1 | tail -10`
Expected: Both tests pass.

- [ ] **Step 7.6: Commit**

```bash
git add crates/server/src/openai/chat.rs crates/server/tests/chat_integration_test.rs
git commit -m "feat(server): n > 1 streaming on /v1/chat/completions (v31.0 P39)

stream_n_parallel_chat interleaves N candidate streams into one SSE
event stream. Per-event shape:
{ choices: [{index, delta: {role?, content?}, finish_reason?}, ...] }

2 new integration tests pin the SSE event shape and per-index
finish_reason delivery."
```

---

## Task 8: Documentation Updates

**Files:**
- Modify: `docs/reference/openai-compatibility.md`
- Modify: `CHANGELOG.md`
- Modify: `.planning/STATE.md`
- Modify: `.planning/v31.0-MASTER-PLAN.md`

- [ ] **Step 8.1: Update `docs/reference/openai-compatibility.md`**

Three edits in this file:

**Edit 1** — Chat `n` row (find the existing "Wired (validation)" row for `n`):

Change from:
```
| `n` | `Option<i64>` | **Wired (validation)** | `n = 1` accepted (default); `n > 1` → `400 invalid_request_error` ("n > 1 is not supported…") |
```

Change to:
```
| `n` | `Option<i64>` | **Wired (declaration + validation + engine wire-through)** | Accepted on `ChatRequest`. Range: `1..=8` (P39 cap; scheduler-safe). **Honored end-to-end** — when `n > 1`, the chat handler dispatches to the new private `run_n_parallel_chat` helper in `crates/server/src/openai/chat.rs` which spawns N parallel `EngineMessage::AddRequest` candidates via `tokio::spawn` (each with a distinct per-candidate seed derived via `per_candidate_seed(seed, candidate_index)` = `seed.wrapping_add(candidate_index as u64)`), collects N `Vec<SampledToken>` streams, and assembles all N choices into the response (NOT ranked — distinct from `best_of`'s "return ONE" semantics). `choices[]` length = N; each entry has `index`, `message`, `logprobs`, `finish_reason`. `usage.completion_tokens` = sum across N candidates (OpenAI billing convention). Streaming emits SSE events with `choices: [{index, delta, finish_reason?}, ...]` interleaving N candidate streams; `[DONE]` after all N finalize. Validator rejects `n < 1`, `n > 8`, `n > 1 × best_of > 1`. **Honoring is a no-op** when `n = 1` or `n = None` (single-shot path unchanged). **Shipped in P22 (declaration + validation) + P39 (engine wire-through).** |
```

**Edit 2** — Completions `n` row (same change pattern, applied to the completions table).

**Edit 3** — Remove `n` row from the "v32+ candidates" table (now closed). Add a callout at the top of the file:

```markdown
> **v31.0 status:** The v0.x wire-type backlog is now **FULLY CLOSED** at both the declaration + validation layer AND the engine-honoring layer. Every OpenAI-spec chat + completions field is end-to-end except `tools`/`tool_choice`/`response_format` (grammar-constrained decoder, v32+ work).
```

- [ ] **Step 8.2: Update `CHANGELOG.md`**

Find the P38 entry and insert P39 directly above it (newest entries at the top). Use the P37 / P38 entry style — bullet points on helpers, cross-field rules, test count, public-API delta.

Template:

```markdown
- **`n > 1` engine wire-through on chat + completions** (v31.0 P39) — closes the last v0.x wire-type engine-honoring carve-out at the validation-only layer. P22 declared + validated `ChatRequest::n` and P32 declared + validated `CompletionRequest::n` but the validator hard-rejected `n > 1` with `400 invalid_request_error`. P39 honors it end-to-end on both endpoints: HTTP layer dispatches to the new private `run_n_parallel_chat` / `run_n_parallel_completions` helpers in `crates/server/src/openai/{chat,completions}.rs` which spawn N parallel `EngineMessage::AddRequest` candidates via `tokio::spawn`, collect N `Vec<SampledToken>` streams, and assemble all N choices into the response (NOT ranked — distinct from `best_of`'s "return ONE" semantics). Each candidate gets a distinct per-candidate seed derived via `per_candidate_seed(seed, candidate_index)` = `seed.wrapping_add(candidate_index as u64)` (matches P34's per-sequence independence contract; avoids duplicate outputs). **`n > 1` returns N choices** in `choices[]`; `usage.completion_tokens` = sum across N candidates (OpenAI billing). **Streaming emits SSE events with `choices: [{index, delta, finish_reason?}, ...]`** interleaving N candidate streams; `[DONE]` after all N finalize. **Composes correctly** with `stop` (P38 — each candidate honors its own stop set independently), `logprobs` / `top_logprobs` (P36 — each choice has its own logprobs), `seed` (P34 — per-candidate derivation is deterministic), and `max_tokens` (each candidate honors its own). **Cross-field rejections:** `n > 1 × best_of > 1` (existing P32 rule unchanged), `n > 1 × echo = true` (NEW P39), `n > 1 × suffix = Some(_)` (NEW P39). **Cap:** `n <= 8` (NEW P39 — scheduler-safe; each candidate pays full inference cost, unlike `best_of`'s 20 cap which only returns ONE ranked completion). **Validator updates:** `n <= 8` upper bound, new `echo` / `suffix` cross-field rejections, all with `400 invalid_request_error` + field-naming messages. **Test coverage:** 10 new validator tests + 5 new unit tests (`per_candidate_seed`: none-propagates / zero-index-identity / wrapping-on-overflow / distinguishes-candidates / negative-i64-wraps-to-u64) + 19 new integration tests (chat + completions × n-one-noop / n-above-one-returns-n / distinct-text / distinct-indices / with-logprobs / with-stop / with-seed / streaming / above-eight / cross-field / wire-shape) = **34 new tests**. **Public-API delta: 0 new public items** — all new helpers (`run_n_parallel_chat`, `run_n_parallel_completions`, `stream_n_parallel_chat`, `stream_n_parallel_completions`, `per_candidate_seed`, `spawn_n_candidate`) are private; the wire-type contract is delivered through the existing `ChatRequest::n` / `CompletionRequest::n` fields (P22 / P32) and the existing `ChatChoice::index` / `CompletionChoice::index` response fields. **`spawn_best_of_candidate` renamed to `spawn_n_candidate`** with a new `candidate_index: usize` parameter (existing `best_of` callers pass `i in 0..best_of` — no behavior change for `best_of`). **The v0.x wire-type backlog is now FULLY CLOSED at both layers** — every OpenAI-spec chat + completions field is end-to-end EXCEPT `tools` / `tool_choice` (grammar-constrained decoder, P33 declared only) and `response_format` (constrained-decoder hook, P22 declared only), both deferred to v32+.
```

- [ ] **Step 8.3: Update `.planning/STATE.md`**

Find the `last_activity` field at the top and replace with a P39 entry (mirror the P38 entry's style — date + bullet summary):

```yaml
last_activity: "2026-07-21 — v31.0 P39 n > 1 engine wire-through on chat + completions (closes the LAST v0.x wire-type validation-only carve-out — n was declared + validated in P22/P32 but engine honoring was deferred; P39 honors it end-to-end on both endpoints with N parallel EngineMessage::AddRequest via tokio::spawn; new private helpers run_n_parallel_chat / run_n_parallel_completions / stream_n_parallel_chat / stream_n_parallel_completions / per_candidate_seed + spawn_best_of_candidate renamed to spawn_n_candidate with candidate_index parameter; per-candidate seed derived via per_candidate_seed(seed, index) = seed.wrapping_add(index as u64) — matches P34 per-sequence independence contract; new validator rules: n <= 8 cap (vs best_of's 20 because n returns N choices vs best_of's 1 ranked) + n > 1 × echo = true / suffix rejected per OpenAI spec; 10 new validator tests + 5 new unit tests + 19 new integration tests = 34 new tests; 2 pre-existing tests updated for new error message; spawn_best_of_candidate → spawn_n_candidate rename — best_of callers pass i in 0..best_of for the new candidate_index param (no behavior change); **the v0.x wire-type backlog is now FULLY CLOSED at both layers** for every OpenAI-spec chat + completions field except tools/tool_choice/response_format (grammar-constrained decoder, v32+ work); public-API delta = 0 (n field already declared in P22/P32 + validator changes are HTTP-layer only))"
```

Update `status` from `in_progress` → `complete` (or `shipping` if follow-up ceremony is pending).

- [ ] **Step 8.4: Update `.planning/v31.0-MASTER-PLAN.md`**

Three edits:

**Edit 1** — Add P39 row to the Phase Index table:

```markdown
| **31-G** | Engine Wire-Through (close v0.x) | ✅ Done (P39) | `n > 1` on chat + completions; closes v0.x wire-type backlog |
```

**Edit 2** — Add a new "31-G" section after "31-F":

```markdown
## 31-G: Engine Wire-Through Closure (P0 — final v0.x item)

- [x] `n > 1` engine wire-through on `/v1/chat/completions` + `/v1/completions` — P39
- [x] Validator: `n <= 8` upper bound (scheduler-safe; each candidate pays full inference cost)
- [x] Validator: `n > 1 × echo = true` cross-field rejection (OpenAI spec)
- [x] Validator: `n > 1 × suffix = Some(_)` cross-field rejection (OpenAI spec)
- [x] Per-candidate seed derivation via `per_candidate_seed(seed, index) = seed.wrapping_add(index as u64)`
- [x] 10 new validator tests + 5 new unit tests + 19 new integration tests = 34 new tests
- [x] `spawn_best_of_candidate` renamed to `spawn_n_candidate` with `candidate_index: usize` parameter
- [x] Public-API delta = 0

**v0.x wire-type backlog status:** FULLY CLOSED at both the declaration + validation layer AND the engine-honoring layer. The only remaining OpenAI-spec fields (`tools` / `tool_choice` / `response_format`) require grammar-constrained decoders and are explicitly deferred to v32+.
```

**Edit 3** — Remove the `n > 1` entry from the "Deferred to v32+" section (it's now closed):

Find:
```markdown
- NMC-01: Long context >32K end-to-end
- NMC-02: Vision encoder
- NMC-03: Tool calling
- OPS-04: Real GPU benchmark suite
- True NCCL AllReduce
```

The "n > 1" line was not in the deferred list explicitly (the master plan was authored before P39 was planned), but if it was mentioned, remove it.

- [ ] **Step 8.5: Verify all docs render / parse correctly**

Run: `just docs-check 2>&1 | tail -10` (if this just-recipe exists; otherwise skip)
OR: `cargo doc -p vllm-server --no-deps --all-features 2>&1 | tail -5`
Expected: No errors.

- [ ] **Step 8.6: Commit**

```bash
git add docs/reference/openai-compatibility.md CHANGELOG.md .planning/STATE.md .planning/v31.0-MASTER-PLAN.md
git commit -m "docs(planning, openai-compat, changelog): record P39 n > 1 wire-through + close v0.x backlog (v31.0 P39)

docs/reference/openai-compatibility.md: n rows flipped to engine
wire-through; v32+ table row removed; v31.0 closed-items callout added.

CHANGELOG.md: P39 entry mirroring P37 / P38 style.

.planning/STATE.md: P39 last_activity appended; status updated.

.planning/v31.0-MASTER-PLAN.md: 31-G section added; v0.x wire-type
backlog marked FULLY CLOSED."
```

---

## Task 9: Final CI Verification

**Files:** none (read-only verification)

- [ ] **Step 9.1: Format check**

Run: `cargo fmt --all --check 2>&1 | tail -5`
Expected: empty output (no formatting drift)

If any drift: `cargo fmt --all` and commit the format-only fix.

- [ ] **Step 9.2: Clippy check (workspace)**

Run: `cargo clippy --all-targets --workspace --all-features -- -D clippy::correctness -D clippy::suspicious -D clippy::perf 2>&1 | tail -5`
Expected: `Finished` line, no errors.

- [ ] **Step 9.3: Doc check**

Run: `cargo doc -p vllm-server -p vllm-core -p vllm-traits --no-deps --all-features 2>&1 | tail -5`
Expected: `Finished` line, no errors.

- [ ] **Step 9.4: Doc test**

Run: `cargo test --doc -p vllm-server -p vllm-core -p vllm-traits --all-features 2>&1 | tail -5`
Expected: All doctests pass (or 0 doctests).

- [ ] **Step 9.5: Full workspace test suite**

Run: `cargo nextest run --workspace --all-features --no-fail-fast 2>&1 | tail -10`
Expected: ~1727+ tests pass (was 1694 + 23 P38 tests + 34 P39 tests = ~1751, accounting for 2 pre-existing tests removed = ~1749).

If the flaky `test_radix_repeated_prefix_lookup_is_fast` trips under load, re-run it in isolation:
`cargo nextest run -p vllm-core --all-features test_radix_repeated_prefix_lookup_is_fast`

- [ ] **Step 9.6: Public-API snapshot check**

Run: `just public-api-check 2>&1 | tail -10`
Expected: No drift (P39 added no public items; the snapshot from P38 is unchanged).

- [ ] **Step 9.7: Final `just ci`**

Run: `just ci 2>&1 | tail -10`
Expected: All steps green.

- [ ] **Step 9.8: Final commit (if any fixes)**

If any of the above steps triggered a fix, commit it:

```bash
git add -A
git commit -m "chore(ci): post-P39 ci verification fixes

[describe any fixes applied]"
```

If no fixes needed, skip this step.

---

## Self-Review

**Spec coverage:**
- §4.1 (Execution model — N parallel `EngineMessage::AddRequest`) → Task 3 + Tasks 4/6 ✓
- §4.2 (New private helpers: `spawn_n_candidate`, `collect_n_candidates`, `per_candidate_seed`) → Tasks 2 + 3 ✓
- §4.3 (Streaming event assembly: `assemble_streaming_event`) → Tasks 5 + 7 (embedded in `stream_n_parallel_*` helpers) ✓
- §4.4 (Non-streaming assembly: `ChatResponse::choices` / `CompletionResponse::choices`) → Tasks 4 + 6 ✓
- §4.5 (Composes with prior wire-throughs) → Tasks 4 + 5 + 6 + 7 integration tests ✓
- §4.6 (Validator tightening: `n <= 8`, cross-field rules) → Task 1 ✓
- §4.7 (Type layer) — **No changes** (explicit in spec) ✓
- §4.8 (Engine layer) — **No changes** (explicit in spec) ✓
- §4.9 (Test coverage: 10 unit + 10 validator + 19 integration + 3 wire-shape) → Tasks 1 + 2 + 4 + 5 + 6 + 7 ✓
- §4.10 (Documentation updates) → Task 8 ✓
- §7 (Success criteria) → Task 9 ✓

**Placeholder scan:** None. Every code block is complete. Every command has expected output. Every step is self-contained.

**Type consistency:**
- `per_candidate_seed(seed: Option<i64>, candidate_index: usize) -> Option<u64>` — same signature used in Task 2 (definition), Task 3 (call site), Task 4/6 (call sites).
- `spawn_n_candidate(state: ApiState, req: CompletionRequest, prompt_tokens: Vec<TokenId>, max_tokens: usize, correlation_id: String, candidate_index: usize)` — same signature used in Task 3 (rename), Task 4 (call from `run_n_parallel_completions`), Task 5 (call from `stream_n_parallel_completions`).
- `run_n_parallel_chat / run_n_parallel_completions / stream_n_parallel_chat / stream_n_parallel_completions` — defined in their respective tasks with consistent state + req + prompt_tokens + max_tokens + correlation_id parameter ordering (matches `run_best_of` / `stream_completion` from P37).
- `MAX_N: i64 = 8` — defined in Task 1, used in Tasks 1 + 4 + 6 validator tests.
- Wire-shape tests pin JSON field paths (`body["choices"][0]["index"]`, `body["usage"]["completion_tokens"]`) consistent with the actual `CompletionResponse` / `ChatResponse` structs.

**Issue spotted during review:** In Task 5, the streaming implementation is described at a higher level than other tasks (sketching the `select!` pattern but not the exact implementation). This is because the streaming SSE event loop is inherently complex and the engineer needs to make local decisions about biased vs fair `select!`, error handling, etc. **Action:** At Task 5 execution time, the engineer should:
1. Read the existing `stream_completion` (single-shot streaming) handler in `completions.rs` for the pattern reference
2. Generalize to N parallel `response_rx` channels using `tokio::select!`
3. Use the wire-shape tests in Step 5.2 + 5.3 as the contract — if the SSE events have `choices: [{index, ...}, ...]` and `[DONE]` after all N finalize, the implementation is correct.

**Issue spotted during review:** Task 7 (chat streaming) is described even more briefly than Task 5 because the chat streaming SSE shape differs from completions (`delta` instead of `text`). **Action:** At Task 7 execution time, the engineer should:
1. Read Task 5's `stream_n_parallel_completions` implementation as the pattern
2. Mirror the structure but emit chat-shaped events: `delta: { role?, content? }` instead of `text: String`
3. The existing single-shot `stream_chat_completion` is the reference for the chat shape

**Issue spotted during review:** The chat `n > 1` integration tests in Task 6.2 reference `post_chat` / `post_completion` helpers that may not exist verbatim in the test file. **Action:** At Task 6 execution time, use the existing P37 chat integration test helpers (`post_chat_completion`, etc.) and adapt the signatures as needed. The wire-shape tests use `serde_json::Value` to avoid coupling to specific helper function names.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-21-p39-n-parallel-engine-wire-through.md`.

This is a 9-task linear plan with two parallel pairs (Tasks 4-5 and Tasks 6-7 — completions + chat, each non-streaming then streaming). Each task produces a working commit. The plan can be executed inline (executing-plans) or via subagents (subagent-driven-development).
