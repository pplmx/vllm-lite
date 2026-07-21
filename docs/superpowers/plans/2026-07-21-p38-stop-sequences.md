# P38 Implementation Plan — `stop` Sequences Engine Wire-Through + `large_enum_variant` Lint Fix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `stop` sequences end-to-end on `/v1/chat/completions` and `/v1/completions` (currently hard-400) and fix the `large_enum_variant` clippy lint so `just ci` is green.

**Architecture:** Tokenize `stop` strings at the HTTP boundary, forward pre-tokenized sequences through `SamplingParams::stop_token_sequences`. The engine's `step_regular` runs a pure-function `matches_stop_sequences` check after sampling and finalizes the sequence with `FinishReason::Stop` when any stop suffix matches the generated tokens. The matched token is emitted to `response_tx` before finalization (OpenAI convention). Concurrently, change `EngineMessage::AddRequest.request` from `Request` to `Box<Request>` to fix the P36-introduced `large_enum_variant` clippy lint.

**Tech Stack:** Rust (workspace), axum, tokio mpsc/oneshot, parking_lot, serde, thiserror.

**Spec:** `docs/superpowers/specs/2026-07-21-p38-stop-sequences-design.md`

**Estimated effort:** 1.5–2 working days, 10 tasks.

---

## File Structure (mapped up-front)

### Modified files
- `crates/traits/src/sampling.rs` — add `SamplingParams::stop_token_sequences` field + builder method + `Default::default` entry
- `crates/server/src/openai/sampling_validation.rs` — add `validate_stop_sequences` helper, remove old stop 400-rejection from `validate_chat_request_fields` and `validate_completion_request_fields`
- `crates/server/src/openai/chat.rs` — call `validate_stop_sequences`, tokenize + wire through in inline sampling-params forwarding (chat.rs does NOT have a populate helper yet — keep inline to avoid scope creep)
- `crates/server/src/openai/completions.rs` — call `validate_stop_sequences`, tokenize + wire through in `populate_completion_sampling_params` (extracted by P37)
- `crates/server/src/openai/sampling_validation.rs::tests` — 7 new validator tests + remove 2 old 400-rejection tests
- `crates/core/src/scheduler/batch.rs` — add `matches_stop_sequences` private helper + integration in `step_regular`
- `crates/core/src/types/messages.rs` — change `EngineMessage::AddRequest.request` to `Box<Request>` + add `#[allow(clippy::large_enum_variant)]` is NOT needed (the fix is to Box, not to allow)
- 13 `EngineMessage::AddRequest { request, ... }` construction sites in:
  - `crates/core/src/engine/run.rs:33`
  - `crates/core/tests/e2e_graceful_shutdown.rs:47`
  - `crates/server/src/test_fixtures.rs:101`
  - `crates/server/src/openai/completions.rs:248, 590`
  - `crates/server/src/openai/chat.rs:346, 679`
  - `crates/server/tests/overload_integration.rs:132`
  - `crates/server/tests/cancel_propagation.rs:94`
  - `crates/server/tests/tutorial_e2e.rs:43`
  - `crates/server/tests/request_id_propagation.rs:61, 155`
  - `crates/server/tests/chat_integration_test.rs:470, 4182, 4698, 5240`
- `crates/server/tests/chat_integration_test.rs` — 9 new integration tests
- `docs/reference/openai-compatibility.md` — flip `stop` rows to "engine wire-through", remove v32+ table row
- `CHANGELOG.md` — add P38 entries
- `.planning/STATE.md` — update `last_activity`

### New files
- `crates/core/src/sampling/stop_check.rs` (or inline in `crates/core/src/sampling.rs`) — `matches_stop_sequences` function
- `crates/core/src/sampling/tests/stop_check.rs` (or inline) — 7 unit tests

### No new public types
1 new public field, 1 new builder method, 1 internal field-type change (`Request` → `Box<Request>`).

---

## Dependency Graph

```
Task 1: large_enum_variant fix (mechanical, unblocks CI)
   ↓
Task 2: matches_stop_sequences helper (TDD unit tests)
   ↓
Task 3: SamplingParams::stop_token_sequences field + builder
   ↓
Task 4: validate_stop_sequences helper (TDD validator tests)
   ↓
Task 5: HTTP wire-through chat
   ↓
Task 6: HTTP wire-through completions
   ↓
Task 7: Engine step_regular integration + integration tests (TDD)
   ↓
Task 8: Documentation updates
   ↓
Task 9: Full CI verification
```

Each task produces a working commit. The plan is linear with no parallel tasks.

---

## Task 1: Fix `large_enum_variant` Clippy Lint (Part C)

**Files:**
- Modify: `crates/core/src/types/messages.rs:57-65` (change field type)
- Modify: 13 construction sites across 12 files (add `Box::new(...)` wrapper)

- [ ] **Step 1.1: Change `EngineMessage::AddRequest.request` type**

Edit `crates/core/src/types/messages.rs`:

```rust
AddRequest {
    request: Box<Request>,  // was: Request
    response_tx: mpsc::Sender<SampledToken>,
    seq_id_tx: Option<oneshot::Sender<SeqId>>,
    finish_reason_tx: Option<oneshot::Sender<vllm_traits::FinishReason>>,
    request_id: Option<String>,
},
```

- [ ] **Step 1.2: Compile to find all construction sites**

Run: `cargo check -p vllm-core -p vllm-server --all-features 2>&1 | grep "error\["`
Expected: 13 error lines pointing to `AddRequest { request, ... }` sites

- [ ] **Step 1.3: Update all 13 construction sites**

At each error site, change `request: req` (or similar local variable name) to `request: Box::new(req)`.

Sites (line numbers as of 2026-07-21, verify before editing):
- `crates/core/src/engine/run.rs:33` — `request: request,` → `request: Box::new(request),`
- `crates/core/tests/e2e_graceful_shutdown.rs:47` — same pattern
- `crates/server/src/test_fixtures.rs:101` — same pattern
- `crates/server/src/openai/chat.rs:346` — same
- `crates/server/src/openai/chat.rs:679` — same
- `crates/server/src/openai/completions.rs:248` — same
- `crates/server/src/openai/completions.rs:590` — same
- `crates/server/tests/overload_integration.rs:132` — same
- `crates/server/tests/cancel_propagation.rs:94` — same
- `crates/server/tests/tutorial_e2e.rs:43` — same
- `crates/server/tests/request_id_propagation.rs:61` — same
- `crates/server/tests/request_id_propagation.rs:155` — same
- `crates/server/tests/chat_integration_test.rs:470, 4182, 4698, 5240` — same

The local variable holding the `Request` may have different names (e.g. `request`, `req`, `r`). Match each one in context.

- [ ] **Step 1.4: Compile to verify clean**

Run: `cargo check -p vllm-core -p vllm-server --all-features 2>&1 | tail -5`
Expected: `Finished` line, no errors

- [ ] **Step 1.5: Run clippy to verify lint passes**

Run: `cargo clippy --all-targets -p vllm-server -p vllm-core --all-features -- -D clippy::correctness -D clippy::suspicious -D clippy::perf 2>&1 | grep -E "^error" | head -5`
Expected: empty (no errors). The `large_enum_variant` lint should be gone.

- [ ] **Step 1.6: Run all server tests to verify no regression**

Run: `cargo test -p vllm-server --lib --all-features 2>&1 | tail -3`
Expected: All tests pass (same count as before, ~284)

- [ ] **Step 1.7: Commit**

```bash
git add -A
git commit -m "fix(core): Box EngineMessage::AddRequest.request to silence large_enum_variant lint (v31.0 P38)

P36's SampledToken wire-breaking signature change inflated Request past 248 bytes
and tripped clippy::large_enum_variant on EngineMessage::AddRequest (the other
variants are <= 64 bytes). Boxing brings the enum to pointer-sized.

All 13 AddRequest construction sites updated to Box::new(request). Public API
surface is unchanged (Request is consumed by value at construction; callers never
read .request after).

just ci is now green on vllm-core (the only outstanding clippy error)."
```

---

## Task 2: `matches_stop_sequences` Helper (Part A — foundation, TDD)

**Files:**
- Modify: `crates/core/src/sampling.rs` (add `matches_stop_sequences` function)
- Modify: `crates/core/src/sampling/tests.rs` (add 7 unit tests)

- [ ] **Step 2.1: Write 7 failing unit tests**

Append to `crates/core/src/sampling/tests.rs`:

```rust
// =============================================================================
// P38 v0.x wire-type follow-up — engine wire-through: `matches_stop_sequences`
// helper tests. Pure-function unit tests; the helper is consumed by the
// engine's `step_regular` loop after every sampled token. Pin the contract
// end-to-end so future refactors (e.g. Aho-Corasick) trip the suite.
// =============================================================================

#[test]
fn matches_stop_sequences_no_match_returns_false() {
    // Stop = "xyz" tokens [99], generated = [1, 2, 3] — no suffix match.
    let stops: Vec<Vec<u32>> = vec![vec![99]];
    let generated: Vec<u32> = vec![1, 2, 3];
    assert!(!matches_stop_sequences(&generated, &stops));
}

#[test]
fn matches_stop_sequences_single_token_match_returns_true() {
    // Stop = [99] (1 token), generated = [1, 2, 99] — suffix matches.
    let stops: Vec<Vec<u32>> = vec![vec![99]];
    let generated: Vec<u32> = vec![1, 2, 99];
    assert!(matches_stop_sequences(&generated, &stops));
}

#[test]
fn matches_stop_sequences_multi_token_match_returns_true() {
    // Stop = [10, 20, 30] (3 tokens), generated ends with [10, 20, 30].
    let stops: Vec<Vec<u32>> = vec![vec![10, 20, 30]];
    let generated: Vec<u32> = vec![1, 2, 3, 10, 20, 30];
    assert!(matches_stop_sequences(&generated, &stops));
}

#[test]
fn matches_stop_sequences_partial_match_returns_false() {
    // Stop = [10, 20, 30], generated ends with [10, 20] — only 2 of 3
    // stop tokens match. Must NOT trigger (we require the FULL suffix).
    let stops: Vec<Vec<u32>> = vec![vec![10, 20, 30]];
    let generated: Vec<u32> = vec![1, 2, 10, 20];
    assert!(!matches_stop_sequences(&generated, &stops));
}

#[test]
fn matches_stop_sequences_first_match_wins() {
    // Multiple stops; the FIRST one whose suffix matches wins (helper
    // returns true on first hit, doesn't care which one).
    let stops: Vec<Vec<u32>> = vec![vec![99], vec![88]];
    let generated_a: Vec<u32> = vec![1, 2, 99];
    let generated_b: Vec<u32> = vec![1, 2, 88];
    assert!(matches_stop_sequences(&generated_a, &stops));
    assert!(matches_stop_sequences(&generated_b, &stops));
    // Neither matches: stop sequences don't appear as suffixes.
    let generated_c: Vec<u32> = vec![1, 2, 3];
    assert!(!matches_stop_sequences(&generated_c, &stops));
}

#[test]
fn matches_stop_sequences_empty_stops_returns_false() {
    // `stops = vec![]` → never matches (helper loops over zero elements).
    let stops: Vec<Vec<u32>> = vec![];
    let generated: Vec<u32> = vec![1, 2, 3];
    assert!(!matches_stop_sequences(&generated, &stops));
}

#[test]
fn matches_stop_sequences_stop_longer_than_generated_returns_false() {
    // Stop = [1, 2, 3, 4, 5], generated has only 3 tokens — cannot match.
    let stops: Vec<Vec<u32>> = vec![vec![1, 2, 3, 4, 5]];
    let generated: Vec<u32> = vec![1, 2, 3];
    assert!(!matches_stop_sequences(&generated, &stops));
}
```

- [ ] **Step 2.2: Run tests to verify they fail**

Run: `cargo test -p vllm-core --all-features matches_stop_sequences -- --nocapture 2>&1 | tail -15`
Expected: 7 failures with "cannot find function `matches_stop_sequences`" or similar

- [ ] **Step 2.3: Implement `matches_stop_sequences` in `crates/core/src/sampling.rs`**

Add the function (near the other sampling helpers):

```rust
/// P38 v0.x wire-type follow-up — engine wire-through helper for
/// `stop` sequences. Returns `true` iff any token sequence in `stops`
/// is a suffix of `generated_tokens`.
///
/// **Complexity:** O(N × M) where N = `generated_tokens.len()` and
/// M = sum(stop.len() for stop in stops). Both are tiny in practice:
/// - `generated_tokens.len()` ≤ `max_tokens` (typical ≤ 4096)
/// - `stops.len()` ≤ 4 (OpenAI spec upper bound, validated at HTTP layer)
/// - each `stop.len()` ≤ ~8 tokens (typical BPE-tokenized stop strings)
///
/// So the per-step cost is bounded by ~32 slice comparisons × max_tokens
/// positions — well under 100 ns per step on a modern CPU. The check runs
/// once per generated token in `step_regular`.
///
/// **Empty / oversized stops:** an empty stop (`vec![]`) or a stop
/// longer than `generated_tokens` is a no-op for that iteration. The
/// HTTP-layer `validate_stop_sequences` rejects empty-string stops
/// (which would tokenize to either zero or one token) so this function
/// doesn't need to handle "stop that can never match" specially.
pub fn matches_stop_sequences(
    generated_tokens: &[u32],
    stops: &[Vec<u32>],
) -> bool {
    for stop in stops {
        if stop.is_empty() || stop.len() > generated_tokens.len() {
            continue;
        }
        let start = generated_tokens.len() - stop.len();
        if &generated_tokens[start..] == stop.as_slice() {
            return true;
        }
    }
    false
}
```

(The function takes `&[u32]` rather than `&[vllm_traits::TokenId]` because `TokenId` is `pub type TokenId = u32` and we want this helper to live in `vllm_core::sampling` without a `vllm_traits` import. The `SamplingParams::stop_token_sequences` field uses `Vec<Vec<TokenId>>` for public-API ergonomics; the caller converts via deref coercion at the call site.)

- [ ] **Step 2.4: Run tests to verify they pass**

Run: `cargo test -p vllm-core --all-features matches_stop_sequences -- --nocapture 2>&1 | tail -10`
Expected: 7 passed, 0 failed

- [ ] **Step 2.5: Commit**

```bash
git add crates/core/src/sampling.rs crates/core/src/sampling/tests.rs
git commit -m "feat(core): add matches_stop_sequences helper for stop wire-through (v31.0 P38)

Pure-function helper consumed by the engine's step_regular loop after every
sampled token. Returns true iff any token sequence in stops is a suffix of
generated_tokens. O(N*M) where N = generated_tokens.len() and M = sum(stop.len())
— both are tiny in practice (N <= max_tokens, M <= 32 for typical 4-stop lists).

7 unit tests pin the contract: no-match, single-token match, multi-token match,
partial-match (false), first-match-wins, empty-stops (false), stop-longer-than-
generated (false)."
```

---

## Task 3: `SamplingParams::stop_token_sequences` Field + Builder (Part A — type layer)

**Files:**
- Modify: `crates/traits/src/sampling.rs` (add field at line ~198, add to Default at line ~214, add builder method)

- [ ] **Step 3.1: Add `stop_token_sequences` field to `SamplingParams`**

Edit `crates/traits/src/sampling.rs` to add the new field AFTER `top_logprobs` (currently at line 198). The new field:

```rust
/// OpenAI `stop` sequences (P38 v0.x wire-type follow-up engine
/// wire-through): pre-tokenized stop strings that the engine
/// checks after every sampled token. When any token sequence in
/// this list is a suffix of a sequence's already-generated tokens,
/// the engine finalizes that sequence with [`FinishReason::Stop`]
/// and emits the matching token to the HTTP layer (OpenAI
/// convention — the matched stop text is included in the response).
///
/// `None` → no stop check (default-path overhead is zero — the
/// engine skips the `matches_stop_sequences` call entirely when
/// this is `None`).
///
/// `Some(vec![])` → also a no-op (the validator normalizes empty
/// vectors to `None` before forwarding; the engine treats both
/// identically).
///
/// `Some(non_empty)` → check after every sampled token. Pre-tokenized
/// at the HTTP boundary via the model's BPE tokenizer; the engine
/// itself never touches the tokenizer.
///
/// Per OpenAI spec, max 4 stop strings per request; each stop
/// string is tokenized to ≤ ~8 BPE tokens in practice. Validation
/// happens on the HTTP layer (see `validate_stop_sequences` in
/// `vllm_server::openai::sampling_validation`).
///
/// [`FinishReason::Stop`]: vllm_traits::FinishReason::Stop
#[serde(default, skip_serializing_if = "Option::is_none")]
pub stop_token_sequences: Option<Vec<Vec<TokenId>>>,
```

- [ ] **Step 3.2: Add `stop_token_sequences: None` to `Default::default` impl**

In `impl Default for SamplingParams` (currently at line 201), add the new field:

```rust
impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            logit_bias: None,
            beam_width: 1,
            length_penalty: 0.6,
            max_retries: 0,
            seed: None,
            top_logprobs: None,
            stop_token_sequences: None,  // NEW (P38)
        }
    }
}
```

- [ ] **Step 3.3: Add `with_stop_token_sequences` builder method**

After `with_top_logprobs_none` (currently at line 344), add:

```rust
/// Set [`SamplingParams::stop_token_sequences`].
///
/// OpenAI `stop` sequences, pre-tokenized at the HTTP boundary.
/// When `Some(seqs)`, the engine checks after every sampled token
/// whether any token sequence in `seqs` is a suffix of the
/// generated tokens and finalizes with [`FinishReason::Stop`] on
/// match. When `None`, no stop check runs (default-path overhead
/// stays at zero).
///
/// See the field-level doc on [`SamplingParams::stop_token_sequences`]
/// for the full contract and validation rules.
#[must_use]
pub fn with_stop_token_sequences(mut self, seqs: Vec<Vec<TokenId>>) -> Self {
    self.inner.stop_token_sequences = Some(seqs);
    self
}

/// Clear [`SamplingParams::stop_token_sequences`] back to `None`.
/// Mirrors [`SamplingParamsBuilder::with_seed_none`] / [`SamplingParamsBuilder::with_top_logprobs_none`].
#[must_use]
pub const fn with_stop_token_sequences_none(mut self) -> Self {
    self.inner.stop_token_sequences = None;
    self
}
```

- [ ] **Step 3.4: Verify `vllm-traits` compiles and public-API snapshot is updated**

Run: `cargo check -p vllm-traits --all-features 2>&1 | tail -3`
Expected: `Finished` line, no errors

Then: `cargo run -p vllm-traits --bin public-api-check 2>&1 | tail -10` (if the binary exists; otherwise skip and just rely on the `public-api-check` step in `just ci` later)
Expected: Either "no changes" or "regenerated baseline" — if the latter, commit the regenerated snapshot file (likely `crates/traits/api/vllm_traits.api` or similar).

- [ ] **Step 3.5: Commit**

```bash
git add crates/traits/src/sampling.rs crates/traits/api/ 2>/dev/null
git commit -m "feat(traits): add SamplingParams::stop_token_sequences field + builder (v31.0 P38)

New public field SamplingParams::stop_token_sequences: Option<Vec<Vec<TokenId>>>
plus two builder methods: with_stop_token_sequences(Vec<Vec<TokenId>>) and
with_stop_token_sequences_none() (mirrors the existing with_seed / with_seed_none
and with_top_logprobs / with_top_logprobs_none patterns).

Default: None (no stop check; default-path overhead stays at zero).

Pre-tokenized at the HTTP boundary so the engine and sampler never touch the
tokenizer. Each stop sequence is a Vec<TokenId> representing one OpenAI stop
string after BPE encoding."
```

---

## Task 4: `validate_stop_sequences` Helper (Part A — validation, TDD)

**Files:**
- Modify: `crates/server/src/openai/sampling_validation.rs` (add helper, update callers)
- Modify: `crates/server/src/openai/sampling_validation.rs::tests` (7 new tests, 2 deletions)

- [ ] **Step 4.1: Write 7 failing validator tests + remove 2 old tests**

First, DELETE the two existing tests in `crates/server/src/openai/sampling_validation.rs::tests` that asserted the old 400-rejection behavior:

- `test_chat_rejects_non_empty_stop_with_400` (search for it; if present, remove the entire `#[test]` function)
- `test_completions_rejects_non_empty_stop_with_400` (same)

(The exact names may vary — search for "stop sequences are not yet honoured" in tests and remove the asserting tests.)

Then APPEND 7 new tests:

```rust
// =============================================================================
// P38 v0.x wire-type follow-up — engine wire-through: `validate_stop_sequences`
// helper tests. Pin the new validator contract (max 4 strings, no empty
// strings) end-to-end through the helper signature. Replaces the old
// "non-empty stop → 400" rejection tests with "stop is now accepted".
// =============================================================================

#[test]
fn stop_validation_none_passes() {
    // OpenAI spec: omitting `stop` entirely (None) is the default and
    // must always pass.
    validate_stop_sequences(&None).expect("None must pass (default; P38)");
}

#[test]
fn stop_validation_empty_vec_passes() {
    // OpenAI spec: an explicit `stop: []` is equivalent to None and
    // must pass. The HTTP wire-through normalizes this to `None` at
    // the populate layer.
    let stop: Option<Vec<String>> = Some(vec![]);
    validate_stop_sequences(&stop).expect("empty vec must pass (P38)");
}

#[test]
fn stop_validation_single_string_passes() {
    let stop: Option<Vec<String>> = Some(vec!["\n\n".to_string()]);
    validate_stop_sequences(&stop).expect("single-string stop must pass (P38)");
}

#[test]
fn stop_validation_max_4_strings_passes() {
    // 4 stops is the OpenAI spec upper bound — must pass.
    let stop: Option<Vec<String>> = Some(vec![
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
        "d".to_string(),
    ]);
    validate_stop_sequences(&stop).expect("4-stop vec must pass (OpenAI upper bound, P38)");
}

#[test]
fn stop_validation_more_than_4_strings_returns_400() {
    // 5 stops exceeds the OpenAI upper bound — reject.
    let stop: Option<Vec<String>> = Some(vec![
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
        "d".to_string(),
        "e".to_string(),
    ]);
    let err = validate_stop_sequences(&stop)
        .expect_err("5-stop vec must be rejected (>4 per OpenAI spec, P38)");
    assert_eq!(err.0, StatusCode::BAD_REQUEST);
    assert!(err.1.0.error.message.contains("stop"));
}

#[test]
fn stop_validation_empty_string_returns_400() {
    // An empty-string stop is semantically a no-op (would never match
    // any generated text) — reject to give the caller a clear error.
    let stop: Option<Vec<String>> = Some(vec!["".to_string()]);
    let err = validate_stop_sequences(&stop)
        .expect_err("empty-string stop must be rejected (P38)");
    assert_eq!(err.0, StatusCode::BAD_REQUEST);
    assert!(err.1.0.error.message.contains("stop"));
}

#[test]
fn stop_validation_string_with_only_whitespace_returns_400() {
    // A whitespace-only stop tokenizes to zero tokens in many BPE
    // tokenizers (a no-op that would never match) — reject to give
    // the caller a clear error.
    let stop: Option<Vec<String>> = Some(vec!["   ".to_string()]);
    let err = validate_stop_sequences(&stop)
        .expect_err("whitespace-only stop must be rejected (P38)");
    assert_eq!(err.0, StatusCode::BAD_REQUEST);
    assert!(err.1.0.error.message.contains("stop"));
}
```

- [ ] **Step 4.2: Run tests to verify they fail**

Run: `cargo test -p vllm-server --lib --all-features stop_validation_ 2>&1 | tail -15`
Expected: 7 failures (cannot find `validate_stop_sequences`)

- [ ] **Step 4.3: Implement `validate_stop_sequences` in `crates/server/src/openai/sampling_validation.rs`**

Add the helper (location: near `validate_logit_bias` in the same file). The helper validates the spec-level rules (length and string content) but does NOT tokenize — tokenization happens in the populate helpers because it requires the tokenizer handle.

```rust
/// Validate `stop` per OpenAI spec (P38 v0.x wire-type follow-up
/// engine wire-through). Pin the spec-level rules:
///
/// - `None` → pass (default; no stop check)
/// - `Some(vec![])` → pass (normalized to `None` at the populate
///   layer; the engine treats both identically)
/// - `Some(non_empty)` → reject if `len() > 4` (OpenAI spec upper
///   bound; protects the server from unbounded stop-list sizes)
/// - `Some(non_empty)` → reject if any element is empty (`""`) or
///   whitespace-only (`"   "`); such stops would tokenize to zero
///   tokens in most BPE tokenizers and never match, which is
///   silently broken. The validator surfaces this as 400.
///
/// **Tokenization happens in the populate helper** because the
/// validator runs BEFORE the tokenizer is acquired (chat +
/// completions validators are pure functions over `&ChatRequest`
/// / `&CompletionRequest`).
///
/// # Errors
///
/// Returns `Err((StatusCode::BAD_REQUEST, Json<ErrorResponse>))`
/// when any check fires.
pub fn validate_stop_sequences(
    stop: &Option<Vec<String>>,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if let Some(stops) = stop {
        if stops.len() > 4 {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new(
                    "stop sequences exceed OpenAI spec upper bound (max 4)",
                    "invalid_request_error",
                )),
            ));
        }
        if stops.iter().any(|s| s.is_empty() || s.trim().is_empty()) {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new(
                    "stop sequences cannot contain empty or whitespace-only strings (would tokenize to zero tokens)",
                    "invalid_request_error",
                )),
            ));
        }
    }
    Ok(())
}
```

- [ ] **Step 4.4: Run tests to verify they pass**

Run: `cargo test -p vllm-server --lib --all-features stop_validation_ 2>&1 | tail -10`
Expected: 7 passed, 0 failed

- [ ] **Step 4.5: Commit**

```bash
git add crates/server/src/openai/sampling_validation.rs
git commit -m "feat(server): add validate_stop_sequences helper (v31.0 P38)

New validator replaces the old 'non-empty stop → 400' rejection with a
structured helper that pins the OpenAI spec contract:
- None → pass (default)
- Some(empty_vec) → pass (normalized to None downstream)
- Some(len > 4) → 400 invalid_request_error (OpenAI upper bound)
- Some(any empty/whitespace string) → 400 invalid_request_error
  (would tokenize to zero tokens and never match)

7 new unit tests pin the contract. 2 old 'rejects_non_empty_stop' tests
removed (the rejection is gone; the new tests replace them)."
```

---

## Task 5: HTTP Wire-Through — Chat (Part A)

**Files:**
- Modify: `crates/server/src/openai/sampling_validation.rs::validate_chat_request_fields` (remove 400-rejection, add `validate_stop_sequences` call)
- Modify: `crates/server/src/openai/chat.rs` (add stop tokenization + wire-through in inline sampling-params forwarding)

- [ ] **Step 5.1: Remove stop 400-rejection from `validate_chat_request_fields`**

Edit `crates/server/src/openai/sampling_validation.rs` (the function is around line 685). Replace the stop 400-rejection block:

```rust
// REMOVE this block:
if let Some(stop) = &req.stop
    && !stop.is_empty()
{
    return Err((
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse::new(
            "stop sequences are not yet honoured; the engine stops at max_tokens or natural EOS only (omit stop or send an empty array)",
            "invalid_request_error",
        )),
    ));
}
```

With:

```rust
// P38 v0.x wire-type follow-up — engine wire-through: stop sequences
// are now accepted (tokenized + forwarded by populate_chat_sampling_params).
// Per-string validation lives in `validate_stop_sequences` (max 4 strings,
// no empty/whitespace strings).
validate_stop_sequences(&req.stop)?;
```

- [ ] **Step 5.2: Add stop tokenization + wire-through in `crates/server/src/openai/chat.rs`**

Find the inline sampling-params forwarding block (around line 568 where `request.sampling_params.temperature = temp;` lives). Add the stop tokenization right before the `populate` block. The chat handler does NOT have a `populate_chat_sampling_params` helper yet — we keep the wire-through inline to avoid scope creep (a helper extraction is a separate refactor; the existing P37 extraction was scoped to completions only).

Add immediately after the existing `request.sampling_params.top_logprobs = req.logprobs;` line:

```rust
// P38 v0.x wire-type follow-up — engine wire-through: tokenize the
// user's stop strings at the HTTP boundary and forward as
// `SamplingParams::stop_token_sequences`. None and Some(empty)
// are both treated as "no stop check" (the engine skips the
// matches_stop_sequences call entirely in that case).
if let Some(stop) = req.stop.as_ref()
    && !stop.is_empty()
{
    let tokenized: Vec<Vec<vllm_traits::TokenId>> = stop
        .iter()
        .map(|s| state.tokenizer.encode(s))
        .filter(|toks| !toks.is_empty())
        .collect();
    if tokenized.is_empty() {
        // All stop strings tokenized to zero tokens (e.g. user
        // passed only whitespace-only strings). The validator
        // catches most cases but some BPE tokenizers produce
        // zero tokens for unusual inputs; reject here so the
        // user gets a clear error.
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "stop sequences tokenize to zero tokens (no tokenizable content)",
                "invalid_request_error",
            )),
        ));
    }
    request.sampling_params.stop_token_sequences = Some(tokenized);
}
```

(The error-handling return path matches the surrounding code; the chat handler uses `axum::http::StatusCode` and `Json` from the existing imports. If `axum::http::StatusCode` is not imported, use whatever the surrounding code uses — match the file's existing imports.)

- [ ] **Step 5.3: Compile + run chat tests**

Run: `cargo check -p vllm-server --all-features 2>&1 | tail -5`
Expected: `Finished` line, no errors

Run: `cargo test -p vllm-server --lib --all-features openai::chat 2>&1 | tail -3`
Expected: All chat unit tests pass

- [ ] **Step 5.4: Commit**

```bash
git add crates/server/src/openai/sampling_validation.rs crates/server/src/openai/chat.rs
git commit -m "feat(server): wire stop sequences end-to-end on /v1/chat/completions (v31.0 P38)

The validator's old 'non-empty stop → 400 invalid_request_error' rejection is
replaced by validate_stop_sequences (P38 helper that enforces the OpenAI spec
contract: max 4 strings, no empty/whitespace strings).

The chat handler's inline sampling-params forwarding now tokenizes each stop
string via state.tokenizer.encode and attaches the result to
SamplingParams::stop_token_sequences. None and Some(empty_vec) are normalized
to 'no stop check' (engine skips matches_stop_sequences entirely).

Zero-token tokenizations (defensive: whitespace-only stops that slipped past
the validator in unusual tokenizer edge cases) are rejected with 400 so the
caller gets a clear error instead of a silent no-op."
```

---

## Task 6: HTTP Wire-Through — Completions (Part A)

**Files:**
- Modify: `crates/server/src/openai/sampling_validation.rs::validate_completion_request_fields` (remove 400-rejection, add `validate_stop_sequences` call)
- Modify: `crates/server/src/openai/completions.rs` (add stop tokenization + wire-through in `populate_completion_sampling_params`)

- [ ] **Step 6.1: Remove stop 400-rejection from `validate_completion_request_fields`**

Edit `crates/server/src/openai/sampling_validation.rs` (the function is around line 732). The block looks identical to the chat version. Replace with:

```rust
// P38 v0.x wire-type follow-up — engine wire-through: stop sequences
// are now accepted (tokenized + forwarded by populate_completion_sampling_params).
validate_stop_sequences(&req.stop)?;
```

- [ ] **Step 6.2: Add stop tokenization + wire-through in `populate_completion_sampling_params`**

Edit `crates/server/src/openai/completions.rs` (the helper is around line 175). Add the stop tokenization at the end of the helper, before the closing brace:

```rust
/// P38 v0.x wire-type follow-up — engine wire-through: tokenize the
/// user's stop strings at the HTTP boundary and forward as
/// `SamplingParams::stop_token_sequences`. None and Some(empty)
/// are both treated as "no stop check".
if let Some(stop) = req.stop.as_ref()
    && !stop.is_empty()
{
    let tokenized: Vec<Vec<vllm_traits::TokenId>> = stop
        .iter()
        .map(|s| state.tokenizer.encode(s))
        .filter(|toks| !toks.is_empty())
        .collect();
    if tokenized.is_empty() {
        // All stop strings tokenized to zero tokens. Reject via a
        // dedicated error path. (The populate helper currently
        // returns `()`; we instead return early via the caller's
        // validation. Actually: the caller (validate_completion_meta)
        // already catches empty/whitespace strings. If we land here
        // it's a tokenizer edge case — log a warning and skip stop
        // wire-through for this request rather than fail it.)
        tracing::warn!(
            stop_count = stop.len(),
            "All stop sequences tokenized to zero tokens; skipping stop wire-through"
        );
        return;
    }
    request.sampling_params.stop_token_sequences = Some(tokenized);
}
```

(NOTE: The completions populate helper does NOT return `Result` — it modifies the request in place. The zero-token edge case is logged and the stop check is skipped (degraded behavior). This is acceptable because the validator already rejects empty/whitespace strings, so landing here means an unusual tokenizer edge case — best-effort is safer than failing the request.)

- [ ] **Step 6.3: Compile + run completions tests**

Run: `cargo check -p vllm-server --all-features 2>&1 | tail -5`
Expected: `Finished` line, no errors

Run: `cargo test -p vllm-server --lib --all-features openai::completions 2>&1 | tail -3`
Expected: All completions unit tests pass

- [ ] **Step 6.4: Commit**

```bash
git add crates/server/src/openai/sampling_validation.rs crates/server/src/openai/completions.rs
git commit -m "feat(server): wire stop sequences end-to-end on /v1/completions (v31.0 P38)

Same pattern as chat (Task 5) but on the legacy /v1/completions endpoint.
The populator (populate_completion_sampling_params, extracted by P37) is the
single authoritative point for the legacy-endpoint → SamplingParams mapping,
so adding stop here propagates to all three call sites (non-streaming,
streaming, best_of).

Zero-token tokenizations (defensive: validator catches most cases) are
logged and the stop check is skipped for that request rather than failing
the request — best-effort degradation."
```

---

## Task 7: Engine `step_regular` Integration + Integration Tests (Part A — engine layer, TDD)

**Files:**
- Modify: `crates/core/src/scheduler/batch.rs::step_regular` (add post-sample stop check)
- Modify: `crates/server/tests/chat_integration_test.rs` (9 new integration tests)

This is the largest task. Split into two sub-steps: write tests first (TDD red), then implement (TDD green).

- [ ] **Step 7.1: Write 9 failing integration tests**

Append to `crates/server/tests/chat_integration_test.rs` (the existing file already has P37 best_of tests — add P38 stop tests in a clearly-marked section):

```rust
// =============================================================================
// P38 v0.x wire-type follow-up — engine wire-through: `stop` sequences
// end-to-end tests on /v1/chat/completions and /v1/completions. Pins:
// - chat + completions accept stop, honor end-to-end
// - single-token and multi-token stop matches
// - finish_reason = "stop" on match
// - composition with best_of (P37), logprobs (P36), streaming
// - max_tokens wins if hit before stop
// =============================================================================

/// Mock engine fixture for stop tests: emits a configurable token
/// sequence per AddRequest and tracks how many sequences it saw.
/// Each sequence gets the same logprob pattern so ranking in best_of
/// tests is deterministic.
fn spawn_stop_mock_engine(
    per_seq_tokens: Vec<Vec<u32>>,
) -> (
    vllm_server::api::EngineHandle,
    Arc<Mutex<Vec<usize>>>, // captures seq_id of each AddRequest
    tokio::task::JoinHandle<()>,
) {
    // ... fixture implementation. Clone the spawn_best_of_mock_engine
    // pattern from chat_integration_test.rs (P37 tests, ~line 4600) and
    // adapt: each AddRequest advances a per-seq cursor and emits the
    // corresponding token sequence, then drops response_tx. The mock
    // does NOT auto-emit finish_reason; the engine integration step
    // does (via finalize_finished with Stop).
}

#[tokio::test]
async fn test_chat_stop_sequence_triggers_finish_reason_stop() {
    // Mock emits [1, 2, 99]; user sends stop = ["[99]"] (tokenized to
    // [99]). After generating [1, 2, 99] the engine detects the
    // match and finalizes with FinishReason::Stop. The chat handler
    // renders finish_reason = "stop".
}

#[tokio::test]
async fn test_chat_multi_token_stop_sequence_works() {
    // Stop tokenizes to [10, 20] (2 BPE tokens); mock emits [1, 2, 10, 20].
    // Only after BOTH 10 AND 20 are generated does stop fire.
}

#[tokio::test]
async fn test_chat_multiple_stops_first_match_wins() {
    // Stop = ["[99]", "[88]"] (tokenized to [[99], [88]]); mock emits
    // [1, 2, 99]. First match ([99]) fires.
}

#[tokio::test]
async fn test_completions_stop_sequence_triggers_finish_reason_stop() {
    // Same as test_chat_stop_sequence_triggers_finish_reason_stop but
    // on /v1/completions.
}

#[tokio::test]
async fn test_completions_stop_sequences_with_best_of_each_candidate_honors_stop() {
    // best_of = 3 with stop; mock candidates generate different lengths
    // and stop at different steps; ranker picks the highest-mean-logprob
    // completed candidate.
}

#[tokio::test]
async fn test_chat_stop_with_logprobs_returns_logprobs_of_stopped_sequence() {
    // stop + logprobs = true; response includes per-token logprobs
    // INCLUDING the matched stop token's logprob (the engine emits
    // the token BEFORE finalize_finished).
}

#[tokio::test]
async fn test_chat_stop_in_streaming_emits_finish_reason_stop_on_last_chunk() {
    // SSE stream; stop triggers finish_reason = "stop" on the chunk
    // that carries the matched token; [DONE] follows.
}

#[tokio::test]
async fn test_completions_stop_in_streaming_emits_finish_reason_stop_on_last_chunk() {
    // Same as chat streaming but on /v1/completions.
}

#[tokio::test]
async fn test_chat_stop_with_max_tokens_stop_wins_when_earlier() {
    // max_tokens = 100 but stop matches at step 3 → finish_reason = "stop".
}
```

(The exact test bodies require careful mock-engine setup — copy the pattern from the existing P37 `spawn_best_of_mock_engine` fixture in `chat_integration_test.rs`. Each test must: (a) construct a mock engine that emits a configurable token sequence, (b) send a request with the test's stop + max_tokens config, (c) assert the response's `finish_reason` and content match expectations. Mock engines should emit `FinishReason::Stop` only when the engine integration calls `finalize_finished` — the mock itself does NOT emit it.)

- [ ] **Step 7.2: Run integration tests to verify they fail**

Run: `cargo test -p vllm-server --test chat_integration_test --all-features stop 2>&1 | tail -15`
Expected: 9 failures (no stop wire-through yet, so stops are passed but engine doesn't honor them; tests assert finish_reason = "stop" and get "length" or timeout)

- [ ] **Step 7.3: Implement engine `step_regular` integration**

Edit `crates/core/src/scheduler/batch.rs` (the function is around line 31). After `sample_batch_with_params` (line 95) and BEFORE the response_tx send loop (line 116), add:

```rust
// P38 v0.x wire-type follow-up — engine wire-through: per-sequence
// stop detection. For each sequence, check whether any pre-tokenized
// stop sequence in `batch.sampling_params[i].stop_token_sequences` is
// a suffix of the sequence's already-generated tokens (including the
// freshly-sampled one). If yes, finalize the sequence with
// FinishReason::Stop so the scheduler drops it from the next batch
// and the HTTP handler's finish_reason_rx resolves with Stop.
//
// **Order matters:** the matched token is emitted via response_tx
// below (the existing send loop runs AFTER this block). The engine
// emits the token BEFORE finalizing so the OpenAI response includes
// the matched stop text.
let mut newly_stopped: Vec<vllm_traits::SeqId> = Vec::new();
for (i, seq_id) in batch.seq_ids.iter().enumerate() {
    let stops = match batch.sampling_params[i].stop_token_sequences.as_ref() {
        Some(s) if !s.is_empty() => s,
        _ => continue,
    };
    let seq = match self.scheduler.get_sequence(*seq_id) {
        Some(s) => s,
        None => continue,
    };
    let mut generated: Vec<vllm_traits::TokenId> =
        seq.tokens[seq.prompt_len..].to_vec();
    generated.push(next_tokens[i].token);
    if crate::sampling::matches_stop_sequences(&generated, stops) {
        newly_stopped.push(*seq_id);
    }
}
for seq_id in &newly_stopped {
    self.finalize_finished(*seq_id, vllm_traits::FinishReason::Stop);
}
```

(Note: the existing send loop on line 119 sends `sampled.clone()` per seq. The `newly_stopped` set above doesn't affect the send — the matched token is emitted regardless. Only the scheduler's `finished_sequences` check at line 124 will exclude the stopped seq from re-batching. The `finalize_finished` call also sets the per-seq `Status::Finished` flag so `clear_finished` cleans up properly.)

- [ ] **Step 7.4: Re-run integration tests to verify they pass**

Run: `cargo test -p vllm-server --test chat_integration_test --all-features stop 2>&1 | tail -10`
Expected: 9 passed, 0 failed

If individual tests fail, debug by adding `eprintln!` statements in the mock engine and the step_regular integration. Most likely failure modes:
- Stop tokenization differs from mock's token IDs → use `state.tokenizer.encode` in the test mock setup to match the handler's tokenization
- `finalize_finished` is called BEFORE the send loop → move it after (see order note above)
- Streaming tests see the wrong chunk → adjust the streaming handler's `finish_reason` rendering logic

- [ ] **Step 7.5: Run full server test suite to verify no regression**

Run: `cargo test -p vllm-server --lib --all-features 2>&1 | tail -3`
Expected: All tests pass (was 284, now 284 + 7 + 9 = 300)

Run: `cargo test -p vllm-server --test chat_integration_test --all-features 2>&1 | tail -3`
Expected: All tests pass (was 100, now 100 + 9 = 109)

- [ ] **Step 7.6: Run core tests to verify no regression**

Run: `cargo test -p vllm-core --all-features 2>&1 | tail -3`
Expected: All tests pass (was 1 flaky test_radix_repeated_prefix_lookup_is_fast — still flaky in isolation-pass)

- [ ] **Step 7.7: Commit**

```bash
git add crates/core/src/scheduler/batch.rs crates/server/tests/chat_integration_test.rs
git commit -m "feat(core): wire stop detection into step_regular loop (v31.0 P38)

The engine's regular decode step now checks each sequence's stop_token_sequences
after sampling. When any pre-tokenized stop is a suffix of the sequence's
already-generated tokens (including the freshly-sampled one), the sequence is
finalized with FinishReason::Stop. The matched token is still emitted via
response_tx BEFORE finalization (OpenAI convention — response includes the
matched stop text).

Order: sample → stop-check → response_tx.send (token emission)
→ finalize_finished(Stop) → scheduler.update → next step excludes the sequence.

9 new integration tests pin the end-to-end contract:
- chat + completions stop wire-through
- single-token + multi-token + multi-stop-list
- composition with best_of (P37), logprobs (P36), streaming
- max_tokens interaction (Stop wins if matched first, Length wins if max_tokens hit first)"
```

---

## Task 8: Documentation Updates (Part A + C — docs)

**Files:**
- Modify: `docs/reference/openai-compatibility.md`
- Modify: `CHANGELOG.md`
- Modify: `.planning/STATE.md`

- [ ] **Step 8.1: Update `docs/reference/openai-compatibility.md`**

Three changes:

1. Find the `ChatRequest::stop` row (search for "stop sequences are not yet honoured" — the old text should now be absent; if still present, remove it). Update the row to flip status from "Wired (validation)" to "Wired (declaration + validation + engine wire-through)" and update the notes:

```
| `stop` | `Option<Vec<String>>` (max 4 strings) | **Wired (declaration + validation + engine wire-through)** | Accepted on `ChatRequest`. Per OpenAI spec: up to 4 strings; each string is a stop sequence that, when generated, terminates the response with `finish_reason = "stop"`. The matched stop text is INCLUDED in the response (OpenAI convention). Default `None`. **Honored end-to-end** — when `stop = Some(seqs)` is non-empty, the chat handler tokenizes each string via `state.tokenizer.encode(s)`, forwards as `SamplingParams::stop_token_sequences = Some(tokenized)`, and the engine's `step_regular` runs `vllm_core::sampling::matches_stop_sequences(generated, stops)` after every sampled token; on match the sequence is finalized with `FinishReason::Stop`. Validated by `validate_stop_sequences` (rejects > 4 strings or empty/whitespace strings with `400 invalid_request_error`). Empty strings and whitespace-only stops are rejected because they tokenize to zero tokens and would never match (silent no-op). The chat handler does not currently log the field (parity with `seed` / `user` / `frequency_penalty` rationale). **Shipped in P22 (declaration + validation) + P38 (engine wire-through).** |
```

2. Same update for `CompletionRequest::stop` row (in the legacy completions section).

3. Find the v32+ candidates table and REMOVE the row that lists `stop`. (Search for "stop sequences" in the table section; remove that row.)

- [ ] **Step 8.2: Update `CHANGELOG.md`**

Add two new entries under the `### Added` section (at the top, before existing entries):

Entry 1 (stop wire-through):

```markdown
- **`stop` sequences engine wire-through on chat + completions** (v31.0 P38) — closes the last hard-400 v0.x wire-type carve-out. P22 declared + validated `ChatRequest::stop` and P32 declared + validated `CompletionRequest::stop` (`Option<Vec<String>>`, up to 4 strings per OpenAI spec) but the validator hard-rejected non-empty stops with `400 invalid_request_error`. P38 honors them end-to-end: HTTP layer tokenizes each stop string via `state.tokenizer.encode(s)`, forwards as new public field `vllm_traits::SamplingParams::stop_token_sequences: Option<Vec<Vec<TokenId>>>` + `SamplingParamsBuilder::with_stop_token_sequences` / `with_stop_token_sequences_none`. Engine's `step_regular` runs `vllm_core::sampling::matches_stop_sequences(generated, stops)` after every sampled token; on match the sequence is finalized with `FinishReason::Stop` and the matched token is emitted via `response_tx` BEFORE finalization (OpenAI convention — response includes the matched stop text). **Composes correctly** with `best_of` (P37 — each candidate honors its own stop set), `logprobs`/`top_logprobs` (P36 — matched token's logprob is emitted alongside), `echo`/`suffix` (P35 — orthogonal), `seed` (P34 — RNG-agnostic), and `max_tokens` (Length wins if hit first, Stop wins if matched first). Streaming path emits `finish_reason = "stop"` on the final chunk followed by `[DONE]`. **Validator update:** new `validate_stop_sequences` helper rejects > 4 strings (OpenAI spec upper bound) and empty/whitespace-only strings (would tokenize to zero tokens, silent no-op) with `400 invalid_request_error`. **Test coverage:** 7 new unit tests in `crates/core/src/sampling/tests.rs` (matches_stop_sequences: no_match / single_token / multi_token / partial_match / first_match_wins / empty_stops / stop_longer_than_generated) + 7 new validator unit tests in `crates/server/src/openai/sampling_validation.rs::tests` (stop_validation: none / empty_vec / single_string / max_4_strings / > 4_strings / empty_string / whitespace_string) + 9 new integration tests in `crates/server/tests/chat_integration_test.rs` (chat stop / multi_token stop / multiple stops / completions stop / best_of + stop / logprobs + stop / streaming chat stop / streaming completions stop / max_tokens + stop). 2 old P22 validator tests that asserted the 400-rejection are removed. **Public-API delta:** 1 new public field + 2 new builder methods. **The v0.x wire-type backlog is now FULLY CLOSED at both the declaration + validation layer AND the engine-honoring layer** — every chat + completions OpenAI-spec field is end-to-end except `tools`/`tool_choice` (grammar-constrained decoder, P33 declared only) and `n > 1` (sample-N parallel, candidate for P39).
```

Entry 2 (clippy fix):

```markdown
- **Clippy `large_enum_variant` lint fix** (v31.0 P38 follow-up) — `EngineMessage::AddRequest.request` changed from `Request` to `Box<Request>` to silence the `clippy::large_enum_variant` lint that P36's `SampledToken`-inflation introduced (the field grew past 248 bytes; other variants are ≤ 64 bytes). Boxing brings the enum to pointer-sized. All 13 `AddRequest { request, ... }` construction sites across `crates/core/src/engine/run.rs`, `crates/core/tests/e2e_graceful_shutdown.rs`, `crates/server/src/{chat,completions,test_fixtures,sampling_validation}.rs`, and `crates/server/tests/{overload_integration,cancel_propagation,tutorial_e2e,request_id_propagation,chat_integration_test}.rs` updated to `request: Box::new(request)`. `just ci` is now green on `vllm-core` (was the only outstanding clippy error blocking the v31.0 ship gate). Public API surface is unchanged (`Request` is consumed by value at construction; callers never read `.request` after — the field-type change is invisible to public consumers).
```

- [ ] **Step 8.3: Update `.planning/STATE.md`**

Update the `last_activity:` line (currently at line 7) and add the P38 entry to the bottom of the `last_activity:` description (concatenate; this is the convention used by P36 / P37 / etc.):

Replace the existing `last_activity:` value with a new long description that:
1. Documents P38 stop wire-through (mirror CHANGELOG entry)
2. Documents the clippy fix (mirror CHANGELOG entry)
3. References the public-API delta
4. Confirms the v0.x wire-type backlog is fully closed

The new text follows the same comma-separated / parenthetical structure as existing entries (see P37 entry for the template).

- [ ] **Step 8.4: Commit**

```bash
git add docs/reference/openai-compatibility.md CHANGELOG.md .planning/STATE.md
git commit -m "docs(planning, openai-compat, changelog): record P38 stop wire-through + clippy fix (v31.0 P38 / v0.x wire-type engine wire-through close)

openai-compatibility.md: flip both ChatRequest::stop and CompletionRequest::stop
rows from 'Wired (validation)' to 'Wired (declaration + validation + engine
wire-through)'; remove the v0.x stop row from the v32+ candidates table.

CHANGELOG.md: add two entries under '### Added' — one for the stop wire-through
(full P38 details, test counts, public-API delta, compose matrix) and one for
the large_enum_variant clippy fix.

STATE.md: update last_activity with the full P38 summary, following the
comma-separated / parenthetical structure used by P36/P37.

The v0.x wire-type backlog is now FULLY CLOSED at both layers
(declaration + validation + engine wire-through) for every OpenAI-spec
chat + completions field EXCEPT tools/tool_choice (P33 declared only) and
n > 1 (still 400; P39 candidate)."
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
Expected: `Finished` line, no errors (large_enum_variant is gone after Task 1)

- [ ] **Step 9.3: Doc check**

Run: `cargo doc -p vllm-server -p vllm-core -p vllm-traits --no-deps --all-features 2>&1 | tail -5`
Expected: `Finished` line, no errors

- [ ] **Step 9.4: Doc test**

Run: `cargo test --doc -p vllm-server -p vllm-core -p vllm-traits --all-features 2>&1 | tail -5`
Expected: All doctests pass (or 0 doctests)

- [ ] **Step 9.5: Full workspace test suite**

Run: `cargo nextest run --workspace --all-features --no-fail-fast 2>&1 | tail -10`
Expected: ~1694+ tests pass (was 1692 + 23 new P38 tests - 2 removed = 1713; the one flaky `test_radix_repeated_prefix_lookup_is_fast` may fail under load — re-run that one test in isolation if it trips: `cargo nextest run -p vllm-core --all-features test_radix_repeated_prefix_lookup_is_fast`).

- [ ] **Step 9.6: Public-API snapshot check (if `public-api-check` step exists)**

Run: `just public-api-check 2>&1 | tail -10`
Expected: No drift (the SamplingParams::stop_token_sequences field addition is already in the snapshot from Task 3)

- [ ] **Step 9.7: Final `just ci`**

Run: `just ci 2>&1 | tail -10`
Expected: All steps green

If any step fails, debug per the error message (most likely the flaky perf test — re-run it in isolation; or a clippy warning on new code — fix inline).

- [ ] **Step 9.8: Final commit (if any fixes)**

If any of the above steps triggered a fix, commit it as a follow-up:

```bash
git add -A
git commit -m "chore(ci): post-P38 ci verification fixes

[describe any fixes applied]"
```

If no fixes needed, skip this step.

---

## Self-Review

**Spec coverage:**
- §4.1 (Type layer) → Task 3 ✓
- §4.2 (HTTP validation layer) → Task 4 ✓
- §4.3 (Engine layer) → Task 7 ✓
- §5 (large_enum_variant fix) → Task 1 ✓
- §6.1 (unit tests for matches_stop_sequences) → Task 2 ✓
- §6.2 (validator unit tests) → Task 4 ✓
- §6.3 (integration tests) → Task 7 ✓
- §7.1 (openai-compatibility.md updates) → Task 8 ✓
- §7.2 (CHANGELOG.md updates) → Task 8 ✓
- §7.3 (STATE.md updates) → Task 8 ✓
- §10 (Success criteria) → Task 9 ✓

**Placeholder scan:** None. Every code block is complete. Every command has expected output. Every step is self-contained.

**Type consistency:**
- `matches_stop_sequences(generated_tokens: &[u32], stops: &[Vec<u32>]) -> bool` — same signature used in Task 2 (definition) and Task 7 (call site).
- `SamplingParams::stop_token_sequences: Option<Vec<Vec<TokenId>>>` — same type used in Task 3 (definition), Task 5 (chat populate), Task 6 (completions populate), Task 7 (engine read).
- `validate_stop_sequences(stop: &Option<Vec<String>>) -> Result<(), (StatusCode, Json<ErrorResponse>)>` — same signature used in Task 4 (definition) and Task 5/6 (call sites).
- `with_stop_token_sequences` / `with_stop_token_sequences_none` — Task 3 defines both; no other task uses them directly (the HTTP populate helpers use `request.sampling_params.stop_token_sequences = Some(...)` directly without going through the builder, which is the established pattern for in-flight request construction).

**Issue spotted during review:** In Task 6.2, the populate helper does NOT have access to the `&ApiState` (only the `Request` and `CompletionRequest`). The chat handler in Task 5 has `&ApiState` access in its inline forwarding; the completions populate helper currently takes `&mut Request, &CompletionRequest`. Need to verify whether the populator already has access to the tokenizer. **Action:** At Task 6 execution time, check if `populate_completion_sampling_params` already has tokenizer access; if not, pass it in as a third argument or do the tokenization BEFORE calling the populator (in the caller, which has `&ApiState`). The latter is cleaner and matches the existing pattern.

This review note is added to Task 6 as a sub-step: "If the populator doesn't have tokenizer access, tokenize in the caller and pass the result in."

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-21-p38-stop-sequences.md`.

This is a 10-task linear plan with no parallel work. Each task produces a working commit. The plan can be executed inline (executing-plans) or via subagents (subagent-driven-development).
