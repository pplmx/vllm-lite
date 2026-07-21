# P38 — `stop` Sequences Engine Wire-Through + `large_enum_variant` Lint Fix

**Date:** 2026-07-21
**Phase:** v31.0 / P38
**Author:** vllm-lite evolution iteration (post-P37 ROI analysis)
**Status:** Approved design (2026-07-21)

---

## 1. Motivation

After P37 closed the last v0.x wire-type engine-honoring carve-out (`best_of` on legacy `/v1/completions`), two residual items remain at the v0.x layer:

1. **`stop` sequences** — `ChatRequest::stop` and `CompletionRequest::stop` are both declared on the wire type, but the HTTP-layer validator (`validate_chat_request_fields` / `validate_completion_request_fields`) currently **rejects non-empty `stop` with `400 invalid_request_error`** ("stop sequences are not yet honoured"). This is a broken OpenAI contract — clients that set `stop` (one of the most-used OpenAI parameters) cannot use vllm-lite at all.

2. **`large_enum_variant` clippy lint** — P36's `SampledToken` wire-breaking signature change inflated the size of `EngineMessage::AddRequest.request` to >248 bytes, dwarfing the other enum variants. The `just ci` clippy step is currently red on `vllm-core` (the only outstanding CI error). Unblocking `just ci` is a pre-condition for the v31.0 "Perfection & Elegance" milestone ship criteria.

Both items have **very high ROI**: small implementation cost, high user-value (fixes a broken contract + unblocks CI gate). Bundling them in one PR keeps the work scoped, atomic, and easy to revert if either piece regresses.

## 2. Goals

- **G1:** `stop` sequences accepted and honored end-to-end on both `/v1/chat/completions` and `/v1/completions`.
- **G2:** `just ci` runs green on the workspace (clippy `-D clippy::correctness -D clippy::suspicious -D clippy::perf`).
- **G3:** Public-API surface increase ≤ 1 new field + 1 new builder method.
- **G4:** No regression to P37 (`best_of`), P36 (`logprobs`/`top_logprobs`), P35 (`echo`/`suffix`), P34 (`seed`), P30 (`logit_bias`), P29 (`frequency_penalty` sign-aware), or P28 (`presence_penalty`).

## 3. Non-Goals

- **N1:** `response_format: json_object` constrained-decoder honoring — v32+ work (declared in P22).
- **N2:** `tools` / `tool_choice` engine honoring — v32+ work (declared in P33).
- **N3:** `n > 1` engine honoring — separate phase (potential P39).
- **N4:** Cross-request stop sequence caching — fresh encode per request (tokenization is sub-millisecond for typical stops ≤ 4 tokens).

## 4. Design — Part A: `stop` Sequences Engine Honoring

### 4.1 Type Layer (`crates/traits/src/types.rs`)

Add a single new public field to `SamplingParams`:

```rust
/// OpenAI-spec `stop` sequences, pre-tokenized at the HTTP boundary.
/// Each inner `Vec<TokenId>` is one stop string tokenized by the
/// model's BPE tokenizer. When the engine detects that any of these
/// token sequences matches the suffix of a sequence's already-generated
/// tokens, it terminates that sequence with `FinishReason::Stop` and
/// emits the matching token to the HTTP layer (OpenAI convention —
/// the matched stop text is included in the response).
///
/// `None` → no stop check (default-path overhead is zero).
/// `Some(vec![])` → no stop check (treated identically to `None` at
/// the HTTP layer; the validator normalizes empty vectors to `None`
/// before forwarding).
/// `Some(non_empty)` → check after every sampled token.
///
/// Pre-tokenization keeps the sampler and engine step-loop free of
/// tokenizer dependencies (the HTTP layer is the only place that
/// needs `state.tokenizer`).
#[serde(default, skip_serializing_if = "Option::is_none")]
pub stop_token_sequences: Option<Vec<Vec<TokenId>>>,
```

Plus a builder method on `SamplingParamsBuilder`:

```rust
pub fn with_stop_token_sequences(
    mut self,
    stops: Vec<Vec<TokenId>>,
) -> Self {
    self.params.stop_token_sequences = Some(stops);
    self
}
```

`Batch.sampling_params` is already `Vec<SamplingParams>` (per-seq; confirmed in `crates/core/src/scheduler/batch_composer/compose/decode.rs:66`), so per-sequence stop sequences are supported natively — no `Batch` struct change needed.

### 4.2 HTTP Validation Layer (`crates/server/src/openai/sampling_validation.rs`)

**Remove** the two existing 400-rejection blocks:

```rust
// REMOVE: in validate_chat_request_fields and validate_completion_request_fields
if let Some(stop) = req.stop.as_ref()
    && !stop.is_empty()
{
    return Err((StatusCode::BAD_REQUEST, Json(ErrorResponse::new(
        "stop sequences are not yet honoured; ...",
        "invalid_request_error",
    ))));
}
```

**Add** a new helper `validate_stop_sequences(stop: &Option<Vec<String>>) -> Result<(), ErrorResponse>`:

```rust
/// Validate `stop` per OpenAI spec:
/// - `None` → pass (default)
/// - `Some(vec![])` → pass (treated as None downstream)
/// - `Some(non_empty)` → reject if any string is empty (OpenAI's
///   stop[] elements are non-empty strings by spec)
/// - Maximum 4 strings per OpenAI spec (some servers enforce this)
fn validate_stop_sequences(
    stop: &Option<Vec<String>>,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if let Some(stops) = stop {
        if stops.len() > 4 {
            return Err(/* 400 invalid_request_error: too many stop sequences */);
        }
        if stops.iter().any(|s| s.is_empty()) {
            return Err(/* 400 invalid_request_error: empty stop string */);
        }
    }
    Ok(())
}
```

**Wire-through** in `populate_chat_sampling_params` (chat) and `populate_completion_sampling_params` (completions, already extracted by P37): if `stop_token_sequences` is needed in the request, the HTTP layer tokenizes each string into `Vec<TokenId>` via `state.tokenizer.encode(&s)` and sets `request.sampling_params.stop_token_sequences = Some(tokenized)`.

**Edge case:** if `state.tokenizer.encode(stop_str)` returns `vec![]` (e.g. whitespace-only stop that decodes to no tokens), reject with `400 invalid_request_error: "stop sequence tokenizes to zero tokens; not a valid stop"`. This protects the engine from a stop that can never match.

### 4.3 Engine Layer (`crates/core/src/scheduler/batch.rs::step_regular`)

After `sample_batch_with_params` returns the new tokens and **before** `self.scheduler.update(...)` and the response_tx send loop:

```rust
// New step: per-sequence stop detection (P38 v0.x wire-type
// engine wire-through). For each sequence, append the freshly
// sampled token to its already-generated tokens and check whether
// any stop_token_sequence is a suffix of the result. If yes:
// finalize_finished(seq_id, FinishReason::Stop) so the scheduler
// drops the sequence from the next batch and the HTTP handler's
// finish_reason_rx resolves with `Stop`.
let mut newly_stopped: Vec<SeqId> = Vec::new();
for (i, seq_id) in batch.seq_ids.iter().enumerate() {
    let Some(stops) = batch.sampling_params[i].stop_token_sequences.as_ref() else {
        continue;
    };
    if stops.is_empty() {
        continue;
    }
    let sampled = &next_tokens[i];
    let mut seq = match self.scheduler.get_sequence(*seq_id) {
        Some(s) => s,
        None => continue,
    };
    // Append the new token to the sequence's already-generated
    // tokens (scheduler.update below will commit it; here we use
    // the in-memory view for the stop check).
    let mut generated: Vec<TokenId> =
        seq.tokens[seq.prompt_len..].to_vec();
    generated.push(sampled.token);

    if matches_stop_sequences(&generated, stops) {
        newly_stopped.push(*seq_id);
    }
}
for seq_id in &newly_stopped {
    self.finalize_finished(*seq_id, FinishReason::Stop);
}
```

Helper (private to engine):

```rust
/// Returns `true` iff any token sequence in `stops` is a suffix of
/// `generated_tokens`. O(N × M) where N = generated_tokens.len() and
/// M = sum(stop.len()) — both are tiny in practice (generated ≤
/// max_tokens, stops ≤ 4 sequences of ≤ 4 tokens each).
fn matches_stop_sequences(
    generated_tokens: &[TokenId],
    stops: &[Vec<TokenId>],
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

**Important behavioral note:** the matched token IS emitted to `response_tx` BEFORE `finalize_finished` is called (the existing `try_send` loop on line 119 happens before this new block conceptually; we re-order to keep the emission first). OpenAI's contract includes the matched stop text in the response. The next step will not include this sequence in the batch (because `finalize_finished` removes it from the active set).

**Streaming behavior:** the existing `finish_reason_tx` oneshot delivers `FinishReason::Stop` to the handler. The chat streaming handler already maps `FinishReason::Stop → "stop"` in its finish-reason chunk. The completions streaming handler does the same in its SSE emission.

### 4.4 Compose with Prior Wire-Throughs

| Interaction | Behavior |
|---|---|
| `stop` × `best_of` (P37) | Each `best_of` candidate carries its own `stop_token_sequences` (the per-candidate `populate_completion_sampling_params` clones the user-supplied stop set into every candidate). The N candidates stop independently. |
| `stop` × `logprobs` (P36) | The matched stop token's `SampledToken::logprob` is emitted alongside it (the engine emits the token BEFORE finalizing). The logprob response shape is unchanged. |
| `stop` × `echo` (P35) | `echo=true` prepends the prompt to the response text (response-side formatting); `stop` controls the sequence's tail (engine-side termination). They compose orthogonally. |
| `stop` × `suffix` (P35) | `suffix` is response-side text appended after generation; `stop` cuts generation. If `stop` matches early, `suffix` still appends to the shorter text (matches OpenAI's behavior). |
| `stop` × `seed` (P34) | `seed` controls RNG draws; `stop` checks after sampling. Orthogonal. |
| `stop` × `max_tokens` | `max_tokens` check runs first (existing path via `scheduler/engine/update.rs`); if `stop` matches BEFORE `max_tokens` is reached, `FinishReason::Stop` wins. If `max_tokens` is reached first, `FinishReason::Length` wins. |
| `stop` × `frequency_penalty` / `presence_penalty` / `logit_bias` | Sampling-time filters (P28/P29/P30) are orthogonal to the post-sample stop check. |
| `stop` × streaming | Engine emits the matched token via the existing per-token `response_tx.try_send`, then closes the channel. Handler emits the token chunk + a final chunk with `finish_reason: "stop"` + `[DONE]` sentinel. |

## 5. Design — Part C: `large_enum_variant` Lint Fix

### 5.1 Root Cause

`crates/core/src/types/messages.rs::EngineMessage::AddRequest` carries `request: Request` directly. After P36 added the `top_logprobs: Option<u32>` field to `SamplingParams`, `Request` grew past 248 bytes. The other enum variants (`GetEmbeddings`, `Shutdown`, `GetMetrics`, etc.) are ≤ 64 bytes. The size delta trips `clippy::large_enum_variant`.

### 5.2 Fix

Change `EngineMessage::AddRequest.request` from `Request` to `Box<Request>`:

```rust
// crates/core/src/types/messages.rs
AddRequest {
    request: Box<Request>,  // was: Request
    response_tx: mpsc::Sender<SampledToken>,
    seq_id_tx: Option<oneshot::Sender<SeqId>>,
    finish_reason_tx: Option<oneshot::Sender<FinishReason>>,
    request_id: Option<String>,
},
```

The other variants are unchanged. The enum now fits in a single 8-byte word (pointer-sized discriminant + padding).

### 5.3 Construction Site Updates

All `EngineMessage::AddRequest { request, ... }` constructions across the workspace must wrap `request` in `Box::new(...)`. Estimated sites:

- `crates/server/src/openai/chat.rs` — non-streaming + streaming chat handlers
- `crates/server/src/openai/completions.rs` — non-streaming + streaming + best_of (P37)
- Test files in `crates/server/tests/*.rs` — many integration tests build mock engines

The compiler will flag every site (the field type changed). Pattern: `request: Box::new(request)` at each call site.

### 5.4 Risk

- `Box<Request>` adds one heap allocation per request. For typical request sizes (< 2 KiB) this is sub-microsecond. The HTTP layer already does `Request::new(...)` which itself allocates the inner `Vec<TokenId>` — Boxing the outer is a marginal extra cost.
- Construction sites must be updated atomically (the compiler enforces this).
- Public API: `EngineMessage` is `pub` in `vllm_core::types` but the `request` field is consumed by value at construction time — callers don't read `.request` after construction. The signature change is therefore invisible to public consumers.

## 6. Test Matrix

### 6.1 Unit Tests — `crates/core/src/sampling/tests.rs` (new file `stop_check.rs` or extend existing)

| Test | What it pins |
|---|---|
| `matches_stop_sequences_no_match_returns_false` | A stop "xyz" doesn't match generated "abc" |
| `matches_stop_sequences_single_token_match` | 1-token stop matches when generated ends with it |
| `matches_stop_sequences_multi_token_match` | 2/3-token stop matches after N steps |
| `matches_stop_sequences_partial_match_returns_false` | Generated ends with first 2 of 3 stop tokens — NO match |
| `matches_stop_sequences_first_match_wins` | Multiple stops in list, first suffix-match returns true |
| `matches_stop_sequences_empty_stops_returns_false` | `stops = vec![]` → never matches |
| `matches_stop_sequences_stop_longer_than_generated_returns_false` | 5-token stop, 3 generated tokens → no match |

### 6.2 Validator Unit Tests — `crates/server/src/openai/sampling_validation.rs::tests`

| Test | What it pins |
|---|---|
| `stop_validation_none_passes` | `stop = None` → OK |
| `stop_validation_empty_vec_passes` | `stop = Some(vec![])` → OK (normalized to None) |
| `stop_validation_single_string_passes` | `stop = Some(vec!["\n\n".into()])` → OK |
| `stop_validation_max_4_strings_passes` | `stop = Some(vec![a, b, c, d])` → OK |
| `stop_validation_more_than_4_strings_returns_400` | `stop = Some(vec![a, b, c, d, e])` → 400 |
| `stop_validation_empty_string_returns_400` | `stop = Some(vec!["".into()])` → 400 |
| `stop_validation_whitespace_string_tokenizes_to_zero_returns_400` | `stop = Some(vec![" ".into()])` and tokenizer produces 0 tokens → 400 |

**Remove** the old tests that asserted the 400-rejection (`test_chat_rejects_non_empty_stop_with_400`, `test_completions_rejects_non_empty_stop_with_400`).

### 6.3 Integration Tests — `crates/server/tests/chat_integration_test.rs`

| Test | What it pins |
|---|---|
| `test_chat_stop_sequence_triggers_finish_reason_stop` | Mock engine emits tokens then a stop match; chat handler returns `finish_reason: "stop"` |
| `test_chat_multi_token_stop_sequence_works` | Stop = "\n\n" (2 BPE tokens); only after BOTH are generated does stop fire |
| `test_chat_multiple_stops_first_match_wins` | Stop = ["a", "b"]; generated ends with "b" → stop fires |
| `test_completions_stop_sequence_triggers_finish_reason_stop` | Same as chat but on `/v1/completions` |
| `test_completions_stop_sequences_with_best_of_each_candidate_honors_stop` | best_of=3 with stop; mock candidates stop at different steps; ranker picks the highest-mean-logprob completed candidate |
| `test_chat_stop_with_logprobs_returns_logprobs_of_stopped_sequence` | stop + logprobs=true; response includes per-token logprobs INCLUDING the matched stop token |
| `test_chat_stop_in_streaming_emits_finish_reason_stop_on_last_chunk` | SSE stream; stop triggers `finish_reason: "stop"` on the final chunk |
| `test_completions_stop_in_streaming_emits_finish_reason_stop_on_last_chunk` | Same on completions |
| `test_chat_stop_with_max_tokens_max_tokens_wins_when_earlier` | stop would match at step 5 but max_tokens=3 → finish_reason="length" |
| `test_chat_stop_with_max_tokens_stop_wins_when_earlier` | max_tokens=100 but stop matches at step 5 → finish_reason="stop" |

### 6.4 large_enum_variant Tests

No new tests — clippy `cargo clippy -D clippy::perf` succeeding is the verification.

## 7. Documentation Updates

### 7.1 `docs/reference/openai-compatibility.md`

| Section | Change |
|---|---|
| `ChatRequest::stop` row | Status: "Wired (validation)" → "Wired (declaration + validation + engine wire-through)". Update notes to describe tokenize-at-HTTP-boundary, post-sample suffix check, FinishReason::Stop, max-4-strings limit, non-empty-string requirement. Cite P38 helpers (`validate_stop_sequences`, `matches_stop_sequences`, `stop_token_sequences` field). |
| `CompletionRequest::stop` row | Same flip. Cite P38. |
| `stop` row in v32+ candidates table | **Remove** (now shipped). |
| `stop` mention in v0.2 candidates table | Update "Shipped in P38" note. |

### 7.2 `CHANGELOG.md`

Add a new "Added" entry under the unreleased / v31.0 section:

> **`stop` sequences engine wire-through on chat + completions** (v31.0 P38) — closes the last hard-400 v0.x wire-type carve-out. P32/P22 declared + validated `ChatRequest::stop` and `CompletionRequest::stop` (`Option<Vec<String>>`) but the validator hard-rejected non-empty stops with `400 invalid_request_error`. P38 honors them end-to-end: HTTP layer tokenizes each stop string via the model tokenizer, forwards as `SamplingParams::stop_token_sequences: Option<Vec<Vec<TokenId>>>`, engine's `step_regular` runs `matches_stop_sequences` after every sampled token and finalizes the sequence with `FinishReason::Stop` when any pre-tokenized stop matches the generated-token suffix. The matched token is emitted to the response (OpenAI convention). Composes correctly with `best_of` (P37 — each candidate honors its own stop set), `logprobs`/`top_logprobs` (P36 — matched token's logprob is emitted), `echo`/`suffix` (P35 — orthogonal), `seed` (P34), and `max_tokens` (Length wins if hit first). Streaming path emits `finish_reason: "stop"` on the final chunk. Validator gains a `validate_stop_sequences` helper: rejects > 4 strings (OpenAI upper bound) and empty strings with 400. …

Plus a separate entry for the clippy fix:

> **Clippy `large_enum_variant` lint fix** (v31.0 P38 follow-up) — `EngineMessage::AddRequest.request` changed from `Request` to `Box<Request>` (P36's `SampledToken`-inflation pushed the field past 248 bytes and tripped the lint). The other `EngineMessage` variants are ≤ 64 bytes; boxing brings the enum to pointer-sized. All `AddRequest { request, ... }` construction sites updated to `request: Box::new(request)`. `just ci` is now green on `vllm-core`.

### 7.3 `.planning/STATE.md`

Update `last_activity` with P38 summary.

## 8. Public API Delta

| Item | New? | Where |
|---|---|---|
| `SamplingParams::stop_token_sequences: Option<Vec<Vec<TokenId>>>` | YES | `crates/traits/src/types.rs` |
| `SamplingParamsBuilder::with_stop_token_sequences(Vec<Vec<TokenId>>)` | YES | same file |
| `EngineMessage::AddRequest.request` type | CHANGED (`Request` → `Box<Request>`) | `crates/core/src/types/messages.rs` |
| All other items | unchanged | n/a |

**0 new public types.** **1 new public field + 1 new builder method.** **1 internal field type change** (consumed-by-value at construction; invisible to public consumers).

## 9. Risks & Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Stop tokenization produces unexpected splits (BPE merges across stop boundary) | Low | Use existing `state.tokenizer.encode(...)`; OpenAI's reference impl uses the same pattern |
| Engine checks stop on EVERY sample → per-step overhead | Negligible | `matches_stop_sequences` is O(N × M) where M ≤ 16 (4 stops × ≤ 4 tokens); runs once per step |
| `Box<Request>` adds per-request allocation | Negligible | < 1 µs per request; HTTP layer already allocates `Request`'s inner Vec |
| `stop` × `best_of` (P37) interaction | Medium | Per-candidate `populate_completion_sampling_params` clones the stop set into every candidate; integration test pins this |
| `stop` × `max_tokens` race | Low | `max_tokens` check runs in `scheduler/update.rs` BEFORE the stop check in `step_regular`; whichever fires first wins, deterministic per-step |
| Streaming + `stop` cancel mid-flight | Low | Client disconnect triggers existing `CancelOnDrop` (REL-01) which sends `CancelRequest`; engine cleans up; the in-flight stop token may or may not reach the client depending on timing — acceptable per OpenAI's "best effort" wording |

## 10. Success Criteria

- [ ] `cargo check -p vllm-server -p vllm-core -p vllm-traits --all-features` green
- [ ] `cargo fmt --all --check` green
- [ ] `cargo clippy --all-targets -p vllm-server --all-features -- -D clippy::correctness -D clippy::suspicious -D clippy::perf` green (including on `vllm-core`)
- [ ] `cargo test -p vllm-server --lib --all-features` 100% passing
- [ ] `cargo test -p vllm-server --test chat_integration_test --all-features` 100% passing
- [ ] `cargo test -p vllm-core --all-features` 100% passing (including new `matches_stop_sequences` tests)
- [ ] `cargo doc -p vllm-server -p vllm-core -p vllm-traits --no-deps --all-features` green
- [ ] `just ci` green end-to-end
- [ ] No regression in P37 best_of tests (run `cargo test -p vllm-server --test chat_integration_test --all-features best_of` → all green)
- [ ] `docs/reference/openai-compatibility.md` updated; v32+ table no longer has `stop` row
- [ ] `CHANGELOG.md` has new P38 entries
- [ ] `.planning/STATE.md` `last_activity` reflects P38
- [ ] Public-API surface delta = { 1 field + 1 builder method + 1 internal field-type change } — matches §8

## 11. Implementation Order (rough)

1. Part C first (smaller, mechanical, unblocks `just ci` for subsequent verification)
2. Part A — type layer (`SamplingParams` field + builder)
3. Part A — HTTP validation layer (validator rewrite + helper)
4. Part A — HTTP wire-through (populate_chat/completion_sampling_params)
5. Part A — engine step loop (`matches_stop_sequences` + integration into `step_regular`)
6. Tests (unit → validator → integration)
7. Docs + CHANGELOG + STATE
8. Full `just ci`

Estimated effort: ~1.5–2 working days.

## 12. Cross-References

- P32 (echo + suffix + best_of declaration) — sister work
- P35 (echo + suffix engine wire-through) — orthogonal compose target
- P36 (logprobs + top_logprobs engine wire-through) — orthogonal compose target
- P37 (best_of engine wire-through) — orthogonal compose target; per-candidate stop set
- `.planning/v31.0-MASTER-PLAN.md` §31-F (Performance) — unrelated; `stop` honoring doesn't belong to performance phase
- `.planning/STATE.md` §"Deferred Items (v32+)" — `stop` is removed from this list after P38 ships

---

**End of design. Next step:** writing-plans skill to produce the executable task plan.
