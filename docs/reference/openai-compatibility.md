# OpenAI API Compatibility Matrix

> **Status (2026-07-16, P23):** v0.x alpha. The matrix below lists every
> field on the OpenAI request/response types that vLLM-lite exposes
> over HTTP, and the current implementation status. Anything not
> listed under "Wired" is either "Declared but not honoured" (silently
> dropped — historically a contract violation we now reject with
> 400 — see API-01 follow-up batches) or "Not declared" (rejected by
> serde at deserialisation).
>
> This file is the single source of truth for what works against
> `/v1/chat/completions`, `/v1/completions`, `/v1/models`, and
> `/v1/embeddings`. Update the matrix when a field's status changes;
> the CHANGELOG entry that flips a field must link here. The
> "v0.2 follow-ups (planned)" section below splits the "Not declared"
> rows into v0.2 candidates and v32+ deferrals so the backlog is
> visible from this single document.

## `/v1/chat/completions`

### Request (`ChatRequest`)

| Field | Type | Status | Notes |
|-------|------|--------|-------|
| `model` | `string` (required) | Wired | Empty string → `400 model is required` |
| `messages` | `Vec<ChatMessage>` (required) | Wired | Empty list → `400 messages is required` |
| `temperature` | `Option<f32>` (0.0–2.0) | Wired | Passed through to engine `SamplingParams.temperature` |
| `top_p` | `Option<f32>` (0.0–1.0) | **Wired** | Forwarded to engine `SamplingParams.top_p`; honours nucleus sampling via `vllm_core::sampling::sample_batch_with_params`. `validate_top_p` rejects `top_p <= 0`, `top_p > 1`, and `NaN` with `400 invalid_request_error` at the HTTP boundary (P9 follow-up). |
| `max_tokens` | `Option<i64>` | Wired | Default 100; cap checked against `max_model_len` |
| `stream` | `Option<bool>` | Wired | `true` → SSE; `false`/missing → unary |
| `n` | `Option<i64>` | **Wired (validation)** | `n = 1` accepted (default); `n > 1` → `400 invalid_request_error` ("n > 1 is not supported…") |
| `stop` | `Option<Vec<String>>` | **Wired (validation)** | `None` or empty array accepted; non-empty → `400 invalid_request_error` ("stop sequences are not yet honoured…") |
| `seed` | `Option<i64>` | **Wired (declaration + tracing pass-through)** | Accepted on `ChatRequest` + `CompletionRequest`; per OpenAI spec any `i64` is accepted (no range / sign validation, no NaN check). Threaded into the `tracing::info!(seed = ?req.seed, ...)` log lines in `openai::chat::{non_stream_chat_completion, stream_chat_completion}` so determinism is at least observable in trace logs. **Honoring is a no-op today** — the engine's sampler reads from `rand`'s thread-local RNG which is currently unseeded; same seed + same model + same prompt does NOT yet produce the same output. Engine-side RNG seeding is v32+ work. The wire-type contract is locked in now so the declaration-only PR doesn't regress to "rejected by serde". **Shipped in P23 (2026-07-16).** |
| `frequency_penalty` | `Option<f32>` (`-2.0`–`2.0`) | **Wired (declaration + validation + engine wire-through)** | Accepted on `ChatRequest`. Per OpenAI spec the valid range is `[-2.0, 2.0]`; the validator rejects NaN / ±infinity / out-of-range values with `400 invalid_request_error`. Honoring is end-to-end for non-negative values via the existing `SamplingParams::repeat_penalty` slot (P2 ARCH-02): the chat handler maps `frequency_penalty` to `repeat_penalty = max(1.0, 1.0 + frequency_penalty)` so positive penalties produce the OpenAI-spec "halve logit on each repetition" semantics. **Negative values are clamped to `repeat_penalty = 1.0` (no penalty)** because the current `apply_repeat_penalty` logit-divide math inverts the sign of negative logits when dividing by a value `< 1.0` — surfacing the OpenAI-spec "boost repetition" semantics requires either a new `apply_repeat_boost` helper or a sign-aware refactor of `apply_repeat_penalty` (v32+ work; the in-range negative case silently degrades to "no penalty" rather than producing garbage output). Threaded into the chat handler's `tracing::info!(...)` log lines as `frequency_penalty = ?req.frequency_penalty` so the request's penalty settings are observable in trace logs even when honoring is partially clamped. **Shipped in P27 (2026-07-18, v0.3 wire-type follow-up).** |
| `presence_penalty` | `Option<f32>` (`-2.0`–`2.0`) | **Wired (declaration + validation + engine wire-through)** | Accepted on `ChatRequest`. Per OpenAI spec the valid range is `[-2.0, 2.0]`; the validator rejects NaN / ±infinity / out-of-range values with `400 invalid_request_error`. **Honoring is end-to-end** via the new `vllm_core::sampling::apply_presence_penalty` helper (added by P28): the helper subtracts `presence_penalty` from the logit of every *distinct* seen token regardless of occurrence count — the OpenAI presence-style semantic. Positive values discourage repetition (encourage new topics); negative values *encourage* repetition (because subtracting a negative is the same as adding to the logit). Unlike `frequency_penalty` (which maps to `repeat_penalty` via a clamped `max(1.0, 1.0 + value)` formula to work around the `apply_repeat_penalty` logit-divide sign-flip bug for negative values), `presence_penalty` is an *additive* bias so the value is forwarded verbatim — no clamping needed. Threaded into the chat handler's `tracing::info!(...)` log lines as `presence_penalty = ?req.presence_penalty` for parity with P21/P22/P23/P27 observability plumbing. **Shipped in P28 (2026-07-18, v0.3 wire-type follow-up engine wire-through).** |
| `logit_bias` | (not declared) | **Not declared** | Rejected by serde |
| `logprobs` | (not declared) | **Not declared** | Rejected by serde |
| `top_logprobs` | (not declared) | **Not declared** | Rejected by serde |
| `tools` / `tool_choice` | (not declared) | **Not declared** | Rejected by serde |
| `response_format` | `Option<ResponseFormat>` | **Wired (declaration + validation)** | Accepted on `ChatRequest` (NOT `CompletionRequest` — the legacy `/v1/completions` endpoint does not support this field per OpenAI spec). v0.2 declares the `ResponseFormat` enum with `Text` + `JsonObject` variants only; `json_schema` (the v0.3 constrained-decoding variant) is rejected at the serde layer (axum returns `422 Unprocessable Entity` for unknown enum variants). Honoring is a no-op — the engine does not enforce JSON syntax via a constrained-decoder hook. Pinned by `validate_chat_response_format` (a documentation-first no-op today; the hook for future strict checks). Threaded into `tracing::info!(response_format = ?req.response_format, ...)` log lines for observability. **Shipped in P22 (2026-07-16).** |
| `user` | `Option<String>` | **Wired (tracing pass-through)** | Accepted on `ChatRequest` + `CompletionRequest`; per OpenAI spec there is no format/length validation. Threaded into the `tracing::info!(user = ?req.user, ...)` log lines in `openai::chat::{chat_completions, stream_chat_completion}` so downstream subscribers (rate-limiter, audit log) can pick it up. Honoring is a no-op today — vllm-lite has no auth/persistence layer that consumes it (P21 v0.2 follow-up). |

### Response (`ChatResponse` / `ChatChunk`)

| Field | Status | Notes |
|-------|--------|-------|
| `id` | Wired | Generated server-side (`req_<8-char-uuid>`) |
| `object` | Wired | Hardcoded `"chat.completion"` (non-stream) / `"chat.completion.chunk"` (stream) |
| `created` | Wired | Unix timestamp seconds |
| `model` | Wired | Echoes the request `model` |
| `choices[].index` | Wired | Always `0` (we don't support `n > 1`) |
| `choices[].message.role` | Wired | Always `"assistant"` |
| `choices[].message.content` | Wired | Generated text |
| `choices[].finish_reason` | Wired | `"stop"` (natural EOS or `Cancelled`) / `"length"` (hit `max_tokens`). API-01 P4 fix wired the engine's `FinishReason` oneshot through. |
| `choices[].logprobs` | **Not declared** | Top-level field doesn't exist; cannot be sent |
| `usage.prompt_tokens` | Wired | From tokenizer encode |
| `usage.completion_tokens` | Wired | Counted server-side |
| `usage.total_tokens` | Wired | `prompt_tokens + completion_tokens` |

### Streaming (SSE) deltas

| Field | Status | Notes |
|-------|--------|-------|
| `choices[].delta.role` | Wired | Only on first chunk |
| `choices[].delta.content` | Wired | Per-token |
| `choices[].finish_reason` | Wired | On the final chunk (P4 batch fix — was missing) |
| `[DONE]` sentinel | Wired | Emitted as a **separate** SSE event after the final chunk (P4 batch fix — pre-fix was crammed into the same `data:` field as the last chunk, breaking strict SSE clients) |

## `/v1/completions`

### Request (`CompletionRequest`)

| Field | Type | Status | Notes |
|-------|------|--------|-------|
| `model` | `Option<String>` | Wired | Optional in OpenAI's legacy endpoint |
| `prompt` | `string` (required) | Wired | Raw text (no chat template). Empty → `400 prompt is required` |
| `temperature` | `Option<f32>` | Wired | Same as chat |
| `max_tokens` | `Option<i64>` | Wired | Default 100 |
| `stream` | `Option<bool>` | Wired | SSE or unary |
| `n` | `Option<i64>` | **Wired (validation)** | Same rejection as chat |
| `stop` | `Option<Vec<String>>` | **Wired (validation)** | Same rejection as chat |
| `top_p` | `Option<f32>` (0.0–1.0) | **Wired** | Same honouring + range check as chat (P9 follow-up) |
| `seed` | `Option<i64>` | **Wired (declaration)** | Accepted on `CompletionRequest` (P23 v0.2). Same contract as the chat endpoint — any `i64` is accepted per OpenAI spec, no range / sign validation, no NaN check. The completions handler does not currently log the field (deferred to avoid adding a new `tracing::info!` line — chat handler logs it). Downstream consumers subscribe via direct field access. Honoring is a no-op today — engine-side RNG seeding is v32+ work. |
| `user` | `Option<String>` | **Wired (declaration)** | Accepted on `CompletionRequest` (P21 v0.2). Same contract as the chat endpoint — no format/length validation per OpenAI spec. The completions handler does not currently log the field (deferred to avoid adding a new `tracing::info!` line — chat handler logs it). Downstream consumers subscribe via direct field access. |
| `frequency_penalty` | `Option<f32>` (`-2.0`–`2.0`) | **Wired (declaration + validation + engine wire-through)** | Accepted on `CompletionRequest`. Same contract as the chat endpoint (see the chat table row above): OpenAI-spec range `[-2.0, 2.0]`; validator rejects NaN / ±infinity / out-of-range with `400`; non-negative values are forwarded to the engine via `repeat_penalty = max(1.0, 1.0 + value)` (negative values clamped to `1.0` for the same `apply_repeat_penalty` logit-divide sign-flip reason documented on the chat endpoint). The completions handler does not currently log the field — adding a new `tracing::info!` line is deferred to keep parity with the `seed` / `user` fields (chat handler logs them, completions handler accepts them at the wire type but does not log). **Shipped in P27 (2026-07-18).** |
| `presence_penalty` | `Option<f32>` (`-2.0`–`2.0`) | **Wired (declaration + validation + engine wire-through)** | Accepted on `CompletionRequest`. Same contract as the chat endpoint (see the chat table row above): OpenAI-spec range `[-2.0, 2.0]`; validator rejects NaN / ±infinity / out-of-range with `400`; honoring is end-to-end via the new `apply_presence_penalty` helper (P28) — the value is forwarded verbatim to `SamplingParams::presence_penalty` (no clamping because presence-style penalty is additive). The completions handler does not currently log the field — same parity rationale as `seed` / `user` / `frequency_penalty` above. **Shipped in P28 (2026-07-18).** |
| `logprobs` / `echo` / `suffix` / `best_of` | (not declared) | **Not declared** | Rejected by serde |

### Response (`CompletionResponse`)

| Field | Status | Notes |
|-------|--------|-------|
| `id`, `object`, `created`, `model` | Wired | Same shape as chat |
| `choices[].text` | Wired | Raw continuation |
| `choices[].index` | Wired | Always `0` |
| `choices[].finish_reason` | Wired | Same as chat (P4 fix) |
| `usage` | Wired | Same shape as chat |

## `/v1/models`

| Behaviour | Status | Notes |
|-----------|--------|-------|
| `GET /v1/models` | Wired | Returns the single loaded model's id + metadata |
| `max_model_len` field | Wired | Exposed when the loaded model declares `max_position_embeddings`; absent otherwise (omit with `skip_serializing_if = "Option::is_none"`). Production-readiness §4. |
| `capabilities` field | Wired | Architecture capabilities for the loaded model (production-readiness §10) |

## `/v1/embeddings`

| Behaviour | Status | Notes |
|-----------|--------|-------|
| `POST /v1/embeddings` | Partial | Refuses with `501 Not Implemented` (code `embeddings_unsupported`) when the loaded architecture is a stub or capabilities couldn't be detected. Production-grade embedding requires a non-stub architecture + a real forward path. Production-readiness §10. |

## `/v1/batches`

| Behaviour | Status | Notes |
|-----------|--------|-------|
| `POST /v1/batches` | **501 Not Implemented** | Always returns 501 with code `server_error` + a documentation pointer. Per API-01 P1 fix — there is no background worker to advance `pending → in_progress → completed`. The handler still validates the request shape so callers get a clear distinction between "malformed" and "not implemented". |
| `GET /v1/batches/{id}` | Wired (read-only) | Returns whatever state the job currently has |
| `GET /v1/batches/{id}/results` | Wired (read-only) | Returns empty array for never-completed jobs |

## v0.2 follow-ups (planned)

The "Not declared" rows in the tables above split into two
categories: fields queued for **v0.2** (the next minor) and
fields deferred to **v32+** (the next major). The split
follows OpenAI's field groupings — sampling knobs (cheaper to
add) land in v0.2; structural features (tool calling, logprobs)
require model-side work and defer to v32+.

**v0.2 candidates** (declaration + HTTP-boundary validation;
honoring depends on engine-side work):

| Field | Type | Why v0.2 (and not v32+) |
|-------|------|-------------------------|
| `seed` | `Option<i64>` | OpenAI spec: "best effort to sample deterministically". Declaration is trivial; honoring requires seeding the sampler's RNG (currently unseeded — the sampler reads from `rand`'s thread-local RNG). The validation contract is the same as `top_p`: accept any integer (per OpenAI spec), forward to engine, log the seed so determinism is at least observable in trace logs. v32+ adds RNG seeding in `vllm_core::sampling`. **Shipped in P23 (2026-07-16)** as a declaration-only PR (mirrors P21 `user` + P22 `response_format` pattern); RNG seeding in v32+ will activate the determinism contract. |
| `user` | `Option<String>` | User identifier for safety / abuse tracking. Declaration + pass-through to `tracing::info!(user = ?req.user, ...)` is trivial; vllm-lite has no auth/persistence layer that would consume it. Honoring is a no-op until a downstream consumer (rate-limiter, audit log) subscribes. **Shipped in P21 (2026-07-16).** |
| `response_format` | `Option<ResponseFormat>` | OpenAI's JSON-mode. Declaration + validation (only `{type: "text"}` and `{type: "json_object"}` accepted in v0.2; the JSON schema subset defers to v0.3 because it requires generating-grammar-constrained output) is small. Honoring requires the sampler to enforce `json_object` mode via a constrained-decoding hook — that hook is v32+ work. v0.2 accepts the field and forwards to the engine which currently treats it as a no-op. **Shipped in P22 (2026-07-16).** Note: P22 chose the minimal declaration-only approach — the field is declared on `ChatRequest`, validated via serde (with a `validate_chat_response_format` documentation-first hook), and threaded into `tracing::info!(response_format = ?req.response_format, ...)`. Engine-side forwarding is deferred to v0.3 / v32+ when the constrained-decoding hook lands. |

**v32+ candidates** (deferred — require model-side work that the
technical due diligence flags as out-of-scope for v0.x):

| Field | Why v32+ |
|-------|----------|
| `logit_bias` | Requires a bias-map injection in the sampler's softmax step. Not implemented in the current sampling module. |
| `logprobs` / `top_logprobs` | Requires the engine to return top-K logprobs per output token. Current `sample_batch_with_params` returns only the sampled token. |
| `presence_penalty` honoring | **Shipped in P28 (2026-07-18)** — new `vllm_core::sampling::apply_presence_penalty` helper subtracts the penalty from the logit of every distinct seen token (presence-style semantic). Wired into `sample_one_with_params` after `apply_repeat_penalty` and before `temperature_sample`. Negative values are honored as-is (encourage repetition) because additive subtraction has no sign-flip issue. |
| `frequency_penalty` boost semantics (negative values) | Honoring for non-negative values is shipped via the existing `repeat_penalty` slot (P27, 2026-07-18): the chat handler maps `frequency_penalty` to `repeat_penalty = max(1.0, 1.0 + value)`. Negative values are clamped to `1.0` (no penalty) because the current `apply_repeat_penalty` logit-divide math inverts the sign of negative logits when dividing by a value `< 1.0`. Surfacing the OpenAI-spec "boost repetition" semantics requires either a new `apply_repeat_boost` helper or a sign-aware refactor of `apply_repeat_penalty`; either is mechanical but out of scope for the declaration PR. v32+ work. |
| `tools` / `tool_choice` | Tool-calling framework requires a grammar-constrained decoder (JSON-schema → grammar) and a per-request tool schema cache. Architecture-level work — v32+. |

**Cross-references:**
- The `seed` item was tracked under `.planning/STATE.md` "Remaining
  open items" with the same `v0.2` tag until P23 closed it
  (2026-07-16) — see the `Public-API delta` bullet in the CHANGELOG
  and the STATE.md "v0.2 wire-type follow-ups" section for the
  closing notes.
- The `response_format` JSON-mode subset was closed by P22
  (2026-07-16) — see the `Public-API delta` bullet in the
  CHANGELOG and the STATE.md "v0.2 wire-type follow-ups" section
  for the closing notes.
- The v0.3 `frequency_penalty` + `presence_penalty` declarations
  are shipped in P27 (2026-07-18); P28 (2026-07-18) closed the
  `presence_penalty` honoring gap by adding a presence-aware
  `apply_presence_penalty` helper to the engine. Both fields are
  now honored end-to-end — `frequency_penalty` via the existing
  `repeat_penalty` slot (frequency-style, clamped for negative
  values), `presence_penalty` via the new helper (presence-style,
  no clamp needed). Only the `frequency_penalty` boost-semantics
  v32+ carve-out (negative values) remains as engine work.

## Error contract

| Code | When | HTTP |
|------|------|------|
| `400 invalid_request_error` | Request fails shape / validation (e.g. `n > 1`, `stop` non-empty, `model` empty, `messages` empty, `prompt_tokens + max_tokens > max_model_len`, `beam_width > 1`) | 400 |
| `400 context_length_exceeded` | `prompt_tokens + max_tokens > max_model_len`. OpenAI-compatible code so SDKs can detect and split. | 400 |
| `401 unauthorized` | Missing/invalid `Authorization` Bearer for a protected endpoint when keys are configured | 401 |
| `404 not_found` | Catch-all for unmapped routes / unrecognized batch IDs | 404 |
| `413 payload_too_large` | Request body > 1 MiB (default; configurable via `with_default_body_limit`) | 413 |
| `429 too_many_requests` | Reserved — currently no rate-limit middleware wired | 429 |
| `500 server_error` | Unexpected engine error / handler panic / serialization failure | 500 |
| `501 not_implemented` | `/v1/batches` create; `/v1/embeddings` when capabilities are missing | 501 |
| `503 admin_disabled` | Admin endpoint (e.g. `/debug/*`, `/shutdown`) hit without API keys configured. SEC-01 fail-closed policy. | 503 |
| `503 engine_unavailable` | Engine channel closed at submission time | 503 |
| `503 engine_overloaded` | Engine mailbox full (capacity `engine_mailbox_capacity`, default 256) | 503 |

## Validation module

All HTTP-boundary validations live in
`crates/server/src/openai/sampling_validation.rs`:

- `validate_sampling_params` — rejects `beam_width > 1`
- `validate_chat_request_fields` — rejects `n > 1`, non-empty `stop`
- `validate_completion_request_fields` — same for `/v1/completions`

Each function returns `Result<(), (StatusCode, Json<ErrorResponse>)>`
so handlers can `?`-propagate into axum's error response.

## How to update this matrix

1. Add the field to `crates/server/src/openai/types.rs` if not already declared.
2. Wire it through in the handler (pass to engine, or validate + 400).
3. Add a unit test in `sampling_validation.rs` (or in `chat.rs`/`completions.rs` for handler-level wiring).
4. Add an integration test in `crates/server/tests/chat_integration_test.rs` (or `completions_*`).
5. Update this matrix.
6. Add a CHANGELOG entry under Unreleased > Changed (or Fixed) that links here.
