# OpenAI API Compatibility Matrix

> **Status (2026-07-15):** v0.x alpha. The matrix below lists every
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
> the CHANGELOG entry that flips a field must link here.

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
| `seed` | (not declared) | **Not declared** | Rejected by serde |
| `frequency_penalty` | (not declared) | **Not declared** | Rejected by serde |
| `presence_penalty` | (not declared) | **Not declared** | Rejected by serde |
| `logit_bias` | (not declared) | **Not declared** | Rejected by serde |
| `logprobs` | (not declared) | **Not declared** | Rejected by serde |
| `top_logprobs` | (not declared) | **Not declared** | Rejected by serde |
| `tools` / `tool_choice` | (not declared) | **Not declared** | Rejected by serde |
| `response_format` | (not declared) | **Not declared** | Rejected by serde |
| `user` | (not declared) | **Not declared** | Rejected by serde |

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
| `seed` | (not declared) | **Not declared** | Rejected by serde |
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
