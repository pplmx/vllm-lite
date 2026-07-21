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
| `stop` | `Option<Vec<String>>` (max 4 strings) | **Wired (declaration + validation + engine wire-through)** | Accepted on `ChatRequest`. Per OpenAI spec: up to 4 strings; each string is a stop sequence that, when generated, terminates the response with `finish_reason = "stop"`. The matched stop text is INCLUDED in the response (OpenAI convention). Default `None`. **Honored end-to-end** — when `stop = Some(seqs)` is non-empty, the chat handler tokenizes each string via `state.tokenizer.encode(s)`, forwards as `SamplingParams::stop_token_sequences = Some(tokenized)`, and the engine's `step_regular` runs `vllm_core::sampling::matches_stop_sequences(generated, stops)` after every sampled token; on match the sequence is finalized with `FinishReason::Stop`. Validated by `validate_stop_sequences` (rejects > 4 strings or empty/whitespace strings with `400 invalid_request_error`). Empty strings and pure-whitespace stops are rejected because they tokenize to zero tokens and would never match (silent no-op). The chat handler does not currently log the field (parity with `seed` / `user` / `frequency_penalty` rationale). **Shipped in P22 (declaration + validation) + P38 (engine wire-through).** |
| `seed` | `Option<i64>` | **Wired (declaration + validation + engine wire-through)** | Accepted on `ChatRequest` + `CompletionRequest`; per OpenAI spec any `i64` is accepted (no range / sign validation, no NaN check). Threaded into the `tracing::info!(seed = ?req.seed, ...)` log lines in `openai::chat::{non_stream_chat_completion, stream_chat_completion}` so determinism is observable in trace logs (P23). **Honoured end-to-end** — `req.seed` (cast to `u64` via `as`, wrapping negatives) is forwarded to `SamplingParams::seed`; the engine's `sample_one_with_params` reads `params.seed` once per call and builds a fresh `rand::rngs::StdRng::seed_from_u64(s)` when `Some(_)`, or falls back to the thread-local default RNG when `None` (P34 v0.2 wire-type follow-up engine wire-through). The OpenAI-spec "same seed + same model + same prompt → same output" contract holds for non-greedy sampling; greedy paths (`temperature = 0` / `top_p = 1.0`) bypass the RNG entirely so seed has no observable effect in those modes — matches OpenAI's "best effort" wording in the spec. `seed = 0` is a valid seed (NOT conflated with `None`). `seed = Some(_)` and `seed = None` are independent per-sequence in `sample_batch_with_params` — sequences with different seeds draw different random thresholds, sequences with the same seed re-seed independently and produce the same draws (correct OpenAI per-request determinism contract). **Shipped in P23 (declaration + tracing) + P34 (engine wire-through).** |
| `frequency_penalty` | `Option<f32>` (`-2.0`–`2.0`) | **Wired (declaration + validation + engine wire-through)** | Accepted on `ChatRequest`. Per OpenAI spec the valid range is `[-2.0, 2.0]`; the validator rejects NaN / ±infinity / out-of-range values with `400 invalid_request_error`. **Honoring is end-to-end for the full OpenAI range** via the sign-aware `apply_repeat_penalty` engine helper (P29 refactor): the chat handler maps `frequency_penalty` to `repeat_penalty = (1.0 + frequency_penalty).max(1e-3)` (the 1e-3 floor prevents divide-by-zero at extreme negative values; mid-range negatives like -0.5 produce a legitimate boost via the sign-aware multiply path). The engine handles positive and negative logits symmetrically — divide for `logit >= 0`, multiply for `logit < 0` — so the penalty direction (penalize vs boost) is consistent regardless of the logit's sign. Positive `frequency_penalty` values produce the OpenAI-spec "halve logit on each repetition" semantics; negative values produce the OpenAI-spec "boost repetition" semantics. Threaded into the chat handler's `tracing::info!(...)` log lines as `frequency_penalty = ?req.frequency_penalty`. **Shipped in P27 (declaration) + P29 (sign-aware engine refactor that closes the boost-semantics carve-out).** |
| `presence_penalty` | `Option<f32>` (`-2.0`–`2.0`) | **Wired (declaration + validation + engine wire-through)** | Accepted on `ChatRequest`. Per OpenAI spec the valid range is `[-2.0, 2.0]`; the validator rejects NaN / ±infinity / out-of-range values with `400 invalid_request_error`. **Honoring is end-to-end** via the new `vllm_core::sampling::apply_presence_penalty` helper (added by P28): the helper subtracts `presence_penalty` from the logit of every *distinct* seen token regardless of occurrence count — the OpenAI presence-style semantic. Positive values discourage repetition (encourage new topics); negative values *encourage* repetition (because subtracting a negative is the same as adding to the logit). Unlike `frequency_penalty` (which maps to `repeat_penalty` via a `max(1e-3, ...)` floor to work around the divisor-formulation divide-by-zero edge case at extreme negatives — but otherwise forwarded verbatim), `presence_penalty` is an *additive* bias so the value is forwarded verbatim with no floor needed. Threaded into the chat handler's `tracing::info!(...)` log lines as `presence_penalty = ?req.presence_penalty` for parity with P21/P22/P23/P27/P29 observability plumbing. **Shipped in P28 (2026-07-18, v0.3 wire-type follow-up engine wire-through).** |
| `logit_bias` | `Option<HashMap<TokenId, f32>>` (`-100`–`100` per value) | **Wired (declaration + validation + engine wire-through)** | Accepted on `ChatRequest`. Per OpenAI spec the valid range is `[-100, 100]` for each bias value; the validator rejects NaN / ±infinity / out-of-range values with `400 invalid_request_error` (error message names the offending token ID). Token IDs are *not* validated — any `TokenId` (which is a `u32`) is accepted, and out-of-vocab IDs are silently ignored at sampling time (matches OpenAI's server behaviour). **Honoring is end-to-end** via the new `vllm_core::sampling::apply_logit_bias` helper (added by P30): the helper adds each map value to the logit at the corresponding token position before the temperature / top-k / top-p pipeline. Positive values *increase* the probability of the biased tokens; negative values *decrease* it — opposite of the `presence_penalty` semantic (which subtracts from seen-token logits). The map iteration order is non-deterministic (HashMap) but the *final logits* are deterministic because each bias is additive and independent per token — so determinism is preserved. Threaded into the chat handler's `tracing::info!(...)` log lines as `logit_bias_len = ?req.logit_bias.as_ref().map(|m| m.len())` (count only, not the full map, to keep log lines bounded for typical maps of up to ~300 entries). **Shipped in P30 (2026-07-18, v0.3 wire-type follow-up engine wire-through).** |
| `logprobs` | `Option<bool>` | **Wired (declaration + validation + engine wire-through)** | Accepted on `ChatRequest`. Per OpenAI chat-completions spec: `logprobs: bool` indicates whether to return the log probability of the sampled token (default `false`). The validator only enforces the cross-field rule with `top_logprobs` — no range check on the bool itself. **Honored end-to-end** via `vllm_core::sampling::sample_one_with_params` (P36 v0.3 wire-type follow-up engine wire-through) — the engine now emits `SampledToken { token, logprob, top_logprobs }` per step and the chat handler renders `choices[0].logprobs.content[]` from that stream (one entry per generated token with `token` + `logprob` + optional `top_logprobs[]` when `top_logprobs > 0`). The chat streaming handler also renders the field per-chunk (each intermediate chunk carries one entry; the finish_reason chunk carries none). The wire-type contract was locked in by P31 (2026-07-18) so the declaration-only PR didn't regress to "rejected by serde" for callers who already send the field. Threaded into the chat handler's `tracing::info!(...)` log lines as `logprobs = ?req.logprobs` for parity with P21/P22/P23/P27/P28/P29/P30 observability plumbing. **Shipped in P31 (declaration + validation); engine wire-through shipped in P36 (2026-07-20).** |
| `top_logprobs` | `Option<u32>` (`0`–`20`) | **Wired (declaration + validation + engine wire-through)** | Accepted on `ChatRequest`. Per OpenAI chat-completions spec: an integer in `[0, 20]` specifying how many of the most likely tokens to return log probabilities for at each position. The validator rejects values > 20 with `400 invalid_request_error`. **Cross-field rule:** `top_logprobs = Some(_)` requires `logprobs = true` (otherwise the field is meaningless — OpenAI only honours it when `logprobs` is enabled); violations return `400 invalid_request_error` with a message naming both fields. **Honored end-to-end** via `vllm_core::sampling::sample_one_with_params`'s new `top_logprobs_of` helper (P36) — when `params.top_logprobs.is_some()`, the sampler runs a partial top-K selection on the post-filter logits and attaches the `(token, logprob)` pairs to `SampledToken::top_logprobs` (sorted by logprob descending, length ≤ `params.top_logprobs`). When `None` or `Some(0)`, no top-K computation runs and `SampledToken::top_logprobs` is empty (default-path overhead stays at zero). The chat handler renders each per-entry `top_logprobs[]` as a nested array inside `choices[0].logprobs.content[i].top_logprobs`. Threaded into the chat handler's `tracing::info!(...)` log lines as `top_logprobs = ?req.top_logprobs`. **Shipped in P31 (declaration + validation); engine wire-through shipped in P36 (2026-07-20).** |
| `tools` | `Option<Vec<Tool>>` | **Wired (declaration + validation)** | Accepted on `ChatRequest` (NOT `CompletionRequest` — legacy endpoint doesn't support tool calling per OpenAI spec). Per OpenAI chat-completions spec: a list of tool definitions the model may invoke. Currently only function-calling tools are supported (the `ToolType` enum has a single `Function` variant); the `function.parameters` field is a JSON Schema stored as `serde_json::Value` to preserve byte-for-byte fidelity (modelling the full JSON-Schema grammar would balloon the type definition without giving vllm-lite any new capability — the engine doesn't process the schema today). Field names + descriptions + parameters are *not* validated against the OpenAI-spec regex (`^[a-zA-Z0-9_-]{1,64}$`) today — the validator is permissive by design until honoring lands. **Honoring is a no-op** today — tool calling requires a grammar-constrained decoder (JSON-schema → grammar) and a per-request tool schema cache. Architecture-level work, tracked as v32+. The wire-type contract is locked in now so the declaration-only PR doesn't regress to "rejected by serde" for callers who already send the field. Validated by `validate_chat_tool_choice` (cross-field rules with `tool_choice`; see that row for the rule set). Threaded into the chat handler's `tracing::info!(...)` log lines as `tools_len = ?req.tools.as_ref().map(|v| v.len())` (count only, not the full array, to keep log lines bounded for typical tool lists of up to ~10 entries). **Shipped in P33 (2026-07-18, v0.x wire-type follow-up declaration + validation; engine wire-through is v32+).** |
| `tool_choice` | `Option<ToolChoice>` | **Wired (declaration + validation)** | Accepted on `ChatRequest` (NOT `CompletionRequest` — legacy endpoint doesn't support it). Modeled as a `#[serde(untagged)]` enum with two variants: `Mode(ToolChoiceMode)` (string modes `"none"` / `"auto"` / `"required"`) and `Specific(ToolChoiceSpecific)` (object form `{"type": "function", "function": {"name": "..."}}`). The `untagged` strategy preserves the wire shape 1:1; the string modes are tried before the object form so a bare `"auto"` string doesn't accidentally deserialize as `Specific { kind: Auto, ... }`. Validated by `validate_chat_tool_choice`: rejects `tool_choice = Some(Required)` when `tools` is `None` / empty (model can't satisfy "must call a tool"); rejects `tool_choice = Some(Specific { function: { name: X } })` when `tools` is `None` / empty OR when no tool in `tools[]` has a matching `name`. Returns `400 invalid_request_error` with field-naming messages. **Honoring is a no-op** today — same rationale as `tools`. The chat handler threads `tool_choice = ?req.tool_choice` into the `tracing::info!(...)` log lines (Debug-printable; the field is small enough to log in full). **Shipped in P33 (2026-07-18, v0.x wire-type follow-up declaration + validation; engine wire-through is v32+).** |
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
| `choices[].logprobs` | **Wired (declaration + engine wire-through)** | The response-side `choices[].logprobs` field is declared on `ChatChoice` (`Option<ChatChoiceLogprobs>`) and rendered from the engine's `SampledToken` stream when the request set `logprobs = true` (P36 v0.3 wire-type follow-up engine wire-through). Shape matches the OpenAI spec 1:1 — `content: Vec<ChatLogprob>` with one entry per generated token, each carrying `token` (decoded string) + `logprob` (`ln(P)` under the actual sampling distribution) + `bytes` (UTF-8 byte representation) + optional `top_logprobs: Vec<ChatLogprob>` (when the request also set `top_logprobs > 0`). When the request did not ask for logprobs, the field is absent from the response JSON (`skip_serializing_if = "Option::is_none"`). The chat streaming handler renders the same shape per-chunk (each intermediate chunk carries one entry; the finish_reason chunk carries none). |
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
| `stop` | `Option<Vec<String>>` (max 4 strings) | **Wired (declaration + validation + engine wire-through)** | Accepted on `CompletionRequest`. Per OpenAI spec: up to 4 strings; each string is a stop sequence that, when generated, terminates the response with `finish_reason = "stop"`. The matched stop text is INCLUDED in the response (OpenAI convention). Default `None`. **Honored end-to-end** — when `stop = Some(seqs)` is non-empty, the completions handler tokenizes each string via `state.tokenizer.encode(s)` in the private `populate_completion_sampling_params` helper (P37), forwards as `SamplingParams::stop_token_sequences = Some(tokenized)`, and the engine's `step_regular` runs `vllm_core::sampling::matches_stop_sequences(generated, stops)` after every sampled token; on match the sequence is finalized with `FinishReason::Stop`. Validated by `validate_stop_sequences` (rejects > 4 strings or empty/whitespace strings with `400 invalid_request_error`). Empty strings and pure-whitespace stops are rejected because they tokenize to zero tokens and would never match (silent no-op). The `best_of` path (P37) inherits the per-candidate stop set automatically — each of the N parallel candidates honors the same stop set independently. The completions handler does not currently log the field (parity with `seed` / `user` / `frequency_penalty` / `presence_penalty` / `logit_bias` / `echo` / `suffix` / `best_of` rationale). **Shipped in P32 (declaration + validation) + P38 (engine wire-through).** |
| `top_p` | `Option<f32>` (0.0–1.0) | **Wired** | Same honouring + range check as chat (P9 follow-up) |
| `seed` | `Option<i64>` | **Wired (declaration + engine wire-through)** | Accepted on `CompletionRequest` (P23 v0.2). Same contract as the chat endpoint — any `i64` is accepted per OpenAI spec, no range / sign validation, no NaN check. The completions handler does not currently log the field (deferred to avoid adding a new `tracing::info!` line — chat handler logs it). Downstream consumers subscribe via direct field access. **Honoured end-to-end** — `req.seed` is cast to `u64` via `as` (wrapping negatives) and forwarded to `SamplingParams::seed`; the engine reads `params.seed` once per `sample_one_with_params` call (P34). See the chat `seed` row above for the full determinism contract + greedy-bypass + per-sequence-independence semantics. |
| `user` | `Option<String>` | **Wired (declaration)** | Accepted on `CompletionRequest` (P21 v0.2). Same contract as the chat endpoint — no format/length validation per OpenAI spec. The completions handler does not currently log the field (deferred to avoid adding a new `tracing::info!` line — chat handler logs it). Downstream consumers subscribe via direct field access. |
| `frequency_penalty` | `Option<f32>` (`-2.0`–`2.0`) | **Wired (declaration + validation + engine wire-through)** | Accepted on `CompletionRequest`. Same contract as the chat endpoint (see the chat table row above): OpenAI-spec range `[-2.0, 2.0]`; validator rejects NaN / ±infinity / out-of-range with `400`; the full OpenAI range is honored end-to-end via the sign-aware `apply_repeat_penalty` engine helper (P29) with the `max(1e-3, 1.0 + value)` floor that prevents divide-by-zero at extreme negatives. The completions handler does not currently log the field — adding a new `tracing::info!` line is deferred to keep parity with the `seed` / `user` fields (chat handler logs them, completions handler accepts them at the wire type but does not log). **Shipped in P27 (declaration) + P29 (sign-aware engine refactor).** |
| `presence_penalty` | `Option<f32>` (`-2.0`–`2.0`) | **Wired (declaration + validation + engine wire-through)** | Accepted on `CompletionRequest`. Same contract as the chat endpoint (see the chat table row above): OpenAI-spec range `[-2.0, 2.0]`; validator rejects NaN / ±infinity / out-of-range with `400`; honoring is end-to-end via the new `apply_presence_penalty` helper (P28) — the value is forwarded verbatim to `SamplingParams::presence_penalty` (no clamping because presence-style penalty is additive). The completions handler does not currently log the field — same parity rationale as `seed` / `user` / `frequency_penalty` above. **Shipped in P28 (2026-07-18).** |
| `logit_bias` | `Option<HashMap<TokenId, f32>>` (`-100`–`100` per value) | **Wired (declaration + validation + engine wire-through)** | Accepted on `CompletionRequest`. Same contract as the chat endpoint (see the chat table row above): OpenAI-spec range `[-100, 100]` per value; validator rejects NaN / ±infinity / out-of-range with `400` (error message names the offending token ID); token IDs are not validated. Honoring is end-to-end via the new `apply_logit_bias` helper (P30) — the map is forwarded verbatim to `SamplingParams::logit_bias`. The completions handler does not currently log the field — same parity rationale as `seed` / `user` / `frequency_penalty` / `presence_penalty` above. **Shipped in P30 (2026-07-18).** |
| `logprobs` | `Option<u32>` (`0`–`5`) | **Wired (declaration + validation + engine wire-through)** | Accepted on `CompletionRequest`. **Type asymmetry vs chat:** per OpenAI legacy-completions spec the field is `int 0..=5` (count of top tokens to return logprobs for), NOT a `bool` like the chat endpoint's `logprobs`. P31 declares both with the correct types per endpoint rather than unifying behind a common representation — the OpenAI spec is intentionally asymmetric. Validator rejects values > 5 with `400 invalid_request_error`. **Honored end-to-end** via `vllm_core::sampling::sample_one_with_params` (P36 v0.3 wire-type follow-up engine wire-through) — when `req.logprobs = Some(n)` the legacy endpoint forwards `n` to `SamplingParams::top_logprobs = Some(n)` and the engine's `top_logprobs_of` helper computes the top-`n` alternatives at each step. The completions handler renders `choices[0].logprobs` (parallel `tokens[]` / `token_logprobs[]` / `top_logprobs[][]` arrays, one entry per generated token) from the engine's `SampledToken` stream. When `req.logprobs = Some(0)`, the container is still emitted (per OpenAI's behavior — clients rely on `choices[0].logprobs` being non-null when the request mentioned `logprobs` at all) but with empty `top_logprobs[]` arrays; `token_logprobs[]` still carries the sampled-token logprob because the engine always populates `SampledToken::logprob`. The completions handler does not currently log the field — same parity rationale as `seed` / `user` / `frequency_penalty` / `presence_penalty` / `logit_bias` above. **Shipped in P31 (declaration + validation); engine wire-through shipped in P36 (2026-07-20).** |
| `echo` | `Option<bool>` | **Wired (declaration + validation + engine wire-through)** | Accepted on `CompletionRequest`. Per OpenAI legacy-completions spec: when `true`, the response echoes the prompt back as a prefix to the generated continuation in the `text` field (instead of just returning the continuation). Default `false`. **Honored end-to-end** — when `echo = Some(true)`, the completions handler prepends the original raw prompt to the generated continuation via the private `apply_completion_meta` helper in `crates/server/src/openai/completions.rs` (single authoritative point shared by the non-streaming and streaming paths). Non-streaming: response `choices[0].text = prompt + completion`. Streaming: the prompt lands on the FIRST non-empty text chunk's `text` field (matches OpenAI's accumulator semantics — clients concatenate chunk `text` fields in order, so prefixing the first chunk puts the prompt at the start). `echo = Some(false)` and `echo = None` produce the pre-P35 behaviour (no prefix) — legacy clients are unaffected. Validated by `validate_completion_meta` (cross-field rule: `echo = true` cannot coexist with `best_of > 1` per OpenAI spec — see `best_of` row for rationale). The completions handler does not currently log the field (parity with the `seed` / `user` / `frequency_penalty` / `presence_penalty` / `logit_bias` rationale). **Shipped in P32 (declaration + validation) + P35 (engine wire-through).** |
| `suffix` | `Option<String>` | **Wired (declaration + validation + engine wire-through)** | Accepted on `CompletionRequest`. Per OpenAI legacy-completions spec: a string that comes after the inserted completion. Useful for code-completion UIs that pre-fill the suffix (e.g. the closing `}` of a function body) and want the model to fill only the gap. Default `None`. **Honored end-to-end** — when `suffix = Some(_)`, the completions handler appends the suffix to the generated continuation via the same private `apply_completion_meta` helper used by `echo`. Non-streaming: response `choices[0].text = completion + suffix` (or `prompt + completion + suffix` when both flags are set). Streaming: the suffix lands on the FINAL chunk's `text` field (the chunk that carries `finish_reason`) — matches OpenAI's accumulator semantics so suffixing the final chunk puts the suffix at the end of the visible response. `suffix = None` produces the pre-P35 behaviour (no postfix). Validated by `validate_completion_meta` (no range / length check per OpenAI spec — any string is accepted). The completions handler does not currently log the field (parity with the rationale above). **Shipped in P32 (declaration + validation) + P35 (engine wire-through).** |
| `best_of` | `Option<u32>` (`1..=20`) | **Wired (declaration + validation + engine wire-through)** | Accepted on `CompletionRequest`. Per OpenAI legacy-completions spec: an integer `>= 1` specifying how many completions to generate server-side, returning the "best" one (highest mean log probability over the generated tokens). Default `1`. **Honored end-to-end** — when `best_of > 1`, the completions handler dispatches to the private `run_best_of` helper in `crates/server/src/openai/completions.rs` which spawns N parallel `EngineMessage::AddRequest` messages (one per candidate), collects N `Vec<SampledToken>` streams, ranks them via the private `rank_by_mean_logprob` helper (arithmetic mean of per-token `SampledToken::logprob`; tie-breaks on the lowest seq_id / slice index — deterministic, no RNG dependency), and returns the single best completion as a JSON `choices[]` (matches OpenAI's contract: `best_of` returns ONE completion, not N). The other N-1 candidates' tokens / logprobs / text are discarded after ranking. The chosen completion's text then flows through the existing `apply_completion_meta` (P35) helper for `echo` / `suffix` and through the existing `build_completion_choice_logprobs` (P36) helper when `logprobs = Some(n)` was requested — so `best_of` composes correctly with the prior wire-throughs. **Upper bound (P37):** `best_of > 20` is rejected with `400 invalid_request_error` by `validate_completion_meta` (matches OpenAI's documented contract; protects the server from N×max_tokens inference cost explosion). **Cross-field rules:** `best_of > 1` cannot coexist with `echo = true` (P32 invariant, unchanged by P37) and `best_of > 1` cannot coexist with `n > 1` (the existing validator already rejects `n > 1` so the rule is automatically upheld). **Streaming interaction (P37):** `stream = true && best_of > 1` is silently accepted and the response is forced to non-streaming JSON (matches OpenAI's current API behavior — the combination is technically accepted but the runtime shape is a single document, not an SSE event stream, because `best_of` requires ranking N candidates before emitting anything). **Per-candidate sampling:** each candidate uses the exact same prompt + sampling config the user submitted — the `populate_completion_sampling_params` helper (extracted in P37 from the inline forwarding) is the single authoritative point for the OpenAI → `SamplingParams` mapping and is called once per candidate so all N receive identical sampling params. **Honoring is a no-op** when `best_of = 1` or `best_of = None` (default-path overhead stays at zero — the single-shot handler short-circuits to the existing P35/P36 path). The completions handler does not currently log the field (parity with `seed` / `user` / `frequency_penalty` / `presence_penalty` / `logit_bias` / `echo` / `suffix`). **Shipped in P32 (2026-07-18, v0.x wire-type follow-up declaration + validation) + P37 (2026-07-20, v0.x wire-type follow-up engine wire-through).** |

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
| `seed` | `Option<i64>` | OpenAI spec: "best effort to sample deterministically". Declaration is trivial; honoring requires seeding the sampler's RNG (currently unseeded — the sampler reads from `rand`'s thread-local RNG). The validation contract is the same as `top_p`: accept any integer (per OpenAI spec), forward to engine, log the seed so determinism is at least observable in trace logs. v32+ adds RNG seeding in `vllm_core::sampling`. **Shipped in P23 (2026-07-16)** as a declaration-only PR (mirrors P21 `user` + P22 `response_format` pattern); **engine wire-through shipped in P34 (2026-07-19)** — the v0.2 wire-type backlog is now fully closed at the engine-honoring layer for the `seed` field; same seed + same model + same prompt now produces the same output for non-greedy sampling. |
| `user` | `Option<String>` | User identifier for safety / abuse tracking. Declaration + pass-through to `tracing::info!(user = ?req.user, ...)` is trivial; vllm-lite has no auth/persistence layer that would consume it. Honoring is a no-op until a downstream consumer (rate-limiter, audit log) subscribes. **Shipped in P21 (2026-07-16).** |
| `response_format` | `Option<ResponseFormat>` | OpenAI's JSON-mode. Declaration + validation (only `{type: "text"}` and `{type: "json_object"}` accepted in v0.2; the JSON schema subset defers to v0.3 because it requires generating-grammar-constrained output) is small. Honoring requires the sampler to enforce `json_object` mode via a constrained-decoding hook — that hook is v32+ work. v0.2 accepts the field and forwards to the engine which currently treats it as a no-op. **Shipped in P22 (2026-07-16).** Note: P22 chose the minimal declaration-only approach — the field is declared on `ChatRequest`, validated via serde (with a `validate_chat_response_format` documentation-first hook), and threaded into `tracing::info!(response_format = ?req.response_format, ...)`. Engine-side forwarding is deferred to v0.3 / v32+ when the constrained-decoding hook lands. |

**v32+ candidates** (deferred — require model-side work that the
technical due diligence flags as out-of-scope for v0.x):

| Field | Why v32+ |
|-------|----------|
| `logit_bias` | **Shipped in P30 (2026-07-18)** — new `vllm_core::sampling::apply_logit_bias` helper adds each map value to the logit at the corresponding token ID before the temperature / top-k / top-p pipeline (additive bias map, deterministic despite non-deterministic HashMap iteration order). Wired into `sample_one_with_params` after `apply_presence_penalty` and before `temperature_sample`. Out-of-vocab token IDs are silently ignored (matches OpenAI's server behaviour); the validator rejects NaN / ±infinity / out-of-range `[-100, 100]` values with `400 invalid_request_error` up front so the engine never sees bad data. |
| `logprobs` / `top_logprobs` | **Fully shipped in P36 (2026-07-20, v0.3 wire-type follow-up engine wire-through)** — both endpoints now honor the request end-to-end. `ChatRequest::logprobs` + `ChatRequest::top_logprobs` are forwarded to `SamplingParams::top_logprobs`; `CompletionRequest::logprobs` is forwarded to the same slot. The engine's `sample_one_with_params` now returns `SampledToken { token, logprob, top_logprobs }` (a wire-breaking signature change that touches 28 files across the workspace; the public type is `vllm_traits::SampledToken`). The HTTP layer renders the OpenAI-spec `choices[].logprobs` shape on the response side (chat: `content[]` with per-token entries; completions: parallel `tokens[]` / `token_logprobs[]` / `top_logprobs[][]` arrays). Streaming chat renders the field per-chunk (each intermediate chunk carries one entry; the finish_reason chunk carries none). Declaration + validation (P31) was the prerequisite; P36 closes the engine wire-through. The logprob-generation primitive that P36 shipped was the prerequisite for P37's `best_of` engine wire-through (which needs `SampledToken::logprob` to rank candidates by mean log probability). |
| `presence_penalty` honoring | **Shipped in P28 (2026-07-18)** — new `vllm_core::sampling::apply_presence_penalty` helper subtracts the penalty from the logit of every distinct seen token (presence-style semantic). Wired into `sample_one_with_params` after `apply_repeat_penalty` and before `temperature_sample`. Negative values are honored as-is (encourage repetition) because additive subtraction has no sign-flip issue. |
| `frequency_penalty` boost semantics (negative values) | **Shipped in P29 (2026-07-18)** — the `apply_repeat_penalty` helper was refactored to be sign-aware (positive logit → divide by `repeat_penalty`, negative logit → multiply by `repeat_penalty`). The chat / completions handlers now forward negative values verbatim modulo a 1e-3 floor (prevents divide-by-zero for `frequency_penalty ≤ -1.0`); mid-range negatives like -0.5 produce a legitimate boost. |
| `tools` / `tool_choice` (engine honoring) | **Declaration-only shipped in P33 (2026-07-18)** — `ChatRequest::tools` (`Option<Vec<Tool>>`) + `ChatRequest::tool_choice` (`Option<ToolChoice>`, a `#[serde(untagged)]` enum with `Mode` and `Specific` variants) are both declared + validated end-to-end through the HTTP boundary. The validator rejects `tool_choice = Some(Required)` when `tools` is `None` / empty, and `tool_choice = Some(Specific { function: { name: X } })` when no tool in `tools[]` matches `X`. **Engine honoring remains v32+ work** — tool calling requires a grammar-constrained decoder (JSON-schema → grammar at sample time, not a separate post-processing step) and a per-request tool schema cache (the JSON Schema in `tools[].function.parameters` is per-request and can't be globally memoized). Architecture-level work — v32+. |
| `echo` / `suffix` / `best_of` (legacy `/v1/completions`) | **All three shipped end-to-end** — `CompletionRequest::echo` (`Option<bool>`) + `CompletionRequest::suffix` (`Option<String>`, any string per OpenAI spec) + `CompletionRequest::best_of` (`Option<u32>`, range `1..=20`) are all declared + validated + honored end-to-end through the HTTP boundary. `echo` + `suffix` engine honoring closed by P35 via the private `apply_completion_meta` helper in `crates/server/src/openai/completions.rs` — non-streaming prepends prompt + appends suffix directly on `choices[0].text`; streaming puts the prompt on the first non-empty text chunk and the suffix on the finish_reason chunk (matches OpenAI's accumulator semantics). `best_of` engine honoring closed by P37 via the private `run_best_of` + `rank_by_mean_logprob` + `populate_completion_sampling_params` helpers in the same file — N parallel `EngineMessage::AddRequest` candidates are spawned via `tokio::spawn`, joined, ranked by mean logprob (ties broken by lowest seq_id), and the chosen candidate's text flows through `apply_completion_meta` (P35) for `echo` / `suffix` and through `build_completion_choice_logprobs` (P36) when `logprobs` was requested. `stream = true && best_of > 1` is silently accepted and the response is forced to non-streaming JSON. **This closes the v0.x wire-type backlog entirely** — only `tools` / `tool_choice` (grammar-constrained decoder, P33 declared only) remains as v32+ work on the legacy + chat endpoints. |

**Cross-references:**
- The `seed` item was tracked under `.planning/STATE.md` "Remaining
  open items" with the same `v0.2` tag until P23 closed it
  (2026-07-16) — see the `Public-API delta` bullet in the CHANGELOG
  and the STATE.md "v0.2 wire-type follow-ups" section for the
  closing notes. **P34 (2026-07-19) closed the engine-honoring
  layer** — `req.seed` (cast to `u64`) is forwarded to
  `SamplingParams::seed`; the engine's `sample_one_with_params`
  builds a fresh `StdRng::seed_from_u64` per call when `Some(_)`.
  Same seed + same model + same prompt now produces the same
  output for non-greedy sampling (greedy paths bypass the RNG).
  The v0.2 wire-type backlog is now fully closed at the
  engine-honoring layer for `seed`.
- The `response_format` JSON-mode subset was closed by P22
  (2026-07-16) — see the `Public-API delta` bullet in the
  CHANGELOG and the STATE.md "v0.2 wire-type follow-ups" section
  for the closing notes.
- The v0.3 `frequency_penalty` + `presence_penalty` declarations
  are shipped in P27 (2026-07-18); P28 (2026-07-18) closed the
  `presence_penalty` honoring gap by adding a presence-aware
  `apply_presence_penalty` helper to the engine; P29 (2026-07-18)
  closed the `frequency_penalty` boost-semantics carve-out by
  refactoring `apply_repeat_penalty` to be sign-aware so negative
  values produce the OpenAI-spec "boost repetition" semantic.
  Both fields are now honored end-to-end across the full
  `[-2.0, 2.0]` OpenAI range.
- The v0.3 `logit_bias` declaration + engine wire-through is
  shipped in P30 (2026-07-18) — new `apply_logit_bias` helper adds
  each map value to the logit at the corresponding token ID (the
  OpenAI-spec additive bias semantic); the validator rejects NaN /
  ±infinity / out-of-range `[-100, 100]` values with `400` up
  front; token IDs are not validated (out-of-vocab IDs are silently
  ignored).
- The v0.3 `logprobs` + `top_logprobs` declaration + validation
  is shipped in P31 (2026-07-18); the engine wire-through is
  shipped in P36 (2026-07-20). `ChatRequest::logprobs`
  (`Option<bool>`) + `ChatRequest::top_logprobs` (`Option<u32>`,
  range `[0, 20]` with the cross-field rule that
  `top_logprobs = Some` requires `logprobs = true`) and
  `CompletionRequest::logprobs` (`Option<u32>`, range `[0, 5]`)
  are all declared and validated end-to-end through the HTTP
  boundary (out-of-range values and cross-field violations return
  `400 invalid_request_error`). The chat handler forwards both
  `logprobs` + `top_logprobs` to `SamplingParams::top_logprobs`;
  the completions handler forwards `logprobs` to the same slot.
  The engine's `sample_one_with_params` now returns
  `SampledToken { token, logprob, top_logprobs }` — a wire-breaking
  signature change that touches 28 files across the workspace
  (the public type is `vllm_traits::SampledToken`, exported from
  `vllm_traits::types`). The HTTP layer renders the OpenAI-spec
  `choices[].logprobs` shape on the response side (chat:
  `content[]` with per-token entries; completions: parallel
  `tokens[]` / `token_logprobs[]` / `top_logprobs[][]` arrays).
  Streaming chat renders the field per-chunk (each intermediate
  chunk carries one entry; the finish_reason chunk carries none).
  This closes the v0.3 wire-type backlog's last carve-out — only
  `tools` / `tool_choice` (grammar-constrained decoder) remains.
- The `/v1/chat/completions` `tools` + `tool_choice` declaration
  + validation is shipped in P33 (2026-07-18) — `ChatRequest::tools`
  (`Option<Vec<Tool>>`) + `ChatRequest::tool_choice`
  (`Option<ToolChoice>`, a `#[serde(untagged)]` enum with
  `Mode(ToolChoiceMode)` and `Specific(ToolChoiceSpecific)`
  variants) are both declared and validated end-to-end through the
  HTTP boundary (cross-field rules reject `tool_choice = Some(Required)`
  with empty tools, and `tool_choice = Some(Specific { ... })` when
  no tool matches the named function — both return `400
  invalid_request_error`). The engine wire-through is v32+ work —
  tool calling requires a grammar-constrained decoder
  (JSON-schema → grammar at sample time) and a per-request tool
  schema cache (the JSON Schema in `tools[].function.parameters`
  is per-request and can't be globally memoized). With P33, the
  v0.x wire-type backlog is fully closed at the declaration +
  validation layer (declaration: P21/P22/P23/P27/P31/P32/P33;
  engine wire-through: P28/P29/P30 — all the v0.3 carve-outs
  closed except the grammar-constrained decoder for `tools`).
- The legacy `/v1/completions` `echo` + `suffix` + `best_of`
  declaration + validation is shipped in P32 (2026-07-18) —
  `CompletionRequest::echo` (`Option<bool>`) +
  `CompletionRequest::suffix` (`Option<String>`, any string per
  OpenAI spec) + `CompletionRequest::best_of` (`Option<u32>`,
  range `1..=20` after the P37 upper-bound addition) are all
  declared and validated end-to-end through the HTTP boundary
  (`best_of = 0`, `best_of > 20`, and the cross-field rule
  violation `echo = true && best_of > 1` all return `400
  invalid_request_error`). **Engine wire-through closed by P35
  (`echo` + `suffix`, via `apply_completion_meta`) and P37
  (`best_of`, via `run_best_of` + `rank_by_mean_logprob` + N-way
  parallel `EngineMessage::AddRequest`).** The v0.x wire-type
  backlog is now fully closed at both the declaration + validation
  layer AND the engine-honoring layer — only `tools` /
  `tool_choice` (grammar-constrained decoder, P33 declared only)
  remains as v32+ work on the chat endpoint.
- The `/v1/chat/completions` `tools` + `tool_choice` engine
  honoring is the only remaining v32+ candidate at the v0.3
  wire-type layer (grammar-constrained decoder + per-request tool
  schema cache). With P33 closing the declaration layer, the
  v0.3 wire-type backlog is now fully closed at the wire-type
  layer (declaration: P21/P22/P23/P27/P31/P32/P33; engine
  wire-through: P28/P29/P30 — all the v0.3 carve-outs closed
  except the grammar-constrained decoder for `tools`).

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
