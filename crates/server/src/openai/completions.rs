//! `OpenAI` legacy Completions endpoint: `POST /v1/completions`. Prompt-string in, completion string out.
use axum::{
    Extension, Json,
    extract::State,
    response::{
        IntoResponse,
        sse::{Event, Sse},
    },
};
use futures::stream;
use std::convert::Infallible;
use tokio::sync::mpsc;

use super::sampling_validation::{validate_completion_request_fields, validate_sampling_params};
use super::types::{
    CompletionChoice, CompletionChoiceLogprobs, CompletionLogprob, CompletionRequest,
    CompletionResponse, ErrorResponse, Usage,
};
use crate::ApiState;
use crate::security::correlation::CorrelationId;

fn should_skip_token_text(tokenizer: &vllm_model::tokenizer::Tokenizer, text: &str) -> bool {
    text.is_empty() || tokenizer.is_special_token(text)
}

fn clean_completion_text(tokenizer: &vllm_model::tokenizer::Tokenizer, text: &str) -> String {
    tokenizer.clean_special_tokens(text)
}

/// Extract the bare [`TokenId`] sequence from a `Vec<SampledToken>`.
/// Used when passing per-token data to the tokenizer (which only
/// understands `&[u32]`); the `logprob` + `top_logprobs` fields are
/// preserved separately for the `CompletionChoice::logprobs`
/// rendering (P36 v0.3 wire-type follow-up engine wire-through).
fn token_ids(sampled: &[vllm_traits::SampledToken]) -> Vec<vllm_traits::TokenId> {
    sampled.iter().map(|s| s.token).collect()
}

/// Build the `CompletionChoiceLogprobs` payload (P36 v0.3 wire-type
/// follow-up engine wire-through) from the engine's per-token
/// `SampledToken` stream. Returns `None` when the request did not
/// ask for logprobs (`req.logprobs.is_none()`).
///
/// When `req.logprobs = Some(0)`, returns `Some` with all-empty
/// parallel arrays — OpenAI's spec keeps the container present (so
/// clients can rely on `choices[0].logprobs` being non-null when
/// the request mentioned `logprobs` at all) but with no top-K
/// alternatives. The sampled-token logprobs are still populated via
/// `token_logprobs` when `req.logprobs = Some(0)` because the
/// engine's `sample_one_with_params` always populates `SampledToken::logprob`.
fn build_completion_choice_logprobs(
    tokenizer: &vllm_model::tokenizer::Tokenizer,
    sampled: &[vllm_traits::SampledToken],
    req_logprobs: Option<u32>,
) -> Option<CompletionChoiceLogprobs> {
    let top_n = req_logprobs?;
    let tokens: Vec<String> = sampled
        .iter()
        .map(|s| tokenizer.decode(&[s.token]))
        .collect();
    let token_logprobs: Vec<f32> = sampled.iter().map(|s| s.logprob).collect();
    let top_logprobs: Vec<Vec<CompletionLogprob>> = sampled
        .iter()
        .map(|s| {
            if top_n == 0 || s.top_logprobs.is_empty() {
                Vec::new()
            } else {
                s.top_logprobs
                    .iter()
                    .take(top_n as usize)
                    .map(|&(tok, lp)| {
                        let text = tokenizer.decode(&[tok]);
                        CompletionLogprob {
                            token: text.clone(),
                            logprob: lp,
                            bytes: Some(text.into_bytes()),
                        }
                    })
                    .collect()
            }
        })
        .collect();
    Some(CompletionChoiceLogprobs {
        tokens,
        token_logprobs,
        top_logprobs,
    })
}

/// Rank N `best_of` candidates by mean logprob (P37 v0.x wire-type
/// follow-up — engine wire-through helper for `best_of`).
///
/// Returns the index of the candidate with the highest mean logprob
/// across its generated tokens. The mean is the simple arithmetic
/// mean of per-token `SampledToken::logprob` values — matches
/// OpenAI's "mean log probability" wording (length-normalized by
/// design; OpenAI does not specify a length-penalty variant here).
///
/// **Tie-breaking:** when two candidates have equal mean logprob (to
/// within `f32::EPSILON`), the one with the lower `seq_id` wins.
/// `seq_id` is monotonically assigned by the scheduler in the order
/// the engine admits the request, so the tie-break is deterministic
/// across runs (no RNG dependency) — important for snapshot /
/// regression testing.
///
/// **Defensive defaults:**
/// - Empty input → returns `0` (no candidates to rank; caller should
///   not reach this in practice because `best_of >= 1` is enforced
///   by `validate_completion_meta`).
/// - Candidate with zero generated tokens → mean logprob is `0.0`
///   (sum is `0`, divide by `1` per OpenAI convention — empty
///   completions are extremely rare but can occur if the engine
///   emits a `finish_reason = length` before any sampled token
///   reaches the HTTP layer).
///
/// Each candidate's `seq_id` is the index of its `Vec<SampledToken>`
/// in the input slice — we use the slice index as the seq_id proxy
/// because the candidates are admitted in order and we don't need
/// the full SeqId for tie-breaking (any monotonically-increasing
/// deterministic value works).
fn rank_by_mean_logprob(candidates: &[Vec<vllm_traits::SampledToken>]) -> usize {
    if candidates.is_empty() {
        return 0;
    }
    candidates
        .iter()
        .enumerate()
        .map(|(i, sampled)| {
            let mean = if sampled.is_empty() {
                0.0
            } else {
                let sum: f32 = sampled.iter().map(|s| s.logprob).sum();
                sum / sampled.len() as f32
            };
            (i, mean)
        })
        // Highest mean logprob wins; ties broken by lowest seq_id
        // (slice index, which is monotonically assigned per the
        // engine's admission order).
        .max_by(|(i_a, mean_a), (i_b, mean_b)| {
            mean_a
                .partial_cmp(mean_b)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| i_b.cmp(i_a)) // lower i wins on tie
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Forward the legacy-completions request's sampling fields onto a
/// fresh `vllm_core::types::Request` (P27/P28/P29/P30/P34/P36
/// sampling-params wire-through; P37 reuses this for the
/// `best_of` per-candidate requests so each candidate gets the
/// exact same sampling config the user submitted).
///
/// The function is the single authoritative point for the
/// legacy-endpoint → `SamplingParams` mapping; both the streaming
/// and non-streaming paths in `completions()` call it, and the
/// `best_of` path calls it N times (once per candidate). Keeping
/// the mapping in one place means the contract stays in sync
/// across all three paths.
///
/// Field-level rationale (see the chat handler for the long-form
/// comments on each mapping):
/// - `temperature`: forwarded verbatim.
/// - `top_p`: forwarded verbatim (validator already range-checked).
/// - `frequency_penalty`: maps to `repeat_penalty = (1.0 + fp).max(1e-3)`
///   per P29's sign-aware engine refactor.
/// - `presence_penalty`: forwarded verbatim (additive bias).
/// - `logit_bias`: cloned verbatim (validator already bounds each value).
/// - `seed`: `i64 as u64` cast (wraps negatives per OpenAI's i64 contract).
/// - `logprobs`: forwarded verbatim to `top_logprobs` (the legacy
///   endpoint's `logprobs` is `u32 0..=5`; the engine treats
///   `Some(0)` as "compute sampled-token logprob only, no top-K").
fn populate_completion_sampling_params(
    request: &mut vllm_core::types::Request,
    req: &CompletionRequest,
) {
    if let Some(temp) = req.temperature {
        request.sampling_params.temperature = temp;
    }
    if let Some(top_p) = req.top_p {
        request.sampling_params.top_p = top_p;
    }
    if let Some(fp) = req.frequency_penalty {
        request.sampling_params.repeat_penalty = (1.0 + fp).max(1e-3);
    }
    if let Some(pp) = req.presence_penalty {
        request.sampling_params.presence_penalty = pp;
    }
    if let Some(ref lb) = req.logit_bias {
        request.sampling_params.logit_bias = Some(lb.clone());
    }
    if let Some(seed) = req.seed {
        request.sampling_params.seed = Some(seed as u64);
    }
    request.sampling_params.top_logprobs = req.logprobs;
}

/// Spawn one of N parallel candidates for a `best_of` request
/// (P37 v0.x wire-type follow-up — engine wire-through helper for
/// `best_of`). Each candidate is an independent `EngineMessage::AddRequest`
/// using the exact same prompt + sampling params the user submitted;
/// the caller (`run_best_of`) ranks the N streams by mean logprob and
/// returns the single best completion.
///
/// Returns the candidate's full token stream alongside its
/// `FinishReason` so the caller can render the chosen completion's
/// `finish_reason` in the response (matches OpenAI's contract:
/// the response's `finish_reason` is the chosen candidate's, not
/// the request's aggregate).
///
/// **Concurrency note:** the candidate is spawned via `tokio::spawn`
/// so all N candidates run in parallel. Each carries its own clone
/// of `ApiState` (cheap — all fields are `Arc`-wrapped) and its own
/// clone of the prompt tokens. The engine's bounded mailbox
/// (`engine_tx.try_send`) ensures a saturated engine surfaces as
/// `503 engine_overloaded` rather than blocking the HTTP handler.
///
/// **Cancellation note:** each candidate is non-streaming-only (best_of
/// never streams — see the validator's silence on `stream + best_of > 1`),
/// so we don't need a `seq_id` round-trip — there is no client
/// disconnect to propagate mid-flight. The candidate runs to natural
/// completion (engine closes `response_tx` after the sequence
/// finishes or hits `max_tokens`).
async fn spawn_best_of_candidate(
    state: ApiState,
    req: CompletionRequest,
    prompt_tokens: Vec<vllm_traits::TokenId>,
    max_tokens: usize,
    correlation_id: String,
) -> Result<
    (Vec<vllm_traits::SampledToken>, vllm_traits::FinishReason),
    (axum::http::StatusCode, Json<ErrorResponse>),
> {
    let total_max = prompt_tokens.len() + max_tokens;
    let mut request = vllm_core::types::Request::new(0, prompt_tokens, total_max);

    populate_completion_sampling_params(&mut request, &req);
    validate_sampling_params(&request.sampling_params)?;

    let (response_tx, mut response_rx) = mpsc::channel(64);
    let (finish_reason_tx, finish_reason_rx) = tokio::sync::oneshot::channel();

    // REL-01: try_send surfaces saturation as 503 instead of blocking.
    state
        .engine_tx
        .try_send(vllm_core::types::EngineMessage::AddRequest {
            request,
            response_tx,
            seq_id_tx: None, // non-streaming — no client disconnect to propagate
            finish_reason_tx: Some(finish_reason_tx),
            request_id: Some(correlation_id),
        })
        .map_err(|e| match e {
            tokio::sync::mpsc::error::TrySendError::Full(_) => overload_response(),
            tokio::sync::mpsc::error::TrySendError::Closed(_) => unavailable_response(),
        })?;

    // Collect the candidate's full token stream.
    let mut tokens = Vec::new();
    while let Some(sampled) = response_rx.recv().await {
        tokens.push(sampled);
    }

    // Engine sends the reason before closing the response channel,
    // so this resolves immediately in the normal case. Fall back to
    // `Stop` only when the oneshot was dropped without a value
    // (e.g. engine panicked between the two steps).
    let finish_reason = finish_reason_rx
        .await
        .unwrap_or(vllm_traits::FinishReason::Stop);
    Ok((tokens, finish_reason))
}

/// Run the `best_of > 1` path (P37 v0.x wire-type follow-up —
/// engine wire-through helper for `best_of`). Spawns N parallel
/// candidates via [`spawn_best_of_candidate`], joins them, ranks
/// by mean logprob via [`rank_by_mean_logprob`], and returns the
/// single best completion in a JSON response (NOT SSE — `best_of`
/// is non-streaming-only, matching OpenAI's contract).
///
/// **Streaming + best_of:** the caller (`completions()`) detects
/// `stream = true && best_of > 1` and silently dispatches here
/// without raising a 400. The OpenAI spec is intentionally
/// permissive on this combination; the runtime behavior is what
/// changes (the response shape becomes a single JSON document
/// instead of an SSE event stream).
///
/// **Partial-failure semantics:** if any of the N candidates fails
/// (engine error / overload / panic), we return the FIRST error
/// observed and discard the other candidates' results. We do NOT
/// issue `EngineMessage::CancelRequest` for the still-running
/// candidates — they will run to natural completion and free their
/// scheduler slots on their own. The cost of this simplification
/// is bounded (each candidate runs at most `max_tokens` steps), and
/// the alternative — per-candidate seq_id tracking + cancel — would
/// add significant complexity for a corner case that should be rare
/// in practice (a `best_of = 5` request where 4 of 5 candidates
/// succeed is overwhelmingly the common path).
///
/// **Logprob interaction:** when `req.logprobs = Some(n)`, the
/// **chosen** completion's per-token logprobs are rendered via
/// [`build_completion_choice_logprobs`] (P36 helper). The other
/// N-1 candidates' logprobs are discarded after ranking.
///
/// **Suffix interaction:** when `req.suffix = Some(_)`, the
/// **chosen** completion's text has the suffix appended via
/// [`apply_completion_meta`] (P35 helper).
async fn run_best_of(
    state: ApiState,
    req: CompletionRequest,
    prompt_tokens: Vec<vllm_traits::TokenId>,
    prompt: String,
    prompt_tokens_len: usize,
    max_tokens: usize,
    correlation_id: CorrelationId,
) -> Result<axum::response::Response, (axum::http::StatusCode, Json<ErrorResponse>)> {
    let n = req.best_of.unwrap_or(1) as usize;

    // Spawn N candidates. Each task is independent and runs in
    // parallel; we send all N `EngineMessage::AddRequest` messages
    // to the engine mailbox in quick succession so the engine sees
    // them as one batch (when N is small enough to fit the bounded
    // mailbox capacity — REL-01 saturation surfaces as the first
    // candidate's 503 try_send error, which we propagate below).
    let mut handles = Vec::with_capacity(n);
    for _ in 0..n {
        let state = state.clone();
        let req = req.clone();
        let prompt_tokens = prompt_tokens.clone();
        let correlation_id = correlation_id.0.clone();
        let candidate = tokio::spawn(spawn_best_of_candidate(
            state,
            req,
            prompt_tokens,
            max_tokens,
            correlation_id,
        ));
        handles.push(candidate);
    }

    // Join all candidates. First failure wins (the other candidates
    // keep running in the background — see the partial-failure
    // comment in the module doc above).
    let mut candidates: Vec<Vec<vllm_traits::SampledToken>> = Vec::with_capacity(n);
    let mut candidate_finish_reasons: Vec<vllm_traits::FinishReason> = Vec::with_capacity(n);
    for handle in handles {
        match handle.await {
            Ok(Ok((tokens, finish_reason))) => {
                candidates.push(tokens);
                candidate_finish_reasons.push(finish_reason);
            }
            Ok(Err(e)) => return Err(e),
            Err(e) => {
                return Err((
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse::new(
                        format!("best_of candidate task panicked: {e}").as_str(),
                        "server_error",
                    )),
                ));
            }
        }
    }

    // Rank by mean logprob — the index of the chosen candidate is
    // also the index into `candidate_finish_reasons`.
    let best_idx = rank_by_mean_logprob(&candidates);
    let best_tokens = &candidates[best_idx];
    let best_finish_reason = &candidate_finish_reasons[best_idx];

    // Apply echo + suffix to the chosen completion's text. The
    // prompt is cloned once here (we move it into the formatter);
    // the other N-1 candidates' texts are discarded (OpenAI's
    // contract: `best_of` returns ONE completion, not N).
    let text = clean_completion_text(
        &state.tokenizer,
        &state.tokenizer.decode(&token_ids(best_tokens)),
    );
    let text = apply_completion_meta(text, &prompt, req.echo, req.suffix.as_deref());

    // Render the chosen completion's finish_reason per OpenAI's
    // contract (same string mapping as the non-streaming single-shot
    // path; see that branch for the full rationale).
    let finish_reason_string = match best_finish_reason {
        vllm_traits::FinishReason::Length => "length",
        vllm_traits::FinishReason::Stop | vllm_traits::FinishReason::Cancelled => "stop",
    };

    let choice = CompletionChoice {
        text,
        index: 0,
        finish_reason: Some(finish_reason_string.to_string()),
        // P36 v0.3 wire-type follow-up engine wire-through: render
        // per-token logprobs (and top-K alternatives when the
        // request asked) from the chosen candidate's SampledToken
        // stream. The other N-1 candidates' logprobs are discarded
        // after ranking — matches OpenAI's "one completion, with
        // its logprobs" contract.
        logprobs: build_completion_choice_logprobs(&state.tokenizer, best_tokens, req.logprobs),
    };

    let usage = Usage::new(prompt_tokens_len, best_tokens.len());
    let response = CompletionResponse::new(
        format!("cmpl-{}", uuid::Uuid::new_v4()),
        req.model.unwrap_or_else(|| "default".to_string()),
        vec![choice],
        usage,
    );

    Ok(Json(response).into_response())
}

/// Apply OpenAI `echo` + `suffix` semantics to a generated
/// completion text (P35 v0.x wire-type follow-up engine
/// wire-through). This is the single authoritative point for the
/// response-side text formatting; both the non-streaming and
/// streaming paths call it so the contract stays in sync.
///
/// Per OpenAI legacy-completions spec:
/// - `echo = true`: prepend the prompt to the generated
///   continuation. The response `text` is `prompt + completion`.
/// - `suffix = Some(_)`: append the suffix to the response `text`.
///   The response `text` is `completion + suffix` (or `prompt +
///   completion + suffix` when both are set).
///
/// Both flags are independent: `echo` only affects the prefix,
/// `suffix` only affects the postfix. The continuation is always
/// preserved verbatim between them. `echo = false` (or omitted)
/// and `suffix = None` produce the pre-P35 behaviour (no prefix /
/// no postfix), so legacy clients are unaffected.
fn apply_completion_meta(
    completion: String,
    prompt: &str,
    echo: Option<bool>,
    suffix: Option<&str>,
) -> String {
    let mut out =
        String::with_capacity(completion.len() + prompt.len() + suffix.map_or(0, str::len));
    if echo == Some(true) {
        out.push_str(prompt);
    }
    out.push_str(&completion);
    if let Some(s) = suffix {
        out.push_str(s);
    }
    out
}

/// OpenAI-compatible `/v1/completions` HTTP handler. Dispatches to streaming
/// (SSE) or non-streaming based on `req.stream`.
///
/// Validates that `prompt` is non-empty and forwards an
/// [`vllm_core::types::EngineMessage::AddRequest`] to the engine for each call.
///
/// # Errors
///
/// Returns `(StatusCode, ErrorResponse)` when:
/// - prompt is empty (`BAD_REQUEST`)
/// - the engine channel is closed (`SERVICE_UNAVAILABLE`, code `engine_unavailable`)
/// - token decoding or SSE serialization fails
///
/// # Panics
///
/// Panics if the streaming path reaches a `seq_id_rx.expect(...)` after
/// the `is_streaming` branch already established that the oneshot was
/// constructed. This invariant is held by the surrounding `if
/// is_streaming { ... }` guard; the panic exists as a tripwire for
/// future refactors that might break the link between the two branches.
pub async fn completions(
    State(state): State<ApiState>,
    Extension(correlation_id): Extension<CorrelationId>,
    Json(req): Json<CompletionRequest>,
) -> Result<axum::response::Response, (axum::http::StatusCode, Json<ErrorResponse>)> {
    // API-01: reject OpenAI fields the engine does not yet honour
    // BEFORE doing any work. Mirror of chat.rs:
    // `validate_chat_request_fields`. Honest 400 > silent degradation.
    validate_completion_request_fields(&req)?;

    if req.prompt.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "prompt is required",
                "invalid_request_error",
            )),
        ));
    }

    let is_streaming = req.stream.unwrap_or(false);
    // Clone the prompt so `req` remains whole for downstream sampling
    // params forwarding + best_of per-candidate requests (P37).
    // The clone is cheap (prompt is typically short) and avoids
    // splitting the borrow on `req` after the sampling-params pass.
    let prompt = req.prompt.clone();
    let prompt_tokens = state.tokenizer.encode(&prompt);
    let prompt_tokens_len = prompt_tokens.len();
    let max_tokens = usize::try_from(req.max_tokens.unwrap_or(100)).unwrap_or(100);
    let total_max = prompt_tokens_len + max_tokens;

    // Production-readiness §4: reject requests whose
    // prompt + max_tokens would exceed the model's context
    // length. See chat.rs for the full rationale.
    if let Some(max_model_len) = state.max_model_len {
        if total_max > max_model_len {
            let message = format!(
                "prompt_tokens ({prompt_tokens_len}) + max_tokens ({max_tokens}) \
                 = {total_max} exceeds the model's context length ({max_model_len})"
            );
            return Err((
                axum::http::StatusCode::BAD_REQUEST,
                Json(ErrorResponse::with_code(
                    &message,
                    "invalid_request_error",
                    "context_length_exceeded",
                )),
            ));
        }
    }

    // P37 v0.x wire-type follow-up — engine wire-through: `best_of`
    // dispatch. When `best_of > 1`, spawn N parallel candidates via
    // `run_best_of`, rank by mean logprob, and return the single
    // best completion as a JSON response. The dispatch MUST happen
    // BEFORE the single-shot `Request::new` (which moves
    // `prompt_tokens`) so the per-candidate requests can reuse the
    // tokenized prompt.
    //
    // **Streaming interaction:** when `stream = true` AND `best_of > 1`,
    // we silently fall back to non-streaming (a single JSON response
    // instead of an SSE event stream). OpenAI's API accepts the
    // combination; the runtime behavior is what changes — the
    // response shape becomes a non-streaming document because
    // `best_of` requires ranking N candidates first.
    if let Some(n) = req.best_of {
        if n > 1 {
            return run_best_of(
                state,
                req,
                prompt_tokens,
                prompt,
                prompt_tokens_len,
                max_tokens,
                correlation_id,
            )
            .await;
        }
    }

    let mut request = vllm_core::types::Request::new(0, prompt_tokens, total_max);

    // Forward all sampling fields (P27/P28/P29/P30/P34/P36 wire-through).
    // The `populate_completion_sampling_params` helper is the single
    // authoritative point for the OpenAI → `SamplingParams` mapping;
    // it is reused by the `best_of` per-candidate requests below so
    // every candidate sees the exact same sampling config the user
    // submitted. See the helper's doc-comment for field-level rationale.
    populate_completion_sampling_params(&mut request, &req);

    // Reject sampling parameters the engine cannot honour (currently
    // beam_width > 1) BEFORE enqueuing — see `sampling_validation`.
    validate_sampling_params(&request.sampling_params)?;

    // Production-readiness recommendation: streaming variants
    // allocate a seq_id round-trip oneshot so we can learn the
    // engine-assigned id and forward `EngineMessage::CancelRequest`
    // if the client disconnects mid-stream. Without this, the
    // engine keeps generating tokens for a caller that has
    // already gone away.
    let (seq_id_tx, seq_id_rx) = if is_streaming {
        let (tx, rx) = tokio::sync::oneshot::channel();
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };

    let (response_tx, mut response_rx) = mpsc::channel(64);
    // API-01 finish_reason propagation (mirrors chat.rs): the engine
    // sends the [`FinishReason`] through this oneshot just before it
    // drops the token response channel, so we can emit the
    // OpenAI-correct `finish_reason` (`"length"` when the sequence hit
    // `max_tokens`, instead of the pre-fix hardcoded `"stop"`).
    let (finish_reason_tx, finish_reason_rx) = tokio::sync::oneshot::channel();

    // REL-01: use `try_send` so a saturated mailbox fails fast with
    // 503 `engine_overloaded` instead of blocking.
    state
        .engine_tx
        .try_send(vllm_core::types::EngineMessage::AddRequest {
            request,
            response_tx,
            seq_id_tx,
            finish_reason_tx: Some(finish_reason_tx),
            // Production-readiness §6: forward the correlation id
            // (same rationale as the chat handler). The engine run
            // loop's `tracing::info_span!("engine.add_request",
            // request_id)` attaches it to every synchronous log
            // line in add_request and its callees.
            request_id: Some(correlation_id.0.clone()),
        })
        .map_err(|e| match e {
            tokio::sync::mpsc::error::TrySendError::Full(_) => overload_response(),
            tokio::sync::mpsc::error::TrySendError::Closed(_) => unavailable_response(),
        })?;

    if is_streaming {
        // Block briefly until the engine assigns the seq_id (see
        // chat.rs for rationale on the 1 s cap).
        let seq_id: vllm_traits::SeqId = match tokio::time::timeout(
            std::time::Duration::from_secs(1),
            seq_id_rx.expect("seq_id_rx is set when is_streaming"),
        )
        .await
        {
            Ok(Ok(id)) => id,
            _ => return Err(unavailable_response()),
        };

        // Drop guard — see `chat::CancelOnDrop` for full rationale.
        let cancel_guard = std::sync::Arc::new(crate::openai::chat::CancelOnDrop {
            engine_tx: state.engine_tx.clone(),
            seq_id: std::sync::atomic::AtomicU64::new(seq_id),
            fired: std::sync::atomic::AtomicBool::new(false),
            request_id: format!(
                "cmpl_{}",
                uuid::Uuid::new_v4().to_string()[..8].to_uppercase()
            ),
        });

        let tokenizer = state.tokenizer.clone();
        // API-01 finish_reason propagation + `[DONE]` split: mirror
        // `chat.rs` — final chunk carries the real `finish_reason`
        // and `[DONE]` is a separate SSE event. See
        // `docs/technical-due-diligence/architecture-performance.md` §5.1.3.
        enum Terminal {
            Streaming,
            EmitDoneSentinel,
            Done,
        }
        // P35 v0.x wire-type follow-up engine wire-through: the
        // streaming path threads `echo` + `suffix` through the SSE
        // event stream. `echo = true` prepends the prompt to the
        // FIRST non-empty text chunk (matches OpenAI's accumulator
        // semantics — clients concatenate chunk `text` fields in
        // order, so prefixing the first chunk puts the prompt at
        // the start of the visible response). `suffix = Some(_)`
        // puts the suffix into the FINAL chunk's `text` field (the
        // chunk that carries `finish_reason`) — same accumulator
        // reasoning: suffixing the final chunk puts the suffix at
        // the end. The `first_chunk_sent` flag tracks whether we've
        // already emitted the prompt-prefixed chunk; we set it on
        // the first chunk that carries non-empty decoded text.
        let prompt_text = prompt.clone();
        let echo_flag = req.echo;
        let suffix_text = req.suffix.clone();
        let stream = stream::unfold(
            (
                response_rx,
                cancel_guard.clone(),
                Some(finish_reason_rx),
                Terminal::Streaming,
                false, // first_chunk_sent
            ),
            move |(mut rx, cancel_guard, mut reason_rx_opt, mut terminal, mut first_chunk_sent)| {
                let tokenizer = tokenizer.clone();
                let prompt_text = prompt_text.clone();
                let echo_flag = echo_flag;
                let suffix_text = suffix_text.clone();
                async move {
                    match terminal {
                        Terminal::Done => None,
                        Terminal::EmitDoneSentinel => {
                            terminal = Terminal::Done;
                            Some((
                                Ok::<Event, Infallible>(Event::default().data("[DONE]")),
                                (rx, cancel_guard, reason_rx_opt, terminal, first_chunk_sent),
                            ))
                        }
                        Terminal::Streaming => match rx.recv().await {
                            Some(sampled) => {
                                let text = tokenizer.decode(&[sampled.token]);
                                if should_skip_token_text(&tokenizer, &text) {
                                    return Some((
                                        Ok::<Event, Infallible>(Event::default().data("")),
                                        (
                                            rx,
                                            cancel_guard,
                                            reason_rx_opt,
                                            terminal,
                                            first_chunk_sent,
                                        ),
                                    ));
                                }
                                // echo=true on the FIRST non-empty
                                // text chunk: prepend prompt.
                                let text = if echo_flag == Some(true) && !first_chunk_sent {
                                    first_chunk_sent = true;
                                    let mut out =
                                        String::with_capacity(prompt_text.len() + text.len());
                                    out.push_str(&prompt_text);
                                    out.push_str(&text);
                                    out
                                } else {
                                    if !first_chunk_sent {
                                        first_chunk_sent = true;
                                    }
                                    text
                                };
                                let chunk = serde_json::json!({
                                    "id": "cmpl-stream",
                                    "object": "text_completion",
                                    "choices": [{
                                        "text": text,
                                        "index": 0,
                                    }]
                                });
                                let sse_payload = chunk.to_string();
                                Some((
                                    Ok(Event::default().data(sse_payload)),
                                    (rx, cancel_guard, reason_rx_opt, terminal, first_chunk_sent),
                                ))
                            }
                            None => {
                                let reason_string = if let Some(rx) = reason_rx_opt.take() {
                                    match rx.await {
                                        Ok(vllm_traits::FinishReason::Length) => "length",
                                        Ok(vllm_traits::FinishReason::Stop) => "stop",
                                        Ok(vllm_traits::FinishReason::Cancelled) => "stop",
                                        Err(_) => "stop",
                                    }
                                } else {
                                    "stop"
                                };
                                // Natural completion — disarm so Drop
                                // doesn't send a redundant CancelRequest.
                                cancel_guard.disarm();
                                // suffix lands on the final chunk's
                                // text field (instead of the default
                                // empty string). Clients concatenate
                                // all chunk `text` fields in order,
                                // so the suffix appears at the end
                                // of the visible response.
                                let text = suffix_text.clone().unwrap_or_default();
                                let chunk = serde_json::json!({
                                    "id": "cmpl-stream",
                                    "object": "text_completion",
                                    "choices": [{
                                        "text": text,
                                        "index": 0,
                                        "finish_reason": reason_string,
                                    }]
                                });
                                let sse_payload = chunk.to_string();
                                terminal = Terminal::EmitDoneSentinel;
                                Some((
                                    Ok(Event::default().data(sse_payload)),
                                    (rx, cancel_guard, reason_rx_opt, terminal, first_chunk_sent),
                                ))
                            }
                        },
                    }
                }
            },
        );

        return Ok(Sse::new(Box::pin(stream)).into_response());
    }

    // 非流式 - 返回普通 JSON
    let mut tokens = Vec::new();
    while let Some(sampled) = response_rx.recv().await {
        tokens.push(sampled);
    }

    // Engine sends the reason before closing the response channel, so
    // this resolves immediately in the normal case. Fall back to
    // `"stop"` only when the oneshot was dropped without a value
    // (e.g. engine panicked between the two steps).
    let finish_reason = match finish_reason_rx.await {
        Ok(vllm_traits::FinishReason::Length) => "length".to_string(),
        Ok(vllm_traits::FinishReason::Stop) => "stop".to_string(),
        Ok(vllm_traits::FinishReason::Cancelled) => "stop".to_string(),
        Err(_) => "stop".to_string(),
    };

    // P35 v0.x wire-type follow-up engine wire-through: apply
    // OpenAI `echo` + `suffix` semantics to the generated
    // continuation before serializing the response. `echo` is
    // applied only when `Some(true)` — `Some(false)` and `None`
    // both preserve the pre-P35 no-prefix behavior so legacy
    // clients are unaffected. `suffix` is applied when
    // `Some(_)`. The original `prompt` is still in scope from
    // line 68 (it's the raw text the user submitted, before
    // tokenization — which is exactly what OpenAI's `echo` spec
    // requires: the response echoes the request body verbatim,
    // not a re-decoded token sequence).
    let text = clean_completion_text(
        &state.tokenizer,
        &state.tokenizer.decode(&token_ids(&tokens)),
    );
    let text = apply_completion_meta(text, &prompt, req.echo, req.suffix.as_deref());
    let choice = CompletionChoice {
        text,
        index: 0,
        finish_reason: Some(finish_reason),
        // P36 v0.3 wire-type follow-up engine wire-through: render
        // per-token logprobs (and top-K alternatives when the
        // request asked) from the engine's SampledToken stream.
        logprobs: build_completion_choice_logprobs(&state.tokenizer, &tokens, req.logprobs),
    };

    let usage = Usage::new(prompt_tokens_len, tokens.len());
    let response = CompletionResponse::new(
        format!("cmpl-{}", uuid::Uuid::new_v4()),
        req.model.unwrap_or_else(|| "default".to_string()),
        vec![choice],
        usage,
    );

    Ok(Json(response).into_response())
}

/// REL-01: 503 response returned when the bounded engine mailbox is
/// saturated (`mpsc::error::TrySendError::Full`). Distinct from
/// `unavailable_response` so clients can implement smarter retry
/// (backoff + jitter) for transient overload.
fn overload_response() -> (axum::http::StatusCode, Json<ErrorResponse>) {
    (
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorResponse::with_code(
            "Engine overloaded; retry with backoff",
            "server_error",
            "engine_overloaded",
        )),
    )
}

/// 503 response returned when the engine channel is closed
/// (`mpsc::error::TrySendError::Closed`). Distinct from
/// `overload_response` so clients know not to retry — the engine is
/// gone.
fn unavailable_response() -> (axum::http::StatusCode, Json<ErrorResponse>) {
    (
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorResponse::with_code(
            "Engine unavailable",
            "server_error",
            "engine_unavailable",
        )),
    )
}

// Unit tests live in `tests.rs` (sibling) to keep this handler file
// under the 800-line soft cap. They cover the empty-prompt validation
// path and the engine-channel error mapping (closed channel → 503
// `engine_unavailable`).
#[cfg(test)]
mod tests;
