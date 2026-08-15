//! Batch job execution: drives a [`BatchJob`] through the engine to
//! completion in a background tokio task.
//!
//! Previously (API-01) `POST /v1/batches` returned 501 because no
//! worker existed to advance a job from `Pending` to `Completed` —
//! the "module exists but capability missing" gap that the technical
//! due diligence flagged. This module is that worker: for each prompt
//! in submission order it builds an engine `AddRequest`, collects the
//! full token stream, records a `BatchResultItem`, and transitions the
//! job to `Completed` (all succeeded) or `Failed` (any errored).
#![allow(clippy::module_name_repetitions)]

use std::sync::Arc;

use tokio::sync::mpsc;
use vllm_model::tokenizer::Tokenizer;
use vllm_traits::SampledToken;

use super::manager::BatchManager;
use super::types::{BatchEndpoint, BatchJob, BatchResultItem};
use crate::ApiState;
use crate::openai::chat_template::ChatTemplate;
use crate::openai::types::ChatMessage;

/// Bounded wait for the engine's assigned `seq_id` (RIL ISS-070 / TASK-083).
///
/// The batch worker is background work (no interactive client), so this is
/// intentionally a little more generous than the 1s bound the sync
/// streaming paths use (`completions.rs:932`) — enough headroom for a real
/// engine mid-forward while still bounding a wedged engine's stall so the
/// job cannot hang `InProgress` forever.
const SEQ_ID_TIMEOUT_SECS: u64 = 3;

/// Spawn a background task that drives `job` to a terminal state.
///
/// The spawned task borrows clones of the engine handle, tokenizer,
/// and batch manager from `state`; the original `job` is consumed so
/// its `id` lives only in the manager map. The caller returns
/// immediately with the `JoinHandle` (for tests to `.await`).
#[must_use]
pub fn spawn_batch_worker(state: &ApiState, job: BatchJob) -> tokio::task::JoinHandle<()> {
    let engine_tx = state.engine_tx.clone();
    let tokenizer = Arc::clone(&state.tokenizer);
    let manager = Arc::clone(&state.batch_manager);
    let architecture = state.architecture;
    let max_model_len = state.max_model_len;
    tokio::spawn(async move {
        run_batch_job(
            job,
            engine_tx,
            tokenizer,
            manager,
            architecture,
            max_model_len,
        )
        .await;
    })
}

/// Drive `job` to `Completed`/`Failed`/`Cancelled` by executing every prompt.
///
/// `max_model_len` is threaded in so each prompt passes the same
/// `check_context_length` gate as the non-batch generation paths — the
/// batch worker must not re-open the ISS-044 `max_tokens`-ceiling bypass
/// that the chat/completions/embeddings endpoints all enforce.
async fn run_batch_job(
    job: BatchJob,
    engine_tx: crate::api::EngineHandle,
    tokenizer: Arc<Tokenizer>,
    manager: Arc<BatchManager>,
    architecture: vllm_model::config::Architecture,
    max_model_len: Option<usize>,
) {
    let job_id = job.id.clone();
    let cancel_requested = std::sync::Arc::clone(&job.cancel_requested);
    // Fail fast if the job vanished between create and pickup (e.g. a
    // concurrent admin action removed it) — nothing to do.
    if !manager.mark_in_progress(&job_id).await {
        return;
    }

    let max_tokens = usize::try_from(job.max_tokens.unwrap_or(100)).unwrap_or(100);

    for (index, prompt) in job.prompts.iter().enumerate() {
        // Cooperative stop: the manager set the flag via
        // `POST /v1/batches/{id}/cancel`. Skip the remaining prompts —
        // partial results already recorded stay readable.
        if cancel_requested.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
        let result = execute_one(
            job.endpoint,
            prompt,
            max_tokens,
            job.temperature,
            architecture,
            max_model_len,
            &job_id,
            index,
            &engine_tx,
            &tokenizer,
            &cancel_requested,
        )
        .await;
        let item = match result {
            Ok(content) => BatchResultItem {
                index,
                status: "succeeded".to_string(),
                content: Some(content),
                error: None,
            },
            Err(message) => BatchResultItem {
                index,
                status: "failed".to_string(),
                content: None,
                error: Some(message),
            },
        };
        manager.add_result(&job_id, item).await;
    }

    // Terminal state. If the user cancelled, `request_cancel` already
    // transitioned the job to `Cancelled` — do not override it. Otherwise
    // any failure marks the whole batch Failed (partial results stay
    // readable via get_batch_results).
    if cancel_requested.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let any_failed = job.prompts.is_empty()
        || manager
            .get_job(&job_id)
            .await
            .is_some_and(|j| j.results.iter().any(|r| r.status == "failed"));
    if any_failed {
        manager.set_failed(&job_id).await;
    } else {
        manager.set_completed(&job_id).await;
    }
}

/// Execute one prompt against the engine and return the generated text.
///
/// `cancel_requested` is the job's shared cancellation flag; when set the
/// in-flight engine sequence is cancelled via
/// [`EngineMessage::CancelRequest`](vllm_core::types::EngineMessage::CancelRequest)
/// (so the engine stops generating for a caller that has gone away) and the
/// collected tokens so far are returned.
///
/// # Errors
///
/// Returns the error message string when the request could not be
/// admitted (engine mailbox full/closed) — surfaced as a failed
/// `BatchResultItem` rather than aborting the whole batch.
async fn execute_one(
    endpoint: BatchEndpoint,
    prompt: &str,
    max_tokens: usize,
    temperature: Option<f32>,
    architecture: vllm_model::config::Architecture,
    max_model_len: Option<usize>,
    job_id: &str,
    index: usize,
    engine_tx: &crate::api::EngineHandle,
    tokenizer: &Tokenizer,
    cancel_requested: &std::sync::atomic::AtomicBool,
) -> Result<String, String> {
    let prompt_text = match endpoint {
        BatchEndpoint::Chat => {
            // One batch prompt maps to a single user message, templated
            // for the loaded architecture — same as the non-batch chat
            // path before tokenization.
            let template = ChatTemplate::for_architecture(architecture);
            let messages = [ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
                name: None,
            }];
            crate::openai::chat::build_prompt_from_messages(template, &messages)
        }
        BatchEndpoint::Completion => prompt.to_string(),
    };

    let prompt_tokens = tokenizer.encode(&prompt_text);
    // Same context-length gate as the chat/completions/embeddings paths:
    // `prompt + max_tokens` must fit `max_model_len` (or the hard
    // `DEFAULT_MAX_GENERATION_TOKENS` ceiling when none is configured).
    // Without this, a batch prompt can carry `max_tokens = i64::MAX`
    // against an unconfigured model and re-open the ISS-044 DoS (unbounded
    // per-sequence token growth until OOM) on the new endpoint.
    if let Err((_, json)) =
        crate::openai::chat::check_context_length(prompt_tokens.len(), max_tokens, max_model_len)
    {
        return Err(json.error.message.clone());
    }
    let mut request = vllm_core::types::Request::new(0, prompt_tokens.clone(), max_tokens);
    if let Some(t) = temperature {
        request.sampling_params.temperature = t;
    }

    let (response_tx, mut response_rx) = mpsc::channel(64);
    let (seq_id_tx, seq_id_rx) = tokio::sync::oneshot::channel();

    engine_tx
        .try_send(vllm_core::types::EngineMessage::AddRequest {
            request: Box::new(request),
            response_tx,
            seq_id_tx: Some(seq_id_tx),
            finish_reason_tx: None,
            // Production-readiness §6: correlation id must be unique per
            // request so engine-side spans are traceable to a specific
            // prompt. Job + index (not prompt length) — a length is not
            // unique across same-sized prompts and breaks log correlation.
            request_id: Some(format!("batch:{job_id}:{index}")),
        })
        .map_err(|e| crate::openai::chat::map_engine_send_error(&e))
        .map_err(|(_, json)| json.error.message.clone())?;

    // Learn the engine-assigned seq_id so a cancellation can cancel this
    // specific sequence (not just skip the next prompt).
    //
    // RIL ISS-070 / TASK-083: the wait is BOUNDED. The sync streaming paths
    // (`completions.rs:932`, `chat.rs:1403`) cap the seq_id await at 1s and
    // fail the request on timeout; without an equivalent bound here, a
    // stalled engine (long/wedged forward before it drains the mailbox and
    // replies) would hang this background worker on `seq_id_rx.await`
    // forever, leaving the batch job `InProgress` indefinitely — and
    // `BatchManager::sweep_expired` only evicts terminal jobs, so the
    // retention is permanent. We use a slightly longer bound than the HTTP
    // paths (3s vs 1s) because the batch worker is background work, not a
    // client-facing stream — the extra headroom tolerates a real engine
    // mid-forward without a spurious failure. On timeout (or a closed
    // channel — engine dropped `seq_id_tx` without sending, e.g. panic
    // between AddRequest processing and the seq_id send) the item fails
    // with a clear message and the worker moves on to the next prompt.
    let seq_id = match tokio::time::timeout(
        std::time::Duration::from_secs(SEQ_ID_TIMEOUT_SECS),
        seq_id_rx,
    )
    .await
    {
        Ok(Ok(seq_id)) => seq_id,
        Ok(Err(_)) => {
            return Err(
                "engine dropped the sequence-id channel before assigning one (stalled or failed)"
                    .to_string(),
            );
        }
        Err(_) => {
            return Err(format!(
                "engine did not assign a sequence id within {SEQ_ID_TIMEOUT_SECS}s"
            ));
        }
    };

    let mut tokens = Vec::new();
    while let Some(sampled) = response_rx.recv().await {
        tokens.push(sampled);
        if cancel_requested.load(std::sync::atomic::Ordering::Relaxed) {
            // Best-effort: cancel the in-flight sequence so the engine
            // stops generating; the channel then closes and we break.
            let _ = engine_tx.try_send(vllm_core::types::EngineMessage::CancelRequest { seq_id });
            break;
        }
    }

    let token_ids: Vec<u32> = tokens.iter().map(|s: &SampledToken| s.token).collect();
    let text = tokenizer.decode(&token_ids[..]);
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::batch::types::BatchStatus;
    use crate::test_fixtures;
    use vllm_model::config::Architecture;

    fn create_state_with_mock_engine() -> (ApiState, tokio::task::JoinHandle<()>) {
        test_fixtures::api_state_with_mock_engine(Architecture::Qwen3, vec![1, 2, 3])
    }

    /// [`ApiState`] whose engine drains the mailbox but **never** replies on
    /// `seq_id_tx` — simulating the engine that assigned a `seq_id`
    /// synchronously in the real run loop but is stalled (long forward /
    /// wedged) so the reply is delayed past the worker's bound. Without the
    /// ISS-070 / TASK-083 `seq_id` timeout, this would hang the batch job in
    /// `InProgress` forever.
    fn create_state_with_stalled_engine() -> ApiState {
        let mut state = test_fixtures::api_state(Architecture::Qwen3);
        let (engine_tx, mut engine_rx) =
            tokio::sync::mpsc::channel::<vllm_core::types::EngineMessage>(256);
        tokio::spawn(async move {
            // Deliberately leak each received message (and with it its
            // `seq_id_tx` / `response_tx` senders) WITHOUT sending — the
            // faithful stalled-engine behaviour. The real engine holds
            // `seq_id_tx` until it drains the mailbox after a (possibly
            // long/wedged) forward; only when it sends does the worker's
            // `seq_id_rx` resolve. Draining the channel but never replying
            // keeps `seq_id_rx` pending, so the worker awaits it forever
            // without a timeout. `mem::forget` keeps the senders alive for
            // the test's lifetime (released at process teardown).
            // SAFETY-free by design: intentional leak, no unsafe.
            while let Some(msg) = engine_rx.recv().await {
                std::mem::forget(msg);
            }
        });
        state.engine_tx = engine_tx;
        state
    }

    #[tokio::test]
    async fn worker_completes_batch_with_results() {
        let (state, _engine) = create_state_with_mock_engine();
        let job = state
            .batch_manager
            .create_job(
                BatchEndpoint::Completion,
                vec!["hello".to_string(), "world".to_string()],
                None,
                Some(10),
                Some(0.7),
            )
            .await;

        let handle = spawn_batch_worker(&state, state.batch_manager.get_job(&job).await.unwrap());
        handle.await.expect("worker task must not panic");

        let finished = state.batch_manager.get_job(&job).await.unwrap();
        assert_eq!(finished.status, BatchStatus::Completed);
        assert_eq!(finished.results.len(), 2);
        assert!(finished.results.iter().all(|r| r.status == "succeeded"));
        assert!(finished.results.iter().all(|r| r.content.is_some()));
        assert!(finished.completed_at.is_some());
    }

    #[tokio::test]
    async fn worker_rejects_prompt_exceeding_context_length() {
        // `max_tokens` beyond the hard ceiling (no max_model_len configured
        // -> check_context_length caps at DEFAULT_MAX_GENERATION_TOKENS) must
        // fail the item locally, never reaching the engine — re-opening the
        // ISS-044 DoS on the batch endpoint is not allowed.
        let (state, _engine) = create_state_with_mock_engine();
        let job = state
            .batch_manager
            .create_job(
                BatchEndpoint::Completion,
                vec!["hello".to_string()],
                None,
                Some(100_000),
                None,
            )
            .await;

        let handle = spawn_batch_worker(&state, state.batch_manager.get_job(&job).await.unwrap());
        handle.await.expect("worker task must not panic");

        let finished = state.batch_manager.get_job(&job).await.unwrap();
        assert_eq!(finished.status, BatchStatus::Failed);
        assert_eq!(finished.results.len(), 1);
        assert_eq!(finished.results[0].status, "failed");
        let error = finished.results[0].error.as_deref().unwrap_or("");
        assert!(
            error.contains("max_tokens") || error.contains("context"),
            "error should describe the context-length rejection, got: {error}"
        );
    }

    #[tokio::test]
    async fn worker_stops_dispatching_when_cancelled_before_start() {
        // Cancellation requested before the worker picks the job up: the
        // loop sees the flag immediately, dispatches nothing, and leaves
        // the job `Cancelled` (no override to Completed/Failed).
        let (state, _engine) = create_state_with_mock_engine();
        let job = state
            .batch_manager
            .create_job(
                BatchEndpoint::Completion,
                vec!["a".to_string(), "b".to_string()],
                None,
                Some(10),
                None,
            )
            .await;
        state.batch_manager.request_cancel(&job).await;

        let handle = spawn_batch_worker(&state, state.batch_manager.get_job(&job).await.unwrap());
        handle.await.expect("worker task must not panic");

        let finished = state.batch_manager.get_job(&job).await.unwrap();
        assert_eq!(finished.status, BatchStatus::Cancelled);
        assert!(finished.results.is_empty(), "nothing should have run");
    }

    #[tokio::test]
    async fn worker_marks_failed_when_engine_unavailable() {
        // `api_state` (the bare fixture) creates an mpsc channel and drops
        // the receiver immediately, so its engine channel is always closed:
        // the worker's try_send errors > each item fails > job is Failed.
        let state = test_fixtures::api_state(Architecture::Qwen3);
        let job = state
            .batch_manager
            .create_job(
                BatchEndpoint::Chat,
                vec!["ping".to_string()],
                None,
                Some(5),
                None,
            )
            .await;

        let handle = spawn_batch_worker(&state, state.batch_manager.get_job(&job).await.unwrap());
        handle.await.expect("worker task must not panic");

        let finished = state.batch_manager.get_job(&job).await.unwrap();
        assert_eq!(finished.status, BatchStatus::Failed);
        assert_eq!(finished.results.len(), 1);
        assert_eq!(finished.results[0].status, "failed");
        assert!(finished.results[0].error.is_some());
    }

    /// RIL ISS-070 / TASK-083: a stalled engine must not hang the batch job
    /// in `InProgress` forever. Pre-fix `execute_one` awaited `seq_id_rx`
    /// with no timeout (worker.rs:211); if the engine never replied (stalled
    /// modelling a long/wedged forward), the worker awaited indefinitely,
    /// and `BatchManager` never evicts in-progress jobs — unbounded
    /// retention (the terminal-only sweep makes it permanent). The worker
    /// now bounds the wait and fails the item instead.
    #[tokio::test]
    async fn worker_fails_item_when_engine_stalls_on_seq_id() {
        let state = create_state_with_stalled_engine();
        let job = state
            .batch_manager
            .create_job(
                BatchEndpoint::Completion,
                vec!["ping".to_string()],
                None,
                Some(5),
                None,
            )
            .await;

        // The worker must terminate within a bounded window even though the
        // engine never replies. 30s hard ceiling is generous vs the 3s bound;
        // a pre-fix hang would time out here and fail the test.
        let handle = spawn_batch_worker(&state, state.batch_manager.get_job(&job).await.unwrap());
        tokio::time::timeout(std::time::Duration::from_secs(30), handle)
            .await
            .expect("worker must not hang forever on a stalled engine")
            .expect("worker task must not panic");

        let finished = state.batch_manager.get_job(&job).await.unwrap();
        assert_eq!(finished.status, BatchStatus::Failed);
        assert_eq!(finished.results.len(), 1);
        assert_eq!(finished.results[0].status, "failed");
        let error = finished.results[0].error.as_deref().unwrap_or("");
        assert!(
            error.contains("sequence id")
                || error.contains("seq_id")
                || error.contains("timed out"),
            "error should describe the seq-id timeout; got: {error}"
        );
    }
}
