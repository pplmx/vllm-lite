//! Engine <-> server control messages.

use tokio::sync::{mpsc, oneshot};

use crate::metrics::MetricsSnapshot;
use crate::types::request::Request;
use vllm_traits::{SeqId, TokenId};

/// Messages the server sends to the engine over the
/// [`crate::engine::Engine`] mailbox. The engine's
/// [`crate::engine::Engine::run`] loop drains these each iteration before
/// stepping.
#[derive(Debug)]
pub enum EngineMessage {
    /// Submit a new generation `request`. Tokens are streamed back to
    /// `response_tx` as they are produced; the channel is closed when the
    /// sequence finishes (either naturally or via cancellation).
    ///
    /// `seq_id_tx` is optional. When supplied, the engine replies
    /// with the assigned [`SeqId`] once the request is admitted
    /// (or with a sentinel `0` if admission was rejected, e.g.
    /// empty prompt). Streaming HTTP handlers use this to learn
    /// the seq_id so they can send [`EngineMessage::CancelRequest`]
    /// if the client disconnects mid-stream — otherwise the
    /// engine would keep generating tokens for a caller that has
    /// already gone away.
    ///
    /// `finish_reason_tx` is optional. When supplied, the engine
    /// sends the [`vllm_traits::FinishReason`] describing why the
    /// sequence stopped (length cap, cancellation, etc.) **before**
    /// dropping `response_tx`. The HTTP layer maps this to OpenAI's
    /// `finish_reason` (`"length"`, `"stop"`, etc.). Pre-fix, the
    /// channel close alone signalled completion and the HTTP layer
    /// hardcoded `finish_reason = "stop"` for every response, even
    /// when the engine actually stopped because the sequence hit
    /// `max_tokens` — see
    /// `docs/technical-due-diligence/architecture-performance.md` §5.1.2
    /// and the v31.0 P4 follow-up batch.
    ///
    /// `request_id` is the correlation id minted (or forwarded) by
    /// the HTTP boundary — see
    /// `docs/technical-due-diligence/production-readiness.md` §6.
    /// When `Some`, the engine run loop enters a
    /// `tracing::info_span!("engine.add_request", request_id)`
    /// around the synchronous `add_request` call so every
    /// engine-side log line for this HTTP request carries the
    /// same id, enabling cross-layer (HTTP → scheduler → engine)
    /// log correlation. When `None` (test fixtures, non-HTTP
    /// callers), the span still enters with `request_id = None`
    /// and the field renders as `null` in the JSON span output.
    AddRequest {
        request: Request,
        response_tx: mpsc::Sender<TokenId>,
        seq_id_tx: Option<oneshot::Sender<SeqId>>,
        finish_reason_tx: Option<oneshot::Sender<vllm_traits::FinishReason>>,
        /// Correlation id forwarded from the HTTP boundary; see
        /// `request_id` field-level doc above.
        request_id: Option<String>,
    },
    /// Cancel an in-flight or queued sequence identified by `seq_id`
    /// (the value returned from the engine when the request was
    /// admitted via [`EngineMessage::AddRequest`]).
    ///
    /// Production-readiness recommendation (input boundary + client
    /// disconnect): when a streaming HTTP client disconnects, the
    /// handler must release the engine-side resources (KV blocks,
    /// sequence slot, response channel) so the next batch isn't held
    /// up generating tokens for a caller that has already gone
    /// away. The engine drops the sequence's `response_tx`, so any
    /// subsequent `send` from the step loop becomes a no-op.
    ///
    /// Unknown `seq_id` is a no-op (the sequence already finished or
    /// was cancelled by another path); the engine logs nothing in
    /// that case so a benign race doesn't pollute operator logs.
    CancelRequest { seq_id: SeqId },
    /// Request a [`MetricsSnapshot`]. The engine responds once with the
    /// current snapshot; the response channel is unbounded because metrics
    /// reads are cheap and rare.
    GetMetrics {
        response_tx: mpsc::UnboundedSender<MetricsSnapshot>,
    },
    /// Compute embeddings for each pre-tokenized `input_tokens` batch.
    /// Returns one float vector per input sequence.
    GetEmbeddings {
        input_tokens: Vec<Vec<TokenId>>,
        response_tx: mpsc::UnboundedSender<Vec<Vec<f32>>>,
    },
    /// Ask the engine to exit its run loop. The server still serves pending
    /// requests up to its shutdown grace period before terminating the
    /// process.
    Shutdown,
}
