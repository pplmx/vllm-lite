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
    AddRequest {
        request: Request,
        response_tx: mpsc::Sender<TokenId>,
        seq_id_tx: Option<oneshot::Sender<SeqId>>,
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
