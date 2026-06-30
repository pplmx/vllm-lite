//! Engine <-> server control messages.

use tokio::sync::mpsc;

use crate::metrics::MetricsSnapshot;
use crate::types::request::Request;
use vllm_traits::TokenId;

/// Messages the server sends to the engine over the
/// [`crate::engine::Engine`] mailbox. The engine's
/// [`crate::engine::Engine::run`] loop drains these each iteration before
/// stepping.
#[derive(Debug)]
pub enum EngineMessage {
    /// Submit a new generation `request`. Tokens are streamed back to
    /// `response_tx` as they are produced; the channel is closed when the
    /// sequence finishes (either naturally or via cancellation).
    AddRequest {
        request: Request,
        response_tx: mpsc::Sender<TokenId>,
    },
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
