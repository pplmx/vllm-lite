//! Engine <-> server control messages.

use tokio::sync::mpsc;

use crate::metrics::MetricsSnapshot;
use crate::types::request::Request;
use vllm_traits::TokenId;

/// EngineMessage: engine message enumeration.
pub enum EngineMessage {
    AddRequest {
        request: Request,
        response_tx: mpsc::Sender<TokenId>,
    },
    GetMetrics {
        response_tx: mpsc::UnboundedSender<MetricsSnapshot>,
    },
    GetEmbeddings {
        input_tokens: Vec<Vec<TokenId>>,
        response_tx: mpsc::UnboundedSender<Vec<Vec<f32>>>,
    },
    Shutdown,
}
