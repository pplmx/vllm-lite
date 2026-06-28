// Sub-module for health, error, cancel, and add_request methods on Engine.
// See mod.rs for the Engine struct definition.

use crate::engine::Engine;
use crate::types::Request;
use tokio::sync::mpsc;
use vllm_traits::{SeqId, TokenId};

impl Engine {
    pub const fn is_healthy(&self) -> bool {
        self.error_count < 10
    }

    pub fn get_last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    pub fn cancel_request(&mut self, seq_id: SeqId) -> bool {
        let canceled = self.scheduler.cancel_request(seq_id);
        if canceled {
            self.response_txs.remove(&seq_id);
            self.scheduler.metrics.remove_per_request(seq_id);
        }
        canceled
    }

    pub fn add_request(&mut self, req: Request, response_tx: mpsc::Sender<TokenId>) -> SeqId {
        // Validate prompt is not empty
        if req.prompt.is_empty() {
            self.last_error = Some("prompt cannot be empty".to_string());
            return 0;
        }

        let seq_id = self.scheduler.add_request(req);
        self.response_txs.insert(seq_id, response_tx);
        seq_id
    }
}
