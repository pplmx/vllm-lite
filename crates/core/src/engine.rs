use crate::error::Result;
use crate::scheduler::Scheduler;
use crate::types::{BatchOutput, EngineMessage, Request, SchedulerConfig, SeqId, TokenId};
use std::collections::HashMap;
use tokio::sync::mpsc;

pub trait ModelBackend: Send + Sync {
    fn forward(
        &self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
    ) -> Result<BatchOutput>;
}

pub struct Engine<M: ModelBackend> {
    pub scheduler: Scheduler,
    pub model: M,
    response_txs: HashMap<SeqId, mpsc::UnboundedSender<TokenId>>,
}

impl<M: ModelBackend> Engine<M> {
    pub fn new(model: M) -> Self {
        Self {
            scheduler: Scheduler::new(),
            model,
            response_txs: HashMap::new(),
        }
    }

    pub fn with_config(model: M, config: SchedulerConfig) -> Self {
        Self {
            scheduler: Scheduler::with_config(config),
            model,
            response_txs: HashMap::new(),
        }
    }

    pub fn add_request(
        &mut self,
        req: Request,
        response_tx: mpsc::UnboundedSender<TokenId>,
    ) -> SeqId {
        let seq_id = self.scheduler.add_request(req);
        self.response_txs.insert(seq_id, response_tx);
        seq_id
    }

    pub fn step(&mut self) -> Result<Vec<(SeqId, TokenId)>> {
        let batch = self.scheduler.build_batch();
        if batch.is_empty() {
            return Ok(vec![]);
        }

        let output = self
            .model
            .forward(&batch.seq_ids, &batch.input_tokens, &batch.positions)?;

        let input_counts: Vec<usize> = batch
            .input_tokens
            .iter()
            .map(|v| v.len())
            .collect::<Vec<_>>();

        self.scheduler
            .update(&batch.seq_ids, &output.next_tokens, &input_counts);

        // Send tokens to response channels
        let mut results = Vec::new();
        for (seq_id, token) in batch.seq_ids.iter().zip(output.next_tokens.iter()) {
            if let Some(tx) = self.response_txs.get(seq_id) {
                let _ = tx.send(*token);
            }
            results.push((*seq_id, *token));
        }

        // Clean up channels for finished sequences
        for seq in self.scheduler.finished_sequences() {
            self.response_txs.remove(&seq.id);
        }

        Ok(results)
    }

    pub fn run(&mut self, mut msg_rx: mpsc::UnboundedReceiver<EngineMessage>) {
        loop {
            while let Ok(msg) = msg_rx.try_recv() {
                match msg {
                    EngineMessage::AddRequest {
                        request,
                        response_tx,
                    } => {
                        self.add_request(request, response_tx);
                    }
                    EngineMessage::Shutdown => return,
                }
            }

            if self.scheduler.has_pending() {
                if let Err(e) = self.step() {
                    eprintln!("Engine step error: {}", e);
                }
            }

            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    pub fn has_pending(&self) -> bool {
        self.scheduler.has_pending()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Request;
    use tokio::sync::mpsc;

    struct StubModel {
        token_to_return: TokenId,
    }

    impl ModelBackend for StubModel {
        fn forward(
            &self,
            seq_ids: &[SeqId],
            _input_tokens: &[Vec<TokenId>],
            _positions: &[Vec<usize>],
        ) -> Result<BatchOutput> {
            Ok(BatchOutput {
                seq_ids: seq_ids.to_vec(),
                next_tokens: seq_ids.iter().map(|_| self.token_to_return).collect(),
            })
        }
    }

    #[test]
    fn test_engine_streaming() {
        let mut engine = Engine::new(StubModel {
            token_to_return: 42,
        });
        let (tx, mut rx) = mpsc::unbounded_channel();

        engine.add_request(Request::new(1, vec![10, 20], 5), tx);

        let out = engine.step().unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(rx.try_recv().unwrap(), 42);

        let out = engine.step().unwrap();
        assert_eq!(out[0], (1, 42));
        assert_eq!(rx.try_recv().unwrap(), 42);

        let out = engine.step().unwrap();
        assert_eq!(out[0], (1, 42));
        assert_eq!(rx.try_recv().unwrap(), 42);

        assert!(!engine.has_pending());
    }

    #[test]
    fn test_engine_multi_request() {
        let mut engine = Engine::new(StubModel {
            token_to_return: 10,
        });
        let (tx1, mut rx1) = mpsc::unbounded_channel();
        let (tx2, mut rx2) = mpsc::unbounded_channel();

        engine.add_request(Request::new(1, vec![10], 3), tx1);
        engine.add_request(Request::new(2, vec![20], 3), tx2);

        // Step 1: both prefill
        engine.step().unwrap();
        assert_eq!(rx1.try_recv().unwrap(), 10);
        assert_eq!(rx2.try_recv().unwrap(), 10);

        // Step 2: both decode
        engine.step().unwrap();
        assert_eq!(rx1.try_recv().unwrap(), 10);
        assert_eq!(rx2.try_recv().unwrap(), 10);

        // Done
        assert!(!engine.has_pending());
    }
}
