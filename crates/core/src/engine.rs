use crate::error::Result;
use crate::scheduler::Scheduler;
use crate::types::{BatchOutput, Request, SeqId, TokenId};

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
}

impl<M: ModelBackend> Engine<M> {
    pub fn new(model: M) -> Self {
        Self {
            scheduler: Scheduler::new(),
            model,
        }
    }

    pub fn add_request(&mut self, req: Request) -> SeqId {
        self.scheduler.add_request(req)
    }

    pub fn step(&mut self) -> Result<Vec<(SeqId, TokenId)>> {
        let batch = self.scheduler.build_batch();
        if batch.is_empty() {
            return Ok(vec![]);
        }

        let output = self
            .model
            .forward(&batch.seq_ids, &batch.input_tokens, &batch.positions)?;

        self.scheduler.update(
            &batch.seq_ids,
            &output.next_tokens,
            &batch
                .input_tokens
                .iter()
                .map(|t| t.len())
                .collect::<Vec<_>>(),
        );
        Ok(batch.seq_ids.into_iter().zip(output.next_tokens).collect())
    }

    pub fn has_pending(&self) -> bool {
        self.scheduler.has_pending()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Request;

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
    fn test_engine_single_request() {
        let mut engine = Engine::new(StubModel {
            token_to_return: 42,
        });
        engine.add_request(Request::new(1, vec![10, 20], 5));

        // Step 1: prefill
        let out = engine.step().unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], (1, 42));

        // Step 2: decode
        let out = engine.step().unwrap();
        assert_eq!(out[0], (1, 42));

        // Step 3: decode → reaches max_tokens=5
        let out = engine.step().unwrap();
        assert_eq!(out[0], (1, 42));

        // Step 4: no more pending
        assert!(!engine.has_pending());
    }
}
