use crate::error::Result;
use crate::metrics::MetricsCollector;
use crate::scheduler::Scheduler;
use crate::types::{BatchOutput, EngineMessage, Request, SchedulerConfig, SeqId, TokenId};
use std::collections::HashMap;
use std::sync::Arc;
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
    pub target_model: Arc<M>,
    pub draft_model: Arc<M>,
    pub max_draft_tokens: usize,
    pub speculative_mode: bool,
    pub error_count: usize,
    pub last_error: Option<String>,
    pub metrics: MetricsCollector,
    response_txs: HashMap<SeqId, mpsc::UnboundedSender<TokenId>>,
}

impl<M: ModelBackend> Engine<M> {
    pub fn new(target_model: M, draft_model: M) -> Self {
        Self::with_config(
            target_model,
            draft_model,
            SchedulerConfig::default(),
            4,
            1024,
        )
    }

    pub fn with_config(
        target_model: M,
        draft_model: M,
        config: SchedulerConfig,
        max_draft_tokens: usize,
        num_kv_blocks: usize,
    ) -> Self {
        Self {
            scheduler: Scheduler::with_config(config, num_kv_blocks),
            target_model: Arc::new(target_model),
            draft_model: Arc::new(draft_model),
            max_draft_tokens,
            speculative_mode: false,
            error_count: 0,
            last_error: None,
            metrics: MetricsCollector::new(),
            response_txs: HashMap::new(),
        }
    }

    pub fn enable_speculative(&mut self) {
        self.speculative_mode = true;
    }

    pub fn is_healthy(&self) -> bool {
        self.error_count < 10
    }

    pub fn get_last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
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
        let start = std::time::Instant::now();
        let batch = self.scheduler.build_batch();
        if batch.is_empty() {
            return Ok(vec![]);
        }

        let output =
            self.target_model
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

        // Record metrics
        if !batch.seq_ids.is_empty() {
            let total_tokens: usize = batch.input_tokens.iter().map(|t| t.len()).sum();
            self.metrics.record_tokens(total_tokens as u64);
            self.metrics.record_batch_size(batch.seq_ids.len());
        }

        let elapsed = start.elapsed().as_millis() as f64;
        if elapsed > 0.0 {
            self.metrics.record_latency(elapsed);
        }

        Ok(results)
    }

    #[allow(dead_code)]
    fn greedy_sample(logits: &[f32]) -> TokenId {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as TokenId)
            .unwrap_or(0)
    }

    pub fn step_speculative(&mut self) -> Result<Vec<(SeqId, TokenId)>> {
        let batch = self.scheduler.build_batch();
        if batch.is_empty() {
            return Ok(vec![]);
        }

        let mut draft_outputs: Vec<Vec<TokenId>> = Vec::new();
        for ((seq_id, tokens), positions) in batch
            .seq_ids
            .iter()
            .zip(batch.input_tokens.iter())
            .zip(batch.positions.iter())
        {
            #[allow(clippy::cloned_ref_to_slice_refs)]
            let mut draft = Vec::new();
            let mut current_tokens = tokens.clone();

            for _ in 0..self.max_draft_tokens {
                #[allow(clippy::cloned_ref_to_slice_refs)]
                let output = self.draft_model.forward(
                    &[*seq_id],
                    &[current_tokens.clone()],
                    &[positions.clone()],
                )?;
                let token = *output.next_tokens.first().unwrap_or(&0);
                draft.push(token);
                current_tokens.push(token);
            }
            draft_outputs.push(draft);
        }

        let target_output =
            self.target_model
                .forward(&batch.seq_ids, &batch.input_tokens, &batch.positions)?;

        let mut results = Vec::new();
        for (i, seq_id) in batch.seq_ids.iter().enumerate() {
            for &tok in &draft_outputs[i] {
                results.push((*seq_id, tok));
            }
            if let Some(&tok) = target_output.next_tokens.get(i) {
                results.push((*seq_id, tok));
            }
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
                    EngineMessage::GetMetrics { response_tx } => {
                        let snapshot = self.metrics.snapshot();
                        let _ = response_tx.send(snapshot);
                    }
                    EngineMessage::Shutdown => return,
                }
            }

            if self.scheduler.has_pending() {
                let result = if self.speculative_mode {
                    self.step_speculative()
                } else {
                    self.step()
                };
                if let Err(e) = result {
                    self.error_count += 1;
                    self.last_error = Some(e.to_string());
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

    #[derive(Clone)]
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
        let stub = StubModel {
            token_to_return: 42,
        };
        let mut engine = Engine::new(stub.clone(), stub);
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
        let stub = StubModel {
            token_to_return: 10,
        };
        let mut engine = Engine::new(stub.clone(), stub);
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

    #[test]
    fn test_engine_no_requests() {
        let stub = StubModel {
            token_to_return: 42,
        };
        let mut engine = Engine::new(stub.clone(), stub);
        let out = engine.step().unwrap();
        assert!(out.is_empty());
    }
}
