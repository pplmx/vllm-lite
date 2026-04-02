mod batch;
mod speculative;

use crate::beam::BeamSequence;
use crate::error::Result;
use crate::metrics::MetricsCollector;
use crate::scheduler::Scheduler;
use crate::types::{EngineMessage, Request, SchedulerConfig};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use vllm_traits::{ModelBackend, SeqId, TokenId};

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
        let max_seqs = config.max_num_seqs;
        Self {
            scheduler: Scheduler::with_config(config, num_kv_blocks),
            target_model: Arc::new(target_model),
            draft_model: Arc::new(draft_model),
            max_draft_tokens,
            speculative_mode: false,
            error_count: 0,
            last_error: None,
            metrics: MetricsCollector::new(),
            response_txs: HashMap::with_capacity(max_seqs),
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

    pub fn step_beam(
        &mut self,
        beam_width: usize,
        length_penalty: f32,
        max_tokens: usize,
    ) -> Result<Vec<BeamSequence>> {
        let _batch = self.scheduler.build_batch();

        let mut results = Vec::new();
        for seq in self.scheduler.running() {
            let beam = self.beam_search(seq, beam_width, length_penalty, max_tokens)?;
            results.push(beam);
        }

        Ok(results)
    }

    fn beam_search(
        &self,
        initial: &crate::types::Sequence,
        beam_width: usize,
        length_penalty: f32,
        max_tokens: usize,
    ) -> Result<BeamSequence> {
        let mut beams = vec![BeamSequence::new(
            initial.tokens.clone(),
            0.0,
            initial.kv_blocks.as_ref().clone(),
        )];

        for _ in 0..max_tokens {
            let mut all_candidates = Vec::new();

            for beam in &beams {
                if beam.tokens.is_empty() {
                    continue;
                }

                let logits = self.target_model.forward_logits(
                    &[0],
                    &[vec![beam.tokens.last().copied().unwrap_or(0)]],
                    &[vec![beam.tokens.len()]],
                )?;

                let top_k = self.get_top_k(&logits[0], beam_width);

                for (token, log_prob) in top_k {
                    let mut new_tokens = beam.tokens.clone();
                    new_tokens.push(token);
                    all_candidates.push(BeamSequence::new(
                        new_tokens,
                        beam.score + log_prob,
                        beam.kv_blocks.as_ref().clone(),
                    ));
                }
            }

            if all_candidates.is_empty() {
                break;
            }

            all_candidates.sort_by(|a, b| {
                let sa = a.score / (a.tokens.len() as f32).powf(length_penalty);
                let sb = b.score / (b.tokens.len() as f32).powf(length_penalty);
                sb.partial_cmp(&sa).unwrap()
            });

            beams = all_candidates.into_iter().take(beam_width).collect();
        }

        Ok(beams
            .into_iter()
            .max_by(|a, b| {
                let sa = a.score / (a.tokens.len() as f32).powf(length_penalty);
                let sb = b.score / (b.tokens.len() as f32).powf(length_penalty);
                sa.partial_cmp(&sb).unwrap()
            })
            .unwrap())
    }

    fn get_top_k(&self, logits: &[f32], k: usize) -> Vec<(TokenId, f32)> {
        let k = k.min(logits.len());
        let mut indexed: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed
            .into_iter()
            .take(k)
            .map(|(i, v)| (i as TokenId, v))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Request;
    use tokio::sync::mpsc;
    use vllm_traits::{BatchOutput, Result, TokenId};

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

        fn forward_logits(
            &self,
            _seq_ids: &[SeqId],
            input_tokens: &[Vec<TokenId>],
            _positions: &[Vec<usize>],
        ) -> Result<Vec<Vec<f32>>> {
            Ok(input_tokens
                .iter()
                .map(|tokens| tokens.iter().map(|_| 0.0).collect())
                .collect())
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

        engine.step().unwrap();
        assert_eq!(rx1.try_recv().unwrap(), 10);
        assert_eq!(rx2.try_recv().unwrap(), 10);

        engine.step().unwrap();
        assert_eq!(rx1.try_recv().unwrap(), 10);
        assert_eq!(rx2.try_recv().unwrap(), 10);

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
