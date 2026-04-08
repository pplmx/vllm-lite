mod speculative;

use crate::beam::BeamSequence;
use crate::error::Result;
use crate::metrics::MetricsCollector;
use crate::scheduler::engine::SchedulerEngine;
use crate::types::{EngineMessage, Request, SchedulerConfig};
use std::cell::RefCell;
use std::collections::HashMap;
use tokio::sync::mpsc;
use vllm_traits::{ModelBackend, SeqId, TokenId};

/// Core inference engine managing requests, scheduling, and model execution.
///
/// The Engine orchestrates the entire inference pipeline:
/// - Receives requests via `add_request`
/// - Schedules batches via the Scheduler
/// - Executes model forward passes
/// - Streams generated tokens back to clients
///
/// # Thread Safety
///
/// The Engine runs on its own dedicated thread (via `run`), using `RefCell`
/// for interior mutability of model references. All external communication
/// happens through mpsc channels (actor pattern).
pub struct Engine<M: ModelBackend> {
    pub scheduler: SchedulerEngine,
    pub target_model: RefCell<Box<M>>,
    pub draft_model: RefCell<Box<M>>,
    pub max_draft_tokens: usize,
    pub speculative_mode: bool,
    pub error_count: usize,
    pub last_error: Option<String>,
    pub metrics: MetricsCollector,
    pub response_txs: HashMap<SeqId, mpsc::Sender<TokenId>>,
    sleep_policy: SleepPolicy,
}

impl<M: ModelBackend> Engine<M> {
    /// Creates a new Engine with default configuration.
    ///
    /// # Arguments
    /// * `target_model` - The primary model for inference
    /// * `draft_model` - The draft model for speculative decoding
    ///
    /// # Example
    /// ```ignore
    /// let engine = Engine::new(my_model, draft_model);
    /// ```
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
            scheduler: SchedulerEngine::new(config, num_kv_blocks),
            target_model: RefCell::new(Box::new(target_model)),
            draft_model: RefCell::new(Box::new(draft_model)),
            max_draft_tokens,
            speculative_mode: false,
            error_count: 0,
            last_error: None,
            metrics: MetricsCollector::new(),
            response_txs: HashMap::with_capacity(max_seqs),
            sleep_policy: SleepPolicy::default(),
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

    pub fn cancel_request(&mut self, seq_id: SeqId) -> bool {
        let canceled = self.scheduler.cancel_request(seq_id);
        if canceled {
            self.response_txs.remove(&seq_id);
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
                        let (used, total) = self.scheduler.get_kv_cache_usage();
                        self.metrics.record_kv_cache_usage(used, total);
                        let snapshot = self.metrics.snapshot();
                        let _ = response_tx.send(snapshot);
                    }
                    EngineMessage::GetEmbeddings {
                        input_tokens,
                        response_tx,
                    } => {
                        let positions: Vec<Vec<usize>> = input_tokens
                            .iter()
                            .map(|tokens| (0..tokens.len()).collect())
                            .collect();
                        match self
                            .target_model
                            .borrow_mut()
                            .embed(&input_tokens, &positions)
                        {
                            Ok(embeddings) => {
                                let _ = response_tx.send(embeddings);
                            }
                            Err(e) => {
                                eprintln!("Embeddings error: {}", e);
                            }
                        }
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

            let has_pending = self.scheduler.has_pending();
            let interval = self.sleep_policy.next_interval(has_pending);
            std::thread::sleep(std::time::Duration::from_millis(interval));
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
            let beam = self.beam_search(&seq, beam_width, length_penalty, max_tokens)?;
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

                let logits = self.target_model.borrow_mut().forward_logits(
                    &[0],
                    &[vec![beam.tokens.last().copied().unwrap_or(0)]],
                    &[vec![beam.tokens.len()]],
                    &[vec![beam.kv_blocks.last().copied().unwrap_or(0)]],
                    &[beam.tokens.len()],
                    &[false],
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
                sb.partial_cmp(&sa)
                    .unwrap_or_else(|| sa.is_nan().cmp(&sb.is_nan()))
            });

            beams = all_candidates.into_iter().take(beam_width).collect();
        }

        Ok(beams
            .into_iter()
            .max_by(|a, b| {
                let sa = a.score / (a.tokens.len() as f32).powf(length_penalty);
                let sb = b.score / (b.tokens.len() as f32).powf(length_penalty);
                sa.partial_cmp(&sb)
                    .unwrap_or_else(|| sa.is_nan().cmp(&sb.is_nan()))
            })
            .unwrap())
    }

    fn get_top_k(&self, logits: &[f32], k: usize) -> Vec<(TokenId, f32)> {
        let k = k.min(logits.len());
        let mut indexed: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or_else(|| a.1.is_nan().cmp(&b.1.is_nan()))
        });
        indexed
            .into_iter()
            .take(k)
            .map(|(i, v)| (i as TokenId, v))
            .collect()
    }
}

pub struct SleepPolicy {
    pub base_interval: u64,
    pub max_interval: u64,
    pub backoff_factor: f64,
    pub consecutive_idle: u32,
}

impl Default for SleepPolicy {
    fn default() -> Self {
        Self {
            base_interval: 1,
            max_interval: 50,
            backoff_factor: 1.5,
            consecutive_idle: 0,
        }
    }
}

impl SleepPolicy {
    pub fn next_interval(&mut self, has_work: bool) -> u64 {
        if has_work {
            self.consecutive_idle = 0;
            return self.base_interval;
        }

        self.consecutive_idle += 1;

        if self.consecutive_idle == 1 {
            return self.base_interval;
        }

        ((self.base_interval as f64) * self.backoff_factor.powi(self.consecutive_idle as i32 - 1))
            .min(self.max_interval as f64) as u64
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
            &mut self,
            seq_ids: &[SeqId],
            _input_tokens: &[Vec<TokenId>],
            _positions: &[Vec<usize>],
            _kv_block_ids: &[Vec<usize>],
            _num_computed_tokens: &[usize],
            _is_prefill: &[bool],
        ) -> Result<BatchOutput> {
            Ok(BatchOutput {
                seq_ids: seq_ids.to_vec(),
                next_tokens: seq_ids.iter().map(|_| self.token_to_return).collect(),
            })
        }

        fn forward_logits(
            &mut self,
            _seq_ids: &[SeqId],
            input_tokens: &[Vec<TokenId>],
            _positions: &[Vec<usize>],
            _kv_block_ids: &[Vec<usize>],
            _num_computed_tokens: &[usize],
            _is_prefill: &[bool],
        ) -> Result<Vec<Vec<f32>>> {
            Ok(input_tokens
                .iter()
                .map(|tokens| tokens.iter().map(|_| 0.0).collect())
                .collect())
        }

        fn embed(
            &mut self,
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
        let (tx, mut rx) = mpsc::channel(64);

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
        let (tx1, mut rx1) = mpsc::channel(64);
        let (tx2, mut rx2) = mpsc::channel(64);

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

    #[test]
    fn test_engine_max_draft_tokens_config() {
        let stub = StubModel {
            token_to_return: 42,
        };
        let config = SchedulerConfig {
            max_num_seqs: 10,
            max_num_batched_tokens: 100,
            max_consecutive_decode: 10,
            enable_pd_separation: true,
            prefill_chunk_size: 512,
            decode_preference_ratio: 0.7,
            enable_priority_scheduling: false,
            enable_dynamic_batching: false,
            min_batch_size: 1,
            max_batch_size: 256,
        };
        let engine = Engine::with_config(stub.clone(), stub, config, 8, 1024);
        assert_eq!(engine.max_draft_tokens, 8);
    }

    #[test]
    fn test_engine_error_tracking() {
        let stub = StubModel {
            token_to_return: 42,
        };
        let mut engine = Engine::new(stub.clone(), stub);
        let (tx, _rx) = mpsc::channel(64);
        engine.add_request(Request::new(1, vec![10], 3), tx);

        let _ = engine.step();

        assert_eq!(engine.error_count, 0);
    }

    #[test]
    fn test_engine_response_channel_cleanup() {
        let stub = StubModel {
            token_to_return: 42,
        };
        let mut engine = Engine::new(stub.clone(), stub);
        let (tx1, _rx1) = mpsc::channel(64);
        let (tx2, _rx2) = mpsc::channel(64);

        engine.add_request(Request::new(1, vec![10], 1), tx1);
        engine.add_request(Request::new(2, vec![20], 1), tx2);

        for _ in 0..3 {
            let _ = engine.step();
        }

        assert!(!engine.has_pending());
    }

    #[test]
    fn test_sleep_policy_immediate_work() {
        let mut policy = SleepPolicy::default();
        let interval = policy.next_interval(true);
        assert_eq!(interval, 1);
        assert_eq!(policy.consecutive_idle, 0);
    }

    #[test]
    fn test_sleep_policy_exponential_backoff() {
        let mut policy = SleepPolicy::default();

        let _ = policy.next_interval(false);
        assert_eq!(policy.consecutive_idle, 1);

        let interval2 = policy.next_interval(false);
        assert_eq!(policy.consecutive_idle, 2);

        let interval3 = policy.next_interval(false);
        assert!(interval3 >= interval2);

        let interval4 = policy.next_interval(true);
        assert_eq!(interval4, 1);
    }

    #[test]
    fn test_sleep_policy_max_interval() {
        let mut policy = SleepPolicy::default();

        for _ in 0..100 {
            policy.next_interval(false);
        }

        let interval = policy.next_interval(false);
        assert!(interval <= policy.max_interval);
    }
}
