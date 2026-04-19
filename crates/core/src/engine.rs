mod speculative;

use crate::beam::BeamSequence;
use crate::error::Result;
use crate::metrics::{EnhancedMetricsCollector, MetricsCollector};
use crate::scheduler::engine::SchedulerEngine;
use crate::speculative::AdaptiveSpeculativeDecoder;
use crate::types::AdaptiveDraftConfig;
use crate::types::{EngineMessage, Request, SchedulerConfig};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tracing::{error, trace};
use vllm_traits::{BatchOutput, BatchPhase, ModelBackend, Result as ModelResult, SeqId, TokenId};

#[cfg(feature = "cuda-graph")]
use vllm_model::kernels::BatchCudaGraphExecutor;

#[cfg(feature = "cuda-graph")]
use vllm_traits::kernels::CudaGraphConfig;

pub struct BoxedModelBackend(Box<dyn ModelBackend>);

impl BoxedModelBackend {
    pub fn new(model: Box<dyn ModelBackend>) -> Self {
        Self(model)
    }
}

impl std::ops::Deref for BoxedModelBackend {
    type Target = dyn ModelBackend;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl std::ops::DerefMut for BoxedModelBackend {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.0
    }
}

impl ModelBackend for BoxedModelBackend {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> ModelResult<BatchOutput> {
        self.0.forward(
            seq_ids,
            input_tokens,
            positions,
            kv_block_ids,
            num_computed_tokens,
            is_prefill,
        )
    }

    fn forward_logits(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> ModelResult<Vec<Vec<f32>>> {
        self.0.forward_logits(
            seq_ids,
            input_tokens,
            positions,
            kv_block_ids,
            num_computed_tokens,
            is_prefill,
        )
    }

    fn embed(
        &mut self,
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
    ) -> ModelResult<Vec<Vec<f32>>> {
        self.0.embed(input_tokens, positions)
    }

    fn vocab_size(&self) -> usize {
        self.0.vocab_size()
    }
}

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
pub struct Engine<M: ModelBackend + 'static> {
    pub scheduler: SchedulerEngine,
    pub target_model: Arc<Mutex<dyn ModelBackend>>,
    pub draft_model: Option<Arc<Mutex<dyn ModelBackend>>>,
    pub max_draft_tokens: usize,
    pub speculative_mode: bool,
    pub error_count: usize,
    pub last_error: Option<String>,
    pub metrics: MetricsCollector,
    pub response_txs: HashMap<SeqId, mpsc::Sender<TokenId>>,
    sleep_policy: SleepPolicy,
    _phantom: PhantomData<M>,
    #[cfg(feature = "cuda-graph")]
    cuda_graph: Option<BatchCudaGraphExecutor>,
    pub adaptive_decoder: Option<AdaptiveSpeculativeDecoder>,
}

impl Engine<BoxedModelBackend> {
    pub fn new_boxed(
        target_model: Box<dyn ModelBackend>,
        draft_model: Option<Box<dyn ModelBackend>>,
    ) -> Self {
        Self::with_config_boxed(
            target_model,
            draft_model,
            SchedulerConfig::default(),
            4,
            1024,
        )
    }

    pub fn with_config_boxed(
        target_model: Box<dyn ModelBackend>,
        draft_model: Option<Box<dyn ModelBackend>>,
        config: SchedulerConfig,
        max_draft_tokens: usize,
        num_kv_blocks: usize,
    ) -> Self {
        let max_seqs = config.max_num_seqs;
        #[cfg(feature = "cuda-graph")]
        let cuda_graph = if config.cuda_graph.enabled {
            let graph_config = CudaGraphConfig {
                enabled: true,
                batch_sizes: config.cuda_graph.batch_sizes.clone(),
                ..Default::default()
            };
            match BatchCudaGraphExecutor::new(graph_config) {
                Ok(executor) => Some(executor),
                Err(e) => {
                    tracing::warn!("Failed to initialize CUDA Graph: {}", e);
                    None
                }
            }
        } else {
            None
        };
        let enhanced_metrics = Arc::new(EnhancedMetricsCollector::new());
        let draft = draft_model.map(|m| {
            Arc::new(Mutex::new(BoxedModelBackend::new(m))) as Arc<Mutex<dyn ModelBackend>>
        });
        Self {
            scheduler: SchedulerEngine::new(config, num_kv_blocks, enhanced_metrics),
            target_model: Arc::new(Mutex::new(BoxedModelBackend::new(target_model)))
                as Arc<Mutex<dyn ModelBackend>>,
            draft_model: draft,
            max_draft_tokens,
            speculative_mode: false,
            error_count: 0,
            last_error: None,
            metrics: MetricsCollector::new(),
            response_txs: HashMap::with_capacity(max_seqs),
            sleep_policy: SleepPolicy::default(),
            _phantom: PhantomData,
            #[cfg(feature = "cuda-graph")]
            cuda_graph,
            adaptive_decoder: None,
        }
    }
}

impl<M: ModelBackend + 'static> Engine<M> {
    /// Creates a new Engine with default configuration.
    ///
    /// # Arguments
    /// * `target_model` - The primary model for inference
    /// * `draft_model` - The draft model for speculative decoding
    ///
    /// # Example
    /// ```ignore
    /// let engine = Engine::new(my_model, Some(draft_model));
    /// ```
    pub fn new(target_model: M, draft_model: Option<M>) -> Self {
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
        draft_model: Option<M>,
        config: SchedulerConfig,
        max_draft_tokens: usize,
        num_kv_blocks: usize,
    ) -> Self {
        let max_seqs = config.max_num_seqs;
        #[cfg(feature = "cuda-graph")]
        let cuda_graph = if config.cuda_graph.enabled {
            let graph_config = CudaGraphConfig {
                enabled: true,
                batch_sizes: config.cuda_graph.batch_sizes.clone(),
                ..Default::default()
            };
            match BatchCudaGraphExecutor::new(graph_config) {
                Ok(executor) => Some(executor),
                Err(e) => {
                    tracing::warn!("Failed to initialize CUDA Graph: {}", e);
                    None
                }
            }
        } else {
            None
        };
        let enhanced_metrics = Arc::new(EnhancedMetricsCollector::new());
        let draft_model =
            draft_model.map(|m| Arc::new(Mutex::new(m)) as Arc<Mutex<dyn ModelBackend>>);
        Self {
            scheduler: SchedulerEngine::new(config, num_kv_blocks, enhanced_metrics),
            target_model: Arc::new(Mutex::new(target_model)),
            draft_model,
            max_draft_tokens,
            speculative_mode: false,
            error_count: 0,
            last_error: None,
            metrics: MetricsCollector::new(),
            response_txs: HashMap::with_capacity(max_seqs),
            sleep_policy: SleepPolicy::default(),
            _phantom: PhantomData,
            #[cfg(feature = "cuda-graph")]
            cuda_graph,
            adaptive_decoder: None,
        }
    }

    #[cfg(feature = "cuda-graph")]
    pub fn capture_cuda_graphs(&mut self) -> crate::error::Result<()> {
        if let Some(ref mut executor) = self.cuda_graph {
            executor
                .capture_all_graphs()
                .map_err(|e| crate::error::EngineError::ModelError(e.to_string()))?;
            tracing::info!("CUDA Graphs captured successfully");
        }
        Ok(())
    }

    #[cfg(not(feature = "cuda-graph"))]
    pub fn capture_cuda_graphs(&mut self) -> crate::error::Result<()> {
        tracing::warn!("CUDA Graph support not enabled");
        Ok(())
    }

    #[cfg(feature = "cuda-graph")]
    pub fn cuda_graph_enabled(&self) -> bool {
        self.cuda_graph.as_ref().is_some_and(|e| e.is_enabled())
    }

    #[cfg(not(feature = "cuda-graph"))]
    pub fn cuda_graph_enabled(&self) -> bool {
        false
    }

    pub fn enable_speculative(&mut self) {
        self.speculative_mode = true;
    }

    /// Enable adaptive speculative decoding
    pub fn enable_adaptive_speculative(&mut self, config: AdaptiveDraftConfig) {
        self.adaptive_decoder = Some(AdaptiveSpeculativeDecoder::new(config));
        self.speculative_mode = true;
    }

    /// Disable adaptive speculative decoding
    pub fn disable_adaptive_speculative(&mut self) {
        self.adaptive_decoder = None;
        self.speculative_mode = false;
    }

    /// Check if adaptive speculative is enabled
    pub fn is_adaptive_speculative_enabled(&self) -> bool {
        self.adaptive_decoder.is_some()
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
        let mut step_count = 0u64;
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
                            .lock()
                            .unwrap()
                            .embed(&input_tokens, &positions)
                        {
                            Ok(embeddings) => {
                                let _ = response_tx.send(embeddings);
                            }
                            Err(e) => {
                                error!(error = %e, "Embeddings error");
                            }
                        }
                    }
                    EngineMessage::Shutdown => return,
                }
            }

            if self.scheduler.has_pending() {
                step_count += 1;
                let result = if self.adaptive_decoder.is_some() {
                    self.step_adaptive_speculative()
                } else if self.speculative_mode {
                    self.step_speculative()
                } else if self.cuda_graph_enabled() {
                    self.step_with_graph()
                } else {
                    self.step()
                };
                if let Err(e) = result {
                    self.error_count += 1;
                    self.last_error = Some(e.to_string());
                    error!(step = step_count, error = %e, "Engine step error");
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

                let logits = self.target_model.lock().unwrap().forward_logits(
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

    #[cfg(feature = "cuda-graph")]
    pub fn step_with_graph(&mut self) -> Result<Vec<(SeqId, TokenId)>> {
        let start = std::time::Instant::now();
        let graph_batch = self.scheduler.build_batch_with_graph();
        if graph_batch.batch_size() == 0 {
            return Ok(vec![]);
        }

        let (output, batch) = match graph_batch {
            crate::scheduler::GraphBatch::Graph(prepared) => {
                let batch = prepared.batch;
                let input_counts: Vec<usize> = batch.input_tokens.iter().map(|v| v.len()).collect();
                let output = if let Some(ref executor) = self.cuda_graph {
                    match executor.execute(&batch) {
                        Ok(output) => output,
                        Err(e) => {
                            tracing::warn!("CUDA Graph execution failed: {}, falling back", e);
                            self.execute_regular(&batch)?
                        }
                    }
                } else {
                    self.execute_regular(&batch)?
                };
                (output, input_counts)
            }
            crate::scheduler::GraphBatch::Regular(batch) => {
                let input_counts: Vec<usize> = batch.input_tokens.iter().map(|v| v.len()).collect();
                let output = self.execute_regular(&batch)?;
                (output, input_counts)
            }
        };

        self.process_output(output, batch, start)
    }

    #[cfg(not(feature = "cuda-graph"))]
    pub fn step_with_graph(&mut self) -> Result<Vec<(SeqId, TokenId)>> {
        tracing::warn!("CUDA Graph support not enabled, using regular step");
        self.step()
    }

    /// Execute regular forward pass (existing logic)
    fn execute_regular(&mut self, batch: &vllm_traits::Batch) -> Result<BatchOutput> {
        let total_tokens: usize = batch.input_tokens.iter().map(|t| t.len()).sum();
        tracing::debug!(
            batch_size = batch.seq_ids.len(),
            total_tokens = total_tokens,
            is_prefill = matches!(batch.phase, BatchPhase::Prefill),
            "Model forward started"
        );

        let start = std::time::Instant::now();
        let result = {
            let mut model = self.target_model.lock().unwrap();
            model.forward(
                &batch.seq_ids,
                &batch.input_tokens,
                &batch.positions,
                &batch.kv_block_ids,
                &batch.num_computed_tokens,
                &batch.is_prefill,
            )
        };
        let elapsed = start.elapsed().as_millis() as u64;

        match result {
            Ok(output) => {
                tracing::debug!(
                    elapsed_ms = elapsed,
                    tokens = output.next_tokens.len(),
                    "Model forward completed"
                );
                Ok(output)
            }
            Err(e) => {
                tracing::error!(error = %e, "Model forward failed");
                Err(crate::error::EngineError::ModelError(e.to_string()))
            }
        }
    }

    /// Process model output and update state
    fn process_output(
        &mut self,
        output: BatchOutput,
        input_counts: Vec<usize>,
        start: std::time::Instant,
    ) -> Result<Vec<(SeqId, TokenId)>> {
        tracing::debug!(
            seq_ids = ?output.seq_ids,
            tokens = ?output.next_tokens,
            "process_output: received model output"
        );

        let mut results = Vec::new();
        for (seq_id, token) in output.seq_ids.iter().zip(&output.next_tokens) {
            trace!(
                seq_id = %seq_id,
                token_id = %token,
                "Token generated"
            );
            tracing::debug!(seq_id = %seq_id, token = %token, "Sending token via channel");
            if let Some(tx) = self.response_txs.get(seq_id) {
                let _ = tx.try_send(*token);
            }
            results.push((*seq_id, *token));
        }

        let seq_ids: Vec<_> = results.iter().map(|(id, _)| *id).collect();
        let tokens: Vec<_> = results.iter().map(|(_, t)| *t).collect();
        self.scheduler.update(&seq_ids, &tokens, &input_counts);

        let finished = self.scheduler.finished_sequences();
        for seq in &finished {
            if let Some(tx) = self.response_txs.remove(&seq.id) {
                drop(tx);
            }
        }
        self.scheduler.clear_finished();

        // Record metrics
        if !results.is_empty() {
            self.metrics.record_tokens(results.len() as u64);
            self.metrics.record_batch_size(results.len());
            let elapsed = start.elapsed().as_millis() as f64;
            if elapsed > 0.0 {
                self.metrics.record_latency(elapsed);
            }
        }

        Ok(results)
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

        fn vocab_size(&self) -> usize {
            151936
        }
    }

    #[test]
    fn test_engine_streaming() {
        let stub = StubModel {
            token_to_return: 42,
        };
        let mut engine = Engine::new(stub.clone(), None);
        let (tx, mut rx) = mpsc::channel(64);

        engine.add_request(Request::new(1, vec![10, 20], 5), tx);

        // First step: prefill, should return at least 1 output (the generated token)
        let out = engine.step().unwrap();
        assert!(!out.is_empty());
        assert_eq!(rx.try_recv().unwrap(), 42);

        // Keep stepping until done
        let mut steps = 0;
        while engine.has_pending() && steps < 10 {
            let out = engine.step().unwrap();
            if !out.is_empty() {
                assert_eq!(out[0], (1, 42));
                assert_eq!(rx.try_recv().unwrap(), 42);
            }
            steps += 1;
        }

        assert!(
            !engine.has_pending(),
            "Sequence should finish after max_tokens"
        );
    }

    #[test]
    fn test_engine_multi_request() {
        let stub = StubModel {
            token_to_return: 10,
        };
        let mut engine = Engine::new(stub.clone(), None);
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
        let mut engine = Engine::new(stub.clone(), None);
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
            ..Default::default()
        };
        let engine = Engine::with_config(stub.clone(), None, config, 8, 1024);
        assert_eq!(engine.max_draft_tokens, 8);
    }

    #[test]
    fn test_engine_error_tracking() {
        let stub = StubModel {
            token_to_return: 42,
        };
        let mut engine = Engine::new(stub.clone(), None);
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
        let mut engine = Engine::new(stub.clone(), None);
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
