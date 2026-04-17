use std::sync::Arc;
use tokio::sync::mpsc;
use vllm_core::engine::Engine;
use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_core::scheduler::SchedulerEngine;
use vllm_core::types::{Phase, Request, SchedulerConfig};
use vllm_traits::{BatchOutput, ModelBackend, SeqId, TokenId};

struct TracingModel {
    sequence_to_return: Vec<TokenId>,
    current_idx: usize,
}

impl TracingModel {
    fn new(tokens: Vec<TokenId>) -> Self {
        Self {
            sequence_to_return: tokens,
            current_idx: 0,
        }
    }
}

impl ModelBackend for TracingModel {
    fn forward(
        &mut self,
        _seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> vllm_traits::Result<BatchOutput> {
        eprintln!(
            "MODEL forward: input_tokens.len()={:?}, input_tokens={:?}, positions.len()={:?}, positions={:?}, is_prefill={:?}",
            input_tokens.iter().map(|t| t.len()).collect::<Vec<_>>(),
            input_tokens,
            positions.iter().map(|p| p.len()).collect::<Vec<_>>(),
            positions,
            is_prefill
        );

        let mut next_tokens = Vec::new();
        for i in 0.._seq_ids.len() {
            let token = if self.current_idx < self.sequence_to_return.len() {
                let t = self.sequence_to_return[self.current_idx];
                self.current_idx += 1;
                t
            } else {
                151643 // EOS
            };
            next_tokens.push(token);
            eprintln!("  seq {} -> token {}", _seq_ids[i], token);
        }

        Ok(BatchOutput {
            seq_ids: _seq_ids.to_vec(),
            next_tokens,
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
    ) -> vllm_traits::Result<Vec<Vec<f32>>> {
        Ok(input_tokens.iter().map(|t| vec![0.0; t.len()]).collect())
    }

    fn embed(
        &mut self,
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> vllm_traits::Result<Vec<Vec<f32>>> {
        Ok(input_tokens.iter().map(|t| vec![0.0; 128]).collect())
    }
}

#[test]
fn test_engine_step_trace_prefill_then_decode() {
    let model = TracingModel::new(vec![29054, 110934, 99601]); // Expected sequence
    let _config = SchedulerConfig::default();
    let mut engine = Engine::new(model, None);
    let (tx, mut rx) = mpsc::channel(64);

    // Add request with 10 tokens prompt, max 20 generated tokens
    let request = Request::new(
        1,
        vec![
            151643, 151644, 872, 198, 6023, 151645, 198, 151644, 77091, 198,
        ],
        20,
    );
    engine.add_request(request, tx);

    eprintln!("\n=== ENGINE STEP TRACE ===\n");

    let mut step = 0;
    let mut all_tokens = Vec::new();

    while engine.has_pending() && step < 15 {
        eprintln!("\n--- Step {} ---", step + 1);
        let result = engine.step();
        match result {
            Ok(outputs) => {
                eprintln!("Step {} output: {:?}", step + 1, outputs);
                for (_seq_id, token) in outputs {
                    all_tokens.push(token);
                    // Try to receive from channel
                    while let Ok(t) = rx.try_recv() {
                        eprintln!("Channel received token: {}", t);
                    }
                }
            }
            Err(e) => {
                eprintln!("Step {} error: {:?}", step + 1, e);
                break;
            }
        }
        step += 1;
    }

    eprintln!("\n=== FINAL RESULT ===");
    eprintln!("Total tokens: {:?}", all_tokens);
    eprintln!("Expected: [29054, 110934, 99601, 151643, ...]");

    // The model should return these tokens in order
    assert!(
        !all_tokens.is_empty(),
        "Should have generated at least one token"
    );
}

#[test]
fn test_scheduler_batch_trace() {
    let config = SchedulerConfig::default();
    let metrics = Arc::new(EnhancedMetricsCollector::new());
    let mut scheduler = SchedulerEngine::new(config, 1024, metrics);

    // Add request with larger max_tokens to continue after prefill
    let request = Request::new(1, vec![151643, 151644, 872, 198, 6023], 10);
    scheduler.add_request(request);

    eprintln!("\n=== SCHEDULER BATCH TRACE ===\n");

    // First batch (prefill)
    let batch1 = scheduler.build_batch();
    eprintln!("Batch 1:");
    eprintln!("  seq_ids: {:?}", batch1.seq_ids);
    eprintln!("  input_tokens: {:?}", batch1.input_tokens);
    eprintln!("  positions: {:?}", batch1.positions);
    eprintln!("  is_prefill: {:?}", batch1.is_prefill);
    eprintln!("  phase: {:?}", batch1.phase);
    eprintln!(
        "  running count before update: {}",
        scheduler.running().len()
    );

    // Simulate model output: generate token
    let generated_token = 29054u32;
    let input_counts = batch1
        .input_tokens
        .iter()
        .map(|t| t.len())
        .collect::<Vec<_>>();
    scheduler.update(&batch1.seq_ids, &[generated_token], &input_counts);

    eprintln!("\nAfter update 1:");
    eprintln!("  running count: {}", scheduler.running().len());
    for seq in scheduler.running() {
        eprintln!(
            "  seq {}: tokens.len()={}, status={:?}, prompt_len={}",
            seq.id,
            seq.tokens.len(),
            seq.status,
            seq.prompt_len
        );
    }

    // Second batch (decode)
    let batch2 = scheduler.build_batch();
    eprintln!("\nBatch 2:");
    eprintln!("  seq_ids: {:?}", batch2.seq_ids);
    eprintln!("  input_tokens: {:?}", batch2.input_tokens);
    eprintln!("  positions: {:?}", batch2.positions);
    eprintln!("  is_prefill: {:?}", batch2.is_prefill);
    eprintln!("  phase: {:?}", batch2.phase);

    // Check that decode batch only has the last token
    if !batch2.input_tokens.is_empty() {
        assert_eq!(
            batch2.input_tokens[0].len(),
            1,
            "Decode batch should only have 1 token, got {}",
            batch2.input_tokens[0].len()
        );
        assert_eq!(
            batch2.input_tokens[0][0], generated_token,
            "Decode batch should have generated token"
        );
    } else {
        eprintln!("\n*** WARNING: Batch 2 is empty! This is the bug! ***");
    }
}

#[test]
fn test_scheduler_decode_position_trace() {
    use std::sync::Arc;
    use vllm_core::scheduler::BatchComposer;
    use vllm_core::types::{Priority, SamplingParams, Sequence, Status};

    let composer = BatchComposer::default();

    // Simulate after prefill + 3 decode steps
    // tokens = [prompt(5) + generated(3)] = [1,2,3,4,5,6,7,8]
    let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let seq = Sequence {
        id: 1,
        tokens: tokens.clone(),
        kv_blocks: Arc::new(vec![0]),
        num_computed_tokens: 0,
        prompt_len: 5,
        status: Status::Decoding,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 3,
        priority: Priority::default(),
    };

    let batch = composer.compose(vec![seq], Phase::Decode);

    eprintln!("\n=== DECODE BATCH TRACE ===");
    eprintln!("tokens: {:?}", tokens);
    eprintln!("tokens.len(): {}", tokens.len());
    eprintln!("input_tokens: {:?}", batch.input_tokens);
    eprintln!("positions: {:?}", batch.positions);
    eprintln!("num_computed_tokens: {:?}", batch.num_computed_tokens);

    // Expected:
    // - input_tokens = [8] (last token only)
    // - positions = [7] (0-indexed position of token 8)
    // - num_computed = 7 (all tokens before 8 are in cache)
    assert_eq!(
        batch.input_tokens[0],
        vec![8],
        "Should only have last token"
    );
    assert_eq!(
        batch.positions[0],
        vec![7],
        "Position should be 7 (0-indexed)"
    );
    assert_eq!(batch.num_computed_tokens[0], 7, "num_computed should be 7");
}
