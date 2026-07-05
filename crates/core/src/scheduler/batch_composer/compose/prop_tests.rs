//! Property-based tests for [`super::BatchComposer`].
//!
//! Each property verifies an invariant the composer must hold for every
//! valid input — not just for hand-picked examples. Lives in a sibling
//! file (under `compose/`) so the proptest dependency does not bleed
//! into the unit-test compilation unit.

use super::*;
use crate::types::{Priority, SamplingParams, Status};
use proptest::prelude::*;
use std::sync::Arc;

fn make_sequence(id: u64, tokens: Vec<u32>, status: Status) -> Sequence {
    Sequence {
        id,
        tokens,
        kv_blocks: Arc::new(vec![usize::try_from(id).expect("bounded test seq id")]),
        num_computed_tokens: 0,
        prompt_len: 3,
        status,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 0,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    }
}

/// Strategy that produces a bounded `BatchCompositionConfig`.
fn arb_config() -> impl Strategy<Value = BatchCompositionConfig> {
    (
        1usize..64,    // max_batch_size
        1usize..4096,  // max_token_budget
        any::<bool>(), // enable_similarity_grouping
    )
        .prop_map(
            |(max_batch_size, max_token_budget, enable_similarity_grouping)| {
                BatchCompositionConfig {
                    max_batch_size,
                    max_token_budget,
                    enable_similarity_grouping,
                }
            },
        )
}

/// Strategy that produces a single `Sequence` with bounded dimensions.
fn arb_sequence() -> impl Strategy<Value = Sequence> {
    (
        0u64..10_000, // id
        0usize..512,  // prompt length (tokens)
        1usize..128,  // max_tokens
    )
        .prop_map(|(id, prompt_len, max_tokens)| {
            let mut seq = make_sequence(
                id,
                vec![0u32; prompt_len],
                if prompt_len == 0 {
                    Status::Decoding
                } else {
                    Status::Waiting
                },
            );
            seq.max_tokens = max_tokens;
            seq
        })
}

/// Strategy that produces a `Vec<Sequence>` with no duplicate ids.
fn arb_unique_sequences(max_len: usize) -> impl Strategy<Value = Vec<Sequence>> {
    proptest::collection::vec(arb_sequence(), 0..=max_len).prop_map(|mut seqs| {
        seqs.sort_by_key(|s| s.id);
        seqs.dedup_by_key(|s| s.id);
        seqs
    })
}

proptest! {
    /// `batch.seq_ids.len() <= config.max_batch_size` for any input.
    #[test]
    fn prop_batch_size_bounded(
        cfg in arb_config(),
        seqs in arb_unique_sequences(50),
    ) {
        let composer = BatchComposer::new(cfg.clone());
        let batch = composer.compose(seqs.clone(), Phase::Decode);
        prop_assert!(
            batch.seq_ids.len() <= cfg.max_batch_size,
            "batch size {} exceeds max_batch_size {}",
            batch.seq_ids.len(),
            cfg.max_batch_size
        );

        let prefill_batch = composer.compose(
            seqs.into_iter().map(|mut s| {
                s.status = Status::Waiting;
                s
            }).collect(),
            Phase::Prefill,
        );
        prop_assert!(
            prefill_batch.seq_ids.len() <= cfg.max_batch_size,
            "prefill batch size {} exceeds max_batch_size {}",
            prefill_batch.seq_ids.len(),
            cfg.max_batch_size
        );
    }

    /// Prefill: `batch.total_tokens <= config.max_token_budget`.
    #[test]
    fn prop_prefill_token_budget_respected(
        cfg in arb_config(),
        seqs in arb_unique_sequences(20),
    ) {
        let composer = BatchComposer::new(cfg.clone());
        let batch = composer.compose(
            seqs.into_iter().map(|mut s| {
                s.status = Status::Waiting;
                s
            }).collect(),
            Phase::Prefill,
        );
        prop_assert!(
            batch.total_tokens <= cfg.max_token_budget,
            "batch total_tokens {} exceeds max_token_budget {}",
            batch.total_tokens,
            cfg.max_token_budget
        );
    }

    /// All parallel Vec fields in the batch have the same length.
    #[test]
    fn prop_batch_internal_consistency(
        cfg in arb_config(),
        seqs in arb_unique_sequences(30),
    ) {
        let composer = BatchComposer::new(cfg);

        for phase in [Phase::Decode, Phase::Prefill] {
            let input: Vec<Sequence> = seqs.iter().cloned().map(|mut s| {
                s.status = match phase {
                    Phase::Decode => Status::Decoding,
                    Phase::Prefill => Status::Waiting,
                };
                s
            }).collect();
            let batch = composer.compose(input, phase);

            let n = batch.seq_ids.len();
            prop_assert_eq!(batch.input_tokens.len(), n, "input_tokens length mismatch");
            prop_assert_eq!(batch.positions.len(), n, "positions length mismatch");
            prop_assert_eq!(batch.kv_block_ids.len(), n, "kv_block_ids length mismatch");
            prop_assert_eq!(batch.num_computed_tokens.len(), n, "num_computed_tokens length mismatch");
            prop_assert_eq!(batch.is_prefill.len(), n, "is_prefill length mismatch");
            prop_assert_eq!(batch.len(), n, "batch.len() mismatch");
        }
    }

    /// Decode: each sequence contributes exactly one token, so
    /// `total_tokens == seq_ids.len()` and every `input_tokens[i]` has length 1.
    #[test]
    fn prop_decode_one_token_per_seq(
        cfg in arb_config(),
        seqs in arb_unique_sequences(40),
    ) {
        let composer = BatchComposer::new(cfg);
        let input: Vec<Sequence> = seqs.into_iter().map(|mut s| {
            s.status = Status::Decoding;
            s
        }).collect();
        let batch = composer.compose(input, Phase::Decode);

        prop_assert_eq!(
            batch.total_tokens,
            batch.seq_ids.len(),
            "decode total_tokens should equal seq_ids.len()"
        );
        for (i, tokens) in batch.input_tokens.iter().enumerate() {
            prop_assert_eq!(
                tokens.len(),
                1,
                "decode input_tokens[{}] should have length 1, got {}",
                i,
                tokens.len()
            );
        }
    }

    /// Prefill: `total_tokens == sum(input_tokens[i].len())`.
    #[test]
    fn prop_prefill_total_tokens_matches_input(
        cfg in arb_config(),
        seqs in arb_unique_sequences(15),
    ) {
        let composer = BatchComposer::new(cfg);
        let input: Vec<Sequence> = seqs.into_iter().map(|mut s| {
            s.status = Status::Waiting;
            s
        }).collect();
        let batch = composer.compose(input, Phase::Prefill);

        let actual: usize = batch.input_tokens.iter().map(Vec::len).sum();
        prop_assert_eq!(
            batch.total_tokens,
            actual,
            "prefill total_tokens {} != sum of input_tokens lengths {}",
            batch.total_tokens,
            actual
        );
    }

    /// `compose` is deterministic: identical input → identical output.
    #[test]
    fn prop_compose_deterministic(
        cfg in arb_config(),
        seqs in arb_unique_sequences(30),
        phase in prop_oneof![Just(Phase::Decode), Just(Phase::Prefill)],
    ) {
        let composer = BatchComposer::new(cfg);
        let input: Vec<Sequence> = seqs.into_iter().map(|mut s| {
            s.status = match phase {
                Phase::Decode => Status::Decoding,
                Phase::Prefill => Status::Waiting,
            };
            s
        }).collect();

        let b1 = composer.compose(input.clone(), phase);
        let b2 = composer.compose(input, phase);

        prop_assert_eq!(b1.seq_ids, b2.seq_ids);
        prop_assert_eq!(b1.input_tokens, b2.input_tokens);
        prop_assert_eq!(b1.total_tokens, b2.total_tokens);
        prop_assert_eq!(b1.positions, b2.positions);
        prop_assert_eq!(b1.is_prefill, b2.is_prefill);
        prop_assert_eq!(b1.phase, b2.phase);
    }

    /// `batch.seq_ids` contains no duplicates.
    #[test]
    fn prop_batch_seq_ids_unique(
        cfg in arb_config(),
        seqs in arb_unique_sequences(40),
    ) {
        let composer = BatchComposer::new(cfg);
        let batch = composer.compose(seqs, Phase::Decode);

        let mut seen = std::collections::HashSet::new();
        for id in &batch.seq_ids {
            prop_assert!(seen.insert(*id), "duplicate seq_id {} in batch", id);
        }
    }
}
