//! `BatchComposer` implementation: builds phase-specific batches (prefill, decode, mixed) from the running + waiting sequence lists.
//!
//! Runs every scheduler tick; the `compose` function is the entry
//! point and is split into prefill-composition + decode-composition
//! sub-passes with a configurable size budget per pass.

// crates/core/src/scheduler/batch_composer/compose/mod.rs
//
// `BatchComposer` implementation: builds phase-specific batches (prefill,
// decode, chunked prefill, packed prefill). Method bodies are split across
// the following submodules by phase:
// - `chunked` — `compose_chunked_prefill` (memory-bounded chunked prefill)
// - `prefill` — `compose_prefill_with_packing` + `compose_prefill_batch`
// - `decode`  — `compose_decode_batch`

use super::validate::{BatchCompositionConfig, ChunkedPrefillConfig};
use crate::types::{Phase, Sequence, SequencePackingConfig};
use vllm_traits::Batch;

mod chunked;
mod decode;
mod prefill;

#[derive(Debug)]
/// Batch composer for building phase-specific batches
pub struct BatchComposer {
    config: BatchCompositionConfig,
    packing_config: SequencePackingConfig,
    chunked_prefill: ChunkedPrefillConfig,
}

impl BatchComposer {
    /// Create a new batch composer with the given configuration
    #[must_use]
    pub fn new(config: BatchCompositionConfig) -> Self {
        Self {
            config,
            packing_config: SequencePackingConfig::default(),
            chunked_prefill: ChunkedPrefillConfig::default(),
        }
    }

    /// Create a new batch composer with custom packing configuration
    #[must_use]
    pub fn with_packing(
        config: BatchCompositionConfig,
        packing_config: SequencePackingConfig,
    ) -> Self {
        Self {
            config,
            packing_config,
            chunked_prefill: ChunkedPrefillConfig::default(),
        }
    }

    /// Create a new batch composer with chunked prefill configuration
    #[must_use]
    pub fn with_chunked_prefill(
        config: BatchCompositionConfig,
        chunked_prefill: ChunkedPrefillConfig,
    ) -> Self {
        Self {
            config,
            packing_config: SequencePackingConfig::default(),
            chunked_prefill,
        }
    }

    /// Compose batch with optional sequence packing for prefill
    #[must_use]
    pub fn compose(&self, sequences: Vec<Sequence>, phase: Phase) -> Batch {
        match phase {
            Phase::Prefill if self.packing_config.enabled && sequences.len() > 1 => {
                self.compose_prefill_with_packing(sequences)
            }
            Phase::Prefill => self.compose_prefill_batch(sequences),
            Phase::Decode => self.compose_decode_batch(sequences),
        }
    }

    /// Compose batch with chunked prefill support
    /// Returns a batch that respects memory constraints by splitting long sequences
    #[must_use]
    pub fn compose_with_chunking(
        &self,
        sequences: Vec<Sequence>,
        phase: Phase,
        available_memory: usize,
    ) -> Batch {
        match phase {
            Phase::Prefill => {
                if self.chunked_prefill.enabled {
                    self.compose_chunked_prefill(sequences, available_memory)
                } else {
                    self.compose_prefill_batch(sequences)
                }
            }
            Phase::Decode => self.compose_decode_batch(sequences),
        }
    }
}

impl Default for BatchComposer {
    fn default() -> Self {
        Self::new(BatchCompositionConfig::default())
    }
}

// Tests live in sibling files so this implementation file stays under
// the 800-line soft cap. Property tests are split into a separate file
// so the `proptest` dependency stays scoped to the proptest build.
#[cfg(test)]
mod tests;

#[cfg(test)]
mod prop_tests;
