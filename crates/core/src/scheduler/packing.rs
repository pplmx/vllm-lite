//! Sequence Packing Optimization
//!
//! Uses Best-Fit Decreasing (BFD) algorithm to group sequences of similar
//! lengths into batches, minimizing padding waste during prefill.
use crate::types::{Sequence, SequencePackingConfig};

/// Result of sequence packing
#[derive(Clone, Debug)]
pub struct PackedBatch {
    pub sequences: Vec<Sequence>,
    pub batch_size: usize,
    pub max_seq_len: usize,
    pub padding_waste: usize,
}

impl PackedBatch {
    fn new() -> Self {
        Self {
            sequences: Vec::new(),
            batch_size: 0,
            max_seq_len: 0,
            padding_waste: 0,
        }
    }

    fn add_sequence(&mut self, seq: Sequence) {
        self.sequences.push(seq);
        self.recalculate_stats();
    }

    fn recalculate_stats(&mut self) {
        self.batch_size = self.sequences.len();
        self.max_seq_len = self
            .sequences
            .iter()
            .map(|s| s.tokens.len())
            .max()
            .unwrap_or(0);
        self.padding_waste = self
            .sequences
            .iter()
            .map(|s| self.max_seq_len - s.tokens.len())
            .sum();
    }
}

/// Packer using Best-Fit Decreasing algorithm
pub struct SequencePacker {
    config: SequencePackingConfig,
}

impl SequencePacker {
    pub fn new(config: SequencePackingConfig) -> Self {
        Self { config }
    }

    /// Pack sequences using Best-Fit Decreasing algorithm
    pub fn pack_sequences(&self, sequences: Vec<Sequence>) -> Vec<PackedBatch> {
        if sequences.is_empty() {
            return vec![];
        }

        if !self.config.enabled {
            return vec![self.create_single_batch(sequences)];
        }

        // Sort by length descending (Decreasing)
        let mut sorted: Vec<Sequence> = sequences;
        sorted.sort_by_key(|s| std::cmp::Reverse(s.tokens.len()));

        let mut batches: Vec<PackedBatch> = Vec::new();

        for seq in sorted {
            // Find best fit batch
            let best_fit = self.find_best_fit(&batches, &seq);
            if let Some(idx) = best_fit {
                batches[idx].add_sequence(seq);
            } else {
                // Create new batch
                let mut batch = PackedBatch::new();
                batch.add_sequence(seq);
                batches.push(batch);
            }
        }

        batches
    }

    /// Find the batch that best fits the sequence
    fn find_best_fit(&self, batches: &[PackedBatch], seq: &Sequence) -> Option<usize> {
        let seq_len = seq.tokens.len();
        batches
            .iter()
            .enumerate()
            .filter(|(_, b)| b.batch_size < self.config.max_batch_size)
            .filter(|(_, b)| {
                // Check length similarity
                let batch_min_len = b
                    .sequences
                    .iter()
                    .map(|s| s.tokens.len())
                    .min()
                    .unwrap_or(seq_len);
                let max_len = b.max_seq_len.max(seq_len);
                let min_len = batch_min_len.min(seq_len);
                let diff = (max_len - min_len) as f32 / max_len as f32;
                diff <= self.config.similarity_threshold
            })
            .min_by(|(_, a), (_, b)| {
                // Best fit = minimum additional padding
                let add_padding_a = a.max_seq_len.max(seq_len) - a.max_seq_len;
                let add_padding_b = b.max_seq_len.max(seq_len) - b.max_seq_len;
                add_padding_a.cmp(&add_padding_b)
            })
            .map(|(idx, _)| idx)
    }

    fn create_single_batch(&self, sequences: Vec<Sequence>) -> PackedBatch {
        let mut batch = PackedBatch::new();
        for seq in sequences {
            batch.add_sequence(seq);
        }
        batch
    }
}

#[cfg(test)]
mod tests;
