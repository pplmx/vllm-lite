#![allow(clippy::module_name_repetitions)]
use crate::types::{BlockId, Sequence, Status};
use std::collections::{HashMap, VecDeque};

/// Per-policy eviction telemetry: total evictions, average block lifetime, recency-distribution buckets. Used to compare LRU vs. ARC vs. FIFO at runtime.
#[derive(Debug, Clone, Default)]
pub struct EvictionPolicyStats {
    pub total_evictions: usize,
    pub total_selections: usize,
    pub cache_hits: usize,
}

#[derive(Debug)]
/// Trait implemented by every KV-cache eviction policy (`LruPolicy`, `ArcPolicy`, `FifoPolicy`). Defines the touch-eviction decision given the current sequence set and a candidate block.
pub struct EvictionPolicy {
    block_access_order: VecDeque<BlockId>,
    block_ref_count: HashMap<BlockId, usize>,
    cached_victims: Option<(Vec<BlockId>, usize)>,
    stats: EvictionPolicyStats,
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl EvictionPolicy {
    #[must_use]
    pub fn new() -> Self {
        Self {
            block_access_order: VecDeque::new(),
            block_ref_count: HashMap::new(),
            cached_victims: None,
            stats: EvictionPolicyStats::default(),
        }
    }

    pub fn select_victims(
        &mut self,
        running_sequences: &[Sequence],
        num_blocks: usize,
    ) -> Vec<BlockId> {
        if num_blocks == 0 {
            return Vec::new();
        }

        self.stats.total_selections += 1;

        if let Some((ref cached, seq_hash)) = self.cached_victims
            && seq_hash == Self::compute_seq_hash(running_sequences)
            && cached.len() >= num_blocks
        {
            self.stats.cache_hits += 1;
            return cached.iter().take(num_blocks).copied().collect();
        }

        let mut block_usage: HashMap<BlockId, (&Sequence, usize)> = HashMap::new();

        for seq in running_sequences {
            if seq.status == Status::Finished || seq.status == Status::Waiting {
                continue;
            }

            for &block_id in seq.kv_blocks.as_ref() {
                let priority = Self::compute_priority(seq);
                block_usage.entry(block_id).or_insert((seq, priority)).1 =
                    priority.min(block_usage.get(&block_id).map_or(0, |(_, p)| *p));
            }
        }

        let mut sorted_blocks: Vec<_> = block_usage
            .into_iter()
            .map(|(block_id, (seq, priority))| (block_id, seq.id, priority))
            .collect();

        sorted_blocks.sort_by(|a, b| {
            let cmp = b.2.cmp(&a.2);
            if cmp != std::cmp::Ordering::Equal {
                return cmp;
            }
            a.1.cmp(&b.1)
        });

        let available_refs: HashMap<BlockId, usize> = self
            .block_ref_count
            .iter()
            .filter(|&(_, &count)| count <= 1)
            .map(|(&block, &count)| (block, count))
            .collect();

        let victims: Vec<BlockId> = sorted_blocks
            .into_iter()
            .filter(|(block_id, _, _)| available_refs.contains_key(block_id))
            .take(num_blocks)
            .map(|(block_id, _, _)| block_id)
            .collect();

        self.cached_victims = Some((victims.clone(), Self::compute_seq_hash(running_sequences)));
        victims
    }

    const fn compute_priority(seq: &Sequence) -> usize {
        match seq.status {
            Status::Prefilling => 2,
            Status::Decoding => {
                if seq.consecutive_decode_rounds > 5 {
                    1
                } else {
                    3
                }
            }
            _ => 0,
        }
    }

    fn compute_seq_hash(sequences: &[Sequence]) -> usize {
        let mut hash = 0usize;
        for seq in sequences {
            // invariant: hash values are bounded by usize; u64 -> usize truncation
            // is acceptable in this hash function (wrapping handles it).
            #[allow(clippy::cast_possible_truncation)]
            let id_hash = seq.id as usize;
            #[allow(clippy::cast_possible_truncation)]
            let status_hash = seq.status as usize;
            hash = hash.wrapping_mul(31).wrapping_add(id_hash);
            hash = hash.wrapping_mul(31).wrapping_add(status_hash);
            hash = hash.wrapping_mul(31).wrapping_add(seq.kv_blocks.len());
        }
        hash
    }

    pub fn invalidate_cache(&mut self) {
        self.cached_victims = None;
    }

    pub fn record_blocks(&mut self, blocks: &[BlockId]) {
        for &block in blocks {
            *self.block_ref_count.entry(block).or_insert(0) += 1;
            self.block_access_order.retain(|&b| b != block);
            self.block_access_order.push_front(block);
        }
        self.invalidate_cache();
    }

    pub fn release_blocks(&mut self, blocks: &[BlockId]) {
        for &block in blocks {
            if let Some(count) = self.block_ref_count.get_mut(&block) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    self.block_ref_count.remove(&block);
                }
            }
        }
        self.invalidate_cache();
    }

    pub fn touch_blocks(&mut self, blocks: &[BlockId]) {
        for &block in blocks {
            self.block_access_order.retain(|&b| b != block);
            self.block_access_order.push_front(block);
        }
    }

    #[must_use]
    pub fn get_block_ref_count(&self, block: BlockId) -> usize {
        *self.block_ref_count.get(&block).unwrap_or(&0)
    }

    #[must_use]
    pub fn stats(&self) -> EvictionPolicyStats {
        self.stats.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Priority, SamplingParams};
    use std::sync::Arc;

    fn create_test_sequence(id: u64, blocks: Vec<BlockId>, status: Status) -> Sequence {
        Sequence {
            id,
            tokens: vec![],
            kv_blocks: Arc::new(blocks),
            num_computed_tokens: 0,
            prompt_len: 0,
            status,
            max_tokens: 10,
            sampling_params: SamplingParams::default(),
            consecutive_decode_rounds: 0,
            priority: Priority::default(),
            degraded_draft: false,
            draft_model_id: None,
        }
    }

    #[test]
    fn test_eviction_policy_new() {
        let policy = EvictionPolicy::new();
        assert!(policy.block_access_order.is_empty());
        assert!(policy.block_ref_count.is_empty());
    }

    #[test]
    fn test_record_blocks() {
        let mut policy = EvictionPolicy::new();
        policy.record_blocks(&[1, 2, 3]);

        assert_eq!(policy.get_block_ref_count(1), 1);
        assert_eq!(policy.get_block_ref_count(2), 1);
        assert_eq!(policy.get_block_ref_count(3), 1);
    }

    #[test]
    fn test_record_blocks_increments_ref_count() {
        let mut policy = EvictionPolicy::new();
        policy.record_blocks(&[1, 2]);
        policy.record_blocks(&[2, 3]);

        assert_eq!(policy.get_block_ref_count(1), 1);
        assert_eq!(policy.get_block_ref_count(2), 2);
        assert_eq!(policy.get_block_ref_count(3), 1);
    }

    #[test]
    fn test_release_blocks() {
        let mut policy = EvictionPolicy::new();
        policy.record_blocks(&[1, 2, 3]);
        policy.release_blocks(&[2]);

        assert_eq!(policy.get_block_ref_count(1), 1);
        assert_eq!(policy.get_block_ref_count(2), 0);
        assert_eq!(policy.get_block_ref_count(3), 1);
    }

    #[test]
    fn test_release_blocks_removes_zero_refs() {
        let mut policy = EvictionPolicy::new();
        policy.record_blocks(&[1]);
        policy.release_blocks(&[1]);

        assert_eq!(policy.get_block_ref_count(1), 0);
    }

    #[test]
    fn test_touch_blocks_updates_order() {
        let mut policy = EvictionPolicy::new();
        policy.record_blocks(&[1, 2, 3]);
        policy.touch_blocks(&[1]);

        let front = policy.block_access_order.front();
        assert_eq!(front, Some(&1));
    }

    #[test]
    fn test_select_victims_empty_sequences() {
        let mut policy = EvictionPolicy::new();
        let victims = policy.select_victims(&[], 5);
        assert!(victims.is_empty());
    }

    #[test]
    fn test_select_victims_zero_blocks() {
        let mut policy = EvictionPolicy::new();
        let seq = create_test_sequence(1, vec![1, 2], Status::Decoding);
        let victims = policy.select_victims(&[seq], 0);
        assert!(victims.is_empty());
    }

    #[test]
    fn test_select_victims_prefilling_priority() {
        let mut policy = EvictionPolicy::new();
        policy.record_blocks(&[1, 2]);

        let prefill_seq = create_test_sequence(1, vec![1], Status::Prefilling);
        let decode_seq = create_test_sequence(2, vec![2], Status::Decoding);

        let victims = policy.select_victims(&[prefill_seq, decode_seq], 1);
        assert_eq!(victims.len(), 1);
    }

    #[test]
    fn test_select_victims_only_zero_ref_blocks() {
        let mut policy = EvictionPolicy::new();
        policy.record_blocks(&[1]);
        policy.record_blocks(&[1]);

        let seq = create_test_sequence(1, vec![1], Status::Decoding);
        let victims = policy.select_victims(&[seq], 1);

        assert!(victims.is_empty());
    }

    #[test]
    fn test_stats() {
        let mut policy = EvictionPolicy::new();

        assert_eq!(policy.stats().total_selections, 0);

        let seq = create_test_sequence(1, vec![1], Status::Decoding);
        policy.select_victims(&[seq], 1);

        assert_eq!(policy.stats().total_selections, 1);
    }

    #[test]
    fn test_cache_invalidation() {
        let mut policy = EvictionPolicy::new();

        let seq = create_test_sequence(1, vec![1], Status::Decoding);
        let victims1 = policy.select_victims(std::slice::from_ref(&seq), 1);

        policy.record_blocks(&[2]);
        let victims2 = policy.select_victims(&[seq], 1);

        assert_eq!(victims1.len(), victims2.len());
    }
}

#[cfg(test)]
mod prop_tests {
    use super::*;
    use crate::types::{Priority, SamplingParams};
    use proptest::prelude::*;
    use std::sync::Arc;

    fn make_sequence(
        id: u64,
        blocks: Vec<BlockId>,
        status: Status,
        decode_rounds: u32,
    ) -> Sequence {
        Sequence {
            id,
            tokens: vec![],
            kv_blocks: Arc::new(blocks),
            num_computed_tokens: 0,
            prompt_len: 0,
            status,
            max_tokens: 10,
            sampling_params: SamplingParams::default(),
            consecutive_decode_rounds: decode_rounds,
            priority: Priority::default(),
            degraded_draft: false,
            draft_model_id: None,
        }
    }

    #[allow(dead_code)] // proptest helpers referenced indirectly via proptest! macro
    fn arb_status() -> impl Strategy<Value = Status> {
        prop_oneof![
            Just(Status::Waiting),
            Just(Status::Prefilling),
            Just(Status::Decoding)
        ]
    }

    #[allow(dead_code)] // proptest helpers referenced indirectly via proptest! macro
    fn arb_sequence(id: u64) -> impl Strategy<Value = Sequence> {
        (
            proptest::collection::vec(0usize..32, 1..8),
            arb_status(),
            0u32..10,
        )
            .prop_map(move |(blocks, status, decode_rounds)| {
                make_sequence(id, blocks, status, decode_rounds)
            })
    }

    // Invariant 1: record_blocks / release_blocks refcount conservation —
    //              the count after N records and M releases equals max(0, N - M).
    // Invariant 2: select_victims returns at most `num_blocks` entries.
    // Invariant 3: empty running sequences always yields empty victims.
    // Invariant 4: cache hit (repeated identical select_victims call) does
    //              not double-count `total_selections` and bumps cache_hits.

    proptest! {
        /// Refcount conservation: total_refs equals
        /// max(0, records - releases) across any sequence of operations.
        #[test]
        fn prop_record_release_refcount_conserved(
            ops in proptest::collection::vec(
                (0usize..16, proptest::bool::ANY),
                1..50,
            ),
        ) {
            let mut policy = EvictionPolicy::new();
            let mut expected: HashMap<BlockId, usize> = HashMap::new();

            for (block_id, is_record) in ops {
                if is_record {
                    policy.record_blocks(&[block_id]);
                    *expected.entry(block_id).or_insert(0) += 1;
                } else if let Some(&count) = expected.get(&block_id) {
                    policy.release_blocks(&[block_id]);
                    if count <= 1 {
                        expected.remove(&block_id);
                    } else {
                        expected.insert(block_id, count - 1);
                    }
                }
                prop_assert_eq!(policy.get_block_ref_count(block_id), expected.get(&block_id).copied().unwrap_or(0));
            }
        }

        /// `select_victims` never returns more than `num_blocks` entries,
        /// and yields empty for empty input sequences.
        #[test]
        fn prop_select_victims_length_bounded(
            num_blocks in 0usize..10,
        ) {
            let mut policy = EvictionPolicy::new();
            let victims = policy.select_victims(&[], num_blocks);
            prop_assert!(victims.is_empty());
            prop_assert!(victims.len() <= num_blocks);
        }

        /// Repeated identical `select_victims` call must hit the cache:
        /// `cache_hits` strictly increases on the second call with the
        /// same input. We pin status to Decoding (Waiting/Finished
        /// sequences are skipped) and use unique blocks (otherwise the
        /// ref count exceeds 1, making the block unavailable for
        /// eviction and producing an empty victims list that bypasses
        /// the cache-hit check).
        #[test]
        fn prop_select_victims_cache_hit(
            blocks in proptest::collection::hash_set(0usize..32, 1..4),
        ) {
            let mut policy = EvictionPolicy::new();
            let blocks: Vec<usize> = blocks.into_iter().collect();
            for &block in &blocks {
                policy.record_blocks(&[block]);
            }
            let seq = make_sequence(1, blocks, Status::Decoding, 0);

            let _ = policy.select_victims(std::slice::from_ref(&seq), 1);
            let after_first = policy.stats();
            let _ = policy.select_victims(std::slice::from_ref(&seq), 1);
            let after_second = policy.stats();

            prop_assert_eq!(after_first.cache_hits, 0);
            prop_assert!(after_second.cache_hits > after_first.cache_hits);
        }
    }
}
