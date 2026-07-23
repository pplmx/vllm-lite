//! Eviction policy for KV-cache blocks under memory pressure: LRU with optional priority weighting and prefix-cache awareness.
//!
//! The default impl is [`EvictionPolicy`] (LRU). Priority-weighted variants
//! are available when `SchedulerConfig.eviction_priority_weight > 0`.
//! Returns the victim block ranges for the allocator to free.
#![allow(clippy::module_name_repetitions)]
use crate::types::{BlockId, Sequence, Status};
use std::collections::{HashMap, VecDeque};

/// Per-policy eviction telemetry: total evictions, average block lifetime, recency-distribution buckets. Used to compare LRU vs. ARC vs. FIFO at runtime.
#[derive(Debug, Clone, Default)]
pub struct EvictionPolicyStats {
    /// Cumulative blocks evicted by this policy.
    pub total_evictions: usize,
    /// Cumulative `select_victims` calls.
    pub total_selections: usize,
    /// Times a cached victim set was reused (avoided re-computation).
    pub cache_hits: usize,
}

#[derive(Debug)]
/// Trait implemented by every KV-cache eviction policy (`LruPolicy`, `ArcPolicy`, `FifoPolicy`). Defines the touch-eviction decision given the current sequence set and a candidate block.
pub struct EvictionPolicy {
    /// LRU access order (front = most recently used).
    block_access_order: VecDeque<BlockId>,
    /// Live reference count per block (across all sequences).
    block_ref_count: HashMap<BlockId, usize>,
    /// Cached victim set with the seq-hash it was computed for.
    cached_victims: Option<(Vec<BlockId>, usize)>,
    /// Cumulative policy statistics.
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

    /// Decrement the refcount for every block in `blocks` and return
    /// the subset that just reached zero — i.e., the blocks that are
    /// now safe for the allocator to free.
    ///
    /// ARCH-01 (technical due diligence): the previous implementation
    /// returned `()` and the caller had no way to know which blocks
    /// could be freed. That forced the caller (`MemoryManager`) to free
    /// every released block unconditionally, which corrupted shared
    /// prefix-cache entries: if sequence A finishes and inserts its
    /// blocks into the prefix cache, the very next sequence reusing
    /// those blocks got back freed memory.
    ///
    /// The new contract is:
    ///   - Every `record_blocks` MUST be paired with exactly one
    ///     `release_blocks`. (Refcount > 0 keeps a block alive.)
    ///   - Returned blocks MUST be passed to the allocator's
    ///     `free()`. (Refcount 0 means no live owner.)
    ///   - Blocks whose refcount is already 0 (drift / double
    ///     release) are reported back too — better to free the
    ///     block than to leak it indefinitely.
    ///
    /// `saturating_sub(1)` is preserved for the refcount path so a
    /// pathological extra release doesn't underflow into a wrap-
    /// around refcount explosion.
    pub fn release_blocks(&mut self, blocks: &[BlockId]) -> Vec<BlockId> {
        let mut freed = Vec::new();
        for &block in blocks {
            let decremented_to_zero = self.block_ref_count.get_mut(&block).is_none_or(|count| {
                *count = count.saturating_sub(1);
                *count == 0
            });
            if decremented_to_zero {
                self.block_ref_count.remove(&block);
                self.block_access_order.retain(|&b| b != block);
                freed.push(block);
            }
        }
        self.invalidate_cache();
        freed
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

// Unit tests are extracted to `tests.rs` and `prop_tests.rs` to keep
// this file under the 800-line soft cap. See those siblings for the
// test surface (record/release refcount, select_victims edge cases,
// stats counters; plus proptest invariants for refcount conservation,
// length bound, and cache-hit behavior).
#[cfg(test)]
mod prop_tests;
#[cfg(test)]
mod tests;
