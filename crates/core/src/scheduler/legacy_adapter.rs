use crate::kv_cache::PrefixCache;
use crate::scheduler::SchedulerStats;
use crate::scheduler::engine::SchedulerEngine;
use crate::scheduler::eviction::EvictionPolicy;
use crate::types::{Batch, Request, SchedulerConfig, SeqId, Sequence};

pub struct LegacySchedulerAdapter {
    engine: SchedulerEngine,
    prefix_cache: PrefixCache,
    eviction_policy: EvictionPolicy,
}

impl LegacySchedulerAdapter {
    pub fn new(config: SchedulerConfig, num_kv_blocks: usize) -> Self {
        Self {
            engine: SchedulerEngine::new(config, num_kv_blocks),
            prefix_cache: PrefixCache::new(),
            eviction_policy: EvictionPolicy::new(),
        }
    }

    pub fn add_request(&mut self, req: Request) -> SeqId {
        self.engine.add_request(req)
    }

    pub fn build_batch(&mut self) -> Batch {
        self.engine.build_batch()
    }

    pub fn update(&mut self, seq_ids: &[SeqId], next_tokens: &[u32], input_token_counts: &[usize]) {
        self.engine.update(seq_ids, next_tokens, input_token_counts)
    }

    pub fn has_pending(&self) -> bool {
        self.engine.has_pending()
    }

    pub fn running_count(&self) -> usize {
        0
    }

    pub fn waiting_count(&self) -> usize {
        self.engine.waiting_count()
    }

    pub fn finished_sequences(&self) -> &[Sequence] {
        &[]
    }

    pub fn running(&self) -> &[Sequence] {
        &[]
    }

    pub fn prefix_cache(&self) -> &PrefixCache {
        &self.prefix_cache
    }

    pub fn eviction(&self) -> &EvictionPolicy {
        &self.eviction_policy
    }

    pub fn get_kv_cache_usage(&self) -> (u64, u64) {
        self.engine.get_kv_cache_usage()
    }

    pub fn get_prefix_cache_stats(&self) -> (u64, u64) {
        (0, 0)
    }

    pub fn get_scheduler_stats(&self) -> SchedulerStats {
        self.engine.get_stats()
    }
}

impl Default for LegacySchedulerAdapter {
    fn default() -> Self {
        Self::new(SchedulerConfig::default(), 1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Request;

    #[test]
    fn test_adapter_add_request() {
        let mut adapter = LegacySchedulerAdapter::default();
        let id = adapter.add_request(Request::new(0, vec![1, 2, 3], 5));
        assert!(id > 0);
    }

    #[test]
    fn test_adapter_has_pending() {
        let mut adapter = LegacySchedulerAdapter::default();
        assert!(!adapter.has_pending());

        adapter.add_request(Request::new(0, vec![1, 2, 3], 5));
        assert!(adapter.has_pending());
    }

    #[test]
    fn test_adapter_build_batch() {
        let mut adapter = LegacySchedulerAdapter::default();
        adapter.add_request(Request::new(0, vec![1, 2, 3], 5));

        let batch = adapter.build_batch();
        assert!(!batch.is_empty() || !adapter.has_pending());
    }
}
