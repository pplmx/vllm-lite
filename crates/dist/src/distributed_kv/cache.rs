use super::{CacheConfig, CacheMessage, NodeId};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub struct DistributedKVCache {
    config: CacheConfig,
    local_cache: Arc<RwLock<HashMap<u64, CacheEntry>>>,
    stats: CacheStats,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CacheEntry {
    key: u64,
    value_hash: u64,
    owner_nodes: Vec<NodeId>,
    state: CacheState,
    last_access: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum CacheState {
    Shared,
    Modified,
    Exclusive,
    Invalid,
}

#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub invalidations: u64,
    pub updates: u64,
}

impl DistributedKVCache {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            local_cache: Arc::new(RwLock::new(HashMap::new())),
            stats: CacheStats::default(),
        }
    }

    pub fn get(&self, key: u64) -> Option<u64> {
        let cache = self.local_cache.read().ok()?;

        if let Some(entry) = cache.get(&key) {
            if entry.state != CacheState::Invalid {
                return Some(entry.value_hash);
            }
        }

        None
    }

    pub fn put(&self, key: u64, value_hash: u64) {
        let owner_nodes = self.compute_owner_nodes(key);

        let entry = CacheEntry {
            key,
            value_hash,
            owner_nodes: owner_nodes.clone(),
            state: if self.config.node_id == owner_nodes[0] {
                CacheState::Exclusive
            } else {
                CacheState::Shared
            },
            last_access: current_timestamp(),
        };

        if let Ok(mut cache) = self.local_cache.write() {
            cache.insert(key, entry);
        }
    }

    pub fn invalidate(&self, key: u64) {
        if let Ok(mut cache) = self.local_cache.write() {
            cache.remove(&key);
        }
    }

    pub fn handle_message(&self, msg: &CacheMessage) -> Option<CacheMessage> {
        match &msg.operation {
            super::protocol::CacheOperation::Read { key, .. } => {
                if let Some(_value_hash) = self.get(*key) {
                    return Some(CacheMessage::new(
                        self.config.node_id,
                        msg.source,
                        super::protocol::CacheOperation::Ack {
                            operation_id: msg.id,
                            success: true,
                        },
                    ));
                }
                None
            }
            super::protocol::CacheOperation::Invalidate { key, .. } => {
                self.invalidate(*key);
                Some(CacheMessage::new(
                    self.config.node_id,
                    msg.source,
                    super::protocol::CacheOperation::Ack {
                        operation_id: msg.id,
                        success: true,
                    },
                ))
            }
            super::protocol::CacheOperation::Update { key, value_hash, .. } => {
                self.put(*key, *value_hash);
                None
            }
            super::protocol::CacheOperation::Write { key, value_hash, .. } => {
                self.put(*key, *value_hash);
                None
            }
            super::protocol::CacheOperation::Ack { .. } => None,
        }
    }

    pub fn stats(&self) -> CacheStats {
        self.stats.clone()
    }

    fn compute_owner_nodes(&self, key: u64) -> Vec<NodeId> {
        let mut nodes = Vec::with_capacity(self.config.replication_factor);
        let base = (key as usize) % self.config.num_nodes;

        for i in 0..self.config.replication_factor {
            let node_id = (base + i) % self.config.num_nodes;
            nodes.push(NodeId::new(node_id));
        }

        nodes
    }

    pub fn memory_usage(&self) -> usize {
        if let Ok(cache) = self.local_cache.read() {
            cache.len() * std::mem::size_of::<CacheEntry>()
        } else {
            0
        }
    }
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_put_get() {
        let config = CacheConfig::new(NodeId(0), 4);
        let cache = DistributedKVCache::new(config);

        cache.put(1, 100);
        let result = cache.get(1);

        assert_eq!(result, Some(100));
    }

    #[test]
    fn test_cache_miss() {
        let config = CacheConfig::new(NodeId(0), 4);
        let cache = DistributedKVCache::new(config);

        let result = cache.get(999);
        assert_eq!(result, None);
    }

    #[test]
    fn test_cache_invalidate() {
        let config = CacheConfig::new(NodeId(0), 4);
        let cache = DistributedKVCache::new(config);

        cache.put(1, 100);
        assert!(cache.get(1).is_some());

        cache.invalidate(1);
        assert!(cache.get(1).is_none());
    }

    #[test]
    fn test_owner_nodes_single_replication() {
        let config = CacheConfig::new(NodeId(0), 4);
        let cache = DistributedKVCache::new(config);

        let owners = cache.compute_owner_nodes(5);
        assert_eq!(owners.len(), 2);
        assert!(owners.contains(&NodeId::new(1))); // 5 % 4 = 1
    }

    #[test]
    fn test_owner_nodes_distribution() {
        let config = CacheConfig::new(NodeId(0), 8);
        let cache = DistributedKVCache::new(config);

        for key in 0..16 {
            let owners = cache.compute_owner_nodes(key);
            assert!(!owners.is_empty());
            assert!(owners.windows(2).all(|w| w[0] != w[1])); // No duplicates
        }
    }
}
