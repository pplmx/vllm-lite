pub mod protocol;
pub mod cache;

pub use cache::DistributedKVCache;
pub use protocol::{CacheMessage, CacheOperation, NodeId};

#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub node_id: NodeId,
    pub num_nodes: usize,
    pub replication_factor: usize,
    pub invalidation_strategy: InvalidationStrategy,
    pub coherence_protocol: CoherenceProtocol,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidationStrategy {
    WriteInvalidate,
    WriteUpdate,
    NoInvalidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoherenceProtocol {
    None,
    MESI,
    Directory,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            node_id: NodeId(0),
            num_nodes: 1,
            replication_factor: 2,
            invalidation_strategy: InvalidationStrategy::WriteInvalidate,
            coherence_protocol: CoherenceProtocol::Directory,
        }
    }
}

impl CacheConfig {
    pub fn new(node_id: NodeId, num_nodes: usize) -> Self {
        Self {
            node_id,
            num_nodes,
            replication_factor: 2.min(num_nodes),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CacheConfig::default();
        assert_eq!(config.node_id, NodeId(0));
        assert_eq!(config.num_nodes, 1);
        assert_eq!(config.replication_factor, 2); // default is 2
    }

    #[test]
    fn test_multi_node_config() {
        let config = CacheConfig::new(NodeId(2), 4);
        assert_eq!(config.node_id, NodeId(2));
        assert_eq!(config.num_nodes, 4);
        assert_eq!(config.replication_factor, 2);
    }

    #[test]
    fn test_invalidation_strategies() {
        let invalidate = InvalidationStrategy::WriteInvalidate;
        let update = InvalidationStrategy::WriteUpdate;
        let none = InvalidationStrategy::NoInvalidation;

        assert_ne!(invalidate, update);
        assert_ne!(update, none);
        assert_ne!(none, invalidate);
    }

    #[test]
    fn test_coherence_protocols() {
        let none = CoherenceProtocol::None;
        let mesi = CoherenceProtocol::MESI;
        let dir = CoherenceProtocol::Directory;

        assert_ne!(none, mesi);
        assert_ne!(mesi, dir);
        assert_ne!(dir, none);
    }
}
