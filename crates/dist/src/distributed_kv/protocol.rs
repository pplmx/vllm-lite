#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct NodeId(pub usize);

impl NodeId {
    pub fn new(id: usize) -> Self {
        Self(id)
    }

    pub fn index(&self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone)]
pub enum CacheOperation {
    Read {
        key: u64,
        requester: NodeId,
    },
    Write {
        key: u64,
        value_hash: u64,
        requester: NodeId,
    },
    Invalidate {
        key: u64,
        requester: NodeId,
    },
    Update {
        key: u64,
        value_hash: u64,
        requester: NodeId,
    },
    Ack {
        operation_id: u64,
        success: bool,
    },
}

#[derive(Debug, Clone)]
pub struct CacheMessage {
    pub id: u64,
    pub source: NodeId,
    pub destination: NodeId,
    pub operation: CacheOperation,
    pub timestamp: u64,
}

impl CacheMessage {
    pub fn new(source: NodeId, destination: NodeId, operation: CacheOperation) -> Self {
        static MSG_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

        Self {
            id: MSG_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            source,
            destination,
            operation,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }

    pub fn read_request(key: u64, from: NodeId, to: NodeId) -> Self {
        Self::new(from, to, CacheOperation::Read { key, requester: from })
    }

    pub fn invalidate(key: u64, from: NodeId, to: NodeId) -> Self {
        Self::new(from, to, CacheOperation::Invalidate { key, requester: from })
    }

    pub fn update(key: u64, hash: u64, from: NodeId, to: NodeId) -> Self {
        Self::new(from, to, CacheOperation::Update { key, value_hash: hash, requester: from })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id() {
        let id = NodeId::new(5);
        assert_eq!(id.index(), 5);
    }

    #[test]
    fn test_node_id_default() {
        let id = NodeId::default();
        assert_eq!(id.index(), 0);
    }

    #[test]
    fn test_cache_message_read_request() {
        let msg = CacheMessage::read_request(123, NodeId(0), NodeId(1));

        assert_eq!(msg.source, NodeId(0));
        assert_eq!(msg.destination, NodeId(1));

        match msg.operation {
            CacheOperation::Read { key, requester } => {
                assert_eq!(key, 123);
                assert_eq!(requester, NodeId(0));
            }
            _ => panic!("Expected Read operation"),
        }
    }

    #[test]
    fn test_cache_message_invalidate() {
        let msg = CacheMessage::invalidate(456, NodeId(1), NodeId(2));

        match msg.operation {
            CacheOperation::Invalidate { key, requester } => {
                assert_eq!(key, 456);
                assert_eq!(requester, NodeId(1));
            }
            _ => panic!("Expected Invalidate operation"),
        }
    }

    #[test]
    fn test_message_id_uniqueness() {
        let msg1 = CacheMessage::read_request(1, NodeId(0), NodeId(1));
        let msg2 = CacheMessage::read_request(2, NodeId(0), NodeId(1));

        assert_ne!(msg1.id, msg2.id);
    }
}
