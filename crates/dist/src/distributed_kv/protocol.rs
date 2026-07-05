//! Wire protocol for the distributed KV-cache: node identifiers, cache operations, and RPC message envelopes.
//!
//! Pure-data definitions; serialization happens at the gRPC boundary in
//! `crate::grpc`. No I/O, no allocation beyond field construction.

/// Opaque newtype identifier for a node. Hashable, comparable, serializable; use this rather than the raw integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct NodeId(pub usize);

impl NodeId {
    #[must_use]
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    #[must_use]
    pub const fn index(&self) -> usize {
        self.0
    }
}

/// Enum of operations supported by Cache. Variants cover the full CRUD + admin verb set; serialized over the wire.
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

/// Wire message for the distributed KV-cache RPC protocol. Wraps a sequence id, op type (get/put/delete), and the corresponding payload.
#[derive(Debug, Clone)]
pub struct CacheMessage {
    /// Monotonic message id (process-local counter).
    pub id: u64,
    /// Originating node id.
    pub source: NodeId,
    /// Target node id (or broadcast group).
    pub destination: NodeId,
    /// The cache operation being requested/responded.
    pub operation: CacheOperation,
    /// Send timestamp in milliseconds since UNIX_EPOCH.
    pub timestamp: u64,
}

impl CacheMessage {
    /// Construct a new instance from the given configuration.
    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    pub fn new(source: NodeId, destination: NodeId, operation: CacheOperation) -> Self {
        static MSG_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

        Self {
            id: MSG_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            source,
            destination,
            operation,
            // invariant: monotonic clock is always >= UNIX_EPOCH; saturating
            // u64 conversion is safe for any realistic timestamp.
            timestamp: u64::try_from(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    // invariant: pre-conditions make this infallible at this call site.
                    .unwrap()
                    .as_millis(),
            )
            .unwrap_or(u64::MAX),
        }
    }

    #[must_use]
    pub fn read_request(key: u64, from: NodeId, to: NodeId) -> Self {
        Self::new(
            from,
            to,
            CacheOperation::Read {
                key,
                requester: from,
            },
        )
    }

    #[must_use]
    pub fn invalidate(key: u64, from: NodeId, to: NodeId) -> Self {
        Self::new(
            from,
            to,
            CacheOperation::Invalidate {
                key,
                requester: from,
            },
        )
    }

    #[must_use]
    pub fn update(key: u64, hash: u64, from: NodeId, to: NodeId) -> Self {
        Self::new(
            from,
            to,
            CacheOperation::Update {
                key,
                value_hash: hash,
                requester: from,
            },
        )
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
