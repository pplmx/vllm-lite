//! Async source of raw KV-block bytes for cross-node transfer.
//!
//! [`BlockDataSource`] is the abstraction
//! [`crate::distributed_kv::DistributedKVCache::fetch_block`] and the
//! `TransferKVBlock` gRPC handler depend on to materialize block bytes.
//! Production implementations will wrap the model crate's
//! `PagedKvCache`; tests provide an in-memory
//! [`MockBlockDataSource`] so the gRPC layer can be exercised without a
//! GPU.
//!
//! ## Design constraints (Phase 31-D)
//!
//! - **Async**: matches the eventual `PagedKvCache` access pattern (the
//!   GPU-resident tensor copy is the kind of work that benefits from
//!   `await`-able suspension, even though the v31-D mocks are
//!   sync-from-RAM).
//! - **Object-safe**: `Arc<dyn BlockDataSource>` is the canonical storage
//!   form, so the trait must NOT use generic methods or `Self` in
//!   non-object-safe positions. `#[async_trait::async_trait]` is used to
//!   add `Send` bounds compatible with dyn dispatch.
//! - **Storage-agnostic**: returns raw `Vec<u8>` so a future GPU-direct
//!   impl can return device-side slices wrapped as bytes without
//!   changing the trait signature.
//! - **Hash carried on the wire, not in the trait**: the `chain_hash`
//!   travels alongside the bytes on the gRPC wire (see
//!   `TransferKvBlockResponse.chain_hash`); the receiver verifies it
//!   against its locally-recorded `value_hash`. This keeps the dist
//!   layer from importing `BlockHasher` (which lives in `vllm-traits`).

use std::collections::HashMap;
use std::fmt;

/// Maximum size (in bytes) of a single `TransferKVBlock` response.
///
/// Sized for Qwen3-7B (≈14 MiB/block at F32) with ~4× headroom for
/// larger models (Qwen3-72B: ~24 MiB/block) or future `BLOCK_SIZE`
/// growth. Applied symmetrically to both the gRPC server's
/// `max_decoding_message_size` / `max_encoding_message_size` and the
/// `PeerClient`'s generated client builder.
///
/// Tonic's default 4 MiB limit is far too small — bumping here is what
/// makes 31-D's block transfer actually work for production-sized
/// blocks.
pub const MAX_BLOCK_TRANSFER_BYTES: usize = 64 * 1024 * 1024;

/// Async source of raw KV-block bytes.
///
/// The [`crate::distributed_kv::DistributedKVCache::fetch_block`] method
/// (fan-out fallback over peers, then local-source fallback) and the
/// `TransferKVBlock` gRPC handler both rely on this trait.
///
/// Implementations are expected to be cheap to share (`Send + Sync`) so
/// the cache can hand the same `Arc<dyn BlockDataSource>` to the gRPC
/// server state at construction time.
#[async_trait::async_trait]
pub trait BlockDataSource: Send + Sync + fmt::Debug {
    /// Fetch the raw bytes for `block_id`.
    ///
    /// Returns [`FetchError::NotFound`] if this source does not hold
    /// the block. Hash verification is **not** the source's job — the
    /// protocol layer compares the wire-carried `chain_hash` against
    /// the receiver's locally-recorded `value_hash`.
    async fn fetch_block(&self, block_id: u64) -> Result<Vec<u8>, FetchError>;

    /// Cheap precheck: does this source hold `block_id` at all?
    ///
    /// Defaults to `true` so implementations that can't answer cheaply
    /// don't have to override it. Production wrappers around
    /// `PagedKvCache` will likely override this with an O(1) lookup
    /// against the per-layer `block_hashes` map.
    async fn has_block(&self, block_id: u64) -> bool {
        let _ = block_id;
        true
    }
}

/// Errors emitted by block-bytes fetching.
///
/// Six variants cover the full failure surface:
///
/// - [`NotFound`](Self::NotFound): nobody in the cluster (peers +
///   local source) holds the block.
/// - [`HashMismatch`](Self::HashMismatch): a peer returned bytes whose
///   advertised `chain_hash` doesn't match the receiver's locally-
///   recorded `value_hash`. Treated as a soft failure — other peers
///   are still tried.
/// - [`SourceUnavailable`](Self::SourceUnavailable): no
///   [`BlockDataSource`] is wired into the local `GrpcState` (server
///   side) or into the local `DistributedKVCache` (client side).
/// - [`NoPeers`](Self::NoPeers): `connect_peers` was never called or
///   `peer_urls` is empty, AND no local source is wired. A pure
///   single-node misconfiguration for the fetch path.
/// - [`AllPeersFailed`](Self::AllPeersFailed): fan-out fallback was
///   exhausted; inner `usize` records how many peers were tried for
///   diagnostics.
/// - [`Transport`](Self::Transport): the underlying tonic RPC failed at
///   the HTTP/2 layer.
#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    #[error("block {0} not held by any peer or local source")]
    NotFound(u64),

    #[error(
        "hash mismatch for block {block_id}: expected {expected_hash:#x}, got {actual_hash:#x}"
    )]
    HashMismatch {
        block_id: u64,
        expected_hash: u64,
        actual_hash: u64,
    },

    #[error("no BlockDataSource wired in (single-node server)")]
    SourceUnavailable,

    #[error("no peers configured and no local source wired")]
    NoPeers,

    #[error("all {0} peers failed for the block transfer")]
    AllPeersFailed(usize),

    #[error("transport error during block transfer")]
    Transport(#[source] tonic::Status),
}

impl From<tonic::Status> for FetchError {
    fn from(s: tonic::Status) -> Self {
        Self::Transport(s)
    }
}

/// In-memory mock used by tests. Stores `HashMap<u64, Vec<u8>>` of
/// block bytes; tests populate it before the gRPC server starts and
/// the mock hands the bytes back to whichever peer asks.
///
/// Public (not `#[cfg(test)]`) so the integration tests in
/// `tests/kv_block_transfer.rs` can construct one — they're separate
/// test binaries and can't see the library's `#[cfg(test)]` items.
/// Production code MUST NOT use this type; it's only a test helper.
///
/// Named `MockBlockDataSource` (with `Mock` prefix) to avoid the
/// `module_name_repetitions` lint: the containing module is
/// `block_data_source`, so a bare `BlockDataSource` struct would
/// trigger `clippy::module_name_repetitions`.
// The `allow` is needed because clippy still flags the trailing
// `Source` substring as repetition of the module's
// `data_source`/`block_data_source` suffix.
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Default)]
pub struct MockBlockDataSource {
    /// `block_id → serialized block bytes`. Cloned on fetch (cheap,
    /// since the mock is only used in tests).
    pub blocks: HashMap<u64, Vec<u8>>,
}

impl MockBlockDataSource {
    /// Construct an empty mock.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a `(block_id, bytes)` pair. Overwrites any prior entry
    /// for the same `block_id`.
    pub fn insert(&mut self, block_id: u64, bytes: Vec<u8>) {
        self.blocks.insert(block_id, bytes);
    }
}

#[async_trait::async_trait]
impl BlockDataSource for MockBlockDataSource {
    async fn fetch_block(&self, block_id: u64) -> Result<Vec<u8>, FetchError> {
        self.blocks
            .get(&block_id)
            .cloned()
            .ok_or(FetchError::NotFound(block_id))
    }

    async fn has_block(&self, block_id: u64) -> bool {
        self.blocks.contains_key(&block_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_fetch_block_returns_inserted_bytes() {
        let mut source = MockBlockDataSource::new();
        source.insert(42, vec![0xDE, 0xAD, 0xBE, 0xEF]);

        let bytes = source.fetch_block(42).await.expect("fetch ok");
        assert_eq!(bytes, vec![0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[tokio::test]
    async fn mock_fetch_block_returns_not_found_for_missing() {
        let source = MockBlockDataSource::new();
        let result = source.fetch_block(999).await;
        assert!(matches!(result, Err(FetchError::NotFound(999))));
    }

    #[tokio::test]
    async fn mock_has_block_matches_insertions() {
        let mut source = MockBlockDataSource::new();
        source.insert(1, vec![0xAA]);

        assert!(source.has_block(1).await);
        assert!(!source.has_block(2).await);
    }

    #[test]
    fn fetch_error_display_messages_are_distinct() {
        // Sanity check that the six variants format differently —
        // catches accidental variant collisions.
        let variants = [
            FetchError::NotFound(0).to_string(),
            FetchError::HashMismatch {
                block_id: 0,
                expected_hash: 1,
                actual_hash: 2,
            }
            .to_string(),
            FetchError::SourceUnavailable.to_string(),
            FetchError::NoPeers.to_string(),
            FetchError::AllPeersFailed(2).to_string(),
            FetchError::Transport(tonic::Status::unavailable("x")).to_string(),
        ];
        let unique: std::collections::HashSet<_> = variants.iter().collect();
        assert_eq!(
            unique.len(),
            variants.len(),
            "every FetchError variant must produce a distinct Display"
        );
    }

    #[test]
    fn fetch_error_from_tonic_status_yields_transport() {
        let status = tonic::Status::unavailable("peer gone");
        let err: FetchError = status.into();
        assert!(matches!(err, FetchError::Transport(_)));
    }
}
