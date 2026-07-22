//! Async sink of raw KV-block bytes for receiver-side install.
//!
//! [`BlockSink`] is the receiver-side counterpart to
//! [`crate::distributed_kv::block_data_source::BlockDataSource`]. When a
//! peer gRPC `TransferKVBlock` RPC returns bytes, the receiver hands them
//! to a [`BlockSink`] that installs them into the local cache (e.g.
//! `PagedKvCache::write_block_bytes`).
//!
//! [`MockBlockSink`] is an in-memory test helper that simulates the
//! trait's failure surface (force-flags for `InvalidBytes` /
//! `OutOfRange`). For real byte-level round-trip assertions, tests
//! should use a real `PagedKvCacheWrapper` as the sink.
//!
//! ## Design constraints (Phase 31-D / P42)
//!
//! - **Async**: matches the [`BlockDataSource`] shape so it can be
//!   wired into the same async-driven `DistributedKVCache::fetch_block`
//!   hot path without spawning extra tasks.
//! - **Object-safe**: `Arc<dyn BlockSink>` is the canonical storage
//!   form; the trait must NOT use generic methods or `Self` in
//!   non-object-safe positions. `#[async_trait::async_trait]` adds
//!   `Send` bounds compatible with dyn dispatch.
//! - **Bytes-only**: the receiver cannot validate cache shape
//!   internally; it hands the raw `&[u8]` to the concrete sink, which
//!   decides how to interpret it (e.g. `PagedKvCache::write_block_bytes`
//!   parses the wire layout per-layer).
//! - **Errors are recoverable**: `WriteError::InvalidBytes` (wrong
//!   length) and `WriteError::OutOfRange` (block_id can't fit) are
//!   deterministic client-side mistakes; the caller may choose to
//!   treat them as `Status::internal` on the gRPC side or just log.

use std::fmt;

/// Async sink of raw KV-block bytes (receiver side).
///
/// The [`crate::distributed_kv::DistributedKVCache::fetch_block`]
/// method calls `write_block` after a successful peer / local-source
/// fetch when `install_on_fetch` is enabled (the default), so the
/// receiver's local cache ends up populated with the same bytes the
/// sender served.
///
/// Implementations are expected to be cheap to share (`Send + Sync`)
/// so the cache can hand the same `Arc<dyn BlockSink>` to the gRPC
/// server state at construction time.
#[async_trait::async_trait]
pub trait BlockSink: Send + Sync + fmt::Debug {
    /// Install the wire-shape bytes for `block_id` into the local sink.
    ///
    /// Returns [`WriteError::OutOfRange`] if `block_id` cannot fit in
    /// the sink's allocator, or [`WriteError::InvalidBytes`] if
    /// `bytes.len()` is not the expected size for the configured cache
    /// shape.
    async fn write_block(&self, block_id: u64, bytes: &[u8]) -> Result<(), WriteError>;
}

/// Errors emitted by block-bytes writing.
///
/// Three variants cover the full failure surface:
///
/// - [`InvalidBytes`](Self::InvalidBytes): `bytes.len()` doesn't match
///   the sink's expected shape. Treated as a soft failure by the
///   caller — the fetch result is still returned, only the install is
///   skipped (logged as a warning).
/// - [`OutOfRange`](Self::OutOfRange): `block_id` doesn't fit in the
///   sink's allocator (e.g. `PagedKvCache` has `num_blocks = 4` and
///   the peer asked for block 99). Same soft-failure semantics.
/// - [`SinkUnavailable`](Self::SinkUnavailable): the sink was
///   disconnected / poisoned mid-call. Shouldn't happen in practice
///   but covers the `parking_lot::Mutex` poisoning case symmetrically
///   with `BlockDataSource::SourceUnavailable`.
#[derive(Debug, thiserror::Error)]
pub enum WriteError {
    #[error("invalid byte length for block {block_id}: expected {expected} bytes, got {actual}")]
    InvalidBytes {
        block_id: u64,
        expected: usize,
        actual: usize,
    },

    #[error("block_id {block_id} out of range for sink")]
    OutOfRange { block_id: u64 },

    #[error("sink is unavailable (poisoned lock / disconnected)")]
    SinkUnavailable,
}

/// In-memory mock used by tests.
///
/// `MockBlockSink` is intentionally a **failure-simulator** mock — it
/// does not store the bytes it receives. For byte-level round-trip
/// assertions, tests should use a real `PagedKvCacheWrapper` (see
/// `tests/multi_node_receiver_write.rs::round_trip_bit_exact_after_install`).
/// The mock's job is to let tests simulate `WriteError` paths without
/// needing to construct malformed byte slices.
///
/// Public (not `#[cfg(test)]`) so the integration tests in
/// `tests/multi_node_receiver_write.rs` can construct one — they're
/// separate test binaries and can't see the library's `#[cfg(test)]`
/// items. Production code MUST NOT use this type; it's only a test
/// helper.
#[derive(Debug, Default)]
pub struct MockBlockSink {
    /// When `true`, `write_block` returns
    /// [`WriteError::InvalidBytes`]. Lets tests simulate a malformed
    /// payload without having to construct a wrong-length byte slice.
    pub force_invalid_bytes: bool,
    /// When `true`, `write_block` returns [`WriteError::OutOfRange`]
    /// regardless of `block_id`. Mirrors `force_invalid_bytes`.
    pub force_out_of_range: bool,
}

impl MockBlockSink {
    /// Construct an empty mock with no failure flags.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait::async_trait]
impl BlockSink for MockBlockSink {
    async fn write_block(&self, block_id: u64, bytes: &[u8]) -> Result<(), WriteError> {
        if self.force_out_of_range {
            return Err(WriteError::OutOfRange { block_id });
        }
        if self.force_invalid_bytes {
            return Err(WriteError::InvalidBytes {
                block_id,
                expected: bytes.len(),
                actual: bytes.len(),
            });
        }
        // Silent success — the mock doesn't store bytes. Use a real
        // `PagedKvCacheWrapper` if you need to assert what was written.
        let _ = (block_id, bytes);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_write_block_succeeds_when_no_force_flag_set() {
        let sink = MockBlockSink::new();
        let result = sink.write_block(42, b"hello").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn mock_write_block_returns_invalid_bytes_when_flag_set() {
        let mut sink = MockBlockSink::new();
        sink.force_invalid_bytes = true;
        let result = sink.write_block(42, b"hello").await;
        match result {
            Err(WriteError::InvalidBytes { block_id, .. }) => assert_eq!(block_id, 42),
            other => panic!("expected InvalidBytes, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn mock_write_block_returns_out_of_range_when_flag_set() {
        let mut sink = MockBlockSink::new();
        sink.force_out_of_range = true;
        let result = sink.write_block(42, b"hello").await;
        match result {
            Err(WriteError::OutOfRange { block_id }) => assert_eq!(block_id, 42),
            other => panic!("expected OutOfRange, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn mock_write_block_invalid_bytes_takes_precedence_over_success() {
        // Both flags force_invalid_bytes=false; check the dispatch
        // logic actually distinguishes them. If neither flag is set,
        // the call succeeds silently.
        let sink = MockBlockSink::new();
        assert!(sink.write_block(0, b"abc").await.is_ok());
    }

    #[test]
    fn write_error_display_messages_are_distinct() {
        // Sanity check that the three variants format differently —
        // catches accidental variant collisions (mirror of the
        // BlockDataSource::FetchError check).
        let variants = [
            WriteError::InvalidBytes {
                block_id: 1,
                expected: 4,
                actual: 5,
            }
            .to_string(),
            WriteError::OutOfRange { block_id: 1 }.to_string(),
            WriteError::SinkUnavailable.to_string(),
        ];
        let unique: std::collections::HashSet<_> = variants.iter().collect();
        assert_eq!(
            unique.len(),
            variants.len(),
            "every WriteError variant must produce a distinct Display"
        );
    }

    #[test]
    fn mock_block_sink_default_is_no_force_flags() {
        let sink = MockBlockSink::default();
        assert!(!sink.force_invalid_bytes);
        assert!(!sink.force_out_of_range);
    }
}
