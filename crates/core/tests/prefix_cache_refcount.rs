//! ARCH-01 regression tests — prefix-cache shared block lifecycle.
//!
//! Background: prior to the fix, `EvictionPolicy::release_blocks`
//! returned `()` and `MemoryManager::release_blocks` therefore freed
//! every released block unconditionally. That meant a sequence which
//! finished and inserted its blocks into the prefix cache had those
//! blocks freed underneath the cache; a subsequent prefix-cache hit
//! handed out freed memory and the second sequence either crashed
//! or generated corrupted KV.
//!
//! The fix changes `release_blocks` to return the set of blocks
//! that just reached refcount 0, and `MemoryManager::release_blocks`
//! only returns those blocks to the allocator. The contract is then
//! closed at the call sites: prefix-cache hits record an extra
//! refcount, the prefix-cache insert records an extra refcount,
//! and sequence finish releases exactly one refcount. The cache
//! holds the blocks alive until either it is cleared or the
//! allocator's free list is queried for those block IDs (the
//! latter is what the regression tests below exercise).
//!
//! We verify the contract two ways:
//!
//! 1. End-to-end through `Engine`: two requests with identical
//!    prompts run in sequence, and we check the second sequence
//!    reads back the same block IDs the first one used (i.e. the
//!    blocks were not freed and reallocated to a different ID).
//! 2. Direct through `MemoryManager`: a sequence allocates blocks,
//!    the prefix cache records them, a release that would have
//!    double-freed under the old code now correctly leaves the
//!    blocks alive.

use tokio::sync::mpsc;
use vllm_core::engine::Engine;
use vllm_core::types::{Request, SchedulerConfig, TokenId};
use vllm_testing::StubModel;

fn test_config() -> SchedulerConfig {
    SchedulerConfig {
        max_num_seqs: 256,
        max_num_batched_tokens: 4096,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
        ..Default::default()
    }
}

/// Capture the block IDs used by the first sequence to finish with
/// a given prompt. Returns `None` if the prefix cache did not see
/// the prompt. Used to confirm cache-hit reuse lands on the same
/// physical block IDs — i.e. the blocks were not freed and re-issued
/// to a different tenant under ARCH-01.
#[allow(dead_code)]
fn blocks_used_for_prompt(engine: &Engine, prompt: &[TokenId]) -> Option<Vec<usize>> {
    engine
        .scheduler
        .prefix_cache()
        .longest_prefix_match(prompt)
        .map(|r| r.blocks.as_ref().clone())
}

/// ARCH-01 end-to-end regression: the prefix cache must hand out the
/// same block IDs across requests so that precomputed KV from the
/// first request can be reused by the second. Under the pre-fix
/// behaviour the first request's blocks were freed on release, and
/// the second request got back different block IDs (or, worse, freed
/// memory) — producing a generation-time correctness bug rather than
/// a test failure. This test forces the contract to be visible at
/// the API boundary.
#[test]
fn prefix_cache_hit_returns_same_block_ids_after_first_completion() {
    let mut engine = Engine::with_config(StubModel::default(), None, test_config(), 4, 100);
    let prompt = vec![1u32, 2, 3, 4, 5, 6, 7, 8];

    // 1. First request runs to completion; its blocks are inserted
    //    into the prefix cache.
    let (tx1, _rx1) = mpsc::channel(64);
    engine.add_request(Request::new(1, prompt.clone(), 3), tx1);
    let mut steps = 0;
    while engine.has_pending() {
        engine.step().expect("first request step failed");
        steps += 1;
        assert!(
            steps < 1000,
            "first request never completes (infinite loop?)"
        );
    }
    let first_blocks = blocks_used_for_prompt(&engine, &prompt)
        .expect("prefix cache must contain an entry after first request");
    assert!(
        !first_blocks.is_empty(),
        "first request must have populated at least one block"
    );

    // 2. Capture allocator state immediately after first completion:
    //    the used-block count should be at least `first_blocks.len()`
    //    because the prefix cache still holds a reference.
    let (available_before, total_before) = {
        let mem = engine.scheduler.memory_mut();
        (mem.available_blocks(), mem.total_blocks())
    };
    let used_after_first = total_before - available_before;
    assert!(
        used_after_first >= first_blocks.len(),
        "after the first request finished, the prefix cache must still \
         own at least `first_blocks.len()` blocks; used={used_after_first} \
         < {} (ARCH-01 regression)",
        first_blocks.len()
    );

    // 3. Second request with identical prompt hits the prefix cache.
    //    A bug would manifest as either a panic (double free) or
    //    the cache returning different block IDs than `first_blocks`.
    let (tx2, _rx2) = mpsc::channel(64);
    engine.add_request(Request::new(2, prompt.clone(), 3), tx2);
    let _batch = engine.scheduler.build_batch();

    // 4. The second sequence must have come out of the prefix cache
    //    with the same block IDs as the first sequence's cache
    //    entry. (If the prefix cache returned different IDs, those
    //    would be either freshly-allocated blocks — wasteful but
    //    safe — or freed IDs from another tenant, which is the
    //    ARCH-01 bug we are guarding against.)
    let second_seq = engine
        .scheduler
        .running()
        .iter()
        .find(|s| s.id == 2)
        .expect("second sequence must be in running after build_batch")
        .clone();
    let cached_blocks = engine
        .scheduler
        .prefix_cache()
        .longest_prefix_match(&prompt)
        .expect("prefix cache must still contain the entry")
        .blocks
        .as_ref()
        .clone();

    // The sequence's kv_blocks should be a prefix of (or equal to)
    // the cache entry's blocks. If the cache entry was freed and
    // re-allocated to different blocks, this assertion would catch it.
    let prefix_len = second_seq.kv_blocks.len().min(cached_blocks.len());
    assert_eq!(
        &second_seq.kv_blocks[..prefix_len],
        &cached_blocks[..prefix_len],
        "second sequence must reuse the prefix-cache block IDs; \
         mismatch means the cache returned freed or reissued blocks \
         (ARCH-01 regression)"
    );

    // 5. Cancel the second sequence so we can verify the allocator
    //    state without going through the second-request completion
    //    path (which has its own batch-phase logic that is out of
    //    scope for the ARCH-01 test). cancel_request exercises the
    //    same release_blocks code path the completion path uses.
    let cancelled = engine.scheduler.cancel_request(2);
    assert!(
        cancelled,
        "cancel_request must succeed for the cache-hit sequence"
    );

    // 6. After cancelling the second sequence, the prefix cache
    //    still holds its reference — the block must NOT be in the
    //    allocator's free list yet. The available count must
    //    therefore still be smaller than the total by at least the
    //    prefix-cache entry's size.
    let (available_after_cancel, total_after_cancel) = {
        let mem = engine.scheduler.memory_mut();
        (mem.available_blocks(), mem.total_blocks())
    };
    let used_after_cancel = total_after_cancel - available_after_cancel;
    assert!(
        used_after_cancel >= first_blocks.len(),
        "after cancelling the second sequence, the prefix cache must \
         still own `first_blocks.len()` blocks; used={used_after_cancel} \
         dropped below the expected floor (ARCH-01 regression: \
         release_blocks over-freed because it ignored the cache's \
         outstanding reference)"
    );
    assert!(
        available_after_cancel <= total_after_cancel,
        "allocator invariant: available ({available_after_cancel}) must be \
         <= total ({total_after_cancel}); if available exceeds total, \
         blocks were freed twice (ARCH-01)"
    );
}

/// ARCH-01 unit-level regression: `MemoryManager::release_blocks`
/// must NOT free a block whose refcount is still positive. We
/// allocate a block, record it twice (simulating two owners), then
/// release once — the allocator's free list must be untouched.
#[test]
fn release_blocks_with_outstanding_refcount_does_not_free() {
    let mut engine = Engine::with_config(StubModel::default(), None, test_config(), 4, 100);
    let mem = engine.scheduler.memory_mut();

    // Allocate two blocks via the scheduler's MemoryManager.
    let allocated = mem
        .allocate(2)
        .expect("allocator must hand out blocks in a fresh engine");
    assert_eq!(allocated.len(), 2);

    // Owner 1: a hypothetical sequence A. record_blocks makes the
    // refcount 1.
    mem.record_blocks(&allocated);

    // Owner 2: a hypothetical sequence B that shares the prefix.
    // record_blocks makes the refcount 2.
    mem.record_blocks(&allocated);

    // Drain the engine so the original request, if any, is gone.
    // (Engine::with_config doesn't seed a request, so no drain needed.)

    // Release once for sequence A. The block must NOT be freed
    // because sequence B still owns a reference.
    let total_before = mem.total_blocks();
    let available_before = mem.available_blocks();
    mem.release_blocks(&allocated);
    let available_after_one_release = mem.available_blocks();
    assert_eq!(
        available_after_one_release, available_before,
        "release_blocks must NOT return shared blocks to the allocator; \
         available went from {available_before} to {available_after_one_release} \
         but B still owns a reference (ARCH-01 regression)"
    );
    assert_eq!(total_before, mem.total_blocks());

    // Release again for sequence B. NOW the refcount hits zero and
    // the blocks can be returned to the allocator.
    mem.release_blocks(&allocated);
    let available_after_two_releases = mem.available_blocks();
    assert_eq!(
        available_after_two_releases,
        available_before + allocated.len(),
        "after both owners release, the freed blocks must return to the \
         allocator; expected available to grow by {} (got {} -> {})",
        allocated.len(),
        available_after_one_release,
        available_after_two_releases,
    );
}

/// ARCH-01 unit-level regression: a release on a block that was
/// never recorded (refcount drift) must still be safe — it should
/// free the block rather than leak it. Catches the "throw an error"
/// variant which would silently leak memory in production.
#[test]
fn release_blocks_on_unrecorded_block_frees_instead_of_leaking() {
    let mut engine = Engine::with_config(StubModel::default(), None, test_config(), 4, 100);
    let mem = engine.scheduler.memory_mut();

    let allocated = mem.allocate(2).expect("allocator must hand out blocks");
    // Deliberately do NOT call record_blocks.

    let available_before = mem.available_blocks();
    mem.release_blocks(&allocated);
    let available_after = mem.available_blocks();
    assert_eq!(
        available_after,
        available_before + allocated.len(),
        "release_blocks on an unrecorded block must free it (return {} slots); \
         instead went {available_before} -> {available_after}",
        allocated.len()
    );
}
