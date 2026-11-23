#![allow(clippy::module_name_repetitions)]
//! `MemoryBudget` — VRAM budget enforcement for target + concurrent drafts
//!
//! v18.0 Multi-Model Speculative Decoding phase 2 (LIFE-02/03, MEM-01..03).
//!
//! The Engine reserves a chunk of "VRAM" for the target model + its KV cache
//! (one-time at engine startup), and individual drafts reserve and release
//! their estimated footprint (`weight_size_estimate_bytes + kv_blocks *
//! BLOCK_BYTES`) as they are loaded and unloaded.
//!
//! `MemoryBudgetExceeded` is the structured error returned when a draft load
//! would push the budget over its limit.

use crate::scheduler::memory::allocator::BLOCK_BYTES;
use crate::speculative::registry::DraftId;
use std::sync::RwLock;

/// Default bytes per KV block, used for VRAM budget estimation.
///
/// Mirrors `crate::scheduler::memory::allocator::BLOCK_BYTES`. Re-exported
/// here for callers that don't want to depend on the scheduler module.
pub const DEFAULT_BLOCK_BYTES: u64 = BLOCK_BYTES as u64;

/// `MemoryBudgetSnapshot`. See the type definition for fields and behavior.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryBudgetSnapshot {
    /// Total VRAM available to speculative decoding.
    pub total_bytes: u64,
    /// Bytes permanently reserved by the target model + its KV cache.
    pub reserved_target_bytes: u64,
    /// Bytes reserved by currently-loaded draft models.
    pub reserved_drafts_bytes: u64,
    /// Bytes actually consumed by live draft tensors (subset of `reserved_drafts`).
    pub used_drafts_bytes: u64,
    /// Remaining bytes that may still be reserved (`total - reserved_target - reserved_drafts`).
    pub available_bytes: u64,
}

#[derive(Debug, thiserror::Error)]
#[error(
    "memory budget exceeded: requested {requested_bytes} bytes, available {available_bytes} bytes (draft_id={draft_id:?})"
)]
/// `MemoryBudgetExceeded`. See the type definition for fields and behavior.
pub struct MemoryBudgetExceeded {
    /// Bytes the failing reservation attempted to claim.
    pub requested_bytes: u64,
    /// Bytes actually free at the time of the failure.
    pub available_bytes: u64,
    /// Draft id whose load triggered the failure, if any.
    pub draft_id: Option<DraftId>,
}

#[derive(Debug)]
struct MemoryBudgetInner {
    total: u64,
    reserved_target: u64,
    reserved_drafts: u64,
    used_drafts: u64,
}

#[derive(Debug)]
/// Per-request memory budget for speculative decoding. Tracks bytes reserved by target + draft model, plus per-sequence draft-token budget.
pub struct MemoryBudget {
    inner: RwLock<MemoryBudgetInner>,
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self::unlimited()
    }
}

impl MemoryBudget {
    #[must_use]
    /// Construct a memory budget with effectively no cap (returns `usize::MAX`).
    ///
    /// # Panics
    ///
    /// Panics only if `u64::MAX` were rejected by [`Self::new`], which cannot
    /// happen: it rejects 0 alone.
    pub fn unlimited() -> Self {
        // invariant: `u64::MAX` is a non-zero literal; `MemoryBudget::new` only rejects 0.
        Self::new(u64::MAX).expect("u64::MAX always > 0")
    }

    /// Construct a new instance from the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `MemoryBudgetExceeded` when `total_bytes` is 0: a zero budget
    /// could never reserve the target model, so it is rejected at
    /// construction rather than at the first reservation.
    pub const fn new(total_bytes: u64) -> Result<Self, MemoryBudgetExceeded> {
        if total_bytes == 0 {
            return Err(MemoryBudgetExceeded {
                requested_bytes: 0,
                available_bytes: 0,
                draft_id: None,
            });
        }
        Ok(Self {
            inner: RwLock::new(MemoryBudgetInner {
                total: total_bytes,
                reserved_target: 0,
                reserved_drafts: 0,
                used_drafts: 0,
            }),
        })
    }

    pub fn snapshot(&self) -> MemoryBudgetSnapshot {
        let inner = self
            .inner
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let reserved_total = inner.reserved_target + inner.reserved_drafts;
        let available = inner.total.saturating_sub(reserved_total);
        MemoryBudgetSnapshot {
            total_bytes: inner.total,
            reserved_target_bytes: inner.reserved_target,
            reserved_drafts_bytes: inner.reserved_drafts,
            used_drafts_bytes: inner.used_drafts,
            available_bytes: available,
        }
    }

    /// Reserve memory for the target model forward pass.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn reserve_target(&self, bytes: u64) -> Result<(), MemoryBudgetExceeded> {
        let mut inner = self
            .inner
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let current_reserved = inner.reserved_target + inner.reserved_drafts;
        let new_reserved = current_reserved.saturating_add(bytes);
        if new_reserved > inner.total {
            let available = inner.total.saturating_sub(current_reserved);
            return Err(MemoryBudgetExceeded {
                requested_bytes: bytes,
                available_bytes: available,
                draft_id: None,
            });
        }
        inner.reserved_target = inner.reserved_target.saturating_add(bytes);
        drop(inner);
        Ok(())
    }

    pub fn release_target(&self) {
        let mut inner = self
            .inner
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.reserved_target = 0;
    }

    /// Atomically attempt to reserve memory for a draft model, returning the actual reservation on success or `None` on failure.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn try_reserve_draft(
        &self,
        bytes: u64,
        draft_id: Option<DraftId>,
    ) -> Result<(), MemoryBudgetExceeded> {
        let mut inner = self
            .inner
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let current_reserved = inner.reserved_target + inner.reserved_drafts;
        let new_reserved = current_reserved.saturating_add(bytes);
        if new_reserved > inner.total {
            let available = inner.total.saturating_sub(current_reserved);
            return Err(MemoryBudgetExceeded {
                requested_bytes: bytes,
                available_bytes: available,
                draft_id,
            });
        }
        inner.reserved_drafts = inner.reserved_drafts.saturating_add(bytes);
        drop(inner);
        Ok(())
    }

    pub fn release_draft(&self, bytes: u64) {
        let mut inner = self
            .inner
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.reserved_drafts = inner.reserved_drafts.saturating_sub(bytes);
        // Also bring used back down if a draft was previously marked used.
        inner.used_drafts = inner.used_drafts.saturating_sub(bytes);
    }

    pub fn record_draft_kv_growth(&self, delta_bytes: i64) {
        let mut inner = self
            .inner
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if delta_bytes >= 0 {
            // invariant: delta_bytes >= 0 in this branch, so the i64 -> u64
            // cast is sign-safe (saturating_add handles any residual bound).
            #[allow(clippy::cast_sign_loss)]
            let delta = u64::try_from(delta_bytes).unwrap_or(u64::MAX);
            inner.used_drafts = inner.used_drafts.saturating_add(delta);
        } else {
            // invariant: -delta_bytes > 0 in this branch, so the negation is
            // positive and sign-safe for u64.
            #[allow(clippy::cast_sign_loss)]
            let delta = u64::try_from(-delta_bytes).unwrap_or(u64::MAX);
            inner.used_drafts = inner.used_drafts.saturating_sub(delta);
        }
    }

    pub fn total_bytes(&self) -> u64 {
        let inner = self
            .inner
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unlimited_budget_accepts_everything() {
        let budget = MemoryBudget::unlimited();
        budget.reserve_target(10_000_000_000).unwrap();
        budget.try_reserve_draft(1_000_000_000, None).unwrap();
        budget.try_reserve_draft(50_000_000_000, None).unwrap();
        let snap = budget.snapshot();
        assert_eq!(snap.total_bytes, u64::MAX);
        assert!(snap.available_bytes > u64::MAX / 2);
    }

    #[test]
    fn test_reserve_target_drains_budget() {
        let budget = MemoryBudget::new(1000).unwrap();
        budget.reserve_target(400).unwrap();
        budget.try_reserve_draft(300, None).unwrap();
        let snap = budget.snapshot();
        assert_eq!(snap.total_bytes, 1000);
        assert_eq!(snap.reserved_target_bytes, 400);
        assert_eq!(snap.reserved_drafts_bytes, 300);
        assert_eq!(snap.available_bytes, 300);
    }

    /// `reserve_target` must charge target AND draft reservations together.
    ///
    /// The guard is `reserved_target + reserved_drafts + bytes > total`.
    /// `cargo mutants` relaxed the `+` joining the two reservations to `-` and
    /// `*` and the `>` to `>=` / `==`, and all four survived: the existing
    /// tests only reserve ONE of the two kinds before checking the limit, so
    /// the cross-term was always zero and every variant agreed.
    ///
    /// With 300 target + 300 draft already reserved, a 400-byte target request
    /// lands EXACTLY on the 1000-byte total (accepted, so `>` not `>=`), and a
    /// 401-byte one does not fit (rejected, so the sum is not a product or a
    /// difference).
    #[test]
    fn reserve_target_charges_both_reservations() {
        let budget = MemoryBudget::new(1000).unwrap();
        budget.reserve_target(300).unwrap();
        budget.try_reserve_draft(300, None).unwrap();

        // 300 + 300 + 400 == 1000: an exact fit must be accepted.
        budget
            .reserve_target(400)
            .expect("an exact fit must be accepted (`>` not `>=`)");
        let snap = budget.snapshot();
        assert_eq!(snap.reserved_target_bytes, 700);
        assert_eq!(snap.available_bytes, 0);

        // One more byte cannot fit.
        let err = budget
            .reserve_target(1)
            .expect_err("one byte over the total must be rejected");
        assert_eq!(err.requested_bytes, 1);
        assert_eq!(
            err.available_bytes, 0,
            "the error must report the real remaining headroom"
        );

        // A draft reservation must be charged the same way, from the same total.
        let budget = MemoryBudget::new(1000).unwrap();
        budget.reserve_target(600).unwrap();
        budget
            .try_reserve_draft(400, None)
            .expect("600 + 400 is an exact fit");
        assert!(
            budget.try_reserve_draft(1, None).is_err(),
            "the budget is exhausted by the exact fit"
        );
    }

    #[test]
    fn test_reserve_draft_fails_when_exceeds() {
        let budget = MemoryBudget::new(1000).unwrap();
        budget.reserve_target(800).unwrap();
        let err = budget.try_reserve_draft(300, None).unwrap_err();
        assert_eq!(err.requested_bytes, 300);
        assert_eq!(err.available_bytes, 200);
        assert!(err.draft_id.is_none());
    }

    /// The DRAFT reservation path must charge both reservations too.
    ///
    /// `try_reserve_draft` repeats the same `reserved_target + reserved_drafts`
    /// sum. `cargo mutants` replaced that `+` with `-` (and the sibling `>`
    /// with `>=` / `==` at `reserve_target`), surviving because the existing
    /// draft tests never pre-reserve a TARGET: with one side zero, `a + b`,
    /// `a - b`, `a * b` all collapse to the same number.
    #[test]
    fn try_reserve_draft_charges_both_reservations() {
        let budget = MemoryBudget::new(1000).unwrap();
        budget.reserve_target(700).unwrap();

        // 700 + 300 == 1000: an exact fit is accepted.
        budget
            .try_reserve_draft(300, None)
            .expect("an exact fit must be accepted");
        assert_eq!(budget.snapshot().available_bytes, 0);

        let err = budget
            .try_reserve_draft(1, None)
            .expect_err("the budget is exhausted");
        assert_eq!(err.available_bytes, 0);

        // Subtraction instead of addition would report 400 bytes of headroom
        // here (700 - 300) and accept a request that does not fit.
        let budget = MemoryBudget::new(1000).unwrap();
        budget.reserve_target(700).unwrap();
        budget.try_reserve_draft(250, None).unwrap();
        let err = budget
            .try_reserve_draft(100, None)
            .expect_err("700 + 250 + 100 exceeds the 1000-byte total");
        assert_eq!(
            err.available_bytes, 50,
            "the error must report 1000 - 950 = 50 bytes of headroom"
        );
    }

    #[test]
    fn test_release_draft_restores_budget() {
        let budget = MemoryBudget::new(1000).unwrap();
        budget.reserve_target(500).unwrap();
        budget.try_reserve_draft(300, None).unwrap();
        budget.release_draft(300);
        assert_eq!(budget.snapshot().reserved_drafts_bytes, 0);
        // Now we can reserve 500
        budget.try_reserve_draft(500, None).unwrap();
    }

    #[test]
    fn test_snapshot_reflects_state() {
        let budget = MemoryBudget::new(2000).unwrap();
        budget.reserve_target(1000).unwrap();
        budget
            .try_reserve_draft(500, Some(DraftId("a".into())))
            .unwrap();
        budget.record_draft_kv_growth(200);
        let snap = budget.snapshot();
        assert_eq!(snap.total_bytes, 2000);
        assert_eq!(snap.reserved_target_bytes, 1000);
        assert_eq!(snap.reserved_drafts_bytes, 500);
        assert_eq!(snap.used_drafts_bytes, 200);
        assert_eq!(snap.available_bytes, 500);
    }

    #[test]
    fn test_record_draft_kv_growth_updates_used() {
        let budget = MemoryBudget::new(2000).unwrap();
        budget.try_reserve_draft(1000, None).unwrap();
        budget.record_draft_kv_growth(100);
        budget.record_draft_kv_growth(50);
        assert_eq!(budget.snapshot().used_drafts_bytes, 150);
        budget.record_draft_kv_growth(-30);
        assert_eq!(budget.snapshot().used_drafts_bytes, 120);
    }

    #[test]
    fn test_zero_total_budget_errors() {
        assert!(MemoryBudget::new(0).is_err());
    }

    #[test]
    fn test_release_target_resets_to_zero() {
        let budget = MemoryBudget::new(1000).unwrap();
        budget.reserve_target(500).unwrap();
        budget.release_target();
        assert_eq!(budget.snapshot().reserved_target_bytes, 0);
    }

    #[test]
    fn test_release_draft_floors_at_zero() {
        let budget = MemoryBudget::new(1000).unwrap();
        budget.release_draft(500); // never reserved; floors at 0
        assert_eq!(budget.snapshot().reserved_drafts_bytes, 0);
    }

    #[test]
    fn test_record_kv_growth_floors_used_at_zero() {
        let budget = MemoryBudget::new(1000).unwrap();
        budget.record_draft_kv_growth(-1000);
        assert_eq!(budget.snapshot().used_drafts_bytes, 0);
    }
}
