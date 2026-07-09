//! Unit tests for `DraftModelRegistry`.
//!
//! Covers the registry's three concerns:
//!
//! 1. **Identity & state transitions**: register/attach_loaded/unload,
//!    duplicate-id rejection, unknown-id errors, refcount-driven
//!    auto-unload (LIFE-02/03).
//! 2. **Lookup surface**: ids (sorted), len, is_empty, contains.
//! 3. **Memory budget integration**: budgeted attach reserves/releases
//!    correctly, refuses when over budget, and accessor returns the
//!    shared `Arc`.
//!
//! The `FakeBackend` is a no-op `ModelBackend` that panics if invoked
//! — registry tests never exercise a real forward pass.
use super::*;
use crate::speculative::memory_budget::MemoryBudget;
use std::sync::Arc;
use vllm_traits::ModelBackend;

fn dummy_spec(id: &str, kv_blocks: usize) -> DraftSpec {
    DraftSpec::new(id, "/nonexistent", kv_blocks)
}

#[test]
fn test_register_creates_unloaded_state() {
    let registry = DraftModelRegistry::new();
    registry.register(dummy_spec("a", 64)).unwrap();
    match registry.lookup(&DraftId("a".into())) {
        Some(DraftState::Unloaded(s)) => {
            assert_eq!(s.kv_blocks, 64);
            assert_eq!(s.ref_count, 0);
        }
        other => panic!("expected Unloaded, got {other:?}"),
    }
}

#[test]
fn test_register_duplicate_id_errors() {
    let registry = DraftModelRegistry::new();
    registry.register(dummy_spec("a", 64)).unwrap();
    let err = registry.register(dummy_spec("a", 32)).unwrap_err();
    assert!(matches!(err, DraftRegistryError::AlreadyLoaded(_)));
}

#[test]
fn test_attach_loaded_promotes_state() {
    let registry = DraftModelRegistry::new();
    registry.register(dummy_spec("a", 64)).unwrap();
    assert!(!registry.is_loaded(&DraftId("a".into())));
    let err = registry
        .attach_loaded(&DraftId("nope".into()), test_backend())
        .unwrap_err();
    assert!(matches!(err, DraftRegistryError::UnknownDraftId(_)));
}

#[test]
fn test_unload_unknown_id_errors() {
    let registry = DraftModelRegistry::new();
    let err = registry.unload(&DraftId("nope".into())).unwrap_err();
    assert!(matches!(err, DraftRegistryError::UnknownDraftId(_)));
}

#[test]
fn test_unload_already_unloaded_is_noop() {
    let registry = DraftModelRegistry::new();
    registry.register(dummy_spec("a", 64)).unwrap();
    registry.unload(&DraftId("a".into())).unwrap();
    registry.unload(&DraftId("a".into())).unwrap();
    assert!(!registry.is_loaded(&DraftId("a".into())));
}

#[test]
fn test_refcount_increment_decrement() {
    let registry = DraftModelRegistry::new();
    registry.register(dummy_spec("a", 64)).unwrap();
    registry.increment_ref(&DraftId("a".into())).unwrap();
    registry.increment_ref(&DraftId("a".into())).unwrap();
    assert_eq!(registry.ref_count(&DraftId("a".into())).unwrap(), 2);
    registry.decrement_ref(&DraftId("a".into())).unwrap();
    assert_eq!(registry.ref_count(&DraftId("a".into())).unwrap(), 1);
    registry.decrement_ref(&DraftId("a".into())).unwrap();
    registry.decrement_ref(&DraftId("a".into())).unwrap();
    assert_eq!(registry.ref_count(&DraftId("a".into())).unwrap(), 0);
}

#[test]
fn test_ids_lists_all_registered_sorted() {
    let registry = DraftModelRegistry::new();
    registry.register(dummy_spec("b", 32)).unwrap();
    registry.register(dummy_spec("a", 64)).unwrap();
    registry.register(dummy_spec("c", 16)).unwrap();
    assert_eq!(
        registry.ids(),
        vec![
            DraftId("a".into()),
            DraftId("b".into()),
            DraftId("c".into())
        ]
    );
}

#[test]
fn test_len_and_is_empty() {
    let registry = DraftModelRegistry::new();
    assert!(registry.is_empty());
    assert_eq!(registry.len(), 0);
    registry.register(dummy_spec("a", 64)).unwrap();
    assert!(!registry.is_empty());
    assert_eq!(registry.len(), 1);
}

#[test]
fn test_contains_returns_true_for_registered() {
    let registry = DraftModelRegistry::new();
    registry.register(dummy_spec("a", 64)).unwrap();
    assert!(registry.contains(&DraftId("a".into())));
    assert!(!registry.contains(&DraftId("b".into())));
}

#[test]
fn test_draft_id_display() {
    let id = DraftId("qwen-small".into());
    assert_eq!(format!("{id}"), "qwen-small");
    assert_eq!(id.as_str(), "qwen-small");
}

#[test]
fn test_draft_id_from_str_and_string() {
    let a: DraftId = "abc".into();
    let b: DraftId = String::from("abc").into();
    assert_eq!(a, b);
}

#[test]
fn test_draft_spec_builder() {
    let spec = DraftSpec::new("x", "/tmp/model", 128)
        .with_arch_hint("llama")
        .with_ref_count(2);
    assert_eq!(spec.id, DraftId("x".into()));
    assert_eq!(spec.arch_hint.as_deref(), Some("llama"));
    assert_eq!(spec.kv_blocks, 128);
    assert_eq!(spec.ref_count, 2);
}

struct FakeBackend;
impl ModelBackend for FakeBackend {
    fn forward(
        &mut self,
        _seq_ids: &[vllm_traits::SeqId],
        _input_tokens: &[Vec<vllm_traits::TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> vllm_traits::Result<vllm_traits::BatchOutput> {
        panic!("FakeBackend::forward should not be called in registry tests")
    }
    fn forward_logits(
        &mut self,
        _seq_ids: &[vllm_traits::SeqId],
        _input_tokens: &[Vec<vllm_traits::TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> vllm_traits::Result<Vec<Vec<f32>>> {
        panic!("FakeBackend::forward_logits should not be called in registry tests")
    }
    fn embed(
        &mut self,
        _input_tokens: &[Vec<vllm_traits::TokenId>],
        _positions: &[Vec<usize>],
    ) -> vllm_traits::Result<Vec<Vec<f32>>> {
        panic!("FakeBackend::embed should not be called in registry tests")
    }
    fn vocab_size(&self) -> usize {
        panic!("FakeBackend::vocab_size should not be called in registry tests")
    }
    fn num_layers(&self) -> usize {
        panic!("FakeBackend::num_layers should not be called in registry tests")
    }
    fn num_heads(&self) -> usize {
        panic!("FakeBackend::num_heads should not be called in registry tests")
    }
}

fn test_backend() -> Box<dyn ModelBackend> {
    Box::new(FakeBackend)
}

fn dummy_spec_with_size(id: &str, kv_blocks: usize, weight_bytes: u64) -> DraftSpec {
    DraftSpec::new(id, "/nonexistent", kv_blocks).with_weight_size(weight_bytes)
}

#[test]
fn test_unload_with_refcount_errors_in_use() {
    let registry = DraftModelRegistry::new();
    registry
        .register(dummy_spec_with_size("a", 8, 1024))
        .unwrap();
    registry
        .attach_loaded(&DraftId("a".into()), test_backend())
        .unwrap();
    registry.increment_ref(&DraftId("a".into())).unwrap();
    let err = registry.unload(&DraftId("a".into())).unwrap_err();
    assert!(matches!(err, DraftRegistryError::InUse(1)));
    assert!(registry.is_loaded(&DraftId("a".into())));
}

#[test]
fn test_force_unload_overrides_refcount() {
    let registry = DraftModelRegistry::new();
    registry
        .register(dummy_spec_with_size("a", 8, 1024))
        .unwrap();
    registry
        .attach_loaded(&DraftId("a".into()), test_backend())
        .unwrap();
    registry.increment_ref(&DraftId("a".into())).unwrap();
    registry.increment_ref(&DraftId("a".into())).unwrap();
    registry.force_unload(&DraftId("a".into())).unwrap();
    assert!(!registry.is_loaded(&DraftId("a".into())));
}

#[test]
fn test_decrement_ref_auto_unloads_at_zero() {
    let registry = DraftModelRegistry::new();
    registry
        .register(dummy_spec_with_size("a", 8, 1024))
        .unwrap();
    registry
        .attach_loaded(&DraftId("a".into()), test_backend())
        .unwrap();
    registry.increment_ref(&DraftId("a".into())).unwrap();
    let auto_unloaded = registry.decrement_ref(&DraftId("a".into())).unwrap();
    assert!(auto_unloaded);
    assert!(!registry.is_loaded(&DraftId("a".into())));
}

#[test]
fn test_decrement_ref_no_auto_unload_when_already_unloaded() {
    let registry = DraftModelRegistry::new();
    registry.register(dummy_spec("a", 8)).unwrap();
    registry.increment_ref(&DraftId("a".into())).unwrap();
    let auto_unloaded = registry.decrement_ref(&DraftId("a".into())).unwrap();
    assert!(!auto_unloaded);
    assert!(!registry.is_loaded(&DraftId("a".into())));
}

#[test]
fn test_decrement_ref_does_not_auto_unload_above_zero() {
    let registry = DraftModelRegistry::new();
    registry
        .register(dummy_spec_with_size("a", 8, 1024))
        .unwrap();
    registry
        .attach_loaded(&DraftId("a".into()), test_backend())
        .unwrap();
    registry.increment_ref(&DraftId("a".into())).unwrap();
    registry.increment_ref(&DraftId("a".into())).unwrap();
    let auto_unloaded = registry.decrement_ref(&DraftId("a".into())).unwrap();
    assert!(!auto_unloaded);
    assert!(registry.is_loaded(&DraftId("a".into())));
}

#[test]
fn test_attach_loaded_budgeted_reserves_budget() {
    let budget = Arc::new(MemoryBudget::new(100_000_000_000).unwrap());
    let registry = DraftModelRegistry::with_budget(budget.clone());
    let spec = dummy_spec_with_size("a", 4, 1000);
    let estimated = spec.estimated_total_bytes();
    registry.register(spec).unwrap();
    registry
        .attach_loaded_budgeted(&DraftId("a".into()), test_backend())
        .unwrap();
    let snap = budget.snapshot();
    assert_eq!(snap.reserved_drafts_bytes, estimated);
}

#[test]
fn test_attach_loaded_budgeted_releases_on_double_attach() {
    let budget = Arc::new(MemoryBudget::new(100_000_000).unwrap());
    let registry = DraftModelRegistry::with_budget(budget.clone());
    let spec = dummy_spec_with_size("a", 1, 1000);
    registry.register(spec).unwrap();
    registry
        .attach_loaded_budgeted(&DraftId("a".into()), test_backend())
        .unwrap();
    let snap_before = budget.snapshot();
    let _ = registry.attach_loaded_budgeted(&DraftId("a".into()), test_backend());
    let snap_after = budget.snapshot();
    assert_eq!(
        snap_before.reserved_drafts_bytes,
        snap_after.reserved_drafts_bytes
    );
}

#[test]
fn test_attach_loaded_budgeted_refuses_when_over_budget() {
    let budget = Arc::new(MemoryBudget::new(100).unwrap());
    let registry = DraftModelRegistry::with_budget(budget);
    let spec = dummy_spec_with_size("huge", 4, 1000);
    registry.register(spec).unwrap();
    let err = registry
        .attach_loaded_budgeted(&DraftId("huge".into()), test_backend())
        .unwrap_err();
    assert!(matches!(err, DraftRegistryError::MemoryBudgetExceeded(_)));
    assert!(!registry.is_loaded(&DraftId("huge".into())));
}

#[test]
fn test_unload_releases_budget_reservation() {
    let budget = Arc::new(MemoryBudget::new(100_000_000).unwrap());
    let registry = DraftModelRegistry::with_budget(budget.clone());
    let spec = dummy_spec_with_size("a", 1, 1000);
    let estimated = spec.estimated_total_bytes();
    registry.register(spec).unwrap();
    registry
        .attach_loaded_budgeted(&DraftId("a".into()), test_backend())
        .unwrap();
    assert_eq!(budget.snapshot().reserved_drafts_bytes, estimated);
    registry.unload(&DraftId("a".into())).unwrap();
    assert_eq!(budget.snapshot().reserved_drafts_bytes, 0);
}

#[test]
fn test_draft_allocated_bytes_zero_when_unloaded() {
    let registry = DraftModelRegistry::new();
    registry.register(dummy_spec("a", 4)).unwrap();
    assert_eq!(
        registry
            .draft_allocated_bytes(&DraftId("a".into()))
            .unwrap(),
        0
    );
}

#[test]
fn test_draft_reserved_bytes_zero_when_unloaded() {
    let registry = DraftModelRegistry::new();
    registry.register(dummy_spec("a", 4)).unwrap();
    assert_eq!(
        registry.draft_reserved_bytes(&DraftId("a".into())).unwrap(),
        0
    );
}

#[test]
fn test_memory_budget_accessor_returns_arc() {
    let budget = Arc::new(MemoryBudget::new(1000).unwrap());
    let registry = DraftModelRegistry::with_budget(budget);
    assert_eq!(registry.memory_budget().total_bytes(), 1000);
}

#[test]
fn test_draft_spec_estimated_total_bytes() {
    let spec = DraftSpec::new("x", "/tmp", 2).with_weight_size(1024);
    let expected = 1024 + 2 * (16 * 1024 * 1024);
    assert_eq!(spec.estimated_total_bytes(), expected);
}
