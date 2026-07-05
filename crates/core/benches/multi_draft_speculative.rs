//! Multi-Model Speculative Decoding (v18.0) benchmark.
//!
//! Compares three configurations end-to-end via the v18.0 building blocks:
//! 1. No draft (baseline pure target decode via fallback path)
//! 2. Self-spec draft (v17 baseline)
//! 3. External draft (v18.0 new path)
//!
//! Uses stub backends and the in-memory `DraftModelRegistry`. Real models and
//! tokenizers are not involved — the goal is to validate the orchestration
//! overhead, not the inference speed.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_core::speculative::{
    DraftId, DraftLoader, DraftModelRegistry, DraftRegistryError, DraftResolver, DraftSpec,
    MemoryBudget, ResolvedDraft,
};
use vllm_traits::{BatchOutput, ModelBackend, ModelError, Result as ModelResult, SeqId, TokenId};

// ───────────────────────── Stub Backend ───────────────────────────

struct BenchBackend {
    id: String,
    counter: AtomicU64,
}

impl BenchBackend {
    fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            counter: AtomicU64::new(0),
        }
    }
}

impl ModelBackend for BenchBackend {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> ModelResult<BatchOutput> {
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        // Mix backend id into the token so different backends produce
        // distinguishable outputs (closer to real behavior).
        let id_bias: u32 = self.id.bytes().map(u32::from).sum();
        let token: TokenId =
            id_bias.wrapping_add(u32::try_from(n).expect("bounded bench counter")) % 32000;
        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens: seq_ids.iter().map(|_| token).collect(),
        })
    }
    fn forward_logits(
        &mut self,
        _seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> ModelResult<Vec<Vec<f32>>> {
        Ok(vec![])
    }
    fn embed(
        &mut self,
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> ModelResult<Vec<Vec<f32>>> {
        Ok(vec![])
    }
    fn vocab_size(&self) -> usize {
        32000
    }
    fn num_layers(&self) -> usize {
        1
    }
    fn num_heads(&self) -> usize {
        1
    }
}

// ───────────────────────── Map Loader ─────────────────────────────

struct BenchLoader {
    backends: Mutex<HashMap<DraftId, Box<dyn ModelBackend>>>,
}

impl BenchLoader {
    fn new() -> Self {
        Self {
            backends: Mutex::new(HashMap::new()),
        }
    }
}

impl DraftLoader for BenchLoader {
    fn load(&self, id: &DraftId) -> std::result::Result<Box<dyn ModelBackend>, DraftRegistryError> {
        self.backends.lock().unwrap().remove(id).map_or_else(
            || {
                Err(DraftRegistryError::Model(
                    id.clone(),
                    ModelError::new(format!("no backend for {id}")),
                ))
            },
            Ok,
        )
    }
}

// ───────────────────────── Harness ────────────────────────────────

struct BenchHarness {
    resolver: Arc<DraftResolver>,
}

fn make_harness(config: &str) -> BenchHarness {
    let budget = Arc::new(MemoryBudget::unlimited());
    let registry = Arc::new(DraftModelRegistry::with_budget(budget));
    let metrics = Arc::new(EnhancedMetricsCollector::new());
    let loader = Arc::new(BenchLoader::new());
    let loader_dyn: Arc<dyn DraftLoader> = loader.clone();
    let self_spec: Arc<Mutex<Box<dyn ModelBackend>>> =
        Arc::new(Mutex::new(Box::new(BenchBackend::new("self-spec"))));

    if config == "external_draft" {
        let spec = DraftSpec::new("external", "/nope", 4).with_weight_size(1_000_000_000);
        registry.register(spec).unwrap();
        loader.backends.lock().unwrap().insert(
            DraftId("external".into()),
            Box::new(BenchBackend::new("external")),
        );
    }

    let resolver = Arc::new(DraftResolver::new(
        registry,
        Some(self_spec),
        loader_dyn,
        metrics,
    ));
    BenchHarness { resolver }
}

fn run_iteration(h: &BenchHarness, config: &str) {
    let draft_id = if config == "external_draft" {
        Some(DraftId("external".into()))
    } else {
        None
    };
    // Simulate 16 decode steps. Each step resolves a draft and runs the
    // resolved backend's forward.
    for _ in 0..16 {
        let resolved = h.resolver.resolve(draft_id.as_ref());
        let backend = match resolved {
            ResolvedDraft::External(b) | ResolvedDraft::SelfSpec(b) => b,
            ResolvedDraft::None => continue,
        };
        let mut guard = backend.lock().unwrap();
        let _ = guard.forward(&[1], &[vec![10]], &[vec![0]], &[vec![0]], &[0], &[false]);
    }
}

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_draft_iter_total");

    for config in ["no_draft", "self_spec", "external_draft"] {
        let h = make_harness(config);
        group.bench_with_input(BenchmarkId::from_parameter(config), config, |b, _cfg| {
            b.iter(|| {
                let start = Instant::now();
                run_iteration(&h, config);
                start.elapsed()
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_throughput);
criterion_main!(benches);
