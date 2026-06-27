# ADR-007: Per-Request Draft Routing (RTE-01..03)

**Date:** 2026-06-27
**Status:** Accepted
**Context version:** v18.0

## Context

v18.0 introduced external draft model support — clients can name a specific draft model in their request, and the engine routes that request to the named draft rather than the default self-speculation path. The motivating use cases were:

- **A/B testing** — route a percentage of traffic to draft model A, another percentage to draft model B, and measure acceptance-rate deltas.
- **Per-tenant drafts** — a chat-completions tenant that always uses a domain-specific draft (e.g. code completion) gets that draft; a general-completions tenant falls back to self-spec.
- **Heterogeneous hardware** — some sequences benefit from a heavier draft (more layers, higher acceptance) when VRAM is available; others must use the cheapest possible draft.

The design question is: at what granularity should draft routing happen?

- **Per-engine** — one draft for all requests. Simple, but defeats the use cases above.
- **Per-batch** — the scheduler picks a single draft for the whole batch. Works if all requests in a batch share the same draft, but requires partitioning the batch by draft before scheduling, which fragments batching efficiency.
- **Per-request** — each request carries its own `draft_model_id`; the resolver picks the right backend at step time. Requests with different drafts can interleave in the same batch.

v18.0 chose per-request resolution because it preserves batching efficiency and matches the natural unit of customisation (the request). The requirement set RTE-01..03 (Request-level routing decisions, request-level draft IDs, request-level fallback) drove this.

The resolution itself must satisfy three properties:

1. **Lazy loading** — a draft should only be loaded into VRAM when a request actually needs it (RTE-01).
2. **Graceful fallback** — if the named draft is missing, unloaded, or fails to load, the request silently falls back to self-spec rather than erroring (FALL-01).
3. **Observability** — every resolution (hit, miss, fallback) must emit a metric so operators can audit routing behaviour.

## Decision

Per-request draft routing is implemented by `DraftResolver::resolve(&self, request_draft_id: Option<&DraftId>) -> ResolvedDraft` (`crates/core/src/speculative/draft_resolver.rs:103`). The engine calls `resolve` on every step for every request, but the cost is one hashmap lookup in the common case (draft already loaded).

The resolution algorithm:

```rust
pub fn resolve(&self, request_draft_id: Option<&DraftId>) -> ResolvedDraft {
    // Case 1: no external draft requested → fallback to self-spec or None
    let id = match request_draft_id {
        None => return self.fallback_to_self_spec_or_none(),
        Some(id) => id,
    };

    // Case 2: draft already loaded → return External(backend)
    if let Some(backend) = self.registry.get_loaded_backend(id) {
        return ResolvedDraft::External(backend);
    }

    // Case 3: draft registered but unloaded → try to load via DraftLoader
    if self.registry.contains(id) {
        match self.loader.load(id) {
            Ok(backend) => match self.registry.attach_loaded_budgeted(id, backend) {
                Ok(()) => return ResolvedDraft::External(/* re-fetched */),
                Err(_) => return self.fallback_to_self_spec_or_none(),
            },
            Err(_) => return self.fallback_to_self_spec_or_none(),
        }
    } else {
        // Case 4: unknown draft id → fallback
        self.fallback_to_self_spec_or_none()
    }
}
```

Three outcomes (`ResolvedDraft`):

- `External(Arc<Mutex<Box<dyn ModelBackend>>>)` — use the named draft.
- `SelfSpec(Arc<Mutex<Box<dyn ModelBackend>>>)` — fall back to self-spec.
- `None` — no speculation at all (pure target decode).

The `DraftLoader` trait (`draft_resolver.rs:58`) is the dependency-inversion seam: the server implements it with the real `ModelLoader`; tests use a `StubLoader` that returns canned backends. The resolver doesn't depend on `vllm_model` directly.

The `NoopLoader` (`draft_resolver.rs:67`) is the placeholder for engine construction paths where no server-side loader has been wired yet — every load attempt fails, every request gets self-spec, and the engine behaves as if external drafts weren't registered.

## Rationale

1. **Per-request resolution preserves batching** — the scheduler can mix requests with different drafts in one batch without partitioning.
2. **Per-request matches the unit of customisation** — clients naturally specify drafts per-request, not per-batch.
3. **Lazy loading keeps VRAM low** — drafts are loaded on first use, evicted on refcount-zero (lifecycle in `draft_registry.rs`), never pre-loaded speculatively.
4. **Graceful fallback preserves availability** — a missing or failed draft is a soft failure, not a request error. The user still gets a response (just without speculation).
5. **Metrics on every resolution** — `metrics.inc_draft_resolution("external"|"self_spec"|"none")` and `metrics.inc_draft_load_failure()` make routing behaviour observable.
6. **Trait seam keeps the resolver testable** — `DraftResolver` doesn't know about `vllm_model::loader`; tests inject a stub loader.

Alternatives considered:

- **Per-engine routing** — rejected; defeats A/B testing and per-tenant use cases.
- **Per-batch routing** — rejected; fragments batching efficiency and adds scheduler complexity.
- **Eager load-on-startup** — rejected; wastes VRAM on drafts that may never be requested.
- **Hard error on missing draft** — rejected; violates availability expectation (a missing draft should not 500 a request that would otherwise succeed).
- **Global lock around resolution** — rejected; per-request resolution is on the hot path; lock contention would tank throughput.

## Consequences

**Positive:**

- Mixed-draft batches work transparently — different requests in the same batch can use different drafts without scheduler awareness.
- New drafts can be added at runtime via `DraftModelRegistry::register` without engine restart.
- Fallback semantics mean a typo in `draft_model_id` is non-fatal — the request still completes, just without speculation.
- Per-resolution metrics make it possible to debug "why isn't request X using draft Y?" from production telemetry.
- The `NoopLoader` placeholder lets `Engine::with_drafts_boxed` constructors work before server-side wiring is complete.

**Negative:**

- Per-request resolution is on the hot path — even when the result is "use self-spec" the call goes through `resolve` (one hashmap lookup, one atomic increment).
- The lazy-load path can cause a latency spike on the *first* request for a cold draft — the engine pauses to load weights.
- The fallback chain (`None` → `SelfSpec` → `External` → `SelfSpec`) is subtle; reading the resolver code requires understanding all four cases (1: None requested; 2: loaded; 3: registered but unloaded; 4: unknown id).
- FALL-02 (runtime draft failure) is *not* handled by the resolver — the engine handles it. This split is intentional but increases the surface area a reader must understand.

**Mitigations / migration paths:**

- The lazy-load latency spike can be hidden by warming popular drafts at startup via `DraftModelRegistry::preload`.
- The four-case algorithm is documented in `draft_resolver.rs:103-165` with explicit case comments.
- The fallback-to-self-spec path is the default; behaviour is "fail safe" by design.
- If per-request resolution becomes a bottleneck, the resolver can be cached per-request (memoize the `ResolvedDraft` for the lifetime of the request) without changing the public API.
