# Roadmap: v16.0 Speculative Decoding

**Created:** 2026-04-28
**Milestone:** v16.0
**Goal:** Implement draft-then-verify token generation for 2-3x throughput improvements on repetitive content.

**Status:** Shipped ✅

---

## Proposed Roadmap

**4 phases** | **17 requirements mapped** | All covered ✓

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|------------------|
| 16.1 | Architecture | Core speculation infrastructure | SPEC-01 | 5 criteria |
| 16.2 | Draft Model | Self-speculation implementation | SPEC-02 | 4 criteria |
| 16.3 | Verification | Parallel verification + early exit | SPEC-03 | 4 criteria |
| 16.4 | Benchmarks | Performance validation | SPEC-04 | 4 criteria |

---

## Phase Details

### Phase 16.1: Architecture

**Goal:** Define core speculation infrastructure with DraftVerifier trait, SpeculativeModel wrapper, and configuration.

**Requirements:** SPEC-01.1, SPEC-01.2, SPEC-01.3, SPEC-01.4

**Success Criteria:**
1. DraftVerifier trait defined with `generate_draft`, `verify`, and `accept` methods
2. SpeculativeModel struct wraps ModelBackend and manages speculation lifecycle
3. SpeculationConfig supports draft count, max depth, temperature
4. RejectionStrategy enum with TokenLevel and BlockLevel variants
5. Unit tests for DraftVerifier trait implementation

---

### Phase 16.2: Draft Model

**Goal:** Implement self-speculation using same model with reduced layer count.

**Requirements:** SPEC-01.5, SPEC-02.1, SPEC-02.2, SPEC-02.3, SPEC-02.4

**Success Criteria:**
1. Self-speculation uses first N layers of target model (configurable per architecture)
2. Layer count configured via config.json or Architecture trait
3. Draft model shares weight tensors with target (zero copy)
4. Draft sampling uses configurable temperature with top-k filtering
5. KV cache reuse: accepted draft KV blocks appended to target KV cache

---

### Phase 16.3: Verification

**Goal:** Implement parallel verification with early termination and KV cache management.

**Requirements:** SPEC-03.1, SPEC-03.2, SPEC-03.3, SPEC-03.4

**Success Criteria:**
1. Parallel verification: single forward pass evaluates all draft tokens
2. Token acceptance: compare target probability vs draft probability, accept if target p > draft p
3. Early termination: stop verification at first rejection, return accepted tokens
4. KV cache append: accepted draft KV blocks copied/appended to target cache

---

### Phase 16.4: Benchmarks

**Goal:** Validate speculation performance with throughput, acceptance rate, and memory metrics.

**Requirements:** SPEC-04.1, SPEC-04.2, SPEC-04.3, SPEC-04.4

**Success Criteria:**
1. Throughput benchmark: >50% speedup on repetitive coding tasks vs standard decoding
2. Acceptance rate: >70% of draft tokens accepted on structured output tasks
3. Memory overhead: <20% additional KV cache memory per request
4. Latency: P99 < 2x standard decoding latency for per-token generation

---

## Dependencies

```
Phase 16.1 (Architecture)
    ↓
Phase 16.2 (Draft Model) ← Phase 16.1
    ↓
Phase 16.3 (Verification) ← Phase 16.2
    ↓
Phase 16.4 (Benchmarks) ← Phase 16.3
```

---

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| SPEC-01.1 | 16.1 | Pending |
| SPEC-01.2 | 16.1 | Pending |
| SPEC-01.3 | 16.1 | Pending |
| SPEC-01.4 | 16.1 | Pending |
| SPEC-01.5 | 16.2 | Pending |
| SPEC-02.1 | 16.2 | Pending |
| SPEC-02.2 | 16.2 | Pending |
| SPEC-02.3 | 16.2 | Pending |
| SPEC-02.4 | 16.2 | Pending |
| SPEC-03.1 | 16.3 | Pending |
| SPEC-03.2 | 16.3 | Pending |
| SPEC-03.3 | 16.3 | Pending |
| SPEC-03.4 | 16.3 | Pending |
| SPEC-04.1 | 16.4 | Pending |
| SPEC-04.2 | 16.4 | Pending |
| SPEC-04.3 | 16.4 | Pending |
| SPEC-04.4 | 16.4 | Pending |

---

*Roadmap created: 2026-04-28*
*Last updated: 2026-04-28*
