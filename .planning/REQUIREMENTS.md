# Requirements: v16.0 Speculative Decoding

**Milestone:** v16.0
**Created:** 2026-04-28

---

## Requirements

### Architecture (SPEC-01)

- [ ] **SPEC-01.1**: DraftVerifier trait defines draft model contract with `generate_draft` and `verify` methods
- [ ] **SPEC-01.2**: SpeculativeModel wrapper wraps ModelBackend with speculative execution logic
- [ ] **SPEC-01.3**: SpeculationConfig allows configuring draft count, max depth, temperature
- [ ] **SPEC-01.4**: RejectionStrategy enum with Accepted/N rejected strategies (token-level, block-level)
- [ ] **SPEC-01.5**: KV cache reuse across draft verification pass (no recomputation of accepted prefixes)

### Draft Model (SPEC-02)

- [ ] **SPEC-02.1**: Self-speculation using same model with reduced layer count (e.g., 4 layers for 32-layer model)
- [ ] **SPEC-02.2**: Layer count configuration per model in config.json
- [ ] **SPEC-02.3**: Draft model shares weights with target model (no extra memory for weights)
- [ ] **SPEC-02.4**: Draft sampling with configurable temperature (lower temp = more conservative)

### Verification (SPEC-03)

- [ ] **SPEC-03.1**: Parallel verification using target model on all draft tokens simultaneously
- [ ] **SPEC-03.2**: Token acceptance based on target vs draft probability comparison
- [ ] **SPEC-03.3**: Early termination when token rejected (no verification of subsequent draft tokens)
- [ ] **SPEC-03.4**: KV cache management for verification pass (append draft KV to target KV)

### Benchmarks (SPEC-04)

- [ ] **SPEC-04.1**: Throughput benchmark comparing speculative vs standard decoding on repetitive tasks
- [ ] **SPEC-04.2**: Acceptance rate metrics (percentage of draft tokens accepted)
- [ ] **SPEC-04.3**: Memory overhead measurement (KV cache, overhead per request)
- [ ] **SPEC-04.4**: Latency percentiles (P50/P95/P99) for first token and per-token

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
