# Roadmap: vllm-lite

## Milestones

- ✅ **v13.0 主机部署** — Phases 13.1-13.3 (shipped 2026-04-27)

## Phase History

<details>
<summary>✅ v13.0 主机部署 (Phases 13.1-13.3) — SHIPPED 2026-04-27</summary>

- [x] Phase 13.1: K8s 基础 (10/10 req, K8S-02 partial) — completed 2026-04-27
- [x] Phase 13.2: 高可用 (7/7 req) — completed 2026-04-27
- [x] Phase 13.3: 安全加固 (6/6 req) — completed 2026-04-27

Full details: [.planning/milestones/v13.0-ROADMAP.md](.planning/milestones/v13.0-ROADMAP.md)

</details>

## Progress

| Phase | Milestone | Requirements | Status |
|-------|-----------|--------------|--------|
| 1-11 | Prior milestones | Various | Complete |
| 12 | Advanced features | AWQ/GPTQ, backpressure, predictive batching | Complete |
| 13.1 | K8s 基础 | 9/10 (K8S-02 partial) | Complete |
| 13.2 | 高可用 | 7/7 | Complete |
| 13.3 | 安全加固 | 6/6 | Complete |

---

## Long-term Vision

- Phase 14: Cross-region replication
- WebAssembly support (out of scope for now)
- Multi-tenant KV cache isolation (requires coherence v2)

---

*Roadmap archived: 2026-04-27*
*Full milestone archives: .planning/milestones/*