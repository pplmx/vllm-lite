# Phase 12e — Public API CI Gate

## Baselines

Committed snapshots of `cargo public-api -p vllm-<crate> --simplified`:

| File | Crate |
|------|-------|
| `traits.txt` | vllm-traits |
| `core.txt` | vllm-core |
| `model.txt` | vllm-model |
| `server.txt` | vllm-server |
| `dist.txt` | vllm-dist |
| `testing.txt` | vllm-testing |

## CI / local

```bash
just public-api-check          # strict — fails on growth without CHANGELOG
bash .planning/phase-12e/check-public-api.sh --no-fail   # report only
```

## Refresh baselines (intentional API change)

After updating `CHANGELOG.md` with the new public items:

```bash
just public-api-baseline
git add .planning/phase-12e/*.txt CHANGELOG.md
```

Requires `cargo install cargo-public-api --locked` (v0.52+).
