# Fuzz Testing (v29.0 + v30.0 Phase L)

## Why

Property-based tests (v28.0, ADR-016) exercise invariants over **structured
inputs** that pass through our own `Arbitrary` generators. They don't catch
panics in **unstructured, adversarial** inputs that come from external
sources: YAML configs, checkpoint headers, model JSON.

These surfaces are common attack vectors for memory unsafety in parsers
(parser panic → process abort → DoS).

## Targets

| Target | Module | Input | Created |
|--------|--------|-------|---------|
| `app_config_yaml` | `crates/server/src/config.rs` | UTF-8 → `serde_saphyr::from_str::<AppConfig>` | v29.0 |
| `safetensors_header` | `safetensors 0.7` | bytes → `SafeTensors::deserialize` | v29.0 |
| `qwen3_config_json` | `crates/model/src/qwen3/config.rs` | bytes → `serde_json::from_slice::<Qwen3Config>` | v29.0 |

The `fuzz_target_1` placeholder (cargo-fuzz default) is excluded from CI.

## Local Usage

```bash
# Install nightly (required for coverage-guided fuzzing)
rustup install nightly

# Build all fuzz targets (first run downloads toolchain + crates)
just fuzz-build

# Quick smoke: 10s × all targets
just fuzz-smoke

# Focused: 60s on one target
just fuzz app_config_yaml

# List all targets
just fuzz-list

# Replay a crash artifact
just fuzz-repro app_config_yaml fuzz/artifacts/app_config_yaml/crash-deadbeef
```

## CI Integration (v30.0 Phase L)

### PR workflow (`.github/workflows/fuzz.yml`)

Triggers on every PR + push to main. Runs `cargo +nightly fuzz run` for
**30 seconds per target** across the matrix (3 targets). Uploads crash
artifacts on failure. Wall-clock: ~3-5 min total.

Corpus is persisted across runs via `actions/cache` (key includes target
name + SHA, restore-keys fall back to most recent corpus for the target).

### Nightly workflow (`.github/workflows/fuzz-nightly.yml`)

Runs on cron (04:00 UTC daily) or manual `workflow_dispatch`. Runs
**5 minutes per target** (15 min wall-clock for 3 targets). Uses a
separate cache key (`-nightly-` prefix) so nightly and PR corpora
don't overwrite each other.

On completion, uploads the grown corpus as an artifact (manual review
before promoting to PR corpus via cache key).

## Corpus Management

| Location | Format | Lifecycle |
|----------|--------|-----------|
| `fuzz/corpus/<TARGET>/` | gitignored, local seed files | Local dev only |
| GitHub Actions cache | compressed tarball | 7-day TTL, shared across runs |
| GitHub Actions artifact | compressed tarball | 90-day retention, manual review |

Promotion flow: nightly run discovers new coverage → artifact upload →
human review → manual update of cache key to promote to PR corpus.

## Crash Handling

When a crash is found:

1. **Local**: cargo-fuzz writes to `fuzz/artifacts/<TARGET>/<hash>`.
   The original crashing input is preserved there.
2. **CI**: workflow fails; crash artifacts auto-uploaded as GitHub Actions
   artifacts (`fuzz-crash-<TARGET>`).
3. **Replay**: `just fuzz-repro TARGET CRASH_FILE` reruns against the
   crashing input — should fail deterministically.
4. **Fix**: write a regression test (preferably in source-file's
   `#[cfg(test)] mod tests {}`), then fix the panic.
5. **Update corpus**: after fix, the original crash input becomes a
   permanent regression seed (manually add to `fuzz/corpus/<TARGET>/`).

## See also

- v29.0 plan: `docs/superpowers/plans/2026-06-28-v29-fuzz-testing.md`
- ADR-017: fuzz testing strategy
- `fuzz/` directory
- `justfile` `fuzz-*` targets
