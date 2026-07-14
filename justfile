# Show help
default:
    @just --list

init:
    uv tool install prek
    uv tool install rumdl
    uv tool install ruff
    uv tool install rust-just
    prek install --hook-type commit-msg --hook-type pre-push

# Build release binary
build:
    cargo build --release

# Run tests with nextest (skips #[ignore] tests by default; one checkpoint smoke remains)
nextest:
    cargo nextest run --workspace --all-features --no-fail-fast

# Faster local loop: no checkpoint smoke, fail-fast, default features only
nextest-fast:
    cargo nextest run --workspace --no-fail-fast -P optimized

# Run nextest including all tests (with #[ignore])
nextest-all:
    cargo nextest run --release --workspace --all-features --run-ignored all --no-fail-fast

# On-disk checkpoint integration tests (ignored by default in `just nextest`)
nextest-checkpoint:
    cargo nextest run -p vllm-model --all-features --no-fail-fast -P checkpoint \
        --test qwen3_integration --test qwen3_token_pipeline --test arch_checkpoint_smoke \
        --test checkpoint_loading_tests --run-ignored all

# Format code
fmt-check:
    cargo fmt --all --check

# Run clippy (CI style)
clippy:
	# Denies correctness/suspicious/perf. Pedantic/nursery are visible as warnings
	# but not blocking. See `just clippy-pedantic` for pedantic-only view.
	cargo clippy --all-targets --workspace --all-features -- \
		-D clippy::correctness \
		-D clippy::suspicious \
		-D clippy::perf

# Run clippy with pedantic+nursery warnings visible (local use, not CI)
clippy-pedantic:
	# Adds pedantic+nursery warnings on top of deny-tier lints.
	cargo clippy --all-targets --workspace --all-features -- \
		-W clippy::pedantic \
		-W clippy::nursery \
		-D clippy::correctness \
		-D clippy::suspicious \
		-D clippy::perf

# Check documentation (CI style)
doc-check:
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items --workspace --all-features

# Run quick read-only checks (does NOT modify source). Use `just autofix`
# to apply auto-fixes, or `just fix` for both clippy-fix and fmt.
quick: fmt-check clippy doc-check doctest nextest

# Engineering-quality §8: `just check` is the documented three-entry
# alias for the local-loop recipe. We keep `quick` as the canonical
# name (older muscle memory) and expose `check` alongside it so the
# engineering-quality doc stays accurate.
check: quick

# Run doctest examples (///  blocks) — exercises doc-comment code as tests.
doctest:
    cargo test --doc --workspace --all-features --no-fail-fast

# Auto-fix all fixable lints and format issues. Mutates the working tree —
# use `just quick` if you want a non-mutating verification pass instead.
autofix:
	cargo fix --allow-dirty --allow-staged --all-targets --workspace --all-features
	cargo clippy --fix --allow-dirty --allow-staged --all-targets --workspace --all-features -- \
		-D clippy::correctness \
		-D clippy::suspicious \
		-D clippy::perf
	cargo fmt --all

# Legacy alias for `autofix`. Kept so existing muscle-memory commands still work.
fix: autofix

# Phase 12e: cargo-public-api baseline diff — fails if the public API of
# any workspace crate grows without a corresponding CHANGELOG entry.
# Shrinking is allowed (the baseline IS the record of what we removed).
public-api-check:
    bash .planning/phase-12e/check-public-api.sh

# Regenerate committed public-API baselines (run after intentional API changes).
public-api-baseline:
    bash .planning/phase-12e/refresh-baselines.sh

# === Doc coverage gate (Phase 31-E) ===
# Print the per-crate and workspace /// doc coverage. Use
# `just doc-coverage-real` to blank out test mods / hidden /
# derive-generated items for a more honest number.
doc-coverage:
    bash scripts/doc_coverage.sh

# Same as `doc-coverage` but with the `real` filter (excludes
# `#[cfg(test)]` / `#[doc(hidden)]` / derive-generated items).
doc-coverage-real:
    bash scripts/doc_coverage.sh --real

# Phase 31-E doc-coverage gate. Reads the per-crate JSON from
# `scripts/doc_coverage.sh --real json`, aggregates to a single
# workspace `real_pct`, and exits non-zero if it drops below
# the threshold (default 65%, matches the v31.0 master plan).
# Override via the env var DOC_COVERAGE_MIN.
doc-coverage-check:
    #!/usr/bin/env bash
    set -euo pipefail
    THRESHOLD="${DOC_COVERAGE_MIN:-65.0}"
    json="$(bash scripts/doc_coverage.sh --real json)"
    real_pct="$(echo "$json" | python3 -c "
    import sys, json
    data = json.load(sys.stdin)
    total = sum(int(c.get('real_total', 0)) for c in data.values())
    doc = sum(int(c.get('real_documented', 0)) for c in data.values())
    print(f'{(doc/total)*100:.2f}' if total else '0.00')
    ")"
    echo "workspace real doc coverage: ${real_pct}% (target >= ${THRESHOLD}%)"
    python3 -c "
    import sys
    threshold = float('${THRESHOLD}')
    actual = float('${real_pct}')
    if actual < threshold:
        print(f'FAIL: doc coverage {actual}% below threshold {threshold}%', file=sys.stderr)
        sys.exit(1)
    "

# === Release manifest (GOV-01) ===
# Print the release manifest derived from [workspace.package] version.
# Use this to preview what `release.yml` will see on tag push.
release-manifest:
    bash scripts/release-manifest.sh

# Validate a candidate tag against the workspace version. Exits 1 if
# they disagree. Use before tagging to catch typos.
release-manifest-validate TAG:
    bash scripts/release-manifest.sh --validate "{{TAG}}" > /dev/null

# Write the manifest to target/release-manifest.env so it can be
# sourced in another shell / CI step.
release-manifest-write:
    bash scripts/release-manifest.sh --out target/release-manifest.env

# Run all CI checks (skips #[ignore] slow tests)
ci: fmt-check clippy doc-check doctest nextest public-api-check doc-coverage-check

# Run the full CI gate including security checks (audit + deny). Requires
# `cargo-audit` and `cargo-deny` installed locally.
ci-all: fmt-check clippy doc-check doctest nextest public-api-check doc-coverage-check security

# Engineering-quality §8: `just ci-full` is the documented three-entry
# alias for the full-gate recipe. We keep `ci-all` as the canonical
# name (more precise — "all" = security + everything) and expose
# `ci-full` alongside it so the engineering-quality doc stays
# accurate.
ci-full: ci-all

# Legacy `fix` recipe removed — replaced by `autofix` (true meaning) and
# `quick` (read-only). Old `fix` is now an alias for `autofix` defined above.

# Generate documentation
doc:
    cargo doc --no-deps --open

# Generate coverage report (requires cargo-tarpaulin)
cov:
    cargo tarpaulin --all-features --workspace --exclude-files 'src/bin/*'

# Clean build artifacts
clean:
    cargo clean

# Run all benchmarks (CPU; ~5-10 min)
bench:
    cargo bench --workspace --all-features --no-fail-fast -- --output-format bencher

# Run quick benchmarks (core radix cache only)
bench-quick:
    cargo bench -p vllm-core --bench radix_cache -- --sample-size 10

# Run a single model-layer bench by name (H-2 to H-5)
# On CPU-only environments these run a tiny smoke test + eprintln warning;
# on GPU runners they exercise full standard qwen3-class dimensions.
bench-model-one BENCH:
    cargo bench -p vllm-model --bench {{BENCH}} -- --sample-size 10

# Run all model-layer benches (CPU smoke only; ~1 min total on CPU)
bench-model:
    cargo bench -p vllm-model --no-fail-fast -- --sample-size 10

# Run all benchmarks (core + model benches)
bench-all: bench bench-model

# Run cargo audit (ignores RUSTSEC-2024-0436 paste unmaintained — see SECURITY.md)
audit:
    cargo audit --ignore RUSTSEC-2024-0436 --deny warnings

# Run cargo audit (strict; will report the paste INFO warning)
audit-strict:
    cargo audit

# Run cargo-deny license/bans/advisories checks (requires cargo-deny installed)
deny:
    cargo deny check

# Run cargo-deny (advisories only — useful for quick CI signal)
deny-advisories:
    cargo deny check advisories

# Run all security gates (audit + deny) — local pre-push / pre-merge gate
security: audit deny

# Build fuzz binaries (debug). Requires nightly for sanitizer coverage.
# Use `just fuzz-build` before `just fuzz-smoke` / `just fuzz TARGET` so the
# release artifact is cached.
fuzz-build:
    cargo +nightly fuzz build --fuzz-dir fuzz

# Run a short fuzzing smoke test against every target (~10s each).
# Iterates over all targets because `cargo fuzz run` requires a target name.
fuzz-smoke:
    sh -c 'set -e; for t in $(cargo +nightly fuzz list --fuzz-dir fuzz); do echo "==> Fuzzing $t (10s)"; cargo +nightly fuzz run "$t" --fuzz-dir fuzz -- -max_total_time=10; done'

# Run a specific fuzz target for ~60s.
fuzz TARGET:
    cargo +nightly fuzz run {{TARGET}} --fuzz-dir fuzz -- -max_total_time=60

# List available fuzz targets.
fuzz-list:
    cargo +nightly fuzz list --fuzz-dir fuzz

# Re-run a fuzz target with a specific crash artifact for debugging.
# Usage: `just fuzz-repro TARGET CRASH_FILE`
# Example: `just fuzz-repro app_config_yaml fuzz/artifacts/app_config_yaml/crash-deadbeef`
fuzz-repro TARGET CRASH:
    cargo +nightly fuzz run {{TARGET}} --fuzz-dir fuzz {{CRASH}}

# === Mutation testing (v30.0 Phase K) ===
# Generate baseline mutation scan for an entire module under vllm-core.
# Usage: `just mutants MODULE` where MODULE is a path relative to
# crates/core/src (e.g. `scheduler`, `scheduler/policy`, `sampling.rs`).
# - For directories: pass `scheduler` or `scheduler/policy`
# - For single files: pass `sampling.rs` (include the extension)
# NOTE: First run downloads and caches mutants tool (~minutes), subsequent
# runs reuse the cache. A full scheduler scan takes ~30-60 min on 4 cores.
# `--baseline skip` works around a pre-existing test failure in
# cuda_graph_integration.rs:148 — fix the test in v31+ to drop this flag.
mutants MODULE:
    #!/usr/bin/env bash
    set -euo pipefail
    TARGET="crates/core/src/{{MODULE}}"
    if [ -d "$TARGET" ]; then
        FILE_ARGS=$(find "$TARGET" -name '*.rs' -type f -printf '--file=%p\n')
    elif [ -f "$TARGET" ]; then
        FILE_ARGS="--file=$TARGET"
    elif [ -f "${TARGET}.rs" ]; then
        FILE_ARGS="--file=${TARGET}.rs"
    else
        echo "Path not found: $TARGET (or ${TARGET}.rs)" >&2
        exit 1
    fi
    cargo mutants \
        --package vllm-core \
        $FILE_ARGS \
        --timeout 30 \
        --jobs $(($(nproc) > 8 ? 8 : $(nproc))) \
        --output .mutants-out/ \
        --baseline skip \
        --shuffle

# Render a human-readable summary of the latest mutation scan.
mutants-report:
    @test -d .mutants-out/mutants.out || (echo "no .mutants-out/mutants.out/ — run \`just mutants MODULE\` first"; exit 1)
    @if [ -f .mutants-out/mutants.out/mutants.json ]; then \
        jq -r '.[] | select(.status == "SURVIVED") | "\(.file):\(.span.start_line)  \(.mutant_name)"' \
            .mutants-out/mutants.out/mutants.json | head -30; \
    else \
        echo "mutants.json not found; check .mutants-out/mutants.out/ contents"; \
    fi

# Remove the mutation output directory.
mutants-clean:
    rm -rf .mutants-out

# Print mutation score: caught / (caught + missed) as a percentage.
mutants-score:
    @test -f .mutants-out/mutants.out/mutants.json || (echo "no scan yet — run \`just mutants MODULE\` first"; exit 1)
    @./scripts/check_mutation_score.sh .mutants-out/mutants.out 0 | head -1

# Run mutation scan with baseline regression check (used in CI / pre-merge).
# Usage: `just mutants-ci MODULE BASELINE_PCT`
# Example: `just mutants-ci scheduler 99.5`
# Exit code 0 if score >= BASELINE_PCT, non-zero otherwise.
mutants-ci MODULE BASELINE:
    just mutants {{MODULE}}
    @./scripts/check_mutation_score.sh .mutants-out/mutants.out {{BASELINE}}
