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

# Run quick: fix/fmt/clippy will be fixed
quick: fix doc-check nextest

# Run all CI checks (skips #[ignore] slow tests)
ci: fmt-check clippy doc-check nextest

# Auto-fix clippy warnings and format
fix:
	cargo fix --allow-dirty --allow-staged --all-targets --workspace --all-features
	cargo clippy --fix --allow-dirty --allow-staged --all-targets --workspace --all-features -- \
		-D clippy::correctness \
		-D clippy::suspicious \
		-D clippy::perf
	cargo fmt --all

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

# === Mutation testing (v30.0 Phase K) ===
# Generate baseline mutation scan for an entire module under vllm-core.
# Usage: `just mutants MODULE` where MODULE is a path relative to
# crates/core/src (e.g. `scheduler`, `scheduler/policy`, `sampling`).
# NOTE: First run downloads and caches mutants tool (~minutes), subsequent
# runs reuse the cache. A full scheduler scan takes ~30-60 min on 4 cores.
# `--baseline skip` works around a pre-existing test failure in
# cuda_graph_integration.rs:148 — fix the test in v31+ to drop this flag.
mutants MODULE:
    cargo mutants \
        --package vllm-core \
        --file "crates/core/src/{{MODULE}}/**/*.rs" \
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
