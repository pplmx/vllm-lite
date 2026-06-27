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
# Denies correctness/suspicious/perf. Pedantic/nursery are visible as warnings
# but not blocking. See `just clippy-pedantic` for pedantic-only view.
clippy:
	cargo clippy --all-targets --workspace --all-features -- \
		-D clippy::correctness \
		-D clippy::suspicious \
		-D clippy::perf

# Run clippy with pedantic+nursery warnings visible (local use, not CI)
clippy-pedantic:
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
    cargo clippy --fix --allow-dirty --allow-staged --all-targets --workspace --all-features -- -D warnings
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
