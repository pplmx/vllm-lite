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

# Run tests with nextest (skips #[ignore] tests by default)
nextest:
    cargo nextest run --workspace --all-features --no-fail-fast

# Run nextest including all tests (with #[ignore])
nextest-all:
    cargo nextest run --release --workspace --all-features --run-ignored all --no-fail-fast

# Format code
fmt:
    cargo fmt --all

# Check formatting (CI style)
fmt-check:
    cargo fmt --all --check

# Run clippy (CI style)
clippy:
    cargo clippy --all-targets --workspace --all-features -- -D warnings

# Check documentation (CI style)
doc-check:
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items --workspace --all-features

# Run quick: fix/fmt/clippy will be fixed
quick: fix doc-check nextest

# Run all CI checks including slow/ignored tests
ci: fmt-check clippy doc-check nextest-all

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
