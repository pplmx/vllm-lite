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

# Run all tests (skips #[ignore] tests by default)
test:
    cargo test --workspace

# Run tests with nextest (skips #[ignore] tests by default)
nextest:
    cargo nextest run --workspace

# Run nextest including all tests (with #[ignore])
nextest-all:
    cargo nextest run --workspace --all-features

# Format code
fmt:
    cargo fmt --all

# Check formatting (CI style)
fmt-check:
    cargo fmt --all --check

# Run clippy (CI style)
clippy:
    cargo clippy --all-targets --workspace -- -D warnings

# Check documentation (CI style)
doc-check:
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items --workspace

# Run all CI checks locally (excluding msrv-check - requires special workspace setup)
# Uses nextest which skips #[ignore] tests by default
ci: fmt-check clippy doc-check test

# Run all CI checks including slow/ignored tests
ci-all: fmt-check clippy doc-check nextest

# Auto-fix clippy warnings and format
fix:
    cargo fix --allow-dirty --allow-staged --all-targets --workspace
    cargo clippy --fix --allow-dirty --allow-staged --all-targets --workspace -- -D warnings
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
