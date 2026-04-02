# Show help
default:
    @just --list

# Build release binary
build:
    cargo build --release

# Run all tests
test:
    cargo test --all-features --workspace

# Run tests by cargo-nextest (a much more modern test runner)
nextest:
    cargo nextest run --all-features --workspace

# Format code
fmt:
    cargo fmt --all

# Check formatting (CI style)
fmt-check:
    cargo fmt --all --check

# Run clippy (CI style)
clippy:
    cargo clippy --all-targets --all-features --workspace -- -D warnings

# Check documentation (CI style)
doc-check:
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items --all-features --workspace

# Run all CI checks locally (excluding msrv-check - requires special workspace setup)
ci: fmt-check clippy doc-check test

# Check Minimum Supported Rust Version (MSRV)
msrv-check:
    cargo msrv verify

# Auto-fix clippy warnings and format
fix:
    cargo clippy --fix --allow-dirty --allow-staged --all-targets --all-features --workspace -- -D warnings
    cargo fmt --all

# Generate documentation
doc:
    cargo doc --no-deps --open

# Generate coverage report (requires cargo-tarpaulin)
coverage:
    cargo tarpaulin --all-features --workspace --exclude-files 'src/bin/*'

# Clean build artifacts
clean:
    cargo clean
