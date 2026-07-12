# Contributing to vLLM-lite

Thank you for your interest in contributing to vLLM-lite! This guide will help you get started.

## Prerequisites

Before building from source, ensure you have:

- **Rust 1.88+** (MSRV; declared in root `Cargo.toml` `[workspace.package].rust-version`)
  — Install via [rustup](https://rustup.rs/)
- **CUDA 12.1+** - Optional, for GPU support
- **CMake 3.18+** - For building some dependencies
- **cargo-nextest** - Required for the test runner (`cargo install cargo-nextest`)

```bash
# Verify installation
rustc --version   # Should be 1.88 or higher
cargo --version
cargo nextest --version
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/pplmx/vllm-lite.git
cd vllm-lite

# Build
cargo build --workspace

# Run tests
cargo test --workspace

# Run the server
cargo run -p vllm-server
```

## Development Workflow

1. **Fork & Clone**
    - Fork the repository on GitHub
    - Clone your fork: `git clone https://github.com/YOUR_USERNAME/vllm-lite.git`

2. **Create a Branch**

   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

3. **Make Changes**
    - Follow the coding standards below
    - Add tests for new features
    - Keep commits atomic and focused

4. **Test Your Changes**

   The canonical pre-push gate is `just ci`, which runs the same checks
   GitHub Actions runs on PRs:

   ```bash
   just ci              # fmt-check + clippy + doc-check + nextest
   just ci-all          # same + security gates (audit + deny)
   ```

   CI has two parallel jobs:
   - **`ci`** — default features (CPU-only; matches `just quick` without
     `--all-features`).
   - **`ci-all-features`** — `--all-features` clippy + doc-check build
     (matches `just ci` / `just clippy` exactly). Catches feature-conditional
     regressions that the default-features job would miss (CI-01).

   Or run individual steps:

   ```bash
   # Format check
   cargo fmt --all --check

   # Lint (tiered deny; matches CI exactly)
   cargo clippy --all-targets --workspace --all-features -- \
       -D clippy::correctness \
       -D clippy::suspicious \
       -D clippy::perf

   # Run tests with nextest (skips #[ignore] by default)
   cargo nextest run --workspace --all-features --no-fail-fast
   ```

5. **Submit a Pull Request**
    - Push to your fork: `git push -u origin feature/your-feature-name`
    - Open a PR against `main`
    - Reference related issues with `Fixes #123` or `Refs #123`
    - Make sure all CI checks are green before requesting review
    - Address review feedback with follow-up commits (don't force-push during review)

## Coding Standards

- **Formatting**: Run `cargo fmt --all` before committing
- **Linting**: `cargo clippy --workspace --all-targets --all-features -- -D clippy::correctness -D clippy::suspicious -D clippy::perf` must pass (matches `just clippy`)
- **Testing**: Add tests for new features; `cargo nextest run --workspace --all-features` must pass
- **Documentation**: Document public APIs with `///` doc comments. Public types and functions **must** have at least one doc-comment sentence; field-level docs on `pub` fields are encouraged (see existing crates for the convention)
- **Errors**: All public error types are typed `thiserror::Error` enums. Do not introduce `Box<dyn Error>` in public APIs (see CLAUDE.md)

## Commit Message Format

```text
<type>(<scope>): <subject>
```

| Type     | Description           |
| -------- | --------------------- |
| feat     | New feature           |
| fix      | Bug fix               |
| refactor | Code restructuring    |
| test     | Adding/updating tests |
| docs     | Documentation         |
| chore    | Maintenance           |

**Example**:

```text
feat(scheduler): add decode-priority batching

- Prioritize decode sequences over prefill
- Add max_num_batched_tokens limit
- Fix chunked prefill tracking
```

## Project Structure

```text
vllm-lite/
├── crates/
│   ├── traits/      # Interface definitions (ModelBackend, shared types)
│   ├── core/        # Engine, Scheduler, KV Cache, metrics
│   ├── model/       # Model implementations, kernels (Qwen3, Llama, Mistral)
│   ├── dist/        # Tensor / pipeline parallelism (feature-gated: multi-node)
│   ├── server/      # HTTP API (OpenAI-compatible), auth, security
│   └── testing/     # Test harness, request factories, slow-model stubs
├── fuzz/            # cargo-fuzz targets (nightly-only)
├── docs/            # Design documents, ADRs, tutorials
└── scripts/         # CI helpers (mutation score check, etc.)
```

## Testing

```bash
# Run all fast tests (default — skips #[ignore] slow tests)
just nextest

# Run all tests including #[ignore] (release mode, slow)
just nextest-all

# Run a specific crate
cargo nextest run -p vllm-core

# Run with output visible
cargo nextest run --workspace -- --nocapture

# Run standard `cargo test` (when nextest isn't available)
cargo test --workspace
```

Test organization conventions:

- Unit tests for module-private behavior live in `#[cfg(test)] mod tests`
  blocks inside the source file.
- Cross-module integration tests live in `crates/<crate>/tests/<topic>.rs`.
- Doctest examples in `///` doc comments are executed by CI via the
  `doc-check` step. They count as part of the test suite.

## Tutorials

A guided, step-by-step path for new contributors lives in
[`docs/tutorial/`](docs/tutorial/):

1. [Setup & Build](docs/tutorial/01-setup.md) — clone, build, verify
2. [Load a Model](docs/tutorial/02-load-model.md) — `ModelLoader` usage
3. [Run Inference](docs/tutorial/03-inference.md) — request lifecycle
4. [Customize the Engine](docs/tutorial/04-customize.md) — custom strategies
5. [Production Deployment](docs/tutorial/05-production.md) — deploy + observability

If you're new to vllm-lite, start with Tutorial 1 and work through them in
order. The tutorials assume no prior knowledge of the codebase.

## Getting Help

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: See README.md and docs/ directory

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
