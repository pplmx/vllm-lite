# Tutorial 1: Setup & Build

Welcome to vllm-lite! In this tutorial we'll go from a fresh clone to a
working build.

## Prerequisites

- **Rust 1.85+** (we use edition 2024). Install via [rustup](https://rustup.rs/):
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
- **Git** — for cloning and version control
- **~5 GB disk** — for Rust toolchain, dependencies, and build artifacts
- **(Optional) CUDA 11.8+** — only needed for GPU inference (Qwen3, Llama
  with CUDA kernels). CPU-only inference works without it.

## Clone & Build

```bash
git clone https://github.com/pplmx/vllm-lite.git
cd vllm-lite
cargo build --workspace
```

The first build downloads ~300 crates and takes 5-15 minutes. Subsequent
builds are incremental (seconds).

## Verify the Build

```bash
# Run all unit + integration tests (skips #[ignore] slow tests)
just nextest

# Or with cargo directly
cargo test --workspace --no-fail-fast
```

Expected: ~1200+ tests pass, 0 failures.

## Install Just (Task Runner)

We use [`just`](https://github.com/casey/just) (a Make alternative) for
common workflows:

```bash
cargo install just --locked
```

Verify: `just --version` → 1.x+

## Code Style

Before committing, run:

```bash
just fmt-check    # cargo fmt --all --check
just clippy       # cargo clippy with workspace lints
just doc-check    # cargo doc --no-deps (0 warnings)
```

All three must pass for CI to merge your PR.

## Next Steps

→ [Tutorial 2: Load a Model](02-load-model.md)
