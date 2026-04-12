# Contributing to vLLM-lite

Thank you for your interest in contributing to vLLM-lite! This guide will help you get started.

## Prerequisites

Before building from source, ensure you have:

- **Rust 1.75+** - Install via [rustup](https://rustup.rs/)
- **CUDA 12.1+** - Optional, for GPU support
- **CMake 3.18+** - For building some dependencies

```bash
# Verify installation
rustc --version   # Should be 1.75 or higher
cargo --version
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

   ```bash
   # Format check
   cargo fmt --all --check

   # Lint
   cargo clippy --workspace -- -D warnings

   # Run tests
   cargo test --workspace
   ```

5. **Submit a Pull Request**
    - Push to your fork
    - Open a PR against `main`
    - Fill out the PR template

## Coding Standards

- **Formatting**: Run `cargo fmt --all` before committing
- **Linting**: `cargo clippy --workspace -- -D warnings` must pass
- **Testing**: Add tests for new features; all tests must pass
- **Documentation**: Document public APIs with `///` doc comments

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
│   ├── traits/      # Interface definitions
│   ├── core/        # Engine, Scheduler, KV Cache
│   ├── model/       # Model implementations, kernels
│   ├── dist/        # Tensor Parallelism
│   └── server/      # HTTP API
├── tests/           # Integration tests
└── docs/            # Design documents
```

## Testing

```bash
# Run all tests
cargo test --workspace

# Run specific crate
cargo test -p vllm-core

# Run with output visible
cargo test --workspace -- --nocapture

# Run only fast tests (skip slow/ignored tests)
just nextest
```

## Getting Help

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: See README.md and docs/ directory

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
