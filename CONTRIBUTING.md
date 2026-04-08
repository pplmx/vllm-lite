# Contributing to vLLM-lite

Thank you for your interest in contributing!

## Development Workflow

1. **Design First**
   - Use `brainstorming` skill to design new features
   - Use `writing-plans` skill to create implementation plans
   - Get design approval before implementation

2. **Implement**
   - Use `subagent-driven-development` for complex tasks
   - Follow AGENTS.md coding standards
   - Run tests before committing

3. **Review & Commit**
   - Use `verification-loop` before committing
   - Ensure clippy and fmt pass
   - Write detailed commit messages

## Quick Commands

```bash
# Build
cargo build --workspace

# Test
cargo test --workspace

# Clippy (required before commit)
cargo clippy --workspace -- -D warnings

# Format
cargo fmt --all
```

## Commit Message Format

```
<type>(<scope>): <subject>
```

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `refactor` | Code restructuring |
| `test` | Adding/updating tests |
| `docs` | Documentation |
| `chore` | Maintenance |

**Example**:

```
feat(scheduler): add decode-priority batching

- Prioritize decode sequences over prefill
- Add max_num_batched_tokens limit
- Fix chunked prefill tracking
```

## Coding Standards

- Follow Rust standard formatting (`cargo fmt`)
- Run clippy lints (`cargo clippy -- -D warnings`)
- Write tests for new features
- Keep commits small and focused
- Write detailed commit messages
- Document public APIs with `///` doc comments

## Testing

```bash
# Run all tests
cargo test --workspace

# Run specific crate
cargo test -p vllm-core

# Run with output
cargo test --workspace -- --nocapture
```

## Getting Help

- Open an issue for bugs or feature requests
- Use discussions for questions
