# Contributing to vLLM-lite

Thank you for your interest in contributing!

## Development Process

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Follow** the development workflow:
   - Use `brainstorming` skill to design new features
   - Use `writing-plans` skill to create implementation plans
   - Use `subagent-driven-development` for implementation
4. **Commit** your changes (`git commit -m 'feat: add amazing feature'`)
5. **Push** to the branch (`git push origin feature/amazing-feature`)
6. **Create** a Pull Request

## Coding Standards

- Follow Rust standard formatting (`cargo fmt`)
- Run clippy lints (`cargo clippy -- -D warnings`)
- Write tests for new features
- Keep commits small and focused
- Write detailed commit messages

## Commit Message Format

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test changes
- `refactor`: Code refactoring
- `chore`: Build/CI changes

Example:
```
feat(scheduler): add decode-priority batching

- Prioritize decode sequences over prefill
- Add max_num_batched_tokens limit
- Fix chunked prefill tracking

Closes #123
```

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
- Disussions for questions