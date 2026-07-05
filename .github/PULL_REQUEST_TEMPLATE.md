<!--
Thanks for contributing to vLLM-lite! Please fill in this template to help
reviewers understand your changes.
-->

## Summary

<!-- One-paragraph description of the change and motivation. -->

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update
- [ ] Refactor (no functional change)
- [ ] Performance improvement
- [ ] Test addition/improvement

## Related Issues

<!-- Use `Fixes #123` to auto-close, or `Refs #123` to link. -->

## Test Plan

<!-- How did you verify this works? Add commands, output, screenshots. -->

- [ ] `just ci` passes locally
- [ ] New tests added (if behavior changed)
- [ ] Existing tests still pass
- [ ] Manual smoke run: `cargo run -p vllm-server` then `curl localhost:8080/v1/models`

## Checklist

- [ ] Code follows the project's coding style (see CONTRIBUTING.md)
- [ ] Self-reviewed my own diff
- [ ] Public APIs documented with `///` doc comments
- [ ] No new `unwrap()` / `expect()` in non-test code paths
- [ ] No new `Box<dyn Error>` in public APIs
- [ ] Commit messages follow `<type>(<scope>): <subject>` format
- [ ] CI is green on this PR
