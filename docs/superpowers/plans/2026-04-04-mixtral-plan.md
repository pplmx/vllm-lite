# Mixtral Implementation Plan

> 10 tasks

## Task 1: Add Mixtral to Architecture
- Add Architecture::Mixtral
- Add MixtralConfig to ModelConfig

## Task 2: Create mixtral module
- Create crates/model/src/mixtral/

## Task 3: Implement MixtralSparseMoe
- Top-2 expert routing
- 8 experts

## Task 4: Implement MixtralBlock
- Use SparseMoe instead of SwiGLU

## Task 5: Implement MixtralModel

## Task 6: Update loader/registry

## Task 7: Tests & verification
