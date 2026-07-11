#!/bin/bash
# Regenerate Phase 12e public-API baselines from the current workspace.
# Run after intentional public-API changes documented in CHANGELOG.md.
set -euo pipefail
cd "$(dirname "$0")/../.."

if ! command -v cargo-public-api >/dev/null 2>&1; then
    echo "Install cargo-public-api first: cargo install cargo-public-api --locked" >&2
    exit 1
fi

BASELINE_DIR=".planning/phase-12e"
crates=(traits core model server dist testing)

for crate in "${crates[@]}"; do
    echo "Refreshing vllm-${crate}..."
    cargo public-api -p "vllm-${crate}" --simplified | sort -u > "${BASELINE_DIR}/${crate}.txt"
done

echo "OK: baselines written to ${BASELINE_DIR}/"
