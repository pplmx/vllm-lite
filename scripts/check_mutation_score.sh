#!/usr/bin/env bash
# scripts/check_mutation_score.sh — fail if mutation score regresses vs baseline.
#
# Usage: scripts/check_mutation_score.sh <scan-dir> <baseline-pct>
#
# scan-dir: directory containing outcomes.json (default: .mutants-out/mutants.out)
# baseline-pct: minimum acceptable mutation score as integer or float (e.g. 70 or 99.5)
#
# Reads <scan-dir>/outcomes.json, computes
# mutation_score = caught / (caught + missed) * 100,
# and exits 1 if the score is below the baseline.
#
# Used by `just mutants-ci` to gate PRs on mutation score regression.

set -euo pipefail

SCAN_DIR="${1:-.mutants-out/mutants.out}"
BASELINE="${2:?usage: $0 <scan-dir> <baseline-pct>}"

if [ ! -f "$SCAN_DIR/outcomes.json" ]; then
    if [ -f "$SCAN_DIR/mutants.json" ]; then
        echo "found mutants.json but no outcomes.json in $SCAN_DIR — was the scan complete?"
    else
        echo "no outcomes.json in $SCAN_DIR — run a mutation scan first"
    fi
    exit 2
fi

CAUGHT=$(jq '.caught | length' "$SCAN_DIR/outcomes.json")
MISSED=$(jq '.missed | length' "$SCAN_DIR/outcomes.json")

TOTAL=$((CAUGHT + MISSED))
if [ "$TOTAL" -eq 0 ]; then
    echo "no caught/missed mutations in scan — check scope"
    exit 2
fi

SCORE=$(awk "BEGIN { printf \"%.2f\", $CAUGHT * 100 / $TOTAL }")

echo "mutation score: ${SCORE}% (caught=$CAUGHT, missed=$MISSED, total=$TOTAL)"
echo "minimum required: ${BASELINE}%"

# Use awk for floating-point comparison
if awk "BEGIN { exit !($SCORE + 0 < $BASELINE + 0) }"; then
    echo "FAIL: mutation score ${SCORE}% below baseline ${BASELINE}%"
    exit 1
fi

echo "PASS"
