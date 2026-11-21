#!/usr/bin/env bash
# scripts/check_mutation_score.sh — fail if mutation score regresses vs baseline.
#
# Usage: scripts/check_mutation_score.sh <scan-dir> <baseline-pct>
#
# scan-dir: directory produced by `cargo mutants --output` (contains
#           caught.txt / missed.txt / timeout.txt / unviable.txt).
# baseline-pct: minimum acceptable mutation score as integer or float
#               (e.g. 70 or 99.5).
#
# Mutation score = caught / (caught + missed) * 100.
#
# What counts as what (cargo-mutants' own outcome files):
#   caught.txt   — a test failed on the mutant. Counts toward the score.
#   missed.txt   — the suite still passed. Counts against it.
#   timeout.txt  — the mutant hung the suite. Counts against it, same as a
#                  miss: nothing detected the change, the test just never
#                  returned. (These used to be silently excluded.)
#   unviable.txt — the mutant did not compile. NOT counted: it says nothing
#                  about test coverage.
#
# NOTE: this script used to read an `outcomes.json` that cargo-mutants does
# not produce (27.x writes the four .txt files above plus `mutants.json`,
# which carries no outcome field at all). It therefore exited 2 on every
# real scan, and the nightly job's render step selected `.status ==
# "SURVIVED"` from `mutants.json` — a field that does not exist — so the
# survived-mutant summary was always empty.

set -euo pipefail

SCAN_DIR="${1:-.mutants-out/mutants.out}"
BASELINE="${2:?usage: $0 <scan-dir> <baseline-pct>}"

for outcome in caught missed timeout; do
    if [ ! -f "$SCAN_DIR/$outcome.txt" ]; then
        echo "no $outcome.txt in $SCAN_DIR — run a mutation scan first" >&2
        exit 2
    fi
done

count_lines() {
    # cargo-mutants writes one mutant per line; an empty file means zero.
    local file="$1"
    if [ ! -s "$file" ]; then
        echo 0
    else
        grep -cve '^[[:space:]]*$' "$file"
    fi
}

CAUGHT=$(count_lines "$SCAN_DIR/caught.txt")
MISSED=$(count_lines "$SCAN_DIR/missed.txt")
TIMEOUT=$(count_lines "$SCAN_DIR/timeout.txt")
NOT_DETECTED=$((MISSED + TIMEOUT))

TOTAL=$((CAUGHT + NOT_DETECTED))
if [ "$TOTAL" -eq 0 ]; then
    echo "no caught/missed/timeout mutations in scan — check scope" >&2
    exit 2
fi

SCORE=$(awk "BEGIN { printf \"%.2f\", $CAUGHT * 100 / $TOTAL }")

echo "mutation score: ${SCORE}% (caught=$CAUGHT, missed=$MISSED, timeout=$TIMEOUT)"
echo "minimum required: ${BASELINE}%"

if awk "BEGIN { exit !($SCORE + 0 < $BASELINE + 0) }"; then
    echo "FAIL: mutation score ${SCORE}% below baseline ${BASELINE}%"
    echo "undetected mutants:"
    cat "$SCAN_DIR/missed.txt" "$SCAN_DIR/timeout.txt"
    exit 1
fi

echo "PASS"
