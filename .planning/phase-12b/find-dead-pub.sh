#!/bin/bash
# Phase 12b v2: find pub items with no production caller.
# Logic: count DISTINCT files containing the name (excluding declaring file).
#
# Categories:
#   USED                - external production callers exist
#   INTEGRATION-TEST    - only integration tests (tests/*.rs at crate root) reference it
#                         → MUST stay `pub` (integration tests are external to crate)
#   UNIT-TEST-ONLY      - only unit tests (src/**/tests.rs, #[cfg(test)] mods) reference it
#                         → can tighten to `pub(crate)`
#   TEST-ONLY (MIXED)   - both unit and integration tests reference it
#                         → MUST stay `pub` (integration tests require external visibility)
#   INTERNAL-ONLY       - only declaring file references it (could be prod-internal or test-internal)
#   TRULY-UNUSED        - no callers anywhere
#
# Critical: integration tests at crates/<crate>/tests/*.rs are EXTERNAL to the crate.
# The Phase 12c commit (fb2697c) bug: previous version of this script lumped unit and
# integration tests into one bucket, leading to broken builds when pub(crate) was applied
# to methods used by tests/*.rs.
#
# Distinction logic (test file classification):
#   * Unit test file: any `tests.rs` or `*_tests.rs` matched INSIDE `crates/<c>/src/`
#     These are #[cfg(test)] mod tests; pub(crate) is sufficient.
#   * Integration test file: any `tests/*.rs` at the crate root (sibling of `src/`)
#     These compile as external crates; only `pub` works.

set -u
cd /workspace/vllm-lite

CRATE_FILTER="${1:-}"
OUT_FILE=".planning/phase-12b/dead-pub-candidates.tsv"

if [ -n "$CRATE_FILTER" ]; then
    FILE_LIST=$(find "crates/${CRATE_FILTER}" -name "*.rs" -path "*/src/*" -not -name "tests.rs" 2>/dev/null)
else
    FILE_LIST=$(find crates -name "*.rs" -path "*/src/*" -not -name "tests.rs" 2>/dev/null)
fi

echo -e "file\tline\tkind\tname\text_prod_files\text_unit_test_files\text_integ_test_files\tverdict" > "$OUT_FILE"

# A file is an integration test if its path matches `crates/<crate>/tests/*.rs`
# A file is a unit test if its path is `crates/<crate>/src/.../tests.rs` or
# contains `#[cfg(test)]` markers (we approximate by checking if the path
# ends in tests.rs under src/, or if the file's basename contains 'test').
#
# Note: a file ending in tests.rs anywhere under src/ is treated as a unit-test
# file by convention in this workspace (see crates/server/src/security/tls/tests.rs,
# crates/core/src/engine/ctor/builder.rs adjacents, etc.).

while IFS= read -r file; do
    [ -z "$file" ] && continue
    # Find pub declarations (free-standing, not in impl blocks for now)
    grep -nE "^[[:space:]]+pub[[:space:]]+(fn|struct|enum|trait|type|const|static)[[:space:]]+[A-Za-z_][A-Za-z0-9_]*" "$file" \
        | grep -vE "pub[[:space:]]*\(" \
        | while IFS=: read -r linenum match; do
        kind=$(echo "$match" | sed -E 's/.*pub[[:space:]]+(fn|struct|enum|trait|type|const|static).*/\1/')
        name=$(echo "$match" | sed -E 's/.*pub[[:space:]]+(fn|struct|enum|trait|type|const|static)[[:space:]]+([A-Za-z_][A-Za-z0-9_]*).*/\2/')

        [ -z "$name" ] && continue
        # Skip names too short (likely false matches)
        [ "${#name}" -lt 3 ] && continue

        # ALL files containing name (with line numbers)
        all_matches=$(grep -rnE "\b${name}\b" --include="*.rs" crates/ 2>/dev/null)

        # Helper: classify a file as unit-test, integ-test, or production.
        # Usage: classify_file
        classify_file() {
            local f="$1"
            # Skip declaring file (handled separately)
            if [ "$f" = "$file" ]; then
                return 1  # declaring file; not an "external" reference
            fi
            # Integration test: path matches crates/<crate>/tests/*.rs
            # i.e. there is /tests/ segment AFTER the crate root
            if echo "$f" | grep -qE "^crates/[^/]+/tests/.*\.rs$"; then
                echo "integ"
                return 0
            fi
            # Unit test: file ends in tests.rs or _test.rs anywhere under src/
            if echo "$f" | grep -qE "/(tests|_tests|test)\.rs$"; then
                echo "unit"
                return 0
            fi
            # Production
            echo "prod"
            return 0
        }

        # Categorize external references.
        ext_prod=0
        ext_unit=0
        ext_integ=0
        same_file=0

        # Stream through distinct external files (exclude declaring file)
        for f in $(echo "$all_matches" | cut -d: -f1 | sort -u); do
            if [ "$f" = "$file" ]; then
                continue
            fi
            cls=$(classify_file "$f" || true)
            case "$cls" in
                prod)  ext_prod=$((ext_prod + 1)) ;;
                unit)  ext_unit=$((ext_unit + 1)) ;;
                integ) ext_integ=$((ext_integ + 1)) ;;
            esac
        done

        # Same-file matches (excluding the declaration line)
        same_file=$(echo "$all_matches" \
            | grep "^${file}:" \
            | grep -v "^${file}:${linenum}:" \
            | wc -l)

        if [ "$ext_prod" -gt 0 ]; then
            verdict="USED"
        elif [ "$ext_integ" -gt 0 ] && [ "$ext_unit" -gt 0 ]; then
            verdict="TEST-ONLY-MIXED"
        elif [ "$ext_integ" -gt 0 ]; then
            verdict="INTEGRATION-TEST"
        elif [ "$ext_unit" -gt 0 ]; then
            verdict="UNIT-TEST-ONLY"
        elif [ "$same_file" -gt 0 ]; then
            # Same-file only — could be production-internal or test-internal
            verdict="INTERNAL-ONLY"
        else
            verdict="TRULY-UNUSED"
        fi

        echo -e "${file}\t${linenum}\t${kind}\t${name}\t${ext_prod}\t${ext_unit}\t${ext_integ}\t${verdict}"
    done
done <<< "$FILE_LIST" >> "$OUT_FILE"

echo "Wrote: $OUT_FILE"
echo ""
echo "=== Verdict summary ==="
awk -F'\t' 'NR>1 {print $NF}' "$OUT_FILE" | sort | uniq -c | sort -rn
echo ""
echo "=== Per crate (TRULY-UNUSED) ==="
awk -F'\t' '$NF == "TRULY-UNUSED" {split($1,a,"/"); print a[2]}' "$OUT_FILE" | sort | uniq -c | sort -rn
echo ""
echo "=== Per crate (UNIT-TEST-ONLY) — safe to tighten to pub(crate) ==="
awk -F'\t' '$NF == "UNIT-TEST-ONLY" {split($1,a,"/"); print a[2]}' "$OUT_FILE" | sort | uniq -c | sort -rn
echo ""
echo "=== Per crate (INTEGRATION-TEST) — must keep pub ==="
awk -F'\t' '$NF == "INTEGRATION-TEST" {split($1,a,"/"); print a[2]}' "$OUT_FILE" | sort | uniq -c | sort -rn
echo ""
echo "=== Per crate (TEST-ONLY-MIXED) — must keep pub ==="
awk -F'\t' '$NF == "TEST-ONLY-MIXED" {split($1,a,"/"); print a[2]}' "$OUT_FILE" | sort | uniq -c | sort -rn
echo ""
echo "=== Per crate (INTERNAL-ONLY) ==="
awk -F'\t' '$NF == "INTERNAL-ONLY" {split($1,a,"/"); print a[2]}' "$OUT_FILE" | sort | uniq -c | sort -rn
echo ""
echo "=== Top 30 TRULY-UNUSED (by crate) ==="
awk -F'\t' '$NF == "TRULY-UNUSED"' "$OUT_FILE" | head -30
