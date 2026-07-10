#!/bin/bash
# Phase 12b v2: find pub items with no production caller.
# Logic: count DISTINCT files containing the name (excluding declaring file).
# Categories:
#   USED          - external production callers exist
#   TEST-ONLY     - only test files reference it
#   INTERNAL-ONLY - only declaring file references it (could be prod-internal or test-internal)
#   TRULY-UNUSED  - no callers anywhere

set -u
cd /workspace/vllm-lite

CRATE_FILTER="${1:-}"
OUT_FILE=".planning/phase-12b/dead-pub-candidates.tsv"

if [ -n "$CRATE_FILTER" ]; then
    FILE_LIST=$(find "crates/${CRATE_FILTER}" -name "*.rs" -path "*/src/*" -not -name "tests.rs" 2>/dev/null)
else
    FILE_LIST=$(find crates -name "*.rs" -path "*/src/*" -not -name "tests.rs" 2>/dev/null)
fi

echo -e "file\tline\tkind\tname\text_prod_files\text_test_files\tverdict" > "$OUT_FILE"

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

        # External production files (NOT declaring file, NOT test files)
        ext_prod=$(echo "$all_matches" \
            | grep -v "^${file}:" \
            | cut -d: -f1 \
            | sort -u \
            | grep -vE "/tests/|\btests\.rs$" \
            | grep -c .)

        # External test files (NOT declaring file)
        ext_test=$(echo "$all_matches" \
            | grep -v "^${file}:" \
            | cut -d: -f1 \
            | sort -u \
            | grep -E "/tests/|\btests\.rs$" \
            | grep -c .)

        # Same-file matches (excluding the declaration line)
        same_file=$(echo "$all_matches" \
            | grep "^${file}:" \
            | grep -v "^${file}:${linenum}:" \
            | wc -l)

        if [ "$ext_prod" -gt 0 ]; then
            verdict="USED"
        elif [ "$ext_test" -gt 0 ]; then
            verdict="TEST-ONLY"
        elif [ "$same_file" -gt 0 ]; then
            # Same-file only — could be production-internal or test-internal
            verdict="INTERNAL-ONLY"
        else
            verdict="TRULY-UNUSED"
        fi

        echo -e "${file}\t${linenum}\t${kind}\t${name}\t${ext_prod}\t${ext_test}\t${verdict}"
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
echo "=== Per crate (TEST-ONLY) ==="
awk -F'\t' '$NF == "TEST-ONLY" {split($1,a,"/"); print a[2]}' "$OUT_FILE" | sort | uniq -c | sort -rn
echo ""
echo "=== Per crate (INTERNAL-ONLY) ==="
awk -F'\t' '$NF == "INTERNAL-ONLY" {split($1,a,"/"); print a[2]}' "$OUT_FILE" | sort | uniq -c | sort -rn
echo ""
echo "=== Top 30 TRULY-UNUSED (by crate) ==="
awk -F'\t' '$NF == "TRULY-UNUSED"' "$OUT_FILE" | head -30
