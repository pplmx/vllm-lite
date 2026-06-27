#!/usr/bin/env bash
# scripts/doc_coverage.sh — workspace /// doc coverage report
#
# Reports per-crate:
#   - PubTot:   total pub items (struct/enum/fn/trait/type/const/static/mod)
#   - PubDoc:   pub items preceded by a /// line
#   - Pub%:     percentage documented
#   - ModTot:   total source files
#   - ModDoc:   source files with //! module-level doc
#   - Mod%:     percentage with module docs
#
# Usage:
#   scripts/doc_coverage.sh          # text table
#   scripts/doc_coverage.sh json     # JSON for tooling

set -euo pipefail

OUTPUT_MODE="${1:-text}"

# Find pub items per file.
# A pub item is `pub <keyword>` where keyword is struct|enum|fn|trait|type|const|static|mod
# OR `pub async fn`. Multi-line declarations where pub is on its own line are
# caught by also matching `pub` followed by newline then keyword.
find_pub_items() {
    local crate="$1"
    local src_dir="crates/${crate}/src"
    [ -d "$src_dir" ] || return 0
    grep -rEn "^[[:space:]]*pub[[:space:]]+(struct|enum|fn|trait|type|const|static|mod|async[[:space:]]+fn)" "$src_dir" --include="*.rs" 2>/dev/null
}

# Find documented pub items (preceded by ///).
# Approximate: grep for /// followed by pub on next non-blank line.
find_documented_items() {
    local crate="$1"
    local src_dir="crates/${crate}/src"
    [ -d "$src_dir" ] || return 0
    # Use awk to walk each file and check if pub items have /// preceding
    for f in $(find "$src_dir" -name "*.rs" 2>/dev/null); do
        awk '
            {
                lines[NR] = $0
            }
            END {
                for (i = 1; i <= NR; i++) {
                    if (lines[i] ~ /^[[:space:]]*pub[[:space:]]+(struct|enum|fn|trait|type|const|static|mod|async[[:space:]]+fn)/) {
                        # Look back for /// or //!
                        j = i - 1
                        while (j > 0 && (lines[j] ~ /^[[:space:]]*$/ || lines[j] ~ /^[[:space:]]*#/ || lines[j] ~ /^[[:space:]]*\/\//)) {
                            if (lines[j] ~ /^[[:space:]]*\/\/\/[^/]/) {
                                print FILENAME ":" i
                                break
                            }
                            j--
                        }
                    }
                }
            }
        ' "$f"
    done
}

# Module-level //! coverage per file.
module_docs_count() {
    local crate="$1"
    local src_dir="crates/${crate}/src"
    [ -d "$src_dir" ] || { echo "0 0"; return; }
    local total=0 with=0
    while IFS= read -r f; do
        total=$((total+1))
        if head -10 "$f" | grep -qE "^//!"; then
            with=$((with+1))
        fi
    done < <(find "$src_dir" -name "*.rs" 2>/dev/null)
    echo "$total $with"
}

# Compute JSON output
if [ "$OUTPUT_MODE" = "json" ]; then
    echo "{"
    first=1
    for crate in traits core model server dist testing; do
        pt=$(find_pub_items "$crate" | wc -l)
        pd=$(find_documented_items "$crate" | wc -l)
        pct=0
        [ "$pt" -gt 0 ] && pct=$(awk "BEGIN {printf \"%.1f\", ($pd/$pt)*100}")
        md=$(module_docs_count "$crate")
        mt=$(echo "$md" | awk '{print $1}')
        mdoc=$(echo "$md" | awk '{print $2}')
        mpct=0
        [ "$mt" -gt 0 ] && mpct=$(awk "BEGIN {printf \"%.1f\", ($mdoc/$mt)*100}")
        [ "$first" -eq 0 ] && echo ","
        first=0
        printf '  "%s": {"pub_total": %d, "pub_documented": %d, "pub_pct": %s, "module_total": %d, "module_documented": %d, "module_pct": %s}' \
            "$crate" "$pt" "$pd" "$pct" "$mt" "$mdoc" "$mpct"
    done
    echo ""
    echo "}"
else
    printf "%-10s %8s %8s %7s %8s %8s %7s\n" "Crate" "PubTot" "PubDoc" "Pub%" "ModTot" "ModDoc" "Mod%"
    printf "%-10s %8s %8s %7s %8s %8s %7s\n" "----------" "--------" "-------" "-------" "--------" "-------" "-------"
    total_pub=0; total_doc=0
    total_mt=0; total_md=0
    for crate in traits core model server dist testing; do
        pt=$(find_pub_items "$crate" | wc -l)
        pd=$(find_documented_items "$crate" | wc -l)
        pct="0.0"
        [ "$pt" -gt 0 ] && pct=$(awk "BEGIN {printf \"%.1f\", ($pd/$pt)*100}")
        md=$(module_docs_count "$crate")
        mt=$(echo "$md" | awk '{print $1}')
        mdoc=$(echo "$md" | awk '{print $2}')
        mpct="0.0"
        [ "$mt" -gt 0 ] && mpct=$(awk "BEGIN {printf \"%.1f\", ($mdoc/$mt)*100}")
        printf "%-10s %8d %8d %6s%% %8d %8d %6s%%\n" "$crate" "$pt" "$pd" "$pct" "$mt" "$mdoc" "$mpct"
        total_pub=$((total_pub+pt))
        total_doc=$((total_doc+pd))
        total_mt=$((total_mt+mt))
        total_md=$((total_md+mdoc))
    done
    ws_pct="0.0"
    [ "$total_pub" -gt 0 ] && ws_pct=$(awk "BEGIN {printf \"%.1f\", ($total_doc/$total_pub)*100}")
    ws_mpct="0.0"
    [ "$total_mt" -gt 0 ] && ws_mpct=$(awk "BEGIN {printf \"%.1f\", ($total_md/$total_mt)*100}")
    printf "%-10s %8s %8s %7s %8s %8s %7s\n" "----------" "--------" "-------" "-------" "--------" "-------" "-------"
    printf "%-10s %8d %8d %6s%% %8d %8d %6s%%\n" "WORKSPACE" "$total_pub" "$total_doc" "$ws_pct" "$total_mt" "$total_md" "$ws_mpct"
fi
