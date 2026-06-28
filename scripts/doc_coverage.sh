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
#   - RealTot:  (--real only) pub items excluding test/hidden/derive-generated
#   - RealDoc:  (--real only) documented subset of RealTot
#   - Real%:    (--real only) percentage of RealTot that is documented
#
# Usage:
#   scripts/doc_coverage.sh            # text table (raw coverage)
#   scripts/doc_coverage.sh --real     # text table with real-coverage columns
#   scripts/doc_coverage.sh json       # JSON for tooling
#   scripts/doc_coverage.sh json --real # JSON with real-coverage fields

set -euo pipefail

# grep returns 1 when there are no matches; we don't want that to abort the script.
# Wrap grep in a function that always exits 0 so `pipefail` doesn't kill us.
grep_always_ok() { grep "$@" || true; }

OUTPUT_MODE="text"
REAL_MODE=0
for arg in "$@"; do
    case "$arg" in
        --real) REAL_MODE=1 ;;
        json)   OUTPUT_MODE="json" ;;
        text)   OUTPUT_MODE="text" ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

# Path to the Python helper that blanks out test mods / #[doc(hidden)] /
# #[derive(...)]-decorated items. Writes the blanked file to stdout.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLANK_PY="${SCRIPT_DIR}/_blank_for_real.py"

# Find pub items per file.
# Optional second arg: "real" — when set, blank out test mods / #[doc(hidden)] /
# #[derive(...)]-decorated items before counting.
find_pub_items() {
    local crate="$1"
    local mode="${2:-raw}"
    local src_dir="crates/${crate}/src"
    [ -d "$src_dir" ] || return 0

    if [ "$mode" = "raw" ]; then
        grep_always_ok -rEn "^[[:space:]]*pub[[:space:]]+(struct|enum|fn|trait|type|const|static|mod|async[[:space:]]+fn)" "$src_dir" --include="*.rs" 2>/dev/null
        return 0
    fi

    # Real mode: blank each file, then grep.
    while IFS= read -r f; do
        python3 "$BLANK_PY" < "$f" | awk -v fname="$f" '{print fname ":" NR ":" $0}' \
            | grep_always_ok -E ":[[:space:]]*pub[[:space:]]+(struct|enum|fn|trait|type|const|static|mod|async[[:space:]]+fn)"
    done < <(find "$src_dir" -name "*.rs" 2>/dev/null)
}

# Find documented pub items (preceded by ///).
find_documented_items() {
    local crate="$1"
    local mode="${2:-raw}"
    local src_dir="crates/${crate}/src"
    [ -d "$src_dir" ] || return 0

    if [ "$mode" = "raw" ]; then
        for f in $(find "$src_dir" -name "*.rs" 2>/dev/null); do
            awk '
                { lines[NR] = $0 }
                END {
                    for (i = 1; i <= NR; i++) {
                        if (lines[i] ~ /^[[:space:]]*pub[[:space:]]+(struct|enum|fn|trait|type|const|static|mod|async[[:space:]]+fn)/) {
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
        return 0
    fi

    # Real mode: blank out, then check for /// preceding pub items.
    while IFS= read -r f; do
        python3 "$BLANK_PY" < "$f" | awk -v fname="$f" '
            { lines[NR] = $0 }
            END {
                for (i = 1; i <= NR; i++) {
                    if (lines[i] ~ /^[[:space:]]*pub[[:space:]]+(struct|enum|fn|trait|type|const|static|mod|async[[:space:]]+fn)/) {
                        j = i - 1
                        while (j > 0 && (lines[j] ~ /^[[:space:]]*$/ || lines[j] ~ /^[[:space:]]*#/ || lines[j] ~ /^[[:space:]]*\/\//)) {
                            if (lines[j] ~ /^[[:space:]]*\/\/\/[^/]/) {
                                print fname ":" i
                                break
                            }
                            j--
                        }
                    }
                }
            }
        '
    done < <(find "$src_dir" -name "*.rs" 2>/dev/null)
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
        pt=$(find_pub_items "$crate" raw | wc -l)
        pd=$(find_documented_items "$crate" raw | wc -l)
        pct=0
        [ "$pt" -gt 0 ] && pct=$(awk "BEGIN {printf \"%.1f\", ($pd/$pt)*100}")
        rt=0; rd=0; rpct=0
        if [ "$REAL_MODE" -eq 1 ]; then
            rt=$(find_pub_items "$crate" real | wc -l)
            rd=$(find_documented_items "$crate" real | wc -l)
            [ "$rt" -gt 0 ] && rpct=$(awk "BEGIN {printf \"%.1f\", ($rd/$rt)*100}")
        fi
        md=$(module_docs_count "$crate")
        mt=$(echo "$md" | awk '{print $1}')
        mdoc=$(echo "$md" | awk '{print $2}')
        mpct=0
        [ "$mt" -gt 0 ] && mpct=$(awk "BEGIN {printf \"%.1f\", ($mdoc/$mt)*100}")
        [ "$first" -eq 0 ] && echo ","
        first=0
        if [ "$REAL_MODE" -eq 1 ]; then
            printf '  "%s": {"pub_total": %d, "pub_documented": %d, "pub_pct": %s, "real_total": %d, "real_documented": %d, "real_pct": %s, "module_total": %d, "module_documented": %d, "module_pct": %s}' \
                "$crate" "$pt" "$pd" "$pct" "$rt" "$rd" "$rpct" "$mt" "$mdoc" "$mpct"
        else
            printf '  "%s": {"pub_total": %d, "pub_documented": %d, "pub_pct": %s, "module_total": %d, "module_documented": %d, "module_pct": %s}' \
                "$crate" "$pt" "$pd" "$pct" "$mt" "$mdoc" "$mpct"
        fi
    done
    echo ""
    echo "}"
else
    if [ "$REAL_MODE" -eq 1 ]; then
        printf "%-10s %8s %8s %7s %8s %8s %7s %8s %8s %7s\n" "Crate" "PubTot" "PubDoc" "Pub%" "RealTot" "RealDoc" "Real%" "ModTot" "ModDoc" "Mod%"
        printf "%-10s %8s %8s %7s %8s %8s %7s %8s %8s %7s\n" "----------" "--------" "-------" "-------" "--------" "-------" "-------" "--------" "-------" "-------"
    else
        printf "%-10s %8s %8s %7s %8s %8s %7s\n" "Crate" "PubTot" "PubDoc" "Pub%" "ModTot" "ModDoc" "Mod%"
        printf "%-10s %8s %8s %7s %8s %8s %7s\n" "----------" "--------" "-------" "-------" "--------" "-------" "-------"
    fi
    total_pub=0; total_doc=0
    total_rt=0; total_rd=0
    total_mt=0; total_md=0
    for crate in traits core model server dist testing; do
        pt=$(find_pub_items "$crate" raw | wc -l)
        pd=$(find_documented_items "$crate" raw | wc -l)
        pct="0.0"
        [ "$pt" -gt 0 ] && pct=$(awk "BEGIN {printf \"%.1f\", ($pd/$pt)*100}")
        rt=0; rd=0; rpct="0.0"
        if [ "$REAL_MODE" -eq 1 ]; then
            rt=$(find_pub_items "$crate" real | wc -l)
            rd=$(find_documented_items "$crate" real | wc -l)
            [ "$rt" -gt 0 ] && rpct=$(awk "BEGIN {printf \"%.1f\", ($rd/$rt)*100}")
        fi
        md=$(module_docs_count "$crate")
        mt=$(echo "$md" | awk '{print $1}')
        mdoc=$(echo "$md" | awk '{print $2}')
        mpct="0.0"
        [ "$mt" -gt 0 ] && mpct=$(awk "BEGIN {printf \"%.1f\", ($mdoc/$mt)*100}")
        if [ "$REAL_MODE" -eq 1 ]; then
            printf "%-10s %8d %8d %6s%% %8d %8d %6s%% %8d %8d %6s%%\n" "$crate" "$pt" "$pd" "$pct" "$rt" "$rd" "$rpct" "$mt" "$mdoc" "$mpct"
        else
            printf "%-10s %8d %8d %6s%% %8d %8d %6s%%\n" "$crate" "$pt" "$pd" "$pct" "$mt" "$mdoc" "$mpct"
        fi
        total_pub=$((total_pub+pt))
        total_doc=$((total_doc+pd))
        total_rt=$((total_rt+rt))
        total_rd=$((total_rd+rd))
        total_mt=$((total_mt+mt))
        total_md=$((total_md+mdoc))
    done
    ws_pct="0.0"
    [ "$total_pub" -gt 0 ] && ws_pct=$(awk "BEGIN {printf \"%.1f\", ($total_doc/$total_pub)*100}")
    ws_mpct="0.0"
    [ "$total_mt" -gt 0 ] && ws_mpct=$(awk "BEGIN {printf \"%.1f\", ($total_md/$total_mt)*100}")
    if [ "$REAL_MODE" -eq 1 ]; then
        ws_rpct="0.0"
        [ "$total_rt" -gt 0 ] && ws_rpct=$(awk "BEGIN {printf \"%.1f\", ($total_rd/$total_rt)*100}")
        printf "%-10s %8s %8s %7s %8s %8s %7s %8s %8s %7s\n" "----------" "--------" "-------" "-------" "--------" "-------" "-------" "--------" "-------" "-------"
        printf "%-10s %8d %8d %6s%% %8d %8d %6s%% %8d %8d %6s%%\n" "WORKSPACE" "$total_pub" "$total_doc" "$ws_pct" "$total_rt" "$total_rd" "$ws_rpct" "$total_mt" "$total_md" "$ws_mpct"
    else
        printf "%-10s %8s %8s %7s %8s %8s %7s\n" "----------" "--------" "-------" "-------" "--------" "-------" "-------"
        printf "%-10s %8d %8d %6s%% %8d %8d %6s%%\n" "WORKSPACE" "$total_pub" "$total_doc" "$ws_pct" "$total_mt" "$total_md" "$ws_mpct"
    fi
fi
