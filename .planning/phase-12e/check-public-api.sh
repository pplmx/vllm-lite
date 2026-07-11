#!/bin/bash
# Phase 12e: cargo-public-api baseline diff for CI.
#
# Compares the current `cargo public-api` output for each workspace crate
# against the committed baseline. If new public-API items appear without a
# corresponding CHANGELOG entry mentioning that crate, the check fails.
#
# This catches unintentional public-API growth going forward. Public-API
# removal is allowed (without a CHANGELOG line) because the baseline is
# the explicit "we shrunk the API" record — the diff just confirms what
# we removed matches what we documented.
#
# Usage:
#   bash .planning/phase-12e/check-public-api.sh              # strict (fail on any change)
#   bash .planning/phase-12e/check-public-api.sh --no-fail    # report only
#
# Exits 0 when public-API is unchanged or shrinks; exits 1 when new items
# appear AND no CHANGELOG line references that crate.

set -u
cd "$(dirname "$0")/../.."

BASELINE_DIR=".planning/phase-12e"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

mode="--strict"
for arg in "$@"; do
    case "$arg" in
        --no-fail) mode="--report-only" ;;
        *) ;;
    esac
done

crates=(traits core model server dist testing)
fail=0
report_lines=()

for crate in "${crates[@]}"; do
    baseline="${BASELINE_DIR}/${crate}.txt"
    if [ ! -f "$baseline" ]; then
        echo "WARN: baseline missing for ${crate} — skipping" >&2
        continue
    fi

    if ! cargo public-api -p "vllm-${crate}" --simplified > "$WORK_DIR/${crate}.txt" 2>/dev/null; then
        echo "FAIL: cargo public-api failed for ${crate}" >&2
        fail=1
        continue
    fi

    # Sort + dedupe — cargo-public-api can emit duplicate lines for
    # blanket impls (e.g. `Clone for Foo` once per `Foo` reference);
    # we only care about the unique set of public items.
    sort -u "$WORK_DIR/${crate}.txt" > "$WORK_DIR/${crate}.sorted.txt"
    sort -u "$baseline" > "$WORK_DIR/${crate}.baseline.sorted.txt"

    # Compute added/removed items
    added=$(comm -23 "$WORK_DIR/${crate}.sorted.txt" "$WORK_DIR/${crate}.baseline.sorted.txt")
    removed=$(comm -13 "$WORK_DIR/${crate}.sorted.txt" "$WORK_DIR/${crate}.baseline.sorted.txt")

    added_count=$(echo -n "$added" | grep -c . || true)
    removed_count=$(echo -n "$removed" | grep -c . || true)

    if [ "$added_count" -eq 0 ] && [ "$removed_count" -eq 0 ]; then
        echo "  ${crate}: unchanged"
        continue
    fi

    if [ "$added_count" -gt 0 ]; then
        echo ""
        echo "NEW public API in vllm-${crate} (+${added_count} items, -${removed_count} removed):"
        echo "$added" | sed 's/^/  + /'
    fi
    if [ "$removed_count" -gt 0 ]; then
        echo "REMOVED public API from vllm-${crate}:"
        echo "$removed" | sed 's/^/  - /'
    fi

    # Shrinking is allowed without CHANGELOG (the baseline IS the record).
    if [ "$added_count" -eq 0 ]; then
        echo "  ${crate}: shrank by ${removed_count} (allowed)"
        continue
    fi

    # Growth: require CHANGELOG entry referencing this crate.
    # `git log --oneline -1 -- CHANGELOG.md` is the most recent commit
    # touching CHANGELOG; its body should mention `crate` or "phase 12c/12d".
    if [ ! -f "CHANGELOG.md" ]; then
        echo "FAIL: public API grew (+${added_count}) in ${crate} but CHANGELOG.md missing" >&2
        fail=1
        continue
    fi

    changelog_msg="$(git log -1 --format='%s%n%b' -- CHANGELOG.md 2>/dev/null || true)"
    if echo "$changelog_msg" | grep -qiE "vllm-${crate}|phase 12|public.?api" ; then
        echo "  ${crate}: grew +${added_count} (CHANGELOG entry present)"
        continue
    fi

    echo "FAIL: public API grew (+${added_count}) in vllm-${crate} but no CHANGELOG entry references it" >&2
    echo "  Add a CHANGELOG.md bullet under the most recent Unreleased section:" >&2
    echo "    - public-api: vllm-${crate} added <items> (Phase XX)" >&2
    fail=1
done

echo ""
if [ "$fail" -eq 0 ]; then
    echo "OK: public-API baseline check passed"
    exit 0
fi

if [ "$mode" = "--report-only" ]; then
    echo "FAIL: public-API grew without CHANGELOG entries (--no-fail: reporting only)"
    exit 0
fi

echo "FAIL: public-API baseline check failed"
exit 1
