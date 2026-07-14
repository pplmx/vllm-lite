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

    # Growth: require a CHANGELOG entry that explicitly tags the change
    # as a public-API modification for this crate. We scan the entire
    # [Unreleased] section (not just the latest commit message) because
    # API growth typically spans many commits within a single milestone.
    # The grep requires an explicit `public-api:` line that mentions this
    # crate's name — incidental mentions (e.g. test counts that list the
    # crate) do NOT count.
    if [ ! -f "CHANGELOG.md" ]; then
        echo "FAIL: public API grew (+${added_count}) in ${crate} but CHANGELOG.md missing" >&2
        fail=1
        continue
    fi

    # Extract the [Unreleased] section, dropping everything before its
    # heading and after the next `## ` heading (next version). CHANGELOG
    # headings sometimes lead with an emoji (e.g. `## 🚀 [Unreleased]`),
    # so we match on the marker `[Unreleased]` and break on any `## `.
    changelog_msg="$(awk '
        /^## .*\[Unreleased\]/ { in_unreleased = 1; next }
        in_unreleased && /^## / { exit }
        in_unreleased            { print }
    ' CHANGELOG.md)"

    # Match a `public-api:` (or `public_api:` / `publicapi:`) marker
    # followed by the crate name. The marker must appear at the start of
    # a bullet (after `-` / `*` / whitespace, optionally with markdown
    # bold `**` wrapping) so prose mentions of "public-API" don't
    # satisfy it.
    if echo "$changelog_msg" | grep -qiE "^\s*[-*]\s*(\*\*)?\s*public[._-]?api[: ].*vllm-${crate}\b" ; then
        echo "  ${crate}: grew +${added_count} (CHANGELOG entry present in [Unreleased])"
        continue
    fi

    echo "FAIL: public API grew (+${added_count}) in vllm-${crate} but no CHANGELOG entry references it" >&2
    echo "  Add a CHANGELOG.md bullet under the [Unreleased] section like:" >&2
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
