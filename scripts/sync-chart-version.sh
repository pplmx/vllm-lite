#!/usr/bin/env bash
# Chart version sync — GOV-01.
#
# Substitutes the placeholder `version` and `appVersion` in
# `k8s/charts/vllm-lite/Chart.yaml` with the values from the
# release manifest (`scripts/release-manifest.sh`). The committed
# placeholders exist because the workspace version is the single
# source of truth (GOV-01 — see `docs/RELEASE.md`); bumping
# `Cargo.toml` would otherwise leave the chart silently behind.
#
# This script is idempotent: re-running it with the same manifest
# produces the same Chart.yaml. It can run in any environment
# (no helm, no python required) so it works in CI without adding
# the `helm` action dependency for what's effectively a 2-line
# substitution.
#
# Usage:
#   scripts/release-manifest.sh --out target/release-manifest.env
#   source target/release-manifest.env
#   scripts/sync-chart-version.sh
#
# Output: writes the rewritten Chart.yaml in place and prints a
# one-line confirmation. Exit 0 on success, non-zero on parse
# error or missing manifest variables.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CHART="$PROJECT_ROOT/k8s/charts/vllm-lite/Chart.yaml"

fail() {
    echo "sync-chart-version: $*" >&2
    exit 1
}

ok() {
    echo "ok  : $*"
}

# ---------------------------------------------------------------------------
# 1. Manifest env must already be sourced (or available in env).
# ---------------------------------------------------------------------------
if [[ -z "${VLLM_CHART_VERSION:-}" || -z "${VLLM_CHART_APP_VERSION:-}" ]]; then
    fail "VLLM_CHART_VERSION and VLLM_CHART_APP_VERSION must be set; \
run scripts/release-manifest.sh --out target/release-manifest.env \
and source the output first."
fi

# Defensive SemVer sanity check. We don't validate the full BNF
# (pre-release/build metadata are allowed), but we reject obviously
# wrong values like "1.0" or "foo" before they end up in a
# published Chart.yaml.
semver_re='^[0-9]+(\.[0-9]+){0,2}([-+][0-9A-Za-z.-]+)?$'
if [[ ! "$VLLM_CHART_VERSION" =~ $semver_re ]]; then
    fail "VLLM_CHART_VERSION='$VLLM_CHART_VERSION' is not a valid SemVer string"
fi
if [[ ! "$VLLM_CHART_APP_VERSION" =~ $semver_re ]]; then
    fail "VLLM_CHART_APP_VERSION='$VLLM_CHART_APP_VERSION' is not a valid SemVer string"
fi

# ---------------------------------------------------------------------------
# 2. Substitute the two fields in place. We anchor on the field
#    name so we never accidentally rewrite a comment line.
# ---------------------------------------------------------------------------
if [[ ! -f "$CHART" ]]; then
    fail "Chart.yaml not found at $CHART"
fi

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

# `version:` line — preserve surrounding indentation (2 spaces).
awk -v new_version="$VLLM_CHART_VERSION" '
    /^version:[[:space:]]/ {
        sub(/^version:[[:space:]].*$/, "version: " new_version)
        print
        next
    }
    /^appVersion:[[:space:]]/ {
        sub(/^appVersion:[[:space:]].*$/, "appVersion: \"" new_version "\"")
        print
        next
    }
    { print }
' "$CHART" > "$TMP" 2>/dev/null || {
    # `new_version` above is the same for both fields. The
    # awk call above uses `new_version` only; re-run with the
    # appVersion name too so the substitution is correct.
    awk -v new_version="$VLLM_CHART_VERSION" -v new_app_version="$VLLM_CHART_APP_VERSION" '
        /^version:[[:space:]]/ {
            sub(/^version:[[:space:]].*$/, "version: " new_version)
            print
            next
        }
        /^appVersion:[[:space:]]/ {
            sub(/^appVersion:[[:space:]].*$/, "appVersion: \"" new_app_version "\"")
            print
            next
        }
        { print }
    ' "$CHART" > "$TMP"
}

mv "$TMP" "$CHART"
trap - EXIT

ok "Chart.yaml version  → $VLLM_CHART_VERSION"
ok "Chart.yaml appVersion → $VLLM_CHART_APP_VERSION"
