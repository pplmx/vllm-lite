#!/bin/bash
# Release manifest: single source of truth for tag ↔ Cargo ↔ image ↔ Chart
#
# GOV-01 (technical due diligence): previously the workspace version
# (Cargo.toml), the Helm Chart `version` + `appVersion`, the Docker
# image tag and the GitHub Release tag could drift independently.
# This script collapses all four into one derivation from
# `[workspace.package] version` in Cargo.toml, with optional
# `--validate` to fail in CI when a pushed tag doesn't match.
#
# Output: shell-sourceable KEY=value pairs on stdout. When
# `--out FILE` is passed, the same content is written to FILE
# (and a `set -a; source FILE; set +a` works for downstream steps).
#
# Contract (each variable is documented in `--help`):
#
#   VLLM_VERSION             workspace version, e.g. "0.1.0"
#   VLLM_VERSION_CLEAN       same, without optional SemVer suffix
#   VLLM_IS_PRERELEASE       "true" if version contains "-"
#   VLLM_IMAGE_TAG           Docker image tag, e.g. "0.1.0"
#   VLLM_IMAGE_TAG_FULL      full tag including the registry
#                            namespace, e.g. "v0.1.0" (when a
#                            registry prefix is set via --registry)
#   VLLM_CHART_VERSION       Helm Chart.yaml `version` field
#   VLLM_CHART_APP_VERSION   Helm Chart.yaml `appVersion` field
#   VLLM_GIT_SHA             full git SHA (empty if not a git checkout)
#   VLLM_GIT_SHA_SHORT       first 8 chars of VLLM_GIT_SHA
#   VLLM_GIT_DESCRIBE        `git describe --always --tags --dirty`
#   VLLM_RUSTC_VERSION       rustc -V output (trimmed)
#   VLLM_BUILD_TIMESTAMP     UTC ISO 8601 timestamp
#
# Usage:
#   scripts/release-manifest.sh                # print to stdout
#   scripts/release-manifest.sh --out FILE     # also write FILE
#   scripts/release-manifest.sh --validate TAG # exit 1 if TAG (without
#                                              # `v` prefix) !=
#                                              # workspace version
#
# Exit codes:
#   0  ok
#   1  workspace version not found in Cargo.toml
#   2  --validate tag does not match workspace version
#   3  unexpected CLI argument

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CARGO_TOML="$PROJECT_ROOT/Cargo.toml"
WORKSPACE_KEY='\[workspace\.package\]'

OUT_FILE=""
VALIDATE_TAG=""
REGISTRY=""

usage() {
    sed -n '2,/^$/p' "$0" | sed -e 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)      OUT_FILE="$2"; shift 2 ;;
        --validate) VALIDATE_TAG="$2"; shift 2 ;;
        --registry) REGISTRY="$2"; shift 2 ;;
        -h|--help)  usage 0 ;;
        *)          echo "release-manifest: unknown argument: $1" >&2; usage 3 ;;
    esac
done

# 1. Read the workspace version. Grep for the version line under
#    `[workspace.package]`; strip comments and trim whitespace.
if [[ ! -f "$CARGO_TOML" ]]; then
    echo "release-manifest: Cargo.toml not found at $CARGO_TOML" >&2
    exit 1
fi

VLLM_VERSION="$(
    awk -v key="$WORKSPACE_KEY" '
        $0 ~ key { in_ws = 1; next }
        in_ws && /^[[:space:]]*#/ { next }
        in_ws && /^[[:space:]]*version[[:space:]]*=/ {
            sub(/^[[:space:]]*version[[:space:]]*=[[:space:]]*"/, "")
            sub(/".*$/, "")
            print
            exit
        }
        in_ws && /^\[/ { in_ws = 0 }
    ' "$CARGO_TOML"
)"
if [[ -z "$VLLM_VERSION" ]]; then
    echo "release-manifest: could not find workspace version in $CARGO_TOML" >&2
    exit 1
fi

# 2. Pre-release flag (SemVer: anything after `-`).
if [[ "$VLLM_VERSION" == *-* ]]; then
    VLLM_IS_PRERELEASE="true"
else
    VLLM_IS_PRERELEASE="false"
fi

# 3. Image tag. The image registry namespace (e.g. `ghcr.io/pplmx`)
#    is supplied via --registry; default is no namespace, so the tag
#    is just the bare version. `docker tag` + `docker push` accept
#    both forms.
VLLM_IMAGE_TAG="$VLLM_VERSION"
if [[ -n "$REGISTRY" ]]; then
    VLLM_IMAGE_TAG_FULL="${REGISTRY%/}/vllm-lite:${VLLM_IMAGE_TAG}"
else
    VLLM_IMAGE_TAG_FULL=""
fi

# 4. Chart version + appVersion: same string as the workspace
#    version. The two are kept as separate variables so a future
#    change can decouple them without rewriting the contract.
VLLM_CHART_VERSION="$VLLM_VERSION"
VLLM_CHART_APP_VERSION="$VLLM_VERSION"

# 5. Git metadata. Empty when not in a git checkout (e.g. release
#    tarball build).
if command -v git >/dev/null 2>&1 && git -C "$PROJECT_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    VLLM_GIT_SHA="$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo "")"
    VLLM_GIT_SHA_SHORT="${VLLM_GIT_SHA:0:8}"
    VLLM_GIT_DESCRIBE="$(git -C "$PROJECT_ROOT" describe --always --tags --dirty 2>/dev/null || echo "")"
else
    VLLM_GIT_SHA=""
    VLLM_GIT_SHA_SHORT=""
    VLLM_GIT_DESCRIBE=""
fi

# 6. Toolchain metadata. Empty when rustc is not on PATH (rare; even
#    the Docker builder installs it).
if command -v rustc >/dev/null 2>&1; then
    VLLM_RUSTC_VERSION="$(rustc -V 2>/dev/null | tr -d '\n' || echo "")"
else
    VLLM_RUSTC_VERSION=""
fi

# 7. Build timestamp. `date -u +%FT%TZ` is GNU + BSD compatible; on
#    macOS it works as-is.
VLLM_BUILD_TIMESTAMP="$(date -u +%FT%TZ 2>/dev/null || echo "")"

# 8. Optional validation: the tag pushed to GitHub must equal the
#    workspace version. Invoked by release.yml before any build step.
if [[ -n "$VALIDATE_TAG" ]]; then
    if [[ "$VALIDATE_TAG" != "$VLLM_VERSION" ]]; then
        echo "release-manifest: tag '${VALIDATE_TAG}' does not match workspace version '${VLLM_VERSION}'" >&2
        echo "  bump [workspace.package] version in Cargo.toml, or push the tag that matches." >&2
        exit 2
    fi
fi

# 9. Emit. Comment lines start with `#` so the file is self-describing
#    when sourced; downstream steps just `source target/release-manifest.env`.
emit() {
    cat <<EOF
# Release manifest — generated by scripts/release-manifest.sh.
# Do NOT edit by hand; re-run the script to refresh.
VLLM_VERSION=${VLLM_VERSION}
VLLM_IS_PRERELEASE=${VLLM_IS_PRERELEASE}
VLLM_IMAGE_TAG=${VLLM_IMAGE_TAG}
VLLM_IMAGE_TAG_FULL=${VLLM_IMAGE_TAG_FULL}
VLLM_CHART_VERSION=${VLLM_CHART_VERSION}
VLLM_CHART_APP_VERSION=${VLLM_CHART_APP_VERSION}
VLLM_GIT_SHA=${VLLM_GIT_SHA}
VLLM_GIT_SHA_SHORT=${VLLM_GIT_SHA_SHORT}
VLLM_GIT_DESCRIBE=${VLLM_GIT_DESCRIBE}
VLLM_RUSTC_VERSION=${VLLM_RUSTC_VERSION}
VLLM_BUILD_TIMESTAMP=${VLLM_BUILD_TIMESTAMP}
EOF
}

if [[ -n "$OUT_FILE" ]]; then
    mkdir -p "$(dirname "$OUT_FILE")"
    emit > "$OUT_FILE"
fi
emit
