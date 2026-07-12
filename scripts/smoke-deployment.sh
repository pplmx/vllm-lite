#!/usr/bin/env bash
# Deployment smoke test — DEP-01.
#
# Verifies the deployment artifacts are mutually consistent and
# reference real paths/flags. Designed to run in any environment
# (no docker, no helm required) so it can guard CI against the
# regressions called out in
# docs/technical-due-diligence/production-readiness.md#dep-01:
#
#   - Dockerfile uses an MSRV that matches Cargo.toml
#   - Dockerfile HEALTHCHECK calls a real endpoint
#   - Dockerfile stages are named so docker-compose `target:` works
#   - docker-compose.yml is syntactically valid YAML
#   - docker-compose paths point at files that exist
#   - docker-compose env vars match what the CLI reads
#   - Helm chart env vars match what the CLI reads
#   - rust-toolchain.toml pins the MSRV
#
# Exit 0 on success, non-zero on the first failure.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

ok() {
    echo "ok  : $*"
}

# ---------------------------------------------------------------------------
# 1. rust-toolchain.toml exists and pins MSRV that matches Cargo.toml.
# ---------------------------------------------------------------------------
if [[ ! -f rust-toolchain.toml ]]; then
    fail "rust-toolchain.toml is missing (DEP-01: MSRV drift risk)"
fi
TOOLCHAIN_CHANNEL="$(grep -E '^channel\s*=' rust-toolchain.toml | head -1 | sed -E 's/.*"([^"]+)".*/\1/')"
CARGO_MSRV="$(grep -E '^rust-version\s*=' Cargo.toml | head -1 | sed -E 's/.*"([^"]+)".*/\1/')"
if [[ -z "$TOOLCHAIN_CHANNEL" || -z "$CARGO_MSRV" ]]; then
    fail "Could not parse rust-toolchain.toml channel or Cargo.toml rust-version"
fi
# channel is e.g. "1.88"; rust-version is e.g. "1.88" or "1.88.x".
# Compare just the major.minor prefix using cut.
TOOLCHAIN_MM="$(echo "$TOOLCHAIN_CHANNEL" | cut -d. -f1-2)"
CARGO_MSRV_MM="$(echo "$CARGO_MSRV" | cut -d. -f1-2)"
if [[ "$TOOLCHAIN_MM" != "$CARGO_MSRV_MM" ]]; then
    fail "rust-toolchain.toml channel ($TOOLCHAIN_CHANNEL) does not match Cargo.toml rust-version ($CARGO_MSRV)"
fi
ok "rust-toolchain.toml pins $TOOLCHAIN_CHANNEL (matches Cargo.toml MSRV $CARGO_MSRV)"

# ---------------------------------------------------------------------------
# 2. Dockerfile: rust version, locked build, named stages, real healthcheck.
# ---------------------------------------------------------------------------
if [[ ! -f Dockerfile ]]; then
    fail "Dockerfile is missing"
fi

# Builder stage must use the same MSRV.
if ! grep -qE "^FROM rust:${CARGO_MSRV_MM}-bookworm AS builder" Dockerfile; then
    fail "Dockerfile builder stage does not pin rust:${CARGO_MSRV_MM} (DEP-01: MSRV drift)"
fi
ok "Dockerfile builder uses rust:${CARGO_MSRV_MM}"

# Must use --locked (or --frozen) so lockfile drift fails loudly.
if ! grep -q "cargo build --locked" Dockerfile; then
    fail "Dockerfile does not use 'cargo build --locked' (DEP-01: lockfile drift)"
fi
ok "Dockerfile uses --locked"

# Runtime stage must be explicitly named (so docker-compose target: works).
if ! grep -q "AS runtime" Dockerfile; then
    fail "Dockerfile final stage is not named 'runtime' (DEP-01: docker-compose target: runtime breaks)"
fi
ok "Dockerfile has a named 'runtime' stage"

# HEALTHCHECK must probe a real endpoint, not a fictional --health-check flag.
# The directive may be multi-line (continued with `\`); use awk to fold
# continued lines before matching.
HEALTHCHECK_BODY="$(awk 'BEGIN{ORS=" "} {gsub(/\\$/, ""); print}' Dockerfile | tr -s ' ' | grep -oE 'HEALTHCHECK [^|;]*' || true)"
if echo "$HEALTHCHECK_BODY" | grep -q "vllm-server --health-check"; then
    fail "Dockerfile HEALTHCHECK calls 'vllm-server --health-check' which is not a real flag (DEP-01)"
fi
if ! echo "$HEALTHCHECK_BODY" | grep -q "/health/live"; then
    fail "Dockerfile HEALTHCHECK does not probe /health/live (DEP-01)"
fi
ok "Dockerfile HEALTHCHECK probes /health/live over HTTP"

# ---------------------------------------------------------------------------
# 3. docker-compose.yml: valid YAML, real paths, matching env vars.
# ---------------------------------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
    fail "python3 required for docker-compose smoke test"
fi
if ! python3 -c "import yaml" 2>/dev/null; then
    fail "PyYAML required for docker-compose smoke test"
fi
if ! python3 -c "import yaml; yaml.safe_load(open('docker-compose.yml'))"; then
    fail "docker-compose.yml is not valid YAML"
fi
ok "docker-compose.yml is valid YAML"

# target: must reference a stage that actually exists in the Dockerfile.
COMPOSE_TARGET="$(python3 - <<'PY'
import yaml
data = yaml.safe_load(open("docker-compose.yml"))
svc = data.get("services", {}).get("vllm-server", {})
build = svc.get("build") or {}
print(build.get("target") or "")
PY
)"
if [[ "$COMPOSE_TARGET" != "runtime" ]]; then
    fail "docker-compose.yml vllm-server.build.target is '$COMPOSE_TARGET', expected 'runtime'"
fi
ok "docker-compose.yml targets 'runtime' stage"

# All ./local bind mounts must point at existing files/dirs.
python3 - <<'PY' || { echo "docker-compose.yml bind mounts reference missing paths" >&2; exit 1; }
import os
import yaml
data = yaml.safe_load(open("docker-compose.yml"))
def check_mounts(svc_name):
    svc = data["services"].get(svc_name, {})
    for vol in svc.get("volumes") or []:
        if isinstance(vol, str) and vol.startswith("./"):
            host = vol.split(":")[0]
            if not os.path.exists(host):
                print(f"  {svc_name}: missing {host}", flush=True)
                return False
    return True
ok = True
for s in ("vllm-server", "prometheus", "grafana", "k6"):
    if s in data["services"] and not check_mounts(s):
        ok = False
if not ok:
    raise SystemExit(1)
PY
ok "docker-compose.yml bind mounts all exist"

# docker-compose must not emit env vars the CLI ignores.
if grep -E '^[[:space:]]*-[[:space:]]*MODEL_PATH[[:space:]]*=' docker-compose.yml; then
    fail "docker-compose.yml emits MODEL_PATH but the CLI reads VLLM_MODEL (DEP-01)"
fi
ok "docker-compose.yml does not emit the obsolete MODEL_PATH env var"

# ---------------------------------------------------------------------------
# 4. Helm chart: env vars match the CLI.
# ---------------------------------------------------------------------------
if [[ ! -f k8s/charts/vllm-lite/templates/deployment.yaml ]]; then
    fail "Helm chart deployment.yaml is missing"
fi
DEPLOYMENT=k8s/charts/vllm-lite/templates/deployment.yaml

# VLLM_MODEL must be present; MODEL_PATH must not.
if ! grep -qE '^\s*-\s*name:\s*VLLM_MODEL\b' "$DEPLOYMENT"; then
    fail "Helm chart deployment.yaml does not emit VLLM_MODEL (DEP-01: chart silently ignored)"
fi
if grep -qE '^\s*-\s*name:\s*MODEL_PATH\b' "$DEPLOYMENT"; then
    fail "Helm chart deployment.yaml still emits MODEL_PATH (DEP-01)"
fi
ok "Helm chart deployment.yaml emits VLLM_MODEL"

# Other env vars must match the CLI (VLLM_ prefix).
if grep -qE '^\s*-\s*name:\s*MAX_BATCH_SIZE\b' "$DEPLOYMENT"; then
    fail "Helm chart emits MAX_BATCH_SIZE but the CLI reads VLLM_MAX_BATCH_SIZE"
fi
if grep -qE '^\s*-\s*name:\s*NUM_KV_BLOCKS\b' "$DEPLOYMENT"; then
    fail "Helm chart emits NUM_KV_BLOCKS but the CLI reads VLLM_KV_BLOCKS"
fi
if grep -qE '^\s*-\s*name:\s*TENSOR_PARALLEL_SIZE\b' "$DEPLOYMENT"; then
    fail "Helm chart emits TENSOR_PARALLEL_SIZE but the CLI reads VLLM_TENSOR_PARALLEL_SIZE"
fi
ok "Helm chart env var names match CLI VLLM_* convention"

# values.yaml must be valid YAML.
if ! python3 -c "import yaml; yaml.safe_load(open('k8s/charts/vllm-lite/values.yaml'))"; then
    fail "Helm values.yaml is not valid YAML"
fi
ok "Helm values.yaml is valid YAML"

echo
echo "deployment smoke test: PASS"
