# Release process

> **Doc navigation:** [`.planning/DOC-MAP.md`](../.planning/DOC-MAP.md)

## Single source of truth

GOV-01 (technical due diligence): the workspace version declared in
[`Cargo.toml`](../Cargo.toml) under `[workspace.package]` is the
*only* number that drives every release surface. All four downstream
artifacts — the Docker image tag, the Helm Chart `version` /
`appVersion`, the GitHub Release tag, and the binary artifact
filenames — are derived from it by
[`scripts/release-manifest.sh`](../scripts/release-manifest.sh).

```text
                    Cargo.toml [workspace.package] version = "0.1.0"
                                      │
                                      ▼
                       scripts/release-manifest.sh
                                      │
              ┌──────────┬───────────┼───────────┬──────────┐
              ▼          ▼           ▼           ▼          ▼
         Dockerfile   Chart.yaml   release.yml  binaries  CHANGELOG
         (OCI labels) (helm pkg)   (GitHub tag) (filename) (cliff)
```

The script writes a `target/release-manifest.env` file with shell-
sourced `KEY=value` pairs. Every release-time tool reads from there
instead of re-deriving from Cargo.toml — so the four surfaces can
never disagree.

## Manifest fields

`scripts/release-manifest.sh` emits:

| Variable                | Meaning                                                    |
| ----------------------- | ---------------------------------------------------------- |
| `VLLM_VERSION`          | Workspace version (e.g. `0.1.0`)                           |
| `VLLM_IS_PRERELEASE`    | `true` if `version` contains `-` (SemVer pre-release)      |
| `VLLM_IMAGE_TAG`        | Docker image tag, bare version (e.g. `0.1.0`)              |
| `VLLM_IMAGE_TAG_FULL`   | With registry prefix (set via `--registry REGISTRY`)       |
| `VLLM_CHART_VERSION`    | Helm Chart.yaml `version` (same as workspace version)      |
| `VLLM_CHART_APP_VERSION`| Helm Chart.yaml `appVersion` (same as workspace version)  |
| `VLLM_GIT_SHA`          | Full git HEAD SHA                                          |
| `VLLM_GIT_SHA_SHORT`    | First 8 chars of `VLLM_GIT_SHA`                            |
| `VLLM_GIT_DESCRIBE`     | `git describe --always --tags --dirty`                     |
| `VLLM_RUSTC_VERSION`    | `rustc -V` output                                          |
| `VLLM_BUILD_TIMESTAMP`  | UTC ISO 8601 timestamp                                     |

## Bumping the version

1. Edit `Cargo.toml` → `[workspace.package] version = "X.Y.Z"`.
2. Add a CHANGELOG entry under `[Unreleased]` describing the change.
3. Run `scripts/release-manifest.sh --validate X.Y.Z` locally to
   catch typos before tagging.
4. Tag the commit: `git tag -a vX.Y.Z -m "Release vX.Y.Z" && git push origin vX.Y.Z`.

The `--validate` flag exits non-zero if the tag doesn't match the
workspace version — `.github/workflows/release.yml` runs the same
check before any build step, so a stray tag fails fast instead of
shipping an inconsistent release.

## Local dry-run

```bash
# Print the manifest (handy for debugging release.yml locally)
scripts/release-manifest.sh

# Validate a candidate tag without writing anything
scripts/release-manifest.sh --validate 0.1.0

# Write to a file you can source in another shell
scripts/release-manifest.sh --out target/release-manifest.env
set -a; source target/release-manifest.env; set +a
echo "$VLLM_VERSION"   # → 0.1.0
```

## What `release.yml` does

1. `meta` job: runs `release-manifest.sh`, exports the variables as
   job outputs, and uploads `target/release-manifest.env` as an
   artifact so the downstream jobs can `source` it.
2. `build` job: cross-compiles `vllm-server` for the OS matrix,
   packages the binary named
   `vllm-server-${{ needs.meta.outputs.version }}-${{ matrix.target }}`,
   and emits a CycloneDX SBOM per target (see "Software Bill of
   Materials" below).
3. `release` job: runs `git-cliff` to generate release notes and
   creates the GitHub Release with the binary + SBOM artifacts
   attached.

Docker build/push and Helm Chart packaging are not yet wired into
`release.yml` (the due-diligence report flagged these as P1 — see
the roadmap for when they'll land). The Dockerfile and Chart.yaml
already accept the manifest fields via build args / `helm --set`
so the wiring is mechanical once the GHCR secret is configured.

## Software Bill of Materials

Every GitHub Release attaches a CycloneDX JSON SBOM per build target,
emitted by the `build` job via
[`anchore/sbom-action`](https://github.com/marketplace/actions/anchore-sbom-action)
(`syft` under the hood). Each SBOM file:

- Lives next to the corresponding binary artifact
  (`sbom-<target>.<ext>`, where the extension depends on the `format`
  parameter; `cyclonedx-json` produces a `.cdx.json`).
- Captures every Rust crate in `Cargo.lock`, vendored C libraries that
  ended up linked into the binary, and any system libraries detected
  by syft's ELF/PE/Mach-O scanners.
- Is uploaded both as a standalone workflow artifact (`sbom-<target>`)
  and as part of the GitHub Release attachment glob
  (`artifacts/**/*` in the `release` job).

**Verify a release locally:**

```bash
# Download the SBOM alongside the binary, then sanity-check it
gh release download v0.1.0 -p 'sbom-x86_64-unknown-linux-gnu*'
ls sbom-x86_64-unknown-linux-gnu.*
jq '.components | length' sbom-x86_64-unknown-linux-gnu.cdx.json
# → number of distinct packages syft detected

# Cross-check a known crate
jq '.components[] | select(.name == "tokio")' sbom-x86_64-unknown-linux-gnu.cdx.json
```

**Why this matters:** downstream consumers running vLLM-lite in
regulated environments (SOC 2 / FedRAMP / air-gapped vendor review)
frequently need an SBOM to satisfy their intake checklist. Emitting
one per build target means the artifact is already in place — no
post-release `cargo metadata` workaround needed.

Checksums (`sha256sum`) and signed build provenance (SLSA / in-toto)
are still missing from `release.yml`; they are tracked as a separate
follow-up against `docs/technical-due-diligence/engineering-quality.md`
§7.

## Why `0.1.0` (and not the internal milestone numbers)

Internal milestones (`v18`, `v22`, `v31.0` in `.planning/`) are
project-internal sequencing aids. They are **not** user-facing
versions. External users see SemVer `0.1.0`, `0.2.0`, … with the
contract that breaking API changes bump the minor, additive
features bump the minor, and patches fix bugs. Until `1.0.0`, the
project reserves the right to make breaking changes in any release;
each one ships with a MIGRATING entry in `CHANGELOG.md`.
