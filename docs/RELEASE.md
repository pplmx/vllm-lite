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
2. `build` job: cross-compiles `vllm-server` for the OS matrix and
   packages the binary named
   `vllm-server-${{ needs.meta.outputs.version }}-${{ matrix.target }}`.
3. `release` job: runs `git-cliff` to generate release notes and
   creates the GitHub Release with the binary artifacts attached.

Docker build/push and Helm Chart packaging are not yet wired into
`release.yml` (the due-diligence report flagged these as P1 — see
the roadmap for when they'll land). The Dockerfile and Chart.yaml
already accept the manifest fields via build args / `helm --set`
so the wiring is mechanical once the GHCR secret is configured.

## Why `0.1.0` (and not the internal milestone numbers)

Internal milestones (`v18`, `v22`, `v31.0` in `.planning/`) are
project-internal sequencing aids. They are **not** user-facing
versions. External users see SemVer `0.1.0`, `0.2.0`, … with the
contract that breaking API changes bump the minor, additive
features bump the minor, and patches fix bugs. Until `1.0.0`, the
project reserves the right to make breaking changes in any release;
each one ships with a MIGRATING entry in `CHANGELOG.md`.
