#!/usr/bin/env python3
"""Check that every `.cargo/mutants.toml` exclusion still matches a real mutant.

An exclusion pattern is line-anchored on purpose: if a refactor moves the code,
the pattern must stop matching so the exclusion lapses loudly instead of
silently covering different code. That safety property is only real if
something checks it, which is what this script does.

A pattern that matches nothing is reported; a pattern that matches more than one
mutant is also reported, because it is broader than its documented rationale.

Usage:
    python3 scripts/check_mutant_exclusions.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

CONFIG = Path(".cargo/mutants.toml")

# Feature flags each file needs for `cargo mutants --list` to see its mutants.
FEATURES = {
    "graph_step.rs": ["--features", "cuda-graph"],
}


def package_for(source: Path) -> str:
    """Map a source path under `crates/<dir>/` to that crate's package name."""
    parts = source.parts
    if len(parts) >= 2 and parts[0] == "crates":
        return f"vllm-{parts[1]}"
    return "vllm-core"


def file_for(pattern: str) -> str | None:
    """Recover the source file a pattern targets from its `<basename>.rs:` prefix."""
    match = re.match(r"([\w\\]+?\.rs):", pattern)
    if match is None:
        return None
    return match.group(1).replace("\\", "")


def main() -> int:
    config = tomllib.loads(CONFIG.read_text())
    patterns: list[str] = config.get("exclude_re", [])
    if not patterns:
        print("no exclusions configured")
        return 0

    listed: dict[str, list[dict]] = {}
    problems = 0

    for pattern in patterns:
        basename = file_for(pattern)
        if basename is None:
            print(f"MALFORMED  {pattern}")
            problems += 1
            continue

        if basename not in listed:
            matches = sorted(Path("crates").rglob(basename))
            if not matches:
                print(f"MISSING    {pattern}  (no source file named {basename})")
                problems += 1
                continue
            # A bare basename such as `mod.rs` matches MANY files across the
            # tree, so every candidate must be listed — picking only the first
            # made the exclusion for a nested module look STALE.
            records: list[dict] = []
            for source in matches:
                # The package comes from the source file's own crate: a
                # hard-coded `vllm-core` made every `crates/traits` (and
                # `crates/model`, `crates/server`) pattern look STALE.
                package = package_for(source)
                completed = subprocess.run(
                    [
                        "cargo",
                        "mutants",
                        "--package",
                        package,
                        "--file",
                        str(source),
                        "--list",
                        "--json",
                        # `--no-config`: without it the exclusions under test are
                        # already applied, so every pattern would look STALE.
                        "--no-config",
                        *FEATURES.get(basename, []),
                    ],
                    capture_output=True,
                    text=True,
                )
                if completed.stdout.strip():
                    records.extend(json.loads(completed.stdout))
            listed[basename] = records

        regex = re.compile(pattern)
        hits = [record["name"] for record in listed[basename] if regex.search(record["name"])]
        if not hits:
            print(f"STALE      {pattern}  (matches no mutant; line likely moved)")
            problems += 1
        elif len(hits) > 1:
            print(f"TOO WIDE   {pattern}  (matches {len(hits)} mutants)")
            for name in hits:
                print(f"             {name}")
            problems += 1
        else:
            print(f"ok         {pattern}")

    print(f"\n{len(patterns)} exclusions checked, {problems} problem(s)")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
