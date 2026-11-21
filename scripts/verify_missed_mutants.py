#!/usr/bin/env python3
"""Re-check cargo-mutants' MISSED verdicts by applying each one by hand.

Why this exists: cargo-mutants has repeatedly reported MISSED for mutants that
a test in this tree DOES kill when the edit is applied manually (see the
CHANGELOG entries for `prefill.rs:168`, `chunked.rs:64` and the verifier's
`bonus_offset`). Treating MISSED as a coverage gap therefore overstates the
debt, and treating it as noise hides real ones. This script decides per mutant
by applying the recorded span edit and running a filtered test command.

Usage:
    python3 scripts/verify_missed_mutants.py <mutants-json> --test-cmd "<cargo test ...>"

The JSON is `cargo mutants --list --json` output (which carries `span` and
`replacement`), filtered to the names the scan reported as missed.

Limits, learned the hard way:

* The recorded `span` is the author's best guess at the mutated expression, not
  a guarantee. Verify a surprising verdict by reading the line before trusting
  it — a mis-attributed column once made a `> budget_room` guard look like a
  `> chunk_cap` one.
* A surviving mutant still needs a human decision between "real gap", "the
  fixture cannot express the difference", and "equivalent by construction".
  This script separates signal from noise; it does not conclude.
* `--test-cmd` must exit non-zero on failure. A `grep`-based command silently
  reports survivors as killed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def apply_span_edit(text: str, span: dict, replacement: str) -> str:
    """Replace the span's text with `replacement`, keeping everything else."""
    start, end = span["start"], span["end"]
    lines = text.splitlines(keepends=True)
    if start["line"] == end["line"]:
        line = lines[start["line"] - 1]
        patched = (
            line[: start["column"] - 1] + replacement + line[end["column"] - 1 :]
        )
        lines[start["line"] - 1] = patched
    else:
        first = lines[start["line"] - 1]
        last = lines[end["line"] - 1]
        merged = first[: start["column"] - 1] + replacement + last[end["column"] - 1 :]
        lines[start["line"] - 1] = merged
        for idx in range(end["line"] - 1, start["line"] - 1, -1):
            del lines[idx]
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mutants_json", type=Path)
    parser.add_argument(
        "--test-cmd",
        required=True,
        help="shell command run for each mutant; non-zero exit means KILLED",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="restrict to mutants whose `name` contains this substring (repeatable)",
    )
    args = parser.parse_args()

    records = json.loads(args.mutants_json.read_text())
    if args.only:
        records = [
            r for r in records if any(sub in r["name"] for sub in args.only)
        ]

    killed: list[str] = []
    survived: list[str] = []
    unviable: list[str] = []

    for record in records:
        path = Path(record["file"])
        original = path.read_text()
        try:
            path.write_text(apply_span_edit(original, record["span"], record["replacement"]))
            completed = subprocess.run(
                args.test_cmd,
                shell=True,
                capture_output=True,
                text=True,
            )
            if "error[E" in completed.stderr or "error: could not compile" in completed.stderr:
                unviable.append(record["name"])
            elif completed.returncode != 0:
                killed.append(record["name"])
            else:
                survived.append(record["name"])
        finally:
            path.write_text(original)

    for label, entries in (
        ("KILLED (cargo-mutants was wrong)", killed),
        ("SURVIVED (genuine gap)", survived),
        ("UNVIABLE (does not compile)", unviable),
    ):
        print(f"\n{label}: {len(entries)}")
        for name in entries:
            print(f"  {name}")

    return 0 if not survived else 1


if __name__ == "__main__":
    sys.exit(main())
