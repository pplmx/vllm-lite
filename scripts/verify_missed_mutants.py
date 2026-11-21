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
import signal
import subprocess
import sys
from pathlib import Path


class StaleSpanError(RuntimeError):
    """The recorded span does not point at the code it describes."""


def span_text(text: str, span: dict) -> str:
    """The exact source text a span covers."""
    start, end = span["start"], span["end"]
    lines = text.splitlines(keepends=True)
    if start["line"] == end["line"]:
        return lines[start["line"] - 1][start["column"] - 1 : end["column"] - 1]
    parts = [lines[start["line"] - 1][start["column"] - 1 :]]
    parts.extend(lines[start["line"] : end["line"] - 1])
    parts.append(lines[end["line"] - 1][: end["column"] - 1])
    return "".join(parts)


def check_span_is_current(text: str, record: dict) -> None:
    """Refuse to apply a span drawn from a different revision of the file.

    `cargo mutants --list --json` records offsets for the file as it stood when
    the list was produced. Reusing that JSON after the file changes edits the
    WRONG line silently - that is how a `-` -> `/` mutation for `log_softmax`
    landed inside `renormalized_top_p_logprobs`. The mutant's own `diff` names
    the original text on its `-` lines, so comparing against it catches the
    drift.
    """
    removed = [
        line[1:]
        for line in record["diff"].splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]
    if not removed:
        return
    expected = removed[0].strip()
    actual = span_text(text, record["span"]).strip()
    # The span may cover a single operator while the diff quotes the whole line,
    # and `fn_value` mutants quote the signature instead of the body. What must
    # hold in every case is that the span's text still appears in the line the
    # mutant claims to edit.
    if actual and actual not in expected:
        raise StaleSpanError(
            f"{record['name']}: span covers {actual!r} but the mutant targets "
            f"{expected!r} - regenerate the mutant list from the current file"
        )


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
    parser.add_argument("mutants_json", type=Path, nargs="?")
    parser.add_argument(
        "--test-cmd",
        required=True,
        help="shell command run for each mutant; non-zero exit means KILLED",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=None,
        help=(
            "source file whose mutant list is regenerated with `cargo mutants "
            "--list --json --no-config` before verifying (repeatable); prefer "
            "this over a hand-carried JSON, which goes stale on any edit"
        ),
    )
    parser.add_argument(
        "--features",
        action="append",
        default=None,
        help="cargo feature for relisting (repeatable)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="per-mutant wall-clock limit for the test command, in seconds",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="restrict to mutants whose `name` contains this substring (repeatable)",
    )
    args = parser.parse_args()

    if args.file:
        combined: list[dict] = []
        for source in args.file:
            command = [
                "cargo", "mutants", "--package", "vllm-core",
                "--file", source, "--list", "--json", "--no-config",
            ]
            for feature in args.features or []:
                command += ["--features", feature]
            completed = subprocess.run(command, capture_output=True, text=True, check=True)
            combined.extend(json.loads(completed.stdout))
        records = combined
    elif args.mutants_json is not None:
        records = json.loads(args.mutants_json.read_text())
    else:
        parser.error("pass a mutants JSON positional argument or --file")
    if args.only:
        records = [
            r for r in records if any(sub in r["name"] for sub in args.only)
        ]

    killed: list[str] = []
    survived: list[str] = []
    unviable: list[str] = []

    pending: dict[Path, str] = {}

    def restore_pending(*_unused: object) -> None:
        for pending_path, pending_text in pending.items():
            pending_path.write_text(pending_text)
        pending.clear()

    def on_signal(signum: int, _frame: object) -> None:
        restore_pending()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    try:
        for record in records:
            path = Path(record["file"])
            original = path.read_text()
            try:
                check_span_is_current(original, record)
            except StaleSpanError as error:
                print(f"STALE SPAN: {error}")
                return 2
            pending[path] = original
            try:
                path.write_text(apply_span_edit(original, record["span"], record["replacement"]))
                try:
                    completed = subprocess.run(
                        args.test_cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=args.timeout,
                    )
                except subprocess.TimeoutExpired:
                    print(f"TIMEOUT after {args.timeout}s: {record['name']}")
                    survived.append(record["name"])
                    continue
                if "error[E]" in completed.stderr or "error: could not compile" in completed.stderr:
                    unviable.append(record["name"])
                elif completed.returncode != 0:
                    killed.append(record["name"])
                else:
                    survived.append(record["name"])
            finally:
                path.write_text(original)
                pending.pop(path, None)
    finally:
        restore_pending()

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
