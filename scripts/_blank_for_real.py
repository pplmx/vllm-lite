#!/usr/bin/env python3
"""_blank_for_real.py — blank out test mods / hidden / derive-decorated pub items.

Reads a Rust source file from stdin, writes the same file to stdout with the
following lines replaced by empty strings (preserving line numbers):

  1. All lines inside any `#[cfg(test)] mod <name> { ... }` block — test code
     is not part of the public API surface and should not count toward
     documentation coverage.
  2. The pub item line itself (and only that line) when the immediately
     preceding attribute block contains any of:
       - `#[doc(hidden)]`  — the item is intentionally undocumented
       - `#[derive(...)]`  — the item is macro-generated boilerplate
       - `#[automatically_derived]` — derive macro output

Line numbers are preserved so downstream tools can still report file:line.
"""
import re
import sys

ATTR_RE = re.compile(r"^\s*#\[\s*(?P<name>[A-Za-z0-9_:(),\.=\s\"]+?)\s*\]\s*$")
PUB_RE = re.compile(
    r"^\s*pub\s+(?:struct|enum|fn|trait|type|const|static|mod|async\s+fn)\b"
)
CFG_TEST_RE = re.compile(r"^\s*#\[\s*cfg\s*\(\s*test\s*\)\s*\]\s*$")
MOD_DECL_RE = re.compile(r"^\s*mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*")
MOD_OPEN_INLINE_RE = re.compile(r"\{")


def find_cfg_test_mod_ranges(lines):
    """Return list of (start_idx, end_idx_exclusive) line ranges to blank.

    A cfg(test) mod block starts at a `#[cfg(test)]` attribute line followed
    (after optional blanks/other attributes) by a `mod <name> {` declaration,
    and ends at the matching closing brace.
    """
    ranges = []
    n = len(lines)
    i = 0
    while i < n:
        if not CFG_TEST_RE.match(lines[i]):
            i += 1
            continue
        # Look ahead up to 6 lines for a mod declaration.
        j = i + 1
        mod_line = None
        while j < n and j < i + 7:
            s = lines[j].strip()
            if s == "":
                j += 1
                continue
            if s.startswith("#["):
                j += 1
                continue
            if MOD_DECL_RE.match(lines[j]):
                mod_line = j
            break
        if mod_line is None:
            i += 1
            continue
        # Find the opening brace for this mod (may be on same line or a later line).
        brace_idx = None
        if "{" in lines[mod_line]:
            brace_idx = mod_line
        else:
            for k in range(mod_line + 1, min(mod_line + 4, n)):
                if "{" in lines[k]:
                    brace_idx = k
                    break
        if brace_idx is None:
            i += 1
            continue
        # Walk forward from brace_idx, counting braces to find the matching close.
        depth = 0
        end_idx = None
        for k in range(brace_idx, n):
            for ch in lines[k]:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end_idx = k + 1
                        break
            if end_idx is not None:
                break
        if end_idx is None:
            end_idx = n  # unmatched — blank to EOF
        ranges.append((i, end_idx))
        i = end_idx
    return ranges


def main():
    content = sys.stdin.read()
    lines = content.split("\n")
    # Preserve trailing newline semantics: split("\n") leaves an empty trailing
    # element if the file ended with "\n". Drop it for processing and re-add.
    trailing_nl = False
    if lines and lines[-1] == "":
        trailing_nl = True
        lines = lines[:-1]

    n = len(lines)

    # Mark lines that should be blanked because they are inside a #[cfg(test)] mod.
    blank = [False] * n
    for start, end in find_cfg_test_mod_ranges(lines):
        for k in range(start, end):
            if k < n:
                blank[k] = True

    # Walk through and detect pub items decorated by hidden/derive/auto-derived.
    # We maintain a rolling window of the most recent 6 attribute lines.
    last_attrs = []
    for idx, line in enumerate(lines):
        if blank[idx]:
            last_attrs = []
            continue
        m = ATTR_RE.match(line)
        if m is not None:
            last_attrs.append(line)
            if len(last_attrs) > 6:
                last_attrs = last_attrs[-6:]
            continue
        if line.strip() == "":
            # blank lines preserve the attribute window
            continue
        # Non-attribute, non-blank line. Check if it's a decorated pub item.
        if PUB_RE.match(line) and last_attrs:
            joined = "\n".join(last_attrs)
            if (
                "doc(hidden)" in joined
                or "derive(" in joined
                or "automatically_derived" in joined
            ):
                blank[idx] = True
        # Any other non-attribute, non-blank line clears the attribute window.
        last_attrs = []

    # Emit blanked lines.
    out_lines = ["" if blank[i] else lines[i] for i in range(n)]
    sys.stdout.write("\n".join(out_lines))
    if trailing_nl:
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
