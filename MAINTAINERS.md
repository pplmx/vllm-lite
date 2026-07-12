# Maintainers

> **Doc navigation:** [`.planning/DOC-MAP.md`](./.planning/DOC-MAP.md)

This file lists the people responsible for reviewing and merging changes to
vLLM-lite. The technical-due-diligence governance report flagged a
single-maintainer bus factor as a risk; entries below are **areas of
responsibility** rather than personal commitments — anyone listed can
vouch for a change in their area.

## Active maintainers

| Maintainer          | GitHub handle        | Areas of responsibility                                | Contact                  |
| ------------------- | -------------------- | ------------------------------------------------------ | ------------------------ |
| Mystvio (founder)   | `@pplmx`             | Architecture, scheduler, model registry, release owner | GitHub Security Advisory |

> Email addresses are intentionally NOT hard-coded here. Open a GitHub
> Security Advisory or Discussion to reach the maintainer team; this
> keeps the public source tree from drifting when ownership rotates.

## How to request a review

1. Open a PR against `main`. CI (`just ci`) must pass before review.
2. Tag the relevant maintainer in the PR description (e.g. `@pplmx`
   for an architecture change).
3. If the PR is **security-sensitive** (auth bypass, model sandbox
   escape, prompt-injection mitigation, etc.), use the
   [private security advisory][sec-adv] workflow instead of a public
   PR. Once a fix lands it can be backported under embargo.

[sec-adv]: https://github.com/pplmx/vllm-lite/security/advisories/new

## Review SLO

- Initial triage within **48 hours** on weekdays.
- Substantive review within **5 business days** for non-trivial PRs.
- Security advisories: see [`SECURITY.md`](./SECURITY.md) — faster
  turnaround is targeted for vulnerability reports.

## Becoming a maintainer

We accept new maintainers when a contributor has shipped multiple
substantial PRs and demonstrates judgement on at least one of:

- Backwards-compatible API additions without breaking the
  `cargo public-api` baseline (`just public-api-check`).
- A bug-fix that surfaces an existing failure mode (not just a code
  typo).
- A review-thread that closes a defect category, not just an instance.

Process: open a PR adding your row to the table above. The existing
maintainers ack and merge. No public ceremony is required.

## Why this file exists

The technical-due-diligence governance section noted that:

- `CODE_OF_CONDUCT.md` referenced `[INSERT CONTACT EMAIL]` (now fixed —
  see the enforcement section).
- `SECURITY.md` had no dedicated private-advisory channel (now fixed).
- `MAINTAINERS.md` was missing entirely (this file).

If you spot another governance gap, please open an issue or PR — the
audit trail of "what was fixed when" lives in
[`.planning/STATE.md`](./.planning/STATE.md) and the git history.
