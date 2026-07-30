#!/usr/bin/env python3
"""Fail if the current live version leaks into any non-allowed file.

Live version = value of ``src/geopulse/_version.__version__``. Bumping
that file is the ONLY step needed to change the released version.
This script defends that contract.

Allowed files (may contain the live version literal):

* ``src/geopulse/_version.py`` — the source of truth itself.
* ``CHANGELOG.md`` — historical release-note entries.

Everywhere else, if the live version literal appears the check fails.
This catches the common drift bug: someone copies the current version
into a docs example, then a later bump changes ``_version.py``, and
the docs silently lie about which release they belong to.

The script does NOT try to catch arbitrary hardcoded version-shaped
strings (e.g. ``"0.5.0"`` written into README for no reason). Doing
that would need per-line escape markers to distinguish milestone
labels (in ``ROADMAP.md``, ``CONTRIBUTING.md``) from live-version
literals. If you need that stricter check later, extend the CHECKED
list with an inline-waiver marker convention.

Invocation
----------
Direct::

    python scripts/check_no_version_drift.py

Pre-commit (already wired in ``.pre-commit-config.yaml``): runs on
every commit, no file arguments needed (``always_run: true``).

Exit codes: 0 if clean, 1 if any hardcoded live-version literal was
found (with a per-line report to stderr).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Read _version.py textually rather than importing it. Avoids requiring
# an editable install inside the pre-commit hook and side-steps any
# import-time errors elsewhere in the package.
_version_src = (ROOT / "src/geopulse/_version.py").read_text(encoding="utf-8")
_match = re.search(r'__version__\s*=\s*"([^"]+)"', _version_src)
if _match is None:
    print("check-no-version-drift: could not parse __version__ from _version.py", file=sys.stderr)
    sys.exit(2)
LIVE = _match.group(1)

# Files that ARE allowed to contain the live version literal.
ALLOWED = {
    "src/geopulse/_version.py",
    "CHANGELOG.md",
}

# Files that MUST NOT contain the live version literal (unless whitelisted
# above). Globs are evaluated relative to the repo root.
CHECKED_GLOBS = [
    "README.md",
    "ROADMAP.md",
    "docs/**/*.md",
    "docs/**/*.rst",
    "docs/conf.py",
    "pyproject.toml",
]


def main() -> int:
    checked_paths: list[Path] = []
    for pat in CHECKED_GLOBS:
        checked_paths.extend(ROOT.glob(pat))

    failures: list[str] = []
    for path in sorted(set(checked_paths)):
        rel = str(path.relative_to(ROOT))
        if rel in ALLOWED:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if LIVE in line:
                failures.append(f"{rel}:{lineno}: {line.strip()}")

    if failures:
        print(
            f"check-no-version-drift: found {len(failures)} hardcoded live-version literal(s)",
            file=sys.stderr,
        )
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        print(
            f"\nThe live version {LIVE!r} must appear only in _version.py and\n"
            f"CHANGELOG.md. Either remove the literal above, or add the file to\n"
            f"the ALLOWED set in scripts/check_no_version_drift.py with a comment\n"
            f"explaining why the waiver is safe.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
