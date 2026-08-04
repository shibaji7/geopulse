#!/usr/bin/env python3
"""Fail if a function's numpy-style docstring drifts from its signature/body.

Catches the two drift bugs described in
`GH-17 <https://github.com/shibaji7/geopulse/issues/17>`_ that neither
``pydoclint`` nor in-tree ``numpydoc-validation`` could catch without
also flagging the project's correct-by-convention style choices
(type hints in both signature and docstring; ``Attributes`` instead of
``Parameters`` on frozen dataclasses):

1. A public function's argument was renamed but its ``Parameters``
   section still documents the old name (and/or is missing the new
   one).
2. A ``raise SomeError(...)`` was added to a function body but
   ``SomeError`` was never added to that function's ``Raises`` section.

Deliberately narrow, per GH-17's acceptance criteria and non-goals:

* Only checks functions/methods whose docstring **already has** a
  ``Parameters`` (or ``Raises``) section. A function with no section
  at all is not flagged here — docstring *coverage* is
  ``interrogate``'s job, not this script's. This is what keeps the
  false-positive rate at zero: we only cross-check claims the
  docstring is already making, never demand new ones.
* ``*args`` / ``**kwargs`` are excluded from the Parameters
  cross-check (rarely renamed, rarely worth the noise).
* ``Raises`` is checked in one direction only: exception classes
  actually raised in the body but absent from the docstring. A
  documented-but-not-directly-raised exception is common when the
  raise happens in a helper the function calls, so that direction is
  intentionally not flagged.
* Does not touch ``Returns`` — out of scope for GH-17; revisit only if
  a real Returns-drift bug is observed (see the issue's "Non-goals").

Invocation
----------
Direct::

    python scripts/check_docstring_drift.py

Pre-commit: wired in ``.pre-commit-config.yaml``, runs on
``src/geopulse/**/*.py`` changes.

Exit codes: 0 if clean, 1 if any drift was found (per-line report to
stderr).
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src" / "geopulse"

# Matches a numpydoc section header, e.g. "Parameters\n----------".
_SECTION_HEADER_RE = re.compile(r"^([A-Za-z][A-Za-z ]*)\n *-{3,}\s*$", re.MULTILINE)

# Matches a numpydoc parameter/attribute entry line, e.g.
# "freqs_Hz : numpy.ndarray" or "a, b : int" (comma-separated names
# sharing one type). Must be indented (body of the section) and not
# itself a continuation/description line.
_PARAM_ENTRY_RE = re.compile(r"^ {0,4}(\*{0,2}[A-Za-z_][A-Za-z0-9_]*(?:, *\*{0,2}[A-Za-z_][A-Za-z0-9_]*)*) *:(?!\w+:)")

# Matches a numpydoc Raises entry: an exception class name alone on its
# own (optionally indented) line, e.g. "ValueError" or "geopulse.GeoPulseError".
_RAISES_ENTRY_RE = re.compile(r"^ {0,4}([A-Za-z_][A-Za-z0-9_.]*)\s*$")

EXCLUDED_PARAM_NAMES = {"self", "cls"}


def _split_sections(docstring: str) -> dict[str, str]:
    """Split a numpydoc-style docstring into ``{section_name: body}``."""
    matches = list(_SECTION_HEADER_RE.finditer(docstring))
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(docstring)
        sections[name] = docstring[start:end]
    return sections


def _documented_params(section_body: str) -> set[str]:
    names: set[str] = set()
    for line in section_body.splitlines():
        m = _PARAM_ENTRY_RE.match(line)
        if not m:
            continue
        for raw in m.group(1).split(","):
            names.add(raw.strip().lstrip("*"))
    return names


def _documented_raises(section_body: str) -> set[str]:
    names: set[str] = set()
    for line in section_body.splitlines():
        if not line.strip():
            continue
        m = _RAISES_ENTRY_RE.match(line)
        if m:
            # Take only the final dotted component, e.g.
            # "geopulse.GeoPulseError" -> "GeoPulseError", so it can be
            # compared against the bare name a raise statement uses.
            names.add(m.group(1).rsplit(".", 1)[-1])
    return names


def _signature_params(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    args = node.args
    names = [a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]]
    return {n for n in names if n not in EXCLUDED_PARAM_NAMES}


def _raised_exception_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Raise) or child.exc is None:
            continue
        target = child.exc
        # `raise SomeError(...)` -> Call(func=Name/Attribute); `raise SomeError` (no call) -> Name/Attribute.
        if isinstance(target, ast.Call):
            target = target.func
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
        # A bare `raise` (re-raise, no target) yields neither Name nor
        # Attribute here and is intentionally skipped.
    return names


def _iter_functions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _qualname(path: Path, node: ast.AST, stack: list[str]) -> str:
    rel = path.relative_to(ROOT)
    dotted = ".".join(stack)
    return f"{rel}:{node.lineno} {dotted}"


def _check_file(path: Path) -> list[str]:
    failures: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return failures

    # Walk with a name stack so nested class/function context is reported.
    def walk(node: ast.AST, stack: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _check_function(child, stack + [child.name])
                walk(child, stack + [child.name])
            elif isinstance(child, ast.ClassDef):
                walk(child, stack + [child.name])
            else:
                walk(child, stack)

    def _check_function(node: ast.FunctionDef | ast.AsyncFunctionDef, stack: list[str]) -> None:
        doc = ast.get_docstring(node)
        if not doc:
            return
        sections = _split_sections(doc)
        loc = _qualname(path, node, stack)

        if "Parameters" in sections:
            documented = _documented_params(sections["Parameters"])
            actual = _signature_params(node)
            missing_from_doc = actual - documented
            missing_from_sig = documented - actual
            if missing_from_doc:
                failures.append(
                    f"{loc}: signature has {sorted(missing_from_doc)} not in Parameters section"
                )
            if missing_from_sig:
                failures.append(
                    f"{loc}: Parameters section documents {sorted(missing_from_sig)} not in signature"
                )

        if "Raises" in sections:
            documented_raises = _documented_raises(sections["Raises"])
            actual_raises = _raised_exception_names(node)
            undocumented = actual_raises - documented_raises
            if undocumented:
                failures.append(
                    f"{loc}: raises {sorted(undocumented)} not listed in Raises section"
                )

    walk(tree, [])
    return failures


def main() -> int:
    all_failures: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        all_failures.extend(_check_file(path))

    if all_failures:
        print(
            f"check-docstring-drift: found {len(all_failures)} docstring/signature mismatch(es)",
            file=sys.stderr,
        )
        for f in all_failures:
            print(f"  {f}", file=sys.stderr)
        print(
            "\nEach line above is either a renamed/added argument not reflected in the\n"
            "Parameters section, or a raised exception not listed in the Raises section.\n"
            "Update the docstring to match the code (or vice versa) and re-commit.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
