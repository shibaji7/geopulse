"""Tests for ``scripts/check_docstring_drift.py``.

Exercises the checker as a library (importing its functions directly)
rather than shelling out, so failures point at exactly which helper
regressed.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "check_docstring_drift.py"
_spec = importlib.util.spec_from_file_location("check_docstring_drift", _SCRIPT_PATH)
drift = importlib.util.module_from_spec(_spec)
sys.modules["check_docstring_drift"] = drift
_spec.loader.exec_module(drift)


def _failures_for(source: str) -> list[str]:
    """Parse ``source`` and return the checker's failures for its one function."""
    import ast

    tree = ast.parse(textwrap.dedent(source))
    func = tree.body[0]
    assert isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef))

    failures: list[str] = []
    doc = ast.get_docstring(func)
    sections = drift._split_sections(doc) if doc else {}

    if "Parameters" in sections:
        documented = drift._documented_params(sections["Parameters"])
        actual = drift._signature_params(func)
        if actual - documented:
            failures.append("missing_from_doc")
        if documented - actual:
            failures.append("missing_from_sig")

    if "Raises" in sections:
        documented_raises = drift._documented_raises(sections["Raises"])
        actual_raises = drift._raised_exception_names(func)
        if actual_raises - documented_raises:
            failures.append("undocumented_raise")

    return failures


def test_matching_params_pass() -> None:
    source = '''
        def f(a, b):
            """Do a thing.

            Parameters
            ----------
            a : int
                First.
            b : int
                Second.
            """
    '''
    assert _failures_for(source) == []


def test_renamed_arg_flagged_both_directions() -> None:
    source = '''
        def f(a, renamed):
            """Do a thing.

            Parameters
            ----------
            a : int
                First.
            old_name : int
                Second.
            """
    '''
    failures = _failures_for(source)
    assert "missing_from_doc" in failures  # `renamed` not documented
    assert "missing_from_sig" in failures  # `old_name` no longer a param


def test_no_parameters_section_is_not_flagged() -> None:
    # No Parameters section at all -> not this checker's job (interrogate's).
    source = '''
        def f(a, b):
            """Do a thing with no structured Parameters section."""
    '''
    assert _failures_for(source) == []


def test_sphinx_role_in_description_not_mistaken_for_param() -> None:
    # Regression test: a continuation line beginning with a word + a
    # Sphinx cross-reference role (":class:`...`") must not be parsed
    # as a new parameter entry.
    source = '''
        def f(impedance):
            """Do a thing.

            Parameters
            ----------
            impedance : Impedance
                Any :class:`~geopulse.earth.impedance.Impedance` subclass; the
                type is not inspected here.
            """
    '''
    assert _failures_for(source) == []


def test_star_args_excluded_from_comparison() -> None:
    source = '''
        def f(a, *args, **kwargs):
            """Do a thing.

            Parameters
            ----------
            a : int
                First.
            """
    '''
    assert _failures_for(source) == []


def test_undocumented_raise_flagged() -> None:
    source = '''
        def f():
            """Do a thing.

            Raises
            ------
            ValueError
                If bad.
            """
            if True:
                raise TypeError("also possible")
    '''
    assert "undocumented_raise" in _failures_for(source)


def test_documented_but_not_directly_raised_is_not_flagged() -> None:
    # Intentionally lenient in this direction: the exception may be
    # raised by a helper this function calls.
    source = '''
        def f():
            """Do a thing.

            Raises
            ------
            ValueError
                If bad (raised by a helper, not directly here).
            """
            helper()
    '''
    assert _failures_for(source) == []


def test_self_and_cls_excluded() -> None:
    source = '''
        def f(self, a):
            """Do a thing.

            Parameters
            ----------
            a : int
                First.
            """
    '''
    assert _failures_for(source) == []


@pytest.mark.parametrize(
    "attr",
    [
        "_split_sections",
        "_documented_params",
        "_documented_raises",
        "_signature_params",
        "_raised_exception_names",
    ],
)
def test_helpers_are_importable(attr: str) -> None:
    assert hasattr(drift, attr)


def test_repo_is_currently_clean() -> None:
    """The whole point of GH-17: this must pass cleanly on ``main``."""
    failures: list[str] = []
    for path in sorted(drift.SRC.rglob("*.py")):
        failures.extend(drift._check_file(path))
    assert failures == [], f"unexpected docstring drift on main: {failures}"
