"""Sphinx configuration for GeoPulse documentation."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the src/ layout importable so autodoc can find geopulse.
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

# --- Project metadata ----------------------------------------------------
project = "GeoPulse"
author = "Shibaji Chakraborty, David Boteler"
copyright = "2026, Shibaji Chakraborty"

# Pull version from the package itself so docs stay in sync with releases.
# No fallback: if the import fails, the docs build must fail loudly. A
# silently-shipped stale version on RTD is worse than a broken build.
# E402 waived because sys.path.insert above must run before this import.
from geopulse._version import __version__ as release  # noqa: E402

version = ".".join(release.split(".")[:2])

# --- Extensions ----------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",  # import and pull docstrings
    "sphinx.ext.autosummary",  # auto-generate module stub pages
    "sphinx.ext.napoleon",  # NumPy/Google docstring parsing
    "sphinx.ext.intersphinx",  # cross-link to numpy/scipy/h5py
    "sphinx.ext.mathjax",  # render docstring equations
    "sphinx.ext.viewcode",  # [source] links on each API entry
    "sphinx_autodoc_typehints",  # render type hints in the sig
    "myst_parser",  # let us .md-include README etc.
]

# --- Autosummary + autodoc knobs -----------------------------------------
autosummary_generate = True  # actually write the module stub pages
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"  # types in the body, not the signature
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False  # merge :rtype: into :returns:

# Suppress warnings from stub modules that intentionally raise
# NotImplementedYetError on import-side attribute access.
autodoc_mock_imports: list[str] = []

# --- MyST (Markdown) -----------------------------------------------------
myst_enable_extensions = [
    "colon_fence",  # ::: fenced blocks
    "deflist",  # definition lists
    "amsmath",  # math environments
    "dollarmath",  # $...$ inline math
    # "linkify" would need `pip install linkify-it-py`; skip for now.
]
myst_heading_anchors = 3  # allow #anchors for h1..h3 in .md

# --- Intersphinx ---------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# --- Source discovery + templating ---------------------------------------
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "api/_autosummary/*.rst.jinja",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"

# --- HTML output ---------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "logo.png"
html_favicon = "favicon.ico"
html_theme_options = {
    "logo_only": False,
    "navigation_depth": 3,
    "collapse_navigation": False,
    "sticky_navigation": True,
}

# Fills the "Edit on GitHub" link + the "View" dropdown at the top of the
# left sidebar in the sphinx_rtd_theme.
html_context = {
    "display_github": True,
    "github_user": "shibaji7",
    "github_repo": "geopulse",
    "github_version": "main",
    "conf_py_path": "/docs/",
    # Extra project-level links surfaced in the footer (rendered by
    # sphinx_rtd_theme when present).
    "project_links": [
        ("GitHub", "https://github.com/shibaji7/geopulse"),
        ("PyPI", "https://pypi.org/project/geopulse/"),
        ("Issues", "https://github.com/shibaji7/geopulse/issues"),
    ],
}

# Silence RTD-theme quirk: it warns on missing custom.css if _static empty.
if not (Path(__file__).parent / "_static").exists():
    (Path(__file__).parent / "_static").mkdir(exist_ok=True)

# --- Sphinx build knobs --------------------------------------------------
# Fail hard on missing cross-references so we catch typos in autosummary.
nitpicky = False  # RTD flips this on if fail_on_warning
# Suppress noisy WARNINGs from Python typing constructs that Sphinx
# can't resolve into intersphinx targets (e.g. `T` TypeVars in Uncertain).
nitpick_ignore_regex = [
    ("py:class", r".*\.T$"),
    ("py:class", r"numpy\..*"),  # numpy classes not in intersphinx yet
    ("py:class", r"h5py\..*"),
]
