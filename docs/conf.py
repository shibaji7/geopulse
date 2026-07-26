"""Sphinx configuration for GeoPulse documentation."""

from __future__ import annotations

project = "GeoPulse"
author = "Shibaji Chakraborty, David Boteler"
copyright = "2026, Shibaji Chakraborty"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "logo.png"
html_favicon = "favicon.ico"

napoleon_numpy_docstring = True
napoleon_google_docstring = False
