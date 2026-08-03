r"""Named figure presets for journal / poster / presentation output.

Applying a preset sets a handful of matplotlib rcParams (font family,
base size, DPI, save-format defaults) *before* the figure is built, and
then a companion ``save_figure`` call resizes to the venue's exact
column width and writes every format the venue requires. Two calls,
one per phase — rcParams have to be set before Text objects are
constructed, and the resize + save happens after the plot is complete.

Level of "smart"
----------------
Deliberately **Level 1**: mechanical, predictable, no auto-relayout.
The preset picks font size and figure width; the caller stays in
charge of everything else. If a plot has too many legend entries to
fit at 89 mm wide, the preset does NOT rearrange it — it emits a
warning and lets you pick a different preset. Auto-layout heuristics
are a rabbit hole (see notes in the module docstring of
:mod:`geopulse.viz.network_map`); Level 2 waits until we hit a real
need.

What each preset carries
------------------------
The registry stores one :class:`FigurePreset` per venue:

* ``width_mm`` — exact column width the venue prints at. Converted to
  inches at save time (matplotlib is inch-native).
* ``aspect_ratio`` — default height / width ratio if the caller didn't
  pass ``aspect=`` at save time. ``None`` means "use the golden ratio".
* ``dpi`` — minimum DPI the venue's production process accepts.
* ``font_family`` — ``"serif"`` (Times-like for AGU / IEEE) or
  ``"sans-serif"`` (for Nature and presentations).
* ``base_font_pt`` — the ``font.size`` rcParam; other font-size
  rcParams (title, tick, legend) scale from this via matplotlib's
  built-in relative-size mapping.
* ``save_formats`` — extensions the venue accepts / prefers. PDF is
  vector; PNG is a raster fallback; EPS is required by a few venues.

Adding a new venue is a one-line dict entry in :data:`PRESETS`.

Not modelled (deliberate scope limits)
--------------------------------------
* **Auto-layout / auto-legend-splitting**. See discussion above.
* **Colour palette selection**. Different domain (see the ``dataviz``
  skill).
* **LaTeX text rendering**. Not enabled by default because it slows
  save time and requires a LaTeX install; callers who want it can
  ``rcParams["text.usetex"] = True`` themselves after
  :func:`apply_preset`.

References
----------
* AGU house style — https://www.agu.org/publish-with-agu (for JGR
  Space Physics and Space Weather column widths and font conventions).
* Nature guide-for-authors — figure sizing 89 mm single-column /
  183 mm double-column, 300 dpi minimum.
* IEEE journal figure guide — 88 mm / 181 mm column widths.
* Hunter, J. D. (2007). Matplotlib: A 2D graphics environment.
  Computing in Science & Engineering, 9(3), 90-95.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.figure import Figure

from geopulse.exceptions import DataError

__all__ = [
    "PRESETS",
    "FigurePreset",
    "apply_preset",
    "save_figure",
]

# The golden ratio, used as the default figure height when the caller
# does not pass an aspect override. Height = width / GOLDEN_RATIO.
GOLDEN_RATIO = 1.618
# Any text object smaller than this at final rendered size triggers a
# readability warning at save time. 6 pt is at the edge of what a
# reviewer can read on a printed page.
MIN_READABLE_PT = 6.0
# Millimetres per inch (matplotlib is inch-native, journals specify mm).
MM_PER_INCH = 25.4


@dataclass(frozen=True)
class FigurePreset:
    """One venue's figure conventions.

    Attributes
    ----------
    name : str
        Registry key. Also used as the default file-stem suffix.
    width_mm : float
        Exact printed column width in millimetres. Journals publish
        strict values (Nature is 89.0 / 183.0; JGR is 89.0 / 183.0
        following AGU house style; IEEE is 88.0 / 181.0).
    aspect_ratio : float or None
        Default figure height / width. ``None`` means "use the golden
        ratio" (height = width / 1.618). Callers can override at save
        time via ``aspect=``.
    dpi : int
        Minimum DPI the venue accepts. Vector output (PDF, EPS) is
        resolution-independent; DPI still matters for embedded raster
        images and for the fallback PNG.
    font_family : str
        ``"serif"`` (AGU / IEEE house style) or ``"sans-serif"``
        (Nature, presentations, posters).
    base_font_pt : float
        Value applied to ``rcParams["font.size"]``. All other font
        sizes (``axes.titlesize``, ``xtick.labelsize``, ...) inherit
        via matplotlib's ``"medium"`` / ``"small"`` relative-size
        mapping.
    save_formats : tuple[str, ...]
        File extensions to write when :func:`save_figure` is called.
        ``"pdf"`` for vector print, ``"png"`` for web / preview,
        ``"eps"`` for the couple of venues that still require it.
    """

    name: str
    width_mm: float
    aspect_ratio: float | None
    dpi: int
    font_family: str
    base_font_pt: float
    save_formats: tuple[str, ...]


PRESETS: dict[str, FigurePreset] = {
    # AGU (JGR Space Physics, Space Weather, GRL) — Times-like serif,
    # 89 mm single-col, 183 mm double-col, 300 dpi.
    "jgr_1col": FigurePreset(
        name="jgr_1col",
        width_mm=89.0,
        aspect_ratio=None,
        dpi=300,
        font_family="serif",
        base_font_pt=8.0,
        save_formats=("pdf", "png"),
    ),
    "jgr_2col": FigurePreset(
        name="jgr_2col",
        width_mm=183.0,
        aspect_ratio=None,
        dpi=300,
        font_family="serif",
        base_font_pt=8.0,
        save_formats=("pdf", "png"),
    ),
    "sw_1col": FigurePreset(
        name="sw_1col",
        width_mm=89.0,
        aspect_ratio=None,
        dpi=300,
        font_family="serif",
        base_font_pt=8.0,
        save_formats=("pdf", "png"),
    ),
    "sw_2col": FigurePreset(
        name="sw_2col",
        width_mm=183.0,
        aspect_ratio=None,
        dpi=300,
        font_family="serif",
        base_font_pt=8.0,
        save_formats=("pdf", "png"),
    ),
    # Nature — sans-serif, 89 / 183 mm, 300 dpi min, PDF + PNG + EPS.
    "nature_1col": FigurePreset(
        name="nature_1col",
        width_mm=89.0,
        aspect_ratio=None,
        dpi=300,
        font_family="sans-serif",
        base_font_pt=7.0,
        save_formats=("pdf", "png", "eps"),
    ),
    "nature_2col": FigurePreset(
        name="nature_2col",
        width_mm=183.0,
        aspect_ratio=None,
        dpi=300,
        font_family="sans-serif",
        base_font_pt=7.0,
        save_formats=("pdf", "png", "eps"),
    ),
    # IEEE (PES / TPD) — Times-like, 88 / 181 mm, 300 dpi.
    "ieee_1col": FigurePreset(
        name="ieee_1col",
        width_mm=88.0,
        aspect_ratio=None,
        dpi=300,
        font_family="serif",
        base_font_pt=8.0,
        save_formats=("pdf", "png"),
    ),
    "ieee_2col": FigurePreset(
        name="ieee_2col",
        width_mm=181.0,
        aspect_ratio=None,
        dpi=300,
        font_family="serif",
        base_font_pt=8.0,
        save_formats=("pdf", "png"),
    ),
    # Poster (AGU Fall Meeting typical panel) — sans-serif, 254 mm wide
    # (10 in), 4:3 by default, 150 dpi is fine at poster viewing distance.
    "agu_poster": FigurePreset(
        name="agu_poster",
        width_mm=254.0,
        aspect_ratio=0.75,
        dpi=150,
        font_family="sans-serif",
        base_font_pt=20.0,
        save_formats=("png",),
    ),
    # Presentation slide (16:9, sans-serif, big fonts) — 254 mm wide is
    # the width of a wide-aspect content region in a typical talk.
    "presentation": FigurePreset(
        name="presentation",
        width_mm=254.0,
        aspect_ratio=9.0 / 16.0,
        dpi=150,
        font_family="sans-serif",
        base_font_pt=20.0,
        save_formats=("png",),
    ),
    # Preprint (arXiv / ESSOAr) — free width, serif, 10 pt, PDF only.
    "preprint": FigurePreset(
        name="preprint",
        width_mm=170.0,
        aspect_ratio=None,
        dpi=200,
        font_family="serif",
        base_font_pt=10.0,
        save_formats=("pdf",),
    ),
}


def apply_preset(preset: str | FigurePreset) -> None:
    """Set matplotlib rcParams to the chosen preset. Applies globally.

    Call this **before** you build the figure. rcParams take effect at
    Text-object construction time; setting them after the plot is
    built has no effect on the labels / titles that already exist.

    Parameters
    ----------
    preset : str or FigurePreset
        Registered preset name (e.g. ``"jgr_2col"``) or a
        :class:`FigurePreset` instance.

    Raises
    ------
    DataError
        If a string preset name is not in :data:`PRESETS`.

    Notes
    -----
    rcParams set by this function:

    * ``font.family``
    * ``font.size``               → ``base_font_pt``
    * ``axes.titlesize``          → ``"medium"``   (relative to base)
    * ``axes.labelsize``          → ``"medium"``
    * ``xtick.labelsize``         → ``"small"``
    * ``ytick.labelsize``         → ``"small"``
    * ``legend.fontsize``         → ``"small"``
    * ``figure.dpi``              → ``dpi``
    * ``savefig.dpi``             → ``dpi``
    * ``savefig.bbox``            → ``"tight"``

    Everything else is left at matplotlib's defaults.

    Examples
    --------
    >>> from geopulse.viz.presets import apply_preset
    >>> apply_preset("jgr_2col")                        # doctest: +SKIP
    >>> # ... build figure ...
    >>> save_figure(fig, "figure3", preset="jgr_2col")  # doctest: +SKIP
    """
    p = _resolve_preset(preset)
    mpl.rcParams.update(
        {
            "font.family": p.font_family,
            "font.size": p.base_font_pt,
            "axes.titlesize": "medium",
            "axes.labelsize": "medium",
            "xtick.labelsize": "small",
            "ytick.labelsize": "small",
            "legend.fontsize": "small",
            "figure.dpi": p.dpi,
            "savefig.dpi": p.dpi,
            "savefig.bbox": "tight",
        }
    )


def save_figure(
    fig: "Figure",
    path: str | Path,
    preset: str | FigurePreset,
    *,
    aspect: float | None = None,
    check_readability: bool = True,
) -> list[Path]:
    """Resize ``fig`` to the preset's exact column width and save.

    Writes one file per extension in ``preset.save_formats``, all at
    ``preset.dpi``. Returns the list of paths written.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save. Its size is set in place (via
        :meth:`~matplotlib.figure.Figure.set_size_inches`).
    path : str or pathlib.Path
        Output filename. Any extension in the path is stripped and
        replaced with each entry in ``preset.save_formats``.
    preset : str or FigurePreset
        Registered preset name or explicit :class:`FigurePreset`.
    aspect : float, optional
        Height / width ratio to use. Overrides the preset's default.
        ``None`` (default) uses the preset's ``aspect_ratio``, or the
        golden-ratio fallback if the preset didn't set one.
    check_readability : bool, optional
        When ``True`` (default), walk every Text object in the figure
        after resize and warn (via :mod:`warnings`) if any is smaller
        than ``MIN_READABLE_PT`` at the final rendered size. Off if
        you deliberately want tiny inset labels.

    Returns
    -------
    list[pathlib.Path]
        Absolute paths of every file written. Order matches
        ``preset.save_formats``.

    Raises
    ------
    DataError
        If ``preset`` isn't recognised, or ``aspect`` isn't positive.

    Notes
    -----
    * Fonts do NOT auto-scale to survive the resize. The preset picked
      the font size for the target width; if the caller then overrides
      the width with a weird aspect ratio, some labels may spill over.
      The readability warning catches that but does not fix it.
    * When saving multiple formats, matplotlib re-rasterises the
      figure once per file (fine for vector, cheap for PNG).

    Examples
    --------
    >>> save_figure(fig, "fig3", "jgr_2col")             # doctest: +SKIP
    [PosixPath('.../fig3.pdf'), PosixPath('.../fig3.png')]
    """
    p = _resolve_preset(preset)
    ratio = _resolve_aspect(aspect, p)

    width_in = p.width_mm / MM_PER_INCH
    height_in = width_in * ratio
    fig.set_size_inches(width_in, height_in)
    fig.set_dpi(p.dpi)

    if check_readability:
        _warn_if_tiny_text(fig, MIN_READABLE_PT)

    out_dir = Path(path).resolve().parent
    stem = Path(path).stem
    written: list[Path] = []
    for ext in p.save_formats:
        out_path = out_dir / f"{stem}.{ext}"
        fig.savefig(out_path, dpi=p.dpi, bbox_inches="tight")
        written.append(out_path)
    return written


def _resolve_preset(preset: str | FigurePreset) -> FigurePreset:
    """Look up a preset by name or return an already-constructed one."""
    if isinstance(preset, FigurePreset):
        return preset
    if preset not in PRESETS:
        raise DataError(f"unknown preset {preset!r}; known: {sorted(PRESETS.keys())}")
    return PRESETS[preset]


def _resolve_aspect(aspect: float | None, preset: FigurePreset) -> float:
    """Pick a valid aspect ratio: caller-override → preset → golden."""
    if aspect is not None:
        if aspect <= 0.0:
            raise DataError(f"aspect must be positive, got {aspect}")
        return float(aspect)
    if preset.aspect_ratio is not None:
        return float(preset.aspect_ratio)
    return 1.0 / GOLDEN_RATIO


def _warn_if_tiny_text(fig: "Figure", min_pt: float) -> None:
    """Emit a warning if any Text object is smaller than ``min_pt``.

    Walks every Axes and every Text child looking for a rendered font
    size below the threshold. This is a warning, not an error — the
    figure still saves. The purpose is to nudge the caller toward a
    wider preset before submission.
    """
    offenders: list[str] = []
    for ax in fig.axes:
        for text in ax.get_children():
            # Only look at Text; skip Line, Patch, etc.
            get_size = getattr(text, "get_fontsize", None)
            if get_size is None:
                continue
            size = float(get_size())
            if size > 0.0 and size < min_pt:
                get_text = getattr(text, "get_text", lambda: "")
                content = str(get_text())[:40]
                offenders.append(f"  {size:.1f} pt: {content!r}")
    if offenders:
        msg = (
            f"save_figure: {len(offenders)} text object(s) render below "
            f"{min_pt:.1f} pt — consider a wider preset:\n" + "\n".join(offenders)
        )
        warnings.warn(msg, UserWarning, stacklevel=3)
