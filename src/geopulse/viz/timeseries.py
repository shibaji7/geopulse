"""Stacked time-series plots for physical signals (B, E, V, I).

Every GeoPulse case study eventually wants the same picture: several
physically-related quantities sharing a time axis, one per subplot row.
Reproducing the boilerplate for every script (subplots, shared x-axis,
per-panel legend, unit scaling, UTC-aware time axis, PNG save) is both
verbose and error-prone.

``plot_timeseries`` collapses that boilerplate into a single call while
still returning the ``Figure`` so the caller can post-tweak anything.

The panel structure is intentionally value-in-SI-units + explicit scaling:
callers hand over tesla / volt / ampere arrays and say "display as nT"
via ``unit_scale=1e9``. That keeps the plotting layer neutral about
units, which matters because the same array can be plotted in different
units in different papers.

References
----------
* Hunter, J. D. (2007). Matplotlib: A 2D graphics environment.
  *Computing in Science & Engineering*, 9(3), 90-95.
  https://doi.org/10.1109/MCSE.2007.55
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from geopulse.exceptions import ShapeMismatchError

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["TimeSeriesPanel", "plot_timeseries"]

_TimeUnit = Literal["s", "min", "h"]
_TIME_UNIT_TO_SECONDS = {"s": 1.0, "min": 60.0, "h": 3600.0}


@dataclass(frozen=True)
class TimeSeriesPanel:
    """One row in a stacked time-series plot.

    Parameters
    ----------
    series : dict[str, numpy.ndarray]
        Mapping from legend label to y-values in SI units. Every array
        must have the same length as the ``time_s`` argument passed to
        :func:`plot_timeseries`. Passing an empty mapping raises
        :class:`ShapeMismatchError`.
    ylabel : str
        Y-axis label, including the display unit (e.g. ``"B (nT)"``).
    unit_scale : float, optional
        Multiplier applied to every series before plotting. Use it to
        convert SI values to whatever unit ``ylabel`` says (Tesla → nT is
        ``1e9``; Volt/metre → mV/m is ``1e3``). Default: ``1.0``.
    ylim : tuple[float, float] or None, optional
        ``(ymin, ymax)`` for the y-axis, in the *displayed* units (i.e.
        after ``unit_scale``). Default: ``None`` (auto-scale).
    show_legend : bool or None, optional
        Whether to draw a legend on this panel. ``None`` means "auto":
        draw when the panel has more than one series. Default: ``None``.
    """

    series: dict[str, np.ndarray]
    ylabel: str
    unit_scale: float = 1.0
    ylim: tuple[float, float] | None = None
    show_legend: bool | None = None


def plot_timeseries(
    time_s: np.ndarray,
    panels: Sequence[TimeSeriesPanel],
    *,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    time_unit: _TimeUnit = "h",
    time_origin_utc: float | None = None,
    savepath: str | Path | None = None,
    axes: Sequence["Axes"] | None = None,
    dpi: int = 120,
) -> "Figure":
    """Stacked time-series plot with a shared x-axis.

    Parameters
    ----------
    time_s : numpy.ndarray
        Time values in seconds. If ``time_origin_utc`` is given these are
        interpreted as UNIX epoch seconds (UTC); otherwise they are
        elapsed seconds since an arbitrary origin.
    panels : Sequence[TimeSeriesPanel]
        One panel per subplot row, in top-to-bottom order.
    title : str, optional
        Suptitle. Default: no suptitle.
    figsize : (float, float), optional
        Figure size in inches. Default: ``(10, 2 * len(panels) + 1)`` —
        roughly two inches per panel plus header room.
    time_unit : {"s", "min", "h"}, optional
        Unit for the numeric time axis. Ignored when ``time_origin_utc``
        is given (a datetime axis is used instead). Default: ``"h"``.
    time_origin_utc : float, optional
        UNIX epoch seconds. If given, ``time_s`` is treated as absolute
        UTC epoch seconds and the x-axis is formatted as datetimes; the
        value labels the axis. Default: ``None`` (numeric axis).
    savepath : str or pathlib.Path, optional
        If given, save the figure with ``dpi=dpi`` and
        ``bbox_inches="tight"``. Default: don't save.
    axes : Sequence[matplotlib.axes.Axes], optional
        Existing axes to draw into (one per panel). Useful when composing
        a larger figure. Default: create a new figure with
        ``subplots(len(panels), 1, sharex=True)``.
    dpi : int, optional
        DPI for ``savepath`` output. Default: ``120``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the panels. When ``axes`` was passed, this
        is the parent figure of the first axis.

    Raises
    ------
    ShapeMismatchError
        If ``panels`` is empty, if any series has a length different from
        ``time_s``, if any panel has an empty ``series`` mapping, or if
        ``axes`` was passed with a length that does not match ``panels``.

    Examples
    --------
    >>> import numpy as np
    >>> from geopulse.viz.timeseries import TimeSeriesPanel, plot_timeseries
    >>> t = np.linspace(0, 3600, 361)
    >>> bx = 10e-9 * np.sin(2 * np.pi * t / 900)          # 10 nT sine
    >>> fig = plot_timeseries(
    ...     t,
    ...     [TimeSeriesPanel({"Bx": bx}, ylabel="B (nT)", unit_scale=1e9)],
    ...     time_unit="min",
    ...     title="Toy signal",
    ... )
    >>> fig.axes[0].get_ylabel()
    'B (nT)'
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    t = np.asarray(time_s, dtype=np.float64)
    n = t.size
    n_panels = len(panels)
    if n_panels == 0:
        raise ShapeMismatchError("plot_timeseries requires at least one panel")

    for k, p in enumerate(panels):
        if not p.series:
            raise ShapeMismatchError(f"panel {k} ({p.ylabel!r}) has no series")
        for label, y in p.series.items():
            y_arr = np.asarray(y)
            if y_arr.shape != (n,):
                raise ShapeMismatchError(
                    f"panel {k} series {label!r} has shape {y_arr.shape}, "
                    f"expected ({n},) to match time_s"
                )

    if axes is not None:
        axes_list = list(axes)
        if len(axes_list) != n_panels:
            raise ShapeMismatchError(
                f"axes length {len(axes_list)} does not match panels ({n_panels})"
            )
        fig = axes_list[0].figure
    else:
        fig_h = figsize if figsize is not None else (10.0, 2.0 * n_panels + 1.0)
        fig, ax_arr = plt.subplots(n_panels, 1, sharex=True, figsize=fig_h)
        axes_list = list(np.atleast_1d(ax_arr))

    if time_origin_utc is not None:
        x = np.array(
            [datetime.fromtimestamp(float(ti), tz=timezone.utc) for ti in t],
            dtype="O",
        )
        origin_str = datetime.fromtimestamp(time_origin_utc, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
        x_label = f"Time (UTC, origin {origin_str})"
    else:
        x = t / _TIME_UNIT_TO_SECONDS[time_unit]
        x_label = f"Time ({time_unit})"

    for ax, panel in zip(axes_list, panels, strict=True):
        for label, y in panel.series.items():
            ax.plot(x, np.asarray(y) * panel.unit_scale, label=label, lw=1.0)
        ax.set_ylabel(panel.ylabel)
        if panel.ylim is not None:
            ax.set_ylim(*panel.ylim)
        ax.grid(alpha=0.3)
        show_leg = panel.show_legend
        if show_leg is None:
            show_leg = len(panel.series) > 1
        if show_leg:
            ax.legend(loc="upper left", framealpha=0.9, fontsize=8)

    axes_list[-1].set_xlabel(x_label)
    if time_origin_utc is not None:
        axes_list[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate()

    if title:
        fig.suptitle(title, y=1.0)

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(str(savepath), dpi=dpi, bbox_inches="tight")

    return fig
