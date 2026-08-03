"""Network topology on a geographic canvas.

Deliberately **matplotlib-only** — no cartopy, no basemap, no
projection library. For network extents up to a few hundred
kilometres a plain equirectangular ``(lon, lat)`` scatter is
visually indistinguishable from a projected map, and it avoids
pulling ``cartopy`` (which lives in the ``[viz]`` optional extra
and pulls a heavy transitive dep chain via GEOS / PROJ).

If a caller needs a projected basemap later, they can build one
themselves with cartopy and hand the resulting Axes back in via
the ``ax`` keyword; :func:`plot_network_map` will draw into it.

What this module renders
------------------------
* One marker per substation / discretisation node, positioned at
  its ``(longitude_deg, latitude_deg)``.
* Optionally, node colour and / or size mapped to a per-node value
  (e.g. GIC magnitude, RMS voltage, temperature).
* Every branch drawn as a straight line segment between its
  endpoints. Line width is uniform; colouring branches by current
  is intentionally out of scope (bar charts are clearer for that).

Design principles
-----------------
* **Draft quality by default**, publication quality by explicit
  choice. If the caller wants an 89-mm-wide JGR figure with 300 DPI
  and Times-family fonts, they call
  :func:`geopulse.viz.presets.apply_preset` before, and
  :func:`geopulse.viz.presets.save_figure` after, this function.
  See :mod:`geopulse.viz.presets`.
* **Ax injection** is first-class. Callers who compose several
  panels into one figure pass their pre-built ``Axes`` object; this
  function draws into it and returns the parent ``Figure``.
* **Log-scale colouring** is opt-in but frequent enough to be a
  first-class kwarg. GIC magnitudes across a network often span
  more than one decade; linear colouring makes the smaller nodes
  invisible.

References
----------
.. [1] Boteler, D. H. (2014). Methodology for simulation of
   geomagnetically induced currents in power systems. J. Space
   Weather Space Clim., 4, A21. (Network representation.)
.. [2] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment.
   Computing in Science & Engineering, 9(3), 90-95.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from geopulse.exceptions import DataError, ShapeMismatchError

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from geopulse.network.base import ConductorNetwork

__all__ = ["plot_network_map"]


def plot_network_map(
    network: "ConductorNetwork",
    node_values: np.ndarray | dict | None = None,
    *,
    ax: Optional["Axes"] = None,
    cmap: str = "viridis",
    title: Optional[str] = None,
    log_scale: bool = False,
    node_size_scale: float = 200.0,
    show_labels: bool = False,
    branch_color: str = "0.5",
    branch_lw: float = 0.8,
    savepath: str | Path | None = None,
    dpi: int = 120,
) -> "Figure":
    r"""Draw a network's substations and transmission lines on a lon/lat canvas.

    Parameters
    ----------
    network : ConductorNetwork
        Any concrete network exposing
        :meth:`~geopulse.network.base.ConductorNetwork.get_nodes` and
        :meth:`~geopulse.network.base.ConductorNetwork.get_branches`.
    node_values : numpy.ndarray, dict, or None, optional
        Per-node scalar (typically GIC magnitude in A) that colours and
        sizes the markers. Accepted forms:

        * ``numpy.ndarray`` of length ``n_nodes`` — ordered to match
          :meth:`~geopulse.network.base.ConductorNetwork.get_nodes`.
        * ``dict[node_id, value]`` — sparse mapping; missing nodes plot
          at the colour-map minimum with a small marker.
        * ``None`` (default) — uniform colour, uniform size.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into. If ``None``, a new figure with
        ``figsize=(9, 6)`` is created. Ax injection is the mechanism
        for composing this map into a larger figure.
    cmap : str, optional
        Matplotlib colormap name. Default: ``"viridis"``.
    title : str, optional
        Axes title. Default: no title.
    log_scale : bool, optional
        When ``True``, both the colour and marker-size mappings use
        ``log10(|value|)``. Useful when node values span > 1 decade.
        Requires all ``node_values`` to be strictly non-zero;
        zero-value nodes plot at the log-min. Default: ``False``.
    node_size_scale : float, optional
        Base marker area (in matplotlib scatter units, ``s=``). When
        ``node_values`` is supplied, this scales linearly (or
        logarithmically) with the normalised value. Default: ``200``.
    show_labels : bool, optional
        If ``True``, print each node's ``node_id`` next to its marker.
        Off by default — labels clutter maps with many substations.
    branch_color : str, optional
        Matplotlib colour for transmission-line segments. Default:
        light grey (``"0.5"``).
    branch_lw : float, optional
        Line-width for transmission-line segments. Default: ``0.8``.
    savepath : str or pathlib.Path, optional
        If given, save the figure to this path with ``bbox_inches="tight"``.
        Default: don't save.
    dpi : int, optional
        DPI for ``savepath`` output. Default: ``120``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the map (parent of ``ax`` when ``ax``
        was injected).

    Raises
    ------
    ShapeMismatchError
        If ``node_values`` is an array whose length differs from the
        number of nodes.
    DataError
        If the network has zero nodes, or ``log_scale=True`` with any
        negative ``node_values``.

    Notes
    -----
    * Coordinates plot as raw ``(lon, lat)`` with equal aspect ratio.
      For extents up to a few hundred km this is visually
      indistinguishable from a proper projection.
    * When a colour-bar is added, it uses the parent ``fig``'s
      colour-bar API. Callers with tight subplot layouts should call
      ``fig.tight_layout()`` after this function returns.

    Examples
    --------
    Minimum call — topology only, no per-node data:

    >>> from geopulse.viz.network_map import plot_network_map           # doctest: +SKIP
    >>> fig = plot_network_map(net)                                      # doctest: +SKIP

    Colour and size nodes by their GIC magnitude:

    >>> gic_A = np.abs(result.node_voltages_V / Ze_diag)                 # doctest: +SKIP
    >>> fig = plot_network_map(net, gic_A, title="Baseline uniform 1 V/km") # doctest: +SKIP

    Compose into a 2-panel figure:

    >>> fig, (ax_left, ax_right) = plt.subplots(1, 2)                    # doctest: +SKIP
    >>> plot_network_map(net, gic_baseline, ax=ax_left)                  # doctest: +SKIP
    >>> plot_network_map(net, gic_mitigated, ax=ax_right)                # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    nodes = list(network.get_nodes())
    branches = list(network.get_branches())
    if not nodes:
        raise DataError("network has zero nodes; nothing to plot")

    n_nodes = len(nodes)
    node_ids = [n.node_id for n in nodes]
    lons = np.array([n.longitude_deg for n in nodes], dtype=np.float64)
    lats = np.array([n.latitude_deg for n in nodes], dtype=np.float64)

    values = _normalise_node_values(node_values, node_ids, n_nodes)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig = ax.figure

    _draw_branches(ax, nodes, branches, branch_color, branch_lw)

    scatter_kwargs = _build_scatter_kwargs(values, cmap, log_scale, node_size_scale)
    sc = ax.scatter(lons, lats, **scatter_kwargs)

    if values is not None:
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("log10(node value)" if log_scale else "node value")

    if show_labels:
        for lon, lat, nid in zip(lons, lats, node_ids, strict=True):
            ax.annotate(nid, (lon, lat), textcoords="offset points", xytext=(4, 4), fontsize=7)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)

    if savepath is not None:
        fig.savefig(str(savepath), dpi=dpi, bbox_inches="tight")

    return fig


def _normalise_node_values(
    node_values: np.ndarray | dict | None,
    node_ids: list[str],
    n_nodes: int,
) -> Optional[np.ndarray]:
    """Convert dict / array / None into a length-``n_nodes`` array (or None).

    Missing dict keys default to 0.0.
    """
    if node_values is None:
        return None
    if isinstance(node_values, dict):
        out = np.zeros(n_nodes, dtype=np.float64)
        for i, nid in enumerate(node_ids):
            if nid in node_values:
                out[i] = float(node_values[nid])
        return out
    arr = np.asarray(node_values, dtype=np.float64)
    if arr.shape != (n_nodes,):
        raise ShapeMismatchError(
            f"node_values shape {arr.shape} does not match n_nodes ({n_nodes},)"
        )
    return arr


def _draw_branches(
    ax: "Axes",
    nodes: list,
    branches: list,
    color: str,
    lw: float,
) -> None:
    """Plot each branch as a straight line segment between its endpoints."""
    node_by_id = {n.node_id: n for n in nodes}
    for br in branches:
        a = node_by_id.get(br.from_node)
        b = node_by_id.get(br.to_node)
        if a is None or b is None:
            continue
        ax.plot(
            [a.longitude_deg, b.longitude_deg],
            [a.latitude_deg, b.latitude_deg],
            color=color,
            lw=lw,
            zorder=1,  # draw under markers
        )


def _build_scatter_kwargs(
    values: Optional[np.ndarray],
    cmap: str,
    log_scale: bool,
    node_size_scale: float,
) -> dict:
    """Compute matplotlib scatter kwargs for the given colour / size mapping."""
    if values is None:
        return {
            "c": "tab:blue",
            "s": node_size_scale,
            "edgecolor": "black",
            "linewidth": 0.4,
            "zorder": 2,
        }
    if log_scale:
        if np.any(values < 0.0):
            raise DataError("log_scale=True requires non-negative node_values")
        # log10 with a tiny floor to avoid -inf at exact zeros.
        floor = max(np.abs(values).max() * 1e-6, 1e-12)
        colours = np.log10(np.maximum(np.abs(values), floor))
    else:
        colours = values

    v_abs = np.abs(values)
    v_max = float(v_abs.max()) or 1.0
    sizes = node_size_scale * (0.2 + 0.8 * v_abs / v_max)
    return {
        "c": colours,
        "s": sizes,
        "cmap": cmap,
        "edgecolor": "black",
        "linewidth": 0.4,
        "zorder": 2,
    }
