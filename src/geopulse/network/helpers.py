r"""Utility helpers for common network-topology mutations and field sampling.

These are the small, composable operations that case-study and paper
scripts repeatedly reach for. Every helper is **pure** — it returns a
new matrix (or arrays), never mutates its input — so a single loaded
network can drive many scenarios (baseline, mitigation, outage) in the
same session without cross-contamination.

Three families of helper live here:

1. **Network mutations** — post-hoc edits to a network's assembled
   matrices ``[Y^n]`` (network admittance) and ``[Z^e]`` (earthing
   impedance):

   * :func:`apply_resistive_blocker` — raise the earthing impedance at
     one or more substations to model a neutral-side blocking device
     (or a neutral-grounding-resistor tighter than baseline).
   * :func:`open_line` — remove a transmission-line branch from
     ``[Y^n]`` (equivalent to opening the line at both ends).
   * :func:`add_tie` — add a new conductive path between two nodes.

2. **Spatial-field sampling** — evaluate a user-supplied
   ``field_fn(x_km, y_km) → (Ex, Ey)`` at each branch midpoint,
   returning the per-branch ``(ex_Vm, ey_Vm)`` arrays that
   :meth:`~geopulse.network.powergrid.PowerGridNetwork.compute_thevenin_voltages`
   already accepts.

3. **Coordinate helpers** — a local equirectangular projection about a
   network's mean lat/lon, adequate for network extents up to a few
   hundred kilometres.

Design notes
------------
* All matrix helpers **copy** their input and return the modified
  matrix. This is slower per-call than in-place mutation but makes
  scenario sweeps trivially correct — no forgotten resets between
  iterations.
* Node references are by string ``node_id`` when convenient, by
  integer index when performance matters (both are supported side by
  side).
* No new abstractions are introduced. Helpers work on the existing
  matrix outputs of
  :meth:`~geopulse.network.base.ConductorNetwork.assemble_network_admittance`
  and
  :meth:`~geopulse.network.base.ConductorNetwork.assemble_earthing_impedance`.

Typical scenario-sweep pattern
------------------------------

.. code-block:: python

    net = PowerGridNetwork.from_file("benchmarks/horton2012/epri21.m")
    Y_base = net.assemble_network_admittance()
    Z_base = net.assemble_earthing_impedance()

    scenarios = {
        "baseline":       (Y_base, Z_base),
        "blocker at sub6": (Y_base, apply_resistive_blocker(Z_base, [6], 1e12)),
        "line 3->5 open":  (open_line(Y_base, 3, 5, g_line=1.0/2.5), Z_base),
    }
    for label, (Y, Z) in scenarios.items():
        result = NAMSolver().solve(net, Y, Z, V_th)
        ...

References
----------
.. [1] Boteler, D. H. (2014). Methodology for simulation of
   geomagnetically induced currents in power systems. J. Space Weather
   Space Clim., 4, A21.
.. [2] Bolduc, L., Granger, M., Paré, G., Saintonge, J., Brophy, L.
   (2005). Development of a DC current-blocking device for transformer
   neutrals. IEEE Trans. Power Delivery, 20(1), 163-168.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from geopulse.exceptions import DataError, ShapeMismatchError
from geopulse.geo import meridian_radius_m, prime_vertical_radius_m

if TYPE_CHECKING:  # pragma: no cover
    from geopulse.network.base import ConductorNetwork

__all__ = [
    "add_tie",
    "apply_resistive_blocker",
    "evaluate_field_at_branch_midpoints",
    "open_line",
]


# ---------------------------------------------------------------------------
# Matrix-level mutations (fast, index-based)
# ---------------------------------------------------------------------------


def apply_resistive_blocker(
    earthing_impedance: np.ndarray,
    node_indices: Sequence[int],
    r_blocker_Ohm: float,
) -> np.ndarray:
    r"""Raise the earthing impedance at chosen nodes to model blockers.

    Returns a **copy** of ``earthing_impedance`` with the diagonal
    entries at every index in ``node_indices`` overwritten by
    ``r_blocker_Ohm``. A large ``r_blocker_Ohm`` (e.g. :math:`10^{12}`
    Ω) models an effectively-open neutral (DC-blocking capacitor);
    a moderate value (e.g. 5–10 Ω) models a neutral-grounding
    resistor tighter than baseline.

    Parameters
    ----------
    earthing_impedance : numpy.ndarray
        Diagonal earthing-impedance matrix as returned by
        :meth:`~geopulse.network.base.ConductorNetwork.assemble_earthing_impedance`.
        Shape ``(n_nodes, n_nodes)``, Ohms.
    node_indices : Sequence[int]
        Row / column indices of the substations to attach a blocker
        to. Duplicates are collapsed. Must all be within
        ``[0, n_nodes)``.
    r_blocker_Ohm : float
        Blocker resistance in Ohms. Must be strictly positive; ``0``
        would represent a bolted neutral-to-earth short (the opposite
        of the blocking intent).

    Returns
    -------
    numpy.ndarray
        A copy of ``earthing_impedance`` with the modifications applied.

    Raises
    ------
    ShapeMismatchError
        If ``earthing_impedance`` is not square 2-D.
    DataError
        If ``r_blocker_Ohm`` is not strictly positive, or if any entry
        in ``node_indices`` is out of range.

    Notes
    -----
    Local mitigation raises current elsewhere in the network. See
    Boteler (2014) § 5 for the redistribution physics. This helper
    **does not** compute the redistribution — that is what the NAM
    solve on the modified matrices reports.

    Examples
    --------
    >>> import numpy as np
    >>> Z = np.diag([1.0, 2.0, 3.0])
    >>> Z2 = apply_resistive_blocker(Z, [1], 1e12)
    >>> float(Z2[1, 1])
    1000000000000.0
    >>> float(Z[1, 1])  # original untouched
    2.0
    """
    z = np.asarray(earthing_impedance)
    if z.ndim != 2 or z.shape[0] != z.shape[1]:
        raise ShapeMismatchError(f"earthing_impedance must be square 2-D, got shape {z.shape}")
    if not np.isfinite(r_blocker_Ohm) or r_blocker_Ohm <= 0.0:
        raise DataError(f"r_blocker_Ohm must be positive-finite, got {r_blocker_Ohm!r}")
    n = int(z.shape[0])
    idx = sorted(set(int(i) for i in node_indices))
    for i in idx:
        if not (0 <= i < n):
            raise DataError(f"node index {i} out of range [0, {n})")
    out = z.astype(np.float64, copy=True)
    for i in idx:
        out[i, i] = float(r_blocker_Ohm)
    return out


def open_line(
    network_admittance: np.ndarray,
    from_idx: int,
    to_idx: int,
    line_conductance_S: float,
) -> np.ndarray:
    r"""Remove a line from the network by cancelling its conductance.

    Returns a **copy** of ``network_admittance`` with the four matrix
    entries that a line ``(i, j)`` of conductance ``g`` contributes —
    ``+g`` on diagonals ``[i,i]`` and ``[j,j]``, ``-g`` on
    off-diagonals ``[i,j]`` and ``[j,i]`` — cancelled out. Equivalent
    to opening the line at both terminal breakers.

    Parameters
    ----------
    network_admittance : numpy.ndarray
        ``[Y^n]`` as returned by
        :meth:`~geopulse.network.base.ConductorNetwork.assemble_network_admittance`.
        Shape ``(n_nodes, n_nodes)``, Siemens.
    from_idx, to_idx : int
        Row / column indices of the line's two terminal nodes. Must
        differ and both be within ``[0, n_nodes)``.
    line_conductance_S : float
        The conductance the line originally contributed, in Siemens
        (``1 / R_line_Ohm``). Must be strictly positive.

    Returns
    -------
    numpy.ndarray
        A copy of ``network_admittance`` with the line removed.

    Raises
    ------
    ShapeMismatchError
        If ``network_admittance`` is not square 2-D.
    DataError
        If ``from_idx == to_idx``, either index is out of range, or
        ``line_conductance_S`` is not strictly positive.

    Notes
    -----
    Physically this models both terminal breakers opening simultaneously.
    Modelling only ONE end opening is a more subtle case (a floating
    stub) not handled here — the NAM active-subspace logic covers it if
    a full stub is desired, but constructing that state cleanly is
    caller work.

    Examples
    --------
    >>> import numpy as np
    >>> Y = np.array([[ 2.0, -2.0,  0.0],
    ...               [-2.0,  3.0, -1.0],
    ...               [ 0.0, -1.0,  1.0]])
    >>> Y2 = open_line(Y, 0, 1, line_conductance_S=2.0)
    >>> Y2                                        # line (0,1) gone
    array([[ 0.,  0.,  0.],
           [ 0.,  1., -1.],
           [ 0., -1.,  1.]])
    """
    y = np.asarray(network_admittance)
    if y.ndim != 2 or y.shape[0] != y.shape[1]:
        raise ShapeMismatchError(f"network_admittance must be square 2-D, got shape {y.shape}")
    n = int(y.shape[0])
    i, j = int(from_idx), int(to_idx)
    if i == j:
        raise DataError(f"from_idx and to_idx must differ, both {i}")
    if not (0 <= i < n) or not (0 <= j < n):
        raise DataError(f"indices ({i}, {j}) out of range [0, {n})")
    if not np.isfinite(line_conductance_S) or line_conductance_S <= 0.0:
        raise DataError(f"line_conductance_S must be positive-finite, got {line_conductance_S!r}")
    out = y.astype(np.float64, copy=True)
    g = float(line_conductance_S)
    out[i, i] -= g
    out[j, j] -= g
    out[i, j] += g
    out[j, i] += g
    return out


def add_tie(
    network_admittance: np.ndarray,
    from_idx: int,
    to_idx: int,
    resistance_Ohm: float,
) -> np.ndarray:
    r"""Add a new conductive tie between two nodes.

    The inverse of :func:`open_line`: returns a copy of
    ``network_admittance`` with the four entries a new line of
    resistance ``R = resistance_Ohm`` would add — ``+1/R`` on
    ``[i,i]`` and ``[j,j]``, ``-1/R`` on ``[i,j]`` and ``[j,i]``.

    Parameters
    ----------
    network_admittance : numpy.ndarray
        ``[Y^n]``, shape ``(n_nodes, n_nodes)``, Siemens.
    from_idx, to_idx : int
        Nodes at the two ends of the new tie. Must differ.
    resistance_Ohm : float
        Resistance of the new tie in Ohms. Must be strictly positive.

    Returns
    -------
    numpy.ndarray
        A copy of ``network_admittance`` with the tie added.

    Raises
    ------
    ShapeMismatchError
        If ``network_admittance`` is not square 2-D.
    DataError
        If indices are equal / out of range, or ``resistance_Ohm`` is
        non-positive.

    Examples
    --------
    >>> import numpy as np
    >>> Y = np.zeros((3, 3))
    >>> Y2 = add_tie(Y, 0, 2, resistance_Ohm=0.5)
    >>> Y2                                        # g = 1/0.5 = 2
    array([[ 2.,  0., -2.],
           [ 0.,  0.,  0.],
           [-2.,  0.,  2.]])
    """
    y = np.asarray(network_admittance)
    if y.ndim != 2 or y.shape[0] != y.shape[1]:
        raise ShapeMismatchError(f"network_admittance must be square 2-D, got shape {y.shape}")
    n = int(y.shape[0])
    i, j = int(from_idx), int(to_idx)
    if i == j:
        raise DataError(f"from_idx and to_idx must differ, both {i}")
    if not (0 <= i < n) or not (0 <= j < n):
        raise DataError(f"indices ({i}, {j}) out of range [0, {n})")
    if not np.isfinite(resistance_Ohm) or resistance_Ohm <= 0.0:
        raise DataError(f"resistance_Ohm must be positive-finite, got {resistance_Ohm!r}")
    out = y.astype(np.float64, copy=True)
    g = 1.0 / float(resistance_Ohm)
    out[i, i] += g
    out[j, j] += g
    out[i, j] -= g
    out[j, i] -= g
    return out


# ---------------------------------------------------------------------------
# Spatial-field sampling
# ---------------------------------------------------------------------------


def evaluate_field_at_branch_midpoints(
    network: ConductorNetwork,
    field_fn: Callable[[float, float], tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    r"""Sample a spatially-varying E-field at each branch midpoint.

    Walks the network's branches, computes the geographic midpoint of
    each, projects it into a local equirectangular ``(x_km, y_km)``
    frame centered on the network's mean lat/lon, and calls
    ``field_fn(x_km, y_km)`` to get the local ``(Ex, Ey)`` in V/m.

    The returned per-branch arrays are the exact shape
    :meth:`~geopulse.network.powergrid.PowerGridNetwork.compute_thevenin_voltages`
    accepts:

    .. code-block:: python

        ex, ey = evaluate_field_at_branch_midpoints(net, gradient_field)
        V_th = net.compute_thevenin_voltages(ex_Vm=ex, ey_Vm=ey)

    Parameters
    ----------
    network : ConductorNetwork
        Any network implementing the ABC. Its
        :meth:`~geopulse.network.base.ConductorNetwork.get_nodes` and
        :meth:`~geopulse.network.base.ConductorNetwork.get_branches`
        are used to build the branch endpoint coordinates.
    field_fn : Callable[[float, float], tuple[float, float]]
        A function of local Cartesian offsets ``(x_km, y_km)`` returning
        ``(Ex, Ey)`` in V/m. Called once per branch at the midpoint.
        Must be pure — no state, no side effects.

    Returns
    -------
    ex_Vm, ey_Vm : numpy.ndarray
        Per-branch E-field components in V/m. Shape ``(n_branches,)``
        each. Ordered to match
        :meth:`~geopulse.network.base.ConductorNetwork.get_branches`.

    Raises
    ------
    DataError
        If the network has zero branches or ``field_fn`` is not callable.

    Notes
    -----
    * Uses a **local equirectangular projection**: the mean of all node
      latitudes / longitudes becomes ``(0, 0)`` in ``(x_km, y_km)``,
      and offsets scale by the WGS84 meridian / prime-vertical radii of
      curvature at the mean latitude. Accurate to well under 1 % over
      network extents up to ~500 km.
    * ``field_fn`` is called with ``x`` positive east, ``y`` positive
      north — the same convention geoscience literature uses. If you
      pass a function that returns ``(Ex, Ey)`` in mV/km, remember the
      returned arrays will need a `× 1e-3` scale before handing to
      ``compute_thevenin_voltages`` (which expects V/m).

    Examples
    --------
    A uniform 1 V/km eastward field:

    >>> from geopulse.network.helpers import evaluate_field_at_branch_midpoints
    >>> net = ...                                             # doctest: +SKIP
    >>> uniform = lambda x_km, y_km: (1e-3, 0.0)              # 1 V/km east
    >>> ex, ey = evaluate_field_at_branch_midpoints(net, uniform)  # doctest: +SKIP
    """
    if not callable(field_fn):
        raise DataError(f"field_fn must be callable, got {type(field_fn).__name__}")

    nodes = list(network.get_nodes())
    branches = list(network.get_branches())
    if not branches:
        raise DataError("network has zero branches; nothing to sample")

    node_by_id = {n.node_id: n for n in nodes}

    # Local projection origin = mean of node coordinates, computed only
    # over nodes with finite lat/lon so a single NaN-coord node elsewhere
    # in the network cannot poison the origin (and therefore every
    # per-branch sample). Branches touching a NaN-coord endpoint are
    # zero-length degenerates (co-located with parent bus) and get
    # (0, 0) — no induced voltage is the right physics for a zero-length
    # branch.
    lats = np.array([n.latitude_deg for n in nodes], dtype=np.float64)
    lons = np.array([n.longitude_deg for n in nodes], dtype=np.float64)
    finite = np.isfinite(lats) & np.isfinite(lons)
    if not np.any(finite):
        raise DataError(
            "network has no node with finite (lat, lon); cannot build projection origin"
        )
    lat0 = float(np.mean(lats[finite]))
    lon0 = float(np.mean(lons[finite]))
    m_per_deg_lat = meridian_radius_m(lat0) * np.pi / 180.0
    m_per_deg_lon = prime_vertical_radius_m(lat0) * float(np.cos(np.radians(lat0))) * np.pi / 180.0

    ex = np.zeros(len(branches), dtype=np.float64)
    ey = np.zeros(len(branches), dtype=np.float64)

    for k, br in enumerate(branches):
        a = node_by_id[br.from_node]
        b = node_by_id[br.to_node]
        lat_mid = 0.5 * (a.latitude_deg + b.latitude_deg)
        lon_mid = 0.5 * (a.longitude_deg + b.longitude_deg)
        if not (np.isfinite(lat_mid) and np.isfinite(lon_mid)):
            continue  # zero-length degenerate branch → leave (0, 0)
        x_km = (lon_mid - lon0) * m_per_deg_lon / 1000.0
        y_km = (lat_mid - lat0) * m_per_deg_lat / 1000.0
        val = field_fn(float(x_km), float(y_km))
        ex[k] = float(val[0])
        ey[k] = float(val[1])

    return ex, ey
