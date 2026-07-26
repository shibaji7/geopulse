"""Power-grid conductor network (substations + transmission lines).

Loads a MATPOWER-GMD ``.m`` file via :mod:`geopulse.io.matpower` and exposes
it through the :class:`~geopulse.network.base.ConductorNetwork` ABC so the
LPM solver can treat it identically to a cable or a pipeline.

Geographic convention
---------------------
Node coordinates are stored in **degrees**. Line integrals of the geoelectric
field use a local equirectangular projection about the network's mean
lat/lon; adequate for a few hundred kilometres of extent (the Horton EPRI21
network spans ~500 km across TN/GA/AL).

References
----------
.. [1] Horton, R., Boteler, D., Overbye, T. J., Pirjola, R., & Dugan, R. C.
   (2012). A test case for the calculation of geomagnetically induced
   currents. IEEE Trans. Power Delivery, 27(4), 2368-2373.
.. [2] Boteler, D. H. (2014). Methodology for simulation of geomagnetically
   induced currents in power systems. J. Space Weather Space Clim., 4, A21.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from geopulse.exceptions import DataError
from geopulse.geo import latlon_to_local_xy_wgs84_m
from geopulse.io.matpower import MatpowerGMDCase, read_matpower_gmd
from geopulse.network.base import Branch, ConductorNetwork, Node

__all__ = ["PowerGridNetwork"]

# WGS84 projection is the default: Horton (2012) benchmark data was computed
# on the WGS84 ellipsoid (paper Appendix eqns A3, A5-A7), so using it here
# tightens the benchmark from ~1 % to sub-percent.
_latlon_to_local_xy_m = latlon_to_local_xy_wgs84_m


class PowerGridNetwork(ConductorNetwork):
    """Bulk power-grid network loaded from a MATPOWER-GMD case.

    Parameters
    ----------
    case : MatpowerGMDCase
        Parsed case (see :func:`geopulse.io.matpower.read_matpower_gmd`).

    Notes
    -----
    The DC node id is the 1-based row position within ``mpc.gmd_bus`` (this
    is what ``mpc.gmd_branch.f_bus`` / ``t_bus`` reference — NOT AC bus
    numbers). See :mod:`geopulse.io.matpower` for the full convention.

    Examples
    --------
    >>> from geopulse.io.matpower import read_matpower_gmd  # doctest: +SKIP
    >>> case = read_matpower_gmd("benchmarks/horton2012/epri21.m")  # doctest: +SKIP
    >>> net = PowerGridNetwork(case)  # doctest: +SKIP
    >>> Y = net.assemble_network_admittance()  # doctest: +SKIP
    """

    def __init__(self, case: MatpowerGMDCase) -> None:
        self.case = case
        # Ordered node/branch id lists — used as the canonical row order of
        # the admittance matrix and result vectors.
        self._node_ids: list[str] = [dn["name"] for dn in case.dc_nodes]
        self._branch_ids: list[str] = [db["name"] for db in case.dc_branches]
        # Fast lookup: dc-node row (1-based) -> matrix index (0-based).
        self._row_to_index: dict[int, int] = {
            dn["row"]: idx for idx, dn in enumerate(case.dc_nodes)
        }

        # Project origin: mean of all AC-bus lat/lon that have coordinates.
        lats = [ll[0] for ll in case.bus_latlon.values()]
        lons = [ll[1] for ll in case.bus_latlon.values()]
        if not lats:
            raise DataError("MATPOWER-GMD case contains no bus lat/lon data")
        self._lat0_deg = float(np.mean(lats))
        self._lon0_deg = float(np.mean(lons))

        # MATPOWER-GMD convention: `parent_index` in gmd_bus means AC bus
        # number for `dc_bus*` nodes but **substation number** for `dc_sub*`
        # nodes (Horton EPRI21 has 8 substations numbered 1..8). The file
        # does not carry an explicit substation → lat/lon map, so we
        # infer substation coordinates by walking zero-length transformer
        # branches that connect each `dc_sub*` node to a `dc_bus*` node
        # (which does have the correct lat/lon).
        latlon_by_row: dict[int, tuple[float, float]] = {}
        for dn in case.dc_nodes:
            name = dn.get("name", "")
            if name.startswith("dc_sub"):
                # Deferred: fill from a connected dc_bus* neighbour below.
                continue
            parent = dn["parent_ac_bus"]
            if parent in case.bus_latlon:
                latlon_by_row[dn["row"]] = case.bus_latlon[parent]

        # Second pass: dc_sub* nodes inherit lat/lon from a zero-length-branch
        # neighbour that already has coordinates.
        deferred = [dn for dn in case.dc_nodes if dn.get("name", "").startswith("dc_sub")]
        for dn in deferred:
            row = dn["row"]
            for db in case.dc_branches:
                if db.get("length_km", 0.0) != 0.0:
                    continue
                if db["from_row"] == row and db["to_row"] in latlon_by_row:
                    latlon_by_row[row] = latlon_by_row[db["to_row"]]
                    break
                if db["to_row"] == row and db["from_row"] in latlon_by_row:
                    latlon_by_row[row] = latlon_by_row[db["from_row"]]
                    break
            # dc_sub* nodes with no connected zero-length branch (Sub 1 via
            # GIC BD capacitor, Sub 7 switching-station-only) remain without
            # coordinates. That is fine: they contribute nothing to V_th.

        self._xy_m: dict[int, tuple[float, float]] = {
            row: _latlon_to_local_xy_m(lat, lon, self._lat0_deg, self._lon0_deg)
            for row, (lat, lon) in latlon_by_row.items()
        }
        self._latlon_by_row = latlon_by_row

    # ------------------------------------------------------------------ ABC

    def get_nodes(self) -> Sequence[Node]:
        """Return one :class:`Node` per DC ground node, in matrix-row order.

        Grounding impedance is stored on the node as ``1 / g_gnd`` (Ohms);
        for ``g_gnd == 0`` (ungrounded / out-of-service), ``numpy.inf`` is
        stored.
        """
        nodes: list[Node] = []
        for dn in self.case.dc_nodes:
            lat, lon = self._latlon_by_row.get(dn["row"], (float("nan"), float("nan")))
            g = dn["g_gnd_S"]
            z_earth = 1.0 / g if g > 0.0 else float("inf")
            nodes.append(
                Node(
                    node_id=dn["name"],
                    latitude_deg=lat,
                    longitude_deg=lon,
                    earthing_impedance_Ohm=z_earth,
                )
            )
        return nodes

    def get_branches(self) -> Sequence[Branch]:
        """Return one :class:`Branch` per in-service DC branch."""
        out: list[Branch] = []
        for db in self.case.dc_branches:
            if db["status"] != 1 or db["resistance_Ohm"] <= 0.0:
                continue
            from_name = self.case.dc_nodes[db["from_row"] - 1]["name"]
            to_name = self.case.dc_nodes[db["to_row"] - 1]["name"]
            out.append(
                Branch(
                    branch_id=db["name"],
                    from_node=from_name,
                    to_node=to_name,
                    resistance_Ohm=db["resistance_Ohm"],
                    length_m=db["length_km"] * 1_000.0,
                )
            )
        return out

    def assemble_network_admittance(self) -> np.ndarray:
        """Assemble the DC nodal admittance matrix ``Y_n`` in Siemens.

        Only line conductances contribute — earthing conductances are
        returned separately by :meth:`assemble_earthing_impedance` so the
        solver can build the LPM matrix ``(1 + Y_n · Z_e)`` explicitly.
        """
        n = len(self.case.dc_nodes)
        Y = np.zeros((n, n), dtype=np.float64)
        for db in self.case.dc_branches:
            if db["status"] != 1 or db["resistance_Ohm"] <= 0.0:
                continue
            i = self._row_to_index[db["from_row"]]
            j = self._row_to_index[db["to_row"]]
            g = 1.0 / db["resistance_Ohm"]
            Y[i, i] += g
            Y[j, j] += g
            Y[i, j] -= g
            Y[j, i] -= g
        return Y

    def assemble_earthing_impedance(self) -> np.ndarray:
        """Return the diagonal earthing-impedance matrix ``Z_e`` in Ohms.

        Ungrounded nodes (``g_gnd == 0``) get ``0`` on the diagonal so that
        ``Y_n · Z_e`` does not develop infinities; the LPM solver treats
        those nodes as carrying no ground current.
        """
        n = len(self.case.dc_nodes)
        Z = np.zeros((n, n), dtype=np.float64)
        for dn in self.case.dc_nodes:
            i = self._row_to_index[dn["row"]]
            g = dn["g_gnd_S"]
            # Ungrounded → z=0 keeps (1 + Y·Z) well-conditioned; the solver
            # detects such nodes separately and reports zero ground current.
            Z[i, i] = 1.0 / g if g > 0.0 else 0.0
        return Z

    def compute_thevenin_voltages(
        self,
        ex_Vm: np.ndarray | float,
        ey_Vm: np.ndarray | float,
        lat_deg: np.ndarray | None = None,
        lon_deg: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute the Thévenin voltage per branch via ``V = ∫ E · dl``.

        The E-field is evaluated at the **midpoint** of each branch. For a
        uniform field, pass floats for ``ex_Vm`` / ``ey_Vm``; for a
        spatially varying field, pass arrays evaluated at every branch
        midpoint (order matches :meth:`get_branches`).

        Parameters
        ----------
        ex_Vm, ey_Vm : float or numpy.ndarray
            E-field components in V/m. Scalar (uniform) or array of length
            ``n_active_branches`` (branch-wise).
        lat_deg, lon_deg : numpy.ndarray, optional
            Ignored; retained for API compatibility with subclasses that use
            geo-referenced E-field grids.

        Returns
        -------
        numpy.ndarray
            Thévenin voltage per branch, in Volts. Length equals the number
            of branches returned by :meth:`get_branches`.

        Notes
        -----
        Per-branch: ``V_th = Ex · L_E + Ey · L_N`` where ``L_E``, ``L_N``
        are the east-west and north-south distances between endpoints
        computed with WGS84 curvature radii evaluated at that segment's
        midpoint latitude (matching Horton (2012) Appendix eqns A3, A7
        exactly). Branches with an endpoint lacking geographic coordinates
        (transformer-internal windings) receive ``V_th = 0``.
        """
        del lat_deg, lon_deg  # noqa: F841 - accepted for API symmetry
        from geopulse.geo import meridian_radius_m, prime_vertical_radius_m

        active = [
            db for db in self.case.dc_branches if db["status"] == 1 and db["resistance_Ohm"] > 0.0
        ]
        n_b = len(active)

        ex = np.broadcast_to(np.asarray(ex_Vm, dtype=np.float64), (n_b,)).copy()
        ey = np.broadcast_to(np.asarray(ey_Vm, dtype=np.float64), (n_b,)).copy()

        V_th = np.zeros(n_b, dtype=np.float64)
        for k, db in enumerate(active):
            ll_from = self._latlon_by_row.get(db["from_row"])
            ll_to = self._latlon_by_row.get(db["to_row"])
            if ll_from is None or ll_to is None:
                # Transformer-internal branch: no geographic length.
                continue
            lat_from, lon_from = ll_from
            lat_to, lon_to = ll_to
            phi_mid_deg = 0.5 * (lat_from + lat_to)
            M = meridian_radius_m(phi_mid_deg)
            N = prime_vertical_radius_m(phi_mid_deg)
            dlat_rad = np.radians(lat_to - lat_from)
            dlon_rad = np.radians(lon_to - lon_from)
            L_N_m = M * dlat_rad
            L_E_m = N * float(np.cos(np.radians(phi_mid_deg))) * dlon_rad
            V_th[k] = ex[k] * L_E_m + ey[k] * L_N_m
        return V_th

    @classmethod
    def from_file(cls, path: str) -> "PowerGridNetwork":
        """Load a network from a MATPOWER-GMD ``.m`` file.

        Parameters
        ----------
        path : str
            Path to the ``.m`` file (e.g. ``benchmarks/horton2012/epri21.m``).

        Returns
        -------
        PowerGridNetwork
        """
        return cls(read_matpower_gmd(Path(path)))
