r"""Railway signalling track-circuit network via DSTL.

A length-``L`` section of two-rail track acts, for GIC purposes, as a
single grounded conductor: geoelectric ``E`` drives a longitudinal EMF
in the rail-to-rail loop, which produces a DC current that a
signalling relay senses. The model here treats the two rails as one
equivalent conductor with distributed:

* longitudinal series impedance ``z`` (Ω/m) — steel rails in parallel
  (roughly half a single rail's per-metre resistance),
* transverse shunt admittance ``y`` (S/m) — rail-to-ballast leakage
  along the whole length,
* **discrete grounding points** at signalling relay stations and
  traction substations, each contributing a low-impedance bond to
  earth in parallel with the distributed shunt at that node.

The DSTL discretisation is the same equivalent-π used by
:class:`~geopulse.network.pipeline.PipelineNetwork` (Boteler 1997);
the difference is that :class:`RailwayNetwork` supports arbitrarily
many discrete grounding points along the track, not just endpoints.
Each equivalent-π segment emits one branch; each discretisation node
carries a shunt admittance from the two adjacent half-sections plus a
grounding-bond contribution when the node coincides (within one
segment length) with a listed ``ground_positions_m`` entry.

Solved via :class:`~geopulse.solver.nam.NAMSolver` with the standard
:math:`([Y^n] + [Y^e]) V = J^e` formulation. The ``V`` returned at
each node is the **rail-to-earth potential** (analogous to
"pipe-to-soil" in the pipeline case); rail-to-rail voltage at a
signalling block boundary is a downstream interpretation left to the
caller.

What this module does **not** model
-----------------------------------
* AC traction-power feeders (catenary, sectioning gaps, autotransformer
  paralleling) — that is a full three-phase network problem, out of
  scope for signalling GIC analysis.
* Rectifier-substation return currents.
* Rail temperature effects on series resistance.
* Track-circuit relay saturation curves. The rail-loop DC current is
  reported; downstream code decides whether it trips a relay.

References
----------
.. [1] Boteler, D. H. (2021). *Modeling geomagnetic interference on
   railway signaling track circuits*. Space Weather, 19(1),
   e2020SW002609. (DOI unverified.)
.. [2] Patterson, C. J., Wild, J. A., Boteler, D. H. (2023).
   *Modelling the impact of geomagnetically induced currents on
   electrified railway signalling systems in the United Kingdom*.
   Space Weather. (DOI unverified.)
.. [3] Boteler, D. H. (1997). *Distributed-source transmission line
   theory for electromagnetic induction studies.* Zurich EMC
   Symposium, 401-408.
.. [4] Eroshenko, E. A., Belov, A. V., Boteler, D., Gaidash, S. P.,
   Lobkov, S. L., Pirjola, R., Trichtchenko, L. (2010). *Effects of
   strong geomagnetic storms on Northern railways in Russia.* Adv.
   Space Res., 46(9), 1102-1110.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from geopulse.exceptions import DataError
from geopulse.geo import meridian_radius_m, prime_vertical_radius_m
from geopulse.network.base import Branch, ConductorNetwork, Node

__all__ = [
    "RailwayNetwork",
    "RailwayParameters",
    "rail_to_earth_voltage_analytic",
]


@dataclass(frozen=True)
class RailwayParameters:
    """Physical DSTL parameters of a straight two-rail track section.

    Attributes
    ----------
    length_m : float
        Total track length in metres.
    series_impedance_Ohm_per_m : float
        Longitudinal series impedance of the rail-to-rail loop in Ω/m
        (both rails in parallel, real for DC). Typical continuous
        welded rail: ~5e-5 to 3e-4 Ω/m depending on rail cross-section
        and joint condition.
    shunt_admittance_S_per_m : float
        Rail-to-earth leakage admittance in S/m. Strongly ballast- and
        weather-dependent; typical dry ballast: 1e-4 to 1e-3 S/m.
    start_lat_deg, start_lon_deg : float
        Coordinates of the ``x = 0`` end.
    end_lat_deg, end_lon_deg : float
        Coordinates of the ``x = L`` end.
    n_segments : int, optional
        Number of equivalent-π segments used to discretise the track.
        Default: 40. The network has ``n_segments + 1`` nodes.
    ground_positions_m : tuple[float, ...], optional
        Along-track positions (metres from ``x = 0``) at which the rail
        loop is bonded to earth through a low-impedance ground (relay
        cabinet, traction substation earth, etc.). The bond attaches to
        the discretisation node closest to each listed position, so
        with a fine ``n_segments`` these can be placed to within
        ``length_m / n_segments`` accuracy. Default: ``(0.0,)`` —
        grounded at the ``x = 0`` end only.
    ground_resistance_Ohm : float, optional
        Bond impedance at each grounding point, in Ω. Default: ``0.5``
        (a typical signalling-cabinet earth-rod value).

    Notes
    -----
    * Bonded grounds are added in parallel with the DSTL shunt at the
      chosen node. A single very-low bond dominates the effective
      earthing impedance at that node.
    * Providing an empty ``ground_positions_m`` is allowed and reduces
      the network to the pipeline analogue (grounded only through the
      distributed ballast shunt).
    """

    length_m: float
    series_impedance_Ohm_per_m: float
    shunt_admittance_S_per_m: float
    start_lat_deg: float
    start_lon_deg: float
    end_lat_deg: float
    end_lon_deg: float
    n_segments: int = 40
    ground_positions_m: tuple[float, ...] = (0.0,)
    ground_resistance_Ohm: float = 0.5

    def __post_init__(self) -> None:
        if self.length_m <= 0:
            raise DataError(f"length_m must be positive, got {self.length_m}")
        if self.series_impedance_Ohm_per_m <= 0:
            raise DataError("series_impedance_Ohm_per_m must be positive")
        if self.shunt_admittance_S_per_m <= 0:
            raise DataError("shunt_admittance_S_per_m must be positive")
        if self.n_segments < 1:
            raise DataError(f"n_segments must be >= 1, got {self.n_segments}")
        if self.ground_resistance_Ohm <= 0:
            raise DataError("ground_resistance_Ohm must be positive")
        for gp in self.ground_positions_m:
            if not (0.0 <= gp <= self.length_m):
                raise DataError(f"ground_positions_m entry {gp} out of range [0, {self.length_m}]")


def rail_to_earth_voltage_analytic(
    E_Vm: float,
    length_m: float,
    series_impedance_Ohm_per_m: float,
    shunt_admittance_S_per_m: float,
    x_m: np.ndarray,
) -> np.ndarray:
    r"""Closed-form rail-to-earth voltage in the *insulated-end* limit.

    Assumes **no bonded grounds** (``ground_positions_m = ()``): the
    only earth path is the distributed rail-to-ballast shunt. In that
    limit the DSTL solution matches the pipeline case exactly::

        V(x) = (E / γ) · sinh(γ · (x − L/2)) / cosh(γ · L / 2)

    Provided as a convenience for validation against the numerical
    solve when the caller explicitly disables grounding bonds. With
    bonded grounds a closed form still exists but requires piecewise
    matching at each bond — use the numerical solver in that case.

    Parameters
    ----------
    E_Vm : float
        Uniform longitudinal E-field along the track, V/m.
    length_m : float
        Track length ``L`` in metres.
    series_impedance_Ohm_per_m, shunt_admittance_S_per_m : float
        DSTL parameters ``z`` and ``y``.
    x_m : numpy.ndarray
        Along-track positions at which to evaluate ``V``, in metres.

    Returns
    -------
    numpy.ndarray
        Rail-to-earth potential in Volts, same shape as ``x_m``.

    Examples
    --------
    >>> import numpy as np
    >>> V = rail_to_earth_voltage_analytic(1e-3, 20_000.0, 1e-4, 5e-4,
    ...                                    np.array([0.0, 10_000.0, 20_000.0]))
    >>> V.shape
    (3,)
    """
    gamma = float(np.sqrt(series_impedance_Ohm_per_m * shunt_admittance_S_per_m))
    return np.asarray(
        (E_Vm / gamma) * np.sinh(gamma * (x_m - length_m / 2)) / np.cosh(gamma * length_m / 2),
        dtype=np.float64,
    )


def _interp_latlon(
    lat0: float, lon0: float, lat1: float, lon1: float, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Linear lat/lon interpolation. Adequate up to a few hundred km."""
    ts = np.linspace(0.0, 1.0, n)
    return lat0 + (lat1 - lat0) * ts, lon0 + (lon1 - lon0) * ts


class RailwayNetwork(ConductorNetwork):
    """Two-rail signalling track modelled by equivalent-π DSTL segments.

    Parameters
    ----------
    params : RailwayParameters
        Track physical + discretisation + grounding parameters.

    Attributes
    ----------
    params : RailwayParameters
    node_positions_m : numpy.ndarray
        Along-track ``x`` coordinate for every node, in metres. Shape
        ``(n_segments + 1,)``.
    ground_node_indices : tuple[int, ...]
        Indices into ``node_positions_m`` where a bonded ground was
        attached (deduplicated: two ``ground_positions_m`` mapping to
        the same discretisation node collapse into a single bond).

    Notes
    -----
    Node ids are ``"rail_x0"``, ``"rail_x1"``, .... Branch ids are
    ``"seg_0"``, ``"seg_1"``, ....

    Examples
    --------
    >>> params = RailwayParameters(
    ...     length_m=20_000.0,
    ...     series_impedance_Ohm_per_m=1e-4,
    ...     shunt_admittance_S_per_m=5e-4,
    ...     start_lat_deg=55.9, start_lon_deg=-3.2,
    ...     end_lat_deg=55.86, end_lon_deg=-3.9,
    ...     n_segments=40,
    ...     ground_positions_m=(0.0, 10_000.0, 20_000.0),
    ...     ground_resistance_Ohm=0.5,
    ... )
    >>> net = RailwayNetwork(params)
    >>> len(list(net.get_nodes()))
    41
    >>> len(list(net.get_branches()))
    40
    >>> net.ground_node_indices
    (0, 20, 40)
    """

    def __init__(self, params: RailwayParameters) -> None:
        self.params = params
        n_nodes = params.n_segments + 1
        self.node_positions_m = np.linspace(0.0, params.length_m, n_nodes)

        self._lats, self._lons = _interp_latlon(
            params.start_lat_deg,
            params.start_lon_deg,
            params.end_lat_deg,
            params.end_lon_deg,
            n_nodes,
        )
        self._node_ids = [f"rail_x{i}" for i in range(n_nodes)]
        self._branch_ids = [f"seg_{i}" for i in range(params.n_segments)]

        z = params.series_impedance_Ohm_per_m
        y = params.shunt_admittance_S_per_m
        self._gamma = float(np.sqrt(z * y))
        self._Z0_Ohm = float(np.sqrt(z / y))
        dl = params.length_m / params.n_segments
        self._segment_length_m = dl
        self._Z_ser_Ohm = self._Z0_Ohm * float(np.sinh(self._gamma * dl))
        self._Y_sh_S = float(np.tanh(self._gamma * dl / 2.0)) / self._Z0_Ohm

        idx_seen: dict[int, None] = {}
        for gp in params.ground_positions_m:
            idx = int(np.argmin(np.abs(self.node_positions_m - gp)))
            idx_seen[idx] = None
        self.ground_node_indices: tuple[int, ...] = tuple(sorted(idx_seen))

    # ---- ConductorNetwork ABC -------------------------------------------

    def get_nodes(self) -> Sequence[Node]:
        """Return one Node per discretisation point.

        Earthing impedance combines the distributed shunt (from
        adjacent π-sections) with any bonded-ground contribution at
        that node, in parallel.
        """
        n_nodes = self.params.n_segments + 1
        Y_sh = self._Y_sh_S
        g_bond = 1.0 / self.params.ground_resistance_Ohm
        bonded = set(self.ground_node_indices)
        nodes: list[Node] = []
        for i in range(n_nodes):
            n_touch = 1 if (i == 0 or i == n_nodes - 1) else 2
            g_total = n_touch * Y_sh
            if i in bonded:
                g_total += g_bond
            z_earth = 1.0 / g_total if g_total > 0 else float("inf")
            nodes.append(
                Node(
                    node_id=self._node_ids[i],
                    latitude_deg=float(self._lats[i]),
                    longitude_deg=float(self._lons[i]),
                    earthing_impedance_Ohm=z_earth,
                )
            )
        return nodes

    def get_branches(self) -> Sequence[Branch]:
        """One equivalent-π branch per segment."""
        return [
            Branch(
                branch_id=self._branch_ids[k],
                from_node=self._node_ids[k],
                to_node=self._node_ids[k + 1],
                resistance_Ohm=self._Z_ser_Ohm,
                length_m=self._segment_length_m,
            )
            for k in range(self.params.n_segments)
        ]

    def assemble_network_admittance(self) -> np.ndarray:
        """Assemble ``Y_n`` from the series conductance of each π-section."""
        n = self.params.n_segments + 1
        Y = np.zeros((n, n), dtype=np.float64)
        g = 1.0 / self._Z_ser_Ohm
        for k in range(self.params.n_segments):
            i, j = k, k + 1
            Y[i, i] += g
            Y[j, j] += g
            Y[i, j] -= g
            Y[j, i] -= g
        return Y

    def assemble_earthing_impedance(self) -> np.ndarray:
        """Diagonal ``Z_e`` from shunt admittances + bonded grounds in parallel."""
        n = self.params.n_segments + 1
        Z = np.zeros((n, n), dtype=np.float64)
        Y_sh = self._Y_sh_S
        g_bond = 1.0 / self.params.ground_resistance_Ohm
        bonded = set(self.ground_node_indices)
        for i in range(n):
            n_touch = 1 if (i == 0 or i == n - 1) else 2
            g_total = n_touch * Y_sh
            if i in bonded:
                g_total += g_bond
            Z[i, i] = 1.0 / g_total
        return Z

    def compute_thevenin_voltages(
        self,
        ex_Vm: np.ndarray | float,
        ey_Vm: np.ndarray | float,
        lat_deg: np.ndarray | None = None,
        lon_deg: np.ndarray | None = None,
    ) -> np.ndarray:
        r"""Per-segment Thévenin voltage from the driving geoelectric field.

        Uses per-segment WGS84 midpoint radii for the great-circle
        offset (same convention as :class:`PipelineNetwork` and
        :class:`PowerGridNetwork`)::

            V_th = E_x * ΔEast + E_y * ΔNorth
        """
        del lat_deg, lon_deg  # accepted for ABC symmetry

        n_seg = self.params.n_segments
        ex = np.broadcast_to(np.asarray(ex_Vm, dtype=np.float64), (n_seg,)).copy()
        ey = np.broadcast_to(np.asarray(ey_Vm, dtype=np.float64), (n_seg,)).copy()

        V_th = np.zeros(n_seg, dtype=np.float64)
        for k in range(n_seg):
            lat_a, lon_a = float(self._lats[k]), float(self._lons[k])
            lat_b, lon_b = float(self._lats[k + 1]), float(self._lons[k + 1])
            phi_mid = 0.5 * (lat_a + lat_b)
            M = meridian_radius_m(phi_mid)
            N = prime_vertical_radius_m(phi_mid)
            dlat_rad = np.radians(lat_b - lat_a)
            dlon_rad = np.radians(lon_b - lon_a)
            L_N_m = M * dlat_rad
            L_E_m = N * float(np.cos(np.radians(phi_mid))) * dlon_rad
            V_th[k] = ex[k] * L_E_m + ey[k] * L_N_m
        return V_th

    # ---- YAML loader ----------------------------------------------------

    @classmethod
    def from_file(cls, path: str) -> "RailwayNetwork":
        """Load a :class:`RailwayNetwork` from a YAML configuration file.

        Expected YAML shape::

            length_m: 20000
            series_impedance_Ohm_per_m: 1.0e-4
            shunt_admittance_S_per_m: 5.0e-4
            start_lat_deg: 55.9
            start_lon_deg: -3.2
            end_lat_deg: 55.86
            end_lon_deg: -3.9
            n_segments: 40                              # optional, default 40
            ground_positions_m: [0.0, 10000.0, 20000.0] # optional
            ground_resistance_Ohm: 0.5                  # optional, default 0.5

        Parameters
        ----------
        path : str
            Path to a YAML file matching the shape above.

        Returns
        -------
        RailwayNetwork

        Raises
        ------
        DataError
            If the file is not readable, is not a mapping, or is missing
            a required key.
        """
        import yaml  # type: ignore[import-untyped]

        p = Path(path)
        if not p.is_file():
            raise DataError(f"RailwayNetwork.from_file: {p} does not exist")
        with p.open("r", encoding="utf-8") as fh:
            data: Any = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise DataError(f"RailwayNetwork.from_file: {p} does not contain a YAML mapping")

        required = (
            "length_m",
            "series_impedance_Ohm_per_m",
            "shunt_admittance_S_per_m",
            "start_lat_deg",
            "start_lon_deg",
            "end_lat_deg",
            "end_lon_deg",
        )
        missing = [k for k in required if k not in data]
        if missing:
            raise DataError(f"RailwayNetwork.from_file: {p} missing keys {missing}")

        ground_positions = tuple(float(x) for x in data.get("ground_positions_m", (0.0,)))
        params = RailwayParameters(
            length_m=float(data["length_m"]),
            series_impedance_Ohm_per_m=float(data["series_impedance_Ohm_per_m"]),
            shunt_admittance_S_per_m=float(data["shunt_admittance_S_per_m"]),
            start_lat_deg=float(data["start_lat_deg"]),
            start_lon_deg=float(data["start_lon_deg"]),
            end_lat_deg=float(data["end_lat_deg"]),
            end_lon_deg=float(data["end_lon_deg"]),
            n_segments=int(data.get("n_segments", 40)),
            ground_positions_m=ground_positions,
            ground_resistance_Ohm=float(data.get("ground_resistance_Ohm", 0.5)),
        )
        return cls(params)
