r"""Pipeline network via Distributed-Source Transmission-Line (DSTL) theory.

A buried pipeline of length ``L`` is modelled as a transmission line with:

* longitudinal series impedance ``z`` (Ω/m) — the pipeline steel + coating
  return path,
* transverse shunt admittance ``y`` (S/m) — leakage to soil through the
  coating,
* a distributed source term equal to the longitudinal geoelectric field
  ``E`` (V/m) acting along the pipe.

Following Boteler & Cookson (1986) and Boteler (1997), the pipeline is
discretised into segments of length ``Δl`` and each segment is replaced by
its **equivalent-π** two-port:

.. code::

              Z_ser
      i o---/\/\/\----o j
        |            |
        Y_sh        Y_sh
        |            |
       gnd          gnd

with

.. math::

    Z_\mathrm{ser}(\Delta\ell) &= Z_0 \sinh(\gamma \Delta\ell),          \\
    Y_\mathrm{sh}(\Delta\ell)  &= \tanh(\gamma \Delta\ell / 2) / Z_0,    \\
    \gamma &= \sqrt{z y},                                                \\
    Z_0    &= \sqrt{z / y}.

In the small-segment limit ``γ·Δℓ → 0`` this collapses to the
distributed-parameter model (``Z_ser → zΔℓ``, ``Y_sh → yΔℓ/2``). Each
segment is emitted as a single :class:`~geopulse.network.base.Branch` with
resistance ``Real(Z_ser)`` between end nodes, and shunt admittances are
summed onto :class:`~geopulse.network.base.Node.earthing_impedance_Ohm`.
:class:`~geopulse.solver.lpm.LPMSolver` then solves the whole thing as a
standard LP network.

Insulated (open-circuit) ends are the default: the pipeline endpoints
carry only their half-segment shunt admittance, so ends are grounded
through the coating alone.

Closed-form validation
----------------------
For a straight pipeline of length ``L``, uniform longitudinal field ``E``,
and insulated ends, the pipe-to-soil voltage is

.. math::

    V(x) = \frac{E}{\gamma}\,\frac{\sinh(\gamma(x - L/2))}{\cosh(\gamma L / 2)},
    \qquad x \in [0, L].

Peak :math:`|V|` sits at the ends and equals :math:`(E/\gamma)\tanh(\gamma L / 2)`.
:func:`pipe_to_soil_voltage_analytic` returns this profile;
:class:`PipelineNetwork` solved via :class:`LPMSolver` reproduces it to
better than 1 % once ``n_segments`` is a few tens.

References
----------
.. [1] Boteler, D. H., & Cookson, M. J. (1986). Telluric currents and their
   effects on pipelines in the Cook Strait region of New Zealand. Materials
   Performance, 25(3), 27-32.
.. [2] Boteler, D. H. (1997). Distributed-source transmission line theory
   for electromagnetic induction studies. In Supplement of the Proceedings
   of the 1997 Zurich EMC Symposium, 401-408.
.. [3] Pulkkinen, A., Pirjola, R., & Viljanen, A. (2001). Determination of
   ground conductivity and system parameters for optimal modeling of
   geomagnetically induced current flow in technological systems. Earth,
   Planets and Space, 53(8), 803-813.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from geopulse.exceptions import DataError, NotImplementedYetError
from geopulse.geo import latlon_to_local_xy_wgs84_m
from geopulse.network.base import Branch, ConductorNetwork, Node

__all__ = [
    "PipelineParameters",
    "PipelineNetwork",
    "pipe_to_soil_voltage_analytic",
]


@dataclass(frozen=True)
class PipelineParameters:
    """Physical DSTL parameters of a straight buried pipeline.

    Attributes
    ----------
    length_m : float
        Total pipeline length in metres.
    series_impedance_Ohm_per_m : float
        Longitudinal series impedance ``z`` in Ω/m. For DC/quasi-DC GIC
        problems this is real (the pipe steel resistance per metre).
    shunt_admittance_S_per_m : float
        Transverse shunt admittance ``y`` in S/m through the coating to
        soil. Real for DC.
    start_lat_deg, start_lon_deg : float
        Coordinates of the ``x = 0`` end.
    end_lat_deg, end_lon_deg : float
        Coordinates of the ``x = L`` end.
    n_segments : int
        Number of equivalent-π segments used to discretise the pipeline.
        The resulting network has ``n_segments + 1`` nodes.

    Notes
    -----
    Discretisation guideline: keep ``γ · Δℓ = γ · L / n_segments`` well
    below unity — the equivalent-π reduces to the true distributed model
    only in the small-segment limit. For typical steel pipelines
    (``z ~ 3 × 10⁻⁴`` Ω/m, ``y ~ 10⁻⁶`` S/m), ``γ ≈ 1.7 × 10⁻⁵`` /m; a
    100-km pipeline needs ``n_segments ≳ 30`` to get sub-percent agreement
    with the closed form.
    """

    length_m: float
    series_impedance_Ohm_per_m: float
    shunt_admittance_S_per_m: float
    start_lat_deg: float
    start_lon_deg: float
    end_lat_deg: float
    end_lon_deg: float
    n_segments: int = 40

    def __post_init__(self) -> None:
        """Validate positivity and segment count."""
        if self.length_m <= 0:
            raise DataError(f"length_m must be positive, got {self.length_m}")
        if self.series_impedance_Ohm_per_m <= 0:
            raise DataError("series_impedance_Ohm_per_m must be positive")
        if self.shunt_admittance_S_per_m <= 0:
            raise DataError("shunt_admittance_S_per_m must be positive")
        if self.n_segments < 1:
            raise DataError(f"n_segments must be >= 1, got {self.n_segments}")


def pipe_to_soil_voltage_analytic(
    E_Vm: float | np.ndarray,
    length_m: float,
    series_impedance_Ohm_per_m: float,
    shunt_admittance_S_per_m: float,
    x_m: np.ndarray,
) -> np.ndarray:
    r"""Closed-form pipe-to-soil voltage for a uniform longitudinal E-field.

    Insulated-end boundary condition (no ground path except through the
    coating shunt). Follows the standard DSTL result [1]_::

        V(x) = (E / γ) · sinh(γ · (x − L/2)) / cosh(γ · L / 2)

    Parameters
    ----------
    E_Vm : float
        Uniform longitudinal E-field along the pipe, in V/m.
    length_m : float
        Pipeline length ``L`` in metres.
    series_impedance_Ohm_per_m, shunt_admittance_S_per_m : float
        DSTL parameters ``z`` and ``y``.
    x_m : numpy.ndarray
        Positions along the pipe at which to evaluate ``V``, in metres.

    Returns
    -------
    numpy.ndarray
        Pipe-to-soil voltage in Volts. Same shape as ``x_m``.

    References
    ----------
    .. [1] Boteler, D. H. (1997). Distributed-source transmission line theory
       for electromagnetic induction studies.

    Examples
    --------
    >>> import numpy as np
    >>> V = pipe_to_soil_voltage_analytic(1e-3, 100_000.0, 3e-4, 1e-6,
    ...                                    np.array([0, 50_000, 100_000]))
    >>> V.shape
    (3,)
    """
    gamma = np.sqrt(series_impedance_Ohm_per_m * shunt_admittance_S_per_m)
    result = (E_Vm / gamma) * np.sinh(gamma * (x_m - length_m / 2)) / np.cosh(gamma * length_m / 2)
    return np.asarray(result, dtype=np.float64)


def _great_circle_midpoints_deg(
    lat0: float, lon0: float, lat1: float, lon1: float, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of lat/lon along a pipeline.

    Straight-line interpolation is adequate for pipeline runs up to a few
    hundred km; for continent-scale pipelines the great-circle correction
    should be added.
    """
    ts = np.linspace(0.0, 1.0, n)
    lats = lat0 + (lat1 - lat0) * ts
    lons = lon0 + (lon1 - lon0) * ts
    return lats, lons


class PipelineNetwork(ConductorNetwork):
    """Straight buried pipeline modelled by equivalent-π DSTL segments.

    Parameters
    ----------
    params : PipelineParameters
        Pipeline physical + discretisation parameters.

    Attributes
    ----------
    params : PipelineParameters
    node_positions_m : numpy.ndarray
        Along-pipe coordinate ``x`` for every node, in metres.
        Shape ``(n_segments + 1,)``.

    Notes
    -----
    Node ids are ``"pipe_x0"``, ``"pipe_x1"``, ..., in order from
    ``x = 0`` to ``x = L``. Branch ids are ``"seg_0"``, ``"seg_1"``, ....

    Examples
    --------
    >>> params = PipelineParameters(
    ...     length_m=100_000.0,
    ...     series_impedance_Ohm_per_m=3e-4,
    ...     shunt_admittance_S_per_m=1e-6,
    ...     start_lat_deg=45.0, start_lon_deg=-75.0,
    ...     end_lat_deg=45.0, end_lon_deg=-73.72,   # ~100 km east
    ...     n_segments=40,
    ... )
    >>> net = PipelineNetwork(params)
    >>> len(list(net.get_nodes()))
    41
    >>> len(list(net.get_branches()))
    40
    """

    def __init__(self, params: PipelineParameters) -> None:
        self.params = params
        n_nodes = params.n_segments + 1
        self.node_positions_m = np.linspace(0.0, params.length_m, n_nodes)

        # Along-pipe geographic coordinates.
        self._lats, self._lons = _great_circle_midpoints_deg(
            params.start_lat_deg,
            params.start_lon_deg,
            params.end_lat_deg,
            params.end_lon_deg,
            n_nodes,
        )
        self._node_ids = [f"pipe_x{i}" for i in range(n_nodes)]
        self._branch_ids = [f"seg_{i}" for i in range(params.n_segments)]

        # DSTL parameters, cached.
        z = params.series_impedance_Ohm_per_m
        y = params.shunt_admittance_S_per_m
        self._gamma = float(np.sqrt(z * y))
        self._Z0_Ohm = float(np.sqrt(z / y))
        dl = params.length_m / params.n_segments
        self._segment_length_m = dl
        # Equivalent-π element values (real, since z and y are real for DC).
        self._Z_ser_Ohm = self._Z0_Ohm * float(np.sinh(self._gamma * dl))
        self._Y_sh_S = float(np.tanh(self._gamma * dl / 2.0)) / self._Z0_Ohm

        # Local equirectangular projection about the pipeline midpoint.
        self._lat0_deg = float((params.start_lat_deg + params.end_lat_deg) / 2.0)
        self._lon0_deg = float((params.start_lon_deg + params.end_lon_deg) / 2.0)

    # ---- Local projection helper ----------------------------------------
    def _xy_m(self, lat_deg: float, lon_deg: float) -> tuple[float, float]:
        return latlon_to_local_xy_wgs84_m(lat_deg, lon_deg, self._lat0_deg, self._lon0_deg)

    # ---- ConductorNetwork ABC -------------------------------------------

    def get_nodes(self) -> Sequence[Node]:
        """Return one Node per discretisation point.

        Interior nodes carry the shunt from two adjacent π-sections
        (``2 · Y_sh``); endpoint nodes carry a single half-section
        (``Y_sh``). ``earthing_impedance_Ohm`` is the reciprocal.
        """
        n_nodes = self.params.n_segments + 1
        Y_sh = self._Y_sh_S
        nodes: list[Node] = []
        for i in range(n_nodes):
            # Endpoint nodes touch one segment; interior nodes touch two.
            n_touch = 1 if (i == 0 or i == n_nodes - 1) else 2
            g_total = n_touch * Y_sh
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
        """Return one Branch per equivalent-π segment."""
        branches: list[Branch] = []
        for k in range(self.params.n_segments):
            branches.append(
                Branch(
                    branch_id=self._branch_ids[k],
                    from_node=self._node_ids[k],
                    to_node=self._node_ids[k + 1],
                    resistance_Ohm=self._Z_ser_Ohm,
                    length_m=self._segment_length_m,
                )
            )
        return branches

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
        """Diagonal ``Z_e`` from the shunt admittances of each π-section."""
        n = self.params.n_segments + 1
        Z = np.zeros((n, n), dtype=np.float64)
        Y_sh = self._Y_sh_S
        for i in range(n):
            n_touch = 1 if (i == 0 or i == n - 1) else 2
            Z[i, i] = 1.0 / (n_touch * Y_sh)
        return Z

    def compute_thevenin_voltages(
        self,
        ex_Vm: np.ndarray | float,
        ey_Vm: np.ndarray | float,
        lat_deg: np.ndarray | None = None,
        lon_deg: np.ndarray | None = None,
    ) -> np.ndarray:
        """Per-segment Thévenin voltage from the geoelectric field.

        ``V_th = E_x · Δx + E_y · Δy`` where ``Δx``, ``Δy`` are the eastward
        and northward offsets of the segment endpoints in the local
        equirectangular projection.
        """
        del lat_deg, lon_deg  # accepted for ABC symmetry
        from geopulse.geo import meridian_radius_m, prime_vertical_radius_m

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

    @classmethod
    def from_file(cls, path: str) -> "PipelineNetwork":
        """Load pipeline parameters from a YAML file (WP4)."""
        raise NotImplementedYetError("PipelineNetwork.from_file", "WP4")
