"""Tests for :class:`geopulse.network.pipeline.PipelineNetwork`.

Critical acceptance test: the equivalent-π discretised network solved
through :class:`NAMSolver` must reproduce the closed-form DSTL pipe-to-soil
voltage :math:`V(x) = (E/\\gamma)\\sinh(\\gamma(x - L/2))/\\cosh(\\gamma L/2)`
for a uniform longitudinal E-field. Spec Section 22 Phase 3 sets 5 %; a
40-segment discretisation clears that with margin.
"""

from __future__ import annotations

import numpy as np
import pytest

from geopulse.exceptions import DataError
from geopulse.network.pipeline import (
    PipelineNetwork,
    PipelineParameters,
    pipe_to_soil_voltage_analytic,
)
from geopulse.solver.nam import NAMSolver


def _make_pipeline(length_m=100_000.0, z=3e-4, y=1e-6, n_seg=40):
    """Straight eastward pipeline of exactly ``length_m`` (WGS84) at 45 N."""
    from geopulse.geo import prime_vertical_radius_m

    # End coordinate chosen so that WGS84 east-west distance = length_m
    # exactly (matches PowerGridNetwork's per-segment WGS84 line integral).
    lat0 = 45.0
    N_m = prime_vertical_radius_m(lat0)
    dlon_deg = np.degrees(length_m / (N_m * np.cos(np.radians(lat0))))
    return PipelineParameters(
        length_m=length_m,
        series_impedance_Ohm_per_m=z,
        shunt_admittance_S_per_m=y,
        start_lat_deg=lat0,
        start_lon_deg=-75.0,
        end_lat_deg=lat0,
        end_lon_deg=-75.0 + dlon_deg,
        n_segments=n_seg,
    )


def test_pipeline_nodes_and_branches_shape():
    net = PipelineNetwork(_make_pipeline())
    nodes = list(net.get_nodes())
    branches = list(net.get_branches())
    assert len(nodes) == 41
    assert len(branches) == 40
    assert nodes[0].node_id == "pipe_x0"
    assert nodes[-1].node_id == "pipe_x40"


def test_pi_section_matches_analytic_uniform_field():
    """40-segment π-network + LPM must match analytic V(x) within 1 %."""
    params = _make_pipeline(length_m=100_000.0, z=3e-4, y=1e-6, n_seg=40)
    net = PipelineNetwork(params)
    Y = net.assemble_network_admittance()
    Z = net.assemble_earthing_impedance()

    # Uniform 1 V/km eastward along a purely eastward pipe.
    V_th = net.compute_thevenin_voltages(ex_Vm=1e-3, ey_Vm=0.0)
    result = NAMSolver().solve(net, Y, Z, V_th)

    V_analytic = pipe_to_soil_voltage_analytic(
        E_Vm=1e-3,
        length_m=params.length_m,
        series_impedance_Ohm_per_m=params.series_impedance_Ohm_per_m,
        shunt_admittance_S_per_m=params.shunt_admittance_S_per_m,
        x_m=net.node_positions_m,
    )

    # Signs: V_th orientation puts the +x-end at the "high" potential end.
    # Compare on absolute-max-scaled residual to be sign-convention-agnostic.
    scale = float(np.max(np.abs(V_analytic)))
    err = np.abs(np.abs(result.node_voltages_V) - np.abs(V_analytic)) / scale
    assert err.max() < 0.01, f"max rel err {err.max():.4f} > 1%"


def test_convergence_with_n_segments():
    """Peak-voltage error must shrink monotonically as n_segments grows."""
    params_kwargs = dict(length_m=100_000.0, z=3e-4, y=1e-6)
    V_ref = pipe_to_soil_voltage_analytic(
        E_Vm=1e-3,
        length_m=params_kwargs["length_m"],
        series_impedance_Ohm_per_m=params_kwargs["z"],
        shunt_admittance_S_per_m=params_kwargs["y"],
        x_m=np.array([params_kwargs["length_m"]]),
    )[0]

    errs = []
    for n_seg in (10, 20, 40, 80):
        net = PipelineNetwork(_make_pipeline(n_seg=n_seg, **params_kwargs))
        Y = net.assemble_network_admittance()
        Z = net.assemble_earthing_impedance()
        V_th = net.compute_thevenin_voltages(ex_Vm=1e-3, ey_Vm=0.0)
        result = NAMSolver().solve(net, Y, Z, V_th)
        # Endpoint voltage magnitude
        errs.append(abs(abs(result.node_voltages_V[-1]) - abs(V_ref)) / abs(V_ref))
    # Should decrease as we refine
    assert errs[0] > errs[-1]
    # 80-segment must be well within 1%
    assert errs[-1] < 0.01


def test_analytic_helper_symmetry_and_peak():
    """Closed-form V(x) is antisymmetric about x = L/2 with peak (E/γ)·tanh(γL/2)."""
    L, z, y, E = 100_000.0, 3e-4, 1e-6, 1e-3
    gamma = np.sqrt(z * y)
    x = np.linspace(0.0, L, 101)
    V = pipe_to_soil_voltage_analytic(E, L, z, y, x)

    # V(0) = -V(L)
    np.testing.assert_allclose(V[0], -V[-1], rtol=1e-12)
    # V(L/2) = 0
    np.testing.assert_allclose(V[50], 0.0, atol=1e-12)
    # Peak = (E/γ) · tanh(γL/2)
    expected_peak = (E / gamma) * np.tanh(gamma * L / 2)
    np.testing.assert_allclose(np.max(np.abs(V)), expected_peak, rtol=1e-12)


def test_zero_field_gives_zero_voltage():
    net = PipelineNetwork(_make_pipeline())
    Y = net.assemble_network_admittance()
    Z = net.assemble_earthing_impedance()
    V_th = net.compute_thevenin_voltages(ex_Vm=0.0, ey_Vm=0.0)
    result = NAMSolver().solve(net, Y, Z, V_th)
    np.testing.assert_allclose(result.node_voltages_V, 0.0, atol=1e-9)


def test_invalid_parameters_raise():
    with pytest.raises(DataError, match="length_m"):
        PipelineParameters(
            length_m=-1.0,
            series_impedance_Ohm_per_m=1e-4,
            shunt_admittance_S_per_m=1e-6,
            start_lat_deg=0.0,
            start_lon_deg=0.0,
            end_lat_deg=0.0,
            end_lon_deg=1.0,
        )
    with pytest.raises(DataError, match="n_segments"):
        PipelineParameters(
            length_m=1000.0,
            series_impedance_Ohm_per_m=1e-4,
            shunt_admittance_S_per_m=1e-6,
            start_lat_deg=0.0,
            start_lon_deg=0.0,
            end_lat_deg=0.0,
            end_lon_deg=1.0,
            n_segments=0,
        )
