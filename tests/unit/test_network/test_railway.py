"""Unit tests for :mod:`geopulse.network.railway`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geopulse.exceptions import DataError
from geopulse.network.railway import (
    RailwayNetwork,
    RailwayParameters,
    rail_to_earth_voltage_analytic,
)
from geopulse.solver.nam import NAMSolver


def _default_params(**overrides) -> RailwayParameters:
    base = {
        "length_m": 20_000.0,
        "series_impedance_Ohm_per_m": 1e-4,
        "shunt_admittance_S_per_m": 5e-4,
        "start_lat_deg": 55.9,
        "start_lon_deg": -3.2,
        "end_lat_deg": 55.86,
        "end_lon_deg": -3.9,
        "n_segments": 40,
        "ground_positions_m": (0.0,),
        "ground_resistance_Ohm": 0.5,
    }
    base.update(overrides)
    return RailwayParameters(**base)


class TestRailwayParameters:
    def test_defaults_ok(self):
        p = _default_params()
        assert p.n_segments == 40
        assert p.ground_resistance_Ohm == 0.5

    @pytest.mark.parametrize(
        "field,bad",
        [
            ("length_m", -1.0),
            ("series_impedance_Ohm_per_m", 0.0),
            ("shunt_admittance_S_per_m", -1e-4),
            ("n_segments", 0),
            ("ground_resistance_Ohm", 0.0),
        ],
    )
    def test_rejects_bad_scalars(self, field, bad):
        with pytest.raises(DataError):
            _default_params(**{field: bad})

    def test_rejects_out_of_range_ground(self):
        with pytest.raises(DataError, match="out of range"):
            _default_params(ground_positions_m=(-100.0,))
        with pytest.raises(DataError, match="out of range"):
            _default_params(ground_positions_m=(30_000.0,))


class TestRailwayNetworkShape:
    def test_node_and_branch_counts(self):
        net = RailwayNetwork(_default_params(n_segments=25))
        assert len(list(net.get_nodes())) == 26
        assert len(list(net.get_branches())) == 25

    def test_matrix_shapes(self):
        net = RailwayNetwork(_default_params(n_segments=10))
        Y = net.assemble_network_admittance()
        Z = net.assemble_earthing_impedance()
        assert Y.shape == (11, 11)
        assert Z.shape == (11, 11)
        # Y symmetric.
        assert np.allclose(Y, Y.T)
        # Z diagonal.
        assert np.allclose(Z - np.diag(np.diag(Z)), 0.0)

    def test_thevenin_vector_length(self):
        net = RailwayNetwork(_default_params(n_segments=15))
        V_th = net.compute_thevenin_voltages(ex_Vm=1e-3, ey_Vm=0.0)
        assert V_th.shape == (15,)


class TestGroundedBonding:
    def test_ground_indices_are_snapped_and_deduped(self):
        # Length 100 with 10 segments → nodes at every 10 m. Two positions
        # 12 and 13 both snap to node index 1; 47 snaps to node 5.
        p = _default_params(length_m=100.0, n_segments=10, ground_positions_m=(12.0, 13.0, 47.0))
        net = RailwayNetwork(p)
        assert net.ground_node_indices == (1, 5)

    def test_bonded_node_has_lower_earthing_impedance(self):
        p_no_bond = _default_params(ground_positions_m=())
        p_with_bond = _default_params(ground_positions_m=(10_000.0,))
        net_no = RailwayNetwork(p_no_bond)
        net_yes = RailwayNetwork(p_with_bond)
        # Middle node (index 20 in 40-seg net) — bond attaches to node 20.
        Z_no = net_no.assemble_earthing_impedance()
        Z_yes = net_yes.assemble_earthing_impedance()
        assert Z_yes[20, 20] < Z_no[20, 20]

    def test_zero_bonds_matches_no_bond_matrix(self):
        p_no_bond = _default_params(ground_positions_m=())
        net = RailwayNetwork(p_no_bond)
        assert net.ground_node_indices == ()
        # Every diagonal entry uses only distributed shunt.
        Z = net.assemble_earthing_impedance()
        assert Z[20, 20] == Z[19, 19]  # both interior nodes → same value


class TestAgainstAnalytic:
    def test_insulated_end_solve_matches_analytic(self):
        """No bonded grounds → numerical solve should match the pipeline analytic."""
        L = 50_000.0
        z = 1e-4
        y = 5e-4
        E = 1e-3
        params = _default_params(
            length_m=L,
            series_impedance_Ohm_per_m=z,
            shunt_admittance_S_per_m=y,
            n_segments=80,
            ground_positions_m=(),
            # E-field is applied purely eastward; align the track E-W so the
            # driving component is longitudinal.
            start_lat_deg=45.0,
            start_lon_deg=-75.0,
            end_lat_deg=45.0,
            end_lon_deg=-75.0 + np.degrees(L / (6_378_137.0 * np.cos(np.radians(45.0)))),
        )
        net = RailwayNetwork(params)
        Y = net.assemble_network_admittance()
        Z = net.assemble_earthing_impedance()
        V_th = net.compute_thevenin_voltages(ex_Vm=E, ey_Vm=0.0)
        r = NAMSolver().solve(net, Y, Z, V_th)

        V_analytic = rail_to_earth_voltage_analytic(E, L, z, y, net.node_positions_m)
        # Numerical π-section solve vs closed form — few-percent agreement
        # at n_segments = 80 is expected for the DSTL discretisation.
        diff = np.max(np.abs(r.node_voltages_V - V_analytic))
        rms = float(np.sqrt(np.mean(V_analytic**2)))
        assert diff / rms < 0.02


class TestFromFile:
    def test_yaml_roundtrip(self, tmp_path: Path):
        yaml_body = """
length_m: 15000.0
series_impedance_Ohm_per_m: 1.0e-4
shunt_admittance_S_per_m: 5.0e-4
start_lat_deg: 55.9
start_lon_deg: -3.2
end_lat_deg: 55.86
end_lon_deg: -3.9
n_segments: 30
ground_positions_m: [0.0, 7500.0, 15000.0]
ground_resistance_Ohm: 0.4
"""
        f = tmp_path / "track.yml"
        f.write_text(yaml_body)
        net = RailwayNetwork.from_file(str(f))
        assert net.params.length_m == 15_000.0
        assert net.params.n_segments == 30
        assert net.params.ground_resistance_Ohm == 0.4
        assert net.ground_node_indices == (0, 15, 30)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(DataError, match="does not exist"):
            RailwayNetwork.from_file(str(tmp_path / "nope.yml"))

    def test_non_mapping_raises(self, tmp_path: Path):
        f = tmp_path / "bad.yml"
        f.write_text("- 1\n- 2\n- 3\n")
        with pytest.raises(DataError, match="mapping"):
            RailwayNetwork.from_file(str(f))

    def test_missing_required_key_raises(self, tmp_path: Path):
        # Missing shunt_admittance_S_per_m.
        f = tmp_path / "partial.yml"
        f.write_text(
            "length_m: 100\n"
            "series_impedance_Ohm_per_m: 1e-4\n"
            "start_lat_deg: 0\n"
            "start_lon_deg: 0\n"
            "end_lat_deg: 0\n"
            "end_lon_deg: 0.001\n"
        )
        with pytest.raises(DataError, match="missing"):
            RailwayNetwork.from_file(str(f))
