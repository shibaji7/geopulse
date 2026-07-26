"""Tests for :mod:`geopulse.io.matpower`."""

from __future__ import annotations

from pathlib import Path

import pytest

from geopulse.exceptions import DataError
from geopulse.io.matpower import read_matpower_gmd

EPRI21 = Path(__file__).parent.parent.parent.parent / "benchmarks" / "horton2012" / "epri21.m"

pytestmark = pytest.mark.skipif(not EPRI21.is_file(), reason="epri21.m benchmark data not present")


def test_epri21_load_shape():
    case = read_matpower_gmd(EPRI21)
    # Horton (2012) benchmark: 21 AC buses, 8 substations, ~30 DC nodes,
    # ~30 DC branches; check ballpark dimensions.
    assert case.n_dc_nodes >= 20
    assert case.n_dc_nodes == len(case.dc_nodes)
    assert len(case.dc_branches) >= 20
    # Horton file uses gapped bus numbering (1-8, 11-21) — 19 AC buses, not 21.
    assert len(case.bus_latlon) == 19


def test_epri21_dc_node_fields():
    case = read_matpower_gmd(EPRI21)
    dn = case.dc_nodes[0]
    assert set(dn.keys()) >= {"row", "parent_ac_bus", "status", "g_gnd_S", "name"}
    assert dn["row"] == 1
    assert dn["g_gnd_S"] >= 0.0
    assert dn["name"].startswith("dc_")


def test_epri21_dc_branch_endpoints_are_row_positions():
    case = read_matpower_gmd(EPRI21)
    # from_row / to_row reference 1-based row positions within dc_nodes.
    valid_rows = {dn["row"] for dn in case.dc_nodes}
    for db in case.dc_branches:
        assert db["from_row"] in valid_rows
        assert db["to_row"] in valid_rows
        assert db["resistance_Ohm"] >= 0.0


def test_missing_file_raises():
    with pytest.raises(DataError, match="not found"):
        read_matpower_gmd("/tmp/definitely_does_not_exist.m")
