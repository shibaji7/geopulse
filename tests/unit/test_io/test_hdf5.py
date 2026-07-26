"""Tests for :mod:`geopulse.io.hdf5`."""

from __future__ import annotations

import numpy as np
import pytest

from geopulse._version import __version__ as GEOPULSE_VERSION
from geopulse.exceptions import DataError
from geopulse.io.hdf5 import CURRENT_SCHEMA_VERSION, read_results, write_results


def test_roundtrip_array_group(tmp_hdf5_path):
    write_results(
        tmp_hdf5_path,
        description="roundtrip test",
        source={
            "time_s": np.arange(10.0),
            "bx_T": np.linspace(0.0, 1.0, 10),
        },
    )
    data = read_results(tmp_hdf5_path)
    assert data["schema_version"] == CURRENT_SCHEMA_VERSION
    assert data["geopulse_version"] == GEOPULSE_VERSION
    assert data["description"] == "roundtrip test"
    np.testing.assert_array_equal(data["source"]["time_s"], np.arange(10.0))


def test_nested_groups_and_scalars(tmp_hdf5_path):
    write_results(
        tmp_hdf5_path,
        results={
            "efield": {
                "Ex_Vm": np.array([1.0, 2.0, 3.0]),
                "Ey_Vm": np.array([0.5, 1.5, 2.5]),
            },
            "meta": {
                "n_freqs": 128,  # int → attribute
                "fs_Hz": 1.0,  # float → attribute
                "algorithm": "planewave",  # str → attribute
            },
        },
    )
    data = read_results(tmp_hdf5_path)
    np.testing.assert_array_equal(data["results"]["efield"]["Ex_Vm"], [1.0, 2.0, 3.0])
    assert int(data["results"]["meta"]["n_freqs"]) == 128
    assert float(data["results"]["meta"]["fs_Hz"]) == 1.0


def test_string_arrays_stored_as_variable_length(tmp_hdf5_path):
    write_results(
        tmp_hdf5_path,
        gic={"node_ids": ["dc_sub1", "dc_sub2", "dc_sub3"]},
    )
    data = read_results(tmp_hdf5_path)
    ids = [s.decode() if isinstance(s, bytes) else s for s in data["gic"]["node_ids"]]
    assert ids == ["dc_sub1", "dc_sub2", "dc_sub3"]


def test_int_and_float_scalar_groups(tmp_hdf5_path):
    write_results(
        tmp_hdf5_path,
        cfg={
            "n_nodes": np.int64(27),  # numpy int scalar
            "rtol": np.float64(1e-6),  # numpy float scalar
        },
    )
    data = read_results(tmp_hdf5_path)
    assert int(data["cfg"]["n_nodes"]) == 27
    assert float(data["cfg"]["rtol"]) == pytest.approx(1e-6)


def test_unsupported_value_type_raises(tmp_hdf5_path):
    with pytest.raises(DataError, match="Unsupported value type"):
        write_results(tmp_hdf5_path, weird={"bad": {1, 2, 3}})  # set is not supported


def test_top_level_group_must_be_dict(tmp_hdf5_path):
    with pytest.raises(DataError, match="must be a dict"):
        write_results(tmp_hdf5_path, oops=[1, 2, 3])  # top-level must be dict


def test_read_missing_schema_version_raises(tmp_hdf5_path):
    import h5py

    with h5py.File(tmp_hdf5_path, "w") as fh:
        fh.create_dataset("something", data=[1, 2, 3])
    with pytest.raises(DataError, match="missing 'schema_version'"):
        read_results(tmp_hdf5_path)
