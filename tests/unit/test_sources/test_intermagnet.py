"""Tests for :class:`geopulse.sources.intermagnet.INTERMAGNETSource`."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from geopulse.constants import NT_TO_T
from geopulse.exceptions import DataError
from geopulse.sources.intermagnet import INTERMAGNETSource


def _write_iaga(path: Path, reporting: str, body: str) -> Path:
    def kv(k: str, v: str) -> str:
        return f" {k:<23}{v:<44}|"

    header = "\n".join(
        [
            kv("Format", "IAGA-2002"),
            kv("Source of Data", "synthetic"),
            kv("Station Name", "Ottawa"),
            kv("IAGA CODE", "OTT"),
            kv("Geodetic Latitude", "45.403"),
            kv("Geodetic Longitude", "284.448"),
            kv("Elevation", "75.000"),
            kv("Reporting", reporting),
            kv("Sensor Orientation", "HDZF"),
            kv("Digital Sampling", "1 minute"),
            kv("Data Interval Type", "Average 1-Minute"),
            kv("Data Type", "Reported"),
        ]
    )
    col = (
        f"DATE       TIME         DOY     OTT{reporting[0]}      "
        f"OTT{reporting[1]}      OTT{reporting[2]}      OTT{reporting[3]}   |"
    )
    path.write_text(header + "\n" + col + "\n" + body)
    return path


def test_xyzf_load_converts_to_tesla(tmp_path):
    body = dedent(
        """\
        2024-05-10 00:00:00.000 131     18000.00  -3800.00  55000.00  57900.00
        2024-05-10 00:01:00.000 131     18001.00  -3801.00  55001.00  57901.00
        """
    )
    p = _write_iaga(tmp_path / "OTT.min", "XYZF", body)
    src = INTERMAGNETSource(str(p))
    b = src.load()
    assert b.station_id == "OTT"
    assert b.bx_T.shape == (2,)
    np.testing.assert_allclose(b.bx_T[0], 18000.0 * NT_TO_T)
    np.testing.assert_allclose(b.by_T[0], -3800.0 * NT_TO_T)
    np.testing.assert_allclose(b.bz_T[0], 55000.0 * NT_TO_T)
    # 1-minute sampling
    np.testing.assert_allclose(b.sampling_rate_Hz, 1 / 60.0)


def test_hdzf_converts_to_xy(tmp_path):
    # H = 20000 nT, D = -720 arc-minutes = -12 deg (declination west).
    # Expected: X = H · cos(D_rad), Y = H · sin(D_rad).
    body = "2024-05-10 00:00:00.000 131     20000.00  -720.00   55000.00  58200.00\n"
    p = _write_iaga(tmp_path / "OTT_hdzf.min", "HDZF", body)
    src = INTERMAGNETSource(str(p), declination_unit="minutes")
    b = src.load()
    D_rad = np.radians(-12.0)
    expected_bx = 20000.0 * np.cos(D_rad) * NT_TO_T
    expected_by = 20000.0 * np.sin(D_rad) * NT_TO_T
    np.testing.assert_allclose(b.bx_T[0], expected_bx, rtol=1e-6)
    np.testing.assert_allclose(b.by_T[0], expected_by, rtol=1e-6)


def test_hdzf_declination_in_degrees(tmp_path):
    body = "2024-05-10 00:00:00.000 131     20000.00  -12.00    55000.00  58200.00\n"
    p = _write_iaga(tmp_path / "OTT_hdzf_deg.min", "HDZF", body)
    src = INTERMAGNETSource(str(p), declination_unit="degrees")
    b = src.load()
    D_rad = np.radians(-12.0)
    np.testing.assert_allclose(b.bx_T[0], 20000.0 * np.cos(D_rad) * NT_TO_T, rtol=1e-6)


def test_load_time_window(tmp_path):
    body = dedent(
        """\
        2024-05-10 00:00:00.000 131     18000.00  -3800.00  55000.00  57900.00
        2024-05-10 00:01:00.000 131     18001.00  -3801.00  55001.00  57901.00
        2024-05-10 00:02:00.000 131     18002.00  -3802.00  55002.00  57902.00
        """
    )
    p = _write_iaga(tmp_path / "OTT.min", "XYZF", body)
    b = INTERMAGNETSource(str(p)).load(
        start_s=0.0,
        end_s=1_715_299_260.0,  # 2024-05-10T00:01:00Z + 1 s
    )
    assert b.bx_T.shape[0] == 2


def test_empty_window_raises(tmp_path):
    body = "2024-05-10 00:00:00.000 131     0 0 0 0\n"
    p = _write_iaga(tmp_path / "OTT.min", "XYZF", body)
    with pytest.raises(DataError, match="No IAGA-2002 samples"):
        INTERMAGNETSource(str(p)).load(start_s=1e10, end_s=2e10)


def test_bad_declination_unit_rejected():
    with pytest.raises(DataError, match="declination_unit"):
        INTERMAGNETSource("whatever.min", declination_unit="radians")
