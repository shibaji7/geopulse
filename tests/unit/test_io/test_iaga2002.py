"""Tests for :mod:`geopulse.io.iaga2002` — synthetic in-memory fixtures."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from geopulse.exceptions import DataError
from geopulse.io.iaga2002 import read_iaga2002


def _write_iaga(path: Path, reporting: str = "XYZF", body_rows: str = "") -> Path:
    # Header keys are 24 chars wide; pad accordingly.
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
    path.write_text(header + "\n" + col + "\n" + body_rows)
    return path


def test_read_xyzf_basic(tmp_path):
    body = dedent(
        """\
        2024-05-10 00:00:00.000 131     18000.00  -3800.00  55000.00  57900.00
        2024-05-10 00:01:00.000 131     18000.10  -3800.20  55000.10  57900.10
        2024-05-10 00:02:00.000 131     18000.20  -3800.40  55000.20  57900.20
        """
    )
    p = _write_iaga(tmp_path / "OTT.min", "XYZF", body)
    f = read_iaga2002(p)
    assert f.station_code == "OTT"
    assert f.reporting == "XYZF"
    # Longitude 284.448 wraps to -75.552
    np.testing.assert_allclose(f.longitude_deg, -75.552, rtol=1e-4)
    assert f.time_utc.shape == (3,)
    np.testing.assert_allclose(f.col1_nT, [18000.0, 18000.1, 18000.2])


def test_missing_sentinels_become_nan(tmp_path):
    body = dedent(
        """\
        2024-05-10 00:00:00.000 131     18000.00  -3800.00  55000.00  57900.00
        2024-05-10 00:01:00.000 131     99999.00  88888.00  55000.10  57900.10
        """
    )
    p = _write_iaga(tmp_path / "OTT.min", "XYZF", body)
    f = read_iaga2002(p)
    assert np.isnan(f.col1_nT[1])
    assert np.isnan(f.col2_nT[1])
    assert not np.isnan(f.col1_nT[0])


def test_hdzf_reporting_preserved(tmp_path):
    body = dedent(
        """\
        2024-05-10 00:00:00.000 131     19000.00  -720.00   55000.00  58200.00
        """
    )
    p = _write_iaga(tmp_path / "OTT_hdzf.min", "HDZF", body)
    f = read_iaga2002(p)
    assert f.reporting == "HDZF"
    # Raw D column (col2) preserved as-is; conversion happens in the source adapter
    assert f.col2_nT[0] == pytest.approx(-720.0)


def test_time_parsed_as_utc_epoch(tmp_path):
    body = "2024-05-10 00:00:00.000 131     18000.00  -3800.00  55000.00  57900.00\n"
    p = _write_iaga(tmp_path / "one.min", "XYZF", body)
    f = read_iaga2002(p)
    # 2024-05-10T00:00:00Z as epoch
    import calendar

    expected = calendar.timegm((2024, 5, 10, 0, 0, 0, 0, 0, 0))
    assert f.time_utc[0] == pytest.approx(expected)


def test_missing_file_raises():
    with pytest.raises(DataError, match="not found"):
        read_iaga2002("/tmp/does_not_exist_iaga.min")


def test_bad_reporting_raises(tmp_path):
    body = "2024-05-10 00:00:00.000 131     0 0 0 0\n"
    p = _write_iaga(tmp_path / "bad.min", "XYZF", body)
    text = p.read_text().replace("Reporting", "Reporting").replace("XYZF", "ABCD")
    p.write_text(text)
    with pytest.raises(DataError, match="reporting orientation"):
        read_iaga2002(p)
