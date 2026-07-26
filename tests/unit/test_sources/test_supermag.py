"""Tests for :class:`geopulse.sources.supermag.SuperMAGSource`."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from geopulse.constants import NT_TO_T
from geopulse.exceptions import DataError
from geopulse.sources.supermag import SuperMAGSource


def _write_csv(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


def test_single_station_load(tmp_path):
    body = dedent(
        """\
        # SuperMAG export
        # baseline: yearly mean
        Date_UTC,IAGA,GEOLAT,GEOLON,N,E,Z
        2024-05-10T00:00:00,OTT,45.4,284.4,18000.0,-3800.0,55000.0
        2024-05-10T00:01:00,OTT,45.4,284.4,18001.0,-3801.0,55001.0
        2024-05-10T00:02:00,OTT,45.4,284.4,18002.0,-3802.0,55002.0
        """
    )
    p = _write_csv(tmp_path / "sm.csv", body)
    b = SuperMAGSource(str(p)).load()
    assert b.station_id == "OTT"
    assert b.bx_T.shape == (3,)
    np.testing.assert_allclose(b.bx_T[0], 18000.0 * NT_TO_T)
    np.testing.assert_allclose(b.latitude_deg, 45.4)
    np.testing.assert_allclose(b.sampling_rate_Hz, 1 / 60.0)


def test_multi_station_selects_by_iaga(tmp_path):
    body = dedent(
        """\
        Date_UTC,IAGA,GEOLAT,GEOLON,N,E,Z
        2024-05-10T00:00:00,OTT,45.4,284.4,18000,-3800,55000
        2024-05-10T00:00:00,BOU,40.1,254.8,25000,-6200,45000
        2024-05-10T00:01:00,OTT,45.4,284.4,18001,-3801,55001
        2024-05-10T00:01:00,BOU,40.1,254.8,25001,-6201,45001
        """
    )
    p = _write_csv(tmp_path / "sm_multi.csv", body)
    b = SuperMAGSource(str(p), iaga="BOU").load()
    assert b.station_id == "BOU"
    np.testing.assert_allclose(b.bx_T[0], 25000.0 * NT_TO_T)
    np.testing.assert_allclose(b.latitude_deg, 40.1)


def test_time_window_filters(tmp_path):
    body = dedent(
        """\
        Date_UTC,IAGA,GEOLAT,GEOLON,N,E,Z
        2024-05-10T00:00:00,OTT,45.4,284.4,10.0,0.0,0.0
        2024-05-10T00:01:00,OTT,45.4,284.4,20.0,0.0,0.0
        2024-05-10T00:02:00,OTT,45.4,284.4,30.0,0.0,0.0
        """
    )
    p = _write_csv(tmp_path / "sm.csv", body)
    # Grab the middle sample only.
    import calendar

    t0 = calendar.timegm((2024, 5, 10, 0, 0, 30, 0, 0, 0))
    t1 = calendar.timegm((2024, 5, 10, 0, 1, 30, 0, 0, 0))
    b = SuperMAGSource(str(p)).load(start_s=t0, end_s=t1)
    assert b.bx_T.shape == (1,)
    np.testing.assert_allclose(b.bx_T[0], 20.0 * NT_TO_T)


def test_missing_columns_raises(tmp_path):
    body = "Date_UTC,IAGA,N,E\n2024-05-10T00:00:00,OTT,1,2\n"  # no Z
    p = _write_csv(tmp_path / "bad.csv", body)
    with pytest.raises(DataError, match="missing required columns"):
        SuperMAGSource(str(p)).load()


def test_missing_file_raises():
    with pytest.raises(DataError, match="not found"):
        SuperMAGSource("/tmp/no_such_supermag.csv").load()


def test_iso_and_space_separated_times(tmp_path):
    body = dedent(
        """\
        Date_UTC,IAGA,GEOLAT,GEOLON,N,E,Z
        2024-05-10T00:00:00,OTT,45.4,284.4,1.0,0.0,0.0
        2024-05-10 00:01:00,OTT,45.4,284.4,2.0,0.0,0.0
        """
    )
    p = _write_csv(tmp_path / "sm_mixed_ts.csv", body)
    b = SuperMAGSource(str(p)).load()
    assert b.bx_T.shape == (2,)
    # 60 s apart
    assert b.time_s[1] - b.time_s[0] == pytest.approx(60.0)
