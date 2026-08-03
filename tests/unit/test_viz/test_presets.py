"""Unit tests for :mod:`geopulse.viz.presets`."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — force headless backend for CI

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from geopulse.exceptions import DataError  # noqa: E402
from geopulse.viz.presets import (  # noqa: E402
    GOLDEN_RATIO,
    MIN_READABLE_PT,
    MM_PER_INCH,
    PRESETS,
    FigurePreset,
    apply_preset,
    save_figure,
)

# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_covers_expected_venues(self):
        names = set(PRESETS)
        assert {
            "jgr_1col",
            "jgr_2col",
            "sw_1col",
            "sw_2col",
            "nature_1col",
            "nature_2col",
            "ieee_1col",
            "ieee_2col",
            "agu_poster",
            "presentation",
            "preprint",
        } <= names

    def test_every_preset_is_frozen_dataclass(self):
        from dataclasses import FrozenInstanceError

        for name, p in PRESETS.items():
            assert isinstance(p, FigurePreset), name
            with pytest.raises(FrozenInstanceError):
                p.width_mm = 999.0  # type: ignore[misc]

    def test_widths_within_sane_range(self):
        # No venue should have a nonsense width. 40 mm - 500 mm covers
        # everything from column extract to full poster.
        for name, p in PRESETS.items():
            assert 40.0 < p.width_mm < 500.0, name

    def test_dpi_meets_venue_minimums(self):
        # Print journals want >= 300; presentations tolerate 150.
        for name, p in PRESETS.items():
            if name in {"agu_poster", "presentation"}:
                assert p.dpi >= 150
            else:
                assert p.dpi >= 200

    def test_save_formats_nonempty(self):
        for name, p in PRESETS.items():
            assert len(p.save_formats) >= 1, name


# ---------------------------------------------------------------------------
# apply_preset
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_rcparams():
    """Snapshot rcParams around every test so presets can't leak."""
    original = mpl.rcParams.copy()
    yield
    mpl.rcParams.update(original)


class TestApplyPreset:
    def test_sets_font_family_and_size(self):
        apply_preset("jgr_2col")
        assert mpl.rcParams["font.family"] == ["serif"]
        assert mpl.rcParams["font.size"] == 8.0

    def test_sans_serif_for_nature(self):
        apply_preset("nature_1col")
        assert mpl.rcParams["font.family"] == ["sans-serif"]
        assert mpl.rcParams["font.size"] == 7.0

    def test_dpi_applied(self):
        apply_preset("preprint")
        assert mpl.rcParams["figure.dpi"] == 200
        assert mpl.rcParams["savefig.dpi"] == 200

    def test_accepts_preset_object_directly(self):
        p = PRESETS["ieee_1col"]
        apply_preset(p)
        assert mpl.rcParams["font.size"] == p.base_font_pt

    def test_unknown_preset_raises(self):
        with pytest.raises(DataError, match="unknown preset"):
            apply_preset("bogus_venue")

    def test_savefig_bbox_tight(self):
        apply_preset("jgr_1col")
        assert mpl.rcParams["savefig.bbox"] == "tight"


# ---------------------------------------------------------------------------
# save_figure
# ---------------------------------------------------------------------------


class TestSaveFigure:
    def test_resizes_to_exact_column_width(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        save_figure(fig, tmp_path / "out", "jgr_1col")
        w, _ = fig.get_size_inches()
        assert w == pytest.approx(89.0 / MM_PER_INCH, rel=1e-6)
        plt.close(fig)

    def test_default_aspect_uses_golden(self, tmp_path):
        # Preset without a fixed aspect_ratio → golden ratio.
        fig, ax = plt.subplots()
        save_figure(fig, tmp_path / "out", "jgr_2col")
        w, h = fig.get_size_inches()
        assert h == pytest.approx(w / GOLDEN_RATIO, rel=1e-6)
        plt.close(fig)

    def test_preset_aspect_used_when_present(self, tmp_path):
        # agu_poster fixes aspect_ratio=0.75 (4:3).
        fig, ax = plt.subplots()
        save_figure(fig, tmp_path / "poster", "agu_poster")
        w, h = fig.get_size_inches()
        assert h == pytest.approx(w * 0.75, rel=1e-6)
        plt.close(fig)

    def test_caller_aspect_overrides_preset(self, tmp_path):
        fig, ax = plt.subplots()
        save_figure(fig, tmp_path / "out", "jgr_1col", aspect=1.0)
        w, h = fig.get_size_inches()
        assert h == pytest.approx(w, rel=1e-6)
        plt.close(fig)

    def test_writes_every_venue_format(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([0, 1])
        out = tmp_path / "figure"
        written = save_figure(fig, out, "nature_1col")  # PDF + PNG + EPS
        assert {p.suffix for p in written} == {".pdf", ".png", ".eps"}
        for p in written:
            assert p.is_file() and p.stat().st_size > 0
        plt.close(fig)

    def test_rejects_bad_aspect(self, tmp_path):
        fig, _ = plt.subplots()
        with pytest.raises(DataError, match="positive"):
            save_figure(fig, tmp_path / "x", "jgr_1col", aspect=0.0)
        with pytest.raises(DataError, match="positive"):
            save_figure(fig, tmp_path / "x", "jgr_1col", aspect=-1.0)
        plt.close(fig)

    def test_rejects_unknown_preset(self, tmp_path):
        fig, _ = plt.subplots()
        with pytest.raises(DataError, match="unknown preset"):
            save_figure(fig, tmp_path / "x", "not_a_venue")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Readability warning
# ---------------------------------------------------------------------------


class TestReadabilityCheck:
    def test_warns_on_tiny_font(self, tmp_path):
        fig, ax = plt.subplots()
        # Force a tiny label — below the 6-pt readability threshold.
        ax.set_title("tiny title", fontsize=3.0)
        with pytest.warns(UserWarning, match="render below"):
            save_figure(fig, tmp_path / "x", "jgr_1col")
        plt.close(fig)

    def test_no_warning_when_fonts_are_reasonable(self, tmp_path):
        fig, ax = plt.subplots()
        ax.set_title("readable title", fontsize=12.0)
        ax.set_xlabel("x", fontsize=10.0)
        # Use an explicit filter so any UserWarning would surface as
        # a test failure.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            save_figure(fig, tmp_path / "x", "jgr_2col")
        plt.close(fig)

    def test_check_readability_toggle_off(self, tmp_path):
        fig, ax = plt.subplots()
        ax.set_title("tiny", fontsize=2.0)  # smaller than MIN_READABLE_PT
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # Should not raise because the check is disabled.
            save_figure(fig, tmp_path / "x", "jgr_1col", check_readability=False)
        plt.close(fig)

    def test_min_readable_constant_is_documented(self):
        # Not a functional test — just guards against someone changing
        # the constant without thought.
        assert MIN_READABLE_PT == 6.0
