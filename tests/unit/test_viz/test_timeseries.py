"""Unit tests for :mod:`geopulse.viz.timeseries`."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402  — force headless backend for CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from geopulse.exceptions import ShapeMismatchError  # noqa: E402
from geopulse.viz.timeseries import TimeSeriesPanel, plot_timeseries  # noqa: E402


@pytest.fixture
def t_and_signals():
    t = np.linspace(0.0, 3600.0, 361)
    bx_T = 10e-9 * np.sin(2 * np.pi * t / 900.0)
    by_T = 5e-9 * np.cos(2 * np.pi * t / 900.0)
    ex_Vm = 1e-4 * np.sin(2 * np.pi * t / 450.0)
    return t, bx_T, by_T, ex_Vm


def test_single_panel_returns_figure_with_right_axes(t_and_signals):
    t, bx_T, _, _ = t_and_signals
    fig = plot_timeseries(
        t,
        [TimeSeriesPanel({"Bx": bx_T}, ylabel="B (nT)", unit_scale=1e9)],
        time_unit="min",
    )
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_ylabel() == "B (nT)"
    assert ax.get_xlabel() == "Time (min)"
    plt.close(fig)


def test_multi_panel_shares_x_axis(t_and_signals):
    t, bx_T, by_T, ex_Vm = t_and_signals
    fig = plot_timeseries(
        t,
        [
            TimeSeriesPanel({"Bx": bx_T, "By": by_T}, ylabel="B (nT)", unit_scale=1e9),
            TimeSeriesPanel({"Ex": ex_Vm}, ylabel="E (mV/m)", unit_scale=1e3),
        ],
    )
    assert len(fig.axes) == 2
    # Shared x means both axes register in the same sibling group.
    top, bot = fig.axes
    siblings = top.get_shared_x_axes().get_siblings(top)
    assert bot in siblings
    # Two-series panel gets a legend, one-series doesn't (auto behaviour).
    assert top.get_legend() is not None
    assert bot.get_legend() is None
    plt.close(fig)


def test_unit_scale_applied_before_plot(t_and_signals):
    t, bx_T, _, _ = t_and_signals
    fig = plot_timeseries(
        t,
        [TimeSeriesPanel({"Bx": bx_T}, ylabel="B (nT)", unit_scale=1e9)],
    )
    line = fig.axes[0].lines[0]
    ydata = line.get_ydata()
    # Expect the plotted values to be Tesla × 1e9 = nT.
    assert np.allclose(ydata, bx_T * 1e9)
    plt.close(fig)


def test_utc_axis_when_origin_given(t_and_signals):
    t, bx_T, _, _ = t_and_signals
    origin = 1_715_299_200.0  # 2024-05-10 00:00 UTC
    t_utc = origin + t
    fig = plot_timeseries(
        t_utc,
        [TimeSeriesPanel({"Bx": bx_T}, ylabel="B (nT)", unit_scale=1e9)],
        time_origin_utc=origin,
    )
    xlabel = fig.axes[0].get_xlabel()
    assert "UTC" in xlabel
    assert "2024-05-10" in xlabel
    plt.close(fig)


def test_ylim_respected(t_and_signals):
    t, bx_T, _, _ = t_and_signals
    fig = plot_timeseries(
        t,
        [
            TimeSeriesPanel(
                {"Bx": bx_T},
                ylabel="B (nT)",
                unit_scale=1e9,
                ylim=(-100.0, 100.0),
            )
        ],
    )
    lo, hi = fig.axes[0].get_ylim()
    assert (lo, hi) == (-100.0, 100.0)
    plt.close(fig)


def test_savepath_writes_file(tmp_path, t_and_signals):
    t, bx_T, _, _ = t_and_signals
    out = tmp_path / "toy.png"
    plot_timeseries(
        t,
        [TimeSeriesPanel({"Bx": bx_T}, ylabel="B (nT)", unit_scale=1e9)],
        savepath=out,
    )
    assert out.is_file()
    assert out.stat().st_size > 0
    plt.close("all")


def test_ax_injection_reuses_supplied_axes(t_and_signals):
    t, bx_T, by_T, _ = t_and_signals
    outer_fig, ax = plt.subplots(figsize=(6, 3))
    fig = plot_timeseries(
        t,
        [TimeSeriesPanel({"Bx": bx_T, "By": by_T}, ylabel="B (nT)", unit_scale=1e9)],
        axes=[ax],
    )
    assert fig is outer_fig
    assert len(ax.lines) == 2
    plt.close(outer_fig)


def test_empty_panels_raises():
    with pytest.raises(ShapeMismatchError, match="at least one panel"):
        plot_timeseries(np.zeros(5), [])


def test_series_length_mismatch_raises(t_and_signals):
    t, _, _, _ = t_and_signals
    bad = np.zeros(t.size + 1)
    with pytest.raises(ShapeMismatchError, match="expected"):
        plot_timeseries(t, [TimeSeriesPanel({"bad": bad}, ylabel="x")])


def test_empty_series_dict_raises(t_and_signals):
    t, _, _, _ = t_and_signals
    with pytest.raises(ShapeMismatchError, match="no series"):
        plot_timeseries(t, [TimeSeriesPanel({}, ylabel="x")])


def test_axes_length_mismatch_raises(t_and_signals):
    t, bx_T, _, _ = t_and_signals
    _, axs = plt.subplots(2, 1)
    with pytest.raises(ShapeMismatchError, match="does not match panels"):
        plot_timeseries(
            t,
            [TimeSeriesPanel({"Bx": bx_T}, ylabel="B (nT)", unit_scale=1e9)],
            axes=list(axs),
        )
    plt.close("all")
