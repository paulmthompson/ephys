"""Tests for scatter-with-marginals plotting."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

from ephys.plotting.scatter_marginals import (
    ScatterMarginalsData,
    ScatterMarginalsOptions,
    diff_inset_bar_polygon_vertices,
    diff_inset_bars,
    diff_inset_default_count_extent,
    diff_inset_symmetric_edges,
    diff_inset_xy_for_difference,
    draw_scatter_marginals_into,
    marginal_histogram_edges,
    resolve_scatter_axis_limits,
    shared_scatter_axis_ticks,
    unity_line_segment,
)


def test_diff_inset_xy_for_difference() -> None:
    """Difference values map to scatter coordinates with correct y - x."""
    assert diff_inset_xy_for_difference(0.0, 15.0) == (15.0, 15.0)
    assert diff_inset_xy_for_difference(-10.0, 15.0) == (20.0, 10.0)
    x, y = diff_inset_xy_for_difference(4.0, 15.0)
    assert y - x == 4.0


def test_diff_inset_bars_overflow() -> None:
    """Overflow tails count values outside the nominal extent."""
    diffs = np.array([-20.0, -5.0, 0.0, 5.0, 20.0])
    bars = diff_inset_bars(diffs, 10.0, 4, overflow_bins=True)
    assert bars[0].is_overflow and bars[0].count == 1
    assert bars[-1].is_overflow and bars[-1].count == 1
    core_total = sum(b.count for b in bars if not b.is_overflow)
    assert core_total == 3


def test_diff_inset_bars_no_overflow_excludes_outliers() -> None:
    """Without overflow, out-of-range values are not counted."""
    diffs = np.array([-20.0, 0.0, 20.0])
    bars = diff_inset_bars(diffs, 10.0, 4, overflow_bins=False)
    assert sum(b.count for b in bars) == 1


def test_draw_diff_inset_overflow_bar_rendered() -> None:
    """An out-of-range difference produces a tail polygon when enabled."""
    data = ScatterMarginalsData(x=np.array([0.0]), y=np.array([25.0]))
    options = ScatterMarginalsOptions(
        diff_inset_center=10.0,
        diff_inset_d_half_extent=10.0,
        diff_inset_overflow_bins=True,
        diff_hist_bins=4,
    )
    fig = plt.figure(figsize=(3.0, 3.0))
    cell = fig.add_gridspec(1, 1)[0, 0]
    draw_scatter_marginals_into(fig, cell, data, options)
    scatter_ax = fig.axes[0]
    assert any(isinstance(p, Polygon) for p in scatter_ax.patches)
    plt.close(fig)


def test_diff_inset_symmetric_edges() -> None:
    """Symmetric edges span -extent to +extent."""
    edges = diff_inset_symmetric_edges(4.0, 4)
    assert edges[0] == -4.0
    assert edges[-1] == 4.0
    assert edges.size == 5


def test_diff_inset_bar_polygon_vertices_y_minus_x() -> None:
    """Bar base corners lie on bin edges in y - x."""
    verts = diff_inset_bar_polygon_vertices(1.0, 3.0, 2.0, 10.0)
    assert verts.shape == (4, 2)
    assert verts[0, 1] - verts[0, 0] == 1.0
    assert verts[1, 1] - verts[1, 0] == 3.0


def test_resolve_scatter_axis_limits_scatter_lim() -> None:
    """scatter_lim sets equal bounds without padding."""
    x = np.array([2.0, 18.0])
    y = np.array([3.0, 17.0])
    lo_x, hi_x, lo_y, hi_y = resolve_scatter_axis_limits(
        x, y, scatter_lim=(0.0, 20.0), xlim=None, ylim=None
    )
    assert (lo_x, hi_x, lo_y, hi_y) == (0.0, 20.0, 0.0, 20.0)


def test_resolve_scatter_axis_limits_separate() -> None:
    """xlim and ylim can differ."""
    x = np.array([0.0, 10.0])
    y = np.array([0.0, 20.0])
    lo_x, hi_x, lo_y, hi_y = resolve_scatter_axis_limits(
        x, y, scatter_lim=None, xlim=(0.0, 10.0), ylim=(0.0, 30.0)
    )
    assert lo_x == 0.0 and hi_x == 10.0
    assert lo_y == 0.0 and hi_y == 30.0


def test_shared_scatter_axis_ticks_within_limits() -> None:
    """Shared tick helper stays inside axis bounds."""
    ticks = shared_scatter_axis_ticks(0.0, 20.0, nbins=6)
    assert ticks.size >= 2
    assert ticks[0] >= 0.0
    assert ticks[-1] <= 20.0


def test_equal_scatter_limits_use_matching_tick_count() -> None:
    """Equal x/y limits get the same tick positions on both axes."""
    data = ScatterMarginalsData(x=np.array([5.0, 15.0]), y=np.array([6.0, 14.0]))
    options = ScatterMarginalsOptions(scatter_lim=(0.0, 20.0))
    fig = plt.figure(figsize=(4.0, 4.0))
    cell = fig.add_gridspec(1, 1)[0, 0]
    draw_scatter_marginals_into(fig, cell, data, options)
    scatter_ax = fig.axes[0]
    assert len(scatter_ax.get_xticks()) == len(scatter_ax.get_yticks())
    assert np.allclose(scatter_ax.get_xticks(), scatter_ax.get_yticks())
    plt.close(fig)


def test_marginal_histogram_edges_match_scatter_limits() -> None:
    """Marginal bins span each scatter axis limit."""
    edges_x, edges_y = marginal_histogram_edges(0.0, 20.0, 0.0, 25.0, 10)
    assert edges_x[0] == 0.0 and edges_x[-1] == 20.0
    assert edges_y[0] == 0.0 and edges_y[-1] == 25.0


def test_unity_line_segment() -> None:
    """Unity segment stays inside both axes."""
    assert unity_line_segment(0.0, 10.0, 0.0, 30.0) == (0.0, 0.0, 10.0, 10.0)


def test_scatter_marginals_options_rejects_invalid_lim() -> None:
    """Limit tuples must have max > min."""
    with pytest.raises(ValueError, match="scatter_lim"):
        ScatterMarginalsOptions(scatter_lim=(10.0, 5.0))


def test_diff_inset_default_count_extent() -> None:
    """Auto count extent is independent of bin count."""
    expected = (2.0 * 4.0 / 10.0) * 0.85
    assert diff_inset_default_count_extent(4.0, 4) == pytest.approx(expected)
    assert diff_inset_default_count_extent(4.0, 20) == pytest.approx(expected)


def _inset_bar_heights(scatter_ax) -> list[float]:
    """Euclidean height of each diff-inset bar polygon on the scatter axis."""
    heights: list[float] = []
    for patch in scatter_ax.patches:
        if not isinstance(patch, Polygon):
            continue
        verts = np.asarray(patch.get_xy())
        base_mid = 0.5 * (verts[0] + verts[1])
        top_mid = 0.5 * (verts[2] + verts[3])
        heights.append(float(np.linalg.norm(top_mid - base_mid)))
    return heights


def test_diff_inset_count_extent_independent_of_bins() -> None:
    """Explicit count_extent keeps tallest bar height when bin count changes."""
    x = np.array([0.0, 5.0, 10.0, 12.0, 14.0])
    y = np.array([2.0, 5.0, 8.0, 9.0, 10.0])
    data = ScatterMarginalsData(x=x, y=y)
    base_kw = {
        "diff_inset_center": 10.0,
        "diff_inset_d_half_extent": 10.0,
        "diff_inset_count_extent": 4.0,
    }
    heights_by_bins: dict[int, float] = {}
    for n_bins in (4, 20):
        options = ScatterMarginalsOptions(diff_hist_bins=n_bins, **base_kw)
        fig = plt.figure(figsize=(3.0, 3.0))
        cell = fig.add_gridspec(1, 1)[0, 0]
        draw_scatter_marginals_into(fig, cell, data, options)
        heights = _inset_bar_heights(fig.axes[0])
        assert heights
        heights_by_bins[n_bins] = max(heights)
        plt.close(fig)
    assert heights_by_bins[4] == pytest.approx(heights_by_bins[20], rel=1e-6)


def test_diff_inset_count_ylim_caps_bar_height() -> None:
    """count_ylim raises the scaling denominator so bars stay shorter."""
    x = np.repeat(0.0, 20)
    y = np.repeat(5.0, 20)
    data = ScatterMarginalsData(x=x, y=y)
    shared_kw = {
        "diff_inset_center": 10.0,
        "diff_inset_d_half_extent": 10.0,
        "diff_inset_count_extent": 4.0,
        "diff_hist_bins": 4,
    }
    fig_auto = plt.figure(figsize=(3.0, 3.0))
    draw_scatter_marginals_into(
        fig_auto,
        fig_auto.add_gridspec(1, 1)[0, 0],
        data,
        ScatterMarginalsOptions(**shared_kw),
    )
    auto_h = max(_inset_bar_heights(fig_auto.axes[0]))
    plt.close(fig_auto)

    fig_capped = plt.figure(figsize=(3.0, 3.0))
    draw_scatter_marginals_into(
        fig_capped,
        fig_capped.add_gridspec(1, 1)[0, 0],
        data,
        ScatterMarginalsOptions(diff_inset_count_ylim=80.0, **shared_kw),
    )
    capped_h = max(_inset_bar_heights(fig_capped.axes[0]))
    plt.close(fig_capped)

    assert auto_h == pytest.approx(4.0, rel=1e-6)
    assert capped_h == pytest.approx(1.0, rel=1e-6)


def test_draw_scatter_marginals_into_creates_axes() -> None:
    """Synthetic data renders without error and leaves child axes on the figure."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 20.0, size=24)
    y = x + rng.normal(0.0, 2.0, size=24)
    data = ScatterMarginalsData(x=x, y=y)
    options = ScatterMarginalsOptions(
        marginal_hist_bins=8,
        diff_hist_bins=6,
        point_marker="o",
        point_size=12.0,
        point_facecolor="none",
        point_edgecolor="0.2",
    )
    fig = plt.figure(figsize=(3.0, 3.0))
    cell = fig.add_gridspec(1, 1)[0, 0]
    draw_scatter_marginals_into(fig, cell, data, options)
    assert len(fig.axes) >= 3
    scatter_ax = fig.axes[0]
    assert not any(isinstance(p, Polygon) for p in scatter_ax.patches)
    plt.close(fig)


def test_draw_scatter_marginals_fixed_limits() -> None:
    """Explicit scatter_lim is applied to the scatter axis."""
    data = ScatterMarginalsData(x=np.array([5.0]), y=np.array([15.0]))
    options = ScatterMarginalsOptions(scatter_lim=(0.0, 25.0))
    fig = plt.figure(figsize=(3.0, 3.0))
    cell = fig.add_gridspec(1, 1)[0, 0]
    draw_scatter_marginals_into(fig, cell, data, options)
    scatter_ax = fig.axes[0]
    assert scatter_ax.get_xlim() == pytest.approx((0.0, 25.0))
    assert scatter_ax.get_ylim() == pytest.approx((0.0, 25.0))
    plt.close(fig)


def test_diff_inset_xtick_labels() -> None:
    """Endpoint difference labels appear when the option is enabled."""
    x = np.array([0.0, 5.0, 10.0])
    y = np.array([2.0, 5.0, 8.0])
    data = ScatterMarginalsData(x=x, y=y)
    options = ScatterMarginalsOptions(
        diff_inset_center=10.0,
        diff_inset_d_half_extent=10.0,
        diff_inset_xtick_labels=True,
        diff_hist_bins=4,
    )
    fig = plt.figure(figsize=(3.0, 3.0))
    cell = fig.add_gridspec(1, 1)[0, 0]
    draw_scatter_marginals_into(fig, cell, data, options)
    scatter_ax = fig.axes[0]
    labels = {t.get_text() for t in scatter_ax.texts}
    assert "-10" in labels
    assert "10" in labels
    plt.close(fig)


def test_draw_scatter_marginals_inset_with_center() -> None:
    """Inset polygons and axis appear when diff_inset_center is set."""
    x = np.array([0.0, 5.0, 10.0])
    y = np.array([2.0, 5.0, 8.0])
    data = ScatterMarginalsData(x=x, y=y)
    options = ScatterMarginalsOptions(
        diff_inset_center=10.0,
        diff_inset_d_half_extent=4.0,
        diff_hist_bins=4,
    )
    fig = plt.figure(figsize=(3.0, 3.0))
    cell = fig.add_gridspec(1, 1)[0, 0]
    draw_scatter_marginals_into(fig, cell, data, options)
    scatter_ax = fig.axes[0]
    polys = [p for p in scatter_ax.patches if isinstance(p, Polygon)]
    assert len(polys) >= 2
    assert len(scatter_ax.lines) >= 1
    for poly in polys:
        assert poly.get_transform() is scatter_ax.transData
    plt.close(fig)
