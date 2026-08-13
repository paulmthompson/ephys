"""Tests for scatter-with-marginals layout helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib import pyplot as plt

from ephys.plotting.scatter_marginals import (
    ScatterMarginalsData,
    ScatterMarginalsOptions,
    align_marginal_axes_to_scatter,
    draw_scatter_marginals_into,
    marginal_gridspec_ratios,
    resolve_diff_inset_diagonal_pos,
    should_draw_diff_inset,
)


def test_marginal_gridspec_ratios_square_limits() -> None:
    """Equal spans leave nominal marginal ratios unchanged."""
    top_h, scatter_h, scatter_w, right_w = marginal_gridspec_ratios(
        10.0,
        10.0,
        marginal_height_ratio=0.22,
        marginal_width_ratio=0.22,
    )
    assert top_h == pytest.approx(0.22)
    assert scatter_h == pytest.approx(1.0)
    assert scatter_w == pytest.approx(1.0)
    assert right_w == pytest.approx(0.22)


def test_marginal_gridspec_ratios_tall_scatter_shrinks_top_row() -> None:
    """A narrower x span shrinks the top marginal row fraction."""
    top_h, _, _, right_w = marginal_gridspec_ratios(
        5.0,
        20.0,
        marginal_height_ratio=0.22,
        marginal_width_ratio=0.22,
    )
    assert top_h == pytest.approx(0.22 * 0.25)
    assert right_w == pytest.approx(0.22 / 0.25)


def test_marginal_gridspec_ratios_rejects_non_positive_span() -> None:
    """Zero spans raise ValueError."""
    with pytest.raises(ValueError, match="positive"):
        marginal_gridspec_ratios(
            0.0,
            10.0,
            marginal_height_ratio=0.22,
            marginal_width_ratio=0.22,
        )


def test_should_draw_diff_inset_enabled_flag() -> None:
    """Explicit enable draws inset without a center."""
    opts = ScatterMarginalsOptions(diff_inset_enabled=True)
    assert should_draw_diff_inset(opts)


def test_should_draw_diff_inset_legacy_center() -> None:
    """Explicit center still enables inset for backward compatibility."""
    opts = ScatterMarginalsOptions(diff_inset_center=50.0)
    assert should_draw_diff_inset(opts)


def test_should_draw_diff_inset_default_off() -> None:
    """Default options do not draw the inset."""
    opts = ScatterMarginalsOptions()
    assert not should_draw_diff_inset(opts)


def test_resolve_diff_inset_diagonal_pos_explicit_center() -> None:
    """Explicit center overrides auto midpoint."""
    opts = ScatterMarginalsOptions(diff_inset_center=12.5)
    pos = resolve_diff_inset_diagonal_pos(opts, 0.0, 20.0, 0.0, 30.0)
    assert pos == pytest.approx(12.5)


def test_resolve_diff_inset_diagonal_pos_auto_center() -> None:
    """Enabled inset without center uses unity-line midpoint."""
    opts = ScatterMarginalsOptions(diff_inset_enabled=True)
    pos = resolve_diff_inset_diagonal_pos(opts, 0.0, 20.0, 0.0, 30.0)
    assert pos == pytest.approx(10.0)


def test_draw_scatter_marginals_top_x_extent_matches_scatter() -> None:
    """Top marginal x bbox matches scatter after aspect-equal alignment."""
    fig = plt.figure(figsize=(4.0, 1.2))
    gs = fig.add_gridspec(1, 1)
    x = np.linspace(2.0, 8.0, 20, dtype=np.float64)
    y = np.linspace(1.0, 25.0, 20, dtype=np.float64)
    data = ScatterMarginalsData(x=x, y=y)
    draw_scatter_marginals_into(
        fig,
        gs[0, 0],
        data,
        ScatterMarginalsOptions(diff_inset_enabled=False),
    )
    fig.canvas.draw()
    scatter_ax = fig.axes[0]
    top_ax = fig.axes[1]
    scatter_pos = scatter_ax.get_position()
    top_pos = top_ax.get_position()
    assert top_pos.x0 == pytest.approx(scatter_pos.x0, abs=0.01)
    assert top_pos.width == pytest.approx(scatter_pos.width, abs=0.01)


def test_align_marginal_axes_to_scatter_matches_y_extent() -> None:
    """Right marginal y bbox matches scatter after explicit alignment."""
    fig, axes = plt.subplots(1, 3, figsize=(4.0, 1.2))
    scatter_ax, top_ax, right_ax = axes
    scatter_ax.set_xlim(0.0, 5.0)
    scatter_ax.set_ylim(0.0, 20.0)
    scatter_ax.set_aspect("equal", adjustable="box")
    align_marginal_axes_to_scatter(scatter_ax, top_ax, right_ax)
    fig.canvas.draw()
    scatter_pos = scatter_ax.get_position()
    right_pos = right_ax.get_position()
    assert right_pos.y0 == pytest.approx(scatter_pos.y0, abs=0.01)
    assert right_pos.height == pytest.approx(scatter_pos.height, abs=0.01)
    assert right_pos.x0 == pytest.approx(scatter_pos.x0 + scatter_pos.width, abs=0.01)
    assert right_pos.width == pytest.approx(top_ax.get_position().height, abs=0.01)
    plt.close(fig)


def test_draw_scatter_marginals_explicit_ticks() -> None:
    """Explicit scatter_ticks set both axis tick positions."""
    fig = plt.figure(figsize=(2.0, 2.0))
    gs = fig.add_gridspec(1, 1)
    data = ScatterMarginalsData(
        x=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        y=np.array([2.0, 3.0, 4.0], dtype=np.float64),
    )
    draw_scatter_marginals_into(
        fig,
        gs[0, 0],
        data,
        ScatterMarginalsOptions(
            scatter_lim=(0.0, 8.0),
            scatter_ticks=(0.0, 8.0),
            diff_inset_enabled=False,
        ),
    )
    scatter_ax = fig.axes[0]
    assert list(scatter_ax.get_xticks()) == pytest.approx([0.0, 8.0])
    assert list(scatter_ax.get_yticks()) == pytest.approx([0.0, 8.0])
    plt.close(fig)


def test_draw_scatter_marginals_right_x_extent_matches_scatter() -> None:
    """Right marginal x0 sits flush with the scatter right edge."""
    fig = plt.figure(figsize=(4.0, 1.2))
    gs = fig.add_gridspec(1, 1)
    x = np.linspace(2.0, 8.0, 20, dtype=np.float64)
    y = np.linspace(1.0, 25.0, 20, dtype=np.float64)
    data = ScatterMarginalsData(x=x, y=y)
    draw_scatter_marginals_into(
        fig,
        gs[0, 0],
        data,
        ScatterMarginalsOptions(diff_inset_enabled=False),
    )
    fig.canvas.draw()
    scatter_ax = fig.axes[0]
    right_ax = fig.axes[2]
    scatter_pos = scatter_ax.get_position()
    right_pos = right_ax.get_position()
    assert right_pos.x0 == pytest.approx(scatter_pos.x0 + scatter_pos.width, abs=0.01)
    assert right_pos.y0 == pytest.approx(scatter_pos.y0, abs=0.01)
    assert right_pos.height == pytest.approx(scatter_pos.height, abs=0.01)
    plt.close(fig)


def test_draw_scatter_marginals_into_adds_diff_inset_patches() -> None:
    """Enabled inset draws polygon patches on the scatter axis."""
    fig = plt.figure(figsize=(2.0, 2.0))
    gs = fig.add_gridspec(1, 1)
    data = ScatterMarginalsData(
        x=np.array([4.0, 5.0, 6.0, 8.0], dtype=np.float64),
        y=np.array([5.0, 4.0, 7.0, 9.0], dtype=np.float64),
    )
    draw_scatter_marginals_into(
        fig,
        gs[0, 0],
        data,
        ScatterMarginalsOptions(
            diff_inset_enabled=True,
            diff_inset_d_half_extent=5.0,
            scatter_lim=(0.0, 10.0),
        ),
    )
    scatter_ax = fig.axes[0]
    assert len(scatter_ax.patches) > 0
    plt.close(fig)
