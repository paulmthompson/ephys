"""Tests for log-log scatter plotting."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib import pyplot as plt

from ephys.plotting.log_log_scatter import (
    LogLogScatterOptions,
    draw_log_log_scatter_into,
    positive_finite_mask,
    resolve_log_log_axis_limits,
)
from ephys.plotting.scatter_marginals import ScatterMarginalsData


def test_positive_finite_mask() -> None:
    """Only strictly positive finite pairs pass the mask."""
    x = np.array([0.0, 1.0, 2.0, np.nan])
    y = np.array([1.0, 0.0, 3.0, 1.0])
    mask = positive_finite_mask(x, y)
    assert mask.tolist() == [False, False, True, False]


def test_resolve_log_log_axis_limits_shared() -> None:
    """scatter_lim applies the same bounds to both axes."""
    x = np.array([1.0, 10.0])
    y = np.array([2.0, 20.0])
    lo_x, hi_x, lo_y, hi_y = resolve_log_log_axis_limits(
        x,
        y,
        scatter_lim=(0.5, 50.0),
        xlim=None,
        ylim=None,
    )
    assert (lo_x, hi_x, lo_y, hi_y) == (0.5, 50.0, 0.5, 50.0)


def test_log_log_scatter_options_rejects_nonpositive_lim() -> None:
    """Explicit limits must be strictly positive."""
    with pytest.raises(ValueError, match="strictly positive"):
        LogLogScatterOptions(scatter_lim=(0.0, 10.0))


def test_draw_log_log_scatter_into_creates_axes() -> None:
    """Renderer produces a log-scaled scatter axis with a unity line."""
    data = ScatterMarginalsData(
        x=np.array([1.0, 2.0, 5.0]),
        y=np.array([1.5, 3.0, 4.0]),
    )
    options = LogLogScatterOptions(scatter_lim=(0.5, 10.0))
    fig = plt.figure(figsize=(3.0, 3.0))
    cell = fig.add_gridspec(1, 1)[0, 0]
    draw_log_log_scatter_into(fig, cell, data, options)
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    assert len(ax.lines) == 1
    assert ax.lines[0].get_linestyle() == "--"
    plt.close(fig)


def test_draw_log_log_scatter_into_placeholder_when_no_positive() -> None:
    """Non-positive samples show a placeholder instead of an empty axis."""
    data = ScatterMarginalsData(x=np.array([0.0, 0.0]), y=np.array([1.0, 2.0]))
    fig = plt.figure(figsize=(3.0, 3.0))
    cell = fig.add_gridspec(1, 1)[0, 0]
    draw_log_log_scatter_into(fig, cell, data, LogLogScatterOptions())
    ax = fig.axes[0]
    assert not ax.axison
    plt.close(fig)
