"""Tests for modulation histogram plotting."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib import pyplot as plt

from ephys.plotting.modulation_histogram import (
    ModulationHistogramOptions,
    draw_modulation_histogram_into,
    histogram_edges,
    log2_ratio,
    log2_ratio_from_paired,
    resolve_histogram_x_limits,
)


def test_log2_ratio_orders_relative_modulation() -> None:
    """Larger relative change yields a larger log₂ ratio than a small one."""
    small = float(log2_ratio(np.array([54.0]), np.array([50.0]))[0])
    large = float(log2_ratio(np.array([14.0]), np.array([10.0]))[0])
    assert large > small
    assert np.isclose(large, np.log2(1.4))
    assert np.isclose(small, np.log2(1.08))


def test_log2_ratio_from_paired_skips_nonpositive() -> None:
    """Non-positive or non-finite pairs are excluded."""
    x = np.array([10.0, 0.0, 5.0, np.nan])
    y = np.array([14.0, 5.0, 0.0, 8.0])
    ratios = log2_ratio_from_paired(x, y)
    assert ratios is not None
    assert ratios.size == 1
    assert np.isclose(float(ratios[0]), np.log2(1.4))


def test_log2_ratio_from_paired_none_when_empty() -> None:
    """All-invalid input returns None."""
    x = np.array([0.0, -1.0])
    y = np.array([1.0, 2.0])
    assert log2_ratio_from_paired(x, y) is None


def test_modulation_histogram_options_rejects_invalid_xlim() -> None:
    """Explicit x limits must have max > min."""
    with pytest.raises(ValueError, match="xlim max"):
        ModulationHistogramOptions(xlim=(2.0, 1.0))


def test_resolve_histogram_x_limits_explicit() -> None:
    """Explicit xlim is returned unchanged."""
    lo, hi = resolve_histogram_x_limits(np.array([0.0, 1.0]), xlim=(-2.0, 2.0))
    assert (lo, hi) == (-2.0, 2.0)


def test_histogram_edges_count() -> None:
    """Edge array length is n_bins + 1."""
    edges = histogram_edges(-2.0, 2.0, 10)
    assert edges.size == 11
    assert edges[0] == -2.0
    assert edges[-1] == 2.0


def test_draw_modulation_histogram_into_creates_axes() -> None:
    """Renderer produces a linear-x histogram with optional zero line."""
    values = np.array([-1.0, 0.0, 0.5, 1.0, 1.5])
    options = ModulationHistogramOptions(xlim=(-2.0, 2.0), show_zero_line=True)
    fig = plt.figure(figsize=(3.0, 2.0))
    cell = fig.add_gridspec(1, 1)[0, 0]
    draw_modulation_histogram_into(fig, cell, values, options)
    ax = fig.axes[0]
    assert ax.get_xscale() == "linear"
    assert len(ax.lines) == 1
    assert ax.lines[0].get_linestyle() == "--"
    assert ax.lines[0].get_xdata()[0] == 0.0
    plt.close(fig)


def test_draw_modulation_histogram_into_placeholder_when_empty() -> None:
    """Non-finite values show a placeholder instead of an empty axis."""
    fig = plt.figure(figsize=(3.0, 2.0))
    cell = fig.add_gridspec(1, 1)[0, 0]
    draw_modulation_histogram_into(
        fig,
        cell,
        np.array([np.nan, np.inf]),
        ModulationHistogramOptions(),
    )
    ax = fig.axes[0]
    assert not ax.axison
    plt.close(fig)
