"""Unit tests for stimulus interval strip rendering in ``ephys.plotting.traces``."""

from __future__ import annotations

from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ephys.plotting.traces import (
    StimulusIntervalStyle,
    plot_stimulus_intervals,
)


def test_plot_stimulus_intervals_draws_spans_and_applies_style() -> None:
    """Intervals render with configured colors, stroke width, and x-limits."""
    fig, ax = plt.subplots(figsize=(2.0, 0.5))
    style = StimulusIntervalStyle(
        facecolor="#00a9ff",
        edgecolor="#112233",
        alpha=0.6,
        linewidth=1.25,
    )
    plot_stimulus_intervals(
        ax,
        [(0.1, 0.25), (0.4, 0.5)],
        (0.0, 1.0),
        style=style,
    )
    try:
        assert ax.get_xlim() == pytest.approx((0.0, 1.0))
        assert len(ax.patches) == 2
        p0: Any = ax.patches[0]
        assert p0.get_x() == pytest.approx(0.1)
        assert p0.get_width() == pytest.approx(0.15)
        assert p0.get_linewidth() == pytest.approx(1.25)
        assert p0.get_facecolor() == pytest.approx(
            mcolors.to_rgba("#00a9ff", alpha=0.6)
        )
        assert p0.get_edgecolor() == pytest.approx(
            mcolors.to_rgba("#112233", alpha=0.6)
        )
    finally:
        plt.close(fig)


def test_plot_stimulus_intervals_empty_and_validation() -> None:
    """Empty intervals set x-limits cleanly; malformed bounds raise ValueError."""
    fig, ax = plt.subplots(figsize=(2.0, 0.5))
    try:
        plot_stimulus_intervals(ax, [], (0.2, 0.8))
        assert ax.get_xlim() == pytest.approx((0.2, 0.8))
        assert len(ax.patches) == 0

        with pytest.raises(ValueError, match="xlim_s"):
            plot_stimulus_intervals(ax, [], (0.8, 0.2))

        with pytest.raises(ValueError, match="shape"):
            plot_stimulus_intervals(ax, np.array([0.1, 0.2, 0.3]), (0.0, 1.0))

        with pytest.raises(ValueError, match="offset_s >= onset_s"):
            plot_stimulus_intervals(ax, [(0.5, 0.2)], (0.0, 1.0))
    finally:
        plt.close(fig)
