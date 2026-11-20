"""Tests for :mod:`ephys.plotting.forest`."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ephys.plotting.forest import ForestHighlightSpec, plot_forest


def test_plot_forest_highlight_targets_sorted_row_index() -> None:
    """Highlighted marker sits at the sorted y-position for the target session."""
    posteriors = {
        "animal": ["M001", "M001", "M002", "M002", "M054", "M054"],
        "session": ["0101_1", "0101_1", "0202_1", "0202_1", "0427_1", "0427_1"],
        "beta_distance": [-0.2, 0.1, 0.4, 0.6, -0.05, 0.15],
    }
    fig, ax = plt.subplots()
    plot_forest(
        ax,
        posteriors,
        "beta_distance",
        "Pole Distance",
        highlight=ForestHighlightSpec(animal="M054", session="0427_1"),
    )
    highlight_lines = [
        artist
        for artist in ax.lines
        if artist.get_marker() == "o" and artist.get_markerfacecolor() == "white"
    ]
    assert len(highlight_lines) == 1
    ydata = np.asarray(highlight_lines[0].get_ydata(), dtype=np.float64)
    assert float(ydata[0]) == 1.0
    plt.close(fig)


def test_plot_forest_validates_columns_and_ignores_nan_samples() -> None:
    """Missing animal/session columns raise; NaN draws are skipped cleanly."""
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="animal"):
        plot_forest(ax, {"beta": [0.1, 0.2]}, "beta", "Bad")

    plot_forest(
        ax,
        {
            "animal": ["M001", "M001"],
            "session": ["0101_1", "0101_1"],
            "beta": [float("nan"), 0.5],
        },
        "beta",
        "With NaN",
    )
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["M001/0101_1"]
    plt.close(fig)
