"""Tests for scatter-with-marginals plotting."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt

from ephys.plotting.scatter_marginals import (
    ScatterMarginalsData,
    ScatterMarginalsOptions,
    draw_scatter_marginals_into,
)


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
    plt.close(fig)
