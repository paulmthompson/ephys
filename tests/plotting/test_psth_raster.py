"""Tests for PSTH step rendering with UniformBins and ax.stairs."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.patches import StepPatch

from ephys.plotting.psth_raster import plot_psth
from ephys.processing.bins import UniformBins


def test_plot_psth_stairs_spans_final_bin_and_limits() -> None:
    """ax.stairs renders all n_bins segments up to bins.stop without truncation."""
    bins = UniformBins.from_width(-0.01, 0.02, 0.01)
    hist = np.array([0.1, 0.4, 0.8], dtype=np.float64)
    fig, ax = plt.subplots()
    plot_psth(ax, hist, bins, max_y=1.0)

    assert ax.get_xlim() == pytest.approx((-0.01, 0.02))
    step_patches = [p for p in ax.patches if isinstance(p, StepPatch)]
    assert len(step_patches) == 1
    verts_x = np.asarray(
        step_patches[0].get_path().vertices,
        dtype=np.float64,
    )[:, 0]
    assert np.min(verts_x) == pytest.approx(-0.01)
    assert np.max(verts_x) == pytest.approx(0.02)
    plt.close(fig)


def test_plot_psth_rejects_mismatched_hist_length() -> None:
    """Mismatched hist and bin count raise ValueError rather than slicing."""
    bins = UniformBins.from_width(0.0, 0.03, 0.01)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="must match bins.n_bins"):
        plot_psth(ax, np.array([0.2, 0.5]), bins, max_y=1.0)
    plt.close(fig)
