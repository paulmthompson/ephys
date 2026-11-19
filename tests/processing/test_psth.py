"""Tests for population PSTH summaries and peak-latency ordering."""

from __future__ import annotations

import numpy as np
import pytest

from ephys.processing.bins import UniformBins
from ephys.processing.psth import (
    burst_trial_fraction_within_onset_window,
    mean_psth_from_relative_spikes,
    peak_bin_center_times_from_mean_psth,
    peak_time_row_order,
)


def test_mean_psth_and_peak_ordering_with_uniform_bins() -> None:
    """Population PSTH helpers accept UniformBins and sort stably by peak."""
    bins = UniformBins.from_width(0.0, 0.03, 0.01)
    trials = [np.array([0.022]), np.array([0.025])]
    mean_counts = mean_psth_from_relative_spikes(trials, bins)
    assert mean_counts is not None
    assert mean_counts.tolist() == pytest.approx([0.0, 0.0, 1.0])

    rows = np.array(
        [
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ],
        dtype=np.float64,
    )
    peaks = peak_bin_center_times_from_mean_psth(rows, bins)
    assert peaks.tolist() == pytest.approx([0.025, 0.005, 0.015])
    assert peak_time_row_order(rows, bins).tolist() == [1, 2, 0]

    with pytest.raises(ValueError, match="inconsistent"):
        peak_bin_center_times_from_mean_psth(rows[:, :2], bins)


def test_burst_trial_fraction_within_onset_window() -> None:
    """Burst fraction counts trials with strictly >1 spike inside window."""
    assert burst_trial_fraction_within_onset_window([]) is None
    trials = [
        np.array([0.001, 0.004]),
        np.array([0.002]),
        np.array([0.0, 0.005, 0.010]),
    ]
    assert burst_trial_fraction_within_onset_window(trials) == pytest.approx(
        2.0 / 3.0
    )
