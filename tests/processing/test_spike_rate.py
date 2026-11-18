"""Tests for :mod:`ephys.processing.spike_rate`."""

from __future__ import annotations

import numpy as np
import pytest

from ephys.processing.spike_rate import per_trial_firing_rates_hz_in_window


def test_per_trial_firing_rates_hz_in_window() -> None:
    """Inclusive window bounds count boundary spikes and normalize by duration."""
    trials = [
        np.array([-0.001, 0.0, 0.0025, 0.005, 0.006], dtype=np.float64),
        np.array([], dtype=np.float64),
    ]
    rates = per_trial_firing_rates_hz_in_window(
        trials,
        window_start_s=0.0,
        window_end_s=0.005,
    )
    assert rates.dtype == np.dtype(np.float64)
    assert rates.tolist() == pytest.approx([3.0 / 0.005, 0.0])


def test_per_trial_firing_rates_hz_in_window_validation() -> None:
    """Reject non-finite or non-positive window durations."""
    trials = [np.array([0.001], dtype=np.float64)]
    with pytest.raises(ValueError, match="positive duration"):
        per_trial_firing_rates_hz_in_window(
            trials,
            window_start_s=0.005,
            window_end_s=0.005,
        )
    with pytest.raises(ValueError, match="finite"):
        per_trial_firing_rates_hz_in_window(
            trials,
            window_start_s=0.0,
            window_end_s=float("nan"),
        )
