"""Tests for :mod:`ephys.processing.spike_rate`."""

from __future__ import annotations

import numpy as np
import pytest

from ephys.processing.spike_rate import (
    mean_and_sem,
    per_trial_firing_rates_hz_in_window,
)


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


def test_mean_and_sem_filters_non_finite_and_handles_small_samples() -> None:
    """Mean and SEM ignore NaN/Inf and handle 0- or 1-sample inputs."""
    mean_hz, sem_hz = mean_and_sem(
        np.array([10.0, 20.0, np.nan, np.inf], dtype=np.float64)
    )
    assert mean_hz == pytest.approx(15.0)
    assert sem_hz == pytest.approx(5.0)

    assert mean_and_sem([5.0]) == (5.0, 0.0)

    m_empty, s_empty = mean_and_sem([float("nan")])
    assert np.isnan(m_empty) and np.isnan(s_empty)
