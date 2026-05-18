"""Tests for tick-interval spike counting helpers."""

from __future__ import annotations

import numpy as np
import pytest

from data_wrangling.spike_times import (
    count_spikes_in_tick_interval,
    count_spikes_in_tick_intervals,
    firing_rate_hz_from_interval_count,
)


def test_count_spikes_inclusive_closed_window() -> None:
    """Inclusive bounds count endpoints."""
    spikes = np.array([0, 5, 5, 10], dtype=np.int64)
    assert count_spikes_in_tick_interval(spikes, 5, 10, inclusive=True) == 3


def test_count_spikes_half_open() -> None:
    """Half-open interval excludes ``window_end_tick``."""
    spikes = np.array([0, 5, 10], dtype=np.int64)
    assert count_spikes_in_tick_interval(spikes, 5, 10, inclusive=False) == 1


def test_count_spikes_unsorted_input() -> None:
    """Spikes are sorted internally before counting."""
    spikes = np.array([20, 5, 15], dtype=np.int64)
    assert count_spikes_in_tick_interval(spikes, 10, 20, inclusive=True) == 2


def test_count_spikes_invalid_window_raises() -> None:
    """End before start raises ``ValueError``."""
    with pytest.raises(ValueError, match="window_end_tick"):
        count_spikes_in_tick_interval(np.array([1]), 10, 5)


def test_count_spikes_in_tick_intervals_vectorized() -> None:
    """Batch counts match per-interval scalar counts."""
    spikes = np.arange(0, 100, 3, dtype=np.int64)
    on = np.array([0, 10], dtype=np.int64)
    off = np.array([9, 30], dtype=np.int64)
    batched = count_spikes_in_tick_intervals(spikes, on, off)
    assert batched.shape == (2,)
    for i in range(2):
        assert int(batched[i]) == count_spikes_in_tick_interval(
            spikes,
            int(on[i]),
            int(off[i]),
        )


def test_firing_rate_hz_from_interval_count() -> None:
    """Hz = count * fs / duration_ticks."""
    hz = firing_rate_hz_from_interval_count(15, 30000, 30000.0)
    assert hz == pytest.approx(15.0)
