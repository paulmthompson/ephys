"""Tests for :mod:`processing.spike_align`."""

from __future__ import annotations

import numpy as np

from ephys.processing.spike_align import (
    apply_trial_order,
    event_ticks_greedy_non_overlapping_half_windows,
    sort_order_by_spike_count_descending,
    spikes_relative_to_events_ticks,
)


def test_dense_onsets_only_earliest_kept() -> None:
    """ISI below ``2 * win_ticks`` keeps one onset per greedy chain."""
    win_ticks = 500
    onset_ticks = np.array([0, 100, 200], dtype=np.int64)
    kept = event_ticks_greedy_non_overlapping_half_windows(
        onset_ticks,
        win_ticks,
    )
    assert kept.shape == (1,)
    assert int(kept[0]) == 0


def test_spaced_onsets_keep_all() -> None:
    """When ``t[i+1] - t[i] >= 2 * win_ticks``, greedy keeps every onset."""
    win_ticks = 500
    onset_ticks = np.array([0, 2000, 4000], dtype=np.int64)
    kept = event_ticks_greedy_non_overlapping_half_windows(
        onset_ticks,
        win_ticks,
    )
    np.testing.assert_array_equal(kept, onset_ticks)


def test_kept_trials_use_full_symmetric_window() -> None:
    """One kept onset still gets the full ``±win_ticks`` spike window."""
    fs = 30_000.0
    win_ticks = 500
    onset_ticks = np.array([0, 100, 200], dtype=np.int64)
    spike_ticks = np.array([-200, 0, 150, 400], dtype=np.int64)

    kept = event_ticks_greedy_non_overlapping_half_windows(
        onset_ticks,
        win_ticks,
    )
    rel = spikes_relative_to_events_ticks(
        spike_ticks,
        kept,
        win_ticks,
        fs,
    )
    assert len(rel) == len(kept) == 1
    # All spike ticks in [-500, 500) relative to onset 0.
    assert rel[0].size == 4


def test_minimum_separation_is_twice_half_window() -> None:
    """Kept centers are at least ``2 * win_ticks`` apart (sorted output)."""
    win_ticks = 100
    onset_ticks = np.array([0, 150, 500, 650, 1200], dtype=np.int64)
    kept = event_ticks_greedy_non_overlapping_half_windows(
        onset_ticks,
        win_ticks,
    )
    assert kept.size >= 2
    for i in range(1, int(kept.size)):
        assert int(kept[i]) - int(kept[i - 1]) >= 2 * win_ticks


def test_empty_events_returns_empty_array() -> None:
    """Empty candidate list returns an empty ``int64`` array."""
    out = event_ticks_greedy_non_overlapping_half_windows(
        np.array([], dtype=np.int64),
        10,
    )
    assert out.size == 0
    assert out.dtype == np.dtype(np.int64)


def test_sort_order_by_spike_count_descending() -> None:
    """Highest spike count first (bottom raster rows after reorder)."""
    rel = [
        np.array([0.0]),
        np.array([0.0, 0.1, 0.2]),
        np.array([], dtype=np.float64),
    ]
    order = sort_order_by_spike_count_descending(rel)
    assert list(order) == [1, 0, 2]
    stacked = apply_trial_order(rel, order)
    assert len(stacked[0]) == 3
    assert len(stacked[1]) == 1
    assert len(stacked[2]) == 0
