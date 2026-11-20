"""Tests for :mod:`processing.spike_align`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ephys.processing.bins import UniformBins
from ephys.processing.spike_align import (
    apply_trial_order,
    bin_centers,
    collect_first_spike_jitter_ms,
    event_ticks_greedy_non_overlapping_half_windows,
    first_spike_time_ms,
    mean_psth_rate_hz,
    ms_to_ticks,
    per_trial_bin_counts,
    seconds_to_ticks,
    sort_order_by_covariate,
    sort_order_by_spike_count_descending,
    sort_order_by_spike_count_then_first_spike,
    spikes_relative_to_events_ticks,
    spikes_relative_to_tick_intervals,
    validate_tick_intervals,
    window_seconds_to_ticks,
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


def test_ms_to_ticks_and_seconds_to_ticks() -> None:
    """Scalar ms and seconds conversions support ceil and round modes."""
    assert ms_to_ticks(2.0, 30_000.0) == 60
    assert ms_to_ticks(0.05, 30_000.0, rounding="ceil") == 2
    assert ms_to_ticks(0.05, 30_000.0, rounding="round") == 2
    assert ms_to_ticks(0.04, 30_000.0, rounding="round") == 1
    assert seconds_to_ticks(0.007, 30_000.0) == 210
    assert seconds_to_ticks(0.0001, 30_000.0, rounding="ceil") == 3
    assert seconds_to_ticks(0.00004, 30_000.0, rounding="ceil") == 2

    with pytest.raises(ValueError, match="sampling_rate_hz"):
        ms_to_ticks(2.0, 0.0)
    with pytest.raises(ValueError, match="seconds must be finite"):
        seconds_to_ticks(float("nan"), 30_000.0)
    bad_mode: Any = "floor"
    with pytest.raises(ValueError, match="Unsupported rounding mode"):
        ms_to_ticks(2.0, 30_000.0, rounding=bad_mode)

    assert window_seconds_to_ticks(1.0, 1.07, 30_000.0) == (30_000, 32_100)
    with pytest.raises(ValueError, match="end_s"):
        window_seconds_to_ticks(2.0, 1.5, 30_000.0)


def test_spikes_relative_to_tick_intervals_bounds_and_validation() -> None:
    """Interval slicing supports inclusive/exclusive offsets and validates shape."""
    spikes = np.array([250, 100, 150, 200, 300], dtype=np.int64)
    onsets = np.array([100, 300], dtype=np.int64)
    offsets = np.array([200, 300], dtype=np.int64)

    incl = spikes_relative_to_tick_intervals(spikes, onsets, offsets, 1000.0)
    assert len(incl) == 2
    assert incl[0].tolist() == pytest.approx([0.0, 0.05, 0.1])
    assert incl[1].size == 0

    excl = spikes_relative_to_tick_intervals(
        spikes,
        onsets,
        offsets,
        1000.0,
        inclusive_offset=False,
    )
    assert excl[0].tolist() == pytest.approx([0.0, 0.05])

    with pytest.raises(ValueError, match="same length"):
        spikes_relative_to_tick_intervals(
            spikes,
            np.array([100, 200], dtype=np.int64),
            np.array([150], dtype=np.int64),
            1000.0,
        )
    with pytest.raises(ValueError, match="after onsets"):
        validate_tick_intervals(onsets, offsets, require_positive_duration=True)


def test_bin_centers_and_first_spike_jitter_ms() -> None:
    """Bin centers and median-centered first-spike jitter handle edge cases."""
    assert bin_centers(np.array([0.0, 1.0, 3.0])).tolist() == pytest.approx([0.5, 2.0])
    with pytest.raises(ValueError, match="at least 2"):
        bin_centers(np.array([0.0]))

    assert np.isnan(first_spike_time_ms(np.array([], dtype=np.float64)))
    assert first_spike_time_ms(np.array([0.025, 0.010, float("nan")])) == pytest.approx(10.0)

    trials = [
        np.array([0.010, 0.020]),
        np.array([], dtype=np.float64),
        np.array([0.012]),
        np.array([0.020]),
    ]
    jitter = collect_first_spike_jitter_ms(trials)
    assert jitter is not None
    assert jitter.tolist() == pytest.approx([-2.0, 0.0, 8.0])
    assert collect_first_spike_jitter_ms([np.array([], dtype=np.float64)]) is None


def test_per_trial_bin_counts_and_mean_psth_rate_hz_with_uniform_bins() -> None:
    """UniformBins integrates directly with per_trial_bin_counts and rate_hz."""
    bins = UniformBins.from_width(0.0, 0.02, 0.01)
    trials = [np.array([0.005, 0.020]), np.array([0.015])]
    counts = per_trial_bin_counts(trials, bins)
    assert [c.tolist() for c in counts] == [[1.0, 1.0], [0.0, 1.0]]
    assert bin_centers(bins).tolist() == pytest.approx([0.005, 0.015])

    rates = mean_psth_rate_hz(trials, bins)
    assert rates is not None
    assert rates.tolist() == pytest.approx([50.0, 100.0])
    assert mean_psth_rate_hz([], bins) is None

    # Large tick offsets subtract in int64 before dividing by fs_hz.
    rel_large = spikes_relative_to_events_ticks(
        np.array([9_999_998], dtype=np.int64),
        np.array([10_000_000], dtype=np.int64),
        10,
        1000.0,
    )
    assert rel_large[0][0] == -0.002


def test_raster_trial_sorting_by_count_first_spike_and_covariate() -> None:
    """Raster sort helpers break count ties by first spike and place NaN covariates last."""
    trials = [
        np.array([0.05, 0.06]),
        np.array([0.01, 0.02]),
        np.array([0.02, 0.03, 0.04]),
        np.array([0.03]),
        np.array([]),
    ]
    order = sort_order_by_spike_count_then_first_spike(trials)
    assert order.tolist() == [4, 3, 1, 0, 2]
    sorted_trials = apply_trial_order(trials, order)
    assert [len(t) for t in sorted_trials] == [0, 1, 2, 2, 3]

    unique_trials = [
        np.array([0.01, 0.02, 0.03]),
        np.array([0.04]),
        np.array([0.05, 0.06]),
    ]
    assert (
        sort_order_by_spike_count_then_first_spike(unique_trials).tolist()
        == sort_order_by_spike_count_descending(unique_trials)[::-1].tolist()
    )

    asc = sort_order_by_covariate(np.array([3.0, np.nan, 1.0, 2.0, np.nan]), ascending=True)
    assert asc.tolist() == [2, 3, 0, 1, 4]
    desc = sort_order_by_covariate(np.array([1.0, np.nan, 3.0, 2.0]), ascending=False)
    assert desc.tolist() == [2, 3, 0, 1]
    ties = sort_order_by_covariate(np.array([2.0, 2.0, 1.0]), ascending=True)
    assert ties.tolist() == [2, 0, 1]
