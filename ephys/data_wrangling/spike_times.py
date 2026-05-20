"""Spike time utilities for event- and interval-based slicing."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np


def count_spikes_in_tick_interval(
    spike_times_ticks: np.ndarray | Any,
    window_start_tick: int,
    window_end_tick: int,
    *,
    inclusive: bool = True,
) -> int:
    """Count spikes whose times fall in a DAQ tick window.

    Parameters
    ----------
    spike_times_ticks
        Spike sample indices in DAQ ticks (any integer dtype). Values are not
        required to be sorted; a sorted copy is used internally.
    window_start_tick, window_end_tick
        Window bounds in ticks. When ``inclusive`` is True (default), spikes
        with ``window_start_tick <= t <= window_end_tick`` are counted.
    inclusive
        If False, use half-open ``[start, end)`` (``end`` excluded).

    Returns
    -------
    int
        Number of spikes in the window.

    Raises
    ------
    ValueError
        If ``window_end_tick < window_start_tick``.

    Notes
    -----
    Empty ``spike_times_ticks`` yields zero without error.
    """
    if window_end_tick < window_start_tick:
        msg = (
            "window_end_tick must be >= window_start_tick; "
            f"got start={window_start_tick}, end={window_end_tick}"
        )
        raise ValueError(msg)
    spikes = np.asarray(spike_times_ticks, dtype=np.int64).ravel()
    if spikes.size == 0:
        return 0
    spikes = np.sort(spikes)
    if inclusive:
        lo = int(window_start_tick)
        hi = int(window_end_tick)
        left = np.searchsorted(spikes, lo, side="left")
        right = np.searchsorted(spikes, hi, side="right")
    else:
        lo = int(window_start_tick)
        hi = int(window_end_tick)
        left = np.searchsorted(spikes, lo, side="left")
        right = np.searchsorted(spikes, hi, side="left")
    return int(right - left)


def count_spikes_in_tick_intervals(
    spike_times_ticks: np.ndarray | Any,
    interval_onset_ticks: np.ndarray | Any,
    interval_offset_ticks: np.ndarray | Any,
    *,
    inclusive: bool = True,
) -> np.ndarray:
    """Count spikes in many closed (or half-open) tick windows at once.

    Parameters
    ----------
    spike_times_ticks
        Spike sample indices in DAQ ticks.
    interval_onset_ticks, interval_offset_ticks
        Same-length arrays of per-interval bounds. Each row ``k`` uses
        ``onset[k]`` and ``offset[k]`` like :func:`count_spikes_in_tick_interval`.
    inclusive
        Passed through to the same semantics as
        :func:`count_spikes_in_tick_interval`.

    Returns
    -------
    np.ndarray
        Integer counts, shape ``(n_intervals,)``.

    Raises
    ------
    ValueError
        If array lengths differ or any offset is before its onset.
    """
    on = np.asarray(interval_onset_ticks, dtype=np.int64).ravel()
    off = np.asarray(interval_offset_ticks, dtype=np.int64).ravel()
    if on.shape != off.shape:
        msg = (
            "interval_onset_ticks and interval_offset_ticks must have the "
            f"same shape; got {on.shape} vs {off.shape}"
        )
        raise ValueError(msg)
    spikes = np.asarray(spike_times_ticks, dtype=np.int64).ravel()
    if spikes.size == 0:
        return np.zeros(on.shape, dtype=np.int64)
    spikes = np.sort(spikes)
    counts = np.empty(on.shape, dtype=np.int64)
    for i in range(on.size):
        if off[i] < on[i]:
            msg = (
                f"Each interval requires offset >= onset; index {i}: onset={on[i]}, offset={off[i]}"
            )
            raise ValueError(msg)
        if inclusive:
            left = np.searchsorted(spikes, int(on[i]), side="left")
            right = np.searchsorted(spikes, int(off[i]), side="right")
        else:
            left = np.searchsorted(spikes, int(on[i]), side="left")
            right = np.searchsorted(spikes, int(off[i]), side="left")
        counts[i] = right - left
    return counts


def firing_rate_hz_from_interval_count(
    spike_count: int,
    duration_ticks: int,
    sampling_rate_hz: float,
) -> float:
    """Convert spike count and tick-span duration to a rate in Hz.

    Parameters
    ----------
    spike_count
        Non-negative spike count in the interval.
    duration_ticks
        Interval length in DAQ ticks (must be positive).
    sampling_rate_hz
        Samples per second (e.g. 30_000).

    Returns
    -------
    float
        ``spike_count * sampling_rate_hz / duration_ticks``.

    Raises
    ------
    ValueError
        If ``duration_ticks <= 0`` or ``spike_count < 0``.
    """
    if duration_ticks <= 0:
        msg = f"duration_ticks must be positive; got {duration_ticks}"
        raise ValueError(msg)
    if spike_count < 0:
        msg = f"spike_count must be non-negative; got {spike_count}"
        raise ValueError(msg)
    return float(spike_count) * float(sampling_rate_hz) / float(duration_ticks)


def get_spikes_at_events(
    spike_times_ticks,
    event_ticks,
    win_ticks,
    sampling_rate_hz=30000,
):
    """
    Find the spikes that occur within a window around events

    Parameters
    ----------
    spike_times_ticks: np.ndarray
    event_ticks: np.ndarray
    win_ticks: int
    sampling_rate_hz: int

    Returns
    -------
    list:

    """

    spikes_in_range_s = []

    for event_tick in event_ticks:
        event_lower_bound_tick = event_tick - win_ticks
        event_upper_bound_tick = event_tick + win_ticks

        spikes_in_range_ticks = np.take(
            spike_times_ticks,
            np.where(
                (event_lower_bound_tick < spike_times_ticks)
                & (spike_times_ticks < event_upper_bound_tick)
            ),
        )[0]

        spikes_in_range_s.append(
            spikes_in_range_ticks / sampling_rate_hz - event_tick / sampling_rate_hz
        )

    return spikes_in_range_s


def sort_by_spike_times(spike_times):
    """
    Given a list of spike times for each trial, return the indices of the trials
    sorted by the latency to the first spike time

    Parameters
    ----------
    spike_times: list
        spike times for each trial

    Returns
    -------
    list:
        indices of trials sorted by latency to first spike time

    """

    sorted_spike_times = []

    for i in range(0, len(spike_times)):
        # Get the first spike time that is greater than zero
        first_spike_time = np.where(spike_times[i] > 0)[0]
        if len(first_spike_time) > 0:
            sorted_spike_times.append(spike_times[i][first_spike_time[0]])
        else:
            sorted_spike_times.append(0.0)

    sorted_order = np.argsort(sorted_spike_times)
    return sorted_order


def remove_spike_times_after_event(spiketimes_s, event_s):
    """
    Remove spike times that occur after an event

    Parameters
    ----------
    spiketimes_s: list
        spike times per trial in seconds
    event_s: list
        event times per trial in seconds

    Returns
    -------
    list:
        copy of spiketimes_s with spike times after event removed

    """

    # Check that the number of trials is the same
    if len(spiketimes_s) != len(event_s):
        raise ValueError("The number of trials in spiketimes_s and event_s must be the same.")

    spiketimes_s_copy = copy.deepcopy(spiketimes_s)

    for i in range(0, len(spiketimes_s_copy)):
        spiketimes_s_copy[i] = spiketimes_s_copy[i][spiketimes_s_copy[i] < event_s[i]]

    return spiketimes_s_copy
