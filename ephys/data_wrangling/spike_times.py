"""Spike time utilities for event- and interval-based counting."""

from __future__ import annotations

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
        ``onset[k]`` and ``offset[k]`` like
        :func:`count_spikes_in_tick_interval`.
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
    on = np.asarray(interval_onset_ticks, dtype=int).ravel()
    off = np.asarray(interval_offset_ticks, dtype=int).ravel()
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
                f"Each interval requires offset >= onset; index {i}: "
                f"onset={on[i]}, offset={off[i]}"
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
