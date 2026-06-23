"""Time series manipulation and resampling utilities."""

from __future__ import annotations

import numpy as np


def sample_hold(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    target_x: np.ndarray,
) -> np.ndarray:
    """Sample-and-hold interpolation onto a new time axis.

    Matches each target coordinate to the most recent preceding sample value
    (zero-order hold). If a target precedes the earliest sample, it is clamped
    to the first available sample value.

    Parameters
    ----------
    sample_x : np.ndarray
        Coordinates of the original samples. Must be monotonically increasing.
    sample_y : np.ndarray
        Values at the original coordinates.
    target_x : np.ndarray
        Coordinates to interpolate onto.

    Returns
    -------
    np.ndarray
        Interpolated values at the target coordinates.
    """
    idx = np.searchsorted(sample_x, target_x, side="right") - 1
    idx = np.clip(idx, 0, len(sample_x) - 1)
    return sample_y[idx]


def event_impulses(
    bin_start_ticks: np.ndarray,
    event_ticks: np.ndarray,
    dt_ticks: int,
) -> np.ndarray:
    """Bin discrete event times into an impulse train.

    Parameters
    ----------
    bin_start_ticks : np.ndarray
        Left edges of the target time bins.
    event_ticks : np.ndarray
        Discrete times of the events.
    dt_ticks : int
        Width of each time bin.

    Returns
    -------
    np.ndarray
        Array of the same length as `bin_start_ticks`, containing the count
        of events that fall within each bin `[start, start + dt)`.
    """
    out = np.zeros(len(bin_start_ticks), dtype=float)
    for event_tick in event_ticks:
        event = (bin_start_ticks <= int(event_tick)) & (
            int(event_tick) < bin_start_ticks + int(dt_ticks)
        )
        out += event.astype(float)
    return out
