"""Firing-rate and summary statistics over spike-time windows and blocks."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def per_trial_firing_rates_hz_in_window(
    rel_per_trial: Sequence[np.ndarray],
    *,
    window_start_s: float = 0.0,
    window_end_s: float = 0.005,
) -> np.ndarray:
    """Compute per-trial firing rate (Hz) inside a fixed relative window.

    Parameters
    ----------
    rel_per_trial
        Per-trial spike times in seconds relative to each alignment event.
    window_start_s, window_end_s
        Inclusive window bounds in seconds relative to event onset.

    Returns
    -------
    np.ndarray
        ``float64`` firing rates (Hz) with length equal to
        ``len(rel_per_trial)``.

    Raises
    ------
    ValueError
        If window bounds are not finite or ``window_end_s <= window_start_s``.
    """
    start_s = float(window_start_s)
    end_s = float(window_end_s)
    if not (math.isfinite(start_s) and math.isfinite(end_s)):
        msg = "window bounds must be finite"
        raise ValueError(msg)
    dur_s = end_s - start_s
    if dur_s <= 0.0:
        msg = "window must have positive duration"
        raise ValueError(msg)

    rates = np.empty(len(rel_per_trial), dtype=np.float64)
    for i, spikes in enumerate(rel_per_trial):
        st = np.asarray(spikes, dtype=np.float64).ravel()
        mask = (st >= start_s) & (st <= end_s)
        rates[i] = float(np.count_nonzero(mask)) / dur_s
    return rates


def mean_and_sem(values: np.ndarray | Sequence[float]) -> tuple[float, float]:
    """Compute sample mean and SEM across finite values.

    Parameters
    ----------
    values
        1D or multi-dimensional numeric observations (e.g., per-trial or
        per-block firing rates in Hz). Flattened and filtered to finite
        entries.

    Returns
    -------
    tuple[float, float]
        ``(mean, sem)`` using sample standard deviation (``ddof=1``). Returns
        ``(nan, nan)`` when no finite values are present, and ``sem == 0.0``
        when only one finite value is present.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    mean_val = float(np.mean(arr))
    if arr.size < 2:
        return (mean_val, 0.0)
    sem_val = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    return (mean_val, sem_val)
