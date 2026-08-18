"""Inter-spike interval and local spike-train variability helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "adjacent_isi_cv2",
    "collect_adjacent_isi_cv2",
    "collect_adjacent_isi_cv2_by_trial",
    "inter_spike_intervals",
]


def inter_spike_intervals(spike_times: np.ndarray) -> np.ndarray:
    """Return sorted adjacent inter-spike intervals for one spike train.

    Parameters
    ----------
    spike_times
        Spike times in a common unit. Values are flattened and sorted before
        intervals are computed.

    Returns
    -------
    numpy.ndarray
        Adjacent inter-spike intervals in the same unit as ``spike_times``.
        Returns an empty array when fewer than two finite spike times are
        present.

    Notes
    -----
    Non-finite spike times are ignored. The caller is responsible for passing
    spike times from a single trial/window when cross-boundary ISIs are not
    meaningful.
    """
    spikes = np.asarray(spike_times, dtype=np.float64).ravel()
    spikes = spikes[np.isfinite(spikes)]
    if spikes.size < 2:
        return np.asarray([], dtype=np.float64)
    spikes = np.sort(spikes, kind="mergesort")
    return np.diff(spikes)


def adjacent_isi_cv2(spike_times: np.ndarray) -> np.ndarray:
    """Return Holt CV2 values for adjacent ISI pairs in one spike train.

    Parameters
    ----------
    spike_times
        Spike times from one trial/window, in any consistent unit.

    Returns
    -------
    numpy.ndarray
        ``2 * abs(ISI[i+1] - ISI[i]) / (ISI[i+1] + ISI[i])`` for each
        adjacent ISI pair with a positive finite denominator. Returns an empty
        array when fewer than two ISIs are available.

    Notes
    -----
    The conventional nonnegative Holt CV2 is returned. To avoid creating
    biologically invalid intervals, call this separately for each trial or
    stimulus window when boundaries matter.
    """
    isi = inter_spike_intervals(spike_times)
    if isi.size < 2:
        return np.asarray([], dtype=np.float64)
    left = isi[:-1]
    right = isi[1:]
    denom = left + right
    valid = np.isfinite(left) & np.isfinite(right) & (denom > 0.0)
    if not np.any(valid):
        return np.asarray([], dtype=np.float64)
    return 2.0 * np.abs(right[valid] - left[valid]) / denom[valid]


def collect_adjacent_isi_cv2_by_trial(
    spike_times_per_trial: Sequence[np.ndarray],
) -> tuple[np.ndarray, ...]:
    """Return per-trial CV2 arrays without crossing trial boundaries.

    Parameters
    ----------
    spike_times_per_trial
        Sequence of per-trial spike-time arrays. Each array is processed
        independently.

    Returns
    -------
    tuple[numpy.ndarray, ...]
        One CV2 array per input trial, preserving trial order. Trials with too
        few spikes contribute empty arrays.
    """
    return tuple(adjacent_isi_cv2(trial) for trial in spike_times_per_trial)


def collect_adjacent_isi_cv2(
    spike_times_per_trial: Sequence[np.ndarray],
) -> np.ndarray:
    """Collect finite Holt CV2 values across independent trials.

    Parameters
    ----------
    spike_times_per_trial
        Sequence of per-trial spike-time arrays.

    Returns
    -------
    numpy.ndarray
        Flattened finite CV2 values from all trials. Empty trials, short
        trials, and invalid adjacent ISI pairs contribute no values.

    Notes
    -----
    The function computes each trial separately before concatenation, so the
    last spike in one trial is never paired with the first spike in the next.
    """
    by_trial = collect_adjacent_isi_cv2_by_trial(spike_times_per_trial)
    nonempty = [values for values in by_trial if values.size > 0]
    if not nonempty:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(nonempty)
