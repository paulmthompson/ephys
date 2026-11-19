"""Inter-spike interval and local spike-train variability helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "TrialCv2Summary",
    "adjacent_isi_cv2",
    "bootstrap_median_ci_by_trial",
    "collect_adjacent_isi_cv2",
    "collect_adjacent_isi_cv2_by_trial",
    "collect_inter_spike_intervals",
    "flatten_finite_trial_values",
    "inter_spike_intervals",
    "summarize_trial_cv2",
]


@dataclass(frozen=True)
class TrialCv2Summary:
    """Adjacent-ISI CV2 summary statistics across independent trials."""

    n_trials: int
    n_spiking_trials: int
    n_cv2_trials: int
    n_cv2_pairs: int
    median_cv2: float
    mean_cv2: float
    ci_low: float
    ci_high: float


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


def flatten_finite_trial_values(
    values_by_trial: Sequence[np.ndarray],
) -> np.ndarray:
    """Return finite ``float64`` values flattened across trial arrays."""
    nonempty = [
        np.asarray(values, dtype=np.float64).ravel()
        for values in values_by_trial
        if values.size > 0
    ]
    if not nonempty:
        return np.asarray([], dtype=np.float64)
    values = np.concatenate(nonempty)
    return values[np.isfinite(values)]


def collect_inter_spike_intervals(
    spike_times_per_trial: Sequence[np.ndarray],
) -> np.ndarray:
    """Collect finite adjacent ISIs across trials without crossing boundaries.

    Parameters
    ----------
    spike_times_per_trial
        Sequence of per-trial spike-time arrays in a consistent unit.

    Returns
    -------
    numpy.ndarray
        Flattened finite within-trial inter-spike intervals.
    """
    isi_by_trial = tuple(inter_spike_intervals(trial) for trial in spike_times_per_trial)
    return flatten_finite_trial_values(isi_by_trial)


def bootstrap_median_ci_by_trial(
    values_by_trial: Sequence[np.ndarray],
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    ci_level: float,
) -> tuple[float, float]:
    """Return a trial-cluster bootstrap percentile CI for the pooled median.

    Parameters
    ----------
    values_by_trial
        Per-trial metric arrays (e.g. from
        :func:`collect_adjacent_isi_cv2_by_trial`). Resampling samples entire
        trials with replacement so within-trial dependence is preserved.
    rng
        NumPy random generator used for bootstrap resampling.
    n_bootstrap
        Number of bootstrap replicates (at least 1).
    ci_level
        Central confidence level strictly between 0 and 1.

    Returns
    -------
    tuple[float, float]
        ``(ci_low, ci_high)``, or ``(nan, nan)`` when no finite values exist.
    """
    if int(n_bootstrap) < 1:
        raise ValueError("n_bootstrap must be at least 1")
    if not 0.0 < float(ci_level) < 1.0:
        raise ValueError("ci_level must be between 0 and 1")

    trials = tuple(values_by_trial)
    if not trials:
        return float("nan"), float("nan")
    observed = flatten_finite_trial_values(trials)
    if observed.size == 0:
        return float("nan"), float("nan")

    medians = np.empty(int(n_bootstrap), dtype=np.float64)
    n_trials = len(trials)
    for i in range(int(n_bootstrap)):
        indices = rng.integers(0, n_trials, size=n_trials)
        sample = flatten_finite_trial_values([trials[int(j)] for j in indices])
        medians[i] = float(np.median(sample)) if sample.size > 0 else float("nan")
    finite = medians[np.isfinite(medians)]
    if finite.size == 0:
        return float("nan"), float("nan")
    alpha = 1.0 - float(ci_level)
    low, high = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def summarize_trial_cv2(
    spike_times_per_trial: Sequence[np.ndarray],
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    ci_level: float,
) -> TrialCv2Summary:
    """Return pooled CV2 statistics and trial-cluster bootstrap CI."""
    trials = tuple(spike_times_per_trial)
    cv2_by_trial = collect_adjacent_isi_cv2_by_trial(trials)
    cv2_values = flatten_finite_trial_values(cv2_by_trial)
    ci_low, ci_high = bootstrap_median_ci_by_trial(
        cv2_by_trial,
        rng=rng,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
    )
    return TrialCv2Summary(
        n_trials=len(trials),
        n_spiking_trials=sum(1 for trial in trials if np.asarray(trial).size > 0),
        n_cv2_trials=sum(1 for values in cv2_by_trial if values.size > 0),
        n_cv2_pairs=int(cv2_values.size),
        median_cv2=(float(np.median(cv2_values)) if cv2_values.size > 0 else float("nan")),
        mean_cv2=(float(np.mean(cv2_values)) if cv2_values.size > 0 else float("nan")),
        ci_low=ci_low,
        ci_high=ci_high,
    )
