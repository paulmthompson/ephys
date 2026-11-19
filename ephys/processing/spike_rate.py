from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d


def calculate_per_trial_spike_counts(
    spikes_per_trial,
    bins,
    smooth=False,
    sigma=2.0,
):
    """

    Parameters
    ----------
    spikes_per_trial
    bins
    smooth
    sigma

    Returns
    -------

    """

    trial_histograms = []
    for trial_spike_times in spikes_per_trial:
        hist, _ = np.histogram(trial_spike_times, bins=bins)
        if smooth:
            hist = gaussian_filter1d(hist.astype(float), sigma=sigma)
        trial_histograms.append(hist)

    return trial_histograms


def calculate_spike_rate(
    trial_histograms,
    bins,
    event_off_s=None,
    return_probabilty=False,
):
    """

    Parameters
    ----------
    trial_histograms: list
        count of spikes in each time bin for each trial.
    bins: np.ndarray
        the bin edges for histogram calculation.
    event_off_s:
        the time when the event ends. If provided, the spike rate will be calculated
        by dividing the number of spikes by the number of trials in each bin.
    return_probabilty: bool
        if True, return the probability of spiking in each bin rather than the spike rate.

    Returns
    -------
    np.ndarray
        the mean spike rate in each bin.
    """

    if event_off_s is not None:
        remove_spike_events = True
    else:
        remove_spike_events = False

    trial_histograms = np.array(trial_histograms)

    hist = np.sum(trial_histograms, axis=0)
    if remove_spike_events:
        num_trials_per_bin = np.zeros(bins.shape[0] - 1)
        for i in range(len(bins) - 1):
            num_trials_per_bin[i] = np.sum(event_off_s >= bins[i])
        spike_rate = hist / num_trials_per_bin
    else:
        spike_rate = hist / len(trial_histograms)

    if not return_probabilty:
        bin_width_s = bins[1] - bins[0]
        spike_rate /= bin_width_s

    return spike_rate


def bootstrap_ci(
    spikes_per_trial,
    bins,
    n_bootstrap=10000,
    ci_percentile=95,
    **kwargs,
):
    """
    Calculate the confidence interval of a spike rate using bootstrap resampling.

    Parameters:
    - spikes_per_trial: list of arrays, each array contains spike times for a trial.
    - bins: array-like, the bin edges for histogram calculation.
    - n_bootstrap: int, number of bootstrap samples.
    - ci_percentile: float, the percentile for the confidence interval.
    - **kwargs: additional keyword arguments for calculate_per_trial_spike_counts.

    Returns:
    - ci_lower: array, the lower bound of the confidence interval for each bin.
    - ci_upper: array, the upper bound of the confidence interval for each bin.
    """
    n_trials = len(spikes_per_trial)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    bootstrap_means = np.zeros((n_bootstrap, len(bin_centers)))

    trial_histograms = calculate_per_trial_spike_counts(
        spikes_per_trial,
        bins,
        **kwargs,
    )

    trial_histograms = np.array(trial_histograms)

    for i in range(n_bootstrap):
        resampled_indices = np.random.choice(n_trials, n_trials, replace=True)
        resampled_histograms = trial_histograms[resampled_indices]
        hist_sum = np.sum(resampled_histograms, axis=0)
        bootstrap_means[i, :] = hist_sum / n_trials / bin_width

    ci_lower = np.percentile(bootstrap_means, (100 - ci_percentile) / 2, axis=0)
    ci_upper = np.percentile(bootstrap_means, 100 - (100 - ci_percentile) / 2, axis=0)
    return ci_lower, ci_upper


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
