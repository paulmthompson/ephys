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
