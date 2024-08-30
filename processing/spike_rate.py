
import numpy as np


def bootstrap_ci(
    spikes_per_trial,
    bins,
    n_bootstrap=10000,
    ci_percentile=95,
):
    """
    Calculate the confidence interval of a spike rate using bootstrap resampling.

    Parameters:
    - spikes_per_trial: list of arrays, each array contains spike times for a trial.
    - bins: array-like, the bin edges for histogram calculation.
    - n_bootstrap: int, number of bootstrap samples.
    - ci_percentile: float, the percentile for the confidence interval.

    Returns:
    - ci_lower: array, the lower bound of the confidence interval for each bin.
    - ci_upper: array, the upper bound of the confidence interval for each bin.
    """
    n_trials = len(spikes_per_trial)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    bootstrap_means = np.zeros((n_bootstrap, len(bin_centers)))

    # Precompute histograms for all trials
    trial_histograms = np.array(
        [np.histogram(trial, bins=bins)[0] for trial in spikes_per_trial]
    )

    for i in range(n_bootstrap):
        resampled_indices = np.random.choice(n_trials, n_trials, replace=True)
        resampled_histograms = trial_histograms[resampled_indices]
        hist_sum = np.sum(resampled_histograms, axis=0)
        bootstrap_means[i, :] = hist_sum / n_trials / bin_width

    ci_lower = np.percentile(bootstrap_means, (100 - ci_percentile) / 2, axis=0)
    ci_upper = np.percentile(bootstrap_means, 100 - (100 - ci_percentile) / 2, axis=0)
    return ci_lower, ci_upper
