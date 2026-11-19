"""Signal processing utilities for electrophysiology data."""

from ephys.processing.basis import log_raised_cosine_basis, raised_cosine_basis
from ephys.processing.resampling import whittaker_shannon_interpolate
from ephys.processing.spike_align import (
    bin_centers,
    collect_first_spike_jitter_ms,
    first_spike_time_ms,
    ms_to_ticks,
    seconds_to_ticks,
    spikes_relative_to_tick_intervals,
    validate_tick_intervals,
    window_seconds_to_ticks,
)
from ephys.processing.spike_intervals import (
    TrialCv2Summary,
    adjacent_isi_cv2,
    bootstrap_median_ci_by_trial,
    collect_adjacent_isi_cv2,
    collect_adjacent_isi_cv2_by_trial,
    collect_inter_spike_intervals,
    flatten_finite_trial_values,
    inter_spike_intervals,
    summarize_trial_cv2,
)
from ephys.processing.spike_rate import (
    mean_and_sem,
    per_trial_firing_rates_hz_in_window,
)

__all__ = [
    "TrialCv2Summary",
    "adjacent_isi_cv2",
    "bin_centers",
    "bootstrap_median_ci_by_trial",
    "collect_adjacent_isi_cv2",
    "collect_adjacent_isi_cv2_by_trial",
    "collect_first_spike_jitter_ms",
    "collect_inter_spike_intervals",
    "first_spike_time_ms",
    "flatten_finite_trial_values",
    "inter_spike_intervals",
    "log_raised_cosine_basis",
    "mean_and_sem",
    "ms_to_ticks",
    "per_trial_firing_rates_hz_in_window",
    "raised_cosine_basis",
    "seconds_to_ticks",
    "spikes_relative_to_tick_intervals",
    "summarize_trial_cv2",
    "validate_tick_intervals",
    "whittaker_shannon_interpolate",
    "window_seconds_to_ticks",
]
