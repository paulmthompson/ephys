"""Signal processing utilities for electrophysiology data."""

from ephys.processing.basis import log_raised_cosine_basis, raised_cosine_basis
from ephys.processing.resampling import whittaker_shannon_interpolate
from ephys.processing.spike_align import (
    ms_to_ticks,
    seconds_to_ticks,
    spikes_relative_to_tick_intervals,
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

__all__ = [
    "TrialCv2Summary",
    "adjacent_isi_cv2",
    "bootstrap_median_ci_by_trial",
    "collect_adjacent_isi_cv2",
    "collect_adjacent_isi_cv2_by_trial",
    "collect_inter_spike_intervals",
    "flatten_finite_trial_values",
    "inter_spike_intervals",
    "log_raised_cosine_basis",
    "ms_to_ticks",
    "raised_cosine_basis",
    "seconds_to_ticks",
    "spikes_relative_to_tick_intervals",
    "summarize_trial_cv2",
    "whittaker_shannon_interpolate",
]
