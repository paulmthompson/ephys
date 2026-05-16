"""Per-trial PSTH summaries and **heatmap-oriented row ordering** for populations.

This module builds mean PSTHs from onset- or offset-aligned spike lists and
derives per-row peak times and stable sort orders. Those statistics are the
primary inputs for **ordering units in population heatmaps** (and for placing
points on contact-onset parallel-coordinate axes) so nearby rows reflect
similar response latencies rather than arbitrary session or unit order.

Spike counting primitives live in :mod:`ephys.processing.spike_align`.
"""

from __future__ import annotations

import numpy as np

from ephys.processing.spike_align import (
    mean_spike_probability_per_bin,
    per_trial_bin_counts,
)


def mean_psth_from_relative_spikes(
    rel_per_trial: list[np.ndarray],
    bin_edges: np.ndarray,
) -> np.ndarray | None:
    """Mean spike count per bin across trials; shape ``(n_bins,)``.

    Parameters
    ----------
    rel_per_trial
        Spike times (seconds) relative to one alignment event per trial.
    bin_edges
        Shared histogram edges in seconds.

    Returns
    -------
    np.ndarray or None
        Mean counts per bin, or ``None`` if there are no trials or bins.
    """
    if not rel_per_trial:
        return None
    counts = per_trial_bin_counts(rel_per_trial, bin_edges)
    if not counts:
        return None
    return mean_spike_probability_per_bin(counts)


def burst_trial_fraction_within_onset_window(
    rel_per_trial: list[np.ndarray],
    *,
    window_end_s: float = 0.005,
    window_start_s: float = 0.0,
) -> float | None:
    """Fraction of trials with more than one spike in an onset-aligned window.

    A trial counts toward the numerator if it has strictly more than one spike
    whose relative time falls in ``[window_start_s, window_end_s]`` (seconds
    after contact onset). The denominator is the number of trials.

    Parameters
    ----------
    rel_per_trial
        Per-trial spike times in seconds relative to contact onset.
    window_end_s
        Right edge of the inclusion window (default ``0.005`` = 5 ms).
    window_start_s
        Left edge of the inclusion window (default ``0`` = onset).

    Returns
    -------
    float or None
        Value in ``[0, 1]``, or ``None`` if there are no trials.
    """
    if not rel_per_trial:
        return None
    n_trials = len(rel_per_trial)
    n_burst_trials = 0
    for rel in rel_per_trial:
        t = np.asarray(rel, dtype=np.float64)
        in_win = (t >= window_start_s) & (t <= window_end_s)
        if int(np.sum(in_win)) > 1:
            n_burst_trials += 1
    return float(n_burst_trials) / float(n_trials)


def peak_bin_center_times_from_mean_psth(
    mean_rows: np.ndarray, bin_edges: np.ndarray
) -> np.ndarray:
    """Time of the maximum mean count in each PSTH row (bin center at argmax).

    This is the per-unit heuristic used to order population heatmaps and to
    place points on contact-onset parallel-coordinate axes.

    Parameters
    ----------
    mean_rows
        Shape ``(n_rows, n_bins)`` (e.g. mean onset PSTH per unit).
    bin_edges
        Histogram edges of length ``n_bins + 1``.

    Returns
    -------
    np.ndarray
        Shape ``(n_rows,)``, time in seconds (bin center of the argmax bin per
        row). Rows of all zeros use the first bin center.
    """
    if mean_rows.size == 0:
        return np.array([], dtype=np.float64)
    n_bins = int(bin_edges.size) - 1
    if n_bins < 1 or mean_rows.shape[1] != n_bins:
        raise ValueError("bin_edges and mean_rows column count are inconsistent")
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    peak_t = np.empty(mean_rows.shape[0], dtype=np.float64)
    for i in range(mean_rows.shape[0]):
        row = mean_rows[i]
        peak_t[i] = float(centers[int(np.argmax(row))])
    return peak_t


def peak_time_row_order(mean_rows: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Row indices sorted by peak-response time (ascending).

    Uses :func:`peak_bin_center_times_from_mean_psth` then a stable argsort.
    Rows with all zeros share the first bin center and sort by original order
    among equals.

    Parameters
    ----------
    mean_rows
        Shape ``(n_rows, n_bins)``.
    bin_edges
        Histogram edges of length ``n_bins + 1``.

    Returns
    -------
    np.ndarray
        Integer row order (length ``n_rows``), ``stable`` sort on peak times.
    """
    if mean_rows.size == 0:
        return np.array([], dtype=np.intp)
    peak_t = peak_bin_center_times_from_mean_psth(mean_rows, bin_edges)
    return np.argsort(peak_t, kind="stable")
