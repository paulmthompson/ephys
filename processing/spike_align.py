"""Align spike trains to DAQ tick events for PSTHs and rasters.

Utilities convert between seconds and sample indices, extract spikes in
per-trial windows around events, histogram relative times, and reorder trials
by first-spike latency. Dense periodic events (e.g. high-rate pico trains) can
use :func:`event_ticks_greedy_non_overlapping_half_windows` to subsample onsets
before gathering so trial windows do not overlap.
"""

from __future__ import annotations

import numpy as np


def spike_times_ticks_from_seconds(
    spike_times_s: np.ndarray,
    sampling_rate_hz: float,
) -> np.ndarray:
    """Convert spike times from seconds to integer DAQ sample indices.

    Parameters
    ----------
    spike_times_s
        Spike times in seconds; any shape is flattened.
    sampling_rate_hz
        Acquisition rate used to scale seconds to ticks.

    Returns
    -------
    np.ndarray
        ``int64`` tick indices, ``round(spike_times_s * sampling_rate_hz)``.

    Notes
    -----
    Values are rounded to the nearest integer tick (not truncated).
    """
    st = np.asarray(spike_times_s, dtype=float).ravel()
    return np.round(st * sampling_rate_hz).astype(np.int64, copy=False)


def spikes_relative_to_events_ticks(
    spike_times_ticks: np.ndarray,
    event_ticks: np.ndarray,
    win_ticks: int,
    sampling_rate_hz: float,
) -> list[np.ndarray]:
    """Spike times in seconds relative to each event, inside a tick window.

    For every event tick, returns spike times within
    ``[event - win_ticks, event + win_ticks)`` (half-open on the right in tick
    space), expressed in seconds relative to that event.

    Parameters
    ----------
    spike_times_ticks
        Spike sample indices; converted to ``int64`` 1D and **sorted in
        place** on the working copy used internally.
    event_ticks
        Alignment sample indices (one per trial or condition row).
    win_ticks
        Half-width of the inclusion window in samples (same units as spike and
        event ticks).
    sampling_rate_hz
        Rate used to convert tick offsets to seconds.

    Returns
    -------
    list[np.ndarray]
        Length ``len(event_ticks)``; entry ``i`` holds ``float64`` relative
        times in seconds for spikes in the window around ``event_ticks[i]``
        (possibly length 0).

    Raises
    ------
    ValueError:
        If ``sampling_rate_hz`` or ``win_ticks`` is not positive.

    Notes
    -----
    Window membership uses ``searchsorted`` on sorted spike ticks; boundary
    behavior matches half-open ``[lo, hi)`` sampling in index space.
    """
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    if win_ticks <= 0:
        raise ValueError("win_ticks must be positive")

    spike_times_ticks = np.sort(
        np.asarray(spike_times_ticks, dtype=np.int64).ravel(),
        kind="mergesort",
    )

    rel_per_trial: list[np.ndarray] = []
    half = int(win_ticks)

    for event_tick in np.asarray(event_ticks, dtype=np.int64).ravel():
        lo = int(event_tick) - half
        hi = int(event_tick) + half
        i0 = int(np.searchsorted(spike_times_ticks, lo, side="left"))
        i1 = int(np.searchsorted(spike_times_ticks, hi, side="left"))
        chunk = spike_times_ticks[i0:i1]
        rel_s = chunk.astype(np.float64) / sampling_rate_hz - float(event_tick) / sampling_rate_hz
        rel_per_trial.append(rel_s)

    return rel_per_trial


def event_ticks_greedy_non_overlapping_half_windows(
    event_ticks: np.ndarray,
    win_ticks: int,
) -> np.ndarray:
    """Subset onset ticks so symmetric half-windows do not overlap.

    Windows follow :func:`spikes_relative_to_events_ticks`:
    ``[t - win_ticks, t + win_ticks)`` per onset. Two onsets at ``t0 < t1``
    have disjoint windows iff ``t1 - t0 >= 2 * win_ticks`` (half-open ticks).

    This function sorts ``event_ticks`` in time and applies a greedy rule:
    keep the earliest onset, then keep the next onset that is at least
    ``2 * win_ticks`` after the last kept onset, and so on. The returned array
    is **chronological** (ascending onsets), usually shorter than the input.

    Parameters
    ----------
    event_ticks
        Candidate alignment sample indices (any order; duplicates allowed).
    win_ticks
        Same half-width passed to :func:`spikes_relative_to_events_ticks`.

    Returns
    -------
    np.ndarray
        ``int64`` 1D, strictly non-decreasing kept onsets. Empty if the input
        is empty.

    Raises
    ------
    ValueError:
        If ``win_ticks`` is not positive.

    Notes
    -----
    This **drops trials** (onsets), not spikes within a trial. After filtering,
    call :func:`spikes_relative_to_events_ticks` to gather the full symmetric
    window for each kept onset without cross-trial window overlap.

    Greedy earliest-first scheduling maximizes the number of kept trials among
    equal-length windows on a line.
    """
    if win_ticks <= 0:
        raise ValueError("win_ticks must be positive")

    events = np.asarray(event_ticks, dtype=np.int64).ravel()
    if events.size == 0:
        return np.empty(0, dtype=np.int64)

    t_sorted = np.sort(events, kind="mergesort")
    half = int(win_ticks)
    min_sep = 2 * half

    kept: list[int] = []
    last_t: int | None = None
    for t in t_sorted:
        tt = int(t)
        if last_t is None or tt - last_t >= min_sep:
            kept.append(tt)
            last_t = tt

    return np.asarray(kept, dtype=np.int64)


def sort_order_by_first_spike(
    spike_times_per_trial_s: list[np.ndarray],
) -> np.ndarray:
    """Return trial indices sorted by latency to the first spike.

    Parameters
    ----------
    spike_times_per_trial_s
        One ``float`` array per trial (seconds relative to a common alignment);
        empty arrays are allowed.

    Returns
    -------
    np.ndarray
        ``intp`` index vector into the input list order (argsort output).

    Notes
    -----
    Trials with no spikes are treated as having first spike time ``-inf`` so
    they sort **last** (legacy convention).
    """
    n = len(spike_times_per_trial_s)
    first_s = np.empty(n, dtype=np.float64)
    for i, arr in enumerate(spike_times_per_trial_s):
        a = np.asarray(arr, dtype=float).ravel()
        if a.size > 0:
            first_s[i] = float(np.min(a))
        else:
            first_s[i] = np.inf
    return np.argsort(first_s, kind="mergesort")


def sort_order_by_spike_count_descending(
    spike_times_per_trial_s: list[np.ndarray],
) -> np.ndarray:
    """Return trial indices sorted by within-trial spike count (descending).

    Parameters
    ----------
    spike_times_per_trial_s
        One 1D array per trial (relative spike times in seconds); empty arrays
        are allowed.

    Returns
    -------
    np.ndarray
        ``intp`` index vector into the input list order (suitable for
        :func:`apply_trial_order`).

    Notes
    -----
    Trials with more spikes in the window are ordered **first** in the returned
    permutation, so after :func:`apply_trial_order` they appear on **lower**
    raster row indices (matching ``plot_raster`` defaults with row ``0`` at
    the bottom). Ties break by stable ascending trial index.
    """
    n = len(spike_times_per_trial_s)
    cnt = np.empty(n, dtype=np.intp)
    for i, arr in enumerate(spike_times_per_trial_s):
        cnt[i] = int(np.asarray(arr, dtype=float).ravel().size)
    return np.argsort(-cnt, kind="mergesort")


def apply_trial_order(
    trials: list[np.ndarray],
    order: np.ndarray,
) -> list[np.ndarray]:
    """Reorder a list of per-trial arrays.

    Parameters
    ----------
    trials
        Parallel per-trial payloads (e.g. relative spike time arrays).
    order
        Row indices, typically from :func:`sort_order_by_first_spike` or
        :func:`sort_order_by_spike_count_descending`.

    Returns
    -------
    list[np.ndarray]
        ``[trials[int(j)] for j in order]`` with the same element dtypes/shapes
        as the inputs.
    """
    return [trials[int(j)] for j in order]


def per_trial_bin_counts(
    spike_times_per_trial_s: list[np.ndarray],
    bin_edges_s: np.ndarray,
) -> list[np.ndarray]:
    """Histogram spike counts per trial using shared bin edges.

    Parameters
    ----------
    spike_times_per_trial_s
        One 1D ``float`` array per trial (seconds in the same reference frame).
    bin_edges_s
        Monotonic bin edges in seconds (length ``n_bins + 1``), shared by all
        trials.

    Returns
    -------
    list[np.ndarray]
        Per-trial count vectors, each length ``len(bin_edges_s) - 1``, dtype
        ``float64`` (same as ``numpy.histogram`` counts cast to float).

    Notes
    -----
    Uses :func:`numpy.histogram` with ``bins=bin_edges_s`` for each trial.
    """
    edges = np.asarray(bin_edges_s, dtype=float).ravel()
    out: list[np.ndarray] = []
    for rel in spike_times_per_trial_s:
        rel_f = np.asarray(rel, dtype=float).ravel()
        counts, _ = np.histogram(rel_f, bins=edges)
        out.append(counts.astype(np.float64, copy=False))
    return out


def mean_spike_probability_per_bin(
    trial_histograms: list[np.ndarray],
) -> np.ndarray:
    """Mean per-trial spike count in each bin across trials.

    Parameters
    ----------
    trial_histograms
        Either a list of 1D count arrays (equal length) or inputs that stack
        to a 2D ``(n_trials, n_bins)`` array of counts.

    Returns
    -------
    np.ndarray
        1D ``float64`` vector of length ``n_bins``, mean over trials.

    Notes
    -----
    If ``trial_histograms`` is a list, it is stacked along axis 0 before
    taking the mean (same as ``np.mean(np.stack(...), axis=0)`` when shapes
    match).
    """
    h = np.asarray(trial_histograms, dtype=float)
    if h.ndim != 2:
        h = np.stack(trial_histograms, axis=0)
    return np.mean(h, axis=0)
