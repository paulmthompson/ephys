"""Layout and geometry utilities for high-density spike rasters.

This module provides tools for managing vertical space in raster plots,
including "compression" (merging multiple trials into single rows) and
precise calculation of marker geometry for Matplotlib's ``eventplot``.

Key functionality:

* Row Merging: :func:`bin_spike_lists` and :func:`merge_spikes_chunk` collapse
  consecutive trials into fewer rows when vertical resolution is limited.
* Geometry Helpers: :func:`lineoffsets_equal_height_band` and
  :func:`effective_raster_linelength` calculate optimal vertical offsets and
  marker lengths to maintain visual consistency across stacked panels.
* Trial Sorting: :func:`sorted_subset_trials` provides convenient subsetting
  and sorting of parallel trial lists by first-spike latency.

These utilities operate directly on lists of numpy arrays (spike times in
seconds) and are agnostic to specific experimental paradigms.
"""

from __future__ import annotations

import numpy as np

from ephys.processing.spike_align import (
    apply_trial_order,
    bin_spike_lists,
    merge_spikes_chunk,
    sort_order_by_first_spike,
    sort_order_by_spike_count_descending,
    sort_order_by_spike_count_then_first_spike,
)

__all__ = [
    "bin_spike_lists",
    "effective_raster_linelength",
    "lineoffsets_equal_height_band",
    "merge_spikes_chunk",
    "sorted_subset_trials",
    "sorted_subset_trials_by_spike_count",
    "sorted_subset_trials_by_spike_count_then_first_spike",
]


def lineoffsets_equal_height_band(
    y0: float,
    band_height: float,
    n_rows: int,
    half: float,
) -> np.ndarray:
    """Compute equally spaced row centers in a fixed vertical band.

    Parameters
    ----------
    y0
        Lower **data** boundary of the band.
    band_height
        Height of the band in data units.
    n_rows
        Number of raster rows; if 0, returns an empty array.
    half
        Half-height of each spike marker in data units (centers stay inside
        ``[y0 + half, y0 + band_height - half]`` when possible).

    Returns
    -------
    np.ndarray
        Shape ``(n_rows,)``, y positions for ``lineoffsets``.
    """
    if n_rows <= 0:
        return np.zeros(0, dtype=float)
    if n_rows == 1:
        return np.array([y0 + band_height / 2.0], dtype=float)
    inner_lo = y0 + half
    inner_hi = y0 + band_height - half
    if inner_hi < inner_lo:
        inner_hi = inner_lo
    return np.linspace(inner_lo, inner_hi, n_rows)


def effective_raster_linelength(
    band_height: float,
    n_rows_band_a: int,
    n_rows_band_b: int,
    linelength_cap: float,
) -> float:
    """Choose spike marker height from band height and row counts (two bands).

    When two stacked bands share the same **data** height, row spacing is set
    by the band with more rows. The returned length is capped by
    ``linelength_cap`` and scaled so markers stay thinner than row spacing.

    Parameters
    ----------
    band_height
        Nominal vertical extent of each band in data units (both bands equal).
    n_rows_band_a, n_rows_band_b
        Raster row counts in the two bands (zero if that band has no rows).
    linelength_cap
        Maximum half-height of spike ticks in data units.

    Returns
    -------
    float
        Half-length of spike markers in data units for ``linelengths``.
    """
    active = [n for n in (n_rows_band_a, n_rows_band_b) if n > 0]
    bh = float(band_height)
    if not active:
        return float(min(linelength_cap, 0.28 * bh))
    n_max = max(active)
    if n_max <= 1:
        return float(min(linelength_cap, 0.28 * bh))
    spacing = bh / float(n_max)
    return float(min(linelength_cap, 0.82 * spacing))


def sorted_subset_trials(
    rel_primary: list[np.ndarray],
    rel_secondary: list[np.ndarray],
    row_indices: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Subset paired trial lists; sort by first-spike latency on the primary.

    Parameters
    ----------
    rel_primary
        Spike times per trial for ordering (e.g. onset-aligned).
    rel_secondary
        Parallel list (e.g. offset-aligned), same permutation as primary.
    row_indices
        Integer indices into both lists (typically from a mask or category).

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray]]
        ``(primary_sorted, secondary_sorted)``. Empty lists if
        ``row_indices`` is empty.

    Notes
    -----
    Order is :func:`ephys.processing.spike_align.sort_order_by_first_spike`
    applied to the subset of ``rel_primary``.
    """
    if row_indices.size == 0:
        return [], []
    r_pri = [rel_primary[int(i)] for i in row_indices]
    order = sort_order_by_first_spike(r_pri)
    r_pri_o = apply_trial_order(r_pri, order)
    r_sec = [rel_secondary[int(i)] for i in row_indices]
    r_sec_o = apply_trial_order(r_sec, order)
    return r_pri_o, r_sec_o


def sorted_subset_trials_by_spike_count(
    rel_primary: list[np.ndarray],
    rel_secondary: list[np.ndarray],
    row_indices: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Subset paired trial lists; sort by spike count for stacked rasters.

    Parameters
    ----------
    rel_primary
        Spike times per trial for ordering (e.g. onset-aligned).
    rel_secondary
        Parallel list (e.g. offset-aligned), same permutation as primary.
    row_indices
        Integer indices into both lists (typically from a mask or category).

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray]]
        ``(primary_sorted, secondary_sorted)``. Empty lists if
        ``row_indices`` is empty.

    Notes
    -----
    Trials are ordered so that **more spikes appear toward the top** of each
    raster band (higher ``lineoffsets``) and **zero-spike trials toward the
    bottom**, by reversing
    :func:`ephys.processing.spike_align.sort_order_by_spike_count_descending`.
    """
    if row_indices.size == 0:
        return [], []
    r_pri = [rel_primary[int(i)] for i in row_indices]
    order = sort_order_by_spike_count_descending(r_pri)[::-1]
    r_pri_o = apply_trial_order(r_pri, order)
    r_sec = [rel_secondary[int(i)] for i in row_indices]
    r_sec_o = apply_trial_order(r_sec, order)
    return r_pri_o, r_sec_o


def sorted_subset_trials_by_spike_count_then_first_spike(
    rel_primary: list[np.ndarray],
    rel_secondary: list[np.ndarray],
    row_indices: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Subset paired trial lists; sort by count then first-spike latency.

    Parameters
    ----------
    rel_primary
        Spike times per trial for ordering (e.g. onset-aligned).
    rel_secondary
        Parallel list (e.g. offset-aligned), same permutation as primary.
    row_indices
        Integer indices into both lists (typically from a mask or category).

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray]]
        ``(primary_sorted, secondary_sorted)``. Empty lists if
        ``row_indices`` is empty.

    Notes
    -----
    Uses :func:`~ephys.processing.spike_align.sort_order_by_spike_count_then_first_spike`
    on the subset of ``rel_primary``.
    """
    if row_indices.size == 0:
        return [], []
    r_pri = [rel_primary[int(i)] for i in row_indices]
    order = sort_order_by_spike_count_then_first_spike(r_pri)
    r_pri_o = apply_trial_order(r_pri, order)
    r_sec = [rel_secondary[int(i)] for i in row_indices]
    r_sec_o = apply_trial_order(r_sec, order)
    return r_pri_o, r_sec_o
