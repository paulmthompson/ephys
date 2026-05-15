"""Compressed event-raster layout: fewer rows than trials when space is tight.

Matplotlib ``eventplot`` draws one horizontal row per list entry. When there
are **more trials than usable vertical pixels** (or than comfortable row
spacing), this module **merges spikes from consecutive trials** into a single
display row so multiple trials share one raster line—see
:func:`bin_spike_lists` and :func:`merge_spikes_chunk`.

It also supplies **row geometry** for ``eventplot``: equally spaced
``lineoffsets`` inside a fixed band (:func:`lineoffsets_equal_height_band`) and
a capped marker half-height derived from band height and row counts across
stacked bands (:func:`effective_raster_linelength`). For paired onset/offset
lists, :func:`sorted_subset_trials` subsets rows and reorders by first-spike
time on the primary alignment.

All helpers are **contact-agnostic**: they operate on parallel lists of per-trial
spike time arrays (seconds) and parameters passed through to
:func:`matplotlib.axes.Axes.eventplot` as ``lineoffsets`` / ``linelengths``.
"""

from __future__ import annotations

import numpy as np

from processing.spike_align import (
    apply_trial_order,
    sort_order_by_first_spike,
)


def merge_spikes_chunk(chunk: list[np.ndarray]) -> np.ndarray:
    """Concatenate spike times from consecutive trials into one raster row.

    Empty trials are skipped; if all are empty, returns an empty float array.

    Parameters
    ----------
    chunk
        Spike arrays for consecutive trials to merge into a single row.

    Returns
    -------
    np.ndarray
        1-D spike times (concatenated), dtype ``float64``.
    """
    parts = [np.asarray(x, dtype=np.float64).ravel() for x in chunk if np.asarray(x).size > 0]
    if not parts:
        return np.array([], dtype=np.float64)
    return np.concatenate(parts)


def bin_spike_lists(
    spike_lists: list[np.ndarray],
    trials_per_bin: int,
) -> list[np.ndarray]:
    """Collapse consecutive trials into fewer rows (spikes overlaid per row).

    Parameters
    ----------
    spike_lists
        One spike array per trial row, in display order.
    trials_per_bin
        Number of consecutive trials merged into each output row; must be >= 1.

    Returns
    -------
    list[np.ndarray]
        Shorter list where each element is from :func:`merge_spikes_chunk` over
        a slice of ``spike_lists``.
    """
    if trials_per_bin <= 1:
        return spike_lists
    out: list[np.ndarray] = []
    for i in range(0, len(spike_lists), trials_per_bin):
        out.append(merge_spikes_chunk(spike_lists[i : i + trials_per_bin]))
    return out


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
