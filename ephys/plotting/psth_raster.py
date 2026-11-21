"""Visualizations for Per-Stimulus Time Histograms (PSTH) and spike rasters.

This module provides standard plotting functions for aligned electrophysiology
data, focusing on two primary views:

* PSTH Rendering: :func:`plot_psth` draws step-style histograms representing
  spike probabilities or rates across binned time.
* Event Rasters: :func:`plot_raster` renders per-trial spike times as horizontal
  tick marks, supporting custom vertical offsets and styling.

These functions are designed to work together in aligned views, sharing time
axes to show both aggregate response patterns and trial-to-trial variability.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import matplotlib.axes
import numpy as np
from pydantic import BaseModel, Field

from ephys.processing.bins import UniformBins

RasterTrialSort = Literal[
    "first_spike",
    "spike_count",
    "spike_count_then_first_spike",
]


class RasterOptions(BaseModel):
    """Configuration for raster plots."""

    trials_per_bin: int | None = Field(
        default=None,
        description="Consecutive trials collapsed into single raster rows.",
    )
    linewidths: float | None = Field(
        default=None,
        description="Cap for spike tick linewidths in data units.",
    )
    event_linewidth: float | None = Field(
        default=0.6,
        description="Matplotlib eventplot stroke width (points).",
    )
    trial_sort: RasterTrialSort | None = Field(
        default=None,
        description=(
            "Per-direction-band raster trial order; None uses the plot "
            "driver default."
        ),
    )


class PsthOptions(BaseModel):
    """Configuration for PSTH grids."""

    starting_time_s: float | None = Field(
        default=None,
        description="Time window start relative to alignment event.",
    )
    bin_size_s: float | None = Field(
        default=None,
        gt=0,
        description="PSTH bin size in seconds.",
    )
    max_psth_y: float | None = Field(
        default=None,
        gt=0,
        description="PSTH y-axis scale limit.",
    )


def plot_psth(
    ax: matplotlib.axes.Axes,
    hist: np.ndarray,
    bin_edges: UniformBins | np.ndarray,
    max_y: float,
    *,
    blank_style: bool = False,
    facecolor: str | tuple[float, ...] = "black",
    edgecolor: str | tuple[float, ...] | None = None,
    alpha: float = 1.0,
    linewidth: float = 0.85,
    zorder: float | None = None,
) -> None:
    """Draw a PSTH as an outline step trace spanning all ``bin_edges``.

    Parameters
    ----------
    ax
        Target Matplotlib axes.
    hist
        1D binned PSTH heights of length ``n_bins``.
    bin_edges
        :class:`~ephys.processing.bins.UniformBins` grid or 1D uniform boundary
        array of length ``n_bins + 1``.
    max_y
        Upper y-axis limit.
    blank_style
        When True, hide all spines and ticks.
    facecolor
        Outline color when ``edgecolor`` is omitted (default black).
    edgecolor
        Step outline color; defaults to ``facecolor``.
    alpha
        Outline alpha (use < 1 for overlays).
    linewidth
        Outline width in points.
    zorder
        Artist stacking order when overlaying multiple PSTHs.

    Raises
    ------
    ValueError
        If ``hist`` length does not match ``bins.n_bins`` or ``bin_edges`` is
        not a valid uniform edge vector.
    """
    bins = (
        bin_edges
        if isinstance(bin_edges, UniformBins)
        else UniformBins.from_edges(bin_edges)
    )
    hist_arr = np.asarray(hist, dtype=float).ravel()
    if hist_arr.size != bins.n_bins:
        msg = (
            f"hist length ({hist_arr.size}) must match bins.n_bins "
            f"({bins.n_bins})"
        )
        raise ValueError(msg)

    ec = edgecolor if edgecolor is not None else facecolor
    step_patch = ax.stairs(
        hist_arr,
        bins.edges,
        baseline=None,
        color=ec,
        linewidth=linewidth,
        alpha=alpha,
    )
    if zorder is not None:
        step_patch.set_zorder(float(zorder))

    ax.set_xlim(*bins.limits)
    ax.set_ylim(0, max_y)

    if blank_style:
        ax.spines[["bottom", "left", "right", "top"]].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    ax.set_xticks([])
    ax.spines[["right", "top"]].set_visible(False)
    ax.tick_params(axis="both", which="both", width=2, color="black")
    ax.set_yticks([0, max_y])
    ax.set_ylabel("Spike prob.")


def plot_raster(
    ax: matplotlib.axes.Axes,
    spike_times_per_trial_s: list[np.ndarray],
    t_start_s: float,
    t_end_s: float,
    *,
    include_zero_line: bool = True,
    blank_style: bool = False,
    lineoffsets: Sequence[float] | np.ndarray | None = None,
    linelengths: float = 0.8,
    event_linewidth: float = 0.6,
    event_zorder: float = 4.0,
) -> None:
    """Event raster: one row per trial.

    Parameters
    ----------
    lineoffsets
        Vertical center for each trial (``orientation='horizontal'``). Length
        must match the number of trials. When omitted, trials use ``0..n-1``.
    linelengths
        Total extent of each marker along the axis orthogonal to spike time
        (data units).
    event_linewidth
        Matplotlib ``eventplot`` stroke width in points.
    event_zorder
        Draw spikes above axis shading / spans (default ``4``).
    """
    n = len(spike_times_per_trial_s)
    if lineoffsets is None:
        lo = np.arange(n, dtype=float)
    else:
        lo = np.asarray(lineoffsets, dtype=float).ravel()
        if lo.size != n:
            msg = "lineoffsets length must match number of trials"
            raise ValueError(msg)

    from matplotlib.collections import EventCollection

    event_collections = ax.eventplot(
        spike_times_per_trial_s,
        color="black",
        linewidths=float(event_linewidth),
        linelengths=linelengths,
        lineoffsets=list(lo),
        orientation="horizontal",
    )
    if isinstance(event_collections, EventCollection):
        event_collections.set_zorder(event_zorder)
    else:
        for coll in event_collections:
            coll.set_zorder(event_zorder)

    ax.set_xlim(t_start_s, t_end_s)
    if lo.size == 0:
        ax.set_ylim(-0.5, 0.5)
    else:
        half = float(linelengths) / 2.0
        ax.set_ylim(float(lo.min()) - half, float(lo.max()) + half)
    ax.tick_params(axis="both", which="both", width=2, color="black")

    if blank_style:
        ax.spines[["bottom", "left", "right", "top"]].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    if lineoffsets is None:
        ax.margins(0, 0)
    ax.spines[["right", "top", "left"]].set_visible(False)
    ax.set_yticks([])
    if include_zero_line:
        ax.axvline(0, color="red", linewidth=1)
    ax.set_xticks([t_start_s, t_end_s])
    ax.set_xlabel("Time (s)")
