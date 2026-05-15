"""Step PSTHs and per-trial spike rasters for aligned electrophysiology views."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.axes
import numpy as np


def plot_psth(
    ax: matplotlib.axes.Axes,
    hist: np.ndarray,
    bin_edges: np.ndarray,
    max_y: float,
    *,
    blank_style: bool = False,
    facecolor: str | tuple[float, ...] = "black",
    edgecolor: str | tuple[float, ...] | None = None,
    alpha: float = 1.0,
    linewidth: float = 0.85,
    zorder: float | None = None,
) -> None:
    """Draw a PSTH as outline-only left-aligned bars on ``bin_edges``.

    Parameters
    ----------
    facecolor
        Outline color when ``edgecolor`` is omitted (default black).
    edgecolor
        Bar outline color; defaults to ``facecolor``.
    alpha
        Outline alpha (use < 1 for overlays).
    linewidth
        Outline width in points.
    zorder
        Artist stacking order when overlaying multiple PSTHs.
    """
    bin_edges = np.asarray(bin_edges, dtype=float).ravel()
    hist = np.asarray(hist, dtype=float).ravel()
    if bin_edges.size < 2:
        return
    n_bins = bin_edges.size - 1
    if hist.size != n_bins:
        n = min(hist.size, n_bins)
        hist = hist[:n]
        bin_edges = bin_edges[: n + 1]
        n_bins = bin_edges.size - 1
    if n_bins < 1:
        return

    t0, t1 = float(bin_edges[0]), float(bin_edges[-1])

    ec = edgecolor if edgecolor is not None else facecolor
    extra: dict[str, float] = {}
    if zorder is not None:
        extra["zorder"] = float(zorder)

    ax.step(
        bin_edges[:-1],
        hist,
        where="post",
        color=ec,
        linewidth=linewidth,
        alpha=alpha,
        **extra,
    )

    ax.set_xlim(t0, t1)
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
    event_zorder: float = 4.0,
) -> None:
    """Event raster: one row per trial.

    Parameters
    ----------
    lineoffsets
        Vertical center for each trial (``orientation='horizontal'``). Length
        must match the number of trials. When omitted, trials use ``0..n-1``.
    linelengths
        Total extent of each marker along the axis orthogonal to spike time.
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

    event_collections =ax.eventplot(
        spike_times_per_trial_s,
        color="black",
        linewidths=0.2,  # 0.6
        linelengths=linelengths,
        lineoffsets=list(lo),
        orientation="horizontal",
    )
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
