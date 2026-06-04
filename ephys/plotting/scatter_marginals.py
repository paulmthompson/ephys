"""Scatter plot with external marginal histograms and a y-x difference inset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpecFromSubplotSpec, SubplotSpec
from matplotlib.patches import Polygon
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class ScatterMarginalsData:
    """Paired x/y samples aligned by index (e.g. per-unit mean rates)."""

    x: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        """Validate that x and y are numpy arrays of the same length."""
        if self.x.shape != self.y.shape:
            raise ValueError("x and y must have the same shape")
        if self.x.size == 0:
            raise ValueError("x and y must have at least one element")
        if not np.all(np.isfinite(self.x)) or not np.all(np.isfinite(self.y)):
            raise ValueError("x and y must be finite")


class ScatterMarginalsOptions(BaseModel):
    """Layout, histogram bins, axis labels, and scatter point styling."""

    model_config = {"frozen": True}

    marginal_height_ratio: float = Field(default=0.22, gt=0.0)
    marginal_width_ratio: float = Field(default=0.22, gt=0.0)
    xlabel: str = "X"
    ylabel: str = "Y"
    marginal_hist_bins: int = Field(default=10, ge=1)
    diff_hist_bins: int = Field(default=10, ge=1)
    point_marker: str = "o"
    point_size: float = Field(default=16.0, gt=0.0)
    point_facecolor: str = "none"
    point_edgecolor: str = "0.15"


def _draw_y_minus_x_hist_45deg_inset(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int,
) -> None:
    """Draw a small y-x histogram in the scatter panel's upper right.

    Bin positions run along ``(1, -1)`` in axes-fraction space so the local
    histogram abscissa is perpendicular to the unity direction ``(1, 1)``.
    Count magnitude extends along ``(1, 1)``.
    """
    diffs = (y - x).astype(np.float64, copy=False)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size < 1:
        return
    counts, edges = np.histogram(diffs, bins=int(n_bins))
    if counts.size == 0 or int(np.max(counts)) <= 0:
        return
    d_min = float(edges[0])
    d_max = float(edges[-1])
    span = d_max - d_min
    if span <= 0.0:
        return
    max_c = float(np.max(counts))
    center = np.array([0.84, 0.84], dtype=np.float64)
    u_perp = np.array([1.0, -1.0], dtype=np.float64)
    u_perp = u_perp / float(np.linalg.norm(u_perp))
    u_para = np.array([1.0, 1.0], dtype=np.float64)
    u_para = u_para / float(np.linalg.norm(u_para))
    scale_bins = 0.16
    scale_cnt = 0.11
    transform = ax.transAxes
    t_mat = np.column_stack([u_perp, u_para])
    for k in range(int(counts.size)):
        e0, e1 = float(edges[k]), float(edges[k + 1])
        b0 = ((e0 - d_min) / span - 0.5) * 2.0 * scale_bins
        b1 = ((e1 - d_min) / span - 0.5) * 2.0 * scale_bins
        h = (float(counts[k]) / max_c) * scale_cnt
        corners_uv = np.array(
            [
                [b0, 0.0],
                [b1, 0.0],
                [b1, h],
                [b0, h],
            ],
            dtype=np.float64,
        )
        xy = center + corners_uv @ t_mat.T
        poly = Polygon(
            xy,
            closed=True,
            facecolor="k",
            edgecolor="k",
            linewidth=0.25,
            alpha=0.9,
            zorder=6,
            transform=transform,
            clip_on=False,
        )
        ax.add_patch(poly)


def draw_scatter_marginals_empty_placeholder(
    fig: Figure,
    cell: SubplotSpec,
    message: str,
) -> None:
    """Draw centered gray placeholder text when data are missing."""
    ax = fig.add_subplot(cell)
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="gray",
    )
    ax.set_axis_off()


def draw_scatter_marginals_into(
    fig: Figure,
    cell: SubplotSpec,
    data: ScatterMarginalsData,
    options: ScatterMarginalsOptions | None = None,
) -> None:
    """Scatter (x vs. y) with external marginals and a 45° y-x diff inset.

    Parameters
    ----------
    fig
        Parent figure.
    cell
        SubplotSpec reserved for the joint panel (scatter + marginals).
    data
        Per-sample ``x`` and ``y`` arrays of equal length.
    options
        Layout, histogram bin counts, axis labels, and scatter styling.
        Uses :class:`ScatterMarginalsOptions` defaults when ``None``.

    Notes
    -----
    Top and right marginals share the scatter limits, use black bars with
    ``align='right'``, and call ``set_axis_off()`` on the marginal axes.
    """
    opts = options or ScatterMarginalsOptions()
    x_arr = data.x
    y_arr = data.y
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not np.any(mask):
        draw_scatter_marginals_empty_placeholder(
            fig,
            cell,
            "No finite x/y values",
        )
        return

    rx = x_arr[mask]
    wy = y_arr[mask]
    inner = GridSpecFromSubplotSpec(
        2,
        2,
        cell,
        height_ratios=[opts.marginal_height_ratio, 1.0],
        width_ratios=[1.0, opts.marginal_width_ratio],
        hspace=0.06,
        wspace=0.06,
    )
    ax_scatter = fig.add_subplot(inner[1, 0])
    ax_top = fig.add_subplot(inner[0, 0], sharex=ax_scatter)
    ax_right = fig.add_subplot(inner[1, 1], sharey=ax_scatter)
    ax_corner = fig.add_subplot(inner[0, 1])
    ax_corner.set_axis_off()

    lo = float(min(np.min(rx), np.min(wy)))
    hi = float(max(np.max(rx), np.max(wy)))
    span = hi - lo
    pad = 0.06 * span if span > 0.0 else 1.0
    lo_p = lo - pad
    hi_p = hi + pad

    n_bins = int(opts.marginal_hist_bins)
    edges = np.linspace(lo_p, hi_p, n_bins + 1, dtype=np.float64)
    edges_list = edges.tolist()

    ax_scatter.set_xlim(lo_p, hi_p)
    ax_scatter.set_ylim(lo_p, hi_p)
    ax_scatter.set_aspect("equal", adjustable="box")
    ax_scatter.scatter(
        rx,
        wy,
        s=opts.point_size,
        marker=opts.point_marker,
        facecolors=opts.point_facecolor,
        edgecolors=opts.point_edgecolor,
        zorder=3,
        clip_on=False,
    )
    ax_scatter.plot(
        [lo_p, hi_p],
        [lo_p, hi_p],
        color="0.45",
        linestyle="--",
        linewidth=0.95,
        alpha=0.75,
        zorder=1,
        clip_on=False,
    )
    ax_scatter.set_xlabel(opts.xlabel)
    ax_scatter.set_ylabel(opts.ylabel)
    ax_scatter.spines["top"].set_visible(False)
    ax_scatter.spines["right"].set_visible(False)

    counts_top, _, _ = ax_top.hist(
        rx,
        bins=edges_list,
        align="right",
        color="k",
        edgecolor="k",
        linewidth=0.25,
        zorder=1,
    )
    ax_top.set_axis_off()
    ct = np.asarray(counts_top)
    top_max = float(np.max(ct)) if ct.size else 0.0
    ax_top.set_ylim(0.0, top_max * 1.02 if top_max > 0.0 else 1.0)

    counts_r, _, _ = ax_right.hist(
        wy,
        bins=edges_list,
        orientation="horizontal",
        align="right",
        color="k",
        edgecolor="k",
        linewidth=0.25,
        zorder=1,
    )
    ax_right.set_axis_off()
    cr = np.asarray(counts_r)
    r_max = float(np.max(cr)) if cr.size else 0.0
    ax_right.set_xlim(0.0, r_max * 1.02 if r_max > 0.0 else 1.0)

    _draw_y_minus_x_hist_45deg_inset(
        ax_scatter,
        rx,
        wy,
        n_bins=int(opts.diff_hist_bins),
    )
