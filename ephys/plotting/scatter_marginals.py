"""Scatter plot with external marginal histograms and a y-x difference inset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpecFromSubplotSpec, SubplotSpec
from matplotlib.patches import Polygon
from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_U_PARA = np.array([1.0, 1.0], dtype=np.float64) / np.sqrt(2.0)


def _format_diff_inset_tick_label(d: float) -> str:
    """Format a difference-axis tick value for display."""
    v = float(d)
    if v == int(v):
        return str(int(v))
    return f"{v:g}"


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
    scatter_lim: tuple[float, float] | None = None
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    marginal_hist_bins: int = Field(default=10, ge=1)
    diff_hist_bins: int = Field(default=10, ge=1)
    diff_inset_center: float | None = None
    diff_inset_d_half_extent: float = Field(default=5.0, gt=0.0)
    diff_inset_count_extent: float | None = Field(default=None, gt=0.0)
    diff_inset_xtick_labels: bool = False
    point_marker: str = "o"
    point_size: float = Field(default=16.0, gt=0.0)
    point_facecolor: str = "none"
    point_edgecolor: str = "0.15"

    @model_validator(mode="after")
    def _validate_axis_limits(self) -> ScatterMarginalsOptions:
        """Require min < max on any explicit limit tuple."""
        for name, lim in (
            ("scatter_lim", self.scatter_lim),
            ("xlim", self.xlim),
            ("ylim", self.ylim),
        ):
            if lim is not None and float(lim[1]) <= float(lim[0]):
                raise ValueError(f"{name} max must be greater than min")
        return self


def resolve_scatter_axis_limits(
    x: np.ndarray,
    y: np.ndarray,
    *,
    scatter_lim: tuple[float, float] | None,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    pad_fraction: float = 0.06,
) -> tuple[float, float, float, float]:
    """Return ``(lo_x, hi_x, lo_y, hi_y)`` for the scatter panel and marginals.

    When ``scatter_lim`` is set, both axes use that range (no padding). Otherwise
    each axis uses ``xlim`` / ``ylim`` when provided, or padded data extrema.
    """
    if scatter_lim is not None:
        lo = float(scatter_lim[0])
        hi = float(scatter_lim[1])
        return lo, hi, lo, hi

    def _axis_bounds(
        values: np.ndarray,
        explicit: tuple[float, float] | None,
    ) -> tuple[float, float]:
        if explicit is not None:
            return float(explicit[0]), float(explicit[1])
        lo_v = float(np.min(values))
        hi_v = float(np.max(values))
        span = hi_v - lo_v
        pad = pad_fraction * span if span > 0.0 else 1.0
        return lo_v - pad, hi_v + pad

    lo_x, hi_x = _axis_bounds(x, xlim)
    lo_y, hi_y = _axis_bounds(y, ylim)
    return lo_x, hi_x, lo_y, hi_y


def unity_line_segment(
    lo_x: float,
    hi_x: float,
    lo_y: float,
    hi_y: float,
) -> tuple[float, float, float, float]:
    """Endpoints of ``y = x`` visible inside both axis limits."""
    lo = max(lo_x, lo_y)
    hi = min(hi_x, hi_y)
    return lo, lo, hi, hi


def _clip_to_limits(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Keep only values inside ``[lo, hi]`` (marginals match visible scatter)."""
    v = np.asarray(values, dtype=np.float64)
    return v[(v >= float(lo)) & (v <= float(hi))]


def marginal_histogram_edges(
    lo_x: float,
    hi_x: float,
    lo_y: float,
    hi_y: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin edges for top (x) and right (y) marginals, aligned to scatter limits."""
    n = int(n_bins)
    edges_x = np.linspace(float(lo_x), float(hi_x), n + 1, dtype=np.float64)
    edges_y = np.linspace(float(lo_y), float(hi_y), n + 1, dtype=np.float64)
    return edges_x, edges_y


def diff_inset_xy_for_difference(d: float, diagonal_pos: float) -> tuple[float, float]:
    """Map ``y - x = d`` to scatter coordinates on the unity line anchor.

    ``diagonal_pos`` is where ``y - x = 0`` on the line ``y = x`` (i.e. ``(s, s)``).
    """
    s = float(diagonal_pos)
    return (s - 0.5 * float(d), s + 0.5 * float(d))


def diff_inset_symmetric_edges(d_half_extent: float, n_bins: int) -> np.ndarray:
    """Bin edges for ``y - x`` over ``[-extent, +extent]``."""
    ext = float(d_half_extent)
    return np.linspace(-ext, ext, int(n_bins) + 1, dtype=np.float64)


def diff_inset_default_count_extent(d_half_extent: float, n_bins: int) -> float:
    """Default max bar height: one bin width along ``(1, 1)`` in data units."""
    n = max(int(n_bins), 1)
    return (2.0 * float(d_half_extent) / float(n)) * 0.85


def diff_inset_bar_polygon_vertices(
    e0: float,
    e1: float,
    count_height: float,
    diagonal_pos: float,
) -> np.ndarray:
    """Vertices of one inset bar in scatter data coordinates.

    ``count_height`` is the displacement along ``(1, 1)`` in data units.
    """
    x0, y0 = diff_inset_xy_for_difference(e0, diagonal_pos)
    x1, y1 = diff_inset_xy_for_difference(e1, diagonal_pos)
    offset = float(count_height) * _U_PARA
    return np.array(
        [
            [x0, y0],
            [x1, y1],
            [x1 + offset[0], y1 + offset[1]],
            [x0 + offset[0], y0 + offset[1]],
        ],
        dtype=np.float64,
    )


def _draw_diff_inset_axis(
    ax: Axes,
    *,
    diagonal_pos: float,
    d_half_extent: float,
    show_xtick_labels: bool = False,
    count_extent: float | None = None,
) -> None:
    """Draw the inset difference baseline in scatter data coordinates."""
    ext = float(d_half_extent)
    x0, y0 = diff_inset_xy_for_difference(-ext, diagonal_pos)
    x1, y1 = diff_inset_xy_for_difference(ext, diagonal_pos)
    axis_kw = {
        "color": "0.35",
        "linewidth": 0.9,
        "zorder": 5,
        "clip_on": True,
    }
    ax.plot([x0, x1], [y0, y1], **axis_kw)
    if not show_xtick_labels:
        return
    scale = float(count_extent) if count_extent is not None else ext * 0.2
    tick_len = 0.12 * scale
    tick = tick_len * _U_PARA
    label_shift = -1.15 * tick_len * _U_PARA
    for d in (-ext, ext):
        px, py = diff_inset_xy_for_difference(d, diagonal_pos)
        ax.plot(
            [px - tick[0], px + tick[0]],
            [py - tick[1], py + tick[1]],
            **axis_kw,
        )
        lx = px + label_shift[0]
        ly = py + label_shift[1]
        ha = "right" if d < 0.0 else "left"
        ax.text(
            lx,
            ly,
            _format_diff_inset_tick_label(d),
            fontsize=7.0,
            color="0.35",
            ha=ha,
            va="center",
            zorder=5,
            clip_on=True,
        )


def _draw_y_minus_x_hist_inset(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    diagonal_pos: float,
    d_half_extent: float,
    count_extent: float | None,
    n_bins: int,
    show_xtick_labels: bool = False,
) -> None:
    """Draw a y-x histogram inset centered at ``y - x = 0`` on the unity line.

    ``diagonal_pos`` sets the anchor ``(s, s)`` on ``y = x``. Bin edges span
    ``[-extent, +extent]`` in difference units. Bars grow along ``(1, 1)``.
    """
    diffs = (y - x).astype(np.float64, copy=False)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size < 1:
        return
    edges = diff_inset_symmetric_edges(d_half_extent, n_bins)
    counts, _ = np.histogram(diffs, bins=edges)
    count_extent_eff = (
        float(count_extent)
        if count_extent is not None
        else diff_inset_default_count_extent(d_half_extent, n_bins)
    )
    _draw_diff_inset_axis(
        ax,
        diagonal_pos=float(diagonal_pos),
        d_half_extent=d_half_extent,
        show_xtick_labels=show_xtick_labels,
        count_extent=count_extent_eff,
    )
    if counts.size == 0 or int(np.max(counts)) <= 0:
        return
    max_c = float(np.max(counts))
    for k in range(int(counts.size)):
        if counts[k] <= 0:
            continue
        e0, e1 = float(edges[k]), float(edges[k + 1])
        h = (float(counts[k]) / max_c) * count_extent_eff
        xy = diff_inset_bar_polygon_vertices(e0, e1, h, diagonal_pos)
        poly = Polygon(
            xy,
            closed=True,
            facecolor="k",
            edgecolor="k",
            linewidth=0.25,
            alpha=0.9,
            zorder=6,
            transform=ax.transData,
            clip_on=True,
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
    """Scatter (x vs. y) with external marginals and an optional y-x diff inset.

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

    Axis limits default to padded data extrema. Set ``scatter_lim=(min, max)`` for
    equal x/y limits, or ``xlim`` / ``ylim`` independently. Marginal histograms use
    the same limits as their scatter axis (x for top, y for right); only points
    inside those limits are counted, with bars aligned to bin centers.

    The difference inset is drawn only when ``diff_inset_center`` is set. It is
    always centered at ``y - x = 0`` on the unity line at ``(s, s)`` for
    ``s = diff_inset_center``, with bins symmetric in ``[-extent, +extent]``.
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

    lo_x, hi_x, lo_y, hi_y = resolve_scatter_axis_limits(
        rx,
        wy,
        scatter_lim=opts.scatter_lim,
        xlim=opts.xlim,
        ylim=opts.ylim,
    )
    u_lo, _, u_hi, _ = unity_line_segment(lo_x, hi_x, lo_y, hi_y)

    n_bins = int(opts.marginal_hist_bins)
    edges_x, edges_y = marginal_histogram_edges(lo_x, hi_x, lo_y, hi_y, n_bins)
    rx_marg = _clip_to_limits(rx, lo_x, hi_x)
    wy_marg = _clip_to_limits(wy, lo_y, hi_y)

    ax_scatter.set_xlim(lo_x, hi_x)
    ax_scatter.set_ylim(lo_y, hi_y)
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
        [u_lo, u_hi],
        [u_lo, u_hi],
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
        rx_marg,
        bins=edges_x.tolist(),
        align="mid",
        color="k",
        edgecolor="k",
        linewidth=0.25,
        zorder=1,
    )
    ax_top.set_xlim(lo_x, hi_x)
    ax_top.set_axis_off()
    ct = np.asarray(counts_top)
    top_max = float(np.max(ct)) if ct.size else 0.0
    ax_top.set_ylim(0.0, top_max * 1.02 if top_max > 0.0 else 1.0)

    counts_r, _, _ = ax_right.hist(
        wy_marg,
        bins=edges_y.tolist(),
        orientation="horizontal",
        align="mid",
        color="k",
        edgecolor="k",
        linewidth=0.25,
        zorder=1,
    )
    ax_right.set_ylim(lo_y, hi_y)
    ax_right.set_axis_off()
    cr = np.asarray(counts_r)
    r_max = float(np.max(cr)) if cr.size else 0.0
    ax_right.set_xlim(0.0, r_max * 1.02 if r_max > 0.0 else 1.0)

    if opts.diff_inset_center is not None:
        _draw_y_minus_x_hist_inset(
            ax_scatter,
            rx,
            wy,
            diagonal_pos=float(opts.diff_inset_center),
            d_half_extent=float(opts.diff_inset_d_half_extent),
            count_extent=opts.diff_inset_count_extent,
            n_bins=int(opts.diff_hist_bins),
            show_xtick_labels=opts.diff_inset_xtick_labels,
        )
