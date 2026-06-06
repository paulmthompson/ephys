"""Log-log scatter plot with an optional unity (y = x) guide line."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from pydantic import BaseModel, Field, model_validator

from ephys.plotting.scatter_marginals import (
    ScatterMarginalsData,
    unity_line_segment,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class LogLogScatterOptions(BaseModel):
    """Axis labels, limits, and scatter point styling for log-log panels."""

    model_config = {"frozen": True}

    xlabel: str = "X"
    ylabel: str = "Y"
    scatter_lim: tuple[float, float] | None = None
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    pad_fraction: float = Field(
        default=0.08,
        ge=0.0,
        description="Symmetric log-space padding when limits are data-derived.",
    )
    point_marker: str = "o"
    point_size: float = Field(default=16.0, gt=0.0)
    point_facecolor: str = "none"
    point_edgecolor: str = "0.15"
    unity_line_color: str = "0.45"
    unity_line_style: str = "--"
    unity_line_width: float = Field(default=0.95, gt=0.0)
    unity_line_alpha: float = Field(default=0.75, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_axis_limits(self) -> LogLogScatterOptions:
        """Require min < max on any explicit limit tuple and positive bounds."""
        for name, lim in (
            ("scatter_lim", self.scatter_lim),
            ("xlim", self.xlim),
            ("ylim", self.ylim),
        ):
            if lim is None:
                continue
            lo, hi = float(lim[0]), float(lim[1])
            if hi <= lo:
                raise ValueError(f"{name} max must be greater than min")
            if lo <= 0.0:
                raise ValueError(f"{name} limits must be strictly positive")
        return self


def positive_finite_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """True where ``x`` and ``y`` are finite and strictly positive."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    return np.isfinite(x_arr) & np.isfinite(y_arr) & (x_arr > 0.0) & (y_arr > 0.0)


def resolve_log_log_axis_limits(
    x: np.ndarray,
    y: np.ndarray,
    *,
    scatter_lim: tuple[float, float] | None,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    pad_fraction: float = 0.08,
) -> tuple[float, float, float, float]:
    """Return ``(lo_x, hi_x, lo_y, hi_y)`` for log-scaled axes.

    When ``scatter_lim`` is set, both axes use that range (no padding). Otherwise
    each axis uses ``xlim`` / ``ylim`` when provided, or padded data extrema in
    log space.
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
        if lo_v <= 0.0 or hi_v <= 0.0:
            raise ValueError("values must be strictly positive for log limits")
        log_lo = float(np.log10(lo_v))
        log_hi = float(np.log10(hi_v))
        span = log_hi - log_lo
        pad = float(pad_fraction) * span if span > 0.0 else 0.1
        return 10.0 ** (log_lo - pad), 10.0 ** (log_hi + pad)

    lo_x, hi_x = _axis_bounds(x, xlim)
    lo_y, hi_y = _axis_bounds(y, ylim)
    return lo_x, hi_x, lo_y, hi_y


def _limits_are_equal(
    lo_x: float,
    hi_x: float,
    lo_y: float,
    hi_y: float,
) -> bool:
    """True when scatter x and y share the same numeric limits."""
    return bool(
        np.isclose(lo_x, lo_y, rtol=0.0, atol=1e-12)
        and np.isclose(hi_x, hi_y, rtol=0.0, atol=1e-12)
    )


def draw_log_log_scatter_empty_placeholder(
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


def _style_log_log_scatter_axes(
    ax: Axes,
    *,
    lo_x: float,
    hi_x: float,
    lo_y: float,
    hi_y: float,
    xlabel: str,
    ylabel: str,
) -> None:
    """Apply log scales, labels, and spine styling."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(lo_y, hi_y)
    if _limits_are_equal(lo_x, hi_x, lo_y, hi_y):
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_log_log_scatter_into(
    fig: Figure,
    cell: SubplotSpec,
    data: ScatterMarginalsData,
    options: LogLogScatterOptions | None = None,
) -> None:
    """Scatter ``x`` vs. ``y`` on log-log axes with a dashed unity line.

    Parameters
    ----------
    fig
        Parent figure.
    cell
        SubplotSpec reserved for the scatter panel.
    data
        Per-sample ``x`` and ``y`` arrays of equal length.
    options
        Axis labels, limits, and scatter styling. Uses
        :class:`LogLogScatterOptions` defaults when ``None``.

    Notes
    -----
    Only finite samples with ``x > 0`` and ``y > 0`` are plotted. Axis limits
    default to padded data extrema in log space. Set ``scatter_lim=(min, max)``
    for equal x/y limits, or ``xlim`` / ``ylim`` independently.
    """
    opts = options or LogLogScatterOptions()
    mask = positive_finite_mask(data.x, data.y)
    if not np.any(mask):
        draw_log_log_scatter_empty_placeholder(
            fig,
            cell,
            "No positive finite x/y values",
        )
        return

    rx = np.asarray(data.x[mask], dtype=np.float64)
    wy = np.asarray(data.y[mask], dtype=np.float64)
    ax = fig.add_subplot(cell)

    lo_x, hi_x, lo_y, hi_y = resolve_log_log_axis_limits(
        rx,
        wy,
        scatter_lim=opts.scatter_lim,
        xlim=opts.xlim,
        ylim=opts.ylim,
        pad_fraction=opts.pad_fraction,
    )
    u_lo, _, u_hi, _ = unity_line_segment(lo_x, hi_x, lo_y, hi_y)

    ax.scatter(
        rx,
        wy,
        s=opts.point_size,
        marker=opts.point_marker,
        facecolors=opts.point_facecolor,
        edgecolors=opts.point_edgecolor,
        zorder=3,
        clip_on=False,
    )
    ax.plot(
        [u_lo, u_hi],
        [u_lo, u_hi],
        color=opts.unity_line_color,
        linestyle=opts.unity_line_style,
        linewidth=opts.unity_line_width,
        alpha=opts.unity_line_alpha,
        zorder=1,
        clip_on=False,
    )
    _style_log_log_scatter_axes(
        ax,
        lo_x=lo_x,
        hi_x=hi_x,
        lo_y=lo_y,
        hi_y=hi_y,
        xlabel=opts.xlabel,
        ylabel=opts.ylabel,
    )
