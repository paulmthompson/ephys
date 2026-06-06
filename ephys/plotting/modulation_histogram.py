"""Histogram of per-unit log₂ fold-change modulation between paired rates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from pydantic import BaseModel, Field, model_validator

from ephys.plotting.log_log_scatter import positive_finite_mask

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class ModulationHistogramOptions(BaseModel):
    """Layout and styling for a log₂ fold-change histogram."""

    model_config = {"frozen": True}

    xlabel: str = "log₂(WIA / Rest)"
    ylabel: str = "Neurons"
    n_bins: int = Field(default=20, ge=1)
    xlim: tuple[float, float] | None = None
    pad_fraction: float = Field(
        default=0.06,
        ge=0.0,
        description="Linear padding on x when limits are data-derived.",
    )
    show_zero_line: bool = True
    zero_line_color: str = "0.45"
    zero_line_style: str = "--"
    zero_line_width: float = Field(default=0.95, gt=0.0)
    zero_line_alpha: float = Field(default=0.75, ge=0.0, le=1.0)
    bar_color: str = "k"
    bar_edgecolor: str = "k"
    bar_linewidth: float = Field(default=0.25, ge=0.0)

    @model_validator(mode="after")
    def _validate_xlim(self) -> ModulationHistogramOptions:
        """Require min < max on any explicit x limit tuple."""
        if self.xlim is not None and float(self.xlim[1]) <= float(self.xlim[0]):
            raise ValueError("xlim max must be greater than min")
        return self


def log2_ratio(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return ``log₂(y / x)`` for paired positive samples.

    Parameters
    ----------
    y
        Numerator rates (e.g. WIA).
    x
        Denominator rates (e.g. rest). Must be the same length as ``y``.

    Returns
    -------
    numpy.ndarray
        ``log₂(y / x)`` as ``float64``.

    Notes
    -----
    Caller must ensure ``x`` and ``y`` are finite and strictly positive.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    if y_arr.shape != x_arr.shape:
        raise ValueError("x and y must have the same shape")
    return np.log2(y_arr / x_arr)


def log2_ratio_from_paired(x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """Compute ``log₂(y / x)`` for finite, strictly positive pairs.

    Parameters
    ----------
    x
        Denominator rates (e.g. rest).
    y
        Numerator rates (e.g. WIA).

    Returns
    -------
    numpy.ndarray or None
        One log₂ ratio per valid pair, or ``None`` when no pairs qualify.
    """
    mask = positive_finite_mask(x, y)
    if not np.any(mask):
        return None
    return log2_ratio(np.asarray(y[mask], dtype=np.float64), np.asarray(x[mask], dtype=np.float64))


def resolve_histogram_x_limits(
    values: np.ndarray,
    *,
    xlim: tuple[float, float] | None,
    pad_fraction: float = 0.06,
) -> tuple[float, float]:
    """Return ``(lo, hi)`` bin edges span for the histogram x-axis."""
    if xlim is not None:
        return float(xlim[0]), float(xlim[1])
    v = np.asarray(values, dtype=np.float64)
    lo_v = float(np.min(v))
    hi_v = float(np.max(v))
    span = hi_v - lo_v
    pad = float(pad_fraction) * span if span > 0.0 else 0.5
    return lo_v - pad, hi_v + pad


def histogram_edges(lo: float, hi: float, n_bins: int) -> np.ndarray:
    """Uniform bin edges over ``[lo, hi]``."""
    return np.linspace(float(lo), float(hi), int(n_bins) + 1, dtype=np.float64)


def draw_modulation_histogram_empty_placeholder(
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


def _style_modulation_histogram_axes(
    ax: Axes,
    *,
    lo: float,
    hi: float,
    xlabel: str,
    ylabel: str,
) -> None:
    """Apply labels, limits, and spine styling."""
    ax.set_xlim(lo, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_modulation_histogram_into(
    fig: Figure,
    cell: SubplotSpec,
    values: np.ndarray,
    options: ModulationHistogramOptions | None = None,
) -> None:
    """Draw a histogram of log₂ fold-change values.

      Parameters
      ----------
      fig
          Parent figure.
      cell
          SubplotSpec reserved for the histogram panel.
      values
          One modulation metric per unit (e.g. ``log₂(WIA / Rest)``).
      options
          Bin count, axis labels, limits, and bar styling. Uses
          :class:`ModulationHistogramOptions` defaults when ``None``.

      Notes
      -----
      The x-axis is linear in log₂ ratio (not log-scaled). A dashed vertical
    reference line at zero marks no modulation when ``show_zero_line`` is set.
    """
    opts = options or ModulationHistogramOptions()
    v = np.asarray(values, dtype=np.float64)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        draw_modulation_histogram_empty_placeholder(
            fig,
            cell,
            "No finite modulation values",
        )
        return

    lo, hi = resolve_histogram_x_limits(
        finite,
        xlim=opts.xlim,
        pad_fraction=opts.pad_fraction,
    )
    edges = histogram_edges(lo, hi, opts.n_bins)
    ax = fig.add_subplot(cell)
    counts, _, _ = ax.hist(
        finite,
        bins=edges.tolist(),
        align="mid",
        color=opts.bar_color,
        edgecolor=opts.bar_edgecolor,
        linewidth=opts.bar_linewidth,
        zorder=2,
    )
    ct = np.asarray(counts)
    ymax = float(np.max(ct)) if ct.size else 0.0
    ax.set_ylim(0.0, ymax * 1.02 if ymax > 0.0 else 1.0)

    if opts.show_zero_line and lo < 0.0 < hi:
        ax.axvline(
            0.0,
            color=opts.zero_line_color,
            linestyle=opts.zero_line_style,
            linewidth=opts.zero_line_width,
            alpha=opts.zero_line_alpha,
            zorder=1,
        )

    _style_modulation_histogram_axes(
        ax,
        lo=lo,
        hi=hi,
        xlabel=opts.xlabel,
        ylabel=opts.ylabel,
    )
