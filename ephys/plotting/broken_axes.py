"""Styling utilities for broken-axis visualizations.

This module provides tools for creating visual breaks between adjacent axes,
a common pattern when displaying data with non-contiguous ranges or when
spanning multiple panels.

Key functionality:

* Broken-Axis Markers: :func:`add_vertical_broken_axis_style` hides internal
  spines and adds diagonal break marks between two horizontally adjacent axes.

These markers use axes coordinates to ensure precise alignment along the
shared boundary regardless of the underlying data scales.
"""

from __future__ import annotations

from matplotlib.axes import Axes

_DIAG_SLOPE = 0.5
_MEW = 0.9


def add_vertical_broken_axis_style(
    ax_left: Axes,
    ax_right: Axes,
    vertical_positions: list[float],
    *,
    color: str = "black",
    marker_size: float = 9.0,
) -> None:
    """Hide inner spines and draw diagonal marks on the shared vertical edge.

    Parameters
    ----------
    ax_left
        Left axes; its right spine is hidden.
    ax_right
        Right axes; its left spine is hidden.
    vertical_positions
        Y positions in axes fraction (0–1) for each diagonal mark pair.
    color
        Color for markers and marker edges.
    marker_size
        Marker size in points, forwarded to `Axes.plot`.

    Notes
    -----
    Diagonal ticks use axes coordinates so marks align with the boundary
    between ``ax_left`` (right edge at *x* = 1 in axes fraction) and
    ``ax_right`` (left edge at *x* = 0).
    """
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)

    d = _DIAG_SLOPE
    for y in vertical_positions:
        ax_left.plot(
            [1.0],
            [y],
            transform=ax_left.transAxes,
            marker=[(-1, -d), (1, d)],
            markersize=marker_size,
            linestyle="none",
            color=color,
            mec=color,
            mew=_MEW,
            clip_on=False,
            zorder=200,
        )
        ax_right.plot(
            [0.0],
            [y],
            transform=ax_right.transAxes,
            marker=[(-1, -d), (1, d)],
            markersize=marker_size,
            linestyle="none",
            color=color,
            mec=color,
            mew=_MEW,
            clip_on=False,
            zorder=200,
        )
