"""Layout utilities for multi-panel grids and publication-quality figures.

This module provides tools for managing complex, multi-panel layouts where
standard Matplotlib decorations (like labels and ticks) often collide or clip.
It promotes a robust layout strategy using dedicated spacer axes to ensure
consistent typography and spacing in tight grids.

Summary of tools:

* :func:`style_x_axis_with_dedicated_label_row`: Styles data-axis ticks while
  placing the axis label in a separate spacer row. This prevents clipping and
  allows for precise control over vertical spacing when panels are stacked.

These utilities are intentionally minimal and depend only on Matplotlib,
encouraging consistent layout patterns across various experimental views.
"""

from __future__ import annotations

from typing import Sequence

from matplotlib.axes import Axes


def style_x_axis_with_dedicated_label_row(
    ax_data: Axes,
    ax_label: Axes,
    label_text: str,
    *,
    xticks: Sequence[float],
    xticklabels: Sequence[str],
    fontsize: float,
    tick_length: float = 3.5,
    tick_width: float = 0.75,
    tick_pad: float = 1.0,
    label_y_anchor: float = 0.0,
) -> None:
    """Style x-axis ticks on data axes and draw labels in a dedicated spacer row.

    This follows the robust layout strategy of anchoring manual text to the bottom
    edge of a dedicated spacer cell below the data axes so glyphs grow upward and
    avoid colliding with x-axis ticks, rather than relying on ``set_xlabel`` or
    ``clip_on=False`` on the data axes.

    Parameters
    ----------
    ax_data
        The data axis containing the plots, where x-ticks will be drawn.
    ax_label
        A dedicated axis (usually ``axis("off")``) below ``ax_data`` where the
        label is drawn.
    label_text
        The text for the x-axis label.
    xticks
        Locations of the ticks.
    xticklabels
        Labels for the ticks.
    fontsize
        Font size for both tick labels and the axis label.
    tick_length, tick_width, tick_pad
        Formatting parameters for ticks on the data axes.
    label_y_anchor
        Vertical position for the label text in ``ax_label``.
    """
    ax_data.set_xticks(list(xticks))
    ax_data.set_xticklabels(list(xticklabels))
    ax_data.tick_params(
        axis="x",
        which="major",
        bottom=True,
        labelbottom=True,
        length=tick_length,
        width=tick_width,
        pad=tick_pad,
        labelsize=fontsize,
    )
    ax_data.spines["bottom"].set_visible(True)

    ax_label.text(
        0.5,
        label_y_anchor,
        label_text,
        ha="center",
        va="bottom",
        transform=ax_label.transAxes,
        fontsize=fontsize,
        clip_on=False,
    )
