"""Matplotlib plotting helpers for electrophysiology figures."""

from .broken_axes import add_vertical_broken_axis_style
from .grid_axes import style_x_axis_with_dedicated_label_row
from .psth_raster import plot_psth, plot_raster

__all__ = [
    "add_vertical_broken_axis_style",
    "plot_psth",
    "plot_raster",
    "style_x_axis_with_dedicated_label_row",
]
