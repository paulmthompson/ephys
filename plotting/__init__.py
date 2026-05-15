"""Matplotlib plotting helpers for electrophysiology figures."""

from .broken_axes import add_vertical_broken_axis_style
from .grid_axes import style_x_axis_with_dedicated_label_row

__all__ = [
    "add_vertical_broken_axis_style",
    "style_x_axis_with_dedicated_label_row",
]
