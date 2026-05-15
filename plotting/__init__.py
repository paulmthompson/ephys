"""Matplotlib plotting helpers for electrophysiology figures."""

from .broken_axes import add_vertical_broken_axis_style
from .grid_axes import style_x_axis_with_dedicated_label_row
from .psth_raster import plot_psth, plot_raster
from .raster_layout import (
    bin_spike_lists,
    effective_raster_linelength,
    lineoffsets_equal_height_band,
    merge_spikes_chunk,
    sorted_subset_trials,
)

__all__ = [
    "add_vertical_broken_axis_style",
    "bin_spike_lists",
    "effective_raster_linelength",
    "lineoffsets_equal_height_band",
    "merge_spikes_chunk",
    "plot_psth",
    "plot_raster",
    "sorted_subset_trials",
    "style_x_axis_with_dedicated_label_row",
]
