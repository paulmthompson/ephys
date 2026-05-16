"""Matplotlib plotting utilities for electrophysiology figures.

This package provides tools for rendering high-quality, publication-ready
visualizations of electrophysiology data. It is organized into several
functional areas:

* Aligned Views: Functions for plotting PSTHs and spike rasters
  (:mod:`.psth_raster`).
* Trace Rendering: Tools for high-density voltage traces and margin scales
  (:mod:`.traces`).
* Layout & Geometry: Helpers for managing vertical space, trial merging,
  and precise marker placement in dense grids (:mod:`.raster_layout`).
* Grid Utilities: Robust strategies for axis labeling and multi-panel
  alignment (:mod:`.grid_axes`, :mod:`.broken_axes`).

These utilities focus on visual consistency, precise control over layout
spacing, and efficient handling of large-scale neural data.
"""

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
