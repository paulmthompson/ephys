"""Voltage trace plotting and manuscript-style trace margin layout.

This module provides tools for rendering high-quality voltage traces with
dedicated margin axes for scale bars and labels. It uses a nested grid layout
to ensure that scale graphics remain clear and consistently placed.

Summary of functionality:

* :func:`plot_voltage_trace`: Renders a voltage vector as a min/max envelope.
* :class:`VoltageTraceOptions`: Configuration for grid ratios, fonts,
  and scale-bar styling.
* :func:`add_stim_trace_margin_block`: Creates a nested ``3×2`` grid for a
  trace, stimulus strip, and margin labels/scales.
* :func:`populate_stim_trace_margin_slot`: Orchestrates the plotting of voltage,
  stimulus, and scale bars in a single call.
* Scale bar helpers: Specialized functions for drawing µV and ms scale bars
  on dedicated margin axes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.axes
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from matplotlib.transforms import Transform, blended_transform_factory
from pydantic import BaseModel

from ephys.processing.envelope import get_min_max_envelope


def _plot_scale_bar_stroke(
    ax: Axes,
    x: tuple[float, float],
    y: tuple[float, float],
    *,
    transform: Transform | None,
    color: str,
    linewidth_pt: float,
    zorder: int = 10,
) -> None:
    """Plot a line segment; ``snap=True`` avoids scale-bar stroke artifacts."""
    ax.plot(
        x,
        y,
        transform=transform if transform is not None else ax.transData,
        color=color,
        linewidth=linewidth_pt,
        solid_capstyle="butt",
        zorder=zorder,
        clip_on=False,
        snap=True,  # This prevents the "shifting" and "thickening" look
    )


def _resolved_scale_bar_linewidth_pt(ax: Axes, s: VoltageTraceOptions) -> float:
    """Return scale-bar ``linewidth`` in Matplotlib points for ``ax``."""
    if s.scale_bar_linewidth_figheight_frac is not None:
        fig = ax.figure
        return float(s.scale_bar_linewidth_figheight_frac * fig.get_figheight() * 72.0)
    return float(s.scale_bar_linewidth_pt)


class VoltageTraceOptions(BaseModel):
    """Tuning for a voltage margin column (:class:`StimTraceActivityTopSlot`).

    Grid ratios are stable defaults for the nested ``3×2`` block; typography,
    scale extents, and colors are the usual knobs to change per figure.

    ``scale_bar_linewidth_pt`` sets the stroke for both the µV and time scale
    bars. Matplotlib ``linewidth`` is in **typographic points** (1/72 inch),
    i.e. display/canvas space: it does not stretch with data limits.

    Alternatively set ``scale_bar_linewidth_figheight_frac`` to a non-``None``
    value: the stroke width in points becomes that fraction times the figure
    height in points (``frac * fig.get_figheight() * 72``), which tracks figure
    size; in that case ``scale_bar_linewidth_pt`` is ignored.
    """

    model_config = {"frozen": True}

    hspace_trace: float = 0.05
    stim_ratio: float = 0.08
    trace_ratio: float = 1.0
    time_row_ratio: float = 0.22
    left_column_ratio: float = 0.2
    wspace_label_trace: float = 0.02
    fontsize: float = 7.0
    microvolt_scale_height: float = 50.0
    time_scale_ms: float = 100.0
    trace_color: str = "#646464"
    label_color: str = "0.15"
    scale_line_color: str = "black"
    scale_bar_linewidth_pt: float = 0.75
    scale_bar_linewidth_figheight_frac: float | None = None


DEFAULT_VOLTAGE_TRACE_OPTIONS = VoltageTraceOptions()


@dataclass(frozen=True)
class StimTraceActivityTopSlot:
    """Stimulus strip + voltage trace with reserved margin axes.

    Nested ``3×2`` grid: row 0 — strip label (left) and stimulus strip (right);
    row 1 — microvolt scale (left) and voltage trace (right); row 2 — spacer
    (left) and time-scale bar (right). ``ax_trace`` shares *x* with
    ``ax_strip`` and ``ax_time_scale``; ``ax_microvolt_scale`` shares *y* with
    ``ax_trace``.
    """

    ax_strip: Axes
    ax_trace: Axes
    ax_strip_label: Axes
    ax_microvolt_scale: Axes
    ax_time_scale: Axes


def add_stim_trace_margin_block(
    fig: Figure,
    gs_cell: SubplotSpec,
    *,
    style: VoltageTraceOptions | None = None,
) -> StimTraceActivityTopSlot:
    """Build the ``3×2`` stimulus / trace / scale margin block in one cell.

    Parameters
    ----------
    fig, gs_cell
        Parent figure and subplot cell to fill.
    style
        Grid and spacing for the margin block. If ``None``, uses
        :data:`DEFAULT_VOLTAGE_TRACE_OPTIONS`.
    """
    s = style or DEFAULT_VOLTAGE_TRACE_OPTIONS
    gs_trace = gs_cell.subgridspec(
        3,
        2,
        height_ratios=[s.stim_ratio, s.trace_ratio, s.time_row_ratio],
        width_ratios=[s.left_column_ratio, 1.0],
        hspace=s.hspace_trace,
        wspace=s.wspace_label_trace,
    )
    ax_trace = fig.add_subplot(gs_trace[1, 1])
    ax_strip = fig.add_subplot(gs_trace[0, 1], sharex=ax_trace)
    ax_time_scale = fig.add_subplot(gs_trace[2, 1], sharex=ax_trace)
    ax_strip_label = fig.add_subplot(gs_trace[0, 0])

    ax_microvolt_scale = fig.add_subplot(gs_trace[1, 0], sharey=ax_trace)
    ax_time_spacer = fig.add_subplot(gs_trace[2, 0])
    ax_strip_label.axis("off")
    ax_microvolt_scale.set_xlim(0.0, 1.0)
    ax_microvolt_scale.axis("off")
    ax_time_scale.set_ylim(0.0, 1.0)
    ax_time_scale.axis("off")
    ax_time_spacer.axis("off")
    return StimTraceActivityTopSlot(
        ax_strip=ax_strip,
        ax_trace=ax_trace,
        ax_strip_label=ax_strip_label,
        ax_microvolt_scale=ax_microvolt_scale,
        ax_time_scale=ax_time_scale,
    )


def draw_margin_strip_label(
    ax_strip_label: Axes,
    text: str,
    *,
    style: VoltageTraceOptions | None = None,
) -> None:
    """Draw centered label text in the left strip-label axes.

    This is the top-row, left-column cell from
    :class:`StimTraceActivityTopSlot`.

    Parameters
    ----------
    ax_strip_label
        From :class:`StimTraceActivityTopSlot`.
    text
        Shown in axes coordinates at the center of the cell.
    style
        Uses ``fontsize`` and ``label_color``. If ``None``, uses
        :data:`DEFAULT_VOLTAGE_TRACE_OPTIONS`.
    """
    s = style or DEFAULT_VOLTAGE_TRACE_OPTIONS
    ax_strip_label.text(
        0.5,
        0.5,
        text,
        transform=ax_strip_label.transAxes,
        ha="center",
        va="center",
        fontsize=s.fontsize,
        color=s.label_color,
    )


def populate_stim_trace_margin_slot(
    slot: StimTraceActivityTopSlot,
    *,
    style: VoltageTraceOptions | None = None,
    voltage: np.ndarray,
    snippet_start_idx: int,
    strip_label: str | None = None,
    draw_stimulus: Callable[[Axes], None] | None = None,
    fs: float = 30000.0,
    bin_size: int = 10,
) -> None:
    """Populate a margin block after :func:`add_stim_trace_margin_block`.

    Runs in order: voltage envelope on ``ax_trace``, optional stimulus on
    ``ax_strip``, optional strip label, microvolt scale, time scale.

    Parameters
    ----------
    slot
        Axes bundle from :func:`add_stim_trace_margin_block`.
    style
        Passed through to drawing helpers. If ``None``, uses
        :data:`DEFAULT_VOLTAGE_TRACE_OPTIONS`.
    voltage, snippet_start_idx
        Arguments to :func:`plot_voltage_trace` for ``slot.ax_trace``.
    strip_label
        If set, drawn with :func:`draw_margin_strip_label` on
        ``slot.ax_strip_label``.
    draw_stimulus
        If set, called as ``draw_stimulus(slot.ax_strip)``. Use
        :func:`figures.plotting.stimuli.plot_contact_stimulus`,
        :func:`figures.plotting.stimuli.plot_laser_stimulus`, or a lambda that
        closes over paths and index windows.
    fs, bin_size
        Passed to :func:`plot_voltage_trace`.
    """
    plot_voltage_trace(
        slot.ax_trace,
        voltage,
        snippet_start_idx,
        fs=fs,
        bin_size=bin_size,
        style=style,
    )
    if draw_stimulus is not None:
        draw_stimulus(slot.ax_strip)
    if strip_label:
        draw_margin_strip_label(slot.ax_strip_label, strip_label, style=style)
    draw_microvolt_scale_bar_uv_axis(
        slot.ax_microvolt_scale,
        slot.ax_trace,
        style=style,
    )
    draw_time_scale_bar_trace_bottom_cell(slot.ax_time_scale, style=style)


def plot_voltage_trace(
    ax: matplotlib.axes.Axes,
    voltage: np.ndarray,
    start_idx: int,
    *,
    fs: float = 30000.0,
    bin_size: int = 10,
    style: VoltageTraceOptions | None = None,
    trace_color: str | None = None,
) -> None:
    """Take voltage vector and draw a min/max envelope on ``ax``.

    Parameters
    ----------
    ax
        Target axes.
    voltage
        A 1D voltage array (snippet or longer recording).
    start_idx
        Added to each envelope bin start index before dividing by ``fs``, so
        the trace is drawn in **absolute** DAQ time when ``voltage`` is a
        snippet slice but you still want axis values to match the original
        recording. For a window-only array where the axis should run from
        snippet onset, pass ``0`` (see Figure 4 stacked snippets).
    fs, bin_size
        Sampling rate (Hz) and envelope bin size; see
        :func:`~ephys.processing.envelope.get_min_max_envelope`.
    style
        When given, ``trace_color`` defaults to ``style.trace_color``.
    trace_color
        If set, overrides ``style.trace_color`` (and the default gray) for the
        envelope and outline strokes.
    """

    voltage = np.asarray(voltage).ravel()
    indices, mins, maxs = get_min_max_envelope(voltage, bin_size)
    time = (indices + start_idx) / fs

    s = style or DEFAULT_VOLTAGE_TRACE_OPTIONS
    color = trace_color if trace_color is not None else s.trace_color

    ax.fill_between(time, mins, maxs, color=color, alpha=1.0, linewidth=0)
    ax.plot(time, mins, color=color, alpha=1.0, linewidth=0.5)
    ax.plot(time, maxs, color=color, alpha=1.0, linewidth=0.5)

    v_max = float(np.max(np.abs(voltage)))
    v_min = -v_max
    v_pad = (v_max - v_min) * 0.05
    ax.set_ylim(v_min - v_pad, v_max + v_pad)
    ax.axis("off")


def infer_microvolts_per_data_unit(ax: Axes) -> float:
    """Infer how many microvolts one y-axis data unit represents.

    Snippets are often stored in µV, but some paths expose **millivolts** on
    the axis (values of order 1). In that case one data unit is 1000 µV.

    Parameters
    ----------
    ax
        Axes after :func:`plot_voltage_trace` (symmetric ``ylim`` about zero).

    Returns
    -------
    float
        Microvolts per one y-axis unit (``1.0`` or ``1000.0``).
    """
    ymin, ymax = ax.get_ylim()
    bound = max(abs(ymin), abs(ymax))
    if bound < 3.0:
        return 1000.0
    return 1.0


def draw_microvolt_scale_bar_uv_axis(
    ax_uv: Axes,
    ax_voltage: Axes,
    *,
    style: VoltageTraceOptions | None = None,
    microvolts_per_axis_unit: float | None = None,
    bar_x: float = 0.86,
    label_anchor_x: float = 0.82,
) -> None:
    """Draw a vertical microvolt scale in the narrow left column (``sharey``).

    ``ax_uv`` uses *x* in ``[0, 1]`` as a dummy column; *y* matches
    ``ax_voltage`` via ``sharey``. The label is anchored with ``ha='right'``
    immediately to the left of the bar (both use nearby *x* in data space).

    Parameters
    ----------
    ax_uv
        Dedicated scale axes (``sharey`` with ``ax_voltage``).
    ax_voltage
        Voltage trace axes (used to infer µV vs mV scaling when needed).
    style
        Font size, bar height (µV), label and line colors. If ``None``, uses
        :data:`DEFAULT_VOLTAGE_TRACE_OPTIONS`.
    microvolts_per_axis_unit
        If ``None``, inferred from ``ax_voltage`` limits.
    bar_x
        Data-space *x* in ``ax_uv`` for the vertical bar (toward the trace).
    label_anchor_x
        Data-space *x* for the text anchor (``ha='right'``), slightly left of
        ``bar_x``.
    """
    s = style or DEFAULT_VOLTAGE_TRACE_OPTIONS
    bar_height_microvolts = s.microvolt_scale_height
    fontsize = s.fontsize
    if microvolts_per_axis_unit is None:
        microvolts_per_axis_unit = infer_microvolts_per_data_unit(ax_voltage)
    height_data = bar_height_microvolts / microvolts_per_axis_unit
    ymin, ymax = ax_uv.get_ylim()
    y_mid = 0.5 * (ymin + ymax)
    y0 = y_mid - 0.5 * height_data
    y1 = y_mid + 0.5 * height_data
    lw = _resolved_scale_bar_linewidth_pt(ax_uv, s)

    trans = blended_transform_factory(ax_uv.transAxes, ax_uv.transData)
    # Now x is in 0-1 (axes coords), y is in data (voltage)
    _plot_scale_bar_stroke(
        ax_uv,
        (
            0.9,
            0.9,
        ),  # bar_x should now be e.g., 0.95 (far right of the margin ax)
        (y0, y1),
        transform=trans,
        color=s.scale_line_color,
        linewidth_pt=lw,
    )
    ax_uv.text(
        min(label_anchor_x, bar_x - 1e-3),
        y_mid,
        f"{int(bar_height_microvolts)} µV",
        ha="right",
        va="center",
        fontsize=fontsize,
        color=s.label_color,
        zorder=10,
    )


def draw_time_scale_bar_trace_bottom_cell(
    ax_time: Axes,
    *,
    style: VoltageTraceOptions | None = None,
    axes_y_bar: float = 1.0,
    axes_y_label: float = 0.82,
    pad_s: float | None = None,
) -> None:
    """Draw a time scale bar in the bottom margin cell (``sharex`` with trace).

    By default the bar and label are placed high in ``ax_time`` (large
    ``axes_y_*`` in blended coordinates) so they sit just under the voltage
    trace row with minimal empty space between.

    Parameters
    ----------
    ax_time
        Bottom-row axes sharing *x* with the voltage trace.
    style
        Font size, bar length from ``time_scale_ms``, label and line colors. If
        ``None``, uses :data:`DEFAULT_VOLTAGE_TRACE_OPTIONS`.
    axes_y_bar, axes_y_label
        Axes-coordinate *y* for the bar and for the label anchor (blended with
        data *x*).
    pad_s
        Gap from the right ``xlim`` to the bar end; default 1.5% of span.
    """
    s = style or DEFAULT_VOLTAGE_TRACE_OPTIONS
    width_s = s.time_scale_ms / 1000.0
    fontsize = s.fontsize
    trans = blended_transform_factory(ax_time.transData, ax_time.transAxes)
    xmin, xmax = ax_time.get_xlim()
    span = xmax - xmin
    if pad_s is None:
        pad_s = 0.015 * span
    x1 = xmax - pad_s
    x0 = x1 - width_s
    label_ms = round(width_s * 1000.0)
    lw = _resolved_scale_bar_linewidth_pt(ax_time, s)
    _plot_scale_bar_stroke(
        ax_time,
        (x0, x1),
        (axes_y_bar, axes_y_bar),
        transform=trans,
        color=s.scale_line_color,
        linewidth_pt=lw,
    )
    ax_time.text(
        0.5 * (x0 + x1),
        axes_y_label,
        f"{label_ms} ms",
        transform=trans,
        ha="center",
        va="top",
        fontsize=fontsize,
        color=s.label_color,
        zorder=10,
    )
