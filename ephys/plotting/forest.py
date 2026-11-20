"""Single-axis forest plot and generic multi-column grids for posteriors."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from pydantic import BaseModel, Field


class ForestTableLike(Protocol):
    """Column-indexed mapping or DataFrame accepted by :func:`plot_forest`."""

    def __contains__(self, key: str, /) -> bool: ...

    def __getitem__(self, key: str, /) -> Any: ...


class ForestColumnSpec(BaseModel):
    """Configuration for a single forest plot column in a multi-column row."""

    parameter: str
    title: str
    xlabel: str = "Coefficient Value"


class ForestGridOptions(BaseModel):
    """Configuration for an N-column forest plot grid."""

    ci_level: float = Field(default=0.89, ge=0.0, le=1.0)
    wspace: float = 0.3
    columns: list[ForestColumnSpec] = Field(default_factory=list)


class ForestHighlightSpec(BaseModel):
    """Highlight one animal/session row in a forest plot."""

    model_config = {"frozen": True}

    animal: str
    session: str
    open_marker: bool = True
    color: str = "#d94801"
    error_color: str | None = None
    zorder: int = 5


FOREST_MARKER_SIZE = 2.5
FOREST_ERROR_BAR_COLOR = "gray"
FOREST_ERROR_BAR_LINEWIDTH = 1.5


def _lighter_mpl_color(color: str, mix: float = 0.5) -> str:
    """Blend a matplotlib color toward white for lighter error bars."""
    rgb = np.array(mcolors.to_rgb(color), dtype=float)
    lighter = rgb + (1.0 - rgb) * mix
    return mcolors.to_hex(np.clip(lighter, 0.0, 1.0))


def _extract_column_values(data: ForestTableLike, key: str) -> list[Any]:
    """Materialize one column from a mapping or DataFrame-like table."""
    col = data[key]
    to_numpy = getattr(col, "to_numpy", None)
    if callable(to_numpy):
        return list(np.asarray(to_numpy()))
    return list(col)


def plot_forest(
    ax: plt.Axes,
    df: ForestTableLike,
    parameter: str,
    title: str,
    ci_level: float = 0.89,
    *,
    neuron_id_axis: bool = False,
    detach_y_spine: bool = False,
    xlabel: str | None = None,
    default_xlabel: str = "Coefficient Value",
    highlight: ForestHighlightSpec | None = None,
) -> None:
    """Plot posterior median and HDIs for each animal/session group.

    Sorts rows by median coefficient. Uses symmetric x-axis limits.

    Parameters
    ----------
    ax
        Target matplotlib axes.
    df
        Posterior draws mapping or DataFrame with ``animal``, ``session``,
        and coefficient column.
    parameter
        Column name for the coefficient to summarize.
    title
        Subplot title.
    ci_level
        Central credible interval width (e.g. 0.89).
    neuron_id_axis
        If True, show a compact Neuron ID y-axis (1 at bottom, n at top).
    detach_y_spine
        If True with ``neuron_id_axis``, offset the left spine outward.
    xlabel
        X-axis label; defaults to ``default_xlabel``.
    default_xlabel
        Fallback x-axis label when ``xlabel`` is None.
    highlight
        Optional animal/session row to emphasize. By default the row is
        redrawn with a white-filled open circle marker using the same edge
        colors as peers.
    """
    if parameter not in df:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        ax.set_axis_off()
        return

    param_raw = _extract_column_values(df, parameter)
    if len(param_raw) == 0:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        ax.set_axis_off()
        return

    if "animal" not in df or "session" not in df:
        raise ValueError(
            "Table must contain 'animal' and 'session' columns."
        )

    animals_raw = _extract_column_values(df, "animal")
    sessions_raw = _extract_column_values(df, "session")
    if not (len(animals_raw) == len(sessions_raw) == len(param_raw)):
        raise ValueError(
            "Columns 'animal', 'session', and parameter must have equal length."
        )

    lower_q = (1.0 - ci_level) / 2.0
    upper_q = 1.0 - lower_q

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for animal_val, session_val, sample_val in zip(
        animals_raw, sessions_raw, param_raw, strict=True
    ):
        if sample_val is None:
            continue
        fval = float(sample_val)
        if np.isnan(fval):
            continue
        grouped[(str(animal_val), str(session_val))].append(fval)

    results: list[tuple[str, str, str, float, float, float]] = []
    for animal, session in sorted(grouped):
        samples = np.asarray(grouped[(animal, session)], dtype=np.float64)
        if samples.size == 0:
            continue
        med = float(np.median(samples))
        lower = float(np.quantile(samples, lower_q))
        upper = float(np.quantile(samples, upper_q))
        results.append(
            (animal, session, f"{animal}/{session}", med, lower, upper)
        )

    if not results:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        ax.set_axis_off()
        return

    results.sort(key=lambda row: row[3])
    n = len(results)
    y_pos = np.arange(n)
    animals_sorted = np.array([row[0] for row in results], dtype=object)
    sessions_sorted = np.array([row[1] for row in results], dtype=object)
    labels_sorted = [row[2] for row in results]
    medians = np.array([row[3] for row in results], dtype=np.float64)
    lowers = np.array([row[4] for row in results], dtype=np.float64)
    uppers = np.array([row[5] for row in results], dtype=np.float64)

    highlight_mask = np.zeros(n, dtype=bool)
    if highlight is not None:
        highlight_mask = (animals_sorted == str(highlight.animal)) & (
            sessions_sorted == str(highlight.session)
        )

    baseline_mask = ~highlight_mask
    if np.any(baseline_mask):
        ax.errorbar(
            medians[baseline_mask],
            y_pos[baseline_mask],
            xerr=[
                medians[baseline_mask] - lowers[baseline_mask],
                uppers[baseline_mask] - medians[baseline_mask],
            ],
            fmt="o",
            color="black",
            capsize=0,
            markersize=FOREST_MARKER_SIZE,
            markerfacecolor="black",
            markeredgecolor="black",
            ecolor=FOREST_ERROR_BAR_COLOR,
            elinewidth=FOREST_ERROR_BAR_LINEWIDTH,
        )

    if highlight is not None and np.any(highlight_mask):
        if highlight.open_marker:
            marker_color = "black"
            marker_facecolor = "white"
            error_color = FOREST_ERROR_BAR_COLOR
        else:
            marker_color = highlight.color
            marker_facecolor = highlight.color
            error_color = highlight.error_color or _lighter_mpl_color(
                marker_color,
            )
        ax.errorbar(
            medians[highlight_mask],
            y_pos[highlight_mask],
            xerr=[
                medians[highlight_mask] - lowers[highlight_mask],
                uppers[highlight_mask] - medians[highlight_mask],
            ],
            fmt="o",
            color=marker_color,
            capsize=0,
            markersize=FOREST_MARKER_SIZE,
            markerfacecolor=marker_facecolor,
            markeredgecolor=marker_color,
            ecolor=error_color,
            elinewidth=FOREST_ERROR_BAR_LINEWIDTH,
            zorder=highlight.zorder,
        )

    ax.axvline(0, color="black", linestyle="--", alpha=0.7, linewidth=1)

    max_abs_val = max(np.max(np.abs(lowers)), np.max(np.abs(uppers)))
    limit_val = max_abs_val * 1.1
    ax.set_xlim(-limit_val, limit_val)

    if neuron_id_axis:
        ax.set_ylabel("Neuron ID")
        if n == 1:
            ax.set_yticks([0])
            ax.set_yticklabels(["1"])
        else:
            ax.set_yticks([0, n - 1])
            ax.set_yticklabels(["1", str(n)])
        ax.spines["left"].set_visible(True)
        if detach_y_spine:
            ax.spines["left"].set_position(("outward", 6))
        ax.spines["left"].set_bounds(0.0, max(0.0, float(n - 1)))
        ax.tick_params(axis="y", length=3)
    else:
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_sorted)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    ax.set_title(title)
    ax.set_xlabel(xlabel if xlabel is not None else default_xlabel)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_forest_grid(
    fig: Figure,
    cell: SubplotSpec,
    df: ForestTableLike,
    options: ForestGridOptions,
) -> list[plt.Axes]:
    """Draw a generic N-column forest plot grid.

    Parameters
    ----------
    fig
        Parent figure.
    cell
        GridSpec cell to subdivide into columns.
    df
        Posterior draws table.
    options
        Generic N-column layout options.

    Returns
    -------
    list[matplotlib.axes.Axes]
        List of generated axes.
    """
    if not options.columns:
        ax = fig.add_subplot(cell)
        ax.set_axis_off()
        return [ax]

    n_cols = len(options.columns)
    gs_forest = cell.subgridspec(1, n_cols, wspace=options.wspace)

    axes = []
    for i, col_spec in enumerate(options.columns):
        ax = fig.add_subplot(gs_forest[0, i])

        plot_forest(
            ax,
            df,
            col_spec.parameter,
            col_spec.title,
            ci_level=options.ci_level,
            neuron_id_axis=(i == 0),
            detach_y_spine=True,
            xlabel=col_spec.xlabel,
        )

        if i > 0:
            ax.set_yticklabels([])

        axes.append(ax)

    return axes
