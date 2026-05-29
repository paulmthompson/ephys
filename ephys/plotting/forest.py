"""Single-axis forest plot and generic multi-column grids for posteriors."""

from __future__ import annotations

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from pydantic import BaseModel, Field


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


def plot_forest(
    ax: plt.Axes,
    df: pd.DataFrame,
    parameter: str,
    title: str,
    ci_level: float = 0.89,
    *,
    neuron_id_axis: bool = False,
    detach_y_spine: bool = False,
    xlabel: str | None = None,
    default_xlabel: str = "Coefficient Value",
) -> None:
    """Plot posterior median and HDIs for each animal/session group.

    Sorts rows by median coefficient. Uses symmetric x-axis limits.

    Parameters
    ----------
    ax
        Target matplotlib axes.
    df
        Posterior draws with ``animal``, ``session``, and coefficient column.
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
    """
    if df.empty or parameter not in df.columns:
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

    if "animal" not in df.columns or "session" not in df.columns:
        raise ValueError("DataFrame must contain 'animal' and 'session' columns.")

    lower_q = (1.0 - ci_level) / 2.0
    upper_q = 1.0 - lower_q

    grouped = df.groupby(["animal", "session"])

    results: list[dict[str, object]] = []
    for key, group in grouped:
        animal, session = cast(tuple[Any, Any], key)
        samples = group[parameter].dropna().values
        if len(samples) == 0:
            continue

        med = np.median(samples)
        lower = np.quantile(samples, lower_q)
        upper = np.quantile(samples, upper_q)
        results.append(
            {
                "label": f"{animal}/{session}",
                "med": med,
                "lower": lower,
                "upper": upper,
            }
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

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("med", ascending=True).reset_index(drop=True)

    y_pos = np.arange(len(res_df))
    medians = res_df["med"].values
    lowers = res_df["lower"].values
    uppers = res_df["upper"].values

    ax.errorbar(
        medians,
        y_pos,
        xerr=[medians - lowers, uppers - medians],
        fmt="o",
        color="black",
        capsize=0,
        markersize=2.5,
        markerfacecolor="black",
        markeredgecolor="black",
        ecolor="gray",
    )

    ax.axvline(0, color="black", linestyle="--", alpha=0.7, linewidth=1)

    max_abs_val = max(np.max(np.abs(lowers)), np.max(np.abs(uppers)))
    limit_val = max_abs_val * 1.1
    ax.set_xlim(-limit_val, limit_val)

    n = len(res_df)
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
        labels = res_df["label"].values
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    ax.set_title(title)
    ax.set_xlabel(xlabel if xlabel is not None else default_xlabel)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_forest_grid(
    fig: Figure,
    cell: SubplotSpec,
    df: pd.DataFrame,
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
