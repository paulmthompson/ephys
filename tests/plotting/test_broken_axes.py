"""Tests for :mod:`plotting.broken_axes`."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ephys.plotting.broken_axes import add_vertical_broken_axis_style


def test_add_vertical_broken_axis_style_hides_spines_and_draws() -> None:
    """Spines are hidden and diagonal markers are added without error."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(4, 2))
    add_vertical_broken_axis_style(ax_l, ax_r, vertical_positions=[0.25, 0.75])
    assert not ax_l.spines["right"].get_visible()
    assert not ax_r.spines["left"].get_visible()
    assert len(ax_l.lines) == 2
    assert len(ax_r.lines) == 2
    plt.close(fig)


def test_add_vertical_broken_axis_style_uses_marker_linewidth() -> None:
    """Explicit and inherited linewidths are forwarded to marker edges."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(4, 2))
    ax_l.spines["right"].set_linewidth(0.5)
    add_vertical_broken_axis_style(
        ax_l,
        ax_r,
        vertical_positions=[0.5],
        marker_linewidth=0.5,
    )
    assert ax_l.lines[0].get_mew() == 0.5
    assert ax_r.lines[0].get_mew() == 0.5
    plt.close(fig)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(4, 2))
    ax_l.spines["right"].set_linewidth(0.35)
    ax_l.spines["right"].set_visible(True)
    add_vertical_broken_axis_style(ax_l, ax_r, vertical_positions=[0.5])
    assert ax_l.lines[0].get_mew() == 0.35
    plt.close(fig)
