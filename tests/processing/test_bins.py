"""Tests for UniformBins grid construction, geometry, and histogramming."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ephys.processing import UniformBins


def test_from_width_avoids_float_drift_and_includes_boundary_spike() -> None:
    """from_width avoids 1e-18 sliver bins and keeps exact stop boundary."""
    # np.arange(-0.01, 0.02 + 0.002, 0.002) previously drifted to
    # 0.019999999999999997 < 0.02, dropping spikes at 0.02 or adding a sliver.
    bins = UniformBins.from_width(-0.01, 0.02, 0.002)
    assert bins.n_bins == 15
    assert bins.limits == (-0.01, 0.02)
    assert bins.span == pytest.approx(0.03)
    assert bins.width == pytest.approx(0.002)
    assert bins.edges.shape == (16,)
    assert bins.edges[0] == -0.01
    assert bins.edges[-1] == 0.02
    assert np.allclose(np.diff(bins.edges), 0.002)

    counts = bins.histogram_counts([-0.01, 0.0, 0.02, float("nan"), 0.5])
    assert counts.dtype == np.int64
    assert counts.shape == (15,)
    assert int(counts[0]) == 1
    assert int(counts[-1]) == 1
    assert int(counts.sum()) == 3


def test_from_width_rejects_non_divisible_span_and_invalid_inputs() -> None:
    """from_width raises ValueError when stop - start is not k * width."""
    with pytest.raises(ValueError, match="not an integer multiple"):
        UniformBins.from_width(0.0, 0.010, 0.003)
    with pytest.raises(ValueError, match="not an integer multiple"):
        UniformBins.from_width(-0.005, 0.045, 0.003)
    with pytest.raises(ValueError, match="width must be positive"):
        UniformBins.from_width(0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="stop must exceed start"):
        UniformBins.from_width(1.0, 1.0, 0.1)


def test_centers_left_right_edges_and_unit_scaling() -> None:
    """Geometry properties and scaled() preserve alignment across units."""
    bins_s = UniformBins.from_count(-0.005, 0.005, 5)
    assert np.allclose(bins_s.left_edges, [-0.005, -0.003, -0.001, 0.001, 0.003])
    assert np.allclose(bins_s.right_edges, [-0.003, -0.001, 0.001, 0.003, 0.005])
    assert np.allclose(bins_s.centers, [-0.004, -0.002, 0.0, 0.002, 0.004])

    bins_ms = bins_s.scaled(1000.0)
    assert bins_ms.limits == pytest.approx((-5.0, 5.0))
    assert bins_ms.width == pytest.approx(2.0)
    assert np.allclose(bins_ms.centers, [-4.0, -2.0, 0.0, 2.0, 4.0])

    with pytest.raises(ValueError, match="factor must be positive"):
        bins_s.scaled(0.0)


def test_from_edges_round_trip_and_non_uniform_rejection() -> None:
    """from_edges accepts uniform arrays and rejects non-uniform/invalid ones."""
    edges = np.linspace(-2.0, 4.0, 13)
    bins = UniformBins.from_edges(edges)
    assert bins == UniformBins(start=-2.0, stop=4.0, n_bins=12)

    with pytest.raises(ValueError, match="not uniformly spaced"):
        UniformBins.from_edges([0.0, 1.0, 3.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        UniformBins.from_edges([0.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="at least 2 boundaries"):
        UniformBins.from_edges([0.0])


def test_constructor_type_and_value_guards() -> None:
    """Direct constructor rejects booleans, non-integers, and empty counts."""
    bad_bool: Any = True
    with pytest.raises(TypeError, match="start"):
        UniformBins(bad_bool, 2.0, 5)
    with pytest.raises(TypeError, match="n_bins"):
        UniformBins(0.0, 1.0, bad_bool)
    bad_float_bins: Any = 2.5
    with pytest.raises(TypeError, match="n_bins"):
        UniformBins(0.0, 1.0, bad_float_bins)
    with pytest.raises(ValueError, match="n_bins"):
        UniformBins(0.0, 1.0, 0)
