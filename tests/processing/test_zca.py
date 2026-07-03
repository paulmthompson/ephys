"""Tests for ZCA whitening fit/apply and npz serialization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ephys.processing.zca import (
    apply_zca_fit,
    apply_zca_whitening,
    fit_zca_whitening,
    load_zca_fit_npz,
    save_zca_fit_npz,
    zca_matrix_from_covariance,
)


def _synthetic_bandpassed_voltage(
    n_channels: int = 4,
    n_samples: int = 500,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shared = rng.standard_normal(n_samples)
    mixing = rng.standard_normal((n_channels, n_channels))
    data = mixing @ np.vstack([shared, rng.standard_normal((n_channels - 1, n_samples))])
    return data.astype(np.float64)


def test_fit_apply_matches_monolithic_apply() -> None:
    """fit + apply reproduces apply_zca_whitening on the same data."""
    voltage = _synthetic_bandpassed_voltage()
    expected = apply_zca_whitening(np.array(voltage, copy=True))
    fit = fit_zca_whitening(voltage)
    actual = apply_zca_fit(np.array(voltage, copy=True), fit)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_fit_does_not_mutate_input() -> None:
    """fit_zca_whitening leaves the input array unchanged."""
    voltage = _synthetic_bandpassed_voltage()
    before = voltage.copy()
    fit_zca_whitening(voltage)
    np.testing.assert_array_equal(voltage, before)


def test_zca_fit_npz_round_trip(tmp_path: Path) -> None:
    """save/load npz preserves whitening behavior."""
    voltage = _synthetic_bandpassed_voltage()
    fit = fit_zca_whitening(voltage, good_channels=[10, 12, 14, 16])
    path = tmp_path / "zca_fit.npz"
    save_zca_fit_npz(path, fit)
    loaded = load_zca_fit_npz(path)
    expected = apply_zca_fit(np.array(voltage, copy=True), fit)
    actual = apply_zca_fit(np.array(voltage, copy=True), loaded)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(loaded.good_channels, fit.good_channels)


def test_row_index_for_channel_maps_probe_indices() -> None:
    """row_index_for_channel resolves probe labels."""
    voltage = _synthetic_bandpassed_voltage(n_channels=3)
    fit = fit_zca_whitening(voltage, good_channels=[2, 5, 9])
    assert fit.row_index_for_channel(5) == 1


def test_row_index_for_dead_channel_raises() -> None:
    """Missing probe channels raise ValueError."""
    voltage = _synthetic_bandpassed_voltage(n_channels=2)
    fit = fit_zca_whitening(voltage, good_channels=[1, 3])
    with pytest.raises(ValueError, match="not in ZCA fit"):
        fit.row_index_for_channel(0)


def test_zca_matrix_from_covariance_is_symmetric() -> None:
    """Whitening matrix built from covariance is symmetric."""
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    matrix = zca_matrix_from_covariance(cov, epsilon=0.1)
    np.testing.assert_allclose(matrix, matrix.T, rtol=1e-10, atol=1e-10)
