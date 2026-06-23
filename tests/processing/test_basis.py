"""Tests for temporal basis functions."""

import numpy as np
import pytest

from ephys.processing.basis import log_raised_cosine_basis, raised_cosine_basis


def test_raised_cosine_basis_normalizes_columns() -> None:
    """Raised-cosine bases have the requested shape and finite columns."""
    basis = raised_cosine_basis(10, 3)

    assert basis.shape == (10, 3)
    assert np.all(np.isfinite(basis))
    assert basis.sum(axis=0).tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_log_raised_cosine_basis_keeps_zero_lag_and_early_resolution() -> None:
    """Log-spaced bases keep zero-lag support with denser early centers."""
    basis = log_raised_cosine_basis(200, 5)

    assert basis.shape == (200, 5)
    assert np.all(np.isfinite(basis))
    assert basis.sum(axis=0).tolist() == pytest.approx([1.0] * 5)
    assert basis[0, 0] > 0.0

    peak_lags = np.argmax(basis, axis=0)
    peak_spacing = np.diff(peak_lags)
    assert np.all(peak_spacing > 0)
    assert peak_spacing[0] < peak_spacing[-1]
