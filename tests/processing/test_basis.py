"""Tests for temporal basis functions."""

import numpy as np
import pytest

from ephys.processing.basis import (
    log_raised_cosine_basis,
    raised_cosine_basis,
    RaisedCosineBasisOptions,
    LogRaisedCosineBasisOptions,
    build_basis,
)
import pydantic


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


def test_build_basis_factory() -> None:
    """The build_basis factory correctly routes to mathematical functions."""
    rc_opts = RaisedCosineBasisOptions(n_lags=10, n_basis=3)
    basis_rc = build_basis(rc_opts)
    assert basis_rc.shape == (10, 3)

    log_opts = LogRaisedCosineBasisOptions(n_lags=20, n_basis=5)
    basis_log = build_basis(log_opts)
    assert basis_log.shape == (20, 5)


def test_basis_options_validation() -> None:
    """Pydantic properly validates negative n_lags or n_basis."""
    with pytest.raises(pydantic.ValidationError):
        RaisedCosineBasisOptions(n_lags=0, n_basis=5)

    with pytest.raises(pydantic.ValidationError):
        RaisedCosineBasisOptions(n_lags=10, n_basis=-1)
