"""Temporal basis functions for GLM and point-process modeling."""

from __future__ import annotations

import numpy as np


def raised_cosine_basis(n_lags: int, n_basis: int) -> np.ndarray:
    """Return normalized raised-cosine basis functions over lag bins.

    Parameters
    ----------
    n_lags
        Number of lag bins.
    n_basis
        Number of basis columns.

    Returns
    -------
    np.ndarray
        ``(n_lags, n_basis)`` basis matrix with columns normalized to unit sum
        when possible.

    Raises
    ------
    ValueError
        If ``n_lags`` or ``n_basis`` is not positive.
    """
    if n_lags <= 0:
        msg = "n_lags must be positive"
        raise ValueError(msg)
    if n_basis <= 0:
        msg = "n_basis must be positive"
        raise ValueError(msg)
    x = np.linspace(0.0, 1.0, n_lags)
    centers = np.linspace(0.0, 1.0, n_basis)
    width = 1.0 if n_basis == 1 else centers[1] - centers[0]
    basis = np.zeros((n_lags, n_basis), dtype=float)
    for idx, center in enumerate(centers):
        phase = (x - center) / width
        values = 0.5 * (1.0 + np.cos(np.clip(phase, -1.0, 1.0) * np.pi))
        values[np.abs(phase) > 1.0] = 0.0
        denom = values.sum()
        basis[:, idx] = values / denom if denom > 0.0 else values
    return basis


def log_raised_cosine_basis(n_lags: int, n_basis: int) -> np.ndarray:
    """Return normalized raised-cosine bases with log-spaced lag support.

    Parameters
    ----------
    n_lags
        Number of lag bins. The first row represents zero lag.
    n_basis
        Number of basis columns.

    Returns
    -------
    np.ndarray
        ``(n_lags, n_basis)`` basis matrix. Centers are evenly spaced after
        warping lag bins by ``log1p`` so early lags receive finer resolution
        while the first basis can still include zero lag.

    Raises
    ------
    ValueError
        If ``n_lags`` or ``n_basis`` is not positive.
    """
    if n_lags <= 0:
        msg = "n_lags must be positive"
        raise ValueError(msg)
    if n_basis <= 0:
        msg = "n_basis must be positive"
        raise ValueError(msg)
    x = np.log1p(np.arange(n_lags, dtype=float))
    centers = np.linspace(float(x[0]), float(x[-1]), n_basis)
    if n_basis == 1:
        width = 1.0
    else:
        width = centers[1] - centers[0]
        if width == 0.0:
            # Degenerate axis (e.g., n_lags == 1): place all mass at lag 0.
            basis = np.zeros((n_lags, n_basis), dtype=float)
            basis[0, :] = 1.0
            return basis
    basis = np.zeros((n_lags, n_basis), dtype=float)
    for idx, center in enumerate(centers):
        phase = (x - center) / width
        values = 0.5 * (1.0 + np.cos(np.clip(phase, -1.0, 1.0) * np.pi))
        values[np.abs(phase) > 1.0] = 0.0
        denom = values.sum()
        basis[:, idx] = values / denom if denom > 0.0 else values
    return basis
