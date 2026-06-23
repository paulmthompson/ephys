"""Temporal basis functions for GLM and point-process modeling."""

from __future__ import annotations

import numpy as np
import pydantic
from typing import Literal, Union


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


class BasisOptions(pydantic.BaseModel):
    """Base options for temporal basis functions."""

    n_lags: int = pydantic.Field(..., gt=0, description="Number of lag bins.")
    n_basis: int = pydantic.Field(..., gt=0, description="Number of basis columns.")


class RaisedCosineBasisOptions(BasisOptions):
    """Options for a standard raised-cosine basis."""

    type: Literal["raised_cosine"] = "raised_cosine"


class LogRaisedCosineBasisOptions(BasisOptions):
    """Options for a log-spaced raised-cosine basis."""

    type: Literal["log_raised_cosine"] = "log_raised_cosine"


AnyBasisOptions = Union[RaisedCosineBasisOptions, LogRaisedCosineBasisOptions]


def build_basis(options: AnyBasisOptions) -> np.ndarray:
    """Generate a basis matrix from a Pydantic options structure.

    Parameters
    ----------
    options
        Configuration specifying the basis type and shape parameters.

    Returns
    -------
    np.ndarray
        The generated basis matrix.
    """
    if isinstance(options, RaisedCosineBasisOptions):
        return raised_cosine_basis(options.n_lags, options.n_basis)
    if isinstance(options, LogRaisedCosineBasisOptions):
        return log_raised_cosine_basis(options.n_lags, options.n_basis)

    msg = f"Unknown basis options type: {type(options)}"
    raise TypeError(msg)


def causal_basis_columns(
    signal: np.ndarray,
    basis: np.ndarray,
) -> list[np.ndarray]:
    """Apply causal temporal basis convolutions to a 1D signal.

    Parameters
    ----------
    signal : np.ndarray
        One-dimensional predictor trace (e.g. continuous kinematics, or
        impulse train).
    basis : np.ndarray
        ``(n_lags, n_basis)`` temporal basis matrix, where the first row
        represents zero lag.

    Returns
    -------
    list[np.ndarray]
        List of ``n_basis`` arrays, each the same length as ``signal``,
        containing the causal convolution of the signal with each column of the
        basis.
    """
    return [
        np.convolve(signal, basis[:, idx], mode="full")[: len(signal)]
        for idx in range(basis.shape[1])
    ]


def history_basis_columns(
    spikes: np.ndarray,
    basis: np.ndarray,
) -> list[np.ndarray]:
    """Compute causal spike history predictor columns.

    Lag the spike train by one bin to avoid using the current bin's spike to
    predict itself, and then apply causal temporal basis convolutions.

    Parameters
    ----------
    spikes : np.ndarray
        One-dimensional binary array of spike occurrences.
    basis : np.ndarray
        ``(n_lags, n_basis)`` temporal basis matrix. The first row (zero
        lag into the basis) is applied to the one-bin delayed spike train,
        so it measures the effect of the immediately preceding time bin.

    Returns
    -------
    list[np.ndarray]
        List of ``n_basis`` arrays representing the convoluted spike history,
        each the same length as the input ``spikes`` array.
    """
    if len(spikes) == 0:
        return [np.zeros(0, dtype=float) for _ in range(basis.shape[1])]
    lagged = np.concatenate(([0.0], spikes[:-1]))
    return causal_basis_columns(lagged, basis)
