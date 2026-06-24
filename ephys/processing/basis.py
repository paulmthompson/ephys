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
    n_leads: int = pydantic.Field(
        0,
        ge=0,
        description=(
            "Number of lead bins before the event-aligned center (τ=0). "
            "When zero, the basis is causal with the first row at zero lag."
        ),
    )
    n_lead_basis: int = pydantic.Field(
        0,
        ge=0,
        description=(
            "Number of basis columns on the pre-event lead axis. Required "
            "when ``n_leads > 0``. Lag-axis columns use ``n_basis``."
        ),
    )
    lead_s: float = pydantic.Field(
        0.0,
        ge=0.0,
        description=(
            "Pre-event lead window duration in seconds. Required when "
            "``n_leads > 0`` so lead knots are spaced evenly in time."
        ),
    )


class RaisedCosineBasisOptions(BasisOptions):
    """Options for a standard raised-cosine basis."""

    type: Literal["raised_cosine"] = "raised_cosine"


class LogRaisedCosineBasisOptions(BasisOptions):
    """Options for a log-spaced raised-cosine basis."""

    type: Literal["log_raised_cosine"] = "log_raised_cosine"


AnyBasisOptions = Union[RaisedCosineBasisOptions, LogRaisedCosineBasisOptions]


def _normalize_basis_column(values: np.ndarray) -> np.ndarray:
    """Normalize a basis column to unit sum when possible."""
    denom = values.sum()
    return values / denom if denom > 0.0 else values


def _raised_cosine_on_times(
    row_times_s: np.ndarray,
    center_s: float,
    width_s: float,
) -> np.ndarray:
    """Return one raised-cosine bump sampled on arbitrary time coordinates."""
    if width_s <= 0.0:
        msg = "width_s must be positive"
        raise ValueError(msg)
    phase = (row_times_s - center_s) / width_s
    values = 0.5 * (1.0 + np.cos(np.clip(phase, -1.0, 1.0) * np.pi))
    values[np.abs(phase) > 1.0] = 0.0
    return _normalize_basis_column(values)


def _signed_row_times_s(
    n_leads: int,
    n_lags: int,
    *,
    lead_s: float,
) -> np.ndarray:
    """Return event-aligned row times in seconds (τ = 0 at onset)."""
    if n_leads > 0:
        if lead_s <= 0.0:
            msg = "lead_s must be positive when n_leads > 0"
            raise ValueError(msg)
        dt_s = lead_s / n_leads
    else:
        if n_lags <= 0:
            msg = "n_lags must be positive"
            raise ValueError(msg)
        # Causal-only bases use unit lag steps; physical scale is unused.
        dt_s = 1.0
    n_rows = n_leads + n_lags
    return (np.arange(n_rows, dtype=float) - n_leads) * dt_s


def _linear_lead_knot_params(
    n_lead_basis: int,
    lead_s: float,
) -> tuple[np.ndarray, float]:
    """Return evenly spaced lead knot centers and shared width in seconds."""
    if n_lead_basis <= 0:
        msg = "n_lead_basis must be positive"
        raise ValueError(msg)
    if lead_s <= 0.0:
        msg = "lead_s must be positive"
        raise ValueError(msg)
    centers_s = -lead_s * (np.arange(n_lead_basis, 0, -1, dtype=float) / n_lead_basis)
    width_s = lead_s / n_lead_basis
    return centers_s, width_s


def _linear_lag_knot_params(
    n_lags: int,
    n_lag_basis: int,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return linear lag knot centers and widths in seconds."""
    if n_lags <= 0 or n_lag_basis <= 0:
        msg = "n_lags and n_lag_basis must be positive"
        raise ValueError(msg)
    lag_times_s = np.arange(n_lags, dtype=float) * dt_s
    centers_s = np.linspace(float(lag_times_s[0]), float(lag_times_s[-1]), n_lag_basis)
    if n_lag_basis == 1:
        width_s = np.full(1, lag_times_s[-1] - lag_times_s[0] + dt_s)
    else:
        width_s = np.full(n_lag_basis, centers_s[1] - centers_s[0])
    return centers_s, width_s


def _log_lag_knot_params(
    n_lags: int,
    n_lag_basis: int,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return log-spaced lag knot centers and widths in seconds."""
    if n_lags <= 0 or n_lag_basis <= 0:
        msg = "n_lags and n_lag_basis must be positive"
        raise ValueError(msg)
    x = np.log1p(np.arange(n_lags, dtype=float))
    centers_x = np.linspace(float(x[0]), float(x[-1]), n_lag_basis)
    if n_lag_basis == 1:
        width_x = 1.0
    else:
        width_x = centers_x[1] - centers_x[0]
        if width_x == 0.0:
            centers_s = np.zeros(n_lag_basis, dtype=float)
            width_s = np.full(n_lag_basis, dt_s)
            return centers_s, width_s
    centers_s = np.expm1(centers_x) * dt_s
    lower_s = np.expm1(centers_x - width_x) * dt_s
    upper_s = np.expm1(centers_x + width_x) * dt_s
    width_s = 0.5 * (upper_s - lower_s)
    # Signed event bases need pre-onset support from post-onset knots.
    width_s = np.maximum(width_s, 1.5 * dt_s)
    return centers_s, width_s


def lead_time_raised_cosine_basis(
    n_leads: int,
    n_lead_basis: int,
    lead_s: float,
) -> np.ndarray:
    """Return raised-cosine bumps evenly spaced in time before τ = 0.

    Parameters
    ----------
    n_leads
        Number of lead bins before the event-aligned center.
    n_lead_basis
        Number of basis columns on the lead axis.
    lead_s
        Lead window duration in seconds. Row ``r`` in ``0 .. n_leads - 1``
        represents time ``(r - n_leads) * lead_s / n_leads``.

    Returns
    -------
    np.ndarray
        ``(n_leads, n_lead_basis)`` basis matrix.

    Raises
    ------
    ValueError
        If any dimension is not positive.
    """
    if n_leads <= 0:
        msg = "n_leads must be positive"
        raise ValueError(msg)
    if n_lead_basis <= 0:
        msg = "n_lead_basis must be positive"
        raise ValueError(msg)
    if lead_s <= 0.0:
        msg = "lead_s must be positive"
        raise ValueError(msg)

    dt_s = lead_s / n_leads
    row_times_s = (np.arange(n_leads, dtype=float) - n_leads) * dt_s
    centers_s = -lead_s * (np.arange(n_lead_basis, 0, -1, dtype=float) / n_lead_basis)
    width_s = lead_s / n_lead_basis
    basis = np.zeros((n_leads, n_lead_basis), dtype=float)
    for idx, center_s in enumerate(centers_s):
        phase = (row_times_s - center_s) / width_s
        values = 0.5 * (1.0 + np.cos(np.clip(phase, -1.0, 1.0) * np.pi))
        values[np.abs(phase) > 1.0] = 0.0
        denom = values.sum()
        basis[:, idx] = values / denom if denom > 0.0 else values
    return basis


def signed_event_basis(
    n_leads: int,
    n_lags: int,
    n_lead_basis: int,
    n_lag_basis: int,
    *,
    lag_kind: Literal["raised_cosine", "log_raised_cosine"],
    lead_s: float = 0.0,
) -> np.ndarray:
    """Return a continuous signed event basis with overlapping knots.

    Each knot is a raised cosine evaluated on the full axis
    ``[-lead_s, lag_max]``. Lead knots are evenly spaced in time on
    ``τ < 0``; lag knots use linear or log spacing on ``τ >= 0``. Every
    time row can receive mass from both pre- and post-onset knots.

    Parameters
    ----------
    n_leads
        Lead bins before the event-aligned center (τ < 0).
    n_lags
        Lag bins at and after the center (τ = 0 .. n_lags - 1).
    n_lead_basis
        Basis columns with centers on the lead axis. Ignored when
        ``n_leads == 0``.
    n_lag_basis
        Basis columns with centers on the lag axis.
    lag_kind
        Knot placement for the post-onset half.
    lead_s
        Lead window duration in seconds. Required when ``n_leads > 0``.

    Returns
    -------
    np.ndarray
        ``(n_leads + n_lags, n_lead_basis + n_lag_basis)`` when
        ``n_leads > 0``, otherwise ``(n_lags, n_lag_basis)``.

    Raises
    ------
    ValueError
        If dimensions are invalid or lead basis is missing when required.
    """
    if n_leads < 0:
        msg = "n_leads must be non-negative"
        raise ValueError(msg)
    if n_lag_basis <= 0:
        msg = "n_lag_basis must be positive"
        raise ValueError(msg)
    if n_lags <= 0:
        msg = "n_lags must be positive"
        raise ValueError(msg)
    if n_leads == 0:
        if lag_kind == "raised_cosine":
            return raised_cosine_basis(n_lags, n_lag_basis)
        return log_raised_cosine_basis(n_lags, n_lag_basis)
    if n_lead_basis <= 0:
        msg = "n_lead_basis must be positive when n_leads > 0"
        raise ValueError(msg)
    if lead_s <= 0.0:
        msg = "lead_s must be positive when n_leads > 0"
        raise ValueError(msg)

    row_times_s = _signed_row_times_s(n_leads, n_lags, lead_s=lead_s)
    dt_s = lead_s / n_leads
    n_cols = n_lead_basis + n_lag_basis
    basis = np.zeros((n_leads + n_lags, n_cols), dtype=float)

    lead_centers_s, lead_width_s = _linear_lead_knot_params(n_lead_basis, lead_s)
    for idx, center_s in enumerate(lead_centers_s):
        basis[:, idx] = _raised_cosine_on_times(row_times_s, center_s, lead_width_s)

    if lag_kind == "raised_cosine":
        lag_centers_s, lag_widths_s = _linear_lag_knot_params(n_lags, n_lag_basis, dt_s)
    else:
        lag_centers_s, lag_widths_s = _log_lag_knot_params(n_lags, n_lag_basis, dt_s)
    for offset, (center_s, width_s) in enumerate(zip(lag_centers_s, lag_widths_s, strict=True)):
        col = n_lead_basis + offset
        basis[:, col] = _raised_cosine_on_times(row_times_s, center_s, width_s)

    return basis


def split_signed_event_basis(
    n_leads: int,
    n_lags: int,
    n_lead_basis: int,
    n_lag_basis: int,
    *,
    lag_kind: Literal["raised_cosine", "log_raised_cosine"],
    lead_s: float = 0.0,
) -> np.ndarray:
    """Backward-compatible alias for :func:`signed_event_basis`."""
    return signed_event_basis(
        n_leads,
        n_lags,
        n_lead_basis,
        n_lag_basis,
        lag_kind=lag_kind,
        lead_s=lead_s,
    )


def signed_raised_cosine_basis(
    n_leads: int,
    n_lags: int,
    n_lag_basis: int,
    *,
    n_lead_basis: int = 0,
    lead_s: float = 0.0,
) -> np.ndarray:
    """Return raised-cosine bases on a signed lag axis.

    Parameters
    ----------
    n_leads
        Lead bins before the event-aligned center (τ < 0).
    n_lags
        Lag bins at and after the center (τ = 0 .. n_lags - 1).
    n_lag_basis
        Number of basis columns on the lag axis.
    n_lead_basis
        Number of basis columns on the lead axis when ``n_leads > 0``.

    Returns
    -------
    np.ndarray
        Continuous signed basis; see :func:`signed_event_basis`.

    Raises
    ------
    ValueError
        If dimensions are invalid.
    """
    return split_signed_event_basis(
        n_leads,
        n_lags,
        n_lead_basis,
        n_lag_basis,
        lag_kind="raised_cosine",
        lead_s=lead_s,
    )


def signed_log_raised_cosine_basis(
    n_leads: int,
    n_lags: int,
    n_lag_basis: int,
    *,
    n_lead_basis: int = 0,
    lead_s: float = 0.0,
) -> np.ndarray:
    """Return log-spaced lag knots on a continuous signed event axis.

    Parameters
    ----------
    n_leads
        Lead bins before the event-aligned center (τ < 0).
    n_lags
        Lag bins at and after the center (τ = 0 .. n_lags - 1).
    n_lag_basis
        Number of basis columns on the lag axis.
    n_lead_basis
        Number of linear raised-cosine columns on the lead axis when
        ``n_leads > 0``.

    Returns
    -------
    np.ndarray
        Continuous signed basis; see :func:`signed_event_basis`.

    Raises
    ------
    ValueError
        If dimensions are invalid.
    """
    return split_signed_event_basis(
        n_leads,
        n_lags,
        n_lead_basis,
        n_lag_basis,
        lag_kind="log_raised_cosine",
        lead_s=lead_s,
    )


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
        return signed_raised_cosine_basis(
            options.n_leads,
            options.n_lags,
            options.n_basis,
            n_lead_basis=options.n_lead_basis,
            lead_s=options.lead_s,
        )
    if isinstance(options, LogRaisedCosineBasisOptions):
        return signed_log_raised_cosine_basis(
            options.n_leads,
            options.n_lags,
            options.n_basis,
            n_lead_basis=options.n_lead_basis,
            lead_s=options.lead_s,
        )

    msg = f"Unknown basis options type: {type(options)}"
    raise TypeError(msg)


def acausal_basis_columns(
    signal: np.ndarray,
    basis: np.ndarray,
    *,
    n_leads: int = 0,
) -> list[np.ndarray]:
    """Apply two-sided temporal basis filtering to a 1D signal.

    Parameters
    ----------
    signal
        One-dimensional predictor trace (for example continuous kinematics or
        an impulse train).
    basis
        Temporal basis matrix. When ``n_leads == 0``, shape is
        ``(n_lags, n_basis)`` with the first row at zero lag (causal). When
        ``n_leads > 0``, shape is ``(n_leads + n_lags, n_basis)`` with row
        ``n_leads`` at the event-aligned center (τ = 0).
    n_leads
        Number of lead bins before τ = 0. Must match the basis construction.

    Returns
    -------
    list[np.ndarray]
        List of ``n_basis`` arrays, each the same length as ``signal``.

    Raises
    ------
    ValueError
        If ``n_leads`` is negative or inconsistent with ``basis`` shape.
    """
    if n_leads < 0:
        msg = "n_leads must be non-negative"
        raise ValueError(msg)
    if n_leads == 0:
        return [
            np.convolve(signal, basis[:, idx], mode="full")[: len(signal)]
            for idx in range(basis.shape[1])
        ]

    n_rows, n_basis = basis.shape
    n_lags = n_rows - n_leads
    if n_lags <= 0:
        msg = "basis must have at least one lag row when n_leads > 0"
        raise ValueError(msg)

    n = len(signal)
    outputs = [np.zeros(n, dtype=float) for _ in range(n_basis)]
    for col_idx in range(n_basis):
        col = outputs[col_idx]
        for tau in range(-n_leads, n_lags):
            row = n_leads + tau
            weight = basis[row, col_idx]
            if weight == 0.0:
                continue
            if tau >= 0:
                col[tau:] += weight * signal[: n - tau]
            else:
                lead = -tau
                col[: n - lead] += weight * signal[lead:]
    return outputs


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
    return acausal_basis_columns(signal, basis, n_leads=0)


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
