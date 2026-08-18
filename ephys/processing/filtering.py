"""Bandpass filtering helpers for multichannel electrophysiology."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.signal import bessel, butter, sosfiltfilt

BandpassFilterType = Literal["butterworth", "bessel"]

DEFAULT_INTAN_FS_HZ = 30_000.0
DEFAULT_INTAN_LOWCUT_HZ = 300.0
DEFAULT_INTAN_HIGHCUT_HZ = 5000.0
DEFAULT_INTAN_BANDPASS_ORDER = 3

__all__ = [
    "BandpassFilterType",
    "DEFAULT_INTAN_BANDPASS_ORDER",
    "DEFAULT_INTAN_FS_HZ",
    "DEFAULT_INTAN_HIGHCUT_HZ",
    "DEFAULT_INTAN_LOWCUT_HZ",
    "design_intan_sos_bandpass",
    "sos_bandpass_filter",
]


def design_intan_sos_bandpass(
    lowcut_hz: float = DEFAULT_INTAN_LOWCUT_HZ,
    highcut_hz: float = DEFAULT_INTAN_HIGHCUT_HZ,
    sampling_rate_hz: float = DEFAULT_INTAN_FS_HZ,
    order: int = DEFAULT_INTAN_BANDPASS_ORDER,
    *,
    filter_type: BandpassFilterType = "butterworth",
) -> np.ndarray:
    """Return SOS coefficients for the standard Intan spike-band pass.

    Parameters
    ----------
    lowcut_hz, highcut_hz
        Bandpass corners in hertz (defaults match Intan preprocessing scripts).
    sampling_rate_hz
        Sampling rate in hertz.
    order
        Filter order.
    filter_type
        ``"butterworth"`` (default) or ``"bessel"``.

    Returns
    -------
    numpy.ndarray
        Second-order-sections array for :func:`scipy.signal.sosfiltfilt`.
    """
    nyq = 0.5 * float(sampling_rate_hz)
    normalized_cutoffs = [float(lowcut_hz) / nyq, float(highcut_hz) / nyq]
    if filter_type == "butterworth":
        design = butter
    elif filter_type == "bessel":
        design = bessel
    else:
        raise ValueError(f"Unsupported filter_type: {filter_type!r}")

    return design(
        int(order),
        normalized_cutoffs,
        btype="band",
        output="sos",
    )


def sos_bandpass_filter(
    data: np.ndarray,
    sos: np.ndarray,
    *,
    axis: int = -1,
) -> np.ndarray:
    """Apply zero-phase SOS bandpass filtering along ``axis``.

    Parameters
    ----------
    data
        Input array (modified in place when the result aliases ``data``).
    sos
        SOS coefficients from :func:`design_intan_sos_bandpass`.
    axis
        Axis along which to filter.

    Returns
    -------
    numpy.ndarray
        Filtered data (same array as ``data`` when filtering is in place).
    """
    return sosfiltfilt(sos, data, axis=axis)
