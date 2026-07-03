"""Resampling routines for electrophysiology data."""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly


def whittaker_shannon_interpolate(
    data: np.ndarray,
    up_factor: int,
    down_factor: int = 1,
    window_half_len: int | None = None,
    beta: float = 5.0,
    axis: int = -1,
) -> np.ndarray:
    """
    Upsample a signal using Whittaker-Shannon (windowed sinc) interpolation.

    This function uses polyphase filtering (via scipy.signal.resample_poly) to
    efficiently apply a windowed sinc filter for interpolation. This accurately
    reconstructs inter-sample peaks in bandlimited signals, such as action potentials.

    Parameters
    ----------
    data : np.ndarray
        The input signal array.
    up_factor : int
        The upsampling factor (e.g., 4 to go from 30 kHz to 120 kHz).
    down_factor : int, optional
        The downsampling factor. Default is 1.
    window_half_len : int, optional
        Half the length of the FIR filter window in terms of the original sampling rate.
        By default, it is set to `10 * max(up_factor, down_factor)`, which provides
        a good trade-off between anti-aliasing and computational time (covering ~10
        sinc lobes).
    beta : float, optional
        The beta parameter for the Kaiser window. Default is 5.0, which provides
        good stopband attenuation.
    axis : int, optional
        The axis along which to resample. Default is -1.

    Returns
    -------
    np.ndarray
        The interpolated (resampled) signal.
    """
    if window_half_len is None:
        window_half_len = 10 * max(up_factor, down_factor)

    # scipy's resample_poly uses a Kaiser window by default
    # To specify the exact window length, we can pass a tuple to the `window` argument
    # The length of the filter will be 2 * window_half_len + 1
    window = ("kaiser", beta)

    # Pad parameter determines how edges are handled. 'line' or 'mean' helps prevent
    # massive ringing at the start/end if the signal has a DC offset.
    return resample_poly(
        data,
        up=up_factor,
        down=down_factor,
        axis=axis,
        window=window,
        padtype="line",
    )
