"""Min/max envelope downsampling for long 1D analog traces.

When plotting voltage or motion traces at full sample rate, Matplotlib must
iterate over millions of points; that is slow, inflates file size, and can
visually **undersample** oscillations (sharp peaks disappear between samples).
Bin-wise **min** and **max** values form a conservative envelope: for each
time bin you keep the extrema actually reached by the waveform, so decimated
curves still show spikes and ripple when you draw the upper and lower bounds
(or a filled band between them) at plotting resolution.

Typical workflow for figure code: choose ``bin_size`` so that the number of
drawn segments matches the effective horizontal pixel budget (or a small
multiple), compute ``(indices, mins, maxs)``, map bin starts to time with the
sampling rate, then plot two polylines or a polygon. This is decimation **for
visualization**; downstream statistics should still use the original samples
when fidelity matters.
"""

from __future__ import annotations

import numpy as np


def get_min_max_envelope(
    y: np.ndarray,
    bin_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsample ``y`` by keeping min and max within each bin.

    Parameters
    ----------
    y
        1D voltage (or other analog) samples (any numeric dtype).
    bin_size
        Number of consecutive samples per bin; must be >= 1.

    Returns
    -------
    indices
        Start index of each bin (length matches ``mins`` / ``maxs``).
    mins, maxs
        Per-bin minimum and maximum of ``y``.
    """
    if bin_size < 1:
        raise ValueError("bin_size must be >= 1")
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")

    n_bins = len(y) // bin_size
    if n_bins == 0:
        idx = np.arange(0, len(y), bin_size, dtype=np.intp)[:0]
        empty = np.array([], dtype=y.dtype)
        return idx, empty, empty

    y_reshaped = y[: n_bins * bin_size].reshape(n_bins, bin_size)
    mins = np.min(y_reshaped, axis=1)
    maxs = np.max(y_reshaped, axis=1)
    indices = np.arange(0, len(y), bin_size, dtype=np.intp)[:n_bins]
    return indices, mins, maxs
