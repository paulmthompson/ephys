"""Variable scaling and normalization utilities."""

from __future__ import annotations

import numpy as np


def zscore(values: np.ndarray) -> np.ndarray:
    """Compute the zero-mean unit-variance scaled version of an array.

    Parameters
    ----------
    values : np.ndarray
        Input array to be scaled.

    Returns
    -------
    np.ndarray
        Scaled array of the same shape as `values`.

    Notes
    -----
    - Non-finite values in the input are ignored when computing the mean and
      standard deviation.
    - If there are no finite values in the array, a ValueError is raised.
    - If the standard deviation of finite values is zero, all finite values
      are set to 0.0.
    - Non-finite values in the input are imputed to 0.0 in the output.
    """
    finite = np.isfinite(values)
    if not np.any(finite):
        msg = "cannot z-score a vector with no finite values"
        raise ValueError(msg)
    out = values.astype(float).copy()
    mean = float(np.mean(out[finite]))
    sd = float(np.std(out[finite], ddof=0))
    if np.isclose(sd, 0.0) or not np.isfinite(sd):
        out[finite] = 0.0
    else:
        out[finite] = (out[finite] - mean) / sd
    out[~finite] = 0.0
    return out
