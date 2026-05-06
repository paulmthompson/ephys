"""ZCA spatial whitening for multichannel voltage arrays.

Applies zero-phase component analysis (ZCA) whitening along the channel
axis for arrays shaped ``(n_channels, n_samples)``. Used to decorrelate
channels while staying close to the original waveform geometry.
"""

import numpy as np


def apply_zca_whitening(
    voltage_matrix, epsilon=10.0, rescale_amplitude=True, robust_cov=True
):
    """Whiten a voltage matrix with ZCA using robust per-channel scaling.

    Each channel is centered with its temporal median. Per-channel scale
    uses the median absolute deviation (MAD) times ``1.4826`` so that it
    matches the standard deviation of Gaussian noise. The covariance of
    centered data is eigendecomposed; ZCA applies ``U (S + epsilon)^(-1/2)
    U^T`` in that basis. When ``robust_cov`` is enabled, covariance is fit
    only on time samples where every channel is within ``4`` robust standard
    deviations of zero, reducing contamination from brief large artifacts.

    Parameters
    ----------
    voltage_matrix : ndarray, shape (n_channels, n_samples)
        Voltage data to whiten. **Modified in place**; the returned array is
        the same object.
    epsilon : float, optional
        Small positive regularizer added to eigenvalues before taking the
        inverse square root. Larger values stabilize inversion when the
        covariance is nearly singular. Default is ``10.0``.
    rescale_amplitude : bool, optional
        If ``True`` (default), multiply the whitened data by the mean
        per-channel robust standard deviation so overall amplitude stays
        comparable to the input scale.
    robust_cov : bool, optional
        If ``True`` (default), estimate the covariance matrix using only
        samples that pass an all-channel artifact gate (see Notes). If
        ``False``, use ``numpy.cov`` on the full centered matrix.

    Returns
    -------
    ndarray
        The whitened data; same array as ``voltage_matrix``.

    Notes
    -----
    This routine mutates ``voltage_matrix`` for memory efficiency. Copy the
    input first if the original array must be preserved.

    If validation fails **after** median centering (for example too few clean
    samples when ``robust_cov`` is ``True``), the array may already be
    median-centered; discard it or pass a copy on retry.

    The artifact gate when ``robust_cov`` is ``True`` keeps sample ``t`` iff
    ``|x[c, t]| < 4 * MAD_c`` for every channel ``c``, where ``MAD_c`` is the
    robust standard deviation of channel ``c`` after median centering.

    Raises
    ------
    TypeError
        If ``voltage_matrix`` is not a :class:`numpy.ndarray` with a
        floating-point dtype, or if ``rescale_amplitude`` / ``robust_cov`` are
        not boolean-like.
    ValueError
        If the array shape is invalid, ``epsilon`` is not positive, there are
        too few samples for covariance estimation, or ``robust_cov`` leaves
        fewer than two clean samples. If this occurs after median centering,
        ``voltage_matrix`` has already been centered in place (see Notes).
    """
    if not isinstance(voltage_matrix, np.ndarray):
        msg = (
            "voltage_matrix must be a numpy.ndarray, got "
            f"{type(voltage_matrix).__name__!r}"
        )
        raise TypeError(msg)
    if voltage_matrix.ndim != 2:
        msg = (
            "voltage_matrix must have shape (n_channels, n_samples); "
            f"got ndim={voltage_matrix.ndim}"
        )
        raise ValueError(msg)
    n_channels, n_samples = voltage_matrix.shape
    if n_channels < 1 or n_samples < 1:
        msg = (
            "voltage_matrix must have at least one channel and one sample; "
            f"got shape {voltage_matrix.shape!r}"
        )
        raise ValueError(msg)
    if not np.issubdtype(voltage_matrix.dtype, np.floating):
        msg = (
            "voltage_matrix must have a floating-point dtype; "
            f"got {voltage_matrix.dtype!r}"
        )
        raise TypeError(msg)
    if epsilon <= 0:
        msg = f"epsilon must be positive, got {epsilon!r}"
        raise ValueError(msg)
    if not isinstance(rescale_amplitude, (bool, np.bool_)):
        msg = (
            "rescale_amplitude must be bool, got "
            f"{type(rescale_amplitude).__name__!r}"
        )
        raise TypeError(msg)
    if not isinstance(robust_cov, (bool, np.bool_)):
        msg = f"robust_cov must be bool, got {type(robust_cov).__name__!r}"
        raise TypeError(msg)

    if not robust_cov and n_samples < 2:
        msg = (
            "Covariance estimation needs at least 2 samples when robust_cov "
            f"is False; got n_samples={n_samples}"
        )
        raise ValueError(msg)

    # 1. To avoid redundant extreme sorting, compute median once and use it
    # everywhere.
    # Note: For bandpassed ephys data, median is effectively 0, but this
    # ensures perfect centering.
    median_vars = np.median(voltage_matrix, axis=1, keepdims=True)
    voltage_matrix -= median_vars

    # 2. Compute a unified, cached Robust Standard Deviation (MAD)
    # Because centered_volt is cleanly centered, we completely skip the nested
    # inner np.median calculation
    robust_std = (
        np.median(np.abs(voltage_matrix), axis=1, keepdims=True) * 1.4826
    )

    # 3. Compute covariance matrix across channels
    if robust_cov:
        # Find time-samples where ALL channels are below 4x std to drop
        # movement artifacts
        is_clean_sample = np.all(
            np.abs(voltage_matrix) < (4 * robust_std), axis=0
        )
        n_clean = int(is_clean_sample.sum())
        if n_clean < 2:
            msg = (
                "robust_cov requires at least 2 samples passing the clean "
                f"artifact gate; got {n_clean}. "
                "Try robust_cov=False or adjust inputs. "
                "voltage_matrix has been median-centered in place."
            )
            raise ValueError(msg)
        clean_centered_volt = voltage_matrix[:, is_clean_sample]
        cov = np.cov(clean_centered_volt)
    else:
        cov = np.cov(voltage_matrix)

    # 4. Eigen decomposition
    U, S, _ = np.linalg.svd(cov)

    # 5. Compute ZCA matrix with epsilon regularization
    zca_matrix = U @ np.diag(1.0 / np.sqrt(S + epsilon)) @ U.T

    # 6. Apply ZCA
    voltage_matrix[:] = zca_matrix @ voltage_matrix

    # 7. Rescale back to approximate original amplitude using our unified
    # robust MAD
    if rescale_amplitude:
        fixed_scalar = np.mean(robust_std)
        voltage_matrix *= fixed_scalar

    return voltage_matrix
