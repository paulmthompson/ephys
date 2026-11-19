"""ZCA spatial whitening for multichannel voltage arrays.

Applies zero-phase component analysis (ZCA) whitening along the channel
axis for arrays shaped ``(n_channels, n_samples)``. Used to decorrelate
channels while staying close to the original waveform geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d

from ephys.processing.filtering import (
    DEFAULT_INTAN_BANDPASS_ORDER,
    DEFAULT_INTAN_FS_HZ,
    DEFAULT_INTAN_HIGHCUT_HZ,
    DEFAULT_INTAN_LOWCUT_HZ,
)

ZCA_FIT_NPZ_VERSION = 2
_MAD_TO_STD = 1.4826
_DEFAULT_ARTIFACT_N_SIGMA = 4.0
_DEFAULT_ARTIFACT_PAD_MS = 0.5
_DEFAULT_CHUNK_SAMPLES = 262_144

__all__ = [
    "ZCA_FIT_NPZ_VERSION",
    "ZcaFit",
    "apply_zca_fit",
    "apply_zca_whitening",
    "fit_zca_whitening",
    "load_zca_fit_npz",
    "save_zca_fit_npz",
    "zca_matrix_from_covariance",
]


@dataclass(frozen=True)
class ZcaFit:
    """Session-level ZCA whitening parameters fit on bandpassed data.

    Parameters
    ----------
    good_channels
        Probe channel indices included in the fit (dead channels excluded).
    covariance
        Channel covariance matrix with shape ``(n_good, n_good)``.
    channel_medians
        Temporal medians per good channel from the fit recording.
    mean_robust_std
        Mean per-channel MAD scale used when ``rescale_amplitude`` is enabled.
    epsilon
        Eigenvalue regularizer used to build the whitening matrix.
    robust_cov
        Whether covariance was estimated with the all-channel artifact gate.
    artifact_pad_samples
        Half-width in samples by which rejected artifact/spike samples were
        dilated before covariance estimation.
    sampling_rate_hz, lowcut_hz, highcut_hz, filter_order
        Bandpass provenance stored for validation at apply time.
    """

    good_channels: np.ndarray
    covariance: np.ndarray
    channel_medians: np.ndarray
    mean_robust_std: float
    epsilon: float
    robust_cov: bool
    artifact_pad_samples: int = 0
    sampling_rate_hz: float = DEFAULT_INTAN_FS_HZ
    lowcut_hz: float = DEFAULT_INTAN_LOWCUT_HZ
    highcut_hz: float = DEFAULT_INTAN_HIGHCUT_HZ
    filter_order: int = DEFAULT_INTAN_BANDPASS_ORDER

    def zca_matrix(self) -> np.ndarray:
        """Return the whitening matrix derived from :attr:`covariance`."""
        return zca_matrix_from_covariance(self.covariance, self.epsilon)

    def validate_filter_params(
        self,
        *,
        sampling_rate_hz: float,
        lowcut_hz: float,
        highcut_hz: float,
        filter_order: int,
    ) -> None:
        """Raise if apply-time bandpass settings differ from the fit."""
        checks = (
            (self.sampling_rate_hz, sampling_rate_hz, "sampling_rate_hz"),
            (self.lowcut_hz, lowcut_hz, "lowcut_hz"),
            (self.highcut_hz, highcut_hz, "highcut_hz"),
            (float(self.filter_order), float(filter_order), "filter_order"),
        )
        for expected, actual, name in checks:
            if float(expected) != float(actual):
                msg = f"ZCA fit {name}={expected!r} does not match apply setting {actual!r}"
                raise ValueError(msg)

    def row_index_for_channel(self, channel: int) -> int:
        """Map a probe channel index to a row in the good-channel matrix."""
        matches = np.flatnonzero(self.good_channels == int(channel))
        if matches.size != 1:
            msg = f"channel {channel} is not in ZCA fit good_channels {self.good_channels.tolist()}"
            raise ValueError(msg)
        return int(matches[0])


def _validate_voltage_matrix(
    voltage_matrix: np.ndarray,
    *,
    epsilon: float,
    rescale_amplitude: bool | None = None,
    robust_cov: bool | None = None,
) -> tuple[int, int]:
    """Validate a ``(n_channels, n_samples)`` float voltage matrix."""
    if not isinstance(voltage_matrix, np.ndarray):
        msg = f"voltage_matrix must be a numpy.ndarray, got {type(voltage_matrix).__name__!r}"
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
        msg = f"voltage_matrix must have a floating-point dtype; got {voltage_matrix.dtype!r}"
        raise TypeError(msg)
    if epsilon <= 0:
        msg = f"epsilon must be positive, got {epsilon!r}"
        raise ValueError(msg)
    if rescale_amplitude is not None and not isinstance(
        rescale_amplitude,
        (bool, np.bool_),
    ):
        msg = f"rescale_amplitude must be bool, got {type(rescale_amplitude).__name__!r}"
        raise TypeError(msg)
    if robust_cov is not None and not isinstance(robust_cov, (bool, np.bool_)):
        msg = f"robust_cov must be bool, got {type(robust_cov).__name__!r}"
        raise TypeError(msg)
    return n_channels, n_samples


def _validate_artifact_pad_samples(artifact_pad_samples: int) -> int:
    """Validate the artifact-mask dilation half-width in samples."""
    if isinstance(artifact_pad_samples, (bool, np.bool_)):
        msg = f"artifact_pad_samples must be a non-negative int, got {artifact_pad_samples!r}"
        raise TypeError(msg)
    if isinstance(artifact_pad_samples, np.integer):
        artifact_pad_samples = int(artifact_pad_samples)
    if not isinstance(artifact_pad_samples, int):
        msg = (
            "artifact_pad_samples must be a non-negative int, got "
            f"{type(artifact_pad_samples).__name__!r}"
        )
        raise TypeError(msg)
    if artifact_pad_samples < 0:
        msg = f"artifact_pad_samples must be non-negative, got {artifact_pad_samples!r}"
        raise ValueError(msg)
    return artifact_pad_samples


def _default_artifact_pad_samples(sampling_rate_hz: float) -> int:
    """Return the default dilation half-width for ``sampling_rate_hz``."""
    return int(round(_DEFAULT_ARTIFACT_PAD_MS * 1e-3 * sampling_rate_hz))


def _resolve_artifact_pad_samples(
    artifact_pad_samples: int | None,
    sampling_rate_hz: float,
) -> int:
    """Resolve ``None`` to the default half-width derived from sample rate."""
    if artifact_pad_samples is None:
        return _default_artifact_pad_samples(sampling_rate_hz)
    return _validate_artifact_pad_samples(artifact_pad_samples)


def _validate_chunk_samples(chunk_samples: int) -> int:
    """Validate positive time-chunk length in samples."""
    if isinstance(chunk_samples, (bool, np.bool_)):
        msg = f"chunk_samples must be a positive int, got {chunk_samples!r}"
        raise TypeError(msg)
    if isinstance(chunk_samples, np.integer):
        chunk_samples = int(chunk_samples)
    if not isinstance(chunk_samples, int):
        msg = f"chunk_samples must be a positive int, got {type(chunk_samples).__name__!r}"
        raise TypeError(msg)
    if chunk_samples < 1:
        msg = f"chunk_samples must be positive, got {chunk_samples!r}"
        raise ValueError(msg)
    return chunk_samples


def _robust_std_per_channel(centered: np.ndarray) -> np.ndarray:
    """Per-channel MAD scale with shape ``(n_channels, 1)``."""
    n_channels, n_samples = centered.shape
    out = np.empty((n_channels, 1), dtype=centered.dtype)
    work = np.empty(n_samples, dtype=centered.dtype)
    for ch in range(n_channels):
        np.abs(centered[ch], out=work)
        out[ch, 0] = np.median(work, overwrite_input=True) * _MAD_TO_STD
    return out


def _dilate_rejected_mask(is_clean: np.ndarray, pad_samples: int) -> np.ndarray:
    """Expand rejected samples by ``pad_samples`` on each side."""
    if pad_samples <= 0:
        return is_clean
    rejected = np.logical_not(is_clean)
    maximum_filter1d(
        rejected,
        size=2 * pad_samples + 1,
        output=rejected,
        mode="constant",
        cval=False,
    )
    return np.logical_not(rejected, out=rejected)


def _clean_sample_mask(
    centered: np.ndarray,
    *,
    n_sigma: float = _DEFAULT_ARTIFACT_N_SIGMA,
    artifact_pad_samples: int = 0,
    chunk_samples: int = _DEFAULT_CHUNK_SAMPLES,
) -> np.ndarray:
    """Return a boolean mask of samples to keep for covariance estimation."""
    artifact_pad_samples = _validate_artifact_pad_samples(artifact_pad_samples)
    chunk_samples = _validate_chunk_samples(chunk_samples)
    n_channels, n_samples = centered.shape
    robust_std = _robust_std_per_channel(centered)
    is_clean = np.ones(n_samples, dtype=bool)
    work = np.empty(min(n_samples, chunk_samples), dtype=centered.dtype)
    for ch in range(n_channels):
        threshold = n_sigma * robust_std[ch, 0]
        row = centered[ch]
        for start in range(0, n_samples, chunk_samples):
            stop = min(start + chunk_samples, n_samples)
            mask_slice = is_clean[start:stop]
            if not np.any(mask_slice):
                continue
            buf = work[: stop - start]
            np.abs(row[start:stop], out=buf)
            mask_slice &= buf < threshold
    return _dilate_rejected_mask(is_clean, artifact_pad_samples)


def _fit_channel_stats_and_mask(
    voltage_matrix: np.ndarray,
    *,
    robust_cov: bool,
    n_sigma: float = _DEFAULT_ARTIFACT_N_SIGMA,
    artifact_pad_samples: int = 0,
    chunk_samples: int = _DEFAULT_CHUNK_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Compute 1D channel medians, MAD scales, and optional clean-sample mask.

    Uses a single ``(n_samples,)`` work buffer so memory scales as
    ``O(n_samples)`` rather than ``O(n_channels * n_samples)``.
    """
    chunk_samples = _validate_chunk_samples(chunk_samples)
    n_channels, n_samples = voltage_matrix.shape
    medians = np.empty(n_channels, dtype=voltage_matrix.dtype)
    robust_std = np.empty(n_channels, dtype=voltage_matrix.dtype)
    is_clean = np.ones(n_samples, dtype=bool) if robust_cov else None
    work = np.empty(n_samples, dtype=voltage_matrix.dtype)

    for ch in range(n_channels):
        row = voltage_matrix[ch]
        np.copyto(work, row)
        median_ch = np.median(work, overwrite_input=True)
        medians[ch] = median_ch

        np.subtract(row, median_ch, out=work)
        np.abs(work, out=work)
        robust_std_ch = np.median(work, overwrite_input=True) * _MAD_TO_STD
        robust_std[ch] = robust_std_ch

        if is_clean is not None:
            threshold = n_sigma * robust_std_ch
            for start in range(0, n_samples, chunk_samples):
                stop = min(start + chunk_samples, n_samples)
                mask_slice = is_clean[start:stop]
                if not np.any(mask_slice):
                    continue
                buf = work[: stop - start]
                np.subtract(row[start:stop], median_ch, out=buf)
                np.abs(buf, out=buf)
                mask_slice &= buf < threshold

    if is_clean is not None:
        is_clean = _dilate_rejected_mask(is_clean, artifact_pad_samples)

    return medians, robust_std, is_clean


def _estimate_chunked_covariance(
    voltage_matrix: np.ndarray,
    channel_medians: np.ndarray,
    *,
    is_clean: np.ndarray | None,
    chunk_samples: int = _DEFAULT_CHUNK_SAMPLES,
) -> np.ndarray:
    """Estimate the ``(n_channels, n_channels)`` sample covariance in chunks."""
    chunk_samples = _validate_chunk_samples(chunk_samples)
    n_channels, n_samples = voltage_matrix.shape

    if is_clean is not None:
        n_used = int(is_clean.sum())
        if n_used < 2:
            msg = (
                "robust_cov requires at least 2 samples passing the clean "
                f"artifact gate; got {n_used}. Try robust_cov=False."
            )
            raise ValueError(msg)
    else:
        n_used = n_samples

    max_block = min(n_samples, chunk_samples)
    chunk_buf = np.empty((n_channels, max_block), dtype=np.float64)
    medians_col = channel_medians[:, np.newaxis]

    def _fill_centered_chunk(start: int, stop: int) -> np.ndarray | None:
        window = voltage_matrix[:, start:stop]
        if is_clean is None:
            selected = window
        else:
            mask_slice = is_clean[start:stop]
            if np.all(mask_slice):
                selected = window
            else:
                cols = np.flatnonzero(mask_slice)
                if cols.size == 0:
                    return None
                selected = window[:, cols]

        n_cols = int(selected.shape[1])
        buf = chunk_buf[:, :n_cols]
        if voltage_matrix.dtype == np.float64:
            np.subtract(selected, medians_col, out=buf)
        else:
            np.copyto(buf, selected - medians_col)
        return buf

    sum_centered = np.zeros(n_channels, dtype=np.float64)
    for start in range(0, n_samples, chunk_samples):
        stop = min(start + chunk_samples, n_samples)
        buf = _fill_centered_chunk(start, stop)
        if buf is not None:
            sum_centered += buf.sum(axis=1)
    mean_centered = (sum_centered / float(n_used))[:, np.newaxis]

    covariance = np.zeros((n_channels, n_channels), dtype=np.float64)
    for start in range(0, n_samples, chunk_samples):
        stop = min(start + chunk_samples, n_samples)
        buf = _fill_centered_chunk(start, stop)
        if buf is not None:
            buf -= mean_centered
            covariance += buf @ buf.T

    covariance /= float(n_used - 1)
    return covariance


def zca_matrix_from_covariance(covariance: np.ndarray, epsilon: float) -> np.ndarray:
    """Build a ZCA whitening matrix from a channel covariance matrix."""
    if epsilon <= 0:
        msg = f"epsilon must be positive, got {epsilon!r}"
        raise ValueError(msg)
    u_mat, singular_values, _ = np.linalg.svd(covariance)
    return u_mat @ np.diag(1.0 / np.sqrt(singular_values + epsilon)) @ u_mat.T


def fit_zca_whitening(
    voltage_matrix: np.ndarray,
    *,
    epsilon: float = 10.0,
    robust_cov: bool = True,
    artifact_pad_samples: int | None = None,
    good_channels: np.ndarray | list[int] | None = None,
    sampling_rate_hz: float = DEFAULT_INTAN_FS_HZ,
    lowcut_hz: float = DEFAULT_INTAN_LOWCUT_HZ,
    highcut_hz: float = DEFAULT_INTAN_HIGHCUT_HZ,
    filter_order: int = DEFAULT_INTAN_BANDPASS_ORDER,
) -> ZcaFit:
    """Fit ZCA whitening on bandpassed multichannel voltage data.

    Parameters
    ----------
    voltage_matrix
        Bandpassed voltage with shape ``(n_good, n_samples)``. Not modified.
    epsilon
        Eigenvalue regularizer (see :func:`apply_zca_whitening`).
    robust_cov
        Use the all-channel artifact gate when estimating covariance.
    artifact_pad_samples
        When ``robust_cov`` is enabled, also reject this many samples on
        either side of each thresholded artifact/spike sample before fitting
        covariance. When ``None`` (default), uses ``0.5`` ms at
        ``sampling_rate_hz`` (15 samples at 30 kHz).
    good_channels
        Probe indices corresponding to each row of ``voltage_matrix``. When
        omitted, defaults to ``0 .. n_good-1``.
    sampling_rate_hz, lowcut_hz, highcut_hz, filter_order
        Bandpass metadata stored in the returned :class:`ZcaFit`.

    Returns
    -------
    ZcaFit
        Session-level fit parameters for :func:`apply_zca_fit`.

    Notes
    -----
    Caller must bandpass-filter raw voltage before fitting. Covariance,
    medians, and robust scales describe the bandpassed signal domain.
    Medians and MAD scales are computed channel-by-channel in a single 1D
    work buffer and covariance is accumulated over time chunks to avoid
    ``(n_good, n_samples)`` temporary allocations.
    """
    artifact_pad_samples = _resolve_artifact_pad_samples(
        artifact_pad_samples,
        sampling_rate_hz,
    )
    n_channels, n_samples = _validate_voltage_matrix(
        voltage_matrix,
        epsilon=epsilon,
        robust_cov=robust_cov,
    )
    if not robust_cov and n_samples < 2:
        msg = (
            "Covariance estimation needs at least 2 samples when robust_cov "
            f"is False; got n_samples={n_samples}"
        )
        raise ValueError(msg)

    if good_channels is None:
        channel_ids = np.arange(n_channels, dtype=np.int64)
    else:
        channel_ids = np.asarray(good_channels, dtype=np.int64).ravel()
        if channel_ids.shape != (n_channels,):
            msg = f"good_channels must have length n_channels={n_channels}; got {channel_ids.shape}"
            raise ValueError(msg)

    channel_medians, robust_std, is_clean = _fit_channel_stats_and_mask(
        voltage_matrix,
        robust_cov=robust_cov,
        artifact_pad_samples=artifact_pad_samples,
    )
    covariance = _estimate_chunked_covariance(
        voltage_matrix,
        channel_medians,
        is_clean=is_clean,
    )

    return ZcaFit(
        good_channels=channel_ids,
        covariance=covariance,
        channel_medians=channel_medians.astype(np.float64),
        mean_robust_std=float(np.mean(robust_std)),
        epsilon=float(epsilon),
        robust_cov=bool(robust_cov),
        artifact_pad_samples=artifact_pad_samples,
        sampling_rate_hz=float(sampling_rate_hz),
        lowcut_hz=float(lowcut_hz),
        highcut_hz=float(highcut_hz),
        filter_order=int(filter_order),
    )


def _apply_zca_fit_chunked(
    voltage_matrix: np.ndarray,
    fit: ZcaFit,
    *,
    rescale_amplitude: bool,
    chunk_samples: int = _DEFAULT_CHUNK_SAMPLES,
) -> np.ndarray:
    """In-place chunked implementation backing :func:`apply_zca_fit`."""
    chunk_samples = _validate_chunk_samples(chunk_samples)
    n_channels, n_samples = voltage_matrix.shape
    medians_col = fit.channel_medians[:, np.newaxis]
    zca_matrix = fit.zca_matrix()
    product_dtype = np.result_type(zca_matrix.dtype, voltage_matrix.dtype)
    chunk_buf = np.empty((n_channels, min(n_samples, chunk_samples)), dtype=product_dtype)

    for start in range(0, n_samples, chunk_samples):
        stop = min(start + chunk_samples, n_samples)
        block = voltage_matrix[:, start:stop]
        block -= medians_col
        buf = chunk_buf[:, : stop - start]
        np.matmul(zca_matrix, block, out=buf)
        block[:] = buf
        if rescale_amplitude:
            block *= fit.mean_robust_std

    return voltage_matrix


def apply_zca_fit(
    voltage_matrix: np.ndarray,
    fit: ZcaFit,
    *,
    rescale_amplitude: bool = True,
) -> np.ndarray:
    """Apply a saved :class:`ZcaFit` to bandpassed multichannel data.

    Parameters
    ----------
    voltage_matrix
        Bandpassed voltage with shape ``(n_good, n_samples)``. **Modified in
        place**; returned array is the same object.
    fit
        Parameters from :func:`fit_zca_whitening`.
    rescale_amplitude
        Multiply whitened data by :attr:`ZcaFit.mean_robust_std` when ``True``.

    Returns
    -------
    numpy.ndarray
        Whitened data; same array as ``voltage_matrix``.

    Notes
    -----
    Caller must bandpass-filter before calling. Session medians from ``fit``
    are subtracted (not snippet-local medians). Matrix multiplication runs in
    time chunks so peak extra memory stays bounded.
    """
    n_channels, _ = _validate_voltage_matrix(
        voltage_matrix,
        epsilon=fit.epsilon,
        rescale_amplitude=rescale_amplitude,
    )
    if n_channels != fit.good_channels.shape[0]:
        msg = (
            "voltage_matrix channel count must match ZcaFit.good_channels; "
            f"got {n_channels}, expected {fit.good_channels.shape[0]}"
        )
        raise ValueError(msg)

    return _apply_zca_fit_chunked(
        voltage_matrix,
        fit,
        rescale_amplitude=rescale_amplitude,
    )


def apply_zca_whitening(
    voltage_matrix: np.ndarray,
    epsilon: float = 10.0,
    rescale_amplitude: bool = True,
    robust_cov: bool = True,
    artifact_pad_samples: int | None = None,
    sampling_rate_hz: float = DEFAULT_INTAN_FS_HZ,
) -> np.ndarray:
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
        ``False``, use all centered samples.
    artifact_pad_samples : int or None, optional
        When ``robust_cov`` is enabled, also reject this many samples on
        either side of each thresholded artifact/spike sample before fitting
        covariance. When ``None`` (default), uses ``0.5`` ms at
        ``sampling_rate_hz`` (15 samples at 30 kHz).
    sampling_rate_hz : float, optional
        Sample rate used to resolve the default ``artifact_pad_samples``.
        Default is :data:`~ephys.processing.filtering.DEFAULT_INTAN_FS_HZ`.

    Returns
    -------
    ndarray
        The whitened data; same array as ``voltage_matrix``.

    Notes
    -----
    This routine mutates ``voltage_matrix`` for memory efficiency. Copy the
    input first if the original array must be preserved.

    The artifact gate when ``robust_cov`` is ``True`` keeps sample ``t`` iff
    ``|x[c, t]| < 4 * MAD_c`` for every channel ``c``, where ``MAD_c`` is the
    robust standard deviation of channel ``c`` after median centering. When
    ``artifact_pad_samples > 0``, samples within that many indices of any
    rejected sample are also excluded from covariance estimation.
    """
    _validate_voltage_matrix(
        voltage_matrix,
        epsilon=epsilon,
        rescale_amplitude=rescale_amplitude,
        robust_cov=robust_cov,
    )
    fit = fit_zca_whitening(
        voltage_matrix,
        epsilon=epsilon,
        robust_cov=robust_cov,
        artifact_pad_samples=artifact_pad_samples,
        sampling_rate_hz=sampling_rate_hz,
    )
    return apply_zca_fit(
        voltage_matrix,
        fit,
        rescale_amplitude=rescale_amplitude,
    )


def save_zca_fit_npz(path: str | Path, fit: ZcaFit) -> None:
    """Write a :class:`ZcaFit` to an ``.npz`` artifact."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        version=np.int32(ZCA_FIT_NPZ_VERSION),
        good_channels=fit.good_channels.astype(np.int64),
        covariance=fit.covariance.astype(np.float64),
        channel_medians=fit.channel_medians.astype(np.float64),
        mean_robust_std=np.float64(fit.mean_robust_std),
        epsilon=np.float64(fit.epsilon),
        robust_cov=np.bool_(fit.robust_cov),
        artifact_pad_samples=np.int32(fit.artifact_pad_samples),
        sampling_rate_hz=np.float64(fit.sampling_rate_hz),
        lowcut_hz=np.float64(fit.lowcut_hz),
        highcut_hz=np.float64(fit.highcut_hz),
        filter_order=np.int32(fit.filter_order),
    )


def load_zca_fit_npz(path: str | Path) -> ZcaFit:
    """Load a :class:`ZcaFit` written by :func:`save_zca_fit_npz`."""
    with np.load(Path(path), allow_pickle=False) as archive:
        version = int(archive["version"])
        if version not in (1, ZCA_FIT_NPZ_VERSION):
            msg = f"Unsupported ZCA fit npz version {version}; expected 1 or {ZCA_FIT_NPZ_VERSION}"
            raise ValueError(msg)
        artifact_pad_samples = 0 if version < 2 else int(archive["artifact_pad_samples"])
        return ZcaFit(
            good_channels=np.asarray(archive["good_channels"], dtype=np.int64),
            covariance=np.asarray(archive["covariance"], dtype=np.float64),
            channel_medians=np.asarray(archive["channel_medians"], dtype=np.float64),
            mean_robust_std=float(archive["mean_robust_std"]),
            epsilon=float(archive["epsilon"]),
            robust_cov=bool(archive["robust_cov"]),
            artifact_pad_samples=artifact_pad_samples,
            sampling_rate_hz=float(archive["sampling_rate_hz"]),
            lowcut_hz=float(archive["lowcut_hz"]),
            highcut_hz=float(archive["highcut_hz"]),
            filter_order=int(archive["filter_order"]),
        )
