"""Tests for ZCA whitening fit/apply and npz serialization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ephys.processing.zca import (
    _apply_zca_fit_chunked,
    _clean_sample_mask,
    _estimate_chunked_covariance,
    _fit_channel_stats_and_mask,
    apply_zca_fit,
    apply_zca_whitening,
    fit_zca_whitening,
    load_zca_fit_npz,
    save_zca_fit_npz,
    zca_matrix_from_covariance,
)


def _synthetic_bandpassed_voltage(
    n_channels: int = 4,
    n_samples: int = 500,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shared = rng.standard_normal(n_samples)
    mixing = rng.standard_normal((n_channels, n_channels))
    data = mixing @ np.vstack([shared, rng.standard_normal((n_channels - 1, n_samples))])
    return data.astype(np.float64)


def _spatially_correlated_noise(
    n_channels: int,
    n_samples: int,
    *,
    seed: int,
    mixing: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return bandpassed-like noise and the spatial mixing matrix."""
    rng = np.random.default_rng(seed)
    if mixing is None:
        mixing = rng.standard_normal((n_channels, n_channels))
    latent = rng.standard_normal((n_channels, n_samples))
    return (mixing @ latent).astype(np.float64), mixing


def _off_diagonal_frobenius_error(
    estimated: np.ndarray,
    target: np.ndarray,
) -> float:
    mask = ~np.eye(estimated.shape[0], dtype=bool)
    return float(np.linalg.norm(estimated[mask] - target[mask]))


def _biphasic_spike(width_samples: int, amplitude: float) -> np.ndarray:
    times = np.linspace(-1.0, 1.0, width_samples)
    return (amplitude * (-times) * np.exp(-4.0 * times**2)).astype(np.float64)


def _inject_spikes(
    data: np.ndarray,
    spike_times: np.ndarray,
    *,
    source_channel: int = 0,
    width_samples: int = 31,
    amplitude: float = 80.0,
    neighbor_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Add translated biphasic events without mutating the input."""
    output = data.copy()
    template = _biphasic_spike(width_samples, amplitude)
    half = width_samples // 2
    n_channels = data.shape[0]
    if neighbor_weights is None:
        neighbor_weights = np.zeros(n_channels, dtype=np.float64)
        neighbor_weights[source_channel] = 1.0
        if source_channel + 1 < n_channels:
            neighbor_weights[source_channel + 1] = 0.35
    for spike_time in spike_times:
        start = int(spike_time) - half
        end = start + width_samples
        if start < 0 or end > data.shape[1]:
            continue
        for channel, weight in enumerate(neighbor_weights):
            if weight == 0.0:
                continue
            output[channel, start:end] += weight * template
    return output


def test_fit_apply_matches_monolithic_apply() -> None:
    """fit + apply reproduces apply_zca_whitening on the same data."""
    voltage = _synthetic_bandpassed_voltage()
    expected = apply_zca_whitening(np.array(voltage, copy=True))
    fit = fit_zca_whitening(voltage)
    actual = apply_zca_fit(np.array(voltage, copy=True), fit)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_fit_does_not_mutate_input() -> None:
    """fit_zca_whitening leaves the input array unchanged."""
    voltage = _synthetic_bandpassed_voltage()
    before = voltage.copy()
    fit_zca_whitening(voltage)
    np.testing.assert_array_equal(voltage, before)


def test_zca_fit_npz_round_trip(tmp_path: Path) -> None:
    """save/load npz preserves whitening behavior."""
    voltage = _synthetic_bandpassed_voltage()
    fit = fit_zca_whitening(
        voltage,
        good_channels=[10, 12, 14, 16],
        artifact_pad_samples=15,
    )
    path = tmp_path / "zca_fit.npz"
    save_zca_fit_npz(path, fit)
    loaded = load_zca_fit_npz(path)
    expected = apply_zca_fit(np.array(voltage, copy=True), fit)
    actual = apply_zca_fit(np.array(voltage, copy=True), loaded)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(loaded.good_channels, fit.good_channels)
    assert loaded.artifact_pad_samples == 15


def test_row_index_for_channel_maps_probe_indices() -> None:
    """row_index_for_channel resolves probe labels."""
    voltage = _synthetic_bandpassed_voltage(n_channels=3)
    fit = fit_zca_whitening(voltage, good_channels=[2, 5, 9])
    assert fit.row_index_for_channel(5) == 1


def test_row_index_for_dead_channel_raises() -> None:
    """Missing probe channels raise ValueError."""
    voltage = _synthetic_bandpassed_voltage(n_channels=2)
    fit = fit_zca_whitening(voltage, good_channels=[1, 3])
    with pytest.raises(ValueError, match="not in ZCA fit"):
        fit.row_index_for_channel(0)


def test_zca_matrix_from_covariance_is_symmetric() -> None:
    """Whitening matrix built from covariance is symmetric."""
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    matrix = zca_matrix_from_covariance(cov, epsilon=0.1)
    np.testing.assert_allclose(matrix, matrix.T, rtol=1e-10, atol=1e-10)


def test_robust_cov_rejects_common_mode_burst() -> None:
    """robust_cov ignores brief all-channel artifacts when fitting covariance."""
    noise, mixing = _spatially_correlated_noise(4, 4000, seed=1)
    target_cov = np.cov(noise)
    contaminated = noise.copy()
    contaminated[:, 1000:1010] += 250.0

    fit_plain = fit_zca_whitening(contaminated, robust_cov=False, epsilon=1.0)
    fit_robust = fit_zca_whitening(
        contaminated,
        robust_cov=True,
        artifact_pad_samples=0,
        epsilon=1.0,
    )

    plain_error = _off_diagonal_frobenius_error(fit_plain.covariance, target_cov)
    robust_error = _off_diagonal_frobenius_error(fit_robust.covariance, target_cov)
    assert robust_error < plain_error


def test_robust_cov_reduces_spike_contamination_in_covariance() -> None:
    """Spike epochs bias covariance less when robust_cov and padding are enabled."""
    noise, _ = _spatially_correlated_noise(4, 6000, seed=2)
    target_cov = np.cov(noise)
    rng = np.random.default_rng(3)
    spike_times = rng.choice(6000, size=40, replace=False)
    contaminated = _inject_spikes(noise, spike_times, amplitude=100.0)

    fit_plain = fit_zca_whitening(contaminated, robust_cov=False, epsilon=1.0)
    fit_robust = fit_zca_whitening(
        contaminated,
        robust_cov=True,
        artifact_pad_samples=15,
        epsilon=1.0,
    )

    plain_error = _off_diagonal_frobenius_error(fit_plain.covariance, target_cov)
    robust_error = _off_diagonal_frobenius_error(fit_robust.covariance, target_cov)
    assert robust_error < plain_error


def test_robust_cov_preserves_more_spike_amplitude_than_plain_fit() -> None:
    """Whitening fit on spike-contaminated covariance suppresses spikes more."""
    noise, _ = _spatially_correlated_noise(4, 6000, seed=4)
    rng = np.random.default_rng(5)
    spike_times = rng.choice(6000, size=40, replace=False)
    contaminated = _inject_spikes(noise, spike_times, amplitude=100.0)
    spike_time = int(spike_times[0])

    fit_plain = fit_zca_whitening(contaminated, robust_cov=False, epsilon=1.0)
    fit_robust = fit_zca_whitening(
        contaminated,
        robust_cov=True,
        artifact_pad_samples=15,
        epsilon=1.0,
    )

    plain_out = apply_zca_fit(np.array(contaminated, copy=True), fit_plain)
    robust_out = apply_zca_fit(np.array(contaminated, copy=True), fit_robust)
    plain_peak = float(np.max(np.abs(plain_out[0, spike_time - 20 : spike_time + 20])))
    robust_peak = float(np.max(np.abs(robust_out[0, spike_time - 20 : spike_time + 20])))
    assert robust_peak > plain_peak


def test_artifact_pad_dilates_rejected_samples() -> None:
    """Padding excludes more samples around thresholded spike epochs."""
    noise, _ = _spatially_correlated_noise(1, 1000, seed=6)
    spike_times = np.array([500])
    contaminated = _inject_spikes(
        noise,
        spike_times,
        source_channel=0,
        width_samples=31,
        amplitude=120.0,
    )
    centered = contaminated - np.median(contaminated, axis=1, keepdims=True)

    mask_no_pad = _clean_sample_mask(centered, artifact_pad_samples=0)
    mask_padded = _clean_sample_mask(centered, artifact_pad_samples=15)

    rejected_no_pad = int((~mask_no_pad).sum())
    rejected_padded = int((~mask_padded).sum())
    assert rejected_padded > rejected_no_pad
    assert rejected_padded >= 2 * 15 + 1


def test_robust_std_stable_at_low_spike_rate() -> None:
    """Rare spikes should not materially inflate the MAD noise scale."""
    noise, _ = _spatially_correlated_noise(4, 8000, seed=7)
    fit_noise = fit_zca_whitening(noise, robust_cov=True)

    rng = np.random.default_rng(8)
    spike_times = rng.choice(8000, size=20, replace=False)
    with_spikes = _inject_spikes(noise, spike_times, amplitude=100.0)
    fit_spikes = fit_zca_whitening(with_spikes, robust_cov=True)

    assert fit_spikes.mean_robust_std <= 1.2 * fit_noise.mean_robust_std


def test_whitening_reduces_shared_noise_correlation() -> None:
    """ZCA lowers cross-channel correlation for spatially shared noise."""
    noise, _ = _spatially_correlated_noise(4, 5000, seed=9)
    raw_corr = np.corrcoef(noise)
    whitened = apply_zca_whitening(
        np.array(noise, copy=True),
        epsilon=1.0,
        robust_cov=True,
        artifact_pad_samples=0,
    )
    white_corr = np.corrcoef(whitened)

    raw_off_diag = np.abs(raw_corr[~np.eye(4, dtype=bool)]).mean()
    white_off_diag = np.abs(white_corr[~np.eye(4, dtype=bool)]).mean()
    assert white_off_diag < raw_off_diag


def test_robust_cov_raises_when_all_samples_rejected() -> None:
    """An all-artifact snippet cannot produce a covariance fit."""
    voltage = np.zeros((2, 100), dtype=np.float64)
    voltage[:, :] = 1000.0
    with pytest.raises(ValueError, match="at least 2 samples"):
        fit_zca_whitening(voltage, robust_cov=True)


def test_artifact_pad_samples_must_be_non_negative() -> None:
    """Negative padding is rejected."""
    voltage = _synthetic_bandpassed_voltage()
    with pytest.raises(ValueError, match="non-negative"):
        fit_zca_whitening(voltage, artifact_pad_samples=-1)


def test_default_artifact_pad_is_half_ms() -> None:
    """Default padding is 0.5 ms derived from sampling_rate_hz."""
    voltage = _synthetic_bandpassed_voltage()
    fit = fit_zca_whitening(voltage)
    assert fit.artifact_pad_samples == 15

    fit_20k = fit_zca_whitening(voltage, sampling_rate_hz=20_000.0)
    assert fit_20k.artifact_pad_samples == 10


def _dense_reference_fit_and_apply(
    voltage_matrix: np.ndarray,
    *,
    epsilon: float = 10.0,
    robust_cov: bool = True,
    artifact_pad_samples: int = 15,
    rescale_amplitude: bool = True,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Reference dense NumPy ZCA implementation used before chunking."""
    medians = np.median(voltage_matrix, axis=1, keepdims=True)
    centered = voltage_matrix - medians
    robust_std = np.median(np.abs(centered), axis=1, keepdims=True) * 1.4826
    if robust_cov:
        rejected = ~np.all(np.abs(centered) < (4.0 * robust_std), axis=0)
        if artifact_pad_samples > 0:
            kernel = np.ones(2 * artifact_pad_samples + 1, dtype=np.int64)
            rejected = np.convolve(rejected.astype(np.int64), kernel, mode="same") > 0
        covariance = np.atleast_2d(np.cov(centered[:, ~rejected]))
    else:
        covariance = np.atleast_2d(np.cov(centered))

    zca_mat = zca_matrix_from_covariance(np.asarray(covariance, dtype=np.float64), epsilon)
    out = voltage_matrix.copy()
    out -= medians.astype(np.float64)
    out[:] = zca_mat @ out
    mean_std = float(np.mean(robust_std))
    if rescale_amplitude:
        out *= mean_std
    return (
        medians.ravel().astype(np.float64),
        mean_std,
        np.asarray(covariance, dtype=np.float64),
        out,
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("robust_cov", [True, False])
@pytest.mark.parametrize("fortran_order", [False, True])
def test_decomposed_fit_and_apply_match_dense_reference(
    dtype: type[np.floating],
    robust_cov: bool,
    fortran_order: bool,
) -> None:
    """Channel-wise + time-chunked ZCA reproduces the dense NumPy formulation."""
    noise, _ = _spatially_correlated_noise(5, 600, seed=17)
    contaminated = _inject_spikes(noise, np.array([120, 310, 450]), amplitude=120.0)
    contaminated[:, 200:260] += 300.0  # spans > 1 chunk when chunk_samples=37
    arr = np.asarray(contaminated, dtype=dtype, order="F" if fortran_order else "C")

    ref_medians, ref_std, ref_cov, ref_out = _dense_reference_fit_and_apply(
        arr,
        epsilon=2.5,
        robust_cov=robust_cov,
        artifact_pad_samples=8,
        rescale_amplitude=True,
    )

    medians, robust_std, is_clean = _fit_channel_stats_and_mask(
        arr,
        robust_cov=robust_cov,
        artifact_pad_samples=8,
        chunk_samples=37,
    )
    cov = _estimate_chunked_covariance(
        arr,
        medians,
        is_clean=is_clean,
        chunk_samples=37,
    )
    np.testing.assert_array_equal(medians.astype(np.float64), ref_medians)
    assert float(np.mean(robust_std)) == pytest.approx(ref_std, rel=0.0, abs=0.0)
    np.testing.assert_allclose(cov, ref_cov, rtol=1e-12, atol=1e-12)

    fit = fit_zca_whitening(
        arr,
        epsilon=2.5,
        robust_cov=robust_cov,
        artifact_pad_samples=8,
    )
    actual_out = np.array(arr, copy=True)
    _apply_zca_fit_chunked(
        actual_out,
        fit,
        rescale_amplitude=True,
        chunk_samples=37,
    )
    rtol = 1e-10 if dtype is np.float64 else 1e-6
    atol = 1e-10 if dtype is np.float64 else 1e-5
    np.testing.assert_allclose(actual_out, ref_out, rtol=rtol, atol=atol)


def test_large_artifact_pad_does_not_overflow_dilation() -> None:
    """Sustained artifacts with pad >= 64 samples stay rejected across their interior."""
    centered = np.zeros((2, 500), dtype=np.float64)
    centered[:, :250] = 1.0
    centered[:, 250:] = -1.0
    centered[:, 150:350] = 50.0
    mask = _clean_sample_mask(centered, artifact_pad_samples=80, chunk_samples=64)
    assert not np.any(mask[70:430])
    assert np.all(mask[:70])
    assert np.all(mask[430:])


def test_single_channel_fit_and_apply() -> None:
    """Single-channel input yields a (1, 1) covariance and valid whitened trace."""
    voltage = _synthetic_bandpassed_voltage(n_channels=2, n_samples=400)[:1, :]
    fit = fit_zca_whitening(voltage, good_channels=[7])
    assert fit.covariance.shape == (1, 1)
    whitened = apply_zca_fit(voltage.copy(), fit)
    assert whitened.shape == voltage.shape
