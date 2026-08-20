"""Tests for spatial reference diagnostics."""

from __future__ import annotations

import numpy as np

from ephys.processing.spatial_diagnostics import (
    compute_spatial_diagnostics,
    format_spatial_diagnostics_line,
    plan_spatial_subsample_indices,
    spatial_reference_hint,
)


def _common_mode_recording(
    n_channels: int = 4,
    n_samples: int = 10_000,
    *,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(n_samples) * 20.0
    independent = rng.standard_normal((n_channels, n_samples)) * 2.0
    return common + independent


def test_plan_spatial_subsample_indices_spreads_across_recording() -> None:
    """Subsample indices cover start, middle, and end."""
    indices = plan_spatial_subsample_indices(100_000, samples_per_segment=1000)
    assert indices.size >= 2
    assert indices.min() == 0
    assert indices.max() >= 99_000
    assert np.all(np.diff(indices) > 0)


def test_common_mode_has_high_off_diagonal_correlation() -> None:
    """Synthetic common-mode data reports high off-diagonal correlation."""
    voltage = _common_mode_recording()
    indices = plan_spatial_subsample_indices(voltage.shape[1])
    diagnostics = compute_spatial_diagnostics(voltage, np.arange(4), indices)
    assert diagnostics.mean_abs_off_diag_corr > 0.8
    assert diagnostics.dominant_eigenvalue_fraction > 0.6
    assert abs(diagnostics.pc1_median_correlation) > 0.9


def test_cmr_like_subtraction_lowers_reported_correlation() -> None:
    """Removing the spatial median lowers diagnostic correlation."""
    voltage = _common_mode_recording()
    indices = plan_spatial_subsample_indices(voltage.shape[1])
    before = compute_spatial_diagnostics(voltage, np.arange(4), indices)
    cmr = voltage - np.median(voltage, axis=0, keepdims=True)
    after = compute_spatial_diagnostics(cmr, np.arange(4), indices)
    assert after.mean_abs_off_diag_corr < before.mean_abs_off_diag_corr
    assert after.pc1_median_correlation is None
    assert after.residual_spatial_median_rms_uV < 1e-6


def test_format_line_includes_hint() -> None:
    """Formatted output includes the recommendation text."""
    voltage = _common_mode_recording()
    indices = plan_spatial_subsample_indices(voltage.shape[1])
    diagnostics = compute_spatial_diagnostics(voltage, np.arange(4), indices)
    line = format_spatial_diagnostics_line("after bandpass", diagnostics)
    assert line.startswith("[after bandpass]")
    assert spatial_reference_hint(diagnostics, label="after bandpass") in line


def test_post_cmr_line_reports_not_applicable_pc1() -> None:
    """Post-CMR diagnostics explain that PC1-vs-median is undefined."""
    voltage = _common_mode_recording()
    indices = plan_spatial_subsample_indices(voltage.shape[1])
    cmr = voltage - np.median(voltage, axis=0, keepdims=True)
    diagnostics = compute_spatial_diagnostics(cmr, np.arange(4), indices)
    line = format_spatial_diagnostics_line("after CMR", diagnostics)
    assert "PC1 vs median r=n/a" in line
    assert "residual median RMS=" in line
