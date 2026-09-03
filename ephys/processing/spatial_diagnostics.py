"""Memory-efficient spatial reference diagnostics for multichannel voltage data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MAD_TO_STD = 1.4826
_DEFAULT_SEGMENT_SAMPLES = 30_000
_DEFAULT_N_SEGMENTS = 3
_INDEX_CHUNK_SIZE = 10_000
_SPATIAL_MEDIAN_MIN_STD_UV = 1e-3

__all__ = [
    "SpatialDiagnostics",
    "compute_spatial_diagnostics",
    "format_spatial_diagnostics_line",
    "plan_spatial_subsample_indices",
    "spatial_reference_hint",
]


@dataclass(frozen=True)
class SpatialDiagnostics:
    """Summary statistics for choosing a spatial reference strategy."""

    mean_abs_off_diag_corr: float
    median_mad_uV: float
    dominant_eigenvalue_fraction: float
    pc1_median_correlation: float | None
    residual_spatial_median_rms_uV: float
    n_samples_used: int


def plan_spatial_subsample_indices(
    n_samples: int,
    *,
    n_segments: int = _DEFAULT_N_SEGMENTS,
    samples_per_segment: int = _DEFAULT_SEGMENT_SAMPLES,
) -> np.ndarray:
    """Return spread sample indices for diagnostics without loading the full recording.

    Indices are taken from the start, middle, and end of the recording (when
    ``n_segments >= 3``), with at most ``samples_per_segment`` evenly spaced
    samples per segment.
    """
    if n_samples < 1:
        msg = f"n_samples must be positive, got {n_samples}"
        raise ValueError(msg)
    if n_samples == 1:
        return np.zeros(1, dtype=np.int64)

    segment_len = min(int(samples_per_segment), n_samples)
    if n_segments <= 1:
        starts = (0,)
    elif n_segments == 2:
        starts = (0, max(0, n_samples - segment_len))
    else:
        starts = (
            0,
            max(0, n_samples // 2 - segment_len // 2),
            max(0, n_samples - segment_len),
        )

    parts: list[np.ndarray] = []
    for start in starts:
        end = min(start + segment_len, n_samples)
        count = end - start
        if count < 1:
            continue
        take = min(segment_len, count)
        stride = max(1, int(np.ceil(count / take)))
        parts.append(np.arange(start, end, stride, dtype=np.int64)[:take])

    if not parts:
        return np.arange(min(n_samples, segment_len), dtype=np.int64)
    return np.unique(np.concatenate(parts))


def compute_spatial_diagnostics(
    voltage_matrix: np.ndarray,
    channel_indices: np.ndarray | list[int],
    sample_indices: np.ndarray,
) -> SpatialDiagnostics:
    """Compute spatial diagnostics on a subsample using bounded working memory.

    Parameters
    ----------
    voltage_matrix
        Voltage array with shape ``(n_channels, n_samples)``.
    channel_indices
        Probe channel rows to include in the analysis.
    sample_indices
        Time indices returned by :func:`plan_spatial_subsample_indices`.

    Returns
    -------
    SpatialDiagnostics
        Metrics estimated from the subsample only.
    """
    channels = np.asarray(channel_indices, dtype=np.int64).ravel()
    indices = np.asarray(sample_indices, dtype=np.int64).ravel()
    if channels.size < 1:
        msg = "channel_indices must contain at least one channel"
        raise ValueError(msg)
    if indices.size < 2:
        msg = (
            "sample_indices must contain at least 2 samples for covariance; "
            f"got {indices.size}"
        )
        raise ValueError(msg)

    n_channels = channels.shape[0]
    n_used = int(indices.shape[0])

    sum_x = np.zeros(n_channels, dtype=np.float64)
    sum_xx = np.zeros((n_channels, n_channels), dtype=np.float64)
    count = 0

    for start in range(0, n_used, _INDEX_CHUNK_SIZE):
        idx = indices[start : start + _INDEX_CHUNK_SIZE]
        block = np.asarray(voltage_matrix[channels][:, idx], dtype=np.float64)
        sum_x += block.sum(axis=1)
        sum_xx += block @ block.T
        count += block.shape[1]

    mean = sum_x / count
    covariance = (sum_xx / count) - np.outer(mean, mean)

    std = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / np.outer(std, std)
    off_diag = correlation[~np.eye(n_channels, dtype=bool)]
    mean_abs_off_diag = float(np.mean(np.abs(off_diag)))

    u_mat, singular_values, _ = np.linalg.svd(covariance)
    evals_sorted = np.sort(singular_values)[::-1]
    eval_sum = float(evals_sorted.sum())
    dominant_fraction = (
        float(evals_sorted[0] / eval_sum) if eval_sum > 0.0 else 0.0
    )

    pc1 = u_mat[:, 0]
    pc1_scores = np.empty(count, dtype=np.float64)
    spatial_median = np.empty(count, dtype=np.float64)
    offset = 0
    for start in range(0, n_used, _INDEX_CHUNK_SIZE):
        idx = indices[start : start + _INDEX_CHUNK_SIZE]
        block = np.asarray(voltage_matrix[channels][:, idx], dtype=np.float64)
        centered = block - mean[:, np.newaxis]
        width = block.shape[1]
        pc1_scores[offset : offset + width] = pc1 @ centered
        spatial_median[offset : offset + width] = np.median(block, axis=0)
        offset += width

    residual_median_rms = float(np.sqrt(np.mean(spatial_median**2)))
    median_std = float(np.std(spatial_median))
    if (
        median_std > _SPATIAL_MEDIAN_MIN_STD_UV
        and np.std(pc1_scores) > 0.0
    ):
        pc1_median_r: float | None = float(
            np.corrcoef(pc1_scores, spatial_median)[0, 1]
        )
    else:
        pc1_median_r = None

    channel_mads = np.empty(n_channels, dtype=np.float64)
    for channel_row in range(n_channels):
        channel_values = np.empty(count, dtype=np.float64)
        offset = 0
        for start in range(0, n_used, _INDEX_CHUNK_SIZE):
            idx = indices[start : start + _INDEX_CHUNK_SIZE]
            block = np.asarray(
                voltage_matrix[channels[channel_row], idx],
                dtype=np.float64,
            )
            width = block.shape[0]
            channel_values[offset : offset + width] = block
            offset += width
        channel_median = float(np.median(channel_values))
        channel_mads[channel_row] = (
            np.median(np.abs(channel_values - channel_median)) * _MAD_TO_STD
        )

    return SpatialDiagnostics(
        mean_abs_off_diag_corr=mean_abs_off_diag,
        median_mad_uV=float(np.median(channel_mads)),
        dominant_eigenvalue_fraction=dominant_fraction,
        pc1_median_correlation=pc1_median_r,
        residual_spatial_median_rms_uV=residual_median_rms,
        n_samples_used=count,
    )


def _format_pc1_median_field(diagnostics: SpatialDiagnostics) -> str:
    """Format the PC1-vs-median field, handling the post-CMR undefined case."""
    if diagnostics.pc1_median_correlation is None:
        return (
            "PC1 vs median r=n/a "
            f"(residual median RMS={diagnostics.residual_spatial_median_rms_uV:.2f} uV)"
        )
    return f"PC1 vs median r={diagnostics.pc1_median_correlation:.3f}"


def spatial_reference_hint(
    diagnostics: SpatialDiagnostics,
    *,
    label: str = "after bandpass",
) -> str:
    """Return a short recommendation based on subsampled diagnostics."""
    normalized = label.strip().lower()
    if "cmr" in normalized and "zca" not in normalized:
        if diagnostics.mean_abs_off_diag_corr >= 0.35:
            if diagnostics.dominant_eigenvalue_fraction >= 0.4:
                return (
                    "residual spatial correlation remains; try CMR then ZCA if sorting "
                    "needs lower correlation"
                )
            return "residual correlation is mixed; CMR alone may still be sufficient"
        return "residual correlation is modest; CMR alone is likely sufficient"
    if "zca" in normalized:
        if diagnostics.mean_abs_off_diag_corr <= 0.1:
            return "channels are largely decorrelated after ZCA"
        return "some spatial correlation remains after ZCA"

    if (
        diagnostics.mean_abs_off_diag_corr >= 0.7
        and diagnostics.dominant_eigenvalue_fraction >= 0.6
    ):
        return "strong common-mode structure; CMR is likely sufficient"
    if diagnostics.mean_abs_off_diag_corr >= 0.35:
        if diagnostics.dominant_eigenvalue_fraction >= 0.4:
            return "spatially correlated; CMR is a good first step"
        return "mixed spatial structure; compare CMR with CMR then ZCA"
    if diagnostics.dominant_eigenvalue_fraction < 0.4:
        return "correlation spread across several spatial modes; ZCA may be appropriate"
    return "moderate spatial correlation; CMR alone may be enough"


def format_spatial_diagnostics_line(
    label: str,
    diagnostics: SpatialDiagnostics,
    *,
    include_hint: bool = True,
) -> str:
    """Format diagnostics for logging during preprocessing."""
    line = (
        f"[{label}] mean |off-corr|={diagnostics.mean_abs_off_diag_corr:.3f}, "
        f"MAD median={diagnostics.median_mad_uV:.1f} uV, "
        f"dominant eval fraction={diagnostics.dominant_eigenvalue_fraction:.2f}, "
        f"{_format_pc1_median_field(diagnostics)} "
        f"(subsample n={diagnostics.n_samples_used})"
    )
    if include_hint:
        line += f" -> {spatial_reference_hint(diagnostics, label=label)}"
    return line
