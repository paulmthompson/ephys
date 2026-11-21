"""Tests for :mod:`ephys.processing.spike_intervals`."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from ephys.processing.spike_intervals import (
    TrialCv2Summary,
    adjacent_isi_cv2,
    bootstrap_median_ci_by_trial,
    collect_adjacent_isi_cv2,
    collect_adjacent_isi_cv2_by_trial,
    collect_inter_spike_intervals,
    collect_triple_spike_isi_diff_ms,
    first_three_spike_isi_pair_ms,
    first_two_spike_isi,
    flatten_finite_trial_values,
    inter_spike_intervals,
    summarize_trial_cv2,
    triple_spike_isi_contrast_from_pair_ms,
    triple_spike_isi_pairs_ms,
)


def test_inter_spike_intervals_sorts_and_filters_nonfinite() -> None:
    """ISI uses sorted finite spikes only."""
    spikes = np.array([0.030, np.nan, 0.010, 0.020], dtype=np.float64)
    isi = inter_spike_intervals(spikes)
    assert isi.tolist() == pytest.approx([0.010, 0.010])


def test_adjacent_isi_cv2_single_value() -> None:
    """Three spikes produce one conventional nonnegative CV2 value."""
    spikes = np.array([0.010, 0.014, 0.020], dtype=np.float64)
    cv2 = adjacent_isi_cv2(spikes)
    assert cv2.tolist() == pytest.approx([0.4])


def test_adjacent_isi_cv2_multiple_pairs() -> None:
    """Four spikes produce two adjacent-ISI CV2 values."""
    spikes = np.array([0.000, 0.002, 0.005, 0.011], dtype=np.float64)
    cv2 = adjacent_isi_cv2(spikes)
    assert cv2.tolist() == pytest.approx([0.4, 2.0 / 3.0])


def test_adjacent_isi_cv2_ignores_zero_denominator_pairs() -> None:
    """Duplicate-only adjacent ISIs with zero denominator are omitted."""
    spikes = np.array([0.010, 0.010, 0.010, 0.020], dtype=np.float64)
    cv2 = adjacent_isi_cv2(spikes)
    assert cv2.tolist() == pytest.approx([2.0])


def test_collect_adjacent_isi_cv2_does_not_cross_trial_boundaries() -> None:
    """Short trials are not stitched together to manufacture CV2 pairs."""
    trials = (
        np.array([0.010, 0.020], dtype=np.float64),
        np.array([0.030, 0.040], dtype=np.float64),
    )
    assert collect_adjacent_isi_cv2(trials).size == 0
    by_trial = collect_adjacent_isi_cv2_by_trial(trials)
    assert len(by_trial) == 2
    assert all(values.size == 0 for values in by_trial)


def test_collect_adjacent_isi_cv2_flattens_trial_values() -> None:
    """Flattened collection preserves only within-trial CV2 values."""
    trials = (
        np.array([0.000, 0.002, 0.005], dtype=np.float64),
        np.array([0.010], dtype=np.float64),
        np.array([0.000, 0.004, 0.005], dtype=np.float64),
    )
    cv2 = collect_adjacent_isi_cv2(trials)
    assert cv2.tolist() == pytest.approx([0.4, 1.2])


def test_collect_inter_spike_intervals_and_flatten_finite() -> None:
    """Pooled within-trial ISIs ignore short trials and non-finite entries."""
    trials = (
        np.array([0.0, 0.01, 0.03], dtype=np.float64),
        np.array([0.05], dtype=np.float64),
        np.array([0.10, 0.12], dtype=np.float64),
    )
    isi = collect_inter_spike_intervals(trials)
    assert isi.tolist() == pytest.approx([0.01, 0.02, 0.02])

    flattened = flatten_finite_trial_values(
        [np.array([1.0, np.nan]), np.array([], dtype=np.float64), np.array([2.5])]
    )
    assert flattened.tolist() == pytest.approx([1.0, 2.5])


def test_summarize_trial_cv2_and_bootstrap_ci() -> None:
    """Trial summary computes counts, median/mean CV2, and bootstrap CI."""
    trials = (
        np.array([0.000, 0.002, 0.005], dtype=np.float64),
        np.array([0.000, 0.004, 0.005], dtype=np.float64),
        np.array([0.010], dtype=np.float64),
        np.array([], dtype=np.float64),
    )
    summary = summarize_trial_cv2(
        trials,
        rng=np.random.default_rng(0),
        n_bootstrap=50,
        ci_level=0.95,
    )
    assert isinstance(summary, TrialCv2Summary)
    assert summary.n_trials == 4
    assert summary.n_spiking_trials == 3
    assert summary.n_cv2_trials == 2
    assert summary.n_cv2_pairs == 2
    assert summary.median_cv2 == pytest.approx(0.8)
    assert summary.mean_cv2 == pytest.approx(0.8)
    assert 0.4 <= summary.ci_low <= summary.ci_high <= 1.2

    with pytest.raises(ValueError, match="n_bootstrap"):
        bootstrap_median_ci_by_trial(
            trials, rng=np.random.default_rng(0), n_bootstrap=0, ci_level=0.95
        )
    with pytest.raises(ValueError, match="ci_level"):
        bootstrap_median_ci_by_trial(
            trials, rng=np.random.default_rng(0), n_bootstrap=10, ci_level=1.0
        )


def test_first_two_and_triple_spike_isi_helpers() -> None:
    """Compute first-two ISI, triple-spike pairs, and absolute/relative contrasts."""
    assert np.isnan(first_two_spike_isi(np.array([0.01], dtype=np.float64)))
    assert first_two_spike_isi(np.array([0.01, 0.014], dtype=np.float64)) == pytest.approx(0.004)

    p_short = first_three_spike_isi_pair_ms(np.array([0.01, 0.02], dtype=np.float64))
    assert np.isnan(p_short[0]) and np.isnan(p_short[1])
    assert first_three_spike_isi_pair_ms(
        np.array([0.019, 0.010, 0.014], dtype=np.float64)
    ) == pytest.approx((4.0, 5.0))

    trials = (
        np.array([0.010, 0.014, 0.020], dtype=np.float64),
        np.array([0.010, 0.020], dtype=np.float64),
    )
    pairs = triple_spike_isi_pairs_ms(trials)
    assert pairs is not None
    assert pairs[0].tolist() == pytest.approx([4.0])
    assert pairs[1].tolist() == pytest.approx([6.0])
    assert triple_spike_isi_pairs_ms((np.array([0.01], dtype=np.float64),)) is None

    diff_abs = collect_triple_spike_isi_diff_ms(trials, mode="absolute")
    diff_rel = collect_triple_spike_isi_diff_ms(trials, mode="relative_diff")
    assert diff_abs is not None and diff_abs.tolist() == pytest.approx([2.0])
    assert diff_rel is not None and diff_rel.tolist() == pytest.approx([0.4])

    with pytest.raises(ValueError, match="unsupported"):
        triple_spike_isi_contrast_from_pair_ms(
            2.0, 4.0, mode=cast(Any, "bad")
        )
