"""Tests for :mod:`ephys.processing.spike_intervals`."""

from __future__ import annotations

import numpy as np
import pytest

from ephys.processing.spike_intervals import (
    adjacent_isi_cv2,
    collect_adjacent_isi_cv2,
    collect_adjacent_isi_cv2_by_trial,
    inter_spike_intervals,
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
