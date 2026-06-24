"""Tests for temporal basis helpers."""

from __future__ import annotations

import numpy as np
import pytest

from ephys.processing.basis import (
    acausal_basis_columns,
    causal_basis_columns,
    lead_time_raised_cosine_basis,
    log_raised_cosine_basis,
    raised_cosine_basis,
    signed_event_basis,
)


def test_causal_basis_matches_acausal_with_zero_leads() -> None:
    """Causal and acausal convolutions agree when n_leads is zero."""
    signal = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
    basis = raised_cosine_basis(3, 2)
    causal = causal_basis_columns(signal, basis)
    acausal = acausal_basis_columns(signal, basis, n_leads=0)
    for left, right in zip(causal, acausal, strict=True):
        np.testing.assert_allclose(left, right)


def test_lead_basis_knots_are_evenly_spaced_in_time() -> None:
    """Lead knots sit at -lead_s and -lead_s/2 for two lead bases over 2 ms."""
    lead_s = 0.002
    dt_s = 0.0001
    n_leads = int(round(lead_s / dt_s))
    basis = lead_time_raised_cosine_basis(n_leads, 2, lead_s)
    row_times_ms = (np.arange(n_leads, dtype=float) - n_leads) * dt_s * 1000.0
    for col in range(2):
        idx = int(np.argmax(basis[:, col]))
        expected_ms = -lead_s * 1000.0 * (2 - col) / 2
        assert row_times_ms[idx] == pytest.approx(expected_ms, abs=0.15)


def test_signed_basis_matches_causal_log_when_no_lead() -> None:
    """Causal log bases are unchanged when event_lead_s is zero."""
    n_lags = 40
    n_basis = 8
    np.testing.assert_allclose(
        signed_event_basis(
            0,
            n_lags,
            0,
            n_basis,
            lag_kind="log_raised_cosine",
        ),
        log_raised_cosine_basis(n_lags, n_basis),
    )


def test_signed_basis_knots_overlap_at_seam() -> None:
    """Lead and lag knots both contribute near contact time."""
    lead_s = 0.002
    dt_s = 0.0001
    n_leads = int(round(lead_s / dt_s))
    basis = signed_event_basis(
        n_leads,
        50,
        n_lead_basis=2,
        n_lag_basis=8,
        lag_kind="log_raised_cosine",
        lead_s=lead_s,
    )
    onset_row = n_leads
    assert np.any(basis[onset_row - 1, :2] > 0.0)
    assert np.any(basis[onset_row - 1, 2:] > 0.0)
    assert np.any(basis[onset_row, 2:] > 0.0)


def test_acausal_spreads_impulse_before_event() -> None:
    """An impulse produces nonzero mass in lead bins when n_leads > 0."""
    lead_s = 0.002
    n_leads = 2
    n_lags = 3
    basis = signed_event_basis(
        n_leads,
        n_lags,
        n_lead_basis=2,
        n_lag_basis=1,
        lag_kind="raised_cosine",
        lead_s=lead_s,
    )
    signal = np.zeros(6, dtype=float)
    signal[3] = 1.0
    out_lead_far = acausal_basis_columns(signal, basis, n_leads=n_leads)[0]
    out_lead_near = acausal_basis_columns(signal, basis, n_leads=n_leads)[1]
    out_lag = acausal_basis_columns(signal, basis, n_leads=n_leads)[2]
    assert out_lead_far[1] > 0.0
    assert out_lead_near[2] > 0.0
    assert out_lag[3] > 0.0


def test_acausal_zero_leads_has_no_pre_event_mass() -> None:
    """Causal filtering leaves bins before the impulse at zero."""
    signal = np.zeros(5, dtype=float)
    signal[2] = 1.0
    basis = raised_cosine_basis(3, 1)
    out = acausal_basis_columns(signal, basis, n_leads=0)[0]
    assert out[0] == 0.0
    assert out[1] == 0.0
    assert out[2] > 0.0


def test_reconstructed_filter_is_continuous_at_onset() -> None:
    """A single lag-heavy coefficient set can elevate pre-onset rows."""
    lead_s = 0.002
    dt_s = 0.0001
    n_leads = int(round(lead_s / dt_s))
    n_lags = 20
    basis = signed_event_basis(
        n_leads,
        n_lags,
        n_lead_basis=2,
        n_lag_basis=4,
        lag_kind="log_raised_cosine",
        lead_s=lead_s,
    )
    beta = np.zeros(basis.shape[1], dtype=float)
    beta[-1] = 1.0
    filt = beta @ basis.T
    assert filt[n_leads - 1] > 0.0
    assert filt[n_leads] > 0.0
