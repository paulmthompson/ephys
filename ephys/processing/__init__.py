"""Signal processing utilities for electrophysiology data."""

from ephys.processing.basis import log_raised_cosine_basis, raised_cosine_basis
from ephys.processing.resampling import whittaker_shannon_interpolate

__all__ = [
    "log_raised_cosine_basis",
    "raised_cosine_basis",
    "whittaker_shannon_interpolate",
]
