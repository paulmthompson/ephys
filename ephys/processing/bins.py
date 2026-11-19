"""Uniform 1D histogram and PSTH bin grids with exact endpoints."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _require_real_scalar(value: object, name: str) -> float:
    """Validate and coerce ``value`` to a finite ``float`` (rejecting bool)."""
    if isinstance(value, bool) or not isinstance(value, int | float | np.number):
        msg = f"{name} must be a real number, got {type(value).__name__}"
        raise TypeError(msg)
    out = float(value)
    if not math.isfinite(out):
        msg = f"{name} must be finite, got {value!r}"
        raise ValueError(msg)
    return out


@dataclass(frozen=True, slots=True)
class UniformBins:
    """Equal-width 1D bins spanning ``[start, stop]`` without float drift.

    Parameters
    ----------
    start
        Left boundary of the first bin.
    stop
        Right boundary of the final bin (must exceed ``start``).
    n_bins
        Number of equal-width bins across ``[start, stop]`` (at least ``1``).

    Notes
    -----
    Boundaries are generated via :func:`numpy.linspace` rather than
    :func:`numpy.arange` so ``edges[0] == start`` and ``edges[-1] == stop``
    hold without trailing ``1e-18`` sliver bins or boundary-spike drops.
    """

    start: float
    stop: float
    n_bins: int

    def __post_init__(self) -> None:
        """Validate finite bounds, ``stop > start``, and positive bin count."""
        lo = _require_real_scalar(self.start, "start")
        hi = _require_real_scalar(self.stop, "stop")
        if hi <= lo:
            msg = f"stop must exceed start; got start={lo!r}, stop={hi!r}"
            raise ValueError(msg)
        if isinstance(self.n_bins, bool) or not isinstance(
            self.n_bins, int | np.integer
        ):
            msg = f"n_bins must be an int, got {type(self.n_bins).__name__}"
            raise TypeError(msg)
        n = int(self.n_bins)
        if n < 1:
            msg = f"n_bins must be >= 1, got {n}"
            raise ValueError(msg)
        object.__setattr__(self, "start", lo)
        object.__setattr__(self, "stop", hi)
        object.__setattr__(self, "n_bins", n)

    @classmethod
    def from_count(
        cls,
        start: float,
        stop: float,
        n_bins: int,
    ) -> UniformBins:
        """Construct ``n_bins`` equal-width bins spanning ``[start, stop]``.

        Parameters
        ----------
        start, stop
            Interval bounds with ``stop > start``.
        n_bins
            Positive number of bins.

        Returns
        -------
        UniformBins
            Uniform bin grid.
        """
        return cls(start=start, stop=stop, n_bins=n_bins)

    @classmethod
    def from_width(
        cls,
        start: float,
        stop: float,
        width: float,
        *,
        rel_tol: float = 1e-9,
    ) -> UniformBins:
        """Construct equal-width bins of size ``width`` over ``[start, stop]``.

        Parameters
        ----------
        start, stop
            Interval bounds with ``stop > start``.
        width
            Target bin width (must evenly divide ``stop - start`` within
            ``rel_tol * width``).
        rel_tol
            Relative floating-point tolerance when checking divisibility.

        Returns
        -------
        UniformBins
            Uniform bin grid whose ``width`` matches ``width`` up to float
            rounding.

        Raises
        ------
        TypeError
            If ``start``, ``stop``, ``width``, or ``rel_tol`` is not numeric.
        ValueError
            If ``width <= 0``, ``stop <= start``, or ``stop - start`` is not an
            integer multiple of ``width``.
        """
        lo = _require_real_scalar(start, "start")
        hi = _require_real_scalar(stop, "stop")
        if hi <= lo:
            msg = f"stop must exceed start; got start={lo!r}, stop={hi!r}"
            raise ValueError(msg)
        w = _require_real_scalar(width, "width")
        if w <= 0.0:
            msg = f"width must be positive, got {width!r}"
            raise ValueError(msg)
        tol_scale = _require_real_scalar(rel_tol, "rel_tol")
        if tol_scale < 0.0:
            msg = f"rel_tol must be non-negative, got {rel_tol!r}"
            raise ValueError(msg)

        span = hi - lo
        n_exact = round(span / w)
        if n_exact < 1 or abs(span - n_exact * w) > tol_scale * w:
            msg = (
                f"Span stop - start = {span:g} ([{lo:g}, {hi:g}]) is not an "
                f"integer multiple of width={w:g}"
            )
            raise ValueError(msg)
        return cls(start=lo, stop=hi, n_bins=int(n_exact))

    @classmethod
    def from_edges(
        cls,
        edges: Sequence[float] | np.ndarray,
        *,
        rel_tol: float = 1e-6,
    ) -> UniformBins:
        """Validate and convert a 1D uniform edge vector into :class:`UniformBins`.

        Parameters
        ----------
        edges
            1D sequence of at least two strictly increasing, uniformly spaced
            bin boundaries.
        rel_tol
            Maximum relative deviation of individual bin widths from the mean
            width.

        Returns
        -------
        UniformBins
            Uniform grid matching ``edges[0]``, ``edges[-1]``, and
            ``len(edges) - 1``.

        Raises
        ------
        ValueError
            If ``edges`` is not 1D, has fewer than 2 finite elements, is not
            strictly increasing, or has non-uniform spacing.
        """
        arr = np.asarray(edges, dtype=np.float64)
        if arr.ndim != 1 or arr.size < 2 or not np.all(np.isfinite(arr)):
            msg = "edges must be a 1D finite array with at least 2 boundaries"
            raise ValueError(msg)
        tol_scale = _require_real_scalar(rel_tol, "rel_tol")
        if tol_scale < 0.0:
            msg = f"rel_tol must be non-negative, got {rel_tol!r}"
            raise ValueError(msg)
        diffs = np.diff(arr)
        if np.any(diffs <= 0.0):
            msg = "edges must be strictly increasing"
            raise ValueError(msg)
        mean_w = float(arr[-1] - arr[0]) / float(arr.size - 1)
        if np.max(np.abs(diffs - mean_w)) > tol_scale * mean_w:
            msg = "edges are not uniformly spaced within rel_tol"
            raise ValueError(msg)
        return cls(
            start=float(arr[0]),
            stop=float(arr[-1]),
            n_bins=int(arr.size - 1),
        )

    @property
    def span(self) -> float:
        """Total interval span ``stop - start``."""
        return self.stop - self.start

    @property
    def width(self) -> float:
        """Width of each bin ``(stop - start) / n_bins``."""
        return (self.stop - self.start) / float(self.n_bins)

    @property
    def limits(self) -> tuple[float, float]:
        """Closed interval endpoints ``(start, stop)`` for axis limits."""
        return (self.start, self.stop)

    @property
    def edges(self) -> np.ndarray:
        """Monotonic boundary array of shape ``(n_bins + 1,)``."""
        return np.linspace(
            self.start,
            self.stop,
            self.n_bins + 1,
            dtype=np.float64,
        )

    @property
    def left_edges(self) -> np.ndarray:
        """Left bin boundaries of shape ``(n_bins,)``."""
        return self.edges[:-1]

    @property
    def right_edges(self) -> np.ndarray:
        """Right bin boundaries of shape ``(n_bins,)``."""
        return self.edges[1:]

    @property
    def centers(self) -> np.ndarray:
        """Midpoint bin centers of shape ``(n_bins,)``."""
        e = self.edges
        return 0.5 * (e[:-1] + e[1:])

    def scaled(self, factor: float) -> UniformBins:
        """Scale ``start`` and ``stop`` by a positive factor (e.g. s -> ms).

        Parameters
        ----------
        factor
            Strictly positive finite scale multiplier.

        Returns
        -------
        UniformBins
            Scaled grid with identical ``n_bins`` and ``width * factor``.
        """
        f = _require_real_scalar(factor, "factor")
        if f <= 0.0:
            msg = f"factor must be positive, got {factor!r}"
            raise ValueError(msg)
        return UniformBins(
            start=self.start * f,
            stop=self.stop * f,
            n_bins=self.n_bins,
        )

    def histogram_counts(
        self,
        values: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Count values falling into each bin via :func:`numpy.histogram`.

        Parameters
        ----------
        values
            1D sample values (e.g. relative spike times). Non-finite entries
            are ignored.

        Returns
        -------
        np.ndarray
            ``int64`` counts of shape ``(n_bins,)``. As in
            :func:`numpy.histogram`, all bins except the last are half-open
            ``[left, right)`` and the final bin is closed ``[left, stop]``.
        """
        arr = np.asarray(values, dtype=np.float64).ravel()
        finite = arr[np.isfinite(arr)]
        counts, _ = np.histogram(finite, bins=self.edges)
        return counts.astype(np.int64, copy=False)
