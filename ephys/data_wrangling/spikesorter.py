"""Load multi-unit spike tables exported from Swindale SpikeSorter CSVs."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from ephys.processing.spike_align import spike_times_ticks_from_seconds


def _parse_spikesorter_data_row(row: list[str]) -> tuple[float, int, int]:
    """Parse one three-column ``(time_s, unit_id, max_channel)`` CSV row."""
    if len(row) < 3:
        msg = f"Expected at least 3 columns in SpikeSorter row, got {row!r}"
        raise ValueError(msg)
    time_s = float(row[0].strip())
    unit_id = int(row[1].strip())
    max_channel = int(row[2].strip())
    if not math.isfinite(time_s):
        msg = f"Non-finite spike time in SpikeSorter row: {row!r}"
        raise ValueError(msg)
    return time_s, unit_id, max_channel


def read_spikesorter_csv(
    csv_path: str | Path,
    sampling_frequency: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a three-column SpikeSorter CSV export into DAQ tick arrays.

    Sorted spike times from SpikeSorter (Swindale lab) are exported as three
    comma-separated columns: spike time in seconds (typically 5 decimal
    places), integer unit ID, and max-SNR channel ID. Files may begin directly
    with numeric spike rows or include leading text header lines.

    Parameters
    ----------
    csv_path
        Path to the SpikeSorter CSV file.
    sampling_frequency
        Acquisition sampling rate in Hz used to convert seconds to DAQ ticks.

    Returns
    -------
    spike_times : np.ndarray
        1D ``int64`` spike times in DAQ ticks, rounded via
        :func:`~ephys.processing.spike_align.spike_times_ticks_from_seconds`.
    spike_labels : np.ndarray
        1D ``int64`` unit IDs aligned with ``spike_times``.
    max_channel : np.ndarray
        1D ``int64`` max-SNR channel IDs aligned with ``spike_times``.

    Raises
    ------
    ValueError
        If ``sampling_frequency`` is non-positive/non-finite or a data row
        after the header cannot be parsed.

    Notes
    -----
    Spike times in seconds are rounded to the nearest integer tick rather than
    truncated, avoiding the ``1``-tick downward shift caused by fixed-decimal
    formatting (e.g. ``0.00193 * 30000.0 == 57.9 -> 58``) and binary float
    representation (e.g. ``0.0021 * 30000.0 == 62.99999999999999 -> 63``).
    """
    fs = float(sampling_frequency)
    if isinstance(sampling_frequency, bool) or not math.isfinite(fs) or fs <= 0.0:
        msg = (
            "sampling_frequency must be finite and > 0; "
            f"got {sampling_frequency!r}"
        )
        raise ValueError(msg)

    times_s: list[float] = []
    labels: list[int] = []
    channels: list[int] = []
    in_leading_header = True

    with Path(csv_path).open(newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for line_no, row in enumerate(reader, start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if in_leading_header:
                try:
                    t_s, unit_id, ch = _parse_spikesorter_data_row(row)
                except ValueError:
                    continue
                in_leading_header = False
            else:
                try:
                    t_s, unit_id, ch = _parse_spikesorter_data_row(row)
                except ValueError as exc:
                    msg = f"Malformed SpikeSorter CSV row at line {line_no}: {row!r}"
                    raise ValueError(msg) from exc
            times_s.append(t_s)
            labels.append(unit_id)
            channels.append(ch)

    if not times_s:
        empty = np.array([], dtype=np.int64)
        return empty, empty.copy(), empty.copy()

    spike_times = spike_times_ticks_from_seconds(
        np.asarray(times_s, dtype=np.float64),
        fs,
    )
    spike_labels = np.asarray(labels, dtype=np.int64)
    max_channel = np.asarray(channels, dtype=np.int64)
    return spike_times, spike_labels, max_channel


def get_spikes(
    csv_path: str | Path,
    sampling_frequency: float,
) -> list[np.ndarray]:
    """Load per-unit DAQ tick arrays from a SpikeSorter CSV file.

    Parameters
    ----------
    csv_path
        Path to the SpikeSorter CSV file.
    sampling_frequency
        Acquisition sampling rate in Hz.

    Returns
    -------
    list[np.ndarray]
        One ``int64`` tick array per unique sorted unit (ordered by ascending
        unit ID).
    """
    spike_times, spike_labels, _ = read_spikesorter_csv(
        csv_path,
        sampling_frequency,
    )
    neuron_ids = np.unique(spike_labels)
    return [spike_times[spike_labels == neuron_id] for neuron_id in neuron_ids]


def read_best_channel(
    csv_path: str | Path,
    sampling_frequency: float,
) -> list[int]:
    """Return the modal max-SNR channel for each unique unit in a CSV file.

    Parameters
    ----------
    csv_path
        Path to the SpikeSorter CSV file.
    sampling_frequency
        Acquisition sampling rate in Hz.

    Returns
    -------
    list[int]
        Modal channel ID for each unique unit (ordered by ascending unit ID,
        breaking ties toward the smaller channel ID).
    """
    _, spike_labels, max_channel = read_spikesorter_csv(
        csv_path,
        sampling_frequency,
    )
    neuron_ids = np.unique(spike_labels)
    best_channels: list[int] = []
    for neuron_id in neuron_ids:
        uniques, counts = np.unique(
            max_channel[spike_labels == neuron_id],
            return_counts=True,
        )
        best_channels.append(int(uniques[np.argmax(counts)]))
    return best_channels
