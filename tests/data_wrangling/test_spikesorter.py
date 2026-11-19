"""Tests for Swindale SpikeSorter CSV loading and tick rounding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ephys.data_wrangling.spikesorter import (
    get_spikes,
    read_best_channel,
    read_spikesorter_csv,
)


def test_read_spikesorter_csv_headerless_real_5_decimal_rows(
    tmp_path: Path,
) -> None:
    """Headerless 5-decimal rows preserve rows 1-2 and round x.9 to x+1."""
    # Verbatim first 5 rows from NTNG1-PrV-Piezo/P005/102525_1/spikes.csv.
    csv_file = tmp_path / "spikes.csv"
    csv_file.write_text(
        "     0.00193,   15,   23\n"
        "     0.00300,   11,   18\n"
        "     0.02063,   25,    4\n"
        "     0.02557,    3,    4\n"
        "     0.02653,   18,   26\n",
        encoding="utf-8",
    )

    ticks, units, channels = read_spikesorter_csv(csv_file, 30_000.0)

    assert ticks.dtype == np.int64
    assert list(ticks) == [58, 90, 619, 767, 796]
    assert list(units) == [15, 11, 25, 3, 18]
    assert list(channels) == [23, 18, 4, 4, 26]


def test_read_spikesorter_csv_ieee754_boundary_and_two_line_header(
    tmp_path: Path,
) -> None:
    """Two-line headers are skipped and IEEE-754 k-1e-14 products round to k."""
    # 63 / 30000 == 0.0021, yet 0.0021 * 30000.0 == 62.99999999999999.
    csv_file = tmp_path / "headered.csv"
    csv_file.write_text(
        "SpikeSorter export\n"
        "time_s,unit_id,max_channel\n"
        "0.0019666666666666665,1,5\n"
        "0.0021,2,12\n"
        "0.004066666666666666,2,14\n"
        "0.0050,2,12\n",
        encoding="utf-8",
    )

    ticks, units, channels = read_spikesorter_csv(csv_file, 30_000.0)
    assert list(ticks) == [59, 63, 122, 150]
    assert list(units) == [1, 2, 2, 2]
    assert list(channels) == [5, 12, 14, 12]

    per_unit = get_spikes(csv_file, 30_000.0)
    assert len(per_unit) == 2
    assert list(per_unit[0]) == [59]
    assert list(per_unit[1]) == [63, 122, 150]

    assert read_best_channel(csv_file, 30_000.0) == [5, 12]


def test_read_spikesorter_csv_empty_and_validation_errors(
    tmp_path: Path,
) -> None:
    """Empty files return empty int64 arrays; bad body rows or rates raise."""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("SpikeSorter export\ntime,unit,ch\n", encoding="utf-8")
    ticks, units, channels = read_spikesorter_csv(empty_file, 30_000.0)
    assert ticks.size == 0 and ticks.dtype == np.int64
    assert units.size == 0 and channels.size == 0

    corrupt = tmp_path / "corrupt.csv"
    corrupt.write_text("0.00300, 1, 4\nbad_row, x, y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed SpikeSorter CSV row"):
        read_spikesorter_csv(corrupt, 30_000.0)

    with pytest.raises(ValueError, match="sampling_frequency"):
        read_spikesorter_csv(empty_file, 0.0)
