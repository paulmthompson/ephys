"""Top-level namespace aliases for electrophysiology utilities.

This package exposes legacy top-level packages under the ``ephys`` namespace:
``ephys.processing``, ``ephys.plotting``, ``ephys.data_wrangling``, and
``ephys.probes``.
"""

from __future__ import annotations

from importlib import import_module
import sys

_ALIASES = ("data_wrangling", "plotting", "processing", "probes")

for _name in _ALIASES:
    _mod = import_module(_name)
    sys.modules[f"{__name__}.{_name}"] = _mod

del _mod
del _name
