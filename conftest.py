"""Repository-level pytest configuration.

The DSH sandbox exports ``OMP_NUM_THREADS=0``, which libgomp rejects with
``libgomp: Invalid value for environment variable OMP_NUM_THREADS`` on every torch
import.  Normalising it here (before any test imports torch) keeps test output clean
without adding a pytest plugin dependency.
"""

from __future__ import annotations

import os

_value = os.environ.get("OMP_NUM_THREADS", "").strip()
if not _value.isdigit() or int(_value) < 1:
    os.environ["OMP_NUM_THREADS"] = "8"
