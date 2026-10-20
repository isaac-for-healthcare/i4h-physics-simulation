# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Type checks for case and constructor settings: a value of the wrong type raises, naming the setting, instead of
being coerced (``bool("false")`` is True, ``int(2.5)`` is 2). Each returns the value as a plain Python scalar."""

from __future__ import annotations

import math
import numbers

import numpy as np


def check_int(label: str, value, minimum: int | None = None) -> int:
    """An integer (``int`` or a NumPy integer; not a bool, not ``2.0``), at least ``minimum`` when given."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{label} must be an integer{_bound(minimum)}, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be an integer{_bound(minimum)}, got {value!r}")
    return int(value)


def check_bool(label: str, value) -> bool:
    """``true``/``false`` (a ``bool`` or NumPy bool); strings and numbers raise."""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be true or false, got {value!r}")
    return bool(value)


def check_float(label: str, value) -> float:
    """A finite real number (an integer is accepted); bools and strings raise."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    return float(value)


def _bound(minimum):
    return "" if minimum is None else f" at least {minimum}"
