# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Debug-mode launch checks.

``alias_checks()`` rebinds ``wp.launch`` so every launch asserts that no array in
``inputs`` shares device memory with an array in ``outputs``. It checks the declared
arguments by byte span, not kernel bodies: a kernel that reads neighbours from an
``outputs`` array is not caught. Host-only, so it runs inside graph capture too.

Contiguous arrays compare by byte span. A regular 2D view (unit element stride,
rows at least a row wide, e.g. a body's ``(E, count)`` view of a world-major block)
is a set of equally spaced rows, and two arrays laid out as rows with the same row
stride overlap only if some pair of rows does: disjoint bodies of one block never
alias. Any other shape falls back to the conservative byte span.

The rebinding is process-global and not thread-safe: while a context is open, every thread's ``wp.launch`` lookups
(Isaac Lab sensors, the Newton viewer) are checked too, and a launch bound earlier (``from warp import launch``) is not.
Contexts nest: only the outermost installs and restores the wrapper, so a launch is checked once and an exception in
an inner context leaves it in place until the outer one exits.
"""

from __future__ import annotations

import contextlib

import warp as wp


def span(a: wp.array) -> tuple[int, int]:
    end = sum((n - 1) * s for n, s in zip(a.shape, a.strides)) + wp.types.type_size_in_bytes(a.dtype)
    return a.ptr, a.ptr + end


def rows(a: wp.array):
    """``(first byte, row bytes, row stride, row count)`` of a contiguous array (one row) or a regular 2D view;
    None for any other layout."""
    size = wp.types.type_size_in_bytes(a.dtype)
    if a.is_contiguous:
        return a.ptr, a.size * size, 0, 1
    if a.ndim == 2 and a.strides[1] == size and a.strides[0] >= a.shape[1] * size:
        return a.ptr, a.shape[1] * size, a.strides[0], a.shape[0]
    return None


def overlap(a: wp.array, b: wp.array) -> bool:
    if a.size == 0 or b.size == 0 or a.device != b.device:
        return False
    lo_a, hi_a = span(a)
    lo_b, hi_b = span(b)
    if not (lo_a < hi_b and lo_b < hi_a):
        return False
    ra, rb = rows(a), rows(b)
    if ra is None or rb is None:
        return True  # conservative
    (pa, la, sa, na), (pb, lb, sb, nb) = ra, rb
    if na == 1 and nb == 1:
        return True  # contiguous spans already intersect
    stride = sa if na > 1 else sb
    if na > 1 and nb > 1 and sa != sb:
        return True  # different row strides: conservative
    # rows i of a and j of b intersect iff -la < k * stride - d < lb with k = i - j and d = pb - pa
    d = pb - pa
    k_lo = max(-(nb - 1), (d - la) // stride + 1)
    k_hi = min(na - 1, -((-(d + lb)) // stride) - 1)
    return k_lo <= k_hi


def assert_no_alias(kernel, inputs, outputs) -> None:
    for a in inputs:
        if not isinstance(a, wp.array):
            continue
        for b in outputs:
            if isinstance(b, wp.array) and overlap(a, b):
                raise RuntimeError(f"{kernel.key}: an input array aliases an output array")


_depth = 0  # open alias_checks() contexts
_real_launch = None  # wp.launch as the outermost context found it


@contextlib.contextmanager
def alias_checks():
    global _depth, _real_launch
    if _depth == 0:
        _real_launch = real_launch = wp.launch

        def checked_launch(kernel, dim, inputs=None, outputs=None, **kwargs):
            inputs = list(inputs or [])
            outputs = list(outputs or [])
            assert_no_alias(kernel, inputs, outputs)
            return real_launch(kernel, dim, inputs=inputs, outputs=outputs, **kwargs)

        wp.launch = checked_launch
    _depth += 1
    try:
        yield
    finally:
        _depth -= 1
        if _depth == 0:
            wp.launch = _real_launch
            _real_launch = None
