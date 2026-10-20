# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The --debug alias check on strided body views: disjoint columns of one allocation do not alias; genuine overlaps,
row-shifted views and irregular slices are caught (the last conservatively); empty views never alias."""

import warp as wp
from i4h_physics.debug import overlap


def block(E=3, W=10):
    return wp.zeros((E * W,), dtype=wp.vec3, device="cuda:0").reshape((E, W))


def test_disjoint_columns_of_one_allocation_do_not_alias():
    a = block()
    assert not overlap(a[:, 0:4], a[:, 4:10])
    assert not overlap(a[:, 6:10], a[:, 0:3])


def test_overlapping_part_views_alias():
    a = block()
    assert overlap(a[:, 0:5], a[:, 4:8])
    assert overlap(a[:, 2:3], a[:, 0:10])


def test_views_starting_on_different_rows():
    a = block()
    assert not overlap(a[0:1, 0:4], a[1:3, 0:4])  # different rows of the same columns
    assert overlap(a[0:2, 0:4], a[1:3, 2:6])  # row 1 shared, columns 2..4 shared
    assert not overlap(a[0:2, 0:4], a[1:3, 5:8])


def test_contiguous_array_against_a_view():
    a = block()
    flat = a.reshape((30,))
    assert overlap(flat, a[:, 3:4])
    assert not overlap(flat[0:3], a[1:3, 0:10])  # the first three elements lie in row 0 only


def test_empty_views_never_alias():
    a = block()
    assert not overlap(a[:, 0:0], a[:, 0:10])


def test_irregular_slice_is_conservative():
    a = block()
    every_other = a[:, 0:10:2]  # element stride 2: not a regular row view
    assert overlap(every_other, a[:, 1:2])  # truly disjoint, but reported (conservative)
