# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Migration evidence must reject silent product loss and numerical format drift."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('record_baseline', Path(__file__).parents[2] / 'scripts/record_baseline.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def compare(ref, cur):
    return baseline.compare(ref, cur, 0, 0, 3)


def test_strict_bytes_and_format():
    zero = np.array([0.], dtype=np.float32)
    assert not compare({'x': zero}, {'x': zero.copy()})
    assert compare({'x': zero}, {'x': -zero})  # signed zero differs in bytes
    assert compare({'x': zero}, {'x': zero.astype(np.float64)})
    assert compare({'x': zero}, {'x': zero.reshape(1, 1)})
    assert compare({'x': zero}, {})
    assert compare({}, {'x': zero})
    for value in (float('nan'), float('inf')):
        bad = np.array([value], dtype=np.float32)
        assert compare({'x': bad}, {'x': bad})


def test_only_frozen_product_noise_is_permitted():
    ref = {'x': np.array([0.], dtype=np.float32), 'noise/x': np.array(.001)}
    assert not compare(ref, {'x': np.array([.002], dtype=np.float32)})
    assert compare(ref, {'x': np.array([.004], dtype=np.float32)})
    assert compare({'x': ref['x']}, {'x': np.array([1e-9], dtype=np.float32)})
    with pytest.raises(ValueError):
        baseline.compare(ref, {'x': ref['x']}, 1e-5, 0, 3)
    with pytest.raises(ValueError):
        baseline.compare(ref, {'x': ref['x']}, 0, 0, 4)


def test_slot_permutation_preserves_records_and_ignores_dead_payload():
    fields = {'particle': np.array([[[7, 3, 99]]], np.int32),
              'rest': np.array([[[[.7, 0, 0], [.3, 0, 0], [123, 0, 0]]]], np.float32),
              'active': np.array([[[1, 1, 0]]], np.int32)}
    expected = baseline.canonical_slots(fields)
    permuted = {k: v[:, :, [2, 0, 1]].copy() for k, v in fields.items()}
    permuted['rest'][0, 0, 0] = -999
    assert not compare(expected, baseline.canonical_slots(permuted))
    permuted['rest'][0, 0, 1, 0] += .1
    assert compare(expected, baseline.canonical_slots(permuted))
    permuted = {k: v.copy() for k, v in fields.items()}
    permuted['active'][0, 0, 1] = 0
    assert compare(expected, baseline.canonical_slots(permuted))


def test_empty_slots_and_consumed_clip_nodes():
    empty = {'active': np.empty((2, 1, 0), np.int32), 'particle': np.empty((2, 1, 0), np.int32)}
    assert baseline.canonical_slots(empty)['particle'].shape == (2, 1, 0)
    clips = {'node': np.array([[4, 2]], np.int32), 'active': np.array([[0, 1]], np.int32)}
    assert 4 in baseline.canonical_slots(clips, discard_inactive=False)['node']
