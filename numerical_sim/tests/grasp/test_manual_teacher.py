# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/liver_grasp"))
from manual_teacher import ManualGeometryTeacher


def test_live_target_tracking_and_stationary_slow_closure():
    initial = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    example = dict(
        capture_root=[0.01, 0, 0],
        capture_particle_positions=[[0, 0, 0]],
        capture_quaternion_xyzw=[0, 0, 0, 1],
        capture_grip=0.6,
    )
    teacher = ManualGeometryTeacher(initial, [[-1] * 3, [1] * 3], example)
    m = dict(geometry=np.zeros((4, 3)), gap=0.0)
    for _ in range(200):
        before = teacher.action.copy()
        teacher.command(m, 0.002, 0.001, 0.3)
        assert np.linalg.norm(teacher.action[:3] - before[:3]) <= 0.00100001
        if teacher.phase == "capture":
            break
    assert teacher.phase == "capture"
    root = teacher.action[:3].copy()
    previous = teacher.action[7]
    m["geometry"][0] = [0.1, 0, 0]
    for _ in range(20):
        teacher.command(m, 0.002, 0.001, 0.3)
        np.testing.assert_array_equal(teacher.action[:3], root)
        assert 0 <= teacher.action[7] - previous <= 0.0100001
        previous = teacher.action[7]
    assert teacher.action[7] <= 0.600001
