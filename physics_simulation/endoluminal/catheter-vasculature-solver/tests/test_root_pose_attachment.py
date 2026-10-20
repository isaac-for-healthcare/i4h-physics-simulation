# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for holding a rod's proximal end on a mount-supplied pose.

``set_root_pose_gpu`` exists because a rate command cannot express a mount. The
tests below are therefore mostly about the difference between the two: that a
pose lands where it was asked to land rather than near it, that the kinematic
root is given a velocity consistent with the move, and that a pose and a rate
issued in the same step do not both happen.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rod(cpu_rod_config):
    from catheter_vasculature_solver import XPBDRodSolver

    return XPBDRodSolver(cpu_rod_config, floor_z=None)


def root_position(rod) -> np.ndarray:
    return rod.positions.detach().cpu().numpy()[0].astype(np.float64)


def root_velocity(rod) -> np.ndarray:
    return rod.velocities.detach().cpu().numpy()[0].astype(np.float64)


def root_orientation(rod) -> np.ndarray:
    return rod.orientations.detach().cpu().numpy()[0].astype(np.float64)


def test_pose_command_places_the_root_exactly(rod):
    """A pose is absolute: the root arrives at it, not merely toward it."""
    target = np.array([[0.021, -0.013, 0.004]], dtype=np.float32)

    rod.set_root_pose_gpu(target)
    rod.step(1.0 / 60.0)

    np.testing.assert_allclose(root_position(rod), target[0], atol=1e-6)


def test_pose_command_does_not_drift_when_repeated(rod):
    """Holding the same pose holds the root still.

    The rail path integrates a rate, so restating a command keeps moving the
    root. Restating a *pose* must not, or a stalled mount would still feed wire.
    """
    target = np.array([[0.010, 0.0, 0.0]], dtype=np.float32)

    for _ in range(5):
        rod.set_root_pose_gpu(target)
        rod.step(1.0 / 60.0)

    np.testing.assert_allclose(root_position(rod), target[0], atol=1e-6)


def test_the_root_teleports_rather_than_travelling(rod):
    """The root reaches its pose carrying no velocity.

    ``_xr_predict_pos`` zeroes the velocity of every zero-inverse-mass particle
    at the top of each substep, so a mount-driven root cannot hand the rod
    momentum -- the motion reaches the wire through the first stretch
    constraint alone. Pinned here because it is the reason
    :meth:`set_root_pose_gpu` takes no timestep.
    """
    start = root_position(rod)
    step = np.array([0.003, 0.0, 0.0], dtype=np.float32)
    target = (start + step).astype(np.float32).reshape(1, 3)

    rod.set_root_pose_gpu(target)
    rod.step(1.0 / 120.0)

    np.testing.assert_allclose(root_position(rod), target[0], atol=1e-6)
    np.testing.assert_allclose(root_velocity(rod), np.zeros(3), atol=1e-9)


def test_orientation_is_optional(rod):
    """Omitting the quaternion places the root without dictating its twist."""
    before = root_orientation(rod)

    rod.set_root_pose_gpu(np.array([[0.005, 0.0, 0.0]], dtype=np.float32))
    rod.step(1.0 / 60.0)

    np.testing.assert_allclose(root_orientation(rod), before, atol=1e-6)


def test_orientation_is_adopted_when_supplied(rod):
    """A supplied quaternion becomes the root's orientation."""
    half = np.sqrt(0.5)
    quat = np.array([[0.0, 0.0, half, half]], dtype=np.float32)

    rod.set_root_pose_gpu(np.array([[0.005, 0.0, 0.0]], dtype=np.float32), orientations=quat)
    rod.step(1.0 / 60.0)

    np.testing.assert_allclose(root_orientation(rod), quat[0], atol=1e-6)


def test_a_pose_supersedes_a_rate_issued_in_the_same_step(rod):
    """Only the last command latched in a step survives.

    Both commands claim the root, so letting both run would have the mount and
    the rail fighting over it. The documented rule is last-writer-wins.
    """
    target = np.array([[0.008, 0.0, 0.0]], dtype=np.float32)

    rod.apply_proximal_control_gpu(np.array([0.5], dtype=np.float32), np.zeros(1, np.float32), 1.0 / 60.0)
    rod.set_root_pose_gpu(target)
    rod.step(1.0 / 60.0)

    np.testing.assert_allclose(root_position(rod), target[0], atol=1e-6)


def test_wrong_shape_is_rejected_at_the_call(rod):
    """Shape errors surface where the caller can see them, not at the next step."""
    with pytest.raises(ValueError, match="positions must hold 1 entries"):
        rod.set_root_pose_gpu(np.zeros((3, 3), dtype=np.float32))
