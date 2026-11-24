# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Hosted episode reset must sample current holder joints before restoring tools."""

import numpy as np
import pytest
import test_coupling
import warp as wp
from i4h_isaaclab.coupling import prepare_mounted_episode
from i4h_physics import PhysicsBinding
from test_coupling import held_catheter, held_probe


@pytest.mark.parametrize("kind", ["instrument", "catheter"])
@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("selected", [0, 1])
def test_hosted_mount_reset_uses_current_joints(kind, captured, selected, monkeypatch):
    # World 1's holder is tilted in world space; the reset stays in component coordinates.
    monkeypatch.setattr(test_coupling, "ENV_XFORMS", [test_coupling.ENV_XFORMS[0],
                        ((1.0, .5, 0.0), (float(np.sin(np.pi / 4)), 0.0, 0.0, float(np.cos(np.pi / 4))))])
    if kind == "instrument":
        _, _, model, state, control, coupled = held_probe("two_way")
    else:
        _, model, state, control, coupled = held_catheter("two_way")
    coupled.mounts.drive(state)
    coupled.restore_episode(state, control)
    articulation_mask = wp.zeros(model.articulation_count, dtype=wp.bool, device=model.device)
    binding = PhysicsBinding.hosted(
        coupled.ours, coupled, lambda: None,
        prepare_episode_reset=lambda mask: prepare_mounted_episode(binding, mask, articulation_mask))
    binding.mark_prepared()
    group = coupled.mounts.groups[0]
    before = group.control.root.numpy().copy()
    particles_before = state.particle_q.numpy().reshape(2, -1, 3).copy()
    peer = 1 - selected
    peer_pose = state.body_q.numpy()[peer].copy()
    peer_mount = group.control.mount_pose.numpy()[peer].copy()
    joints = state.joint_q.numpy()
    joints[:] += 0.04  # both worlds have stale FK; only the selected world may be updated
    state.joint_q.assign(joints)
    group.force.fill_(wp.vec3(3.0))
    group.torque.fill_(wp.vec3(2.0))
    ids = wp.array([selected], dtype=wp.int32, device=model.device)
    if captured:
        # Capture the public episode reset itself, including mask selection.
        with wp.ScopedCapture(device=model.device) as capture:
            binding.reset_physics(ids)
        wp.capture_launch(capture.graph)
    else:
        binding.reset_physics(ids)
    after = group.control.root.numpy()
    np.testing.assert_allclose(after[selected, 0], before[selected, 0] + [0, 0, .04], atol=1e-6)
    np.testing.assert_array_equal(after[peer], before[peer])
    particles = state.particle_q.numpy().reshape(2, -1, 3)
    np.testing.assert_array_equal(particles[peer], particles_before[peer])
    if kind == "catheter":
        np.testing.assert_allclose(particles[selected], particles_before[selected] + [0, 0, .04], atol=1e-6)
    np.testing.assert_array_equal(state.body_q.numpy()[peer], peer_pose)
    np.testing.assert_array_equal(group.control.mount_pose.numpy()[peer], peer_mount)
    assert not group.force.numpy()[selected].any() and not group.torque.numpy()[selected].any()
    np.testing.assert_array_equal(group.force.numpy()[peer], 3.0)
    np.testing.assert_array_equal(group.torque.numpy()[peer], 2.0)

    # Cache-only reset and continuation must not sample the edited holder joints.
    arrays = [state.particle_q, state.particle_qd, group.control.mount_pose,
              group.control.root, group.force, group.torque]
    saved = model.state(), model.control()
    binding.save(*saved)
    expected = [a.numpy().copy() for a in arrays]
    joint_values = state.joint_q.numpy()
    joint_values[:] += .03
    state.joint_q.assign(joint_values)
    coupled.reset(state, world_mask=binding.world_mask, flags=0)
    binding.restore(*saved)
    assert binding.reset_physics([]) is False
    for array, value in zip(arrays, expected):
        np.testing.assert_array_equal(array.numpy(), value)
    # Release the hosted callback's reference cycle here, rather than allowing
    # a later test's garbage collection to free this model's device allocation.
    binding.invalidate()
