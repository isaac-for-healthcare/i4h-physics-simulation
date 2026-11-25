# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Holding the catheter's root on its drive body inside every coupled substep.

Two-way coupling is only closed when the body feels the rod and the rod is held
where the body is. These tests pin the second half: the root lands on the mount
in the same substep the rigid solve moved it, keeps its twist relative to the
body, rebinds per env on reset, and refuses to fight a task's own root command.
The closed-loop tests run the real coupler and follower kernels against a
spring-held rod to pin the stability limit of the staggered scheme.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

wp = pytest.importorskip("warp", reason="needs warp installed")

from catheter_vasculature_solver import RodConfig, RodParticleBridge, RodParticleRange, XPBDRodSolver  # noqa: E402
from catheter_vasculature_solver.isaaclab_integration.proxy_coupling import (  # noqa: E402
    DriveReactionCoupler,
    DriveRootFollower,
)

DEVICE = "cpu"
DT = 1.0 / 240.0


def make_rod(num_envs: int = 1) -> XPBDRodSolver:
    cfg = RodConfig()
    cfg.device = DEVICE
    cfg.geometry.num_segments = 4
    cfg.geometry.segment_length = 0.02
    cfg.solver.num_substeps = 1
    cfg.solver.gravity = (0.0, 0.0, 0.0)
    return XPBDRodSolver(cfg, num_envs=num_envs, floor_z=None)


def body_states(poses):
    """Return a body_q array from ``(position, xyzw)`` pairs."""
    return wp.array(
        [wp.transform(wp.vec3(*p), wp.quat(*q)) for p, q in poses], dtype=wp.transform, device=DEVICE,
    )


def quat_z(angle: float):
    return (0.0, 0.0, float(np.sin(angle / 2.0)), float(np.cos(angle / 2.0)))


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def same_rotation(a, b) -> bool:
    return abs(abs(float(np.dot(a, b))) - 1.0) < 1e-5


IDENTITY = (0.0, 0.0, 0.0, 1.0)


def test_root_lands_on_the_mount_in_the_same_substep() -> None:
    rod = make_rod()
    follower = DriveRootFollower(
        num_envs=1, root_stride=rod.num_points, drive_bodies=[1], mount_local=[(0.1, 0.0, 0.0)], device=DEVICE,
    )
    begin = SimpleNamespace(body_q=body_states([((0, 0, 0), IDENTITY), ((0, 0, 0), IDENTITY)]))
    end = SimpleNamespace(body_q=body_states([((0, 0, 0), IDENTITY), ((0.3, -0.2, 0.05), quat_z(np.pi / 2))]))

    follower.sync(begin, end, rod)
    rod.step(DT)

    np.testing.assert_allclose(rod.position_array.numpy()[0], [0.3, -0.1, 0.05], atol=1e-6)


def test_root_keeps_its_twist_relative_to_the_body() -> None:
    """The bind captures the root against the body, so a turn carries the wire's frame with it."""
    rod = make_rod()
    initial = rod.orientation_array.numpy()[0].copy()
    follower = DriveRootFollower(num_envs=1, root_stride=rod.num_points, drive_bodies=[0], device=DEVICE)
    rest = SimpleNamespace(body_q=body_states([((0, 0, 0), IDENTITY)]))
    turned = SimpleNamespace(body_q=body_states([((0, 0, 0), quat_z(0.4))]))

    follower.sync(rest, rest, rod)
    assert same_rotation(follower.root_orientations.numpy()[0], initial)
    rod.step(DT)

    follower.sync(rest, turned, rod)
    assert same_rotation(follower.root_orientations.numpy()[0], quat_mul(quat_z(0.4), initial))


def test_reset_rebinds_only_the_masked_env() -> None:
    rod = make_rod(num_envs=2)
    initial = rod.orientation_array.numpy()[0].copy()
    follower = DriveRootFollower(num_envs=2, root_stride=rod.num_points, drive_bodies=[0, 1], device=DEVICE)
    rest = SimpleNamespace(body_q=body_states([((0, 0, 0), IDENTITY)] * 2))
    turned = SimpleNamespace(body_q=body_states([((0, 0, 0), quat_z(0.4))] * 2))
    follower.sync(rest, rest, rod)
    rod.step(DT)

    follower.reset(world_mask=wp.array([False, True], dtype=wp.bool, device=DEVICE))
    follower.sync(turned, turned, rod)

    orientations = follower.root_orientations.numpy()
    assert same_rotation(orientations[0], quat_mul(quat_z(0.4), initial))
    assert same_rotation(orientations[1], initial)


def test_refuses_to_override_a_latched_root_command() -> None:
    rod = make_rod()
    follower = DriveRootFollower(num_envs=1, root_stride=rod.num_points, drive_bodies=[0], device=DEVICE)
    state = SimpleNamespace(body_q=body_states([((0, 0, 0), IDENTITY)]))
    rod.apply_proximal_control_gpu([0.5], [0.0], DT)

    with pytest.raises(RuntimeError, match="already latched"):
        follower.sync(state, state, rod)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"drive_bodies": [-1]}, "negative"),
        ({"drive_bodies": [0, 1]}, "num_envs"),
        ({"drive_bodies": [0], "mount_local": [(0.0, 0.0)]}, "3-vector"),
    ],
)
def test_rejects_malformed_inputs(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        DriveRootFollower(num_envs=1, root_stride=5, device=DEVICE, **kwargs)


@pytest.mark.parametrize("mode", ["one_way", "two_way"])
def test_root_sync_without_a_drive_body_fails_at_build(rod_managers, mode) -> None:
    cfg = rod_managers.coupled_cfg(coupling_mode=mode, drive_root_sync=True)

    with pytest.raises(ValueError, match="drive_body_name or drive_body_index"):
        rod_managers.coupled._build_follower(SimpleNamespace(body_count=2, device=DEVICE), cfg)


def test_coupled_step_holds_the_root_on_this_substeps_body_pose(rod_managers, monkeypatch) -> None:
    """The sync sits between the rigid and rod solves, so there is no substep of lag."""
    rod = make_rod(num_envs=2)
    bridge = RodParticleBridge(rod, RodParticleRange(0, rod.num_points * 2, 2))

    def advance(state_in, state_out, control, contacts, dt):
        bridge.read_from(state_in)
        bridge.advance(dt)
        bridge.write_to(state_out)

    states = [SimpleNamespace(
        particle_q=wp.clone(rod.position_array),
        particle_qd=wp.clone(rod.velocity_array),
        body_q=body_states([((0, 0, 0), IDENTITY)] * 2),
    ) for _ in range(2)]
    state_0, state_1 = states

    def rigid_step(state_in, state_out, control, contacts, dt):
        poses = state_in.body_q.numpy()
        poses[:, 0] += 0.002
        poses[1, 1] -= 0.001
        state_out.body_q.assign(wp.array(poses, dtype=wp.transform, device=DEVICE))

    follower = DriveRootFollower(num_envs=2, root_stride=rod.num_points, drive_bodies=[0, 1], device=DEVICE)
    monkeypatch.setattr(rod_managers.coupled, "_rigid_solver", SimpleNamespace(step=rigid_step))
    monkeypatch.setattr(rod_managers.coupled, "_rod_solver", SimpleNamespace(rod=rod, step=advance))
    monkeypatch.setattr(rod_managers.coupled, "_follower", follower)
    monkeypatch.setattr(rod_managers.base, "_collision_pipeline", None, raising=False)
    monkeypatch.setattr(rod_managers.base, "_contacts", None, raising=False)

    for _ in range(5):
        rod_managers.coupled._step_solver(state_0, state_1, None, None, DT)
        roots = state_1.particle_q.numpy()[::rod.num_points]
        np.testing.assert_allclose(roots, state_1.body_q.numpy()[:, :3], atol=1e-6)
        state_0, state_1 = state_1, state_0


class SpringHeldRod:
    """A rod pinned at its tip, reduced to a spring of stiffness ``k`` at the root.

    Stands in for a stiff catheter wedged against the vessel: the reaction on
    the holder grows with how far the root is pushed past ``wall``. It honours
    the parts of the rod solver's interface the follower and coupler use.
    """

    def __init__(self, stiffness: float, wall: float = 0.0) -> None:
        self.k = stiffness
        self.wall = wall
        self.root_x = 0.0
        self.has_pending_control = False
        self.orientation_array = wp.array([wp.quat_identity()], dtype=wp.quat, device=DEVICE)

    def set_root_pose_gpu(self, positions, orientations=None) -> None:
        self.root_x = float(positions.numpy()[0][0])

    def proximal_wrench(self, out=None):
        force = -self.k * (self.root_x - self.wall)
        out.assign(wp.array([[force, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=wp.spatial_vector, device=DEVICE))
        return out


def run_closed_loop(stiffness_ratio: float, substeps: int = 2000, mass: float = 1.0, push: float = 1.0):
    """Drive a free body into a spring-held rod; return the reaction history.

    ``stiffness_ratio`` is ``k·dt²/M``. The rigid side is a semi-implicit Euler
    step of a body pushed by ``push`` plus the lagged reaction the coupler injects.
    """
    rod = SpringHeldRod(stiffness_ratio * mass / DT**2)
    coupler = DriveReactionCoupler(
        body_count=1, num_envs=1, root_offset=0, root_stride=1, drive_bodies=[0], relaxation=1.0, device=DEVICE,
    )
    follower = DriveRootFollower(num_envs=1, root_stride=1, drive_bodies=[0], device=DEVICE)

    def state(x):
        return SimpleNamespace(
            body_q=body_states([((x, 0, 0), IDENTITY)]),
            body_f=wp.zeros(1, dtype=wp.spatial_vector, device=DEVICE),
            particle_q=wp.zeros(1, dtype=wp.vec3, device=DEVICE),
        )

    x = v = 0.0
    history = []
    for _ in range(substeps):
        begin = state(x)
        coupler.apply(begin)
        force = push + float(begin.body_f.numpy()[0][0])
        v += DT * force / mass
        x += DT * v
        end = state(x)
        follower.sync(begin, end, rod)
        coupler.harvest(end, rod)
        history.append(float(coupler.coupling_forces.numpy()[0][0]))
    return np.asarray(history)


@pytest.mark.parametrize("stiffness_ratio", [0.1, 1.0, 3.0])
def test_closed_loop_stays_bounded_below_the_symplectic_limit(stiffness_ratio) -> None:
    """Including where an unrewound reaction was expected to alternate and blow up."""
    history = run_closed_loop(stiffness_ratio)
    assert np.all(np.isfinite(history))
    assert np.abs(history[-200:]).max() < 4.0


def test_closed_loop_diverges_past_the_symplectic_limit() -> None:
    """``k·dt²/M > 4`` is where relaxation or more substeps are needed."""
    history = run_closed_loop(5.0, substeps=200)
    assert not np.isfinite(history).all() or np.abs(history[-20:]).max() > 1e6
