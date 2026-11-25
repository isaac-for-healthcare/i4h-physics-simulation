# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The rod as a participant in Newton's coupled-solver framework.

The hook tests drive :class:`CoupledCathRodSolver` the way ``SolverCoupledProxy``
does, through a stand-in for the rod entry's view, so they run on the Newton this
package pins. The end-to-end test needs ``newton>=1.8``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

wp = pytest.importorskip("warp", reason="needs warp installed")
pytest.importorskip("newton", reason="needs newton installed")

from catheter_vasculature_solver import newton_coupled_rod  # noqa: E402
from catheter_vasculature_solver import RodConfig, RodParticleRange, XPBDRodSolver  # noqa: E402
from catheter_vasculature_solver.isaaclab_integration.proxy_coupling import harvest_drive_reaction_kernel  # noqa: E402
from catheter_vasculature_solver.newton_coupled_rod import CoupledCathRodSolver, build_coupled_rod_solver  # noqa: E402

DEVICE = "cpu"
DT = 1.0 / 240.0
IDENTITY = (0.0, 0.0, 0.0, 1.0)
PARTICLE_Q = 1 << 2


def make_rod(num_envs: int = 1, gravity=(0.0, 0.0, 0.0)) -> XPBDRodSolver:
    cfg = RodConfig()
    cfg.device = DEVICE
    cfg.geometry.num_segments = 4
    cfg.geometry.segment_length = 0.02
    cfg.solver.num_substeps = 1
    cfg.solver.gravity = gravity
    return XPBDRodSolver(cfg, num_envs=num_envs, floor_z=None)


def rod_view(rod, *, body_inv_mass, body_com):
    """The rod entry's view: the rod's particles and the proxy bodies, in local ids."""
    return SimpleNamespace(
        device=DEVICE,
        particle_count=rod.num_points * rod.num_envs,
        body_inv_mass=wp.array(body_inv_mass, dtype=wp.float32, device=DEVICE),
        body_com=wp.array(body_com, dtype=wp.vec3, device=DEVICE),
    )


def rod_state(rod, body_q, body_qd):
    return SimpleNamespace(
        particle_q=wp.array(rod.position_array.numpy(), dtype=wp.vec3, device=DEVICE),
        particle_qd=wp.zeros(rod.num_points * rod.num_envs, dtype=wp.vec3, device=DEVICE),
        body_q=wp.array([wp.transform(wp.vec3(*p), wp.quat(*q)) for p, q in body_q], dtype=wp.transform,
                        device=DEVICE),
        body_qd=wp.array([wp.spatial_vector(*v) for v in body_qd], dtype=wp.spatial_vector, device=DEVICE),
    )


def make_participant(rod, *, drive_bodies, mount_local, body_inv_mass, body_com):
    view = rod_view(rod, body_inv_mass=body_inv_mass, body_com=body_com)
    particle_range = RodParticleRange(offset=0, count=rod.num_points * rod.num_envs, num_envs=rod.num_envs)
    return CoupledCathRodSolver(view, rod, particle_range, drive_bodies=drive_bodies, mount_local=mount_local)


def global_forces(count, entries):
    forces = np.zeros((count, 6), dtype=np.float32)
    for index, wrench in entries.items():
        forces[index] = wrench
    return wp.array(forces, dtype=wp.spatial_vector, device=DEVICE)


@pytest.fixture
def state_flags(monkeypatch):
    if newton_coupled_rod.StateFlags is None:
        monkeypatch.setattr(newton_coupled_rod, "StateFlags", SimpleNamespace(PARTICLE_Q=PARTICLE_Q))
    return int(newton_coupled_rod.StateFlags.PARTICLE_Q)


def test_rewind_starts_the_root_on_the_mount_with_the_free_velocity() -> None:
    """Local proxy 1 is global body 7; the lagged reaction is taken out at the proxy's mass."""
    rod = make_rod()
    mount = (0.1, 0.0, 0.0)
    com = (0.02, 0.0, 0.0)
    inv_mass = 0.5
    solver = make_participant(rod, drive_bodies=[7], mount_local=[mount],
                              body_inv_mass=[0.0, inv_mass], body_com=[(0, 0, 0), com])
    state = rod_state(rod, [((0, 0, 0), IDENTITY), ((0.3, -0.2, 0.05), IDENTITY)],
                      [(0, 0, 0, 0, 0, 0), (1.0, 0.0, 0.0, 0.0, 0.0, 2.0)])
    reaction = (4.0, -2.0, 0.0, 0.0, 0.0, 0.0)

    solver.coupling_rewind_proxy_body(
        wp.array([-1, 7], dtype=int, device=DEVICE), state, global_forces(8, {7: reaction}), None, DT,
    )

    np.testing.assert_allclose(solver.root_positions.numpy()[0], [0.4, -0.2, 0.05], atol=1e-6)
    free = np.array([1.0, 0.0, 0.0]) - DT * inv_mass * np.array(reaction[:3])
    spin = np.cross([0.0, 0.0, 2.0], np.subtract(mount, com))
    np.testing.assert_allclose(solver.root_velocities.numpy()[0], free + spin, atol=1e-6)
    np.testing.assert_allclose(solver.root_inv_masses.numpy()[0], inv_mass)
    assert rod.has_pending_control


def test_harvest_replaces_the_guess_with_the_rods_reaction_on_the_drive_body() -> None:
    rod = make_rod(gravity=(0.0, 0.0, -9.81))
    mount = [(0.0, 0.0, 0.05)]
    solver = make_participant(rod, drive_bodies=[3], mount_local=mount, body_inv_mass=[1.0], body_com=[(0, 0, 0.01)])
    state = rod_state(rod, [((0, 0, 0), IDENTITY)], [(0, 0, 0, 0, 0, 0)])
    rod.step(DT)
    assert np.linalg.norm(rod.proximal_wrench().numpy()[0]) > 0.0
    out = global_forces(4, {3: (9.0, 9.0, 9.0, 9.0, 9.0, 9.0)})

    solver.coupling_harvest_proxy_wrenches(
        wp.array([3], dtype=int, device=DEVICE), out,
        body_qd_before=None, state=state, state_out=state, contacts=None, dt=DT,
    )

    expected = wp.zeros(4, dtype=wp.spatial_vector, device=DEVICE)
    wp.launch(harvest_drive_reaction_kernel, dim=1, inputs=[
        rod.proximal_wrench(), wp.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0.01)], dtype=wp.vec3, device=DEVICE),
        wp.array(mount, dtype=wp.vec3, device=DEVICE),
        wp.array([wp.transform_identity()] * 4, dtype=wp.transform, device=DEVICE),
        wp.array([3], dtype=wp.int32, device=DEVICE), expected,
    ], device=DEVICE)
    np.testing.assert_allclose(out.numpy()[3], expected.numpy()[3], atol=1e-6)
    np.testing.assert_allclose(out.numpy()[:3], 0.0)


def test_iteration_restart_re_solves_the_step_bit_for_bit(state_flags) -> None:
    rod = make_rod(num_envs=2)
    solver = make_participant(rod, drive_bodies=[0, 1], mount_local=None,
                              body_inv_mass=[2.0, 2.0], body_com=[(0, 0, 0)] * 2)
    solver.coupling_notify_iteration_restarts(True)
    state = rod_state(rod, [((0, 0, 0), IDENTITY)] * 2, [(0.5, 0, 0, 0, 0, 0)] * 2)
    state_out = rod_state(rod, [((0, 0, 0), IDENTITY)] * 2, [(0, 0, 0, 0, 0, 0)] * 2)
    proxies = wp.array([0, 1], dtype=int, device=DEVICE)
    forces = global_forces(2, {0: (1.0, 0, 0, 0, 0, 0)})

    def one_pass(restart):
        solver.coupling_notify_input_state_update(state, state_flags, iteration_restart=restart, dt=DT)
        solver.coupling_rewind_proxy_body(proxies, state, forces, None, DT)
        solver.step(state, state_out, None, None, DT)
        return rod.position_array.numpy().copy(), rod.orientation_array.numpy().copy()

    first = one_pass(False)
    second = one_pass(True)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_rewind_refuses_to_override_a_latched_root_command() -> None:
    rod = make_rod()
    solver = make_participant(rod, drive_bodies=[0], mount_local=None, body_inv_mass=[1.0], body_com=[(0, 0, 0)])
    state = rod_state(rod, [((0, 0, 0), IDENTITY)], [(0, 0, 0, 0, 0, 0)])
    rod.apply_proximal_control_gpu([0.5], [0.0], DT)

    with pytest.raises(RuntimeError, match="already latched"):
        solver.coupling_rewind_proxy_body(wp.array([0], dtype=int, device=DEVICE), state,
                                          global_forces(1, {}), None, DT)


def test_a_drive_body_that_is_not_a_proxy_is_rejected() -> None:
    rod = make_rod()
    solver = make_participant(rod, drive_bodies=[5], mount_local=None, body_inv_mass=[1.0], body_com=[(0, 0, 0)])
    state = rod_state(rod, [((0, 0, 0), IDENTITY)], [(0, 0, 0, 0, 0, 0)])

    with pytest.raises(RuntimeError, match="not proxies"):
        solver.coupling_rewind_proxy_body(wp.array([4], dtype=int, device=DEVICE), state,
                                          global_forces(6, {}), None, DT)


def test_builder_needs_the_coupled_framework_and_two_iterations(monkeypatch) -> None:
    rod = make_rod()
    particle_range = RodParticleRange(offset=0, count=rod.num_points, num_envs=1)
    model = SimpleNamespace(body_count=1, joint_count=0, shape_count=0, particle_count=rod.num_points)

    monkeypatch.setattr(newton_coupled_rod, "SolverCoupledProxy", None)
    with pytest.raises(ImportError, match="newton>=1.8"):
        build_coupled_rod_solver(model, rod, particle_range, rigid_solver=None, drive_bodies=[0])

    monkeypatch.setattr(newton_coupled_rod, "SolverCoupledProxy", object())
    with pytest.raises(ValueError, match="2 or more"):
        build_coupled_rod_solver(model, rod, particle_range, rigid_solver=None, drive_bodies=[0], iterations=1)


def test_held_rod_follows_a_body_through_solver_coupled_proxy() -> None:
    """End to end on Newton's own coupler: a free body carries the rod's root on its mount."""
    if not newton_coupled_rod.coupled_framework_available():
        pytest.skip("needs newton>=1.8 (newton.solvers.experimental.coupled)")
    import newton
    from catheter_vasculature_solver.isaaclab_integration import add_catheter_rod_to_builder
    from newton.solvers import SolverSemiImplicit

    cfg = RodConfig()
    cfg.device = DEVICE
    cfg.geometry.num_segments = 4
    cfg.geometry.segment_length = 0.02
    cfg.solver.num_substeps = 1
    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_body(xform=wp.transform_identity(), mass=1.0)
    builder.add_shape_sphere(body, radius=0.01)
    particle_range = add_catheter_rod_to_builder(builder, cfg, start=np.zeros(3))
    model = builder.finalize(device=DEVICE)
    rod = XPBDRodSolver(cfg, num_envs=1, floor_z=None)
    solver = build_coupled_rod_solver(
        model, rod, particle_range,
        rigid_solver=lambda view: SolverSemiImplicit(view),
        drive_bodies=[body], iterations=2,
    )
    state_0, state_1 = model.state(), model.state()
    state_0.body_qd.assign([wp.spatial_vector(0.2, 0.0, 0.0, 0.0, 0.0, 0.0)])

    for _ in range(10):
        solver.step(state_0, state_1, model.control(), None, DT)
        state_0, state_1 = state_1, state_0

    body_x = state_0.body_q.numpy()[body][:3]
    root = state_0.particle_q.numpy()[particle_range.offset]
    assert np.all(np.isfinite(state_0.particle_q.numpy()))
    np.testing.assert_allclose(root, body_x, atol=1e-3)
