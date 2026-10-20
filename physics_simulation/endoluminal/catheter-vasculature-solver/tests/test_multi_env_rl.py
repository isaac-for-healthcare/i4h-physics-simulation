# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the batched multi-env path used by RL training.

Three properties matter here and each has its own group below:

* **Parity** — a batched env must reproduce the single-env result exactly.
  Anything else means the batch is not simulating the same physics.
* **Independence** — envs must not influence one another, which is what makes
  per-env deformable vessels usable as separate episodes.
* **Reset** — resetting one env must leave its neighbours untouched, including
  their vessel deformation.

Runs on CPU so CI covers the batched kernels, and additionally on CUDA when a
device is present (those cases carry the ``gpu`` marker).
"""

from __future__ import annotations

import numpy as np
import pytest

wp = pytest.importorskip("warp")
wp.init()

from catheter_vasculature_solver import CathRodSolver, RodConfig, XPBDRodSolver  # noqa: E402
from catheter_vasculature_solver.vessel_deformation import (  # noqa: E402
    CenterlineData,
    CenterlineDynamicsParams,
    CenterlineVesselRuntime,
    build_centerline_tree,
)
from catheter_vasculature_solver.vessel_deformation.pbd_cosserat import CosseratRod  # noqa: E402
from catheter_vasculature_solver.vessel_deformation.pbd_cosserat import step as cosserat_step  # noqa: E402

ROD_HEIGHT = 0.5
SEGMENT_LENGTH = 0.02
NUM_SEGMENTS = 16
LUMEN_RADIUS = 0.02
CATHETER_RADIUS = 0.002
NUM_ENVS = 4


def _device_params():
    """CPU always; CUDA additionally when the driver exposes a device."""
    params = [pytest.param("cpu", id="cpu")]
    try:
        has_cuda = wp.get_cuda_device_count() > 0
    except Exception:  # pragma: no cover - no CUDA driver at all
        has_cuda = False
    if has_cuda:
        params.append(pytest.param("cuda:0", id="cuda", marks=pytest.mark.gpu))
    return params


DEVICES = _device_params()


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _curved_lumen(curvature: float, length: float, n_seg: int = 12):
    """Centerline along +X whose axis bends to ``+curvature`` in Y distally."""
    x = np.linspace(0.0, length, n_seg + 1).astype(np.float32)
    y = (curvature * (x / length) ** 2).astype(np.float32)
    z = np.full(n_seg + 1, ROD_HEIGHT, dtype=np.float32)
    starts = np.stack([x[:-1], y[:-1], z[:-1]], axis=1)
    ends = np.stack([x[1:], y[1:], z[1:]], axis=1)
    r = np.full(n_seg, LUMEN_RADIUS, dtype=np.float32)
    return build_centerline_tree(
        CenterlineData(starts, ends, np.zeros(n_seg, dtype=np.int32), r, r, r, r)
    )


def _rod_config(device: str) -> RodConfig:
    cfg = RodConfig()
    cfg.device = device
    cfg.geometry.num_segments = NUM_SEGMENTS
    cfg.geometry.segment_length = SEGMENT_LENGTH
    return cfg


def _vessel(device: str, num_envs: int, response: float, env_positions=None):
    tree = _curved_lumen(0.06, SEGMENT_LENGTH * NUM_SEGMENTS * 1.25)
    return CenterlineVesselRuntime.from_tree(
        tree,
        device=device,
        num_envs=num_envs,
        env_positions=env_positions,
        params=CenterlineDynamicsParams(iterations=2, root_locked=True),
        catheter_radius=CATHETER_RADIUS,
        max_distance=0.5,
        two_way=response > 0.0,
        vessel_response=response,
        catheter_max_delta=0.01,
        vessel_max_delta=0.01,
    )


def _solver(device: str, num_envs: int, *, runtime=None, collision_mesh=None,
            collision_enabled=False, track_enabled=False) -> CathRodSolver:
    cfg = _rod_config(device)
    return CathRodSolver(
        cfg,
        num_envs=num_envs,
        collision_mesh=collision_mesh,
        track_start=np.array([0.0, 0.0, ROD_HEIGHT], dtype=np.float32),
        track_dir=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        track_length=0.6,
        tip_num_edges=6,
        particle_radius=CATHETER_RADIUS,
        segment_length=SEGMENT_LENGTH,
        collision_enabled=collision_enabled,
        track_enabled=track_enabled,
        centerline_runtime=runtime,
    )


def _advance(solver, cfg_dt: float, steps: int = 30):
    for _ in range(steps):
        solver.step(cfg_dt)


def _tube_mesh(device: str, radius: float = LUMEN_RADIUS, length: float = 0.7,
               rings: int = 20, sides: int = 10):
    """Closed tube around the rod axis, as static collision geometry."""
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    xs = np.linspace(-0.05, length, rings)
    vertices = [
        [x, radius * np.cos(a), ROD_HEIGHT + radius * np.sin(a)] for x in xs for a in angles
    ]
    faces: list[list[int]] = []
    for i in range(rings - 1):
        base, nxt = i * sides, (i + 1) * sides
        for j in range(sides):
            k = (j + 1) % sides
            faces += [[base + j, nxt + j, nxt + k], [base + j, nxt + k, base + k]]
    return wp.Mesh(
        points=wp.array(np.asarray(vertices, dtype=np.float32), dtype=wp.vec3, device=device),
        indices=wp.array(np.asarray(faces, dtype=np.int32).reshape(-1), dtype=wp.int32, device=device),
    )


# --------------------------------------------------------------------------- #
# Cosserat vessel batching
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["pbd", "scr"])
def test_batched_cosserat_matches_single_rod(device, mode):
    """Every replica must reproduce the single-rod trajectory exactly."""
    positions = np.array(
        [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0], [0.4, 0, 0],
         [0.5, 0.1, 0], [0.6, 0.2, 0], [0.5, -0.1, 0], [0.6, -0.2, 0]],
        dtype=np.float32,
    )
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (4, 7), (7, 8)]
    kwargs = dict(edges=edges, fixed_nodes=[0], fixed_segments=[0], orientation_mode=mode)

    single = CosseratRod(positions, device, **kwargs)
    batched = CosseratRod(positions, device, num_envs=NUM_ENVS, **kwargs)
    for _ in range(20):
        cosserat_step(single, 0.005, gravity=(0.0, 0.0, -9.81), iterations=3)
        cosserat_step(batched, 0.005, gravity=(0.0, 0.0, -9.81), iterations=3)

    reference = single.positions()
    per_env = batched.positions_per_env()
    assert per_env.shape == (NUM_ENVS, single.n_nodes, 3)
    for env in range(NUM_ENVS):
        np.testing.assert_allclose(per_env[env], reference, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_batched_cosserat_keeps_launch_count_constant(device):
    """Replicas join existing colors, so projection launches must not grow."""
    positions = np.linspace(0.0, 0.4, 9, dtype=np.float32)[:, None] * np.array([1, 0, 0], np.float32)
    single = CosseratRod(positions, device, fixed_nodes=[0])
    batched = CosseratRod(positions, device, num_envs=32, fixed_nodes=[0])

    assert len(batched.stretch_groups) == len(single.stretch_groups)
    assert len(batched.bend_groups) == len(single.bend_groups)
    # Same colors, each 32x wider.
    for wide, narrow in zip(batched.stretch_groups, single.stretch_groups, strict=True):
        assert wide.shape[0] == narrow.shape[0] * 32


@pytest.mark.parametrize("device", DEVICES)
def test_per_env_geometry_gives_per_env_rest_state(device):
    """Per-env positions must produce per-env rest lengths, not a shared one."""
    base = np.linspace(0.0, 0.4, 9, dtype=np.float32)[:, None] * np.array([1, 0, 0], np.float32)
    scaled = np.stack([base * (1.0 + 0.25 * e) for e in range(3)]).astype(np.float32)
    rod = CosseratRod(scaled, device, num_envs=3, fixed_nodes=[0])

    totals = rod.rest_len.numpy().reshape(3, -1).sum(axis=1)
    assert totals[0] < totals[1] < totals[2]


def test_cosserat_rejects_env_position_mismatch():
    with pytest.raises(ValueError, match="num_envs"):
        CosseratRod(np.zeros((2, 5, 3), dtype=np.float32), "cpu", num_envs=3)


# --------------------------------------------------------------------------- #
# Rod reset
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("device", DEVICES)
def test_reset_restores_only_selected_envs(device):
    """Selective reset is the core RL requirement: neighbours must be untouched."""
    cfg = _rod_config(device)
    solver = XPBDRodSolver(cfg, num_envs=NUM_ENVS)
    rest = solver.positions.detach().cpu().numpy().copy()

    # Drive envs differently so their states are distinguishable.
    push = np.array([0.0, 0.3, 0.0, 0.6], dtype=np.float32)
    solver.apply_proximal_control_gpu(push, np.zeros(NUM_ENVS, dtype=np.float32), cfg.solver.dt)
    _advance(solver, cfg.solver.dt, steps=25)
    moved = solver.positions.detach().cpu().numpy().copy()
    assert np.abs(moved[3] - rest[3]).max() > 1.0e-4  # env 3 really moved

    solver.reset([1, 3])
    after = solver.positions.detach().cpu().numpy()
    velocities = solver.velocities.detach().cpu().numpy()

    for env in (1, 3):
        np.testing.assert_allclose(after[env], rest[env], rtol=0, atol=1.0e-6)
        np.testing.assert_allclose(velocities[env], 0.0, rtol=0, atol=1.0e-6)
    for env in (0, 2):
        np.testing.assert_allclose(after[env], moved[env], rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_reset_all_and_empty(device):
    cfg = _rod_config(device)
    solver = XPBDRodSolver(cfg, num_envs=NUM_ENVS)
    rest = solver.positions.detach().cpu().numpy().copy()
    _advance(solver, cfg.solver.dt, steps=10)

    solver.reset([])  # no-op
    solver.reset()  # everything
    np.testing.assert_allclose(solver.positions.detach().cpu().numpy(), rest, rtol=0, atol=1.0e-6)


def test_reset_rejects_out_of_range_env():
    solver = XPBDRodSolver(_rod_config("cpu"), num_envs=2)
    with pytest.raises(IndexError, match="out of range"):
        solver.reset([2])


@pytest.mark.parametrize("device", DEVICES)
def test_single_env_reset_works(device):
    """The single-env path shares the reset kernels, so it must reset too."""
    cfg = _rod_config(device)
    solver = XPBDRodSolver(cfg, num_envs=1)
    rest = solver.positions.detach().cpu().numpy().copy()
    solver.apply_proximal_control(0.5, 0.0, cfg.solver.dt)
    _advance(solver, cfg.solver.dt, steps=20)

    solver.reset()
    np.testing.assert_allclose(solver.positions.detach().cpu().numpy(), rest, rtol=0, atol=1.0e-6)


# --------------------------------------------------------------------------- #
# Per-env actuation and diagnostics
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("device", DEVICES)
def test_set_tip_bend_applies_on_batched_path(device):
    """Regression: set_tip_bend used to write only the unread single-env buffer.

    It returned without error and had exactly zero effect on a batched solve, so
    every env silently simulated a straight-tip catheter.
    """
    cfg = _rod_config(device)

    def tip_after(bend: bool):
        solver = _solver(device, NUM_ENVS)
        if bend:
            solver.set_tip_bend(0.9)
        _advance(solver, cfg.solver.dt, steps=20)
        return solver.positions.detach().cpu().numpy()

    straight, bent = tip_after(False), tip_after(True)
    assert np.abs(straight - bent).max() > 1.0e-4
    # And it must apply to every env, not just the first.
    for env in range(NUM_ENVS):
        assert np.abs(straight[env] - bent[env]).max() > 1.0e-4


@pytest.mark.parametrize("device", DEVICES)
def test_set_root_position_is_per_env(device):
    solver = _solver(device, NUM_ENVS)
    targets = np.array(
        [[0.0, 0.0, ROD_HEIGHT], [0.0, 0.1, ROD_HEIGHT]], dtype=np.float32
    )
    solver.set_root_position(targets, env_ids=[0, 2])

    roots = solver.positions.detach().cpu().numpy()[:, 0, :]
    np.testing.assert_allclose(roots[0], targets[0], rtol=0, atol=1.0e-6)
    np.testing.assert_allclose(roots[2], targets[1], rtol=0, atol=1.0e-6)
    # Broadcast form applies one position to every env.
    solver.set_root_position(np.array([0.0, -0.2, ROD_HEIGHT], dtype=np.float32))
    roots = solver.positions.detach().cpu().numpy()[:, 0, :]
    for env in range(NUM_ENVS):
        np.testing.assert_allclose(roots[env, 1], -0.2, rtol=0, atol=1.0e-6)


def test_set_root_position_rejects_bad_shape():
    solver = _solver("cpu", NUM_ENVS)
    with pytest.raises(ValueError, match="rows"):
        solver.set_root_position(np.zeros((3, 3), dtype=np.float32), env_ids=[0, 1])


@pytest.mark.parametrize("device", DEVICES)
def test_diagnostics_are_per_env(device):
    cfg = _rod_config(device)
    solver = XPBDRodSolver(cfg, num_envs=NUM_ENVS)
    _advance(solver, cfg.solver.dt, steps=5)

    assert solver.max_delta_lambda_per_env.shape == (NUM_ENVS,)
    assert solver.max_correction_per_env.shape == (NUM_ENVS,)
    # The scalar summary stays the batch-wide maximum.
    assert solver.max_delta_lambda == pytest.approx(
        float(solver.max_delta_lambda_per_env.max()), rel=1.0e-6
    )


@pytest.mark.parametrize("device", DEVICES)
def test_reset_scopes_diagnostics_to_reset_envs(device):
    """Reset envs must not report a stale residual, and others must keep theirs.

    A trainer reads diagnostics between ``reset()`` and the next ``step()``, so
    clearing nothing makes a fresh env look like it is diverging, and clearing
    everything hides a genuinely diverging env that is still mid-episode.
    """
    cfg = _rod_config(device)
    solver = XPBDRodSolver(cfg, num_envs=NUM_ENVS)
    solver.apply_proximal_control_gpu(
        np.full(NUM_ENVS, 0.4, dtype=np.float32), np.zeros(NUM_ENVS, dtype=np.float32), cfg.solver.dt
    )
    _advance(solver, cfg.solver.dt, steps=15)

    before = solver.max_delta_lambda_per_env.copy()
    assert before.max() > 0.0, "no residual to test against"

    solver.reset([1])
    after = solver.max_delta_lambda_per_env
    assert after[1] == 0.0
    for env in (0, 2, 3):
        assert after[env] == pytest.approx(before[env], rel=0, abs=0)


# --------------------------------------------------------------------------- #
# Batched vessel: parity, independence, reset
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("response", [0.0, 0.5, 0.9])
def test_batched_vessel_matches_single_env(device, response):
    """One-way and two-way containment must be identical to the single-env path."""
    cfg = _rod_config(device)

    single = _solver(device, 1, runtime=_vessel(device, 1, response))
    _advance(single, cfg.solver.dt)
    ref_catheter = single.positions.detach().cpu().numpy()
    ref_wall = single.centerline_runtime.positions.copy()

    runtime = _vessel(device, NUM_ENVS, response)
    batched = _solver(device, NUM_ENVS, runtime=runtime)
    _advance(batched, cfg.solver.dt)
    catheter = batched.positions.detach().cpu().numpy()
    wall = runtime.positions_per_env

    for env in range(NUM_ENVS):
        np.testing.assert_allclose(catheter[env], ref_catheter, rtol=0, atol=1.0e-6)
        np.testing.assert_allclose(wall[env], ref_wall, rtol=0, atol=1.0e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_per_env_vessels_are_independent(device):
    """Distinct anatomy per env must give distinct tool, wall and contact state."""
    cfg = _rod_config(device)
    length = SEGMENT_LENGTH * NUM_SEGMENTS * 1.25
    curvatures = [0.02, 0.05, 0.08, 0.11]
    env_positions = np.stack(
        [np.asarray(_curved_lumen(c, length).positions, dtype=np.float32) for c in curvatures]
    )

    runtime = _vessel(device, NUM_ENVS, 0.6, env_positions=env_positions)
    solver = _solver(device, NUM_ENVS, runtime=runtime)
    rest_wall = runtime.positions_per_env.copy()
    _advance(solver, cfg.solver.dt)

    tips = solver.positions.detach().cpu().numpy()[:, -1, 1]
    wall_motion = np.abs(runtime.positions_per_env - rest_wall).reshape(NUM_ENVS, -1).max(axis=1)
    depths = solver.contact_depth_per_env

    # Straighter lumen deflects the tool less and moves its wall less.
    assert np.all(np.diff(tips) > 0.0), f"tips not ordered by curvature: {tips}"
    assert np.all(np.diff(wall_motion) > 0.0), f"wall motion not ordered: {wall_motion}"
    assert len(set(np.round(depths, 9))) > 1
    assert solver.contact_count_per_env.shape == (NUM_ENVS,)


@pytest.mark.parametrize("device", DEVICES)
def test_vessel_reset_is_per_env(device):
    """Resetting one env must leave its neighbours' wall deformation intact."""
    cfg = _rod_config(device)
    runtime = _vessel(device, NUM_ENVS, 0.6)
    solver = _solver(device, NUM_ENVS, runtime=runtime)
    rest_wall = runtime.positions_per_env.copy()
    _advance(solver, cfg.solver.dt)

    deformed = runtime.positions_per_env.copy()
    assert np.abs(deformed[2] - rest_wall[2]).max() > 1.0e-5  # it really deformed

    solver.reset([1])
    after = runtime.positions_per_env
    np.testing.assert_allclose(after[1], rest_wall[1], rtol=0, atol=1.0e-6)
    for env in (0, 2, 3):
        np.testing.assert_allclose(after[env], deformed[env], rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_reset_keeps_other_envs_contact_history(device):
    """A partial reset must not erase contact state for continuing episodes.

    Rewards and terminations are built from ``contact_depth_per_env`` after the
    trainer resets the envs that finished, so zeroing the whole accumulator
    would report no wall contact for envs that are in fact pressed against it.
    """
    cfg = _rod_config(device)
    runtime = _vessel(device, NUM_ENVS, 0.6)
    solver = _solver(device, NUM_ENVS, runtime=runtime)
    _advance(solver, cfg.solver.dt)

    depth_before = solver.contact_depth_per_env.copy()
    count_before = solver.contact_count_per_env.copy()
    assert depth_before.max() > 0.0, "no contact to test against"

    solver.reset([1])
    depth_after = solver.contact_depth_per_env
    count_after = solver.contact_count_per_env

    assert depth_after[1] == 0.0
    assert count_after[1] == 0
    for env in (0, 2, 3):
        assert depth_after[env] == pytest.approx(depth_before[env], rel=0, abs=0)
        assert count_after[env] == count_before[env]


def test_solver_rejects_vessel_env_mismatch():
    """A shared vessel across envs would silently couple episodes."""
    runtime = _vessel("cpu", 2, 0.5)
    with pytest.raises(ValueError, match="num_envs"):
        _solver("cpu", 4, runtime=runtime)


def test_runtime_reset_rejects_out_of_range_env():
    runtime = _vessel("cpu", 2, 0.5)
    with pytest.raises(IndexError, match="out of range"):
        runtime.reset([5])


@pytest.mark.parametrize("device", DEVICES)
def test_skinning_selects_one_env(device):
    """Skinning is a render path, so it follows one explicitly chosen env."""
    runtime = _vessel(device, NUM_ENVS, 0.5)
    nodes = np.asarray(runtime.tree.positions, dtype=np.float32)
    sides = 6
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ring = np.stack([np.zeros(sides), np.cos(angles), np.sin(angles)], axis=1) * LUMEN_RADIUS
    vertices = np.concatenate([n + ring for n in nodes]).astype(np.float32)
    triangles = []
    for i in range(len(nodes) - 1):
        base, nxt = i * sides, (i + 1) * sides
        for j in range(sides):
            k = (j + 1) % sides
            triangles += [[base + j, nxt + j, nxt + k], [base + j, nxt + k, base + k]]

    runtime.attach_surface(vertices, np.asarray(triangles, dtype=np.int32), env=2)
    assert runtime.surface_positions is not None
    assert np.isfinite(runtime.surface_positions).all()

    with pytest.raises(IndexError, match="out of range"):
        runtime.attach_surface(vertices, np.asarray(triangles, dtype=np.int32), env=NUM_ENVS)


# --------------------------------------------------------------------------- #
# Batched static mesh collision and track guidance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    ("collision", "track", "mesh_edge"),
    [(True, False, False), (True, False, True), (False, True, False), (True, True, True)],
    ids=["sdf", "sdf+edge", "track", "all"],
)
def test_batched_static_mesh_and_track_match_single_env(device, collision, track, mesh_edge):
    """The static mesh is shared, so batching must not change any env's result."""
    cfg = _rod_config(device)

    def run(num_envs: int):
        solver = _solver(
            device,
            num_envs,
            collision_mesh=_tube_mesh(device),
            collision_enabled=collision,
            track_enabled=track,
        )
        solver.mesh_edge_collision_enabled = mesh_edge
        _advance(solver, cfg.solver.dt, steps=25)
        return solver.positions.detach().cpu().numpy()

    reference, batched = run(1), run(NUM_ENVS)
    for env in range(NUM_ENVS):
        np.testing.assert_allclose(batched[env], reference, rtol=0, atol=1.0e-6)
