# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the public entry points that accept caller-supplied buffers.

A GPU-resident caller is the normal case for this solver: Isaac Lab hands
``reset`` a CUDA index tensor, and a batched controller keeps its per-env
velocities on device to avoid a host sync per substep. ``np.asarray`` cannot
read device memory, so every one of those entry points has to copy across
explicitly.

The regression these cover is asymmetry rather than a missing feature. The
batched control path always marshalled a torch tensor correctly; the single-rod
fallback beside it went straight to ``np.asarray`` and raised. Since the
batched workspace only exists for ``num_envs > 1``, a one-environment scene hit
the broken branch with the same call the batched one accepted.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

wp = pytest.importorskip("warp")
wp.init()

from catheter_vasculature_solver import CathRodSolver, RodConfig, XPBDRodSolver  # noqa: E402
from catheter_vasculature_solver.vessel_deformation import (  # noqa: E402
    CenterlineData,
    CenterlineDynamicsParams,
    CenterlineVesselRuntime,
    build_centerline_tree,
)

NUM_SEGMENTS = 8
NUM_ENVS = 4
DT = 1.0 / 240.0
PUSH = 0.4


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


def _solver(device: str, num_envs: int) -> XPBDRodSolver:
    cfg = RodConfig()
    cfg.device = device
    cfg.geometry.num_segments = NUM_SEGMENTS
    return XPBDRodSolver(cfg, num_envs=num_envs)


def _root(solver: XPBDRodSolver) -> np.ndarray:
    """The root particle, which is the one proximal control moves."""
    positions = solver.positions.detach().cpu().numpy()
    return positions[0].copy() if solver.num_envs == 1 else positions[:, 0, :].copy()


def _push_via(solver: XPBDRodSolver, container) -> np.ndarray:
    """Apply ``container`` as the push velocity and return the resulting root.

    The step is what applies the command: control is latched, not written
    straight into the workspace. Without it these would compare one unmoved rod
    against another and pass whatever the marshalling did.
    """
    rest = _root(solver)
    zeros = np.zeros(solver.num_envs, dtype=np.float32)
    solver.apply_proximal_control_gpu(container, zeros, DT)
    solver.step(DT)
    root = _root(solver)
    assert np.abs(root - rest).max() > 0.0, "control had no effect, so the comparison is vacuous"
    return root


# --------------------------------------------------------------------------- #
# Proximal control, single environment
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("device", DEVICES)
def test_single_env_control_accepts_a_tensor_on_the_solver_device(device):
    """The regression: one environment plus a device tensor used to raise."""
    expected = _push_via(_solver(device, 1), np.full(1, PUSH, dtype=np.float32))
    actual = _push_via(_solver(device, 1), torch.full((1,), PUSH, device=device))

    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "container",
    [
        pytest.param(np.full(1, PUSH, dtype=np.float32), id="ndarray"),
        pytest.param([PUSH], id="list"),
        pytest.param(PUSH, id="scalar"),
        pytest.param(torch.full((1,), PUSH), id="cpu-tensor"),
    ],
)
def test_single_env_control_agrees_across_containers(container):
    """The documented accepted types have to produce one result, not four."""
    expected = _push_via(_solver("cpu", 1), np.full(1, PUSH, dtype=np.float32))

    np.testing.assert_allclose(_push_via(_solver("cpu", 1), container), expected, rtol=0, atol=0)


def test_single_env_control_accepts_a_warp_array():
    expected = _push_via(_solver("cpu", 1), np.full(1, PUSH, dtype=np.float32))
    warp_push = wp.array(np.full(1, PUSH, dtype=np.float32), dtype=wp.float32, device="cpu")

    np.testing.assert_allclose(_push_via(_solver("cpu", 1), warp_push), expected, rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# Proximal control, batched
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("device", DEVICES)
def test_batched_control_accepts_a_tensor_on_the_solver_device(device):
    """The batched path already worked; this pins it against the single-env fix."""
    pushes = np.array([0.0, 0.2, 0.4, 0.6], dtype=np.float32)
    expected = _push_via(_solver(device, NUM_ENVS), pushes)
    actual = _push_via(_solver(device, NUM_ENVS), torch.as_tensor(pushes, device=device))

    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_batched_control_accepts_a_non_contiguous_column(device):
    """IsaacLab action terms slice a ``(num_envs, 2)`` command into columns.

    Each column is a strided view, so a conversion that flattens without
    compacting first would read the rotation channel as the push velocity.
    """
    pushes = np.array([0.0, 0.2, 0.4, 0.6], dtype=np.float32)
    rotations = np.array([9.0, 9.0, 9.0, 9.0], dtype=np.float32)
    commands = torch.as_tensor(np.stack([pushes, rotations], axis=1), device=device)
    assert not commands[:, 0].is_contiguous()

    expected = _push_via(_solver(device, NUM_ENVS), pushes)
    actual = _push_via(_solver(device, NUM_ENVS), commands[:, 0])

    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


def test_batched_control_broadcasts_a_single_value():
    """A single value means "same command everywhere", not a length mismatch."""
    expected = _push_via(_solver("cpu", NUM_ENVS), np.full(NUM_ENVS, PUSH, dtype=np.float32))

    np.testing.assert_allclose(_push_via(_solver("cpu", NUM_ENVS), torch.full((1,), PUSH)), expected, rtol=0, atol=0)


def test_batched_control_rejects_a_mismatched_length():
    """Rejected on the call, not deferred to the step the command is applied on."""
    solver = _solver("cpu", NUM_ENVS)
    with pytest.raises(ValueError, match="per-env array of length"):
        solver.apply_proximal_control_gpu(torch.zeros(NUM_ENVS + 1), torch.zeros(NUM_ENVS), DT)


# --------------------------------------------------------------------------- #
# Control is latched until the step
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("num_envs", [1, NUM_ENVS])
def test_control_does_nothing_until_the_rod_is_stepped(num_envs):
    """The command has to outlive the gap between commanding and stepping.

    Newton opens its step by copying its own particle buffer over the rod's, so
    a command written into the workspace on the call would be erased before it
    could act. Holding it until the step is what makes it survive.
    """
    solver = _solver("cpu", num_envs)
    rest = _root(solver)

    solver.apply_proximal_control_gpu(np.full(num_envs, PUSH, dtype=np.float32), np.zeros(num_envs, np.float32), DT)
    np.testing.assert_allclose(_root(solver), rest, rtol=0, atol=0)

    solver.step(DT)
    assert np.abs(_root(solver) - rest).max() > 0.0


@pytest.mark.parametrize("num_envs", [1, NUM_ENVS])
def test_only_the_latest_command_is_applied(num_envs):
    """Two commands without a step in between are not summed."""
    once = _solver("cpu", num_envs)
    once.apply_proximal_control_gpu(np.full(num_envs, PUSH, dtype=np.float32), np.zeros(num_envs, np.float32), DT)
    once.step(DT)

    twice = _solver("cpu", num_envs)
    twice.apply_proximal_control_gpu(np.full(num_envs, 9.0 * PUSH, np.float32), np.zeros(num_envs, np.float32), DT)
    twice.apply_proximal_control_gpu(np.full(num_envs, PUSH, dtype=np.float32), np.zeros(num_envs, np.float32), DT)
    twice.step(DT)

    np.testing.assert_allclose(_root(twice), _root(once), rtol=0, atol=0)


@pytest.mark.parametrize("num_envs", [1, NUM_ENVS])
def test_a_command_does_not_carry_across_a_reset(num_envs):
    """It belongs to the episode being torn down."""
    solver = _solver("cpu", num_envs)
    rest = _root(solver)

    solver.apply_proximal_control_gpu(np.full(num_envs, PUSH, dtype=np.float32), np.zeros(num_envs, np.float32), DT)
    solver.reset()
    solver.step(DT)

    np.testing.assert_allclose(_root(solver), rest, rtol=0, atol=1.0e-6)


def test_the_scalar_entry_point_is_latched_too():
    """``apply_proximal_control`` is the single-rod path and takes the same route."""
    solver = _solver("cpu", 1)
    rest = _root(solver)

    solver.apply_proximal_control(PUSH, 0.0, DT)
    np.testing.assert_allclose(_root(solver), rest, rtol=0, atol=0)

    solver.step(DT)
    assert np.abs(_root(solver) - rest).max() > 0.0


# --------------------------------------------------------------------------- #
# Reset
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("device", DEVICES)
def test_reset_accepts_an_index_tensor_on_the_solver_device(device):
    """``ManagerBasedEnv.reset`` forwards the device tensor it builds."""
    solver = _solver(device, NUM_ENVS)
    rest = _root(solver)
    _push_via(solver, np.full(NUM_ENVS, PUSH, dtype=np.float32))
    for _ in range(20):
        solver.step(DT)
    moved = _root(solver)
    assert np.abs(moved - rest).max() > 1.0e-4, "rod did not move, so the assertion below is vacuous"

    solver.reset(torch.tensor([0, 2], dtype=torch.int32, device=device))

    after = _root(solver)
    for env in (0, 2):
        np.testing.assert_allclose(after[env], rest[env], rtol=0, atol=1.0e-6)
    for env in (1, 3):
        np.testing.assert_allclose(after[env], moved[env], rtol=0, atol=0)


def test_reset_accepts_a_warp_index_array():
    solver = _solver("cpu", NUM_ENVS)
    rest = _root(solver)
    _push_via(solver, np.full(NUM_ENVS, PUSH, dtype=np.float32))
    for _ in range(20):
        solver.step(DT)

    solver.reset(wp.array([1], dtype=wp.int32, device="cpu"))

    np.testing.assert_allclose(_root(solver)[1], rest[1], rtol=0, atol=1.0e-6)


def test_reset_still_range_checks_a_tensor():
    """Bringing ids to the host must not skip the bounds check they feed."""
    solver = _solver("cpu", 2)
    with pytest.raises(IndexError, match="out of range"):
        solver.reset(torch.tensor([2], dtype=torch.int32))


# --------------------------------------------------------------------------- #
# Reset with a vessel attached
# --------------------------------------------------------------------------- #
ROD_HEIGHT = 0.5  # XPBDRodSolver default initial height
SEGMENT_LENGTH = 0.02
LUMEN_RADIUS = 0.02
CATHETER_RADIUS = 0.002


def _lumen_runtime(device: str, num_envs: int) -> CenterlineVesselRuntime:
    """Straight lumen along +X at the rod's height, one replica per env."""
    length = SEGMENT_LENGTH * NUM_SEGMENTS * 1.25
    x = np.linspace(0.0, length, 9).astype(np.float32)
    zeros = np.zeros(8, dtype=np.float32)
    height = np.full(9, ROD_HEIGHT, dtype=np.float32)
    starts = np.stack([x[:-1], zeros, height[:-1]], axis=1)
    ends = np.stack([x[1:], zeros, height[1:]], axis=1)
    radii = np.full(8, LUMEN_RADIUS, dtype=np.float32)
    tree = build_centerline_tree(
        CenterlineData(starts, ends, np.zeros(8, dtype=np.int32), radii, radii, radii, radii)
    )
    return CenterlineVesselRuntime.from_tree(
        tree,
        device=device,
        params=CenterlineDynamicsParams(iterations=2, root_locked=True),
        num_envs=num_envs,
        catheter_radius=CATHETER_RADIUS,
        max_distance=0.5,
    )


def _cath_solver(device: str, runtime: CenterlineVesselRuntime) -> CathRodSolver:
    cfg = RodConfig()
    cfg.device = device
    cfg.geometry.num_segments = NUM_SEGMENTS
    cfg.geometry.segment_length = SEGMENT_LENGTH
    return CathRodSolver(
        cfg,
        collision_mesh=None,
        track_start=np.array([0.0, 0.0, ROD_HEIGHT], dtype=np.float32),
        track_dir=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        track_length=0.5,
        tip_num_edges=4,
        particle_radius=CATHETER_RADIUS,
        segment_length=SEGMENT_LENGTH,
        collision_enabled=False,
        track_enabled=False,
        centerline_runtime=runtime,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_reset_with_a_vessel_accepts_an_index_tensor(device):
    """The rod forwards the caller's ids to the vessel, which validates its own.

    Two independent validators means two chances to assume a host buffer, and
    the vessel's is only reached once a wall is attached -- which is the
    configuration the workflow actually runs.
    """
    runtime = _lumen_runtime(device, num_envs=1)
    solver = _cath_solver(device, runtime)

    solver.reset(torch.tensor([0], dtype=torch.int32, device=device))


def test_vessel_reset_still_range_checks_a_tensor():
    runtime = _lumen_runtime("cpu", num_envs=1)

    with pytest.raises(IndexError, match="out of range"):
        runtime.reset(torch.tensor([1], dtype=torch.int32))


# --------------------------------------------------------------------------- #
# Root position and orientation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("device", DEVICES)
def test_set_root_position_accepts_a_tensor_on_the_solver_device(device):
    runtime = _lumen_runtime(device, num_envs=1)
    solver = _cath_solver(device, runtime)
    target = torch.tensor([0.01, 0.0, ROD_HEIGHT], device=device)

    solver.set_root_position(target)

    np.testing.assert_allclose(
        solver.positions.detach().cpu().numpy()[0],
        target.cpu().numpy(),
        rtol=0,
        atol=1.0e-6,
    )



@pytest.mark.parametrize("device", DEVICES)
def test_set_root_orientation_accepts_a_tensor_on_the_solver_device(device):
    solver = _solver(device, 1)
    quaternion = torch.tensor([0.0, 0.0, 0.7071068, 0.7071068], device=device)

    solver.set_root_orientation(0, quaternion)

    np.testing.assert_allclose(
        solver.orientations.detach().cpu().numpy()[0],
        quaternion.cpu().numpy(),
        rtol=0,
        atol=1.0e-6,
    )


def test_set_root_orientation_still_rejects_a_wrong_width():
    solver = _solver("cpu", 1)
    with pytest.raises(ValueError, match="4-vector"):
        solver.set_root_orientation(0, torch.zeros(3))
