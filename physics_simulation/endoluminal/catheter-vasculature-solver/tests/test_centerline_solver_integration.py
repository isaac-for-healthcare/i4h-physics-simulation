# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CathRodSolver ↔ centerline vessel co-simulation.

The scenario throughout is a straight catheter advanced into a lumen that curves
away in +Y: the tool enters at the inlet and must be deflected by the wall. That
makes the two-way response directly observable — a rigid wall bends the tool, a
compliant wall yields instead.
"""

from __future__ import annotations

import numpy as np
import pytest

wp = pytest.importorskip("warp")
wp.init()

from catheter_vasculature_solver import CathRodSolver, RodConfig  # noqa: E402
from catheter_vasculature_solver.vessel_deformation import (  # noqa: E402
    CenterlineData,
    CenterlineDynamicsParams,
    CenterlineVesselRuntime,
    build_centerline_tree,
)

ROD_HEIGHT = 0.5  # XPBDRodSolver default initial_height
SEGMENT_LENGTH = 0.02
NUM_SEGMENTS = 24
ROD_LENGTH = SEGMENT_LENGTH * NUM_SEGMENTS
LUMEN_LENGTH = ROD_LENGTH * 1.25
LUMEN_RADIUS = 0.02
CATHETER_RADIUS = 0.002
CURVATURE = 0.06  # lumen axis offset in +Y at the distal end
SETTLE_STEPS = 60


def _lumen_tree(curvature: float = CURVATURE, n_seg: int = 16, radius: float = LUMEN_RADIUS):
    """Lumen along +X whose axis bends to ``+curvature`` in Y at the distal end.

    The inlet coincides with the (pinned) catheter root, so the tool starts
    inside the vessel and progressively loads the outer wall of the bend.
    """
    x = np.linspace(0.0, LUMEN_LENGTH, n_seg + 1).astype(np.float32)
    y = (curvature * (x / LUMEN_LENGTH) ** 2).astype(np.float32)
    z = np.full(n_seg + 1, ROD_HEIGHT, dtype=np.float32)
    starts = np.stack([x[:-1], y[:-1], z[:-1]], axis=1)
    ends = np.stack([x[1:], y[1:], z[1:]], axis=1)
    r = np.full(n_seg, radius, dtype=np.float32)
    return build_centerline_tree(
        CenterlineData(starts, ends, np.zeros(n_seg, dtype=np.int32), r, r, r, r)
    )


def _config(num_segments: int = NUM_SEGMENTS):
    cfg = RodConfig()
    cfg.device = "cpu"
    cfg.geometry.num_segments = num_segments
    cfg.geometry.segment_length = SEGMENT_LENGTH
    cfg.solver.newton_iterations = 4
    return cfg


def _runtime(device: str = "cpu", curvature: float = CURVATURE, **kwargs):
    kwargs.setdefault("catheter_radius", CATHETER_RADIUS)
    kwargs.setdefault("max_distance", 0.5)
    return CenterlineVesselRuntime.from_tree(
        _lumen_tree(curvature=curvature),
        device=device,
        params=CenterlineDynamicsParams(iterations=2, root_locked=True),
        **kwargs,
    )


def _solver(cfg, runtime=None, **kwargs):
    # Track guidance pulls the rod back onto the straight insertion rail, which
    # would fight the wall; these tests isolate the vessel interaction.
    kwargs.setdefault("track_enabled", False)
    return CathRodSolver(
        cfg,
        collision_mesh=None,
        track_start=np.array([0.0, 0.0, ROD_HEIGHT], dtype=np.float32),
        track_dir=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        track_length=0.5,
        tip_num_edges=10,
        particle_radius=CATHETER_RADIUS,
        segment_length=cfg.geometry.segment_length,
        collision_enabled=False,
        centerline_runtime=runtime,
        **kwargs,
    )


def _settle(runtime=None, steps: int = SETTLE_STEPS, cfg=None, **solver_kwargs):
    """Step a solver to rest; return (tip deflection in +Y, peak wall motion)."""
    cfg = cfg or _config()
    solver = _solver(cfg, runtime, **solver_kwargs)
    rest = None if runtime is None else runtime.positions.copy()
    for _ in range(steps):
        solver.step(cfg.solver.dt)

    positions = solver.positions.detach().cpu().numpy()
    assert np.isfinite(positions).all()
    wall_motion = 0.0 if rest is None else float(np.abs(runtime.positions - rest).max())
    return float(positions[-1, 1]), wall_motion


# ────────────────────────────────────────────────────────────── co-simulation


def test_solver_steps_with_centerline_runtime():
    cfg = _config()
    runtime = _runtime()
    solver = _solver(cfg, runtime)

    for _ in range(10):
        solver.step(cfg.solver.dt)

    catheter = solver.positions.detach().cpu().numpy()
    assert catheter.shape == (cfg.geometry.num_segments + 1, 3)
    assert np.isfinite(catheter).all()
    assert runtime.positions.shape == (runtime.rod.n_nodes, 3)
    assert np.isfinite(runtime.positions).all()


def test_curved_lumen_deflects_catheter():
    """Containment is what makes the tool follow the vessel instead of going straight."""
    straight_tip, _ = _settle(None)
    guided_tip, wall_motion = _settle(_runtime(two_way=False))

    assert abs(straight_tip) < 1.0e-3  # no wall: the stiff rod stays straight
    assert guided_tip > 0.02  # pushed a good fraction of the way around the bend
    assert wall_motion == pytest.approx(0.0, abs=1.0e-9)  # one-way: wall is rigid


def test_compliant_wall_yields_instead_of_deflecting_catheter():
    """Two-way contact trades tool deflection for wall displacement."""
    rigid_tip, rigid_wall = _settle(_runtime(two_way=False))
    compliant_tip, compliant_wall = _settle(_runtime(two_way=True, vessel_response=0.8))

    assert compliant_wall > 1.0e-3
    assert rigid_wall < compliant_wall
    assert compliant_tip < rigid_tip


def test_vessel_response_scales_wall_compliance():
    """Higher vessel_response ⇒ more wall motion, less tool deflection."""
    stiff_tip, stiff_wall = _settle(_runtime(two_way=True, vessel_response=0.2))
    soft_tip, soft_wall = _settle(_runtime(two_way=True, vessel_response=0.9))

    assert soft_wall > stiff_wall
    assert soft_tip < stiff_tip


def test_vessel_is_advanced_once_per_substep():
    """predict/finalize must bracket every substep, not just every step."""
    cfg = _config()
    cfg.solver.num_substeps = 3
    runtime = _runtime()
    solver = _solver(cfg, runtime)

    calls = {"predict": 0, "finalize": 0}
    predict, finalize = runtime.predict, runtime.finalize

    def counting(name, fn):
        def wrapped(dt):
            calls[name] += 1
            return fn(dt)

        return wrapped

    runtime.predict = counting("predict", predict)
    runtime.finalize = counting("finalize", finalize)

    solver.step(cfg.solver.dt)
    assert calls == {"predict": 3, "finalize": 3}


def test_containment_can_be_disabled_while_vessel_still_simulates():
    cfg = _config()
    runtime = _runtime(two_way=True, vessel_response=0.8)
    solver = _solver(cfg, runtime, centerline_containment_enabled=False)
    rest = runtime.positions.copy()

    for _ in range(SETTLE_STEPS):
        solver.step(cfg.solver.dt)

    assert not solver.centerline_containment_enabled
    # The vessel still integrates, but never sees the catheter — and the
    # catheter never sees the wall, so it stays straight.
    assert np.allclose(runtime.positions, rest, atol=1.0e-5)
    assert abs(float(solver.positions.detach().cpu().numpy()[-1, 1])) < 1.0e-3


# ─────────────────────────────────────────────────────────── stage selection


@pytest.mark.parametrize("stage", ["pre", "post"])
def test_containment_runs_once_per_substep_at_configured_stage(stage):
    """Regression: containment must not fire in both hooks in a single substep."""
    cfg = _config()
    solver = _solver(
        cfg,
        _runtime(),
        centerline_containment_stage=stage,
        # Both static-mesh stages on: the centerline stage stays independent.
        collision_pre_constraints_enabled=True,
        collision_post_constraints_enabled=True,
    )

    calls = []
    original = solver._project_centerline_containment

    def counting(ws):
        calls.append(stage)
        return original(ws)

    solver._project_centerline_containment = counting

    solver.step(cfg.solver.dt)
    assert len(calls) == cfg.solver.num_substeps


def test_invalid_centerline_stage_is_rejected():
    with pytest.raises(ValueError, match="stage"):
        _solver(_config(), _runtime(), centerline_containment_stage="during")

    solver = _solver(_config(), _runtime())
    with pytest.raises(ValueError, match="stage"):
        solver.set_centerline_runtime(_runtime(), stage="sometimes")


# ───────────────────────────────────────────────────────── unsupported setups


def test_cuda_graph_capture_disabled_with_centerline():
    cfg = _config()
    solver = _solver(cfg, _runtime())
    assert solver._can_use_cuda_graph(cfg.solver.dt) is False

    solver.set_centerline_runtime(None)
    assert solver.centerline_runtime is None
    assert not solver.centerline_containment_enabled


def test_batched_path_rejects_centerline_runtime():
    with pytest.raises(NotImplementedError, match="single-environment"):
        _solver(_config(), _runtime(), num_envs=2)

    batched = _solver(_config(), None, num_envs=2)
    with pytest.raises(NotImplementedError, match="single-environment"):
        batched.set_centerline_runtime(_runtime())
