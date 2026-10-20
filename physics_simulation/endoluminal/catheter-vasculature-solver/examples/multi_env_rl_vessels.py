#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batched envs, one deformable vessel each — the shape of an RL rollout.

Each env owns an independent centerline vessel, so two-way contact in one env
cannot move another env's wall. That independence is what lets the batch stand in
for parallel episodes. Vessel memory scales with ``num_envs``; containment does
not scale in *launches*, because all envs are contained in one batched launch per
collision iteration.

The script walks the three things a trainer needs:

1. **Per-env anatomy.** Every env gets its own lumen curvature, so the batch
   covers a distribution of vessels rather than one repeated case.
2. **Per-env observations.** Tip position, deepest wall penetration and contact
   extent all come back per env, which is what rewards and terminations read.
3. **Per-env reset.** ``solver.reset(env_ids)`` returns the selected envs' rod
   *and* vessel to rest and leaves every other env mid-episode.

Runs on CPU by default, no GPU required::

    python examples/multi_env_rl_vessels.py

Pass a CUDA device to run a larger batch, where batching actually pays::

    python examples/multi_env_rl_vessels.py --device cuda:0 --num-envs 64
"""

from __future__ import annotations

import argparse

import numpy as np

ROD_HEIGHT = 0.5  # XPBDRodSolver default initial_height
SEGMENT_LENGTH = 0.02
NUM_SEGMENTS = 24
LUMEN_RADIUS = 0.02
CATHETER_RADIUS = 0.002


def build_lumen(curvature: float, length: float, n_seg: int = 16):
    """Centerline tree along +X whose axis bends to ``+curvature`` in Y distally."""
    from catheter_vasculature_solver.vessel_deformation import CenterlineData, build_centerline_tree

    x = np.linspace(0.0, length, n_seg + 1).astype(np.float32)
    y = (curvature * (x / length) ** 2).astype(np.float32)
    z = np.full(n_seg + 1, ROD_HEIGHT, dtype=np.float32)
    starts = np.stack([x[:-1], y[:-1], z[:-1]], axis=1)
    ends = np.stack([x[1:], y[1:], z[1:]], axis=1)
    r = np.full(n_seg, LUMEN_RADIUS, dtype=np.float32)
    return build_centerline_tree(
        CenterlineData(starts, ends, np.zeros(n_seg, dtype=np.int32), r, r, r, r)
    )


def build_batch(device: str, num_envs: int, vessel_response: float):
    """One solver over ``num_envs`` envs, each with its own vessel and anatomy."""
    from catheter_vasculature_solver import CathRodSolver, RodConfig
    from catheter_vasculature_solver.vessel_deformation import CenterlineDynamicsParams, CenterlineVesselRuntime

    length = SEGMENT_LENGTH * NUM_SEGMENTS * 1.25
    # Sweep curvature across the batch: one anatomy per env.
    curvatures = np.linspace(0.02, 0.11, num_envs)
    env_positions = np.stack(
        [np.asarray(build_lumen(c, length).positions, dtype=np.float32) for c in curvatures]
    )

    runtime = CenterlineVesselRuntime.from_tree(
        build_lumen(float(curvatures[0]), length),
        device=device,
        num_envs=num_envs,
        env_positions=env_positions,
        params=CenterlineDynamicsParams(iterations=2, root_locked=True),
        catheter_radius=CATHETER_RADIUS,
        max_distance=0.5,
        two_way=vessel_response > 0.0,
        vessel_response=vessel_response,
        catheter_max_delta=0.01,
        vessel_max_delta=0.01,
    )

    cfg = RodConfig()
    cfg.device = device
    cfg.geometry.num_segments = NUM_SEGMENTS
    cfg.geometry.segment_length = SEGMENT_LENGTH
    solver = CathRodSolver(
        cfg,
        num_envs=num_envs,
        collision_mesh=None,
        track_start=np.array([0.0, 0.0, ROD_HEIGHT], dtype=np.float32),
        track_dir=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        track_length=0.6,
        tip_num_edges=10,
        particle_radius=CATHETER_RADIUS,
        segment_length=SEGMENT_LENGTH,
        collision_enabled=False,
        track_enabled=False,
        centerline_runtime=runtime,
    )
    return cfg, solver, runtime, curvatures


def observations(solver, runtime, num_envs: int) -> dict[str, np.ndarray]:
    """The per-env signals a policy would consume, all without a per-env loop."""
    positions = solver.positions.detach().cpu().numpy().reshape(num_envs, -1, 3)
    return {
        "tip_y_mm": positions[:, -1, 1] * 1.0e3,
        "wall_depth_mm": solver.contact_depth_per_env * 1.0e3,
        "contact_samples": solver.contact_count_per_env,
        "solver_residual": solver.max_delta_lambda_per_env,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="Warp device, e.g. cpu or cuda:0")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument(
        "--vessel-response",
        type=float,
        default=0.6,
        help="0 keeps walls rigid (one-way); >0 lets the wall yield (two-way)",
    )
    args = parser.parse_args()

    import warp as wp

    wp.init()

    cfg, solver, runtime, curvatures = build_batch(args.device, args.num_envs, args.vessel_response)
    rest_wall = runtime.positions_per_env.copy()

    for _ in range(args.steps):
        solver.step(cfg.solver.dt)

    obs = observations(solver, runtime, args.num_envs)
    wall_motion = (
        np.abs(runtime.positions_per_env - rest_wall).reshape(args.num_envs, -1).max(axis=1) * 1.0e3
    )

    print(f"device={args.device}  num_envs={args.num_envs}  vessel_response={args.vessel_response}")
    print(f"vessel nodes held: {runtime.rod.n_nodes} ({runtime.nodes_per_env} per env)")
    print()
    print("  env  curvature   tip_y     wall_moved   deepest_penetration   contacts")
    shown = min(args.num_envs, 8)
    for env in range(shown):
        print(
            f"  {env:3d}  {curvatures[env]:8.3f}  {obs['tip_y_mm'][env]:7.3f}mm  "
            f"{wall_motion[env]:8.3f}mm  {obs['wall_depth_mm'][env]:15.3f}mm  "
            f"{obs['contact_samples'][env]:8d}"
        )
    if shown < args.num_envs:
        print(f"  ... {args.num_envs - shown} more envs")

    # Straighter lumens should deflect the tool less: the batch really is a
    # distribution of anatomies, not one case repeated num_envs times. Envs whose
    # lumen is straight enough that the tool never reaches the wall all sit at the
    # same undeflected tip, so the ordering is non-decreasing rather than strict --
    # with a fine curvature sweep, several envs legitimately tie at zero contact.
    tips = obs["tip_y_mm"]
    monotone = bool(np.all(np.diff(tips) >= -1.0e-6))
    touching = int(np.count_nonzero(obs["contact_samples"] > 0))
    print()
    print(f"tip deflection non-decreasing in curvature: {monotone}")
    print(
        f"envs in contact with the wall: {touching}/{args.num_envs} "
        f"(the rest are too straight to touch it)"
    )

    # Reset half the batch, as a trainer would on termination, and show that the
    # other half keeps its deformation.
    terminated = np.arange(0, args.num_envs, 2)
    deformed = runtime.positions_per_env.copy()
    solver.reset(terminated)
    after = runtime.positions_per_env

    reset_err = np.abs(after[terminated] - rest_wall[terminated]).max()
    survivors = np.setdiff1d(np.arange(args.num_envs), terminated)
    survivor_err = (
        np.abs(after[survivors] - deformed[survivors]).max() if survivors.size else 0.0
    )
    print(f"reset envs {terminated.tolist()[:6]}{'...' if terminated.size > 6 else ''}")
    print(f"  reset envs returned to rest wall (max err {reset_err:.2e} m)")
    print(f"  untouched envs kept their deformation (max drift {survivor_err:.2e} m)")


if __name__ == "__main__":
    main()
