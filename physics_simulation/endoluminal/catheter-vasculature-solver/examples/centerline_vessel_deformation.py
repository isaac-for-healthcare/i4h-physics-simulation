#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deformable centerline vessel co-simulated with an XPBD catheter.

Builds a lumen that curves away from the insertion axis, advances a catheter into
it, and reports how the load is shared between tool and wall. Run it twice — once
rigid, once compliant — and the trade-off is the whole point of two-way contact:

* ``vessel_response = 0`` (one-way): the wall is rigid, so the *tool* bends.
* ``vessel_response > 0`` (two-way): the wall yields, so the tool bends less.

A tube surface is skinned to the centerline, so ``surface_positions`` /
``surface_normals`` are what a viewer or fluoro renderer would draw.

Runs on CPU by default, no GPU required::

    python examples/centerline_vessel_deformation.py

Pass a CUDA device to run the same scenarios on GPU, where the extra steps are cheap::

    python examples/centerline_vessel_deformation.py --device cuda:0 --steps 240
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from catheter_vasculature_solver.vessel_deformation import CenterlineTree

ROD_HEIGHT = 0.5  # XPBDRodSolver default initial_height
SEGMENT_LENGTH = 0.02
NUM_SEGMENTS = 24
LUMEN_RADIUS = 0.02
CATHETER_RADIUS = 0.002


def build_lumen(curvature: float, length: float, n_seg: int = 16, radius: float = LUMEN_RADIUS) -> CenterlineTree:
    """Centerline tree along +X whose axis bends to ``+curvature`` in Y distally."""
    from catheter_vasculature_solver.vessel_deformation import CenterlineData, build_centerline_tree

    x = np.linspace(0.0, length, n_seg + 1).astype(np.float32)
    y = (curvature * (x / length) ** 2).astype(np.float32)
    z = np.full(n_seg + 1, ROD_HEIGHT, dtype=np.float32)
    starts = np.stack([x[:-1], y[:-1], z[:-1]], axis=1)
    ends = np.stack([x[1:], y[1:], z[1:]], axis=1)
    r = np.full(n_seg, radius, dtype=np.float32)
    return build_centerline_tree(
        CenterlineData(starts, ends, np.zeros(n_seg, dtype=np.int32), r, r, r, r)
    )


def build_tube_surface(
    tree: CenterlineTree, radius: float = LUMEN_RADIUS, sides: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """One ring of vertices per centerline node, triangulated into a closed tube."""
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ring = np.stack([np.zeros(sides), np.cos(angles), np.sin(angles)], axis=1) * radius
    vertices = np.concatenate([node + ring for node in tree.positions]).astype(np.float32)

    triangles = []
    for i in range(len(tree.positions) - 1):
        base, nxt = i * sides, (i + 1) * sides
        for j in range(sides):
            k = (j + 1) % sides
            triangles.append([base + j, nxt + j, nxt + k])
            triangles.append([base + j, nxt + k, base + k])
    return vertices, np.asarray(triangles, dtype=np.int32)


def run(vessel_response: float, args) -> tuple[float, float]:
    """Advance one scenario; return (tip deflection in Y, peak wall motion)."""
    from catheter_vasculature_solver import CathRodSolver, RodConfig
    from catheter_vasculature_solver.vessel_deformation import CenterlineDynamicsParams, CenterlineVesselRuntime

    rod_length = SEGMENT_LENGTH * NUM_SEGMENTS
    tree = build_lumen(args.curvature, rod_length * 1.25)

    runtime = CenterlineVesselRuntime.from_tree(
        tree,
        device=args.device,
        params=CenterlineDynamicsParams(iterations=2, root_locked=True),
        catheter_radius=CATHETER_RADIUS,
        max_distance=0.5,
        two_way=vessel_response > 0.0,
        vessel_response=vessel_response,
        # Stabilizers: cap how far tool and wall may move within one substep.
        catheter_max_delta=0.01,
        vessel_max_delta=0.01,
    )

    if args.skin:
        vertices, triangles = build_tube_surface(tree)
        runtime.attach_surface(vertices, triangles)

    cfg = RodConfig()
    cfg.device = args.device
    cfg.geometry.num_segments = NUM_SEGMENTS
    cfg.geometry.segment_length = SEGMENT_LENGTH

    solver = CathRodSolver(
        cfg,
        collision_mesh=None,
        track_start=np.array([0.0, 0.0, ROD_HEIGHT], dtype=np.float32),
        track_dir=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        track_length=0.5,
        tip_num_edges=10,
        particle_radius=CATHETER_RADIUS,
        segment_length=cfg.geometry.segment_length,
        collision_enabled=False,  # no static mesh; the live lumen contains the tool
        track_enabled=False,
        centerline_runtime=runtime,
    )

    wall_rest = runtime.positions.copy()
    for _ in range(args.steps):
        solver.step(cfg.solver.dt)

    catheter = solver.positions.detach().cpu().numpy()
    tip_deflection = float(catheter[-1, 1])
    wall_motion = float(np.abs(runtime.positions - wall_rest).max())

    if args.skin:
        surface = runtime.surface_positions
        assert surface is not None and np.isfinite(surface).all()

    return tip_deflection, wall_motion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="warp device (cpu, cuda:0)")
    parser.add_argument("--steps", type=int, default=60, help="solver steps per scenario")
    parser.add_argument("--curvature", type=float, default=0.06, help="lumen Y offset at the outlet [m]")
    parser.add_argument("--no-skin", dest="skin", action="store_false", help="skip surface skinning")
    args = parser.parse_args()

    import warp as wp

    wp.init()

    print("Centerline Cosserat vessel + XPBD catheter")
    print(f"device={args.device} steps={args.steps} curvature={args.curvature} m")
    print("-" * 68)
    print(f"{'vessel_response':>16} | {'tip deflection [mm]':>20} | {'wall motion [mm]':>17}")
    print("-" * 68)

    for response in (0.0, 0.5, 0.9):
        tip, wall = run(response, args)
        print(f"{response:>16.2f} | {tip * 1e3:>20.3f} | {wall * 1e3:>17.3f}")

    print("-" * 68)
    print("A compliant wall (higher response) absorbs the load, so the tool deflects less.")
    if args.skin:
        print("Surface skinning is on: read runtime.surface_positions / surface_normals to render.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
