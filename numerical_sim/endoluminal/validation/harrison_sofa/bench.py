# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SOFA corotational beam FEM, point-on-segment guides, central transverse displacement or force."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import Sofa.Core
import Sofa.Simulation
import SofaRuntime

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from harrison_inputs import SLOPES  # noqa: E402

DT = 0.001
EDGES = 24
R = 0.0004445


def run(wire, force_control, slope=None, radius=R, output=None, inner_radius=0.0):
    if slope is None:
        slope = SLOPES[wire]
    root = Sofa.Core.Node("root")
    root.gravity = [0, 0, 0]
    root.dt = DT
    root.addObject("FreeMotionAnimationLoop")
    solver = root.addObject(
        "BlockGaussSeidelConstraintSolver",
        maxIterations=100,
        tolerance=1e-12,
        computeConstraintForces=True,
    )
    beam = root.addChild("beam")
    beam.addObject("EulerImplicitSolver", rayleighMass=10, rayleighStiffness=0.0001)
    beam.addObject("BTDLinearSolver", template="BTDMatrix6d")
    points = [[float(x), 0, 0, 0, 0, 0, 1] for x in np.linspace(-0.03, 0.03, EDGES + 1)]
    dofs = beam.addObject(
        "MechanicalObject", name="dofs", template="Rigid3d", position=points
    )
    beam.addObject("MeshTopology", lines=[[i, i + 1] for i in range(EDGES)])
    h = 0.06 / EDGES
    axial = (radius**2 + inner_radius**2) / 2
    transverse = (3 * (radius**2 + inner_radius**2) + h**2) / 12
    beam.addObject(
        "UniformMass",
        vertexMass=f"{0.006 / (EDGES + 1)} 1 {axial} 0 0 0 {transverse} 0 0 0 {transverse}",
    )
    rigidity = slope * 1000 * 0.04**3 / 48
    beam.addObject(
        "BeamFEMForceField",
        radius=radius,
        radiusInner=inner_radius,
        youngModulus=rigidity / (np.pi * (radius**4 - inner_radius**4) / 4),
        poissonRatio=0.3,
    )
    beam.addObject("LinearSolverConstraintCorrection")
    if not force_control:
        beam.addObject(
            "LinearMovementProjectiveConstraint",
            indices=[EDGES // 2],
            keyTimes=[0, 15, 16],
            movements=[
                [0, 0, 0, 0, 0, 0],
                [0, -0.005, 0, 0, 0, 0],
                [0, -0.005, 0, 0, 0, 0],
            ],
        )
    load = beam.addObject(
        "ConstantForceField", indices=[EDGES // 2], forces=[[0, 0, 0, 0, 0, 0]]
    )
    anchors = root.addChild("anchors")
    anchors.addObject(
        "MechanicalObject",
        name="points",
        template="Rigid3d",
        position=[[-0.02, 0, 0, 0, 0, 0, 1], [0.02, 0, 0, 0, 0, 0, 1]],
    )
    guides = []
    for j, i in enumerate([3, 20]):
        guides.append(
            root.addObject(
                "SlidingLagrangianConstraint",
                name=f"guide{j}",
                template="Rigid3d",
                object1="@anchors/points",
                object2="@beam/dofs",
                sliding_point=j,
                axis_1=i,
                axis_2=i + 1,
            )
        )
    Sofa.Simulation.init(root)
    folder = (
        Path(output)
        if output
        else HERE / ("force_control" if force_control else "traces")
    )
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{wire}.csv").open("w") as stream:
        out = csv.writer(stream)
        out.writerow(
            [
                "time_s",
                "deflection_mm",
                "applied_force_N" if force_control else "nose_reaction_N",
                "max_guide_gap_mm",
                "normal_support_vertical_N",
                "max_tangential_impulse_Ns",
            ]
        )
        out.writerow([0] * 6)
        for step in range(1, 24001 if force_control else 15009):
            force = slope * step * DT / 3
            if force_control:
                load.forces.value = [[0, -force, 0, 0, 0, 0]]
            Sofa.Simulation.animate(root, DT)
            if step % 16:
                continue
            q = dofs.position.array()[:, :3]
            gaps = []
            vertical = axial_impulse = 0.0
            impulses = solver.constraintForces.array().reshape(2, 4)
            for x, guide, lam in zip([-0.02, 0.02], guides, impulses):
                i = int(guide.axis_1.value)
                edge = q[i + 1] - q[i]
                vertical += (
                    np.linalg.norm(lam[:2]) * abs(edge[0]) / np.linalg.norm(edge) / DT
                )
                axial_impulse = max(axial_impulse, abs(lam[2]), abs(lam[3]))
                t = np.clip((np.array([x, 0, 0]) - q[i]) @ edge / (edge @ edge), 0, 1)
                gaps.append(np.linalg.norm(q[i] + t * edge - [x, 0, 0]))
                i = int(np.clip(np.searchsorted(q[:, 0], x) - 1, 0, EDGES - 1))
                guide.axis_1.value = i
                guide.axis_2.value = i + 1
            d = -q[EDGES // 2, 1] * 1000
            out.writerow(
                [
                    step * DT,
                    d,
                    force if force_control else dofs.force.array()[EDGES // 2, 1],
                    1000 * max(gaps),
                    vertical,
                    axial_impulse,
                ]
            )
            if not np.isfinite(d) or d >= 5:
                break
    Sofa.Simulation.unload(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wire", nargs="*", default=list(SLOPES))
    parser.add_argument("--force-control", action="store_true")
    parser.add_argument("--ei-n-mm2", type=float)
    parser.add_argument("--radius-mm", type=float, default=R * 1000)
    parser.add_argument("--inner-radius-mm", type=float, default=0.0)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not 0 <= args.inner_radius_mm < args.radius_mm or not np.isfinite(
        args.radius_mm
    ):
        parser.error("Radii must be finite with 0 <= inner radius < outer radius.")
    if args.ei_n_mm2 is not None and not (
        np.isfinite(args.ei_n_mm2) and args.ei_n_mm2 > 0
    ):
        parser.error("Bending rigidity must be finite and positive.")
    SofaRuntime.importPlugin("Sofa.Component")
    for w in args.wire:
        run(
            w,
            args.force_control,
            None if args.ei_n_mm2 is None else 48 * args.ei_n_mm2 / 40**3,
            args.radius_mm / 1000,
            args.output,
            args.inner_radius_mm / 1000,
        )
