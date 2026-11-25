# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Newton SolverVBD cable with point-on-segment D6 guides. Internal units are mm, g, ms and N."""

import argparse
import csv
import sys
from contextlib import ExitStack
from pathlib import Path

import newton
import numpy as np
import warp as wp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from harrison_inputs import SLOPES  # noqa: E402

N = 27  # odd count places the nose and both guides at capsule centers initially
L = 60.0 / N
R = 0.4445
GUIDE_STIFFNESS = 1000.0  # N/mm


@wp.kernel
def apply_fixture(
    force_control: int,
    slopes: wp.array(dtype=float),
    forces: wp.array(dtype=wp.spatial_vector),
    centers: wp.array(dtype=int),
    guides: wp.array(dtype=int),
    parents: wp.array(dtype=int),
    anchors: wp.array(dtype=wp.vec3),
    poses: wp.array(dtype=wp.transform),
    clock: wp.array(dtype=int),
    dt_ms: float,
    joint_frame: wp.array(dtype=wp.transform),
):
    i = wp.tid()
    center = centers[i]
    pose = poses[center]
    pos = wp.transform_get_translation(pose)
    travel = float(clock[0]) * dt_ms * 0.001 / 3.0
    if force_control:
        forces[center] = wp.spatial_vector(0.0, 0.0, -slopes[i] * travel, 0.0, 0.0, 0.0)
    else:
        pos[2] = -wp.min(5.0, travel)
        poses[center] = wp.transform(pos, wp.transform_get_rotation(pose))
    for side in range(2):
        j = i * 2 + side
        local = wp.transform_point(wp.transform_inverse(poses[parents[j]]), anchors[j])
        joint_frame[guides[j]] = wp.transform(
            wp.vec3(0.0, 0.0, local[2]), wp.quat_identity()
        )


@wp.kernel
def tick(clock: wp.array(dtype=int)):
    clock[0] += 1


def tangent(p):
    x, y, z, w = p[3:].astype(float)
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def run(
    wires,
    device,
    iterations,
    substeps=16,
    output=None,
    force_control=False,
    slope=None,
    radius=R,
    inner_radius=0.0,
):
    slopes_by_wire = {w: SLOPES[w] if slope is None else slope for w in wires}
    output = Path(output or HERE / ("force_control" if force_control else "traces"))
    output.mkdir(parents=True, exist_ok=True)
    wp.set_device(device)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    centers, joints, parents, anchors = [], [], [], []
    for i, wire in enumerate(wires):
        y = 100.0 * i
        k = slopes_by_wire[wire]
        ei = k * 40.0**3 / 48
        rod = newton.Rod.create_straight(
            start=(-30.0, y, 0.0),
            direction=(1.0, 0.0, 0.0),
            length=60.0,
            segment_count=N,
            radius=radius,
        )
        cfg = newton.ModelBuilder.ShapeConfig(
            density=(6.0 / N) / (np.pi * radius**2 * (L + 4 * radius / 3))
        )
        ids, _ = b.add_rod(
            rod=rod,
            cfg=cfg,
            bend_stiffness=ei / L,
            stretch_stiffness=4 * ei / ((radius**2 + inner_radius**2) * L),
            shear_stiffness=4 * ei / (2.6 * (radius**2 + inner_radius**2) * L),
            twist_stiffness=ei / (1.3 * L),
            label=wire,
        )
        center = ids[N // 2]
        centers.append(center)
        if not force_control:
            b.body_mass[center] = b.body_inv_mass[center] = 0.0
            b.body_inertia[center] = b.body_inv_inertia[center] = wp.mat33(0.0)

        axis = newton.ModelBuilder.JointDofConfig
        for j, x in [(4, -20.0), (22, 20.0)]:
            point = (x, y, 0.0)
            anchor = b.add_body(
                is_kinematic=True,
                xform=wp.transform(point, wp.quat_identity()),
                lock_inertia=True,
            )
            joint = b.add_joint_d6(
                ids[j],
                anchor,
                linear_axes=[axis(axis=(0, 0, 1))],
                angular_axes=[axis(axis=v) for v in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]],
            )
            joints.append(joint)
            parents.append(ids[j])
            anchors.append(point)
    b.color()
    model = b.finalize()
    solver = newton.solvers.SolverVBD(
        model, iterations=iterations, rigid_joint_linear_ke=GUIDE_STIFFNESS
    )
    states = [model.state(), model.state()]
    clock = wp.zeros(1, dtype=int)
    controls = model.control()
    arrays = [
        wp.array(centers, dtype=int),
        wp.array(joints, dtype=int),
        wp.array(parents, dtype=int),
        wp.array(anchors, dtype=wp.vec3),
    ]
    slopes = wp.array([slopes_by_wire[w] for w in wires], dtype=float)

    def frame():
        for s in range(substeps):
            old, new = states[s % 2], states[1 - s % 2]
            old.clear_forces()
            wp.launch(tick, 1, [clock])
            wp.launch(
                apply_fixture,
                len(wires),
                [
                    int(force_control),
                    slopes,
                    old.body_f,
                    *arrays,
                    old.body_q,
                    clock,
                    16.0 / substeps,
                    model.joint_X_p,
                ],
            )
            solver.step(old, new, controls, None, 16.0 / substeps)

    if wp.get_device().is_cuda:
        with wp.ScopedCapture() as capture:
            frame()
    with ExitStack() as stack:
        writers = []
        for w in wires:
            out = csv.writer(stack.enter_context((output / f"{w}.csv").open("w")))
            out.writerow(
                [
                    "time_s",
                    "deflection_mm",
                    "nose_reaction_N",
                    "max_guide_gap_mm",
                    "max_abs_guide_offset_mm",
                    "support_reaction_N",
                    "applied_force_N",
                ]
            )
            out.writerow([0] * 7)
            writers.append(out)
        finished = set()
        for t in range(1, 939):
            if wp.get_device().is_cuda:
                wp.capture_launch(capture.graph)
            else:
                frame()
            q = states[0].body_q.numpy()
            for i, wire in enumerate(wires):
                if i in finished:
                    continue
                d = -q[centers[i], 2]
                gap = offset = 0.0
                support = 0.0
                for j in [2 * i, 2 * i + 1]:
                    pose = q[parents[j]].astype(float)
                    axis = tangent(pose)
                    diff = np.asarray(anchors[j]) - pose[:3]
                    along = axis @ diff
                    support += GUIDE_STIFFNESS * (diff - along * axis)[2]
                    gap = max(
                        gap, np.linalg.norm(diff - np.clip(along, -L / 2, L / 2) * axis)
                    )
                    offset = max(offset, abs(along))
                # Elastic force from the two rod joints adjoining the nose.
                c = centers[i]
                f = 0.0
                ei = slopes_by_wire[wire] * 40**3 / 48
                axial = 4 * ei / ((radius**2 + inner_radius**2) * L)
                shear = axial / 2.6
                for k, sign in [(c - 1, -1.0), (c, 1.0)]:
                    tp, tc = tangent(q[k]), tangent(q[k + 1])
                    strain = q[k + 1, :3].astype(float) - q[k, :3] - 0.5 * L * (tp + tc)
                    f += (
                        sign
                        * (shear * strain + (axial - shear) * (tp @ strain) * tp)[2]
                    )
                writers[i].writerow(
                    [
                        t * 0.016,
                        d,
                        f,
                        gap,
                        offset,
                        support,
                        slopes_by_wire[wire] * t * 0.016 / 3 if force_control else 0.0,
                    ]
                )
                if (
                    not np.isfinite(q[centers[i]]).all()
                    or d >= 5
                    or offset > L / 2
                    or gap > 1.0
                ):
                    finished.add(i)
            if len(finished) == len(wires):
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wire", nargs="*", default=list(SLOPES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--substeps", type=int, default=16)
    parser.add_argument("--output")
    parser.add_argument("--force-control", action="store_true")
    parser.add_argument("--ei-n-mm2", type=float)
    parser.add_argument("--radius-mm", type=float, default=R)
    parser.add_argument("--inner-radius-mm", type=float, default=0.0)
    args = parser.parse_args()
    if not 0 <= args.inner_radius_mm < args.radius_mm or not np.isfinite(
        args.radius_mm
    ):
        parser.error("Radii must be finite with 0 <= inner radius < outer radius.")
    if args.ei_n_mm2 is not None and not (
        np.isfinite(args.ei_n_mm2) and args.ei_n_mm2 > 0
    ):
        parser.error("Bending rigidity must be finite and positive.")
    if args.substeps <= 0 or args.substeps % 2:
        parser.error("Use a positive even substep count.")
    run(
        args.wire,
        args.device,
        args.iterations,
        args.substeps,
        args.output,
        args.force_control,
        None if args.ei_n_mm2 is None else 48 * args.ei_n_mm2 / 40**3,
        args.radius_mm,
        args.inner_radius_mm,
    )
