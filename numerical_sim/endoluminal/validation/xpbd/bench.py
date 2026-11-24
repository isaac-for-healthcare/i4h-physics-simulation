# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Newton 1.8 catheter benchmark. Units are mm, g, ms, and N.

See validation/README.md for the pinned checkout and both iteration settings.
"""

import argparse
import csv
from itertools import pairwise
from pathlib import Path

import newton
import numpy as np
import warp as wp

N = 27
L = 60 / N


@wp.kernel
def move(
    center: wp.array(dtype=int),
    poses: wp.array(dtype=wp.transform),
    step: wp.array(dtype=int),
):
    i = wp.tid()
    p = poses[center[i]]
    pos = wp.transform_get_translation(p)
    pos[2] = -wp.min(float(step[0] + 1) / 3000.0, 5.0)
    poses[center[i]] = wp.transform(pos, wp.transform_get_rotation(p))


@wp.kernel
def tick(step: wp.array(dtype=int)):
    step[0] += 1


def run(iterations, output):
    input_path = (
        Path(__file__).resolve().parents[1] / "imperial/catheter_bending/inputs.csv"
    )
    with input_path.open() as stream:
        cases = [
            (
                row["device"],
                float(row["EI_N_mm2"]),
                float(row["OD_mm"]),
                float(row["ID_mm"]),
            )
            for row in csv.DictReader(stream)
        ]
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    centers, supports, guided, fixed = [], [], [], []
    rot = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), np.pi / 2)
    xyz = np.eye(3).tolist()
    axis = newton.ModelBuilder.JointDofConfig
    for i, (name, ei, od, inner) in enumerate(cases):
        native = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        rod = newton.Rod.create_straight(
            start=(-30.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0),
            length=60.0,
            segment_count=N,
            radius=od / 2,
        )
        axial = 16 * ei / ((od**2 + inner**2) * L)
        cfg = newton.ModelBuilder.ShapeConfig(
            density=(6 / N) / (np.pi * (od / 2) ** 2 * (L + 2 * od / 3))
        )
        native.add_rod(
            rod=rod,
            cfg=cfg,
            bend_stiffness=ei / L,
            stretch_stiffness=axial,
            shear_stiffness=axial / 2.6,
            twist_stiffness=ei / (1.3 * L),
        )
        model = native.finalize()
        assert (
            np.count_nonzero(model.joint_type.numpy() == newton.JointType.ROD) == N - 1
        )
        try:
            newton.solvers.SolverXPBD(model)
        except ValueError as e:
            if str(e) != "SolverXPBD does not support JointType.ROD joints.":
                raise
            print(name, e, flush=True)
        else:
            raise RuntimeError("Native rod accepted. Reassess the capability finding.")

        ids = []
        start_joint = b.joint_count
        for j in range(N):
            pose = wp.transform((-30 + (j + 0.5) * L, i * 100.0, 0.0), rot)
            link = b.add_link(xform=pose, lock_inertia=j == N // 2)
            b.add_shape_capsule(link, radius=od / 2, half_height=L / 2, cfg=cfg)
            ids.append(link)
        centers.append(ids[N // 2])
        for parent, child in pairwise(ids):
            b.add_joint_d6(
                parent,
                child,
                parent_xform=wp.transform((0.0, 0.0, L / 2), wp.quat_identity()),
                child_xform=wp.transform((0.0, 0.0, -L / 2), wp.quat_identity()),
                linear_axes=[
                    axis(axis=v, target_ke=k)
                    for v, k in zip(xyz, [axial / 2.6, axial / 2.6, axial])
                ],
                angular_axes=[
                    axis(axis=v, target_ke=k)
                    for v, k in zip(xyz, [ei / L, ei / L, ei / (1.3 * L)])
                ],
            )
        for j, x in [(4, -20.0), (22, 20.0)]:
            point = (x, i * 100.0, 0.0)
            anchor = b.add_link(xform=wp.transform(point, rot), lock_inertia=True)
            # Material joint frame is fixed at the capsule center. The support
            # can slide along local z. Transverse coordinates are hard constraints.
            b.add_joint_d6(
                ids[j],
                anchor,
                linear_axes=[axis(axis=xyz[2])],
                angular_axes=[axis(axis=v) for v in xyz],
            )
            supports.append(anchor)
            guided.append(ids[j])
            fixed.append(point)
        b.add_articulation(list(range(start_joint, b.joint_count)))
    model = b.finalize()
    assert np.all(model.body_inv_mass.numpy()[centers + supports] == 0)
    model.request_state_attributes("body_parent_f")
    solver = newton.solvers.SolverXPBD(model, iterations=iterations)
    states = [model.state(), model.state()]
    control = model.control()
    clock = wp.zeros(1, dtype=int)
    center_array = wp.array(centers, dtype=int)
    with wp.ScopedCapture() as cap:
        for i in range(16):
            a, b = states[i % 2], states[1 - i % 2]
            wp.launch(move, len(cases), [center_array, a.body_q, clock])
            wp.launch(tick, 1, [clock])
            solver.step(a, b, control, None, 1.0)
    centers = np.array(centers)
    rows = []
    for frame in range(1, 939):
        wp.capture_launch(cap.graph)
        q = states[0].body_q.numpy()
        forces = states[0].body_parent_f.numpy()
        assert np.isfinite(q).all() and np.isfinite(forces).all()
        # Parent-side reaction on driven center from the right joint has the
        # opposite sign of the child-side force reported on its right neighbour.
        nose = forces[centers, 2] - forces[centers + 1, 2]
        support = -forces[supports, 2].reshape(-1, 2).sum(axis=1)
        for i, (name, *_) in enumerate(cases):
            offsets = []
            gaps = []
            for j in [2 * i, 2 * i + 1]:
                pose = q[guided[j]]
                x, y, z, w = pose[3:].astype(float)
                axis_z = np.array(
                    [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]
                )
                diff = np.array(fixed[j]) - pose[:3]
                offset = diff @ axis_z
                offsets.append(abs(offset))
                gaps.append(np.linalg.norm(diff - offset * axis_z))
            rows.append(
                [
                    name,
                    frame * 0.016,
                    -q[centers[i], 2],
                    nose[i],
                    support[i],
                    max(gaps),
                    max(offsets),
                ]
            )
    with output.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "catheter",
                "time_s",
                "displacement_mm",
                "nose_N",
                "supports_N",
                "guide_gap_mm",
                "guide_offset_mm",
            ]
        )
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wp.set_device("cuda:0")
    run(args.iterations, args.output)
