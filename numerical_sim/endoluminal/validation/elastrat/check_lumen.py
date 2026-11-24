# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect saved inserted shapes against the actual inward-wound, open phantom mesh."""

import csv

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom
from run import HERE, REF, inputs, inserted


@wp.kernel
def surface_distance(
    mesh: wp.uint64, points: wp.array(dtype=wp.vec3), gaps: wp.array(dtype=float)
):
    i = wp.tid()
    hit = wp.mesh_query_point_sign_normal(mesh, points[i], 1.0)
    if hit.result:
        near = wp.mesh_eval_position(mesh, hit.face, hit.u, hit.v)
        gaps[i] = -hit.sign * wp.length(points[i] - near)


def distances(mesh, points):
    q = wp.array(points, dtype=wp.vec3, device="cpu")
    d = wp.full(len(points), float("nan"), dtype=float, device="cpu")
    wp.launch(surface_distance, len(points), inputs=[mesh.id, q, d], device="cpu")
    mm = d.numpy() * 1000
    assert np.isfinite(mm).all()
    return mm


def check(group):
    folder = HERE / group
    stage = Usd.Stage.Open(str(REF.parent / "phantom.usdc"))
    usd = UsdGeom.Mesh(stage.GetPrimAtPath("/Phantom"))
    mesh = wp.Mesh(
        wp.array(usd.GetPointsAttr().Get(), dtype=wp.vec3, device="cpu"),
        wp.array(usd.GetFaceVertexIndicesAttr().Get(), dtype=int, device="cpu"),
    )
    rows = []
    for line in csv.DictReader((folder / "results.csv").open()):
        key = line["reference"]
        ref = np.load(REF / (key + ".npy"))
        sim = inserted(np.load(folder / (key + ".npy")), ref)
        sr, rr = distances(mesh, sim), distances(mesh, ref)
        rows.append(
            dict(
                reference=key,
                sim_count=len(sr),
                sim_outside_over_1mm=int(sum(sr > 1)),
                sim_max_outside_mm=max(0.0, max(sr)),
                ct_count=len(rr),
                ct_outside_over_1mm=int(sum(rr > 1)),
                ct_max_outside_mm=max(0.0, max(rr)),
            )
        )
    expected = {r["ct_prefix"] + str(i) for r in inputs(group) for i in (1, 2, 3)}
    if {r["reference"] for r in rows} != expected or len(rows) != len(expected):
        raise RuntimeError(f"Incomplete reference set for {group}")
    with (folder / "lumen_check.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    for group in ("catheters", "guidewires", "guidewires_profiled"):
        check(group)
