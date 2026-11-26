# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export a benchmark scene as i4h builds it, for the SOFA replicas (``sofa_aorta.py``, ``sofa_tissue.py``).

    uv run python scripts/benchmark/export_scene.py aorta --out aorta.npz
    uv run python scripts/benchmark/export_scene.py tissue --out tissue.npz --record-envs 64 --frames 240 --seed 0

``aorta`` writes the vessel mesh in world coordinates and the catheter's rest nodes. ``tissue`` writes the tissue
particles and per-part tetrahedra and surface triangles, the part connectors (particle -> triangle, barycentric), the
pinned particles, the cavity mesh, and the instruments' collision spheres: their radii and, for ``--record-envs``
environments, their positions at every physics step under the same uniform random actions as ``i4h_scaling.py
--controls random``, so SOFA replays exactly the tool motion i4h simulated. Run from ``numerical_sim/`` with
``I4H_DATA`` set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def export_aorta(out: Path) -> None:
    from i4h_host.composition import load_case, read_case

    sim = load_case(read_case("aorta_static"), device="cpu", envs=1)
    vessel = sim.scene.static_meshes[0]
    np.savez_compressed(
        out,
        vessel_points=vessel.points.astype(np.float64),
        vessel_triangles=vessel.indices.astype(np.int32),
        rod_nodes=sim.scene.q.numpy().astype(np.float64),
        dt=sim.dt,
        substeps=8,
        rod_radius=0.02,
        node_mass=1.0,
        track_length=6.35,
        insertion_rate=0.5,
        rotation_rate=1.5,
    )


def random_actions(rng, low, high, envs):
    """Uniform normalised actions in [-1, 1] mapped onto ``[low, high]`` (Isaac Lab's ``map_actions``)."""
    u = rng.uniform(-1.0, 1.0, size=(envs, low.shape[0])).astype(np.float32)
    return (low + high) / 2.0 + u * (high - low) / 2.0


def export_tissue(out: Path, record_envs: int, frames: int, seed: int) -> None:
    import warp as wp
    from i4h_host.composition import load_case, read_case

    case = read_case(str(HERE / "chole_tissue.yaml"))
    sim = load_case(case, device="cuda:0", envs=max(record_envs, 1))
    sc = sim.scene
    body = sc.body("tissue")
    tet = body.tet
    q0 = sc.q.numpy()[: body.count].astype(np.float64)
    parts = list(body.parts)
    tets, tet_part = np.asarray(tet.tets).reshape(-1, 4), np.asarray(tet.part).reshape(-1)
    tris = np.asarray(tet.tris).reshape(-1, 3)
    con = body.connectors
    cavity = sc.static_meshes[0]
    control = next(s for s in sim.systems if type(s).__name__ == "InstrumentControl")
    sim.prepare()
    low, high = (np.asarray(b, np.float32) for b in control.action_bounds(0.5))
    rng = np.random.default_rng(seed)
    spheres = []
    for _ in range(frames):
        control.actions.assign(random_actions(rng, low, high, sc.envs).reshape(sc.envs, control.count, -1))
        sim.step()
        spheres.append(control.sphere_q.numpy()[:record_envs].copy())
    wp.synchronize()
    fields = {
        "points": q0,
        "inv_mass": sc.inv_mass.numpy()[: body.count].astype(np.float64),
        "parts": np.array(parts),
        "dt": sim.dt,
        "substeps": 8,
        "gravity": np.array([0.0, -9.81, 0.0]),
        "cavity_points": cavity.points.astype(np.float64),
        "cavity_triangles": cavity.indices.astype(np.int32),
        "pin": np.asarray(body.pin.index, np.int32).reshape(-1),
        "connector_particle": np.asarray(con.particle, np.int32).reshape(-1),
        "connector_triangle": np.asarray(con.tri, np.int32).reshape(-1, 3),
        "connector_bary": np.asarray(con.bar, np.float64).reshape(-1, 3),
        "connector_group": np.asarray(con.group, np.int32).reshape(-1),
        "connector_stiffness": np.asarray(con.stiffness, np.float64).reshape(-1),
        "sphere_radius": control.sphere_radius.numpy().astype(np.float64),
        "sphere_trajectory": np.asarray(spheres, np.float64).transpose(1, 0, 2, 3),  # (envs, frames, spheres, 3)
        "contact_parts": np.array(["fat", "gallbladder"]),
    }
    cut_start = np.asarray(tet.cut_start).reshape(-1)
    for k, name in enumerate(parts):
        part = body.parts[name]  # deletable parts reserve cut-face slots after their authored surface
        fields[f"tets_{name}"] = tets[tet_part == k].astype(np.int32)
        fields[f"tris_{name}"] = tris[part.tri_start:cut_start[k]].astype(np.int32)
    np.savez_compressed(out, **fields)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scene", choices=["aorta", "tissue"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--record-envs", type=int, default=64)
    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.scene == "aorta":
        export_aorta(args.out)
    else:
        export_tissue(args.out, args.record_envs, args.frames, args.seed)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
