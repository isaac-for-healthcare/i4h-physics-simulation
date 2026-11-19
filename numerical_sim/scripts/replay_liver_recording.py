# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Local package paths must be set before importing simulation modules.
"""Replay physical device rows from authored reset and report first-acquisition retention."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for package in (
    "physics",
    "surgical",
    "standalone",
    "assessment",
    "isaaclab",
    "endoluminal",
):
    sys.path.insert(0, str(root / package / "src"))
import numpy as np
from i4h_surgical.config import load_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    folder = args.recording
    out = folder / "physics_replay"
    out.mkdir(exist_ok=True)
    actions = np.load(folder / "trajectory_primary.npy", allow_pickle=False)
    sim = load_case(folder / "case.yaml", data=args.data)
    sim.prepare()
    ic = next(s for s in sim.systems if s.id == "instrument_control")
    v = next(s for s in sim.systems if s.id == "volumetric_grasp")
    surface = next(s for s in sim.systems if s.id == "grasp")
    body = sim.scene.body("tissue")
    threshold = float(ic.grasp_threshold.numpy()[0])
    closed = False
    episode = None
    episodes = []
    positions = []
    rows = []
    roots = []
    angles = []
    for step, action in enumerate(actions):
        now = bool(action[7] >= threshold)
        if now and not closed:
            episode = dict(
                close_step=step,
                acquisition_step=None,
                acquisition_size=0,
                particle_ids=[],
                sphere_ids=[],
                release_step=None,
                minimum_retained=None,
                max_radial_residual_m=0.0,
                max_acquired_displacement_m=0.0,
            )
            episodes.append(episode)
            original = None
            lost = None
        if not now and closed:
            episode["release_step"] = step
        ic.actions.assign(action.reshape(1, 1, 10))
        sim.step()
        q = sim.scene.field("q", body).numpy()[0]
        a = v.active.numpy()[0, 0]
        particle = v.particle.numpy()[0, 0]
        sphere = v.sphere.numpy()[0, 0]
        rest = v.rest.numpy()[0, 0]
        spheres = ic.sphere_q.numpy()[0]
        count = int(v.count.numpy()[0, 0])
        sc = int(surface.count.numpy()[0, 0])
        retained = 0
        residual_max = 0.0
        if now and episode is not None:
            if episode["acquisition_step"] is None and count:
                slots = np.flatnonzero(a)
                ids = particle[slots].copy()
                sp = sphere[slots].copy()
                r = rest[slots].copy()
                original = (slots, ids, sp, r, q[ids].copy())
                lost = np.zeros(len(slots), bool)
                episode.update(
                    acquisition_step=step,
                    acquisition_size=len(slots),
                    particle_ids=ids.tolist(),
                    sphere_ids=sp.tolist(),
                    minimum_retained=len(slots),
                )
            if original is not None:
                slots, ids, sp, r, q0 = original
                lost |= (
                    (a[slots] == 0)
                    | (particle[slots] != ids)
                    | (sphere[slots] != sp)
                    | np.any(rest[slots] != r, axis=1)
                )
                residual = np.abs(
                    np.linalg.norm(q[ids] - spheres[sp], axis=1)
                    - np.linalg.norm(r, axis=1)
                )
                retained = int(np.count_nonzero(~lost & (residual <= 0.002)))
                residual_max = float(residual.max())
                episode["minimum_retained"] = min(episode["minimum_retained"], retained)
                episode["max_radial_residual_m"] = max(
                    episode["max_radial_residual_m"], residual_max
                )
                episode["max_acquired_displacement_m"] = max(
                    episode["max_acquired_displacement_m"],
                    float(np.linalg.norm(q[ids] - q0, axis=1).max()),
                )
                episode["last_closed_step"] = step
        rows.append(
            dict(
                step=step,
                time_s=(step + 1) * sim.dt,
                grip=float(action[7]),
                volume_count=count,
                surface_count=sc,
                original_retained=retained,
                max_radial_residual_m=residual_max,
                finite=bool(np.isfinite(q).all()),
            )
        )
        positions.append(q)
        roots.append(ic.root.numpy()[0, 0])
        angles.append(float(ic.jaw_angle.numpy()[0, 0]))
        closed = now
    for e in episodes:
        e["capture_hold_s"] = (
            0
            if e["acquisition_step"] is None
            else (e["last_closed_step"] - e["acquisition_step"] + 1) * sim.dt
        )
        e["retained_half_throughout"] = (
            e["acquisition_size"] > 0
            and e["minimum_retained"] >= (e["acquisition_size"] + 1) // 2
        )
    report = dict(
        samples=len(actions),
        duration_s=len(actions) * sim.dt,
        threshold=threshold,
        finite=all(r["finite"] for r in rows),
        episodes=episodes,
        captured_episodes=sum(e["acquisition_size"] > 0 for e in episodes),
        max_volume_count=max(r["volume_count"] for r in rows),
        max_surface_count=max(r["surface_count"] for r in rows),
        recording_sha256=hashlib.sha256(
            (folder / "trajectory_primary.npy").read_bytes()
        ).hexdigest(),
        limitation="Physics reconstruction from authored reset; original interactive tissue state, resets and wall-clock timing were not recorded.",
    )
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    np.savez_compressed(
        out / "states.npz",
        positions=positions,
        root=roots,
        jaw_angle=angles,
        actions=actions,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
