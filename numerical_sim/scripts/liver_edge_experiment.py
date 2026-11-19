# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Local package paths must be set before importing simulation modules.
"""Exploratory seeded edge grasping. Separate from the frozen qualification campaign."""

import argparse
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
import warp as wp
from i4h_surgical.config import load_case
from i4h_surgical.grasp_geometry import normals
from i4h_surgical.grasp_probe import GraspTargetProbe
from i4h_surgical.grasp_snapshot import HarnessSnapshot
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--attempts", type=int, default=8)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    sim = load_case("liver_pinned_grasp_1x", data=args.data)
    sim.prepare()
    p = GraspTargetProbe(sim.physics)
    # Reuse original disk bundle after checking every array and immutable identity against its own metadata.
    # Source identity is unchanged: this experiment lives in scripts, outside numerical packages.
    old = root / "artifacts/liver-grasp-reviewed"
    ident = json.loads((old / "identity.json").read_text())
    snap = HarnessSnapshot(sim.physics, ident, {})
    snap.load(old / "settled_snapshot.npz")
    snap.restore({})
    q = sim.scene.field("q", p.body).numpy()[0]
    az, el = np.deg2rad([20, 15])
    bx = np.array([np.cos(az), 0, -np.sin(az)])
    bz = np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
    by = np.cross(bz, bx)
    projected = q @ np.array([bx, by, bz]).T
    lo = projected[:, :2].min(0)
    hi = projected[:, :2].max(0)
    uv = np.column_stack(
        (
            (projected[:, 0] - lo[0]) / (hi[0] - lo[0]),
            1 - (projected[:, 1] - lo[1]) / (hi[1] - lo[1]),
        )
    )
    # Approximate user annotation in anatomy-normalized front X/Y projection.
    curve = np.array(
        [
            [0.03, 0.58],
            [0.12, 0.61],
            [0.23, 0.62],
            [0.35, 0.69],
            [0.46, 0.78],
            [0.57, 0.87],
        ]
    )
    distance = np.full(len(q), np.inf)
    for a, b in zip(curve[:-1], curve[1:]):
        t = np.clip(((uv - a) @ (b - a)) / np.dot(b - a, b - a), 0, 1)
        distance = np.minimum(
            distance, np.linalg.norm(uv - a - t[:, None] * (b - a), axis=1)
        )
    normal, valid = normals(q, p.triangles)
    eligible = (distance < 0.065) & p.eligible & valid
    candidates = np.flatnonzero(eligible)
    # Frontmost layer in each local projected neighbourhood prevents selecting the back wall.
    candidates = np.array(
        [
            i
            for i in candidates
            if projected[i, 2]
            >= projected[np.linalg.norm(uv - uv[i], axis=1) < 0.035, 2].max() - 0.008
        ],
        dtype=int,
    )
    if not len(candidates):
        raise RuntimeError("annotation mapping produced no candidates")
    rng = np.random.default_rng(args.seed)
    targets = rng.choice(
        candidates, size=min(args.attempts, len(candidates)), replace=False
    )
    image = Image.new("RGB", (900, 850), (20, 25, 33))
    draw = ImageDraw.Draw(image)
    screen = uv * [760, 700] + [70, 70]
    for tri in p.triangles[np.argsort(projected[p.triangles, 2].mean(1))]:
        draw.polygon([tuple(v) for v in screen[tri]], fill=(130, 60, 48))
    for i in candidates:
        x, y = screen[i]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(30, 240, 65))
    for i in targets:
        x, y = screen[i]
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline="yellow", width=2)
        draw.text((x + 8, y), str(i), fill="white")
    draw.text(
        (20, 20),
        "Approximate marked front-edge region: green candidates / yellow seeded samples",
        fill="white",
    )
    image.save(out / "edge_selection.png")
    (out / "manifest.json").write_text(
        json.dumps(
            dict(
                seed=args.seed,
                candidates=candidates.tolist(),
                targets=targets.tolist(),
                annotation="approximate anatomy-normalized front projection; review selection image",
                curve=curve.tolist(),
            ),
            indent=2,
        )
    )
    ic = p.ic
    sign = ic.sphere_jaw_sign.numpy()
    fixed = p.jaws.numpy()
    left = fixed[sign[fixed] > 0]
    right = fixed[sign[fixed] < 0]
    results = []
    for target in targets:
        snap.restore({}, p)
        p.select(int(target))
        action = ic.actions.numpy()
        frames = []
        logs = []
        acquired = False
        closed_tick = 0
        n = normal[target]
        tangent = np.array([1.0, -0.65, 0.0])
        tangent -= n * np.dot(n, tangent)
        tangent /= np.linalg.norm(tangent)
        # Jaw hinge X follows edge; opening Y follows tissue normal; forward -Z approaches edge.
        x = tangent
        y = n
        z = np.cross(x, y)
        z /= np.linalg.norm(z)
        y = np.cross(z, x)
        rotation = np.array(
            wp.quat_from_matrix(
                wp.mat33(*np.column_stack((x, y, z)).astype(np.float32).ravel())
            ),
            dtype=np.float32,
        )
        action[0, 0, 3:7] = rotation
        phase = "align"
        hold = 0
        for step in range(900):
            if step % 2 == 0:
                live = sim.scene.field("q", p.body).numpy()[0, target]
                spheres = ic.sphere_q.numpy()[0]
                # Distal spheres on each jaw form an opposed pair. Midpoint must straddle the edge.
                left_sphere = left[np.argmin(spheres[left] @ z)]
                right_sphere = right[np.argmin(spheres[right] @ z)]
                midpoint = (spheres[left_sphere] + spheres[right_sphere]) / 2
                if phase == "align":
                    delta = live + z * 0.006 - midpoint
                    if np.linalg.norm(delta) < 0.0004:
                        phase = "insert"
                elif phase == "insert":
                    delta = live - midpoint
                    if np.linalg.norm(delta) < 0.00035:
                        phase = "close"
                        p.arm()
                        closed_tick = 0
                else:
                    delta = np.zeros(3)
                if phase in ("align", "insert"):
                    limit = 0.001 if phase == "align" else 0.0001
                    action[0, 0, :3] += delta * min(
                        1, limit / max(np.linalg.norm(delta), 1e-12)
                    )
                elif phase == "close":
                    closed_tick += 1
                    action[0, 0, 7] = min(
                        1, closed_tick / 60
                    )  # two seconds of gradual closure
                if acquired:
                    phase = "hold"
                    hold += 1
                ic.actions.assign(action)
            sim.step()
            p.sample()
            m = p.read()
            qnow = sim.scene.field("q", p.body).numpy()[0]
            count = int(m["counts"][0, 0])
            acquired = bool(m["latched"][0])
            logs.append(
                dict(
                    step=step,
                    phase=phase,
                    grip=float(action[0, 0, 7]),
                    capture=count,
                    patch=int(m["counts"][0, 1]),
                    wrong_patch=int(m["counts"][0, 5]),
                    retained=int(m["counts"][0, 3]),
                    jaw_angle=float(ic.jaw_angle.numpy()[0, 0]),
                )
            )
            frames.append(qnow)
            if acquired and (
                m["counts"][0, 5] or m["counts"][0, 3] < (m["counts"][0, 1] + 1) // 2
            ):
                break
            if hold >= 30 or closed_tick >= 90:
                break
        result = dict(
            target=int(target),
            steps=step + 1,
            captured=count,
            patch=int(m["counts"][0, 1]),
            wrong_patch=bool(m["counts"][0, 5]),
            retained=int(m["counts"][0, 3]),
            held_half_second=hold >= 30,
        )
        results.append(result)
        print(result, flush=True)
        np.savez_compressed(out / f"target_{target}.npz", positions=frames)
        (out / f"target_{target}.json").write_text(json.dumps(logs))
        (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
