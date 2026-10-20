# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""600-step anatomy regression; controls only, pins/reset/diagnostics in every representation.

Run each case/mode in a fresh process (CUDA resources are released at exit).
Captured/eager long-horizon drift is reported, never used to widen parity tolerances.
"""

import argparse
import json
import subprocess
import sys

import numpy as np
from i4h_endoluminal.systems.rods.catheter_control import CatheterControl
from i4h_isaaclab.composition import load_case

CASES = ("aorta_shell", "aorta_centreline")


def qualify(case, graph, steps=600):
    sim = load_case(case, envs=2, debug=True)
    sim.cuda_graph = graph
    sim.prepare()
    assert (sim.graph is not None) == graph
    scene = sim.scene
    control = next(s for s in sim.systems if isinstance(s, CatheterControl))
    initial = {b.name: scene.field("q", b).numpy().copy() for b in scene.bodies}
    max_depth = {s.id: 0.0 for s in sim.systems if hasattr(s, "contact_depth")}
    total_count = {s: 0 for s in max_depth}
    for i in range(steps):
        actions = np.zeros((scene.envs, 3), np.float32)
        actions[:, 0] = np.linspace(0.4, 0.7, scene.envs)
        if i < 120:
            actions[:, 1] = 0.15
            actions[:, 2] = 0.1 * (1 if i < 60 else -1)
        control.actions.assign(actions)
        sim.step()
        if (i+1) % 60 == 0:
            assert np.isfinite(scene.q.numpy()).all() and np.isfinite(scene.qd.numpy()).all()
            for s in sim.systems:
                if s.id in max_depth:
                    depths, counts = s.contact_depth.numpy(), s.contact_count.numpy()
                    assert np.isfinite(depths).all() and (depths >= 0).all() and (counts >= 0).all()
                    max_depth[s.id] = max(max_depth[s.id], float(depths.max()))
                    total_count[s.id] += int(counts.sum())
    deformation = {}
    for b in scene.bodies:
        q = scene.field("q", b).numpy()
        pinned = (b.rest.inv_mass == 0) & (b is not control.body)
        np.testing.assert_array_equal(q[pinned], initial[b.name][pinned])
        deformation[b.name] = float(np.max(np.abs(q-initial[b.name])))
    keep = {b.name: scene.field("q", b).numpy()[1:].copy() for b in scene.bodies}
    sim.reset([0])
    for b in scene.bodies:
        q = scene.field("q", b).numpy()
        np.testing.assert_array_equal(q[1:], keep[b.name])
        np.testing.assert_array_equal(q[0], initial[b.name][0])
    for s in sim.systems:
        if s.id in max_depth:
            assert s.contact_count.numpy()[0] == 0 and s.contact_depth.numpy()[0] == 0
    print(json.dumps({"case": case, "graph": graph, "steps": steps, "envs": scene.envs,
                      "deformation": deformation, "depth_sample_max": max_depth,
                      "count_sample_sum": total_count, "pins_reset_finite": "passed"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    if args.case:
        qualify(args.case, not args.no_cuda_graph, args.steps)
    else:
        for case in CASES:
            for eager in (False, True):
                cmd = [sys.executable, __file__, "--case", case, "--steps", str(args.steps)]
                if eager:
                    cmd.append("--no-cuda-graph")
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
