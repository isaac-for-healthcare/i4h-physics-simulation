# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Local package paths must be set before importing simulation modules.
"""Separate manual-informed development campaign, preserving the original frozen results."""

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
sys.path.insert(0, str(root / "examples/liver_grasp"))
import i4h_surgical.grasp_qualification as qualification
import numpy as np
from i4h_surgical.config import load_case, read_case
from i4h_surgical.grasp_probe import GraspTargetProbe
from i4h_surgical.grasp_qualification import attempt
from i4h_surgical.grasp_snapshot import HarnessSnapshot, identity
from manual_teacher import ManualGeometryTeacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix", action="store_true")
    args = parser.parse_args()
    out = args.output
    out.mkdir(exist_ok=True, parents=True)
    old = root / "artifacts/liver-grasp-reviewed"
    cfg = read_case("liver_pinned_grasp_1x", data=args.data)
    sim = load_case(cfg)
    sim.prepare()
    probe = GraspTargetProbe(sim.physics)
    ident = identity(sim.physics, dict(case=cfg, device="cuda:0", worlds=1), args.data)
    snapshot = HarnessSnapshot(sim.physics, ident, {})
    snapshot.load(old / "settled_snapshot.npz")
    examples = json.loads(
        (
            root
            / "recordings/liver/20261119_141840/physics_replay/successful_examples.json"
        ).read_text()
    )
    if args.matrix:
        development = json.loads(
            (root / "artifacts/liver-grasp-manual-teacher/results.json").read_text()
        )
        selected = {r["source_episode"] for r in development if r["status"] == "passed"}
        examples = [e for e in examples if e["episode"] in selected]
    (out / "frozen_examples.json").write_text(json.dumps(examples, indent=2))
    results = []
    for example in examples:
        target = example["particle_ids"][0]
        if not probe.eligible[target]:
            results.append(
                dict(
                    target=target,
                    status="excluded",
                    reason="patch_or_attachment_filter",
                )
            )
            continue
        snapshot.restore({})
        initial = probe.ic.actions.numpy()[0, 0]
        q = sim.scene.field("q", probe.body).numpy()[0, target]
        goal = (
            q
            + np.array(example["capture_root"])
            - np.array(example["capture_particle_positions"][0])
        )
        points = np.array(
            [
                initial[:3],
                initial[:3] + [0.005, 0, 0],
                initial[:3] - [0.005, 0, 0],
                goal,
                goal + [0, 0.01, 0],
            ]
        )
        frozen = dict(
            workspace=[(points.min(0) - 0.02).tolist(), (points.max(0) + 0.02).tolist()]
        )
        # Frozen per-example geometry; controller continuously follows the live material target.
        qualification.GraspTeacher = (
            lambda initial, workspace, dt: ManualGeometryTeacher(
                initial, workspace, example, dt
            )
        )
        for captured in (False, True) if args.matrix else (True,):
            sim.cuda_graph = captured
            sim.prepare()
            for offset in (0.0, -0.005, 0.005) if args.matrix else (0.0,):
                name = f"episode_{example['episode']}_{'captured' if captured else 'eager'}_x{offset:+.3f}"
                result = attempt(
                    sim, probe, snapshot, {}, target, offset, frozen, out / name
                )
                result["source_episode"] = example["episode"]
                results.append(result)
                print(
                    result["source_episode"],
                    target,
                    result["mode"],
                    offset,
                    result["status"],
                    result["failure"],
                    result["acquisition_size"],
                    flush=True,
                )
                (out / "results.json").write_text(json.dumps(results, indent=2))
    (out / "description.json").write_text(
        json.dumps(
            dict(
                stage="development, not frozen qualification",
                snapshot_hash=snapshot.hash,
                source="manual successful capture relative geometry",
                mechanics="unchanged",
                training="not started",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
