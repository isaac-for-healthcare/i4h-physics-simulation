# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recover asynchronous demonstration row provenance and verify exact start observations."""

import argparse
import heapq
import json
from pathlib import Path

import numpy as np
from training_env import ROOT, LiverEnv, schedule, write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/liver-policy-20261119"
    )
    parser.add_argument(
        "--data", default=str(ROOT.parent.parent / "i4h-isaaclab/numerical_sim/data")
    )
    parser.add_argument("--verify-starts", action="store_true")
    args = parser.parse_args()
    folder = args.output / "dataset"
    manifest = json.loads((folder / "manifest.json").read_text())
    rows = manifest["episodes"]
    worlds = manifest["identity"]["worlds"]
    dispatched = schedule(len(rows), manifest["seed"])
    by_seed = {r["seed"]: r for r in rows}
    lengths = {}
    for r in rows:
        with np.load(folder / r["file"]) as episode:
            lengths[r["seed"]] = len(episode["observations"])
            assert lengths[r["seed"]] == (r["steps"] + 1) // 2
    # Each control tick reports newly finished rows in ascending row order. New starts
    # happen after that tick; no physics is advanced while selected rows are reset.
    queue = []
    for e, spec in enumerate(dispatched[:worlds]):
        heapq.heappush(queue, (lengths[spec["seed"]], e, e, 0))
    following = worlds
    provenance = []
    for record in rows:
        finish, e, index, start = heapq.heappop(queue)
        spec = dispatched[index]
        assert record["seed"] == spec["seed"], (
            f"completion/dispatch reconstruction failed at {record['file']}"
        )
        assert record["target_index"] == spec["target_index"]
        assert record["jitter_m"] == spec["jitter_m"]
        provenance.append(
            dict(
                file=record["file"],
                seed=spec["seed"],
                target_index=spec["target_index"],
                canonical_row=e,
                dispatch_index=index,
                start_control_tick=start,
                end_control_tick=finish,
                snapshot_hash=manifest["snapshot_hash"],
            )
        )
        if following < len(dispatched):
            next_spec = dispatched[following]
            heapq.heappush(
                queue, (finish + lengths[next_spec["seed"]], e, following, finish)
            )
            following += 1
    assert not queue and following == len(rows)
    write_json(
        folder / "provenance_rows.json",
        dict(
            method="Exact completion-order and control-length reconstruction from the seeded dispatcher",
            snapshot=f"../snapshots/settled_{worlds}.npz",
            episodes=provenance,
        ),
    )
    result = dict(
        episodes=len(rows),
        unique_dispatch_indices=len({r["dispatch_index"] for r in provenance}),
        canonical_rows=list(range(worlds)),
        dispatch_reconstruction="passed",
        exact_start_checks=[],
    )
    if args.verify_starts:
        env = LiverEnv(args.data, worlds, args.output / "snapshots", manifest["seed"])
        assert env.snapshot.hash == manifest["snapshot_hash"]
        chosen = []
        for target in range(5):
            chosen.extend([r for r in provenance if r["target_index"] == target][:2])
        for row in chosen:
            record = by_seed[row["seed"]]
            spec = {k: record[k] for k in ("seed", "target_index", "jitter_m")}
            obs = env.reset([row["canonical_row"]], [spec])["policy"][
                row["canonical_row"]
            ].numpy()
            with np.load(folder / row["file"]) as episode:
                np.testing.assert_array_equal(obs, episode["observations"][0])
            result["exact_start_checks"].append(
                dict(file=row["file"], canonical_row=row["canonical_row"], passed=True)
            )
    write_json(args.output / "dataset_row_provenance_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
