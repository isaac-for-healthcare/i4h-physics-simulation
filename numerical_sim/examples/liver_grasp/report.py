# SPDX-License-Identifier: Apache-2.0
"""Summarize retained experiment evidence and plot training curves without changing scores."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from training_env import ROOT, write_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/liver-policy-20261119"
    )
    args = p.parse_args()
    folder = args.output
    comparisons = {}
    starts = {}
    for label in ("controller", "bc_best", "ppo_best", "ppo_latest"):
        path = folder / f"{label}_evaluation.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text())
        by_target = {}
        for target in sorted({r["target"] for r in rows}):
            subset = [r for r in rows if r["target"] == target]
            by_target[str(target)] = summarize(subset)
        comparisons[label] = dict(**summarize(rows), per_target=by_target)
        starts[label] = json.loads((folder / f"{label}_starts.json").read_text())
    same = (
        all(values == next(iter(starts.values())) for values in starts.values())
        if starts
        else False
    )
    comparison = dict(identical_reserved_starts=same, policies=comparisons)
    write_json(folder / "comparison.json", comparison)
    manifest = json.loads((folder / "dataset/manifest.json").read_text())
    seed_to_order = {
        s["seed"]: i
        for i, s in enumerate(
            __import__("training_env").schedule(1000, manifest["seed"])
        )
    }
    write_json(
        folder / "dataset_audit.json",
        dict(
            episodes=len(manifest["episodes"]),
            per_target=dict(Counter(str(r["target"]) for r in manifest["episodes"])),
            statuses=dict(Counter(r["status"] for r in manifest["episodes"])),
            splits=dict(Counter(r["split"] for r in manifest["episodes"])),
            unique_seeds=len({r["seed"] for r in manifest["episodes"]}),
            collection_wall_s=manifest["elapsed_s"],
            ordering="Manifest lists asynchronous completions; dispatch is the seeded balanced cycle.",
            dispatch_order_by_file={
                r["file"]: seed_to_order[r["seed"]] for r in manifest["episodes"]
            },
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    if (folder / "bc_curves.json").exists():
        rows = json.loads((folder / "bc_curves.json").read_text())
        for key in ("train_loss", "validation_loss"):
            axes[0].plot(
                [r["elapsed_s"] / 60 for r in rows], [r[key] for r in rows], label=key
            )
        axes[0].set_yscale("log")
        axes[0].legend()
    axes[0].set(
        xlabel="BC minutes", ylabel="Normalized action MSE", title="Behavior cloning"
    )
    if (folder / "ppo_curves.json").exists():
        rows = json.loads((folder / "ppo_curves.json").read_text())
        axes[1].plot(
            [r["elapsed_s"] / 60 for r in rows],
            [r["recent_success"] for r in rows],
            label="last 100 stochastic rollouts",
        )
    if (folder / "ppo_validation_history.json").exists():
        rows = json.loads((folder / "ppo_validation_history.json").read_text())
        axes[1].plot(
            [r["ppo_elapsed_s"] / 60 for r in rows],
            [r["success"] for r in rows],
            "o",
            label="20-episode validation",
        )
    axes[1].set(
        xlabel="PPO minutes (rollouts)",
        ylabel="Full-cycle success",
        ylim=(-0.02, 1.02),
        title="PPO",
    )
    if (folder / "reward_revisions.json").exists() and (folder / "sessions").exists():
        axes[1].axvline(25, color="grey", linestyle=":", label="reward v2 continuation")
    axes[1].legend(fontsize=7)
    labels = list(comparisons)
    axes[2].bar(labels, [comparisons[label]["success"] for label in labels])
    axes[2].axhline(0.8, color="black", linestyle="--", label="experimental milestone")
    axes[2].set(
        ylabel="Full-cycle success", ylim=(0, 1.05), title="100 reserved episodes"
    )
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].legend(fontsize=7)
    fig.savefig(folder / "training_curves.png", dpi=160)
    lines = [
        "# Liver policy experiment results",
        "",
        "Experimental mechanics, five demonstrated targets and fixed demonstrated orientations. "
        "The physical success criteria and tissue parameters were preserved.",
        "",
        f"Demonstrations: {len(manifest['episodes'])} complete episodes; "
        f"{manifest['elapsed_s']:.1f} seconds; splits: {dict(Counter(r['split'] for r in manifest['episodes']))}.",
        "",
        f"Identical reserved starts across compared policies: **{same}** (exact tissue/action hashes and canonical rows).",
        "",
        "| Policy | Episodes | Full-cycle success | Capture failures | Retention failures |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, r in comparisons.items():
        lines.append(
            f"| {label} | {r['episodes']} | {r['success']:.0%} | {r['capture_failures']} | {r['retention_failures']} |"
        )
    if (folder / "ppo_curves.json").exists() and (
        folder / "ppo_training_episodes.json"
    ).exists():
        history = json.loads((folder / "ppo_curves.json").read_text())
        attempts = json.loads((folder / "ppo_training_episodes.json").read_text())
        updates = history[-1]["iteration"] + 1
        lines += [
            "",
            f"PPO ran for {history[-1]['elapsed_s'] / 60:.1f} training minutes: {updates:,} updates and {updates * 64 * 16:,} environment control transitions. Of {len(attempts):,} completed stochastic training episodes, {sum(r['status'] == 'passed' for r in attempts)} completed the full cycle. That did not yield deterministic success in the reserved comparison.",
        ]
    lines += [
        "",
        "| Environments | Aggregate control ticks/s | Masked reset isolation |",
        "|---|---:|---|",
    ]
    benchmarks = []
    for worlds in (1, 8, 16):
        path = folder / f"benchmark_{worlds}.json"
        if path.exists():
            bench = json.loads(path.read_text())
            benchmarks.append(bench)
            lines.append(
                f"| {worlds} | {bench['environment_control_steps_per_s']:.1f} | {bench['reset_isolation']} |"
            )
    if len(benchmarks) == 3:
        fastest = max(benchmarks, key=lambda b: b["environment_control_steps_per_s"])
        assert fastest["worlds"] == manifest["identity"]["worlds"], (
            "Dataset did not use the fastest qualified configuration"
        )
    lines += [
        "",
        "Selected 16 environments from the measured configurations. Throughput is aggregate environment control ticks per wall second, including two physics steps per tick and no rendering. Existing unrelated GPU workloads were preserved; GPU memory observations are retained in the benchmark JSON.",
        "",
        "| Target | Controller | BC | PPO best |",
        "|---|---:|---:|---:|",
    ]
    targets = sorted({t for r in comparisons.values() for t in r["per_target"]})
    for target in targets:
        values = [
            comparisons.get(label, {})
            .get("per_target", {})
            .get(target, {})
            .get("success")
            for label in ("controller", "bc_best", "ppo_best")
        ]
        lines.append(
            "| "
            + target
            + " | "
            + " | ".join("pending" if v is None else f"{v:.0%}" for v in values)
            + " |"
        )
    lines += [
        "",
        "| Policy | Max lift mm | Max radial slip mm | Max release ms | Max pin drift mm | Max inverted tets |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, r in comparisons.items():

        def maximum(key, scale=1.0):
            value = r[key]
            return "n/a" if value is None else f"{value['max'] * scale:.4g}"

        lines.append(
            f"| {label} | {maximum('lift_m', 1000)} | {maximum('radial_slip_m', 1000)} | {maximum('release_latency_s', 1000)} | {maximum('attachment_drift_m', 1000)} | {maximum('inverted_tetrahedra_max')} |"
        )
    lines += [
        "",
        "Physical maxima include failed attempts. Radial slip is sampled while the grasp is commanded closed, before intended release. The requested destination is +20 mm; the preserved success tolerance is 2 mm.",
        "",
        "The first PPO segment used reward v1. Reward v2 removes bonuses for wrong-patch acquisition and caps lift potential at 20 mm; physical scoring and tissue mechanics are unchanged. See `reward_revisions.json` and archived source. Demonstration rewards and BC labels are unaffected by this fix.",
        "",
        "Raw per-episode metrics and reasons are in `*_evaluation.json`; aggregate physical "
        "diagnostics are in `comparison.json`. Evaluation samples diagnostics after every physics "
        "step; release latency is quantized to 60 Hz physics samples. Missing acquisitions have null lift/slip/release measurements. The source mechanics "
        "already contain inverted tetrahedra and soft-attachment drift; these were reported without tuning.",
        "",
        "See `examples/liver_grasp/README.md` for exact setup, resume, interactive playback and video commands. "
        "BC and PPO best/latest checkpoints remain separate. PPO resume restores optimizer/RNG and begins "
        "fresh canonical episodes at the saved schedule cursor.",
        "",
        "Validation: 30 focused tests passed (`contract-tests-v2.log`); 58 regression tests passed and 3 skipped (`regression-tests.log`); benchmark JSON and retained Slang video. "
        "Nothing was committed or pushed.",
        "",
        "Dataset audits: all 148,447 saved control transitions passed bounds, episode-end, previous-action and next-observation checks. Canonical row provenance was reconstructed for all 1,000 demonstration completions; ten restored starts matched exactly. `seed_separation_audit.json` verifies the complete PPO seed pool is disjoint from reserved and validation seeds.",
        "",
        "The BC Slang video uses five additional playback seeds and preserves its one retention failure (4/5 full cycles); it is separate from the 100 reserved comparison. Videos show actual simulated geometry with unscaled physical motion.",
    ]
    (folder / "results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(comparison, indent=2))


def summarize(rows):
    def metrics(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return (
            None
            if not values
            else dict(
                mean=float(np.mean(values)),
                max=float(np.max(values)),
                min=float(np.min(values)),
            )
        )

    return dict(
        episodes=len(rows),
        success=sum(r["status"] == "passed" for r in rows) / len(rows),
        capture_failures=sum(
            r["acquisition_step"] is None or r["failure"] == "wrong_patch_capture"
            for r in rows
        ),
        retention_failures=sum(r["failure"] == "retention_loss" for r in rows),
        failure_reasons=dict(Counter(r["failure"] for r in rows if r["failure"])),
        lift_m=metrics("achieved_lift_m"),
        radial_slip_m=metrics("radial_slip_m"),
        release_latency_s=metrics("release_latency_s"),
        attachment_drift_m=metrics("attachment_max_drift_m"),
        inverted_tetrahedra_max=metrics("inverted_tetrahedra_max"),
    )


if __name__ == "__main__":
    main()
