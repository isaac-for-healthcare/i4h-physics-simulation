# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize the two native backend campaigns using the existing Harrison digitization."""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from harrison_inputs import SLOPES  # noqa: E402

REFERENCE = ROOT / "harrison_i4h"
BACKENDS = {
    "SOFA": ROOT / "harrison_sofa",
    "Newton VBD": ROOT / "harrison_vbd",
}


def records(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    paper = records(REFERENCE / "summary.csv")
    figure = records(REFERENCE / "figure2_digitized.csv")
    group = records(REFERENCE / "figure2_group.csv")
    grid, axes = plt.subplots(2, 3, figsize=(12, 7.3), layout="constrained")
    for backend, folder in BACKENDS.items():
        results = []
        for i, (wire, k) in enumerate(SLOPES.items()):
            meta = paper[i]
            assert np.isclose(k, float(meta["target_N_mm"]))
            a = np.genfromtxt(
                folder / "traces" / f"{wire}.csv", names=True, delimiter=","
            )
            x, y = a["deflection_mm"], a["nose_reaction_N"]
            assert (
                len(x) == 939
                and x[-1] >= 4.9999
                and np.all(np.diff(x) > 0)
                and np.isfinite(y).all()
            )
            window = x <= 0.3
            slope = np.polyfit(x[window], y[window], 1)[0]
            force = np.interp(4.407081545, x, y)
            support = a[
                "normal_support_vertical_N"
                if backend == "SOFA"
                else "support_reaction_N"
            ]
            row = dict(
                wire=meta["wire"],
                paper_slope_N_mm=k,
                measured_slope_N_mm=slope,
                initial_error_pct=100 * (slope / k - 1),
                within_paper_sd=abs(slope - k) <= float(meta["sd_N_mm"]),
                force_at_4p407mm_N=force,
                paper_individual_force_N=None,
                force_error_pct=None,
                seven_point_rmse_N=None,
                peak_guide_gap_mm=a["max_guide_gap_mm"].max(),
                peak_nose_support_difference_N=np.abs(y - support).max(),
            )
            controlled = np.genfromtxt(
                folder / "force_control" / f"{wire}.csv", names=True, delimiter=","
            )
            row["force_control_final_mm"] = controlled["deflection_mm"][-1]
            row["force_control_final_applied_N"] = controlled["applied_force_N"][-1]
            if backend == "SOFA":
                assert controlled["deflection_mm"][-1] >= 5
                row["force_control_at_4p407mm_N"] = np.interp(
                    4.407081545,
                    controlled["deflection_mm"],
                    controlled["applied_force_N"],
                )
            else:
                refined = np.genfromtxt(
                    folder / "refinement_32_iterations" / f"{wire}.csv",
                    names=True,
                    delimiter=",",
                )
                assert len(refined) == 939 and refined["deflection_mm"][-1] >= 4.9999
                window32 = refined["deflection_mm"] <= 0.3
                slope32 = np.polyfit(
                    refined["deflection_mm"][window32],
                    refined["nose_reaction_N"][window32],
                    1,
                )[0]
                row["slope_32_iterations_N_mm"] = slope32
                row["within_paper_sd_32_iterations"] = abs(slope32 - k) <= float(
                    meta["sd_N_mm"]
                )
            ax = axes.flat[i - 6 if i >= 6 else 5]
            ax.plot(
                x,
                y,
                color="#177463" if backend == "SOFA" else "#864f9d",
                linestyle="-" if backend == "SOFA" else "--",
                linewidth=1.4,
                label=backend if i >= 6 or i == 0 else None,
            )
            if i >= 6:
                pts = [r for r in figure if r["wire"] == meta["wire"]]
                xx = np.array([float(r["published_travel_mm"]) - 0.55 for r in pts])
                yy = np.array([float(r["load_N"]) for r in pts])
                row.update(
                    paper_individual_force_N=yy[-1],
                    force_error_pct=100 * (force / yy[-1] - 1),
                    seven_point_rmse_N=np.mean((np.interp(xx, x, y) - yy) ** 2) ** 0.5,
                )
                if backend == "SOFA":
                    ax.scatter(
                        xx, yy, s=16, color="black", label="Harrison Fig. 2", zorder=3
                    )
                ax.set_title(meta["wire"])
            results.append(row)
        with (folder / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    band = [r for r in group if r["wire"] == "amplatz"]
    axes.flat[5].fill_between(
        [float(r["published_travel_mm"]) - 0.55 for r in band],
        [float(r["group_low_N"]) for r in band],
        [float(r["group_high_N"]) for r in band],
        color="gray",
        alpha=0.35,
        label="Paper group",
    )
    axes.flat[5].set_title("Six overlapping guidewires")
    for ax in axes.flat:
        ax.set(
            xlim=(0, 5.05),
            ylim=(0, None),
            xlabel="Midpoint deflection (mm)",
            ylabel="Force (N)",
        )
        ax.grid(alpha=0.2)
    axes.flat[0].legend(frameon=False)
    axes.flat[5].legend(frameon=False)
    grid.suptitle("Displacement-driven bending · 40 mm span · 1 ms solver step")
    grid.savefig(BACKENDS["Newton VBD"] / "backend_comparison.png", dpi=180)


if __name__ == "__main__":
    main()
