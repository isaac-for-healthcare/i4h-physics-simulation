# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run all 11 uncompensated 1 ms shafts and compare against the Harrison data."""

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent))
from harrison_inputs import SLOPES  # noqa: E402

out = here / "16ms_16substeps"
out.mkdir(exist_ok=True)
rows = []
with (here / "summary.csv").open() as f:
    paper = list(csv.DictReader(f))
with (here / "figure2_digitized.csv").open() as f:
    figure = list(csv.DictReader(f))
aligned_travel = max(float(point["published_travel_mm"]) for point in figure) - 0.55
for (slug, target), meta, modulus in zip(
    SLOPES.items(),
    paper,
    [9.5, 11.4, 12.3, 12.5, 14.5, 17.0, 29.2, 60.3, 65.4, 139.6, 158.4],
    strict=True,
):
    assert np.isclose(target, float(meta["target_N_mm"]))
    path = out / (slug + ".csv")
    subprocess.run(
        [
            sys.executable,
            here.parent / "guidewire_solver_bench.py",
            "--wire",
            slug,
            "--dt",
            ".016",
            "--substeps",
            "16",
            "--output",
            path,
        ],
        check=True,
    )
    trace = np.loadtxt(path, delimiter=",", skiprows=1)
    assert trace.shape == (939, 5) and np.isfinite(trace).all() and trace[-1, 0] == 5
    initial = trace[trace[:, 0] <= 0.3]
    slope = np.polyfit(initial[:, 0], initial[:, 1], 1)[0]
    force = np.interp(aligned_travel, trace[:, 0], trace[:, 1])
    points = [float(r["load_N"]) for r in figure if r["wire"] == meta["wire"]]
    rows.append(
        [
            meta["wire"],
            modulus,
            target,
            slope,
            100 * (slope / target - 1),
            abs(slope - target) <= float(meta["sd_N_mm"]),
            points[-1] if points else "",
            force,
            100 * (force / points[-1] - 1) if points else "",
            trace[:, 4].max(),
        ]
    )
with (out / "comparison.csv").open("w") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "wire",
            "effective_E_GPa",
            "paper_slope_N_mm",
            "simulated_slope_N_mm",
            "slope_error_percent",
            "inside_one_SD",
            "paper_individual_force_N",
            "sim_force_at_4p407mm_N",
            "individual_force_error_percent",
            "max_sampled_gap_mm",
        ]
    )
    writer.writerows(rows)
