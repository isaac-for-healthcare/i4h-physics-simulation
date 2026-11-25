# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check complete loading traces and tabulate nose forces at matched displacements."""

import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def summarize(row, solver):
    path = HERE / "traces" / solver / (row["slug"] + ".csv")
    trace = np.genfromtxt(path, names=True, delimiter=",")
    if solver == "i4h":
        x, y = trace["displacement_mm"], trace["force_N"]
        gap = trace["guide_gap_mm"]
        support = trace["left_y_N"] + trace["right_y_N"]
    else:
        x, y = trace["deflection_mm"], trace["nose_reaction_N"]
        gap = trace["max_guide_gap_mm"]
        support = trace[
            "normal_support_vertical_N" if solver == "sofa" else "support_reaction_N"
        ]
    assert len(x) == 939 and np.all(np.diff(x) > 0) and abs(x[-1] - 5) < 1e-5, path
    assert all(np.isfinite(trace[k]).all() for k in trace.dtype.names), path
    if solver == "vbd":
        assert np.max(trace["max_abs_guide_offset_mm"]) < 60 / 27 / 2, path
    initial = x <= 0.3
    slope = np.polyfit(x[initial], y[initial], 1)[0]
    return dict(
        device=row["device"],
        solver=solver,
        initial_slope_N_mm=slope,
        difference_from_beam_pct=100 * (slope / float(row["beam_slope_N_mm"]) - 1),
        force_at_4mm_N=np.interp(4.0, x, y),
        force_at_5mm_N=y[-1],
        max_guide_gap_mm=max(gap),
        max_nose_support_difference_N=max(abs(y - support)),
        samples=len(x),
    )


if __name__ == "__main__":
    with (HERE / "inputs.csv").open() as f:
        rows = [
            summarize(row, solver)
            for row in csv.DictReader(f)
            for solver in ("i4h", "sofa", "vbd")
        ]
    assert len(rows) == 12
    with (HERE / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
