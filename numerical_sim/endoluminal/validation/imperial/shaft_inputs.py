# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recompute uniform shaft EI from the Cook P1 curve and Imperial axial profiles."""

import csv
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
curve = np.genfromtxt(root / "digitized/digitized_P1.csv", delimiter=",", names=True)
fit = curve[(curve["crosshead_mm"] >= 2) & (curve["crosshead_mm"] <= 5)]
slope = np.polyfit(fit["crosshead_mm"], fit["force_N"], 1)[0]
cook_ei = slope * 40**3 / 48
stations = {}
with (root / "measured/flexural_modulus_profiles_SI.csv").open() as stream:
    for row in csv.DictReader(stream):
        x = float(row["distance_from_tip_m"])
        if row["kind"] == "guidewire" and x >= 0.30:
            stations.setdefault(row["device"], []).append(
                (x, float(row["flexural_modulus_Pa"]) / 1e6)
            )
means = {}
for name, samples in stations.items():
    x, y = np.array(sorted(samples)).T
    means[name] = np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2) / (x[-1] - x[0])
with (root / "derived/uniform_shaft_parameters.csv").open("w") as stream:
    writer = csv.writer(stream)
    writer.writerow(
        [
            "device",
            "shaft_mean_MPa_from_0p30m",
            "ratio_to_Cook",
            "EI_N_mm2",
            "EI_N_m2",
            "basis",
        ]
    )
    for name, mean in means.items():
        ratio = mean / means["Cook Straight"]
        ei = cook_ei * ratio
        writer.writerow(
            [
                name,
                mean,
                ratio,
                ei,
                ei * 1e-6,
                "P1 calibrated"
                if name == "Cook Straight"
                else "estimated using common effective section",
            ]
        )
