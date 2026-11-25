# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the CSV summaries and figure. Run with NumPy and Matplotlib."""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from harrison_inputs import SLOPES  # noqa: E402

SD = [0.028, 0.007, 0.016, 0.016, 0.013, 0.021, 0.021, 0.021, 0.027, 0.057, 0.033]
NAMES = [
    "Amplatz",
    "Fixed Core Heavy Duty",
    "Rosen Heavy Duty (12.3 GPa)",
    "Newton Heavy Duty",
    "Rosen Heavy Duty (14.5 GPa)",
    "Amplatz Stiff",
    "Amplatz Extra Stiff",
    "Amplatz Super Stiff",
    "Amplatz Ultra Stiff",
    "Backup Meier",
    "Lunderquist Extra Stiff",
]
# Centers of the five labeled traces at seven unobscured abscissae in the
# original 572 x 399 embedded JPEG (Figure 2, PDF page 3, image 0).
PIXEL_X = np.array([241, 290, 335, 382, 427, 475, 520])
PIXEL_Y = [
    [297.5, 289.5, 283.5, 276.5, 270.5, 263.5, 258.5],
    [275.5, 259.5, 247, 232.5, 219.5, 208, 199],
    [269, 252, 237.5, 222.5, 209.5, 196.5, 184.5],
    [222.5, 187.5, 158.5, 130.5, 105, 81, 61.5],
    [218.5, 179.5, 147.5, 114.5, 85.5, 59.5, 38.5],
]
travel = (PIXEL_X - 58) * 5 / 466
configs = [
    "250us_24e",
    "125us_24e",
    "125us_48e",
    "62p5us_24e",
    "62p5us_24e_rate",
    "100us_24e",
]


def trace(wire, config, endpoint):
    a = np.loadtxt(ROOT / "traces" / f"{wire}_{config}.csv", delimiter=",", skiprows=1)
    if not (
        np.isfinite(a).all()
        and np.isclose(a[-1, 0], endpoint)
        and np.all(np.diff(a[:, 0]) > 0)
    ):
        raise ValueError(f"Incomplete or nonfinite trace for {wire} {config}")
    return a


def save(name, rows):
    with (ROOT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    summary, curve_metrics, digitized = [], [], []
    fixture_checks = []
    with (ROOT / "fits.csv").open() as stream:
        fits = {
            (r["wire"], r["configuration"]): float(r["slope_N_mm"])
            for r in csv.DictReader(stream)
        }
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    for i, (wire, k) in enumerate(SLOPES.items()):
        row = dict(wire=NAMES[i], target_N_mm=k, sd_N_mm=SD[i])
        for c in configs:
            full = c in ("250us_24e", "100us_24e") or (i >= 6 and c == configs[3])
            a = trace(wire, c, 5 if full else 0.3)
            row[c] = fits[wire, c]
            fixture_checks.append(
                dict(
                    wire=NAMES[i],
                    configuration=c,
                    stroke_mm=a[-1, 0],
                    max_guide_gap_mm=a[:, 4].max(),
                    max_vertical_imbalance_N=np.max(
                        np.abs(a[:, 1] - a[:, 2:4].sum(axis=1))
                    ),
                    max_support_asymmetry_N=np.max(np.abs(a[:, 2] - a[:, 3])),
                )
            )
            if c == "100us_24e":
                row["force_at_5mm_N"] = a[-1, 1]
                row["max_guide_gap_mm"] = a[:, 4].max()
                row["max_force_imbalance_N"] = np.abs(
                    a[:, 1] - a[:, 2:4].sum(axis=1)
                ).max()
        # Last second of the held-load run, independent of the nose readout.
        held = trace(wire, "load_250us_24e", 5)
        window = held[held[:, 0] > 4]
        row.update(
            held_deflection_mm=window[:, 2].mean(),
            held_range_mm=np.ptp(window[:, 2]),
            left_N=window[:, 3].mean(),
            right_N=window[:, 4].mean(),
        )
        summary.append(row)
        if i < 6:
            continue
        ax = axes.flat[i - 6]
        measured = (318.5 - np.array(PIXEL_Y[i - 6])) * 16 / 311
        for x, f in zip(travel, measured):
            digitized.append(dict(wire=NAMES[i], published_travel_mm=x, load_N=f))
        ax.plot(travel - 0.55, measured, "ko", markersize=3, label="Figure 2")
        metric = dict(wire=NAMES[i], last_measured_N=measured[-1])
        for c, color, label in [
            ("250us_24e", "#b45d28", "250 µs"),
            ("100us_24e", "#418348", "100 µs"),
            ("62p5us_24e", "#18648a", "62.5 µs"),
        ]:
            a = trace(wire, c, 5)
            ax.plot(a[:, 0], a[:, 1], color=color, lw=1.3, label=label)
            for offset in [0.45, 0.55, 0.65]:
                pred = np.interp(travel - offset, a[:, 0], a[:, 1])
                metric[f"{c}_rmse_N_offset_{offset}"] = np.sqrt(
                    np.mean((pred - measured) ** 2)
                )
                if offset == 0.55:
                    metric[c + "_last_N"] = pred[-1]
        curve_metrics.append(metric)
        ax.set_title(NAMES[i])
        ax.set_xlabel("Deflection after contact (mm)")
        ax.set_ylabel("Force (N)")
        ax.grid(alpha=0.2)
        ax.set_xlim(0, 5)
        ax.set_ylim(bottom=0)
    ax = axes.flat[-1]
    low = (318.5 - np.array([305.5, 302.5, 300.5, 297.5, 294.5])) * 16 / 311
    high = (318.5 - np.array([301.5, 297.5, 295, 291, 288.5])) * 16 / 311
    ax.fill_between(
        travel[2:] - 0.55, low, high, color="gray", alpha=0.55, label="Figure 2 band"
    )
    band_rows = []
    for wire in list(SLOPES)[:6]:
        a = trace(wire, "100us_24e", 5)
        ax.plot(a[:, 0], a[:, 1], color="#418348", lw=0.9)
        pred = np.interp(travel[2:] - 0.55, a[:, 0], a[:, 1])
        for x, lo, hi, y in zip(travel[2:], low, high, pred):
            band_rows.append(
                dict(
                    wire=wire,
                    published_travel_mm=x,
                    group_low_N=lo,
                    group_high_N=hi,
                    simulated_N=y,
                )
            )
    ax.set(
        title="Six overlapping guidewires",
        xlabel="Deflection after contact (mm)",
        ylabel="Force (N)",
        xlim=(0, 5),
        ylim=(0, None),
    )
    ax.grid(alpha=0.2)
    axes.flat[0].legend(frameon=False)
    ax.legend(frameon=False)
    fig.suptitle("Guidewire shaft bending · 40 mm support span")
    fig.tight_layout()
    fig.savefig(ROOT / "figure2_comparison.png", dpi=180)
    save("summary.csv", summary)
    save("fixture_checks.csv", fixture_checks)
    save("figure2_metrics.csv", curve_metrics)
    save("figure2_digitized.csv", digitized)
    save("figure2_group.csv", band_rows)


if __name__ == "__main__":
    main()
