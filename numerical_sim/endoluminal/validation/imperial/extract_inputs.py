# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract the Imperial modulus profiles and Cook Straight force curve in metres, pascals, millimetres and newtons."""

import csv
from pathlib import Path

import numpy as np
import xlrd
from PIL import Image

root = Path(__file__).resolve().parent
catheters = [
    "Cordis Angiographic 4 Fr",
    "Terumo Straight 5 Fr",
    "Cook Royal Flush 5 Fr",
    "Terumo Straight 4 Fr",
]
wires = [
    "Cook Straight",
    "Cook Safe-T-J Curved",
    "Cook Rosen Curved",
    "Amplatz Super Stiff",
    "Bentson",
    "Terumo Angled",
    "Terumo Stiff Angled",
]
stations = []
for kind, names, first_row in [("catheter", catheters, 7), ("guidewire", wires, 1)]:
    sheet = xlrd.open_workbook(root / "sources" / f"{kind}_modulus.xls").sheet_by_index(
        0
    )
    for col, name in enumerate(names, 1):
        for row in range(first_row, sheet.nrows):
            modulus = sheet.cell(row, col)
            if modulus.ctype == xlrd.XL_CELL_NUMBER:
                stations.append(
                    (kind, name, sheet.cell_value(row, 0) * 0.01, modulus.value * 1e9)
                )
assert len(stations) == 676
with (root / "measured/flexural_modulus_profiles_SI.csv").open(
    "w", newline=""
) as stream:
    writer = csv.writer(stream)
    writer.writerow(["kind", "device", "distance_from_tip_m", "flexural_modulus_Pa"])
    writer.writerows(stations)

# Figure 4 calibrates 90 horizontal pixels per mm and 191 vertical pixels per N.
# Follow the upper magenta P1 pixel in each column inside the plot axes.
pixels = np.asarray(
    Image.open(root / "sources/cook_straight_figure4.png").convert("RGB")
)
magenta = (pixels == [255, 0, 255]).all(axis=2)
curve = []
for x in range(79, 530):
    hits = np.flatnonzero(magenta[18:248, x])
    if hits.size:
        curve.append(((x - 79) / 90, (246 - (18 + hits[0])) / 191))
assert len(curve) == 388
np.savetxt(
    root / "digitized/digitized_P1.csv",
    curve,
    delimiter=",",
    header="crosshead_mm,force_N",
    comments="",
)
