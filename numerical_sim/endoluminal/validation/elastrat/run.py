# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run physical-scale Elastrat cases and compare insertion at CT arc lengths."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from i4h_endoluminal.config import load_case
from i4h_physics.mesh_utils import rotation_matrix_degrees

HERE = Path(__file__).resolve().parent
REF = HERE / "phantom/references"
SPACING = 0.0025
SPEED = 0.025


def reference_error(points, reference):
    """Physical-metre coordinates; select mass points beyond the reference entry plane.

    References are ordered distal tip to proximal entry. The outward plane normal
    averages the last eight proximal displacements. Empty selections have no RMS.
    """
    points, reference = np.asarray(points, float), np.asarray(reference, float)
    outward = np.sum(reference[-1] - reference[-min(len(reference), 9) : -1], axis=0)
    norm = np.linalg.norm(outward)
    if norm <= 1e-12:
        raise ValueError("reference has no usable entry tangent")
    selected = points[(points - reference[-1]) @ (outward / norm) < 0]
    if len(selected) == 0:
        return {"count": 0, "rms_mm": None}
    squared = np.sum((selected[:, None] - reference[None]) ** 2, axis=-1).min(axis=1)
    return {"count": len(selected), "rms_mm": float(np.sqrt(squared.mean()) * 1000)}


def make_case(row):
    cfg = yaml.safe_load((HERE / "base.yaml").read_text())
    cfg["sim"].update(dt=0.016, substeps=16, iterations=8)
    asset = cfg["assets"]["aorta"]
    scale = asset["scale"]
    asset.update(
        scale=1.0, translation=(np.array(asset["translation"]) / scale).tolist()
    )
    catheter = cfg["assets"]["catheter"]
    catheter.update(
        start=[-0.0175, 0.0, 0.05],
        segment_length=SPACING,
        radius=float(row["OD_mm"]) / 2000,
        mass=0.006 / 25,
    )
    p, rod, ctrl, wall = cfg["systems"]
    young = 4 * float(row["EI_N_mm2"]) * 1e-6 / SPACING**2
    p.update(damping=0.5)
    rod.update(
        young=young,
        bend=1.0,
        torsion=young / 1.3,
        inertia="physical",
        angular_damping=0.5,
    )
    ctrl.update(
        track_length=0.4775,
        rates=[SPEED, 1.5, 1.0],
        upper=[0.44, 1e9, 1.5],
        max_delta=0.0005,
    )
    wall.update(search=0.0045)
    cfg["render"]["camera"]["position"] = (
        np.array(cfg["render"]["camera"]["position"]) / scale
    ).tolist()
    cfg["render"]["camera"].update(near=0.001, far=5.0)
    case = HERE / row["group"] / (row["slug"] + ".yaml")
    case.parent.mkdir(parents=True, exist_ok=True)
    if row["group"] == "guidewires_profiled":
        configure_profile(cfg, row, case)
    case.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return case, asset


def arc(p):
    return np.linalg.norm(np.diff(p, axis=0), axis=1).sum()


def inserted(points, ref):
    # App entry plane, with a linearly interpolated crossing. Follow the distal,
    # contiguous part of the rod even if a more proximal part crosses elsewhere.
    outward = (ref[-1] - ref[-9:-1]).sum(axis=0)
    signed = (points - ref[-1]) @ outward
    if signed[-1] >= 0:
        return points[:0]
    outside = np.flatnonzero(signed >= 0)
    if not len(outside):
        return points
    i = outside[-1]
    u = signed[i] / (signed[i] - signed[i + 1])
    return np.vstack((points[i] + u * (points[i + 1] - points[i]), points[i + 1 :]))


def run(row):
    case, asset = make_case(row)
    sim = load_case(str(case), device="cpu", data=str(REF.parent))
    if row["group"] == "guidewires_profiled":
        from i4h_standalone.app.ui import load_overrides

        load_overrides(
            sim, yaml.safe_load(case.with_suffix(".overrides.yaml").read_text())
        )
    body = sim.scene.body("catheter")
    ctrl = next(s for s in sim.systems if hasattr(s, "track_length"))
    ctrl.actions.assign(np.array([[1.0, 0.0, 0.0]], np.float32))
    refs = {
        row["ct_prefix"] + str(i): np.load(REF / (row["ct_prefix"] + str(i) + ".npy"))
        for i in (1, 2, 3)
    }
    rot = rotation_matrix_degrees(asset["rotation_euler_degrees"])
    best = {}
    completed = set()
    for step in range(1, 1101):
        sim.step()
        points = (
            sim.scene.field("q", body).numpy()[0].astype(float) - asset["translation"]
        ) @ rot
        if not np.isfinite(points).all():
            raise RuntimeError(f"nonfinite rod: {case} frame {step}")
        for key, ref in refs.items():
            if key in completed:
                continue
            seg = inserted(points, ref)
            length = arc(seg)
            target = arc(ref)
            if len(seg) > 1 and (
                key not in best or abs(length - target) < abs(best[key][0] - target)
            ):
                best[key] = length, step, points.copy(), seg.copy()
            if length >= target:
                completed.add(key)
        if len(completed) == 3:
            break
    results = []
    for key, ref in refs.items():
        if key not in best:
            raise RuntimeError(f"{key}: no simulated points crossed entry")
        length, frame, points, seg = best[key]
        np.save(case.parent / (key + ".npy"), points)
        distances = np.linalg.norm(ref[:, None] - seg[None, :], axis=2).min(axis=1)
        result = dict(
            reference=key,
            matched=key in completed,
            frame=frame,
            feed_mm=frame * 0.016 * SPEED * 1000,
            reference_length_mm=arc(ref) * 1000,
            simulated_length_mm=length * 1000,
            tip_error_mm=np.linalg.norm(points[-1] - ref[0]) * 1000,
            **reference_error(points, ref),
            reverse_rms_mm=np.sqrt(np.mean(distances**2)) * 1000,
        )
        print(result, flush=True)
        results.append(result)
    return results


IMPERIAL = HERE.parent / "imperial"
WIRES = {
    "Cook Straight": "cookStraight",
    "Cook Safe-T-J Curved": "cookJ",
    "Cook Rosen Curved": "rosen",
    "Amplatz Super Stiff": "amplatz",
    "Bentson": "bentson",
    "Terumo Stiff Angled": "terumoStiff",
}


# Radius and straight terminal length in mm. Sweep angles are approximations
# from the unloaded planar drawing, with one common initial roll for all wires.
TIP_ARCS = {
    "cookJ": (3.75, 180.0, 3.0),
    "rosen": (2.75, 210.0, 2.7),
    "terumoStiff": (9.5, 90.0, 0.0),
}


def configure_profile(cfg, row, case):
    with (IMPERIAL / "measured/flexural_modulus_profiles_SI.csv").open() as stream:
        stations = sorted(
            (float(r["distance_from_tip_m"]), float(r["flexural_modulus_Pa"]))
            for r in csv.DictReader(stream)
            if r["device"] == row["device"]
        )
    x, modulus = np.array(stations).T
    edges = cfg["assets"]["catheter"]["count"] - 1
    regions, values = [], {}
    tip = TIP_ARCS.get(row["slug"])
    for i in range(edges):
        name = f"edge{i}"
        regions.append(dict(name=name, interval=[i / edges, (i + 1) / edges]))
        lo = (edges - i - 1) * SPACING
        ratio = float(np.interp(lo + SPACING / 2, x, modulus)) / (
            float(row["shaft_mean_MPa_from_0p30m"]) * 1e6
        )
        changes = dict(bend_multiplier=ratio, twist_multiplier=ratio)
        if tip:
            radius, degrees, terminal = tip
            start = terminal / 1000
            end = start + radius / 1000 * np.deg2rad(degrees)
            overlap = max(0.0, min(lo + SPACING, end) - max(lo, start))
            changes["curvature_x"] = float(overlap / SPACING / (radius / 1000))
        values[name] = changes
    cfg["assets"]["catheter"]["regions"] = dict(schema_version=1, regions=regions)
    cfg["systems"].insert(
        1, dict(type="rod_direct_material", id="material", body="catheter")
    )
    case.with_suffix(".overrides.yaml").write_text(
        yaml.safe_dump(
            dict(regions=dict(catheter=dict(material=values))), sort_keys=False
        )
    )


def inputs(group):
    if group == "catheters":
        with (IMPERIAL / "catheter_bending/inputs.csv").open() as stream:
            return [
                dict(r, group=group) for r in csv.DictReader(stream) if r["ct_prefix"]
            ]
    with (IMPERIAL / "derived/uniform_shaft_parameters.csv").open() as stream:
        measured = {r["device"]: r for r in csv.DictReader(stream)}
    return [
        dict(measured[name], group=group, OD_mm=0.889, slug=stem, ct_prefix=stem)
        for name, stem in WIRES.items()
    ]


def campaign(group):
    rows = inputs(group)
    with ProcessPoolExecutor(max_workers=len(rows)) as pool:
        results = [r for instrument in pool.map(run, rows) for r in instrument]
    with (HERE / group / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "group",
        choices=["guidewires", "guidewires_profiled", "catheters", "all"],
        default="all",
        nargs="?",
    )
    group = parser.parse_args().group
    for selected in (
        ["guidewires", "guidewires_profiled", "catheters"]
        if group == "all"
        else [group]
    ):
        campaign(selected)
