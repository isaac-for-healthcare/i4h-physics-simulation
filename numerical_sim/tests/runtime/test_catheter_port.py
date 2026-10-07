# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""F5: the workflow catheter's features ported to i4h_endoluminal, opt-in (the shipped cases keep omniendo's numerics):
``rod_direct.inertia: physical`` (each node's cylinder inertia instead of the unit one) and
``catheter_control.tip_mapping: polyline`` (the tip polyline turns by the commanded angle). The parity harness
(scripts/parity/catheter.py) compares them with the workflow's reference; these tests pin the mechanics."""

import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from i4h_isaaclab.composition import load_case

SEG, R, N = 0.4 / 120, 5.0e-4, 121
MASS = 7800.0 * math.pi * R * R * SEG  # the workflow's 0.5 mm steel-like wire, 2e-5 kg per node


def case(inertia="unit", tip_mapping="per_edge", gravity=(0.0, 0.0, -9.81), tip_edges=8, envs=1, mass=MASS):
    return {"sim": {"device": "cuda:0", "envs": envs, "dt": 1.0 / 120, "substeps": 1, "iterations": 1, "cuda_graph": False,
                    "gravity": list(gravity)},
            "assets": {"catheter": {"type": "rod", "start": [0.0, 0.0, 0.5], "direction": [1.0, 0.0, 0.0], "count": N,
                                    "segment_length": SEG, "radius": R, "mass": mass}},
            "systems": [{"type": "particles", "damping": 0.99},
                        {"type": "rod_direct", "id": "rod", "body": "catheter", "young": 1.0e9, "torsion": 3.846e8,
                         "bend": 4 * 0.1 * (math.pi * R**4 / 4) / SEG**2, "twist": 4 * 0.4 * (math.pi * R**4 / 2) / SEG**2,
                         "inertia": inertia},
                        {"type": "catheter_control", "rod": "rod", "track_length": 0.0, "tip_edges": tip_edges,
                         "tip_mapping": tip_mapping}]}


def system(sim, name):
    return next(s for s in sim.systems if type(s).__name__ == name)


def test_physical_inertia_lets_a_light_wire_bend():
    """With unit rotational inertia a 2e-5 kg-per-node wire cannot turn its frames and stays straight under gravity;
    with each node's cylinder inertia it sags like the workflow's (the parity harness puts the tip at z ~0.11)."""
    tips = {}
    for inertia in ("unit", "physical"):
        sim = load_case(case(inertia))
        for _ in range(240):
            sim.step()
        tips[inertia] = sim.scene.field("q", sim.scene.body("catheter")).numpy()[0, -1]
    assert abs(tips["unit"][2] - 0.5) < 1e-3, "unit inertia: rigid"
    assert tips["physical"][2] < 0.3, "physical inertia: the wire hangs"
    assert np.isfinite(tips["physical"]).all()


def test_cylinder_inertia_values():
    sim = load_case(case("physical"))
    local = system(sim, "RodDirect").inertia_local.numpy()
    assert np.all(local[0] == 0.0), "the locked root has none"
    np.testing.assert_allclose(local[1:, 0], 12.0 / (MASS * (3 * R * R + SEG * SEG)), rtol=1e-4)
    np.testing.assert_allclose(local[1:, 2], 2.0 / (MASS * R * R), rtol=1e-4)


def test_unit_inertia_is_the_default_and_structural():
    sim = load_case(case())
    rod = system(sim, "RodDirect")
    assert rod.inertia == "unit" and rod.inv_inertia.shape == (1,)
    rod.inertia = "physical"
    with pytest.raises(ValueError, match="fixed at construction"):
        sim.prepare()
    with pytest.raises(ValueError, match="'unit' or 'physical'"):
        load_case(case("cylinder"))


@pytest.mark.parametrize("mapping, per_edge", [("per_edge", lambda a, n: a / n),
                                               ("polyline", lambda a, n: math.sin(a / (2 * n - 1)))])
def test_tip_mapping(mapping, per_edge):
    sim = load_case(case(tip_mapping=mapping, gravity=(0.0, 0.0, 0.0)))
    cc = system(sim, "CatheterControl")
    cc.actions.assign(np.float32([[0.0, 0.0, 0.6]]))  # tip rate 0.6 rad/s at rates[2] = 1
    for _ in range(60):
        sim.step()
    angle = float(cc.targets.numpy()[0][2])
    offsets = cc.controls.bend_offset.numpy()[0]
    np.testing.assert_allclose(offsets[-8:, 0], per_edge(angle, 8), rtol=1e-5)
    assert np.all(offsets[:-8] == 0.0)
    with pytest.raises(ValueError, match="tip_mapping"):
        load_case(case(tip_mapping="arc"))


def test_parity_compare_of_identical_runs_passes():
    path = Path(__file__).parents[2] / "scripts" / "parity" / "catheter.py"
    spec = importlib.util.spec_from_file_location("parity_catheter", path)
    parity = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = parity  # dataclasses resolve their module
    spec.loader.exec_module(parity)
    rod = parity.Rod(segments=12, length=0.04)
    data = {"rod": json.dumps(dataclasses.asdict(rod))}
    rng = np.random.default_rng(0)
    for name in ("sag", "insertion", "tip_bend", "twist"):
        for suffix in ("", "@half"):
            x = np.linspace(0.0, 0.04, 13)
            frames = np.stack([np.stack([x, 0.001 * np.sin(x * t), 0.5 + 0.001 * np.cos(x * t)], 1) for t in range(1, 21)])
            data[f"{name}{suffix}/positions"] = frames + rng.normal(0, 1e-9, frames.shape)
            data[f"{name}{suffix}/reaction"] = np.tile([0.0, 0.0, -0.02], (20, 1))
    rows = parity.compare(data, dict(data))
    assert all(r["pass"] is not False for r in rows) and sum(r["pass"] is True for r in rows) == 14


def test_parity_tube_bend_keeps_ours_in_the_lumen():
    """The parity harness's bent lumen (ours side): inserted 0.15 m, the wire's tip turns the corner and every node stays
    inside the 2 mm lumen (one-sided containment, the reference's settings)."""
    path = Path(__file__).parents[2] / "scripts" / "parity" / "catheter.py"
    spec = importlib.util.spec_from_file_location("parity_catheter_tube", path)
    parity = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = parity
    spec.loader.exec_module(parity)
    rod = parity.Rod()
    lumen = parity.tube("tube_bend")
    assert np.allclose(lumen[-1], [0.5, 0.35, 0.5]) and parity.tube("sag") is None
    gravity, _, command = parity.scenarios(rod)["tube_bend"]
    frames, _ = parity.run_ours(rod, gravity, 360, command, lumen=lumen)
    q = frames[-1]
    seg = lumen[1:] - lumen[:-1]
    t = np.clip(((q[:, None] - lumen[None, :-1]) * seg[None]).sum(-1) / (seg**2).sum(-1)[None], 0.0, 1.0)
    gap = np.linalg.norm(q[:, None] - (lumen[None, :-1] + t[..., None] * seg[None]), axis=-1).min(axis=1)
    assert gap.max() <= parity.TUBE_RADIUS - rod.radius + 2e-4, gap.max()
    assert q[-1, 1] > 0.05, "the tip turned into the +y leg"



def test_physical_inertia_follows_each_envs_mass():
    """Physical inertia is per environment: one geometry variant may carry different particle masses per env, and
    each env's frames then rotate with its own nodes' inertia, like an independently built rod of that mass."""
    batch = load_case(case("physical", envs=2))
    body = batch.scene.body("catheter")
    inv_mass = batch.scene.field("inv_mass", body)
    rows = inv_mass.numpy()
    rows[1] = rows[1] / 10.0  # env 1's nodes ten times heavier (the locked root stays at 0)
    inv_mass.assign(rows)
    batch.prepare()  # the physical inertia is recomputed from each env's rows
    local = system(batch, "RodDirect").inertia_local.numpy().reshape(2, N, 3)
    np.testing.assert_allclose(local[1, 1:], local[0, 1:] / 10.0, rtol=1e-5)
    heavy = load_case(case("physical", mass=MASS * 10.0))
    for _ in range(120):
        batch.step()
        heavy.step()
    q = batch.scene.field("q", body).numpy()
    np.testing.assert_allclose(q[1], heavy.scene.field("q", heavy.scene.body("catheter")).numpy()[0], atol=1e-5)
