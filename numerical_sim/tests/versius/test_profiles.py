# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from i4h_isaaclab.composition import read_case
from i4h_isaaclab.versius.profiles import profile


def test_original_chole_mechanics_and_support_are_preserved():
    original = read_case("chole_tet_med")
    selected = profile("chole")
    assert selected.case["assets"] == {
        k: v for k, v in original["assets"].items() if k in ("tissue", "cavity")
    }
    keep = {
        "particles",
        "tet",
        "surface",
        "attachments",
        "connectors",
        "tissue_contact",
        "bounds",
    }
    for system in original["systems"]:
        if system["type"] in keep or (
            system["type"] == "static_contact" and system.get("body") == "tissue"
        ):
            assert system in selected.case["systems"]
    assert selected.case["sim"]["substeps"] == 8
    assert selected.case["sim"]["iterations"] == 8
    assert np.isclose(np.linalg.norm(selected.pose[1]), 1.0)
    assert profile("chole").patch_ids == selected.patch_ids


def test_block_is_fixed_fixture_with_original_tet_iterations():
    selected = profile("block")
    assert selected.case["assets"]["block"]["fix_top"]
    assert selected.case["sim"]["substeps"] == 4
    assert selected.case["sim"]["iterations"] == 8
    assert not any(
        x["type"] in {"grasp", "cutting", "fluid", "thermal"}
        for x in selected.case["systems"]
    )


def test_life_size_geometry_support_materials_and_reference_settings():
    from i4h_isaaclab.composition import load_scene

    original = profile("chole")
    life = profile("chole-1x")
    full = load_scene(original.case, device="cpu", envs=1)
    small = load_scene(life.case, device="cpu", envs=1)
    a, b = full.body("tissue"), small.body("tissue")
    np.testing.assert_allclose(b.rest.q0, a.rest.q0 / 7, rtol=0, atol=1e-6)
    np.testing.assert_allclose(b.rest.radius, a.rest.radius / 7, rtol=1e-5)
    np.testing.assert_array_equal(b.rest.inv_mass, a.rest.inv_mass)
    np.testing.assert_array_equal(b.pin.index, a.pin.index)
    np.testing.assert_array_equal(b.tet.tets, a.tet.tets)
    np.testing.assert_array_equal(b.tet.spring, a.tet.spring)
    np.testing.assert_array_equal(b.tet.volume, a.tet.volume)
    np.testing.assert_allclose(b.connectors.rest, a.connectors.rest / 7, rtol=1e-6)
    np.testing.assert_allclose(
        small.static_meshes[0].points,
        full.static_meshes[0].points / 7,
        rtol=0,
        atol=1e-6,
    )
    assert life.geometry_scale == 1 / 7
    systems = {x.get("id", x["type"]): x for x in life.case["systems"]}
    assert systems["particles"]["max_velocity"] == 0.5
    assert systems["instrument_control"]["max_velocity"] == 0.3
    assert systems["static_contact"]["parts"] == ["liver", "fat", "gallbladder"]
    assert systems["static_contact"]["search"] == 0.00285714
    assert np.isclose(systems["tet"]["volume_epsilon"], 1e-8 / 7**4, rtol=1e-5, atol=0)


def test_life_size_volume_cutoff_preserves_original_active_constraints():
    from i4h_isaaclab.composition import load_scene

    def denominator(name):
        p = profile(name)
        body = load_scene(p.case, device="cpu", envs=1).body("tissue")
        q, w, t = body.rest.q0[0], body.rest.inv_mass[0], body.tet.tets[0]
        p0, p1, p2, p3 = [q[t[:, i]] for i in range(4)]
        gradients = [
            np.cross(p1 - p2, p3 - p2),
            np.cross(p2 - p0, p3 - p0),
            np.cross(p0 - p1, p3 - p1),
            np.cross(p1 - p0, p2 - p0),
        ]
        return sum(
            w[t[:, i]] * ((gradient / 6) ** 2).sum(1)
            for i, gradient in enumerate(gradients)
        )

    full, small = denominator("chole"), denominator("chole-1x")
    epsilon=next(s["volume_epsilon"] for s in profile("chole-1x").case["systems"] if s["type"]=="tet")
    np.testing.assert_array_equal(full > 1e-8, small > epsilon)
    assert (small > 1e-8).mean() < 0.2
