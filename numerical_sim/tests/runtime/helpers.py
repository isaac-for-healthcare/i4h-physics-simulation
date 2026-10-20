# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers and skip markers shared by the test modules (test modules import from here, never from each other)."""

from __future__ import annotations

import copy
import importlib.util
import os
import time

import numpy as np
import pytest
import warp as wp
from i4h_endoluminal.systems.rods.catheter_control import CatheterControl
from i4h_isaaclab.composition import data_dir
from i4h_physics import Scene, Stepper
from i4h_physics.loaders import add_rod
from i4h_physics.scene import ACTIVE_INT
from i4h_physics.systems.mechanics.particles import Particles
from i4h_physics.systems.mechanics.surface import deactivate_tet
from i4h_physics.systems.mechanics.tet import Tet
from i4h_physics.systems.rods.rod_direct import RodDirect
from i4h_physics.systems.rods.rod_pbd import RodPbd
from i4h_surgical.systems.fluids.emitter import Emitter
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
from i4h_surgical.systems.mechanics.surface import Surface

# --- data and optional-dependency gates ---------------------------------------------------------------------------

needs_external_data = pytest.mark.skipif(not os.environ.get("I4H_DATA"), reason="external case requires I4H_DATA")

CHOLE = data_dir() / "choly-mid"
VESSELS = data_dir() / "vessels"
needs_chole = pytest.mark.skipif(not os.environ.get("I4H_DATA") and not CHOLE.is_dir(), reason=f"chole data not found at {CHOLE}")
needs_vessels = pytest.mark.skipif(not os.environ.get("I4H_DATA") and not (VESSELS / "AortaWithVesselsStatic.usdc").is_file(),
                                   reason=f"vessel meshes not found under {VESSELS}")
RENDER_TESTS = os.environ.get("I4H_RENDER_TESTS") == "1"
HAVE_SLANG = importlib.util.find_spec("slangpy") is not None
render_tests = pytest.mark.skipif(not RENDER_TESTS, reason="set I4H_RENDER_TESTS=1")  # offscreen Warp/Newton rendering
slang_tests = pytest.mark.skipif(not (RENDER_TESTS and HAVE_SLANG), reason="set I4H_RENDER_TESTS=1 with slangpy installed")

# --- systems and stepping -----------------------------------------------------------------------------------------

PUSH = np.array([0.0, 0.0, 0.045], dtype=np.float32)  # sphere (r 0.015) from below into the bottom face (z 0.05) of env 0


def find(sim, cls):
    return next(s for s in sim.systems if isinstance(s, cls))


def push_actions(control, envs):
    """(E, n, 10) actions: every env's probe at rest, env 0's pushed into the block; identity rotation, jaw open."""
    actions = np.zeros((envs, control.count, 10), np.float32)
    actions[:, :, :3] = control.rest[None]
    actions[:, :, 6] = 1.0
    actions[0, :, :3] = PUSH
    return actions


def rest_actions(control, envs):
    """(E, n, 10) actions: every tool at rest, identity rotation, jaw open."""
    actions = np.zeros((envs, control.count, 10), np.float32)
    actions[:, :, :3] = control.rest[None]
    actions[:, :, 6] = 1.0
    return actions


def run(sim, steps, actions=None):
    """Step ``steps`` times; ``actions`` is an (E, n, 10) array or a callable of time returning one."""
    control = find(sim, InstrumentControl)
    for step in range(steps):
        if callable(actions):
            control.actions.assign(actions(step * sim.dt))
        elif actions is not None:
            control.actions.assign(actions)
        sim.step()


def run_rates(sim, steps, rates):
    """Step with ``rates`` (E, 3) normalised catheter rates, or a callable of time returning them."""
    control = find(sim, CatheterControl)
    for k in range(steps):
        r = rates(k * sim.dt) if callable(rates) else rates
        control.actions.assign(np.ascontiguousarray(r, dtype=np.float32))
        sim.step()


def free_rod(envs=2, backend=RodDirect, tip_edges=8):
    scene = Scene("cuda:0", envs, (0.0, 0.0, 0.0))
    add_rod(scene, "rod", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], 33, 0.05, 0.02, mass=1.0)
    scene.allocate()
    rod = backend(scene, body="rod") if backend is RodPbd else backend(scene, body="rod", torsion=1.0e9, bend=0.1)
    ctl = CatheterControl(scene, rod=rod, track_length=1.0, tip_edges=tip_edges)
    return Stepper.create(scene, [Particles(scene, damping=0.99), rod, ctl], dt=1 / 60, substeps=8, iterations=1), rod, ctl


def quiet(sim):
    """Silence the case's emitter so a test owns every alive particle."""
    for s in sim.systems:
        if isinstance(s, Emitter):
            s.enabled_env.fill_(0.0)
    return sim


def wait_until(predicate, timeout: float = 10.0) -> None:
    deadline = time.perf_counter() + timeout
    while not predicate():
        if time.perf_counter() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.005)


# --- haptics (stub devices only) ----------------------------------------------------------------------------------


def haptic_case(tmp_path, envs=2, handle=False, **haptics):
    """A two_worlds copy whose probe is driven by a fake device pressing into the block, with a haptics: block; with ``handle``
    an instrument box (listed first) whose channel 0 the pad mounts."""
    import yaml
    from i4h_isaaclab.composition import resolve_case

    cfg = yaml.safe_load(resolve_case("two_worlds").read_text())
    cfg["sim"]["envs"] = envs
    cfg["instruments"]["probe"]["input"] = "pad"
    cfg["inputs"] = {"pad": {"backend": "fake", "settings": {"position": [0.0, 0.0, 0.05], "io_ms": 0.05}}}
    if handle:
        cfg["inputs"] = {"box": {"backend": "ibcontroller", "settings": {"channels": 2}}, **cfg["inputs"]}
        cfg["inputs"]["pad"]["settings"]["handle"] = {"box": "box", "channel": 0}
    cfg["haptics"] = {"enabled": True, "newtons_per_unit": 1e-3, "max_force_n": 0.5, "low_pass_hz": 0.0, "max_slew_n_per_s": 0.0,
                      "switch_ramp_s": 0.0, "stale_timeout_s": 0.1, **haptics}
    path = tmp_path / "two_worlds_haptic.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


# --- regional materials and the region workspace ------------------------------------------------------------------


def regional_case(tmp_path):
    """An aorta_centreline copy with regional catheter and vessel materials and a region workspace: the catheter uses
    the shared instrument regions, the vessel tree names branches 0-2 (the other branches stay unassigned)."""
    import yaml
    from i4h_isaaclab.composition import resolve_case

    cfg = yaml.safe_load(resolve_case("aorta_centreline").read_text())
    sidecar = tmp_path / "aorta_regions.yaml"
    sidecar.write_text(yaml.safe_dump({"schema_version": 1, "default": "unassigned",
                                       "branches": {0: "branch_0", 1: "branch_1", 2: "branch_2"}}))
    cfg["assets"]["catheter"]["regions"] = "${DATA}/vessels/instrument_regions.yaml"
    cfg["assets"]["aorta"]["centreline"]["regions"] = str(sidecar)
    materials = {"rod_direct": {"type": "rod_direct_material", "id": "catheter_material", "body": "catheter"},
                 "rod_pbd": {"type": "rod_pbd_material", "id": "vessel_material", "body": "aorta"}}
    systems = []
    for system in cfg["systems"]:  # each material runs before the rod it scales
        if system["type"] in materials:
            systems.append(materials[system["type"]])
        systems.append(system)
    cfg["systems"] = systems
    instrument = ("young_multiplier", "torsion_multiplier", "bend_multiplier", "twist_multiplier", "curvature_x", "curvature_y")
    cfg["ui"] = {
        "title": "Regional vessel and instrument properties",
        "region_bodies": ["aorta", "catheter"],
        "groups": [
            {"title": "Vessel regions",
             "knobs": [["vessel_material", f"region_{name}"] for name in ("stretch", "bend", "twist")]},
            {"title": "Instrument regions", "knobs": [["catheter_material", f"region_{name}"] for name in instrument]},
            {"title": "Catheter controls",
             "knobs": [["rod", "young"], ["rod", "torsion"], ["catheter_control", "track_stiffness"], ["catheter_control", "rates"]]},
            {"title": "Containment",
             "knobs": [["lumen_containment", "max_distance"], ["lumen_containment", "vessel_response"]],
             "readouts": [["lumen_containment", name] for name in ("contact_depth", "contact_count", "candidate_overflow")]},
        ],
        "actions": ["pause", "step", "reset_env", "reset_all", "restore_defaults", "save", "load", "camera"],
    }
    path = tmp_path / "aorta_regions_case.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


# --- deterministic cases (the batched-vs-single oracle) ---------------------------------------------------------

STOCHASTIC = {"emitter", "smoke", "bleeding", "suction"}
SPACER = {"type": "rod", "start": [0.35, 0.3, 0.2], "direction": [0.0, 1.0, 0.0], "count": 7,
          "segment_length": 0.02, "radius": 0.004, "mass": 0.01}


def deterministic(case: dict, spacer_before: str) -> dict:
    """Drop the stochastic systems and insert the spacer asset before ``spacer_before``."""
    case = copy.deepcopy(case)
    case["systems"] = [s for s in case["systems"] if s["type"] not in STOCHASTIC]
    assets = {}
    for name, asset in case["assets"].items():
        if name == spacer_before:
            assets["spacer"] = dict(SPACER)
        assets[name] = asset
    case["assets"] = assets
    return case


# --- geometry and topology ---------------------------------------------------------------------------------------


def tet_volumes(q, tets):
    p0, p1, p2, p3 = (q[tets[:, k]] for k in range(4))
    return np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0


@wp.kernel
def signed_phi(mesh: wp.uint64, q: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.float32)):
    """omniendo's phi with sign_scale -1: a node inside the lumen with radius clearance has phi <= -radius."""
    i = wp.tid()
    r = wp.mesh_query_point_sign_normal(mesh, q[i], 10.0)
    out[i] = -wp.length(q[i] - wp.mesh_eval_position(mesh, r.face, r.u, r.v)) * r.sign


@wp.kernel
def kill_tet(
    env: wp.int32,
    t: wp.int32,
    variant: wp.array(dtype=wp.int32),
    count: wp.int32,
    tets: wp.array2d(dtype=wp.vec4i),
    tet_edges: wp.array3d(dtype=wp.int32),
    active: wp.array2d(dtype=wp.int32),
    edge_active: wp.array2d(dtype=wp.int32),
    support: wp.array(dtype=wp.int32),
    edge_support: wp.array2d(dtype=wp.int32),
    flags: wp.array2d(dtype=wp.int32),
    deleted: wp.array(dtype=wp.int32),
    dirty: wp.array(dtype=wp.int32),
):
    deactivate_tet(env, t, variant[env], count, tets, tet_edges, active, edge_active, support, edge_support, flags,
                   deleted, dirty)


def kill(sim, surface, env, t):
    """Remove one tet the way the deleters do (thermal, cutting and stretch breaking share deactivate_tet)."""
    b = sim.scene
    body = surface.body
    wp.launch(kill_tet, dim=1, inputs=[int(env), int(t), surface.variant, body.count, surface.tets, surface.tet_edges],
              outputs=[surface.active, surface.edge_active, surface.support, surface.edge_support, b.field("flags", body),
                       surface.deleted, surface.dirty], device=b.device)


def live_surface_is_consistent(sim, env):
    scene, body = sim.scene, sim.scene.bodies[0]
    surface, tet = find(sim, Surface), find(sim, Tet)
    tris, tri_tet = surface.tris.numpy()[env], surface.tri_tet.numpy()[env]
    active = tet.active.numpy()[env]
    flags = scene.view(scene.flags, body).numpy()[env]
    live = tri_tet >= 0
    assert active[tri_tet[live]].all()
    assert (flags[tris[live].reshape(-1)] & ACTIVE_INT).all()
    assert (tris[~live][:, 0] == tris[~live][:, 1]).all()
    support = surface.support.numpy().reshape(scene.envs, -1)[env]
    assert support.sum() == 4 * active.sum() and np.array_equal(support > 0, (flags & ACTIVE_INT) != 0)
