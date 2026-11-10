# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Physical topology survives rebind, including cut slots and pending work."""
import newton
import numpy as np
import pytest
from helpers import find, kill
from i4h_isaaclab.composition import maps, read_case, sim_settings
from i4h_physics import Stepper, author_case
from i4h_physics.surface_storage import STATE_FIELDS
from i4h_physics.systems.mechanics.surface import Surface
from i4h_standalone.render.snapshots import Snapshot


def state_bytes(state):
    return {key: getattr(state.i4h, "surface_"+key).numpy().tobytes() for key in STATE_FIELDS}


def faces(sf, e):
    tris, owners = sf.tris.numpy()[e], sf.tri_tet.numpy()[e]
    return sorted((int(owner), tuple(map(int, tri))) for tri, owner in zip(tris, owners, strict=True) if owner >= 0)


@pytest.mark.parametrize("model_kind", ["pbd", "xpbd"])
@pytest.mark.parametrize("capture", [False, True])
def test_rebinding_cut_slots_pending_dirty_and_overflow(model_kind, capture):
    cfg, loaders, constructors = maps(read_case("two_worlds"))
    next(s for s in cfg["systems"] if s["type"] == "tet")["model"] = model_kind
    a = author_case(cfg, loaders, constructors)
    b = newton.ModelBuilder()
    for e in range(a.scene.envs):
        b.begin_world(gravity=a.scene.gravity_vector)
        a.add_world(b, e)
        b.end_world()
    a.complete(b)
    model = b.finalize(device="cuda:0")
    model.particle_grid = None
    state, other = model.state(), model.state()
    untouched, templates = state_bytes(other), state_bytes(model)
    control = model.control()
    scene, systems = a.bind(model, state, control)
    sim = Stepper.create(scene, systems, control=control, **sim_settings(cfg, False))
    sim.cuda_graph = capture
    sim.prepare()
    sf = find(sim, Surface)
    # Emit one actual cut before binding; cut slot identity stays authored -1.
    owner_before = model.i4h.surface_face_owner.numpy().tobytes()
    t = int(np.flatnonzero(sf.active.numpy()[0])[7])
    kill(sim, sf, 0, t)
    sf.step_end(scene, sim.dt)
    assert sf.cut_count.numpy()[0].sum() > 0
    before = state_bytes(state)
    rebound_scene, rebound_systems = a.bind(model, state, control)
    rebound = Stepper.create(rebound_scene, rebound_systems, control=control, **sim_settings(cfg, False))
    rebound.cuda_graph = capture
    rebound.prepare()
    assert before == state_bytes(state)
    fresh = find(rebound, Surface)
    assert fresh.tris.ptr == sf.tris.ptr
    assert fresh.tri_tet.ptr == sf.tri_tet.ptr
    snapshot = Snapshot(rebound)
    assert fresh in snapshot.tris
    # Deletion remains pending across binding. Existing overflow stays latched.
    t = int(np.flatnonzero(sf.active.numpy()[0])[8])
    kill(sim, sf, 0, t)
    sf.overflow.fill_(1)
    sf.cut_capacity.fill_(0)  # a valid live Model edit; count still exceeds capacity
    pending = state_bytes(state)
    third_scene, third_systems = a.bind(model, state, control)
    third = Stepper.create(third_scene, third_systems, control=control, **sim_settings(cfg, False))
    third_sf = find(third, Surface)
    assert pending == state_bytes(state) and third_sf.dirty.numpy()[0] == 1
    # Independent selected State starts with the same physical pending history.
    comparison = model.state()
    for key in STATE_FIELDS:
        getattr(comparison.i4h, "surface_"+key).assign(getattr(state.i4h, "surface_"+key))
    for key in ("active", "edge_active"):
        getattr(comparison.i4h, "tet_"+key).assign(getattr(state.i4h, "tet_"+key))
    ref_scene, ref_systems = a.bind(model, comparison, model.control())
    ref = next(s for s in ref_systems if isinstance(s, Surface))
    third_sf.step_end(third_scene, sim.dt)
    ref.step_end(ref_scene, sim.dt)
    assert faces(third_sf, 0) == faces(ref, 0)
    np.testing.assert_array_equal(third_sf.cut_count.numpy(), ref.cut_count.numpy())
    assert third_sf.cut_count.numpy()[0].sum() > 0
    np.testing.assert_array_equal(third_sf.overflow.numpy(), 1)
    np.testing.assert_array_equal(third_sf.dirty.numpy(), 0)
    assert owner_before == model.i4h.surface_face_owner.numpy().tobytes()
    assert untouched == state_bytes(other) and templates == state_bytes(model)
    # Preparation/capture does not implicitly consume pending work. Execute the
    # existing scheduler only after history assertions to cover both solvers.
    rebound.step()
    assert untouched == state_bytes(other) and templates == state_bytes(model)
