# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Interaction history (centreline, residue, hook, grasps, clips): lifecycle contracts that ownership must preserve."""
import numpy as np
import pytest
import warp as wp
from interaction_helpers import (
    HISTORY,
    LATER,
    MODES,
    Coupled,
    Episode,
    authored,
    copy_container,
    first_difference,
    history,
    poison,
    runtime,
    set_hook,
    snapshot,
    system,
)
from ownership_helpers import assert_native_alias

# What an episode reset clears; every other history array keeps its bytes (deliberately stale).
RESET = {
    ("Hook", "active"): 0, ("Hook", "verts"): -1, ("Hook", "tet"): -1,
    ("Grasp", "active"): 0, ("Grasp", "prim"): -1, ("Grasp", "count"): 0, ("Grasp", "pending"): 0,
    ("VolumetricGrasp", "particle"): -1, ("VolumetricGrasp", "active"): 0, ("VolumetricGrasp", "sphere"): 0,
    ("VolumetricGrasp", "count"): 0, ("VolumetricGrasp", "pending"): 0, ("VolumetricGrasp", "claim"): 0,
    ("Clipping", "node"): -1, ("Clipping", "active"): 0, ("Clipping", "count"): 0, ("Clipping", "clipped"): 0,
}
CENTRELINE = (("Centreline", "positions"), ("Centreline", "active_count"))


def rows(sim, key, env):
    """Every component of one world's rows (NumPy exposes vector dtypes as a trailing axis)."""
    return getattr(system(sim, key[0]), key[1]).numpy().reshape(sim.scene.envs, -1)[env]


@pytest.mark.parametrize("device,captured", MODES)
def test_episode_reset_clears_only_its_fields_in_selected_worlds(device, captured):
    cfg, a, m = authored(device)
    sim = runtime(cfg, a, m, captured)
    poison(sim)
    before = history(sim)
    centre = system(sim, "Centreline")
    rest, full = centre.rest.numpy()[centre.variant.numpy()[1]], centre.clamp_count.numpy()[centre.variant.numpy()[1]]
    sim.reset_physics(mask=wp.array([False, True], dtype=wp.bool, device=device))
    after = history(sim)
    for key in before:
        old = np.frombuffer(before[key], np.uint8).reshape(2, -1)
        new = np.frombuffer(after[key], np.uint8).reshape(2, -1)
        np.testing.assert_array_equal(new[0], old[0], err_msg=f"unselected world changed: {key}")
        if key in RESET:
            np.testing.assert_array_equal(rows(sim, key, 1), RESET[key], err_msg=str(key))
        elif key not in CENTRELINE:
            np.testing.assert_array_equal(new[1], old[1], err_msg=f"reset cleared a field it leaves stale: {key}")
    np.testing.assert_array_equal(centre.positions.numpy()[1], rest)
    np.testing.assert_array_equal(centre.active_count.numpy()[1], full)


@pytest.mark.parametrize("device,captured", MODES)
def test_cache_only_preparation_preserves_history(device, captured):
    cfg, a, m = authored(device)
    sim = runtime(cfg, a, m, captured)
    poison(sim)
    before = history(sim)
    sim.prepare()
    sim.prepare()
    assert history(sim) == before


@pytest.mark.parametrize("device,captured", MODES)
def test_disabled_hook_preparation_clears_its_anchor_flag_everywhere(device, captured):
    """A named exception: a disabled hook must release the contact suppression that aliases its entry anchor."""
    cfg, a, m = authored(device, disable=("hook",))
    sim = runtime(cfg, a, m, captured, prepare=False)
    poison(sim)
    before = history(sim)
    sim.prepare()
    after = history(sim)
    assert not system(sim, "Hook").active.numpy().any()
    assert not system(sim, "InstrumentContact").suppress_active.numpy().any()
    assert {k: v for k, v in after.items() if k != ("Hook", "active")} == {k: v for k, v in before.items() if k != ("Hook", "active")}


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_hook_entry_views_alias_the_anchor_arrays(device):
    cfg, a, m = authored(device)
    sim = runtime(cfg, a, m)
    hook, contact = system(sim, "Hook"), system(sim, "InstrumentContact")
    poison(sim)
    np.testing.assert_array_equal(contact.suppress_active.numpy(), hook.active.numpy()[:, 0])
    np.testing.assert_array_equal(contact.suppress_verts.numpy(), hook.verts.numpy()[:, 0])
    np.testing.assert_array_equal(hook.entry_active.numpy(), hook.active.numpy()[:, 0])
    np.testing.assert_array_equal(hook.entry_tet.numpy(), hook.tet.numpy()[:, 0])
    assert contact.suppress_active.ptr == hook.active.ptr and contact.suppress_verts.ptr == hook.verts.ptr
    assert system(sim, "Bleeding").clipped is system(sim, "Clipping").clipped


def test_history_set_is_the_inventoried_one():
    assert sum(len(v) for v in HISTORY.values()) == 36


def moved(sim, shift):
    """Translate every particle of the centreline's body in the bound State; returns the expected ring centres."""
    centre = system(sim, "Centreline")
    q = sim.scene.field("q", centre.body)
    host = q.numpy() + np.float32(shift)
    wp.copy(q, wp.array(host, dtype=wp.vec3, device=q.device))
    return centre.positions.numpy() + np.float32(shift)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_cache_refresh_recomputes_centreline_from_a_moved_state(device):
    cfg, a, m = authored(device)
    sim = runtime(cfg, a, m)
    expected = moved(sim, [0.01, 0.0, 0.0])
    sim.solver.reset(sim.state, flags=0)
    np.testing.assert_allclose(system(sim, "Centreline").positions.numpy(), expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_binding_refreshes_only_the_centreline_values_derived_from_the_state(device):
    """The named exception to "binding writes nothing": the tracker follows the particles of the State it is bound
    to. On a State saved after a completed step that rewrites identical bytes."""
    cfg, a, m = authored(device)
    state, control = m.state(), m.control()
    sim = runtime(cfg, a, m, state=state, control=control)
    for _ in range(3):
        sim.step()
    saved = history(sim)
    assert history(runtime(cfg, a, m, state=state, control=control, prepare=False)) == saved
    expected = moved(sim, [0.01, 0.0, 0.0])
    fresh = runtime(cfg, a, m, state=state, control=control, prepare=False)
    np.testing.assert_allclose(system(fresh, "Centreline").positions.numpy(), expected, rtol=0, atol=1e-6)
    after = history(fresh)
    assert {k: v for k, v in after.items() if k[0] != "Centreline"} == {k: v for k, v in saved.items() if k[0] != "Centreline"}


@pytest.mark.parametrize("device,captured", MODES)
def test_fresh_runtime_resumes_an_episode_with_active_interaction_history(device, captured):
    """Save State and Control mid-episode and bind a fresh runtime to the copy: binding changes no byte in any
    mode, and on the CPU both runtimes then step bitwise equal.

    The saved episode holds a surface grasp, a volumetric grasp, a set hook, an applied clip and tool-share residue.
    Thermal, bleeding and the fluids keep history that C4.2 migrates; they are disabled here.
    """
    cfg, a, m = authored(device, disable=LATER)
    state, control = m.state(), m.control()
    sim = runtime(cfg, a, m, captured, state=state, control=control)
    episode = Episode(sim)
    episode.run(45, False)
    episode.run(15, True)
    set_hook(sim)
    for name in ("InstrumentContact", "Grasp", "VolumetricGrasp"):  # shares the apply kernel never reduces
        share = getattr(system(sim, name), "delta_k")
        host = share.numpy()
        host[5, 0] = (1.0e-30, 0.0, 0.0)
        wp.copy(share, wp.array(host, dtype=wp.vec3, device=device))
    episode.run(2, True)
    for name, key in (("Grasp", "active"), ("VolumetricGrasp", "active"), ("Hook", "active"), ("Clipping", "active"),
                      ("Clipping", "clipped"), ("InstrumentContact", "delta_k"), ("Grasp", "delta_k"), ("VolumetricGrasp", "delta_k")):
        if name == "Hook" and device != "cpu":
            continue  # the injected anchor is not always retained for two CUDA steps (the run is not repeatable there)
        assert getattr(system(sim, name), key).numpy().any(), f"the episode left {name}.{key} at its default"
    saved_state, saved_control = m.state(), m.control()
    copy_container(state, saved_state)
    copy_container(control, saved_control)
    saved = snapshot(saved_state, saved_control)
    resumed = runtime(cfg, a, m, captured, state=saved_state, control=saved_control)
    Episode(resumed)
    assert snapshot(saved_state, saved_control) == saved, first_difference(snapshot(saved_state, saved_control), saved)
    for closed in (True, False):
        episode.run(6, closed, resumed)
        if device == "cpu":
            left, right = snapshot(state, control), snapshot(saved_state, saved_control)
            assert left == right, f"resume diverged (jaws closed: {closed}) at {first_difference(left, right)}"
    # Observed on CUDA: two runtimes restored from one checkpoint differ from each other after a single step, in eager
    # and captured execution alike, so stepping this case is not bitwise repeatable there. The bitwise claim is made
    # on the CPU, where it is; what capture could break is checked separately in
    # test_runtimes_on_separate_containers_do_not_touch_each_other.
    for container in (state, saved_state):
        assert np.isfinite(container.particle_q.numpy()).all()


# Construction values that are not zero; every other history array starts at zero.
FILLS = {("Hook", "verts"): -1, ("Hook", "tet"): -1, ("Grasp", "prim"): -1, ("Grasp", "tet"): -1,
         ("VolumetricGrasp", "particle"): -1, ("Clipping", "node"): -1}


@pytest.mark.parametrize("device,captured", MODES)
def test_templates_identities_and_independent_containers(device, captured):
    cfg, a, m = authored(device)
    state, control = m.state(), m.control()
    sim = runtime(cfg, a, m, captured, state=state, control=control)
    scene, centre = sim.scene, system(sim, "Centreline")
    E, n = scene.envs, system(sim, "InstrumentControl").count
    variant = centre.variant.numpy()
    for key, raw in history(sim).items():
        values = getattr(system(sim, key[0]), key[1]).numpy()
        if key == ("Centreline", "positions"):
            np.testing.assert_array_equal(values, centre.rest.numpy()[variant])
        elif key == ("Centreline", "active_count"):
            np.testing.assert_array_equal(values, centre.clamp_count.numpy()[variant])
        else:
            assert (values == FILLS.get(key, 0)).all(), key
    records = scene.interaction_domains
    kinds = {"centreline", "instrument_contact", "hook", "grasp", "volumetric_grasp", "clipping"}
    assert {r["kind"] for r in records.values()} >= kinds  # tissue and fluid history has its own module
    names, global_rows = set(), {}
    for sid, record in records.items():
        if record["kind"] not in kinds:
            continue
        owner = next(s for s in sim.systems if s.id == sid)
        for key, (f, vector, shape) in record["fields"].items():
            backing, live = getattr(state.i4h, f.name), getattr(owner, key)
            assert f.owner.value == "state" and f.name not in names and live.shape == tuple(shape)
            names.add(f.name)
            global_rows[f.name] = (backing, assert_native_alias(live, backing, f))
            assert getattr(m.i4h, f.name).ptr != backing.ptr != getattr(m.state().i4h, f.name).ptr
        for label, d in record["domains"].items():
            if not d.custom:
                continue
            worlds = getattr(m.i4h, d.frequency[4:] + "_world").numpy().reshape(d.shape)
            owners = getattr(m.i4h, d.frequency[4:] + "_owner").numpy().reshape(d.shape)
            np.testing.assert_array_equal(worlds, np.broadcast_to(np.arange(E).reshape((E,) + (1,) * (len(d.shape) - 1)), d.shape))
            if label == "node":
                assert (owners == -1).all()
            elif label == "slot" and record["kind"] != "clipping":
                np.testing.assert_array_equal(owners, np.broadcast_to((np.arange(E)[:, None] * n + np.arange(n)[None])[..., None], d.shape))
            else:  # one tool owns every hook anchor and every clip slot of a world
                tool = system(sim, "InstrumentControl").names.index(record["instrument"])
                np.testing.assert_array_equal(owners, np.broadcast_to(np.arange(E).reshape((E,) + (1,) * (len(d.shape) - 1)) * n + tool, d.shape))
    assert len(names) == 36
    # The tool shares add no rows: they are columns of the surface's existing particle rows.
    body = system(sim, "Grasp").body
    assert all(records[sid]["fields"]["delta_k"][0].domain.shape == (E, body.count) and "particle" not in records[sid]["domains"]
               for sid in ("instrument_contact", "grasp", "volumetric_grasp"))
    assert system(sim, "Grasp").delta_k.shape == (E * body.count, max(n, 1)) and system(sim, "Grasp").delta_k.dtype == wp.vec3
    # Live history never writes templates or another container.
    template = snapshot(m)
    other = m.state()
    untouched = snapshot(other)
    poison(sim)
    assert snapshot(m) == template and snapshot(other) == untouched
    for backing, before in global_rows.values():
        if before is not None:
            assert backing.numpy()[-1:].tobytes() == before


@pytest.mark.parametrize("device,captured", MODES)
@pytest.mark.parametrize("name,key", [("Grasp", "active"), ("Hook", "verts"), ("Clipping", "clipped"), ("Centreline", "positions"),
                                      ("VolumetricGrasp", "delta_k"), ("InstrumentContact", "suppress_active")])
def test_replaced_history_requires_a_fresh_binding(device, captured, name, key):
    cfg, a, m = authored(device)
    sim = runtime(cfg, a, m, captured)
    owner = system(sim, name)
    original = getattr(owner, key)
    setattr(owner, key, wp.clone(original))
    before = history(sim)
    with pytest.raises(RuntimeError, match="replaced"):
        sim.prepare()
    assert history(sim) == before
    setattr(owner, key, original)
    sim.prepare()
    assert history(sim) == before


def test_declarations_are_checked_before_allocation():
    from i4h_physics import Scene
    from i4h_physics.loaders import add_tet_grid
    from i4h_surgical.instrument_storage import declare_instruments
    from i4h_surgical.interaction_storage import (
        declare_clipping,
        declare_grasp,
        declare_hook,
        declare_instrument_contact,
        declare_volumetric_grasp,
    )
    from i4h_surgical.loaders.instrument_usd import add_instrument

    scene = Scene("cpu", 2, gravity=(0.0, 0.0, 0.0))
    add_tet_grid(scene, "block", [2, 2, 2], 0.02, radius=0.005)
    add_instrument(scene, "ball", kind="sphere", radius=0.02)
    declare_volumetric_grasp(scene, body="block")
    with pytest.raises(ValueError, match="duplicate storage declaration"):
        declare_volumetric_grasp(scene, body="block")
    declare_instrument_contact(scene)
    declare_grasp(scene)
    declare_hook(scene, instrument="ball")
    declare_clipping(scene, instrument="ball")
    with pytest.raises(ValueError, match="declare it first"):
        scene.allocate()  # nothing declared the control, the surface or the centreline these systems refer to
    late = Scene("cpu", 2, gravity=(0.0, 0.0, 0.0))
    add_tet_grid(late, "block", [2, 2, 2], 0.02, radius=0.005)
    add_instrument(late, "ball", kind="sphere", radius=0.02)
    declare_instruments(late)
    late.allocate()
    with pytest.raises(RuntimeError, match="before allocation"):
        declare_volumetric_grasp(late, body="block")


def test_changed_capacity_after_declaration_is_rejected():
    from test_volumetric_grasp_storage import block_and_grasper

    with pytest.raises(ValueError, match="slots changed after declaration"):
        block_and_grasper(declared=32, built=64)
    sim, grasp = block_and_grasper(declared=64, built=64)
    record = sim.scene.interaction_domains["volumetric_grasp"]
    assert "particle" in record["domains"] and record["fields"]["delta_k"][0].domain is record["domains"]["particle"]
    assert grasp.delta_k.shape == (2 * grasp.body.count, 1) and grasp.particle.shape == (2, 1, 64)
    owners = sim.model.i4h.volumetric_grasp_particle_rows_owner.numpy().reshape(2, grasp.body.count)
    np.testing.assert_array_equal(owners, sim.scene.field("q", grasp.body).shape[1] * np.arange(2)[:, None] + np.arange(grasp.body.count)[None]
                                  + owners[0, 0])


def test_contact_records_are_rewritten_before_they_are_read():
    """Scratch poisoning on the CPU, where stepping is repeatable: stale records and counts change nothing."""
    results = []
    for poisoned in (False, True):
        cfg, a, m = authored("cpu", disable=LATER)
        state, control = m.state(), m.control()
        sim = runtime(cfg, a, m, state=state, control=control)
        episode = Episode(sim)
        episode.run(45, False)
        if poisoned:
            contact = system(sim, "InstrumentContact")
            for key in ("record_count", "rec_tri", "rec_tet", "rec_pen", "rec_t"):
                getattr(contact, key).fill_(3)
        episode.run(4, True)
        results.append(snapshot(state, control))
    assert results[0] == results[1], first_difference(*results)


def test_mounted_tool_resumes_with_a_held_grasp_and_a_pending_reaction():
    """The combination C3 could not claim: a tool carried by a rigid body, holding tissue, with the tissue's reaction
    pending on the body. CPU, bitwise; the heavy rigid test solver keeps all of its history in native State."""
    cfg, a, m = authored("cpu", disable=LATER, mount="grasper")
    state, control = m.state(), m.control()
    sim = Coupled(cfg, a, m, state, control)
    episode = Episode(sim)
    tool = system(sim, "InstrumentControl").names.index("grasper")
    poses = state.body_q.numpy()
    poses[:, :3] = episode.targets[0, tool, :3]  # the holder carries the grasper to its tissue target
    state.body_q.assign(poses)
    episode.run(10, False)
    episode.run(12, True)
    group = sim.mounts.groups[0]
    assert system(sim, "InstrumentControl").mounted.numpy()[tool] == 1
    assert system(sim, "VolumetricGrasp").active.numpy()[:, tool].any() or system(sim, "Grasp").active.numpy()[:, tool].any()
    assert group.force.numpy().any(), "no reaction is pending on the holder"
    saved_state, saved_control = m.state(), m.control()
    copy_container(state, saved_state)
    copy_container(control, saved_control)
    saved = snapshot(saved_state, saved_control)
    resumed = Coupled(cfg, a, m, saved_state, saved_control)
    Episode(resumed)
    assert snapshot(saved_state, saved_control) == saved, first_difference(snapshot(saved_state, saved_control), saved)
    for closed in (True, False):
        episode.run(5, closed, resumed)
        left, right = snapshot(state, control), snapshot(saved_state, saved_control)
        assert left == right, f"resume diverged (jaws closed: {closed}) at {first_difference(left, right)}"
    assert np.isfinite(state.particle_q.numpy()).all() and np.isfinite(state.body_q.numpy()).all()


@pytest.mark.parametrize("device,captured", MODES)
def test_runtimes_on_separate_containers_do_not_touch_each_other(device, captured):
    """Deterministic in every mode: a (captured) runtime writes only the State and Control it was bound to, and its
    history arrays are views of that State."""
    cfg, a, m = authored(device, disable=LATER)
    state, control = m.state(), m.control()
    sim = runtime(cfg, a, m, captured, state=state, control=control)
    episode = Episode(sim)
    episode.run(20, False)
    episode.run(5, True)
    saved_state, saved_control = m.state(), m.control()
    copy_container(state, saved_state)
    copy_container(control, saved_control)
    resumed = runtime(cfg, a, m, captured, state=saved_state, control=saved_control)
    other = Episode(resumed)
    for sid, record in resumed.scene.interaction_domains.items():
        for key, (f, _, _) in record["fields"].items():
            for bound, container in ((resumed, saved_state), (sim, state)):
                owner = next(x for x in bound.systems if x.id == sid)
                owner = owner.appearance if record["kind"] == "appearance" else owner  # the surface's colour object
                assert getattr(owner, key).ptr == getattr(container.i4h, f.name).ptr
    first = snapshot(state, control)
    other.run(3, True)
    assert snapshot(state, control) == first, first_difference(snapshot(state, control), first)
    second = snapshot(saved_state, saved_control)
    assert second != first
    episode.run(3, True)
    assert snapshot(saved_state, saved_control) == second, first_difference(snapshot(saved_state, saved_control), second)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_contact_without_instruments_keeps_one_share_column(device):
    """The contact system on a scene without instruments: its tool shares keep one column, and it steps. (The grasp
    systems read the scene's instrument records and were never constructible without instruments.)"""
    from i4h_physics import Scene, Stepper, declare_surface, declare_tet
    from i4h_physics.loaders import add_tet_grid
    from i4h_physics.systems.mechanics.particles import Particles
    from i4h_physics.systems.mechanics.surface import Surface
    from i4h_physics.systems.mechanics.tet import Tet
    from i4h_surgical.instrument_storage import declare_instruments
    from i4h_surgical.interaction_storage import declare_instrument_contact
    from i4h_surgical.systems.contact.instrument_contact import InstrumentContact
    from i4h_surgical.systems.instruments.instrument_control import InstrumentControl

    scene = Scene(device, 2, gravity=(0.0, 0.0, 0.0))
    add_tet_grid(scene, "block", [2, 2, 2], 0.02, radius=0.005)
    declare_tet(scene, "block", system_id="tet")
    declare_surface(scene)
    declare_instruments(scene)
    declare_instrument_contact(scene)
    scene.allocate()
    control, tet = InstrumentControl(scene), Tet(scene, "block")
    surface = Surface(scene, tet=tet)
    contact = InstrumentContact(scene, instruments=control, surface=surface)
    sim = Stepper.create(scene, [Particles(scene), tet, surface, control, contact], dt=1.0 / 60.0, substeps=1, iterations=1)
    sim.prepare()
    sim.step()
    assert control.count == 0 and contact.delta_k.shape == contact.delta_jaw.shape == (2 * scene.body("block").count, 1)
    assert contact.delta_k.ptr == sim.state.i4h.instrument_contact_delta_k.ptr
    assert np.isfinite(scene.q.numpy()).all()
