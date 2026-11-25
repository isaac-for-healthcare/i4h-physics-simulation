# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tissue, fluid and containment history (heat, colour, fluid counters, smoke, candidates): lifecycle contracts
that ownership must preserve."""
import numpy as np
import pytest
import warp as wp
from interaction_helpers import MODES, authored, copy_container, first_difference, runtime, snapshot, system
from ownership_helpers import assert_native_alias
from tissue_helpers import CONTAINMENT, RESET, TISSUE, TissueEpisode, aorta, field, history, inject, poison, world_rows

KINDS = {"thermal", "appearance", "fluid", "bleeding", "emitter", "smoke", "suction", "graph_containment"}
# Newton attribute of every array, in the order of TISSUE and CONTAINMENT.
NAMES = {
    ("Thermal", "heat"): "thermal_heat", ("Thermal", "burn"): "thermal_burn", ("Thermal", "coag"): "thermal_history_coag",
    ("Surface", "appearance.visual"): "surface_visual",
    ("Fluid", "cursor"): "fluid_cursor", ("Fluid", "overflow"): "fluid_overflow",
    ("Bleeding", "cut"): "bleeding_history_cut", ("Bleeding", "source"): "bleeding_history_source",
    ("Bleeding", "acc"): "bleeding_history_acc", ("Bleeding", "serial"): "bleeding_serial",
    ("Emitter", "acc"): "emitter_acc", ("Emitter", "serial"): "emitter_serial",
    ("Smoke", "pos_age"): "smoke_pos_age", ("Smoke", "vel_life"): "smoke_vel_life", ("Smoke", "cursor"): "smoke_cursor",
    ("Smoke", "acc"): "smoke_acc", ("Smoke", "serial"): "smoke_serial",
    ("Suction", "removed"): "suction_history_removed",
    **{("GraphContainment", key): f"lumen_containment_{key}" for key in CONTAINMENT["GraphContainment"]},
}


def cases():
    return [pytest.param(authored, TISSUE, id="chole"), pytest.param(aorta, CONTAINMENT, id="aorta_centreline")]


def test_history_set_is_the_inventoried_one():
    assert sum(len(v) for v in TISSUE.values()) + sum(len(v) for v in CONTAINMENT.values()) == 25


@pytest.mark.parametrize("build,names", cases())
@pytest.mark.parametrize("device,captured", MODES)
def test_episode_reset_clears_history_in_selected_worlds_only(device, captured, build, names):
    cfg, a, m = build(device)
    sim = runtime(cfg, a, m, captured)
    poison(sim, names)
    before = history(sim, names)
    sim.reset_physics(mask=wp.array([False, True], dtype=wp.bool, device=device))
    after = history(sim, names)
    for key in before:
        old = np.frombuffer(before[key], np.uint8).reshape(2, -1)
        new = np.frombuffer(after[key], np.uint8).reshape(2, -1)
        np.testing.assert_array_equal(new[0], old[0], err_msg=f"unselected world changed: {key}")
        np.testing.assert_array_equal(world_rows(sim, *key, 1), RESET.get(key, 0), err_msg=str(key))


@pytest.mark.parametrize("device,captured", MODES)
def test_cache_only_preparation_preserves_tissue_history(device, captured):
    cfg, a, m = authored(device)
    sim = runtime(cfg, a, m, captured)
    poison(sim, TISSUE)
    before = history(sim, TISSUE)
    sim.prepare()
    sim.prepare()
    assert history(sim, TISSUE) == before


@pytest.mark.parametrize("device,captured", MODES)
def test_containment_preparation_forces_a_rebuild_and_keeps_the_cache_bytes(device, captured):
    """A named exception: preparation invalidates the candidate cache in every world, because the spline settings
    it protects are not part of the cache's validity record."""
    cfg, a, m = aorta(device)
    sim = runtime(cfg, a, m, captured)
    poison(sim, CONTAINMENT)
    before = history(sim, CONTAINMENT)
    sim.prepare()
    after = history(sim, CONTAINMENT)
    key = ("GraphContainment", "rebuild")
    np.testing.assert_array_equal(system(sim, "GraphContainment").rebuild.numpy(), 1)
    assert {k: v for k, v in after.items() if k != key} == {k: v for k, v in before.items() if k != key}


@pytest.mark.parametrize("device,captured", MODES)
def test_smoke_serial_counts_steps_and_its_source_point_is_an_output(device, captured):
    """``serial`` seeds the spawn random numbers and advances once per step in every world. ``centre`` is written
    only while a source exists and no kernel reads it, so it is a product, not history."""
    cfg, a, m = authored(device)
    sim = runtime(cfg, a, m, captured)
    smoke = system(sim, "Smoke")
    for _ in range(3):
        sim.step()
    np.testing.assert_array_equal(smoke.serial.numpy(), 3)
    sim.reset_physics(mask=wp.array([True, False], dtype=wp.bool, device=device))
    np.testing.assert_array_equal(smoke.serial.numpy(), [0, 3])


@pytest.mark.parametrize("build,names", cases())
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_history_is_newton_state_with_templates_and_independent_containers(device, build, names):
    cfg, a, m = build(device)
    state, control, other = m.state(), m.control(), m.state()
    before = snapshot(state, control)
    sim = runtime(cfg, a, m, state=state, control=control, prepare=False)
    after = snapshot(state, control)
    declared, global_rows = set(), {}
    records = {sid: r for sid, r in sim.scene.interaction_domains.items() if r["kind"] in KINDS}
    declarations = {f.name: f for r in records.values() for f, _, _ in r["fields"].values()}
    for name, keys in names.items():
        for key in keys:
            attribute = NAMES[name, key]
            live, backing = field(sim, name, key), getattr(state.i4h, attribute)
            global_rows[attribute] = (backing, assert_native_alias(live, backing, declarations[attribute]))
            assert len({backing.ptr, getattr(m.i4h, attribute).ptr, getattr(other.i4h, attribute).ptr}) == 3
            assert after[0][f"i4h:{attribute}"] == before[0][f"i4h:{attribute}"], f"binding wrote {attribute}"
            np.testing.assert_array_equal(live.numpy(), RESET.get((name, key), 0), err_msg=attribute)
            declared.add(attribute)
    assert {f.name for r in records.values() for f, _, _ in r["fields"].values()} == declared
    poison(sim, names)
    for attribute in declared:  # live history never writes the template or another State
        template, untouched = getattr(m.i4h, attribute).numpy(), getattr(other.i4h, attribute).numpy()
        np.testing.assert_array_equal(template, untouched)
        assert getattr(state.i4h, attribute).numpy().tobytes() != template.tobytes()
    for backing, before in global_rows.values():
        if before is not None:
            assert backing.numpy()[-1:].tobytes() == before


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_shapes_are_the_ones_the_kernels_take_and_new_rows_carry_their_owners(device):
    cfg, a, m = authored(device)
    sim = runtime(cfg, a, m)
    E, body = sim.scene.envs, system(sim, "Thermal").body
    T, N, K = system(sim, "Surface").tets.shape[1], body.count, system(sim, "Centreline").positions.shape[1]
    nf, count = system(sim, "Fluid").nf, system(sim, "Smoke").count
    shapes = {("Thermal", "heat"): ((E, T), wp.float32), ("Thermal", "burn"): ((E, T), wp.float32), ("Thermal", "coag"): ((E * N,), wp.float32),
              ("Surface", "appearance.visual"): ((E * N,), wp.vec4f), ("Fluid", "cursor"): ((E, nf), wp.int32),
              ("Fluid", "overflow"): ((E, nf), wp.int32), ("Bleeding", "cut"): ((E, K), wp.int32), ("Bleeding", "source"): ((E, K), wp.int32),
              ("Bleeding", "acc"): ((E, K), wp.float32), ("Bleeding", "serial"): ((E,), wp.int32), ("Emitter", "acc"): ((E,), wp.float32),
              ("Emitter", "serial"): ((E,), wp.int32), ("Smoke", "pos_age"): ((E * count,), wp.vec4f),
              ("Smoke", "vel_life"): ((E * count,), wp.vec4f), ("Smoke", "cursor"): ((E,), wp.int32), ("Smoke", "acc"): ((E,), wp.float32),
              ("Smoke", "serial"): ((E,), wp.int32), ("Suction", "removed"): ((E, nf), wp.int32)}
    for (name, key), (shape, dtype) in shapes.items():
        live = field(sim, name, key)
        native = wp.zeros(shape, dtype=dtype, device=device)
        assert (live.shape, live.strides, live.dtype) == (native.shape, native.strides, native.dtype), (name, key)
    # One colour array, reached by every system that paints and by the surface itself.
    visual = system(sim, "Surface").appearance.visual
    painters = [s for s in sim.systems if getattr(getattr(s, "surface", None), "appearance", None) is not None]
    assert {type(s).__name__ for s in painters} >= {"Thermal", "Cutting", "ContactPaint"}
    assert all(s.surface.appearance.visual is visual for s in painters)
    assert system(sim, "Smoke").thermal.heat.ptr == sim.solver.state.i4h.thermal_heat.ptr
    # Colour and coagulation add no rows; the fluid pools and the smoke particles do, one world each.
    for frequency, rows in (("fluid_pool_rows", (E, nf)), ("smoke_particle_rows", (E, count))):
        worlds = getattr(m.i4h, frequency + "_world").numpy().reshape(rows)
        np.testing.assert_array_equal(worlds, np.broadcast_to(np.arange(E)[:, None], rows))
        assert (getattr(m.i4h, frequency + "_owner").numpy() == -1).all()
    assert not hasattr(m.i4h, "surface_support_rows_world") or m.i4h.thermal_history_coag.shape == (E * N,)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_containment_rows_are_owned_by_catheter_and_tree_particles(device):
    cfg, a, m = aorta(device)
    sim = runtime(cfg, a, m)
    gc = system(sim, "GraphContainment")
    E, n, Pc, Pt, cap = sim.scene.envs, gc.samples, gc.Pc, gc.Pt, gc.capacity
    assert gc.cand_idx.shape == (E * n * cap,) and gc.cand_count.shape == gc.cand_overflow.shape == gc.ref_samples.shape == (E * n,)
    assert gc.ref_tree.shape == (E * Pt,) and gc.rebuild.shape == gc.built.shape == (E,) and gc.built.dtype == wp.vec2
    world_of = m.particle_world.numpy()
    anchor = np.concatenate([np.arange(Pc), np.repeat(np.arange(Pc - 1), 3)])
    for label, count, local, part in (("sample", n, anchor, gc.body), ("tree", Pt, np.arange(Pt), gc.centreline)):
        worlds = getattr(m.i4h, f"lumen_containment_{label}_rows_world").numpy().reshape(E, count)
        owners = getattr(m.i4h, f"lumen_containment_{label}_rows_owner").numpy().reshape(E, count)
        np.testing.assert_array_equal(worlds, np.broadcast_to(np.arange(E)[:, None], (E, count)))
        np.testing.assert_array_equal(world_of[owners], worlds)
        np.testing.assert_array_equal(owners - owners[:, :1], np.broadcast_to(local - local[0], (E, count)))
        q = sim.scene.field("q", part).numpy()
        np.testing.assert_array_equal(m.particle_q.numpy()[owners], q[:, local])


@pytest.mark.parametrize("device,captured", MODES)
@pytest.mark.parametrize("build,name,key", [
    pytest.param(authored, "Thermal", "heat", id="heat"), pytest.param(authored, "Thermal", "coag", id="coag"),
    pytest.param(authored, "Surface", "appearance.visual", id="colour"), pytest.param(authored, "Fluid", "cursor", id="cursor"),
    pytest.param(authored, "Bleeding", "cut", id="cut"), pytest.param(authored, "Smoke", "pos_age", id="smoke"),
    pytest.param(authored, "Suction", "removed", id="removed"), pytest.param(aorta, "GraphContainment", "cand_idx", id="candidates"),
    pytest.param(aorta, "GraphContainment", "built", id="built")])
def test_replaced_history_requires_a_fresh_binding(device, captured, build, name, key):
    cfg, a, m = build(device)
    sim = runtime(cfg, a, m, captured)
    owner, attribute = system(sim, name), key.split(".")[-1]
    for part in key.split(".")[:-1]:
        owner = getattr(owner, part)
    original = getattr(owner, attribute)
    setattr(owner, attribute, wp.clone(original))
    with pytest.raises(RuntimeError, match="replaced"):
        sim.prepare()
    setattr(owner, attribute, original)
    sim.prepare()


def test_declarations_are_checked_before_allocation():
    from i4h_physics import Scene
    from i4h_physics.containment_storage import declare_graph_containment
    from i4h_physics.fluid_storage import declare_fluid
    from i4h_physics.loaders import add_fluid, add_tet_grid
    from i4h_surgical.interaction_storage import (
        declare_appearance,
        declare_bleeding,
        declare_smoke,
        declare_suction,
        declare_thermal,
    )

    def scene():
        s = Scene("cpu", 2, gravity=(0.0, 0.0, 0.0))
        add_tet_grid(s, "block", [2, 2, 2], 0.02, radius=0.005)
        add_fluid(s, "water", 16, mass=0.001, radius=0.003, interaction_radius=0.012, rest_distance=0.006)
        return s
    for declare, missing in ((lambda s: declare_suction(s), "fluid"), (lambda s: declare_bleeding(s), "centreline"),
                             (lambda s: declare_thermal(s), "surface"), (lambda s: declare_appearance(s), "surface")):
        s = scene()
        declare(s)
        with pytest.raises(ValueError, match=f"not a declared {missing}; declare it first"):
            s.allocate()
    s = scene()
    declare_fluid(s)
    declare_smoke(s, count=8)
    for again in (declare_fluid, declare_smoke):
        with pytest.raises(ValueError, match="duplicate storage declaration"):
            again(s)
    with pytest.raises(ValueError, match="centreline tree"):
        declare_graph_containment(s, body="block", centreline="water")
        s.allocate()
    late = scene()
    late.allocate()
    for declare in (declare_fluid, declare_smoke, declare_suction, declare_thermal, declare_appearance):
        with pytest.raises(RuntimeError, match="before allocation"):
            declare(late)


@pytest.mark.parametrize("build,name,settings,message", [
    pytest.param(aorta, "GraphContainment", dict(capacity=32), "capacity changed after declaration", id="candidates"),
    pytest.param(authored, "Smoke", dict(count=64), "count changed after declaration", id="smoke")])
def test_changed_capacity_after_declaration_is_rejected(build, name, settings, message):
    cfg, a, m = build("cpu")
    sim = runtime(cfg, a, m, prepare=False)
    old = system(sim, name)
    refs = (dict(instruments=old.instruments, contacts=old.contacts, thermal=old.thermal, instrument=old.instrument) if name == "Smoke"
            else dict(body=old.body.name, centreline=old.centreline.name))
    with pytest.raises(ValueError, match=message):
        type(old)(sim.scene, id=old.id, **refs, **settings)


def chole_episode(sim):
    """Travel, close and energise, then add what stepping does not reach; two more steps let the kernels burn,
    paint, age smoke and bleed from that state."""
    episode = TissueEpisode(sim)
    episode.run(45, False)
    episode.run(15, True)
    inject(sim)
    episode.run(2, True)
    return episode


@pytest.mark.parametrize("device,captured", MODES)
def test_fresh_runtime_resumes_an_episode_with_tissue_and_fluid_history(device, captured):
    """Save State and Control with heat, burn, coagulation, colour, blood, irrigation and smoke present and bind a
    fresh runtime to the copy: binding changes no byte in any mode, and on the CPU both runtimes then step bitwise
    equal.

    Fluid contact keeps one thing outside State: the shape of its triangle tree, built from the positions at
    construction and only refitted afterwards. ``save_i4h`` rebuilds that tree in the saving runtime, which is what
    a fresh binding does implicitly. Without the rebuild the two diverge in the pool positions.
    """
    cfg, a, m = authored(device)
    state, control = m.state(), m.control()
    sim = runtime(cfg, a, m, captured, state=state, control=control)
    episode = chole_episode(sim)
    for name, keys in TISSUE.items():
        for key in keys:
            if (name, key) != ("Smoke", "acc"):  # the spawn kernel zeroes it in every step without a smoking contact
                assert field(sim, name, key).numpy().any(), f"the episode left {name}.{key} at its default"
    saved_state, saved_control = m.state(), m.control()
    sim.physics.save(saved_state, saved_control)
    saved = snapshot(saved_state, saved_control)
    assert saved == snapshot(state, control)
    resumed = runtime(cfg, a, m, captured, state=saved_state, control=saved_control)
    TissueEpisode(resumed)
    assert snapshot(saved_state, saved_control) == saved, first_difference(snapshot(saved_state, saved_control), saved)
    if device == "cpu":
        for closed in (True, False):
            episode.run(6, closed, resumed)
            left, right = snapshot(state, control), snapshot(saved_state, saved_control)
            assert left == right, f"resume diverged (jaws closed: {closed}) at {first_difference(left, right)}"
    for container in (state, saved_state):
        assert np.isfinite(container.particle_q.numpy()).all()


def drive(sim, steps, *others):
    for runtime_ in (sim, *others):
        system(runtime_, "CatheterControl").actions.assign(np.tile(np.float32([0.1, 0.0, 0.0]), (runtime_.scene.envs, 1)))
    for _ in range(steps):
        for runtime_ in (sim, *others):
            runtime_.step()


@pytest.mark.parametrize("device,captured", MODES)
def test_fresh_runtime_resumes_a_catheter_in_graph_containment(device, captured):
    """Preparation forces a candidate rebuild, so the reference is the uninterrupted runtime prepared again at the
    save point. On the CPU the fresh binding then steps bitwise equal to it."""
    cfg, a, m = aorta(device)
    state, control = m.state(), m.control()
    sim = runtime(cfg, a, m, captured, state=state, control=control)
    drive(sim, 160)  # the 6.35 catheter reaches the wall after ~140 steps at 0.1
    gc = system(sim, "GraphContainment")
    assert gc.cand_count.numpy().any() and gc.ref_samples.numpy().any() and gc.built.numpy().any()
    saved_state, saved_control = m.state(), m.control()
    copy_container(state, saved_state)
    copy_container(control, saved_control)
    saved = snapshot(saved_state, saved_control)
    resumed = runtime(cfg, a, m, captured, state=saved_state, control=saved_control)
    rebuilt = {k: v for k, v in snapshot(saved_state, saved_control)[0].items() if v != saved[0][k]}
    assert set(rebuilt) <= {"i4h:lumen_containment_rebuild"}, sorted(rebuilt)  # the one write preparation makes
    assert snapshot(saved_state, saved_control)[1] == saved[1]
    if device == "cpu":
        sim.prepare()
        for _ in range(4):
            drive(sim, 5, resumed)
            left, right = snapshot(state, control), snapshot(saved_state, saved_control)
            assert left == right, f"resume diverged at {first_difference(left, right)}"
        assert system(sim, "GraphContainment").contact_count.numpy().any()


@pytest.mark.parametrize("build,advance", [pytest.param(authored, lambda sim: chole_episode(sim), id="chole"),
                                           pytest.param(aorta, lambda sim: drive(sim, 20), id="aorta_centreline")])
@pytest.mark.parametrize("device,captured", MODES)
def test_runtimes_on_separate_containers_do_not_touch_each_other(device, captured, build, advance):
    """Deterministic in every mode: a (captured) runtime writes only the State and Control it was bound to."""
    cfg, a, m = build(device)
    state, control = m.state(), m.control()
    sim = runtime(cfg, a, m, captured, state=state, control=control)
    advance(sim)
    saved_state, saved_control = m.state(), m.control()
    copy_container(state, saved_state)
    copy_container(control, saved_control)
    resumed = runtime(cfg, a, m, captured, state=saved_state, control=saved_control)
    step = (lambda s: TissueEpisode(s).run(3, True)) if build is authored else (lambda s: drive(s, 3))
    first = snapshot(state, control)
    step(resumed)
    assert snapshot(state, control) == first, first_difference(snapshot(state, control), first)
    second = snapshot(saved_state, saved_control)
    assert second != first
    step(sim)
    assert snapshot(saved_state, saved_control) == second, first_difference(snapshot(saved_state, saved_control), second)


def test_restored_candidate_cache_is_used_as_saved():
    """Restoring State into a runtime that is already prepared (no new preparation, so no forced rebuild): the
    restored runtime projects from the saved candidate lists, rebuilds in the same steps as the uninterrupted one
    and steps bitwise equal to it."""
    cfg, a, m = aorta("cpu")
    state, control = m.state(), m.control()
    sim = runtime(cfg, a, m, state=state, control=control)
    drive(sim, 42)  # the following step keeps the cache in the uninterrupted run (checked below)
    saved_state, saved_control = m.state(), m.control()
    resumed = runtime(cfg, a, m, state=saved_state, control=saved_control)
    copy_container(state, saved_state)  # after preparation: this restores ``rebuild`` as saved
    copy_container(control, saved_control)
    ours, theirs = system(sim, "GraphContainment"), system(resumed, "GraphContainment")
    assert not theirs.rebuild.numpy().any() and theirs.cand_count.numpy().any()
    rebuilds = []
    for _ in range(12):
        before = [x.ref_samples.numpy().copy() for x in (ours, theirs)]
        drive(sim, 1, resumed)
        left, right = snapshot(state, control), snapshot(saved_state, saved_control)
        assert left == right, f"restore diverged at {first_difference(left, right)}"
        rebuilt = [bool((x.ref_samples.numpy() != old).any()) for x, old in zip((ours, theirs), before)]
        assert rebuilt[0] == rebuilt[1]
        rebuilds.append(rebuilt[0])
    assert not rebuilds[0], "the first step rebuilt the cache, so the restored candidate lists were never read"
    assert any(rebuilds), "no later rebuild, so rebuild timing after the restore was not compared"


@pytest.mark.parametrize("build,names", cases())
@pytest.mark.parametrize("device,captured", MODES)
def test_empty_and_repeated_reset_masks(device, captured, build, names):
    """An empty selection writes nothing anywhere; resetting the same world twice equals resetting it once."""
    cfg, a, m = build(device)
    state, control = m.state(), m.control()
    sim = runtime(cfg, a, m, captured, state=state, control=control)
    poison(sim, names)
    before = snapshot(state, control)
    sim.reset_physics(mask=wp.array([False, False], dtype=wp.bool, device=device))
    sim.reset_physics([])
    assert snapshot(state, control) == before
    mask = wp.array([False, True], dtype=wp.bool, device=device)
    sim.reset_physics(mask=mask)
    once = snapshot(state, control)
    sim.reset_physics(mask=mask)
    assert snapshot(state, control) == once


@pytest.mark.parametrize("build,advance,written", [
    pytest.param(authored, lambda sim: chole_episode(sim), set(), id="chole"),
    pytest.param(aorta, lambda sim: drive(sim, 20), {"i4h:lumen_containment_rebuild"}, id="aorta_centreline")])
@pytest.mark.parametrize("device,captured", MODES)
def test_preparation_and_cache_reset_write_only_the_recorded_history(device, captured, build, advance, written):
    """The record of what a continuation must not be routed through: on a stepped State, preparation and the
    cache-only reset write nothing but graph containment's rebuild request."""
    cfg, a, m = build(device)
    state, control = m.state(), m.control()
    sim = runtime(cfg, a, m, captured, state=state, control=control)
    advance(sim)
    for write in (sim.prepare, lambda: sim.solver.reset(state, None, flags=0)):
        before = snapshot(state, control)
        write()
        after = snapshot(state, control)
        changed = {key for old, new in zip(before, after) for key in old if old[key] != new[key]}
        assert changed <= written, sorted(changed)
