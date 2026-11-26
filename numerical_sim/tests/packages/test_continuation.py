# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Continuation: ``save_i4h`` and ``restore_i4h`` copy exactly the rows i4h owns, and a runtime that restores a saved
State and Control continues like the one that saved them.

Bitwise resume is asserted where stepping a case is repeatable: every mode for the rod and mount fixtures, the CPU
for chole and the two aorta cases (on CUDA two runtimes restored from one checkpoint already differ from each other
there). Tip evidence, procedure state and a rigid solver's own history are outside the call.
"""
import newton
import numpy as np
import pytest
import test_mount_owned_storage as mount_case
import test_rod_owned_storage as rod_case
import warp as wp
from i4h_physics import author_case
from i4h_physics import config as physical
from i4h_physics.storage import Owner
from interaction_helpers import Coupled, Episode, authored, runtime, set_hook, system
from ownership_helpers import MODES, arrays, copy_container, first_difference, snapshot
from tissue_helpers import TissueEpisode, aorta, inject

NATIVE = ("particle_q", "particle_qd", "particle_f")
BODY = ("body_q", "body_qd", "body_f")


def row_worlds(scene):
    """World of every backing row the call may write, per snapshot key: -1 shared, -2 foreign or padding. Derived
    from the declared row identities and the particle placement, independently of the copy's own row maps."""
    d, m, E = scene.storage.declarations, scene.model, scene.envs
    particles = np.full(max(1, m.particle_count), -2)
    if scene.main.width and m.particle_count:
        base, stride, col = scene.main.place(E)
        for world in range(E):
            particles[base + col + world * stride:][:scene.main.width] = world
    table = {}
    for key, field in d.fields.items():
        if field.owner is Owner.MODEL:
            continue
        if field.domain.custom:
            worlds = np.asarray(d.rows(field.domain)[0]) if field.domain.count else np.full(1, -2)
        else:
            if field.domain.frequency == newton.Model.AttributeFrequency.PARTICLE:
                worlds = particles
            else:
                container = scene.state if field.owner is Owner.STATE else scene.control
                rows = getattr(container.i4h, d.name(key)).shape[0]
                worlds = np.full(rows, -2)
                worlds[:E] = np.arange(E)
        table["i4h:" + d.name(key)] = (int(field.owner is Owner.CONTROL), worlds)
    if m.particle_count:
        table.update({name: (0, particles) for name in NATIVE})
    return table


def expected(destination, source, scene, selected=None):
    """``destination`` with the owned rows of the selected worlds taken from ``source`` (``selected``: ``(E+1,)``
    host bools, last slot for shared rows; None for all)."""
    E = scene.envs
    selected = np.ones(E + 1, bool) if selected is None else np.asarray(selected)
    result = [dict(part) for part in destination]
    for key, (part, worlds) in row_worlds(scene).items():
        if key not in destination[part]:
            continue
        rows = np.frombuffer(destination[part][key], np.uint8).reshape(len(worlds), -1).copy()
        take = (worlds != -2) & selected[np.where(worlds == -1, E, np.maximum(worlds, 0))]
        rows[take] = np.frombuffer(source[part][key], np.uint8).reshape(len(worlds), -1)[take]
        result[part][key] = rows.tobytes()
    return result


def poison(seed, *containers):
    """Arbitrary bytes in every array of the containers (no physical meaning; the copy must not care)."""
    rng = np.random.default_rng(seed)
    for container in containers:
        for array in arrays(container).values():
            if array.size:
                host = array.numpy().copy()
                raw = host.view(np.uint8)
                raw[...] = rng.integers(1, 255, raw.shape, dtype=np.uint8)
                wp.copy(array, wp.array(host, dtype=array.dtype, device=array.device))
    wp.synchronize()


def rods(device, captured=False):
    """Three worlds of a tree rod with drag, behind one global and one per-world foreign particle."""
    a, b, m = rod_case.authored(device, "pbd", True, tree_body=True)
    state, control = m.state(), m.control()
    sim, systems = rod_case.runtime(a, m, captured, state=state, control=control)
    return a, m, sim, systems, state, control


def mask_of(device, *selected):
    return wp.array(selected, dtype=wp.bool, device=device)


@pytest.mark.parametrize("device,captured", MODES)
def test_save_and_restore_copy_exactly_the_owned_rows(device, captured):
    a, m, sim, _, state, control = rods(device, captured)
    saved = m.state(), m.control()
    poison(1, state, control)
    poison(2, *saved)
    model, live, other = snapshot(m), snapshot(state, control), snapshot(*saved)
    worlds = row_worlds(sim.scene)
    assert (worlds["particle_q"][1] == -2).sum() == 4 and (worlds["particle_q"][1] >= 0).any()  # foreign rows exist
    sim.physics.save(*saved)
    assert snapshot(state, control) == live and snapshot(m) == model
    after = snapshot(*saved)
    assert after == expected(other, live, sim.scene), first_difference(after, expected(other, live, sim.scene))
    assert after != other
    sim.physics.restore(*saved)  # the owned rows are already equal: restoring them is the identity
    assert snapshot(state, control) == live and snapshot(*saved) == after and snapshot(m) == model
    poison(3, state, control)
    live = snapshot(state, control)
    sim.physics.restore(*saved)
    restored = snapshot(state, control)
    assert restored == expected(live, after, sim.scene), first_difference(restored, expected(live, after, sim.scene))
    assert snapshot(*saved) == after and snapshot(m) == model


@pytest.mark.parametrize("device,captured", MODES)
def test_masks_select_worlds_and_shared_rows(device, captured):
    a, m, sim, _, state, control = rods(device, captured)
    saved = m.state(), m.control()
    poison(4, state, control)
    poison(5, *saved)
    live, other = snapshot(state, control), snapshot(*saved)
    sim.solver.save_i4h(*saved, mask_of(device, False, False, False, False))
    assert sim.physics.save(*saved, []) is False
    sim.physics.save(*saved, mask=mask_of(device, False, False, False))
    assert snapshot(*saved) == other and snapshot(state, control) == live
    for repeat in range(2):  # a repeated selection equals a single one
        sim.physics.save(*saved, [1])
        want = expected(other, live, sim.scene, [False, True, False, False])
        assert snapshot(*saved) == want, first_difference(snapshot(*saved), want)
    sim.physics.save(*saved, wp.array([2], dtype=wp.int32, device=device))
    want = expected(other, live, sim.scene, [False, True, True, False])
    assert snapshot(*saved) == want, first_difference(snapshot(*saved), want)
    sim.solver.save_i4h(*saved, mask_of(device, True, False, False, True))  # the last slot adds the shared rows
    assert snapshot(*saved) == expected(other, live, sim.scene)
    poison(6, state, control)
    live, checkpoint = snapshot(state, control), snapshot(*saved)
    sim.physics.restore(*saved, mask=mask_of(device, False, False, True))
    want = expected(live, checkpoint, sim.scene, [False, False, True, False])
    assert snapshot(state, control) == want, first_difference(snapshot(state, control), want)
    assert snapshot(*saved) == checkpoint


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_pool_rows_and_tissue_history_follow_the_world_mask(device):
    """Chole: fluid pool rows are stored body-major, so one world's rows are not contiguous."""
    cfg, a, m = authored(device)
    state, control = m.state(), m.control()
    sim = runtime(cfg, a, m, state=state, control=control)
    episode = TissueEpisode(sim)
    episode.run(20, True)
    saved = m.state(), m.control()
    live, other = snapshot(state, control), snapshot(*saved)
    worlds = row_worlds(sim.scene)["i4h:pool_q"][1]
    assert np.flatnonzero(np.diff(worlds)).size > 1, "pool rows are expected to interleave worlds"
    sim.physics.save(*saved, [1])
    want = expected(other, live, sim.scene, [False, True, False])
    assert want != other and snapshot(*saved) == want, first_difference(snapshot(*saved), want)
    assert snapshot(state, control) == live


def test_rejections_write_nothing():
    a, m, sim, _, state, control = rods("cpu")
    saved = m.state(), m.control()
    poison(7, state, control)
    live, other, model = snapshot(state, control), snapshot(*saved), snapshot(m)
    a2, b2, m2 = rod_case.authored("cpu", "pbd", True, tree_body=True)
    second = b2.finalize(device="cpu")
    shared = m.control(clone_variables=False)
    cases = [
        ((m2.state(), m2.control()), ValueError, "foreign"),  # another authoring
        ((m.state(), m2.control()), ValueError, "foreign"),
        ((state, saved[1]), ValueError, "other than the bound"),
        ((saved[0], control), ValueError, "other than the bound"),
        ((saved[0], shared), ValueError, "initialization templates"),
    ]
    for containers, error, message in cases:
        for call in (sim.physics.save, sim.physics.restore):
            with pytest.raises(error, match=message):
                call(*containers)
    on_templates = rod_case.runtime(a, m, state=m.state(), control=shared)[0]  # bound to the template Control
    on_templates.physics.save(*saved)  # reading it is allowed
    with pytest.raises(ValueError, match="rewrite them"):
        on_templates.physics.restore(*saved)
    sim.physics.save(*saved)
    poison(8, *saved)
    other = snapshot(*saved)
    with pytest.raises(ValueError, match="foreign"):  # a second finalization of the same builder is another Model
        rod_case.runtime(a2, m2)[0].physics.save(second.state(), second.control())
    with pytest.raises(ValueError):
        sim.solver.save_i4h(*saved, mask_of("cpu", True, True, True))  # (E,) is not a world mask
    name = next(key for key in arrays(saved[0]) if key.startswith("i4h:")).split(":")[1]
    original = getattr(saved[0].i4h, name)
    setattr(saved[0].i4h, name, wp.zeros(original.shape[0] + 1, dtype=original.dtype, device="cpu"))
    with pytest.raises(ValueError, match="incompatible continuation buffer"):
        sim.physics.save(*saved)
    setattr(saved[0].i4h, name, original)
    assert snapshot(state, control) == live and snapshot(*saved) == other and snapshot(m) == model
    assert snapshot(shared) == snapshot(m.control(clone_variables=False))
    sim.physics.save(*saved)  # the same pair is accepted once it is whole again
    setattr(state.i4h, name, wp.clone(getattr(state.i4h, name)))
    with pytest.raises(RuntimeError, match="replaced"):
        sim.physics.restore(*saved)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_continuation_preserves_native_world_global_rows(device):
    """The host's trailing WORLD row is foreign even when the shared-row mask slot is selected."""
    cfg, a, m = aorta(device)
    if m._attribute_frequency_count(newton.Model.AttributeFrequency.WORLD) == m.world_count:
        pytest.skip("this Newton revision has no global WORLD row")
    state, control = m.state(), m.control()
    sim = runtime(cfg, a, m, state=state, control=control)
    saved = m.state(), m.control()
    poison(21, state, control)
    poison(22, *saved)
    live, other = snapshot(state, control), snapshot(*saved)
    worlds = row_worlds(sim.scene)
    assert any(key.startswith("i4h:") and rows[-1] == -2 for key, (_, rows) in worlds.items())
    sim.physics.save(*saved)
    assert snapshot(*saved) == expected(other, live, sim.scene)
    poison(23, state, control)
    before, checkpoint = snapshot(state, control), snapshot(*saved)
    sim.solver.restore_i4h(*saved, wp.ones(m.world_count + 1, dtype=wp.bool, device=device))
    assert snapshot(state, control) == expected(before, checkpoint, sim.scene)


def test_repeated_calls_allocate_nothing():
    if not wp.is_cuda_available():
        pytest.skip("device memory pool counters need CUDA")
    a, m, sim, _, state, control = rods("cuda:0")
    saved = m.state(), m.control()
    mask = mask_of("cuda:0", False, True, False, False)
    for call in (sim.physics.save, sim.physics.restore):  # first use builds the views of the other pair
        call(*saved)
        sim.solver.save_i4h(*saved, mask)
    wp.synchronize()
    used = wp.get_mempool_used_mem_current("cuda:0")
    for _ in range(3):
        sim.physics.save(*saved)
        sim.physics.restore(*saved, [1])
        sim.solver.restore_i4h(*saved, mask)
    wp.synchronize()
    assert wp.get_mempool_used_mem_current("cuda:0") == used


def resume(saving, fresh, advance, state, control, steps, *, bodies=False):
    """Save from ``saving``, restore into ``fresh`` runtimes (already bound and prepared on their own containers) and
    into the saving runtime after it advanced: all must reproduce the uninterrupted run. Returns the first array
    that differs, or None."""
    model = state_model(saving)
    checkpoint = model.state(), model.control()
    target(saving).save(*checkpoint)
    if bodies:
        for name in BODY:
            wp.copy(getattr(checkpoint[0], name), getattr(state, name))
    assert snapshot(*checkpoint) == snapshot(state, control), first_difference(snapshot(*checkpoint), snapshot(state, control))
    for other, other_state, other_control in fresh:
        target(other).restore(*checkpoint)
        if bodies:
            for name in BODY:
                wp.copy(getattr(other_state, name), getattr(checkpoint[0], name))
        assert snapshot(other_state, other_control) == snapshot(*checkpoint)
    advance(steps, saving, *(other for other, _, _ in fresh))
    reference = snapshot(state, control)
    for _, other_state, other_control in fresh:
        difference = first_difference(reference, snapshot(other_state, other_control))
        if difference:
            return f"fresh runtime: {difference}"
    target(saving).restore(*checkpoint)
    if bodies:
        for name in BODY:
            wp.copy(getattr(state, name), getattr(checkpoint[0], name))
    advance(steps, saving)
    difference = first_difference(reference, snapshot(state, control))
    return f"replay on the saving runtime: {difference}" if difference else None


def state_model(sim):
    return sim.scene.model


def target(sim):
    """The object carrying ``save`` and ``restore``: the physics binding, or a coupled runtime given both names."""
    return sim if hasattr(sim, "save") else sim.physics


@pytest.mark.parametrize("device,captured", MODES)
def test_rod_with_an_active_drag_resumes_in_every_mode(device, captured):
    a, m, sim, systems, state, control = rods(device, captured)
    systems["d0"].begin(1, 1, [0.1, 0.02, 0])
    for _ in range(3):
        sim.step()
    assert systems["d0"].active.numpy().any() and systems["r0"].omega.numpy().any()
    fresh = []
    for _ in range(2):
        s, c = m.state(), m.control()
        fresh.append((rod_case.runtime(a, m, captured, state=s, control=c)[0], s, c))

    def advance(steps, *sims):
        for _ in range(steps):
            for runtime_ in sims:
                runtime_.step()

    assert resume(sim, fresh, advance, state, control, 4) is None


@pytest.mark.parametrize("device,captured", MODES)
def test_mounted_tools_with_a_pending_reaction_resume_in_every_mode(device, captured):
    a, b, m = mount_case.authored(device)
    r = mount_case.runtime(a, m, captured)
    for _ in range(3):
        r.step()
    assert r.mounts.groups[0].force.numpy().any(), "no reaction is pending"
    checkpoint = m.state(), m.control()
    r.coupled.save_i4h(*checkpoint)
    for name in BODY:
        wp.copy(getattr(checkpoint[0], name), getattr(r.state, name))
    assert snapshot(*checkpoint) == snapshot(r.state, r.control)
    resumed = mount_case.runtime(a, m, captured)
    resumed.coupled.restore_i4h(*checkpoint)
    for name in BODY:
        wp.copy(getattr(resumed.state, name), getattr(checkpoint[0], name))
    for _ in range(4):
        r.step()
        resumed.step()
        left, right = snapshot(r.state, r.control), snapshot(resumed.state, resumed.control)
        assert left == right, first_difference(left, right)
    r.coupled.restore_i4h(*checkpoint)
    for name in BODY:
        wp.copy(getattr(r.state, name), getattr(checkpoint[0], name))
    for _ in range(4):
        r.step()
    assert snapshot(r.state, r.control) == left, first_difference(snapshot(r.state, r.control), left)


def chole_episode(sim, early=()):
    """Grasp, volumetric grasp, clip, hook, residue, heat, burn, colour, blood, irrigation and smoke at the save
    point. ``early``: a State and Control that receive the episode as it is after 45 steps. The hook anchor, one residue value per share array, hot fat, smoke particles, the cut node and three
    counters are written directly (the script does not reach them); two more steps act on all but the hook anchor."""
    episode = TissueEpisode(sim)
    episode.run(45, False)
    for container, earlier in zip((sim.scene.state, sim.scene.control), early):
        copy_container(container, earlier)
    episode.run(15, True)
    for name in ("InstrumentContact", "Grasp", "VolumetricGrasp"):
        share = system(sim, name).delta_k
        host = share.numpy()
        host[5, 0] = (1.0e-30, 0.0, 0.0)
        wp.copy(share, wp.array(host, dtype=wp.vec3, device=share.device))
    inject(sim)
    episode.run(2, True)
    set_hook(sim)  # released within a step of this energised episode, so it is set at the save point itself
    return episode


@pytest.mark.parametrize("device,captured", MODES)
def test_chole_episode_resumes(device, captured):
    cfg, a, m = authored(device)
    state, control = m.state(), m.control()
    sim = runtime(cfg, a, m, captured, state=state, control=control)
    early = m.state(), m.control()
    episode = chole_episode(sim, early)
    live = {("Grasp", "active"), ("VolumetricGrasp", "active"), ("Clipping", "clipped"), ("Thermal", "heat"), ("Thermal", "burn"),
            ("Thermal", "coag"), ("Bleeding", "source"), ("Smoke", "pos_age"), ("Fluid", "cursor"), ("InstrumentContact", "delta_k")}
    for name, key in sorted(live | ({("Hook", "active")} if device == "cpu" else set())):
        assert getattr(system(sim, name), key).numpy().any(), f"the episode left {name}.{key} at its default"
    assert system(sim, "Surface").appearance.visual.numpy().any()
    fresh = []
    for _ in range(1 if device == "cpu" else 0):
        # Bound to the episode as it was 17 steps earlier: its fluid contact tree is built from other positions than
        # the saving runtime's, so the comparison fails unless save and restore rebuild both trees.
        s, c = early
        other = runtime(cfg, a, m, captured, state=s, control=c)
        Episode(other)  # the live clip reach every runtime of the comparison sets
        fresh.append((other, s, c))

    def advance(steps, *sims):
        episode.run(steps // 2, True, *sims[1:])
        episode.run(steps // 2, False, *sims[1:])

    if device == "cpu":
        assert resume(sim, fresh, advance, state, control, 12) is None
    else:  # stepping is not repeatable on CUDA: the copy is exact and a restored runtime steps to finite values
        checkpoint = m.state(), m.control()
        sim.physics.save(*checkpoint)
        assert snapshot(*checkpoint) == snapshot(state, control)
        episode.run(2, True)
        sim.physics.restore(*checkpoint)
        assert snapshot(*checkpoint) == snapshot(state, control)
        episode.run(2, True)
        assert np.isfinite(state.particle_q.numpy()).all()


def test_chole_with_tissue_contact_resumes():
    """The case ships its liver-gallbladder contact disabled; enabled, its search tree is consulted every step."""
    cfg, a, m = authored("cpu", enable=("tissue_contact",))
    state, control = m.state(), m.control()
    sim = runtime(cfg, a, m, state=state, control=control)
    episode = TissueEpisode(sim)
    episode.run(45, False)
    episode.run(15, True)
    contact = system(sim, "PartContact")
    assert contact.enabled and contact.candidate_count.numpy().any() and contact.contact_count.numpy().any()
    s, c = m.state(), m.control()
    other = runtime(cfg, a, m, state=s, control=c)
    Episode(other)

    def advance(steps, *sims):
        episode.run(steps // 2, True, *sims[1:])
        episode.run(steps // 2, False, *sims[1:])

    assert resume(sim, [(other, s, c)], advance, state, control, 12) is None
    assert contact.contact_count.numpy().any()


def test_mounted_chole_tool_resumes_with_a_held_grasp_and_a_pending_reaction():
    cfg, a, m = authored("cpu", mount="grasper")
    state, control = m.state(), m.control()
    sim = Coupled(cfg, a, m, state, control)
    episode = Episode(sim)
    tool = system(sim, "InstrumentControl").names.index("grasper")
    poses = state.body_q.numpy()
    poses[:, :3] = episode.targets[0, tool, :3]
    state.body_q.assign(poses)
    episode.run(10, False)
    episode.run(12, True)
    assert system(sim, "VolumetricGrasp").active.numpy()[:, tool].any() or system(sim, "Grasp").active.numpy()[:, tool].any()
    assert sim.mounts.groups[0].force.numpy().any(), "no reaction is pending on the holder"
    s, c = m.state(), m.control()
    other = Coupled(cfg, a, m, s, c)
    Episode(other)

    def advance(steps, *sims):
        episode.run(steps // 2, True, *sims[1:])
        episode.run(steps // 2, False, *sims[1:])

    sim.save, sim.restore = sim.solver.save_i4h, sim.solver.restore_i4h
    other.save, other.restore = other.solver.save_i4h, other.solver.restore_i4h
    assert resume(sim, [(other, s, c)], advance, state, control, 10, bodies=True) is None


def endoluminal(case, device, captured):
    if case == "aorta_centreline":
        return aorta(device)
    from i4h_endoluminal.config import LOADERS, SYSTEMS, read_case
    cfg = read_case(case)
    a = author_case(cfg, loaders={**physical.LOADERS, **LOADERS}, systems={**physical.SYSTEMS, **SYSTEMS}, device=device, envs=2)
    b = newton.ModelBuilder()
    a.declare(b)
    for world in range(2):
        b.begin_world(gravity=a.scene.gravity_vector)
        a.add_world(b, world)
        b.end_world()
    a.complete(b)
    m = b.finalize(device=device)
    m.particle_grid = None
    return cfg, a, m


@pytest.mark.parametrize("case,steps,rate,contact", [("aorta_centreline", 160, 0.1, "GraphContainment"),
                                                     ("aorta_shell", 120, 0.3, "ShellContainment")])
@pytest.mark.parametrize("device,captured", MODES)
def test_catheter_cases_resume(device, captured, case, steps, rate, contact):
    """A driven catheter in wall contact; ``aorta_centreline`` also holds an active drag and a candidate cache whose
    rebuild request preparation forces: the restore brings back the saved request."""
    cfg, a, m = endoluminal(case, device, captured)
    state, control = m.state(), m.control()
    sim = runtime(cfg, a, m, captured, state=state, control=control)

    def advance(count, *sims):
        for _ in range(count):
            for runtime_ in sims:
                system(runtime_, "CatheterControl").actions.assign(np.tile(np.float32([rate, 0.0, 0.0]), (2, 1)))
                runtime_.step()

    advance(steps, sim)
    drag = next((s for s in sim.systems if type(s).__name__ == "Drag"), None)
    if drag is not None:
        tip = sim.scene.field("q", drag.body).numpy()[1, 3]
        drag.begin(1, 3, tip + np.float32([0.002, 0.0, 0.0]))
        advance(2, sim)
        assert drag.active.numpy().any()
    assert system(sim, contact).contact_count.numpy().any(), "the catheter is not in contact at the save point"
    if device != "cpu":  # two restored runtimes differ from each other on CUDA: exact copy only
        checkpoint = m.state(), m.control()
        sim.physics.save(*checkpoint)
        advance(2, sim)
        sim.physics.restore(*checkpoint)
        assert snapshot(*checkpoint) == snapshot(state, control)
        return
    s, c = m.state(), m.control()
    fresh = [(runtime(cfg, a, m, captured, state=s, control=c), s, c)]
    assert resume(sim, fresh, advance, state, control, 12) is None


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_portable_world_rows_are_independent_and_validate_before_writing(device):
    _, _, sim, _, state, control = rods(device)
    c = sim.physics.solver.continuation
    poison(71, state, control)
    saved = c.export_worlds([2, 0])
    before = snapshot(state, control)
    invalid = {k: v.copy() for k, v in saved.items()}
    last = next(reversed(invalid))
    invalid[last] = invalid[last].astype(np.float64)
    with pytest.raises(ValueError, match="shape/dtype"):
        c.import_worlds(invalid, [0, 2])
    assert snapshot(state, control) == before
    c.import_worlds(saved, [0, 2])
    swapped = c.export_worlds([0, 2])
    for key in saved:
        assert swapped[key].tobytes() == saved[key].tobytes(), key


def test_portable_snapshot_replays_in_a_fresh_single_world_runtime():
    from i4h_host.composition import load_case

    def build(envs):
        return load_case({
            "sim": {"device": "cpu", "envs": envs, "dt": 1/120, "gravity": [0, 0, 0]},
            "assets": {"catheter": {"type": "rod", "start": [0, 0, 0], "direction": [1, 0, 0],
                                    "count": 9, "segment_length": .01, "radius": .0005, "mass": .001}},
            "systems": [{"type": "particles"}, {"type": "rod_direct", "id": "rod", "body": "catheter"},
                        {"type": "catheter_control", "rod": "rod", "track_length": 0, "tip_edges": 2,
                         "track_path": [[0, 0, 0], [.1, 0, 0]], "distal_free_nodes": 5}],
        })

    original, replay = build(3), build(1)
    control = next(s for s in original.systems if s.id == "catheter_control")
    control.actions.assign(np.array([[.01, .2, .3]] * 3, np.float32))
    for _ in range(3):
        original.step()
    saved = original.physics.solver.continuation.export_worlds([2])
    replay.physics.solver.continuation.import_worlds(saved)
    actual = replay.physics.solver.continuation.export_worlds()
    for key in saved:
        assert actual[key].tobytes() == saved[key].tobytes(), key
    for _ in range(3):
        original.step()
        replay.step()
        np.testing.assert_array_equal(original.scene.field("q", original.scene.body("catheter")).numpy()[2],
                                      replay.scene.field("q", replay.scene.body("catheter")).numpy()[0])
