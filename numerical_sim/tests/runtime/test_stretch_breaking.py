# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M4 stretch breaking: over-stretched tets and edges of env 0 break and paint damage and blood; env 1 is untouched."""

import numpy as np
from helpers import find, live_surface_is_consistent, needs_chole
from i4h_isaaclab.composition import load_case
from i4h_physics.systems.mechanics.tet import Tet
from i4h_surgical.systems.mechanics.stretch_breaking import StretchBreaking
from i4h_surgical.systems.mechanics.surface import Surface


def test_stretch_breaks_env_locally():
    sim = load_case("two_worlds")
    scene, body = sim.scene, sim.scene.bodies[0]
    surface, tet, breaking = find(sim, Surface), find(sim, Tet), find(sim, StretchBreaking)
    sim.step()
    graph = sim.graph
    sdt = sim.dt / sim.substeps
    active_before, edges_before = tet.active.numpy().copy(), tet.edge_active.numpy().copy()

    # nothing near rest breaks or tints, even at a low threshold
    breaking.threshold.fill_(1.5)
    breaking.damage_start.fill_(1.1)
    breaking.damage_full.fill_(1.4)
    breaking.substep_end(scene, sdt)
    breaking.step_end(scene, sim.dt)
    assert surface.deleted.numpy().tolist() == [0, 0] and not surface.appearance.visual.numpy().any()

    # pull env 0's bottom layer three cells down: the tets touching it stretch far past 1.5
    q = scene.view(scene.q, body).numpy()
    flags = scene.view(scene.flags, body).numpy()
    z = q[0][:, 2]
    bottom = np.flatnonzero((flags[0] != 0) & (z < z[flags[0] != 0].min() + 1e-4))
    q[0][bottom, 2] -= 0.06
    scene.field("q", body).assign(q)  # the body's (E, count) view, not the whole block
    breaking.substep_end(scene, sdt)
    breaking.step_end(scene, sim.dt)
    surface.step_end(scene, sim.dt)
    v = body.variant_of_env[0]
    m = body.tet
    touching = np.flatnonzero(np.isin(m.tets[v], bottom).any(axis=1) & (m.active0[v] != 0))
    active = tet.active.numpy()
    assert (active[0][touching] == 0).all() and active[0].sum() == active_before[0].sum() - len(touching)
    assert np.array_equal(active[1], active_before[1]) and surface.deleted.numpy().tolist() == [len(touching), 0]
    edge_active = tet.edge_active.numpy()
    stretched = np.flatnonzero(np.isin(m.edges[v], bottom).sum(axis=1) == 1)  # edges from the bottom layer upward
    assert (edge_active[0][stretched] == 0).all() and np.array_equal(edge_active[1], edges_before[1])
    assert edge_active[0].sum() < edges_before[0].sum() - len(stretched)  # tets took their other edges along
    visual = surface.appearance.visual.numpy().reshape(scene.envs, -1, 4)
    painted = np.flatnonzero(visual[0][:, 0] > 0.0)
    assert set(bottom) <= set(painted) and visual[0][bottom, 0].min() == 1.0  # full damage at 3x stretch
    assert (visual[0][bottom, 2] == 0.5).all() and not visual[0][:, 1].any() and not visual[1].any()  # blood once, no coag
    live_surface_is_consistent(sim, 0)
    assert np.isfinite(scene.q.numpy()).all()

    # the accumulator is clean between substeps; with the shipped thresholds the hanging blocks paint nothing
    assert not breaking.stretch.numpy().any()
    breaking.threshold.fill_(6.0)
    breaking.damage_start.fill_(4.0)
    breaking.damage_full.fill_(5.5)
    visual_before = surface.appearance.visual.numpy().copy()
    for _ in range(5):
        sim.step()
    assert np.array_equal(surface.appearance.visual.numpy(), visual_before)
    assert sim.graph is graph and np.isfinite(scene.q.numpy()).all()

    # the edge breaker is structural
    breaking.break_edges = False
    try:
        sim.step()
        assert False, "structural change accepted"
    except RuntimeError:
        pass
    breaking.break_edges = True

    sim.reset([0])
    assert np.array_equal(tet.active.numpy(), active_before) and np.array_equal(tet.edge_active.numpy(), edges_before)
    assert not surface.appearance.visual.numpy().any() and surface.deleted.numpy().tolist() == [0, 0]


# --- M9b: per-part targeting, the chole tuning, ring death -> bleeding cut -------------------------------------------

import pytest  # noqa: E402
from i4h_isaaclab.composition import read_case  # noqa: E402
from i4h_surgical.systems.fluids.bleeding import Bleeding  # noqa: E402
from i4h_surgical.systems.instruments.grasp import Grasp  # noqa: E402
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl  # noqa: E402
from i4h_surgical.systems.instruments.volumetric_grasp import VolumetricGrasp  # noqa: E402
from i4h_surgical.systems.mechanics.centreline import Centreline  # noqa: E402


@needs_chole
def test_parts_mask():
    sim = load_case("chole_tet_med", envs=1)
    scene, surface, breaking = sim.scene, find(sim, Surface), find(sim, StretchBreaking)
    body = scene.body('tissue')
    assert breaking.parts == ("fat",) and breaking.target.numpy().tolist() == [0, 1, 0] and breaking.enabled
    assert StretchBreaking(scene, surface=surface).target.numpy().tolist() == body.tet.deletable.tolist()  # empty: every deletable part
    assert StretchBreaking(scene, surface=surface, parts="fat").parts == ("fat",)
    with pytest.raises(ValueError, match="not deletable"):
        StretchBreaking(scene, surface=surface, parts=["liver"])
    with pytest.raises(ValueError, match="unknown part"):
        StretchBreaking(scene, surface=surface, parts=["spleen"])
    sim.step()
    breaking.parts = ()
    with pytest.raises(RuntimeError, match="structural"):
        sim.step()
    with pytest.raises(ValueError, match="fixed"):
        sim.prepare()


def chole_at_rest(envs, threshold=None):
    cfg = read_case("chole_tet_med")
    if threshold is not None:
        for entry in cfg["systems"]:
            if entry["type"] == "stretch_breaking":
                entry["threshold"], entry["damage_start"], entry["damage_full"] = threshold, threshold, 2 * threshold
    sim = load_case(cfg, envs=envs)
    control = find(sim, InstrumentControl)
    actions = np.zeros((envs, control.count, 10), np.float32)
    actions[:, :, :3] = control.rest[None]
    actions[:, :, 6] = 1.0
    return sim, control, actions


def pull_fat(sim, control, actions, env=0, retract_steps=90, record=None):
    """The grasper onto the fat surface point nearest its rest pose, jaws open; close; retract toward the trocar."""
    scene, body = sim.scene, sim.scene.body('tissue')
    k = control.names.index("grasper")
    m, fat = body.tet, body.parts["fat"]
    q = scene.view(scene.q, body).numpy()[env]
    centroids = q[m.tris[0][np.flatnonzero(m.tri_part == fat.index)]].mean(axis=1)
    target = centroids[np.argmin(np.linalg.norm(centroids - control.rest[k], axis=1))]
    for _ in range(60):
        actions[env, k, :3] = target
        control.actions.assign(actions)
        sim.step()
        if record is not None:
            record()
    actions[env, k, 7] = 1.0
    for _ in range(20):
        control.actions.assign(actions)
        sim.step()
        if record is not None:
            record()
    trocar = control.trocar.numpy()[k]
    for _ in range(retract_steps):
        root = control.root.numpy()[env, k]
        d = trocar - root
        actions[env, k, :3] = root + d / np.linalg.norm(d) * 0.02
        control.actions.assign(actions)
        sim.step()
        if record is not None:
            record()
    return k


@needs_chole
def test_chole_stretch_measurement(capsys):
    """The tuning observable behind the case values (prints with -s): the fat's largest stretch while settling and under a
    grasp-and-pull with breaking disabled by a huge threshold."""
    sim, control, actions = chole_at_rest(1, threshold=1.0e9)
    breaking = find(sim, StretchBreaking)
    settle = []
    for _ in range(300):
        control.actions.assign(actions)
        sim.step()
        settle.append(float(breaking.max_stretch.numpy()[0]))
    settle = np.asarray(settle)
    pull = []
    pull_fat(sim, control, actions, record=lambda: pull.append(float(breaking.max_stretch.numpy()[0])))
    case = next(e for e in read_case("chole_tet_med")["systems"] if e["type"] == "stretch_breaking")
    with capsys.disabled():
        print(f"\nfat stretch: settle transient {settle[:100].max():.2f}, steady {settle[100:].max():.2f}; grasp and pull {max(pull):.1f}; "
              f"case damage {case['damage_start']}..{case['damage_full']}, threshold {case['threshold']}")
    assert settle[100:].max() < case["damage_start"] < case["threshold"] < max(pull)  # the case values sit between settling and pulling
    assert find(sim, Surface).deleted.numpy()[0] == 0  # nothing broke at the huge threshold


@needs_chole
def test_chole_settle_does_not_tear():
    sim, control, actions = chole_at_rest(2)
    breaking, surface = find(sim, StretchBreaking), find(sim, Surface)
    worst = np.zeros(2, np.float32)
    for step in range(600):
        control.actions.assign(actions)
        sim.step()
        if step >= 100:
            worst = np.maximum(worst, breaking.max_stretch.numpy())
    assert surface.deleted.numpy().tolist() == [0, 0]
    assert (worst < breaking.damage_start.numpy()).all(), worst
    assert not surface.appearance.visual.numpy().reshape(2, -1, 4)[:, :, 0].any()  # no damage tint at rest
    assert np.isfinite(sim.scene.q.numpy()).all()


@needs_chole
def test_chole_pull_tears_the_fat_with_damage_first():
    sim, control, actions = chole_at_rest(2)
    scene, body = sim.scene, sim.scene.body('tissue')
    _breaking, surface, tet = find(sim, StretchBreaking), find(sim, Surface), find(sim, Tet)
    grasp, vg = find(sim, Grasp), find(sim, VolumetricGrasp)
    fat = body.parts["fat"]
    seen_damage_before_tear = []

    def record():
        deleted = int(surface.deleted.numpy()[0])
        if deleted == 0:
            damage = surface.appearance.visual.numpy().reshape(2, -1, 4)[0][fat.particle_start : fat.particle_start + fat.particle_count, 0]
            seen_damage_before_tear.append(float(damage.max()))

    k = pull_fat(sim, control, actions, record=record)
    deleted = surface.deleted.numpy()
    assert deleted[0] > 0 and deleted[1] == 0, deleted
    assert max(seen_damage_before_tear) > 0.0  # the fat showed damage before the first tet went
    active0 = body.tet.active0[body.variant_of_env[0]]
    removed = np.flatnonzero((active0 != 0) & (tet.active.numpy()[0] == 0))
    assert (body.tet.part[removed] == fat.index).all()  # only the fat tears
    visual = surface.appearance.visual.numpy().reshape(2, -1, 4)
    others = np.ones(body.count, bool)
    others[fat.particle_start : fat.particle_start + fat.particle_count] = False
    assert not visual[0][others, 0].any() and not visual[1].any()
    assert grasp.count.numpy()[0, k] + vg.count.numpy()[0, k] >= 0 and np.isfinite(scene.q.numpy()).all()
    live_surface_is_consistent(sim, 0)


@needs_chole
def test_ring_death_marks_a_bleeding_cut():
    """A clamp ring cut down to one live particle (torn or burnt away) counts as a cut of its node: the artery bleeds."""
    from helpers import kill

    sim = load_case("chole_tet_med", envs=2)
    _scene, body = sim.scene, sim.scene.body('tissue')
    surface, centreline, bleeding = find(sim, Surface), find(sim, Centreline), find(sim, Bleeding)
    c, m, v = body.centrelines, body.tet, body.variant_of_env[0]
    for _ in range(5):
        sim.step()
    node = (bleeding.node_lo + bleeding.node_hi) // 2
    ring = c.clamp[0, c.clamp_start[0, node] : c.clamp_start[0, node] + c.clamp_count[0, node]]
    assert len(ring) >= 2 and centreline.active_count.numpy()[0, node] == len(ring)
    for pid in ring[1:]:  # every tet incident to all but one clamp particle of env 0
        for t in np.flatnonzero((m.tets[v] == pid).any(axis=1) & (surface.tet.active.numpy()[0] != 0)):
            kill(sim, surface, 0, int(t))
    sim.step()
    assert centreline.active_count.numpy()[0, node] < 2 and centreline.active_count.numpy()[1, node] == len(ring)
    assert bleeding.cut.numpy()[0, node] == 1 and bleeding.cut.numpy()[1].sum() == 0
    sim.step()
    assert bleeding.sources.numpy()[0] > 0 and bleeding.sources.numpy()[1] == 0
    sim.reset([0])
    assert bleeding.cut.numpy()[0].sum() == 0 and centreline.active_count.numpy()[0, node] == len(ring)
