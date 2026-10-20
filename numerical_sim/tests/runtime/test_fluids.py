# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M6 fluids: pool bodies, the position-based solve, emitters, contact, suction, bleeding."""

import numpy as np
import pytest
import warp as wp
from helpers import find, needs_chole, quiet, rest_actions, run
from i4h_isaaclab.composition import load_case, read_case
from i4h_physics.scene import ACTIVE_INT
from i4h_physics.systems.contact.fluid_contact import FluidContact
from i4h_physics.systems.fluids.fluid import Fluid
from i4h_physics.systems.mechanics.bounds import Bounds
from i4h_physics.systems.mechanics.tet import Tet
from i4h_surgical.systems.fluids.bleeding import Bleeding
from i4h_surgical.systems.fluids.emitter import Emitter
from i4h_surgical.systems.fluids.suction import Suction
from i4h_surgical.systems.instruments.clipping import Clipping
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl


def activate(scene, body, env, positions, velocity=(0.0, 0.0, 0.0)):
    """Bring env's first ``len(positions)`` pool slots to life at ``positions`` (host writes, between steps)."""
    n = len(positions)
    for name, value in (("q", positions), ("q_prev", positions), ("qd", np.tile(np.float32(velocity), (n, 1)))):
        view = scene.field(name, body)
        host = view.numpy()
        host[env, :n] = value
        view.assign(host)
    view = scene.field("flags", body)
    host = view.numpy()
    host[env, :n] = ACTIVE_INT
    view.assign(host)


def lattice(origin, dims, spacing, jitter=0.1, seed=0):
    """A rest-spaced block with a little deterministic jitter: a perfectly aligned column dropped onto the hard
    floor clamp lands its particles on the same point, and two coincident particles have no gradient between
    them to separate again (emitters always jitter, so only a synthetic lattice can reach that state)."""
    i, j, k = np.mgrid[: dims[0], : dims[1], : dims[2]]
    grid = np.stack([i, j, k], axis=-1).reshape(-1, 3).astype(np.float32)
    noise = np.random.default_rng(seed).uniform(-jitter, jitter, grid.shape).astype(np.float32)
    return ((grid + noise) * spacing + np.float32(origin)).astype(np.float32)


def alive_positions(scene, body, env):
    q = scene.field("q", body).numpy()[env]
    flags = scene.field("flags", body).numpy()[env]
    return q[(flags & ACTIVE_INT) != 0]


def test_fluid_pool_body(case):
    sim = quiet(load_case(case))
    scene = sim.scene
    water = scene.body("water")
    assert not water.integrate and water.fluid is not None and list(water.parts) == ["water"]
    assert (scene.field("flags", water).numpy() == 0).all()
    q0 = scene.field("q0", water).numpy()
    assert q0[..., 2].max() < -100.0  # parked 1 km along gravity (two_worlds is Z-up)
    assert np.allclose(scene.field("inv_mass", water).numpy(), 1.0 / 0.001)
    fluid = find(sim, Fluid)
    assert fluid.index == {"water": 0} and fluid.iterations == [4] and fluid.rest_density[0] > 0.0
    run(sim, 20)
    assert np.array_equal(scene.field("q", water).numpy(), q0), "a parked pool never moves"
    assert (fluid.alive_count.numpy() == 0).all()


def test_column_settles_to_a_blob(case):
    sim = quiet(load_case(case))
    scene = sim.scene
    water = scene.body("water")
    fluid = find(sim, Fluid)
    rd = fluid.rest_distance[0]
    column = lattice((-0.32, -0.015, 0.05), (6, 6, 7), rd)  # 252 particles, a rest-spaced column above the floor
    activate(scene, water, 0, column)
    run(sim, 150)
    q = alive_positions(scene, water, 0)
    assert q.shape[0] == column.shape[0] and np.isfinite(q).all()
    assert q[:, 2].min() >= -1e-5, "the bounds floor holds the water (bounds is the last projector in the list)"
    assert q[:, 2].max() < 0.05, "the column fell"
    assert 0.003 < q[:, 2].max() and np.ptp(q[:, 0]) < 0.12 and np.ptp(q[:, 1]) < 0.12, "cohesion keeps a blob, not a sheet"
    density = fluid.density[0].numpy().reshape(scene.envs, -1)[0][: column.shape[0]]
    assert 0.8 < density.max() < 1.7 and 0.3 < density.mean() < 1.5, (density.max(), density.mean())
    d = np.linalg.norm(q[:, None] - q[None], axis=-1) + np.eye(len(q))
    assert d.min() > 0.3 * rd, "particles do not collapse onto each other"
    v = scene.field("qd", water).numpy()[0][: column.shape[0]]
    assert np.linalg.norm(v, axis=1).max() < 0.2, "the blob has come to rest"
    alive = fluid.alive_count.numpy()
    assert alive[0, 0] == column.shape[0] and alive[1, 0] == 0
    assert np.array_equal(scene.field("q", water).numpy()[1], scene.field("q0", water).numpy()[1]), "env 1 parked"


def test_viscosity_damps_a_shear(case):
    def shear(viscosity):
        sim = quiet(load_case(case))
        scene = sim.scene
        water = scene.body("water")
        fluid = find(sim, Fluid)
        fluid.viscosity.fill_(viscosity)
        fluid.cohesion.fill_(0.0)  # cohesion alone couples the halves; measure XSPH on its own
        block = lattice((-0.32, -0.015, 0.0), (6, 6, 6), fluid.rest_distance[0])
        top = block[:, 2] > 0.015
        activate(scene, water, 0, block)
        qd = scene.field("qd", water)
        host = qd.numpy()
        host[0, : len(block), 0] = np.where(top, 0.1, -0.1)  # opposite x velocities in the top and bottom halves
        qd.assign(host)
        run(sim, 10)
        v = scene.field("qd", water).numpy()[0, : len(block), 0]
        return float(v[top].mean() - v[~top].mean())
    free = shear(0.0)
    assert free > 0.05 and abs(shear(0.2)) < 0.8 * free


def test_two_pools_one_grid_over_replays(case):
    cfg = read_case(case)
    cfg["assets"]["oil"] = {"type": "fluid", "count": 128, "mass": 0.001, "radius": 0.004, "interaction_radius": 0.02,
                            "rest_distance": 0.01, "iterations": 2}
    sims = [quiet(load_case(cfg)), quiet(load_case(cfg))]
    sims[1].cuda_graph = False
    for sim in sims:
        scene = sim.scene
        fluid = find(sim, Fluid)
        assert fluid.nf == 2 and fluid.cell == pytest.approx(0.012)
        activate(scene, scene.body("water"), 0, lattice((-0.32, -0.015, 0.05), (5, 5, 5), 0.006))
        activate(scene, scene.body("oil"), 1, lattice((0.3, -0.3, 0.05), (4, 4, 4), 0.01))
        run(sim, 100)
    a, b = sims
    assert a.graph is not None and b.graph is None
    for name, env in (("water", 0), ("oil", 1)):
        qa, qb = alive_positions(a.scene, a.scene.body(name), env), alive_positions(b.scene, b.scene.body(name), env)
        assert np.isfinite(qa).all() and np.allclose(qa, qb, rtol=1e-4, atol=1e-3), name
        fluid = find(a, Fluid)
        density = fluid.density[fluid.index[name]].numpy().reshape(2, -1)[env]
        assert density.max() > 0.2, f"{name}: its grid descriptor still answers after replays"
    fa = find(a, Fluid)
    assert (fa.alive_count.numpy() == [[125, 0], [0, 64]]).all()


def test_captured_matches_uncaptured_fluid(case):
    sims = [quiet(load_case(case)), quiet(load_case(case))]
    sims[1].cuda_graph = False
    for sim in sims:
        activate(sim.scene, sim.scene.body("water"), 0, lattice((-0.32, -0.015, 0.05), (6, 6, 6), 0.006))
        run(sim, 60)
    qa = alive_positions(sims[0].scene, sims[0].scene.body("water"), 0)
    qb = alive_positions(sims[1].scene, sims[1].scene.body("water"), 0)
    assert np.allclose(qa, qb, rtol=1e-4, atol=1e-3)


def test_fluid_masked_reset(case):
    sim = quiet(load_case(case))
    scene = sim.scene
    water = scene.body("water")
    fluid = find(sim, Fluid)
    for env in (0, 1):
        activate(scene, water, env, lattice((-0.32, -0.015, 0.05), (4, 4, 4), 0.006))
    run(sim, 30)
    before = scene.field("q", water).numpy()[0].copy()
    sim.reset([1])
    after = scene.field("q", water).numpy()
    assert np.array_equal(after[0], before)
    assert np.array_equal(after[1], scene.field("q0", water).numpy()[1])
    assert (scene.field("flags", water).numpy()[1] == 0).all()
    assert list(fluid.alive_count.numpy()[:, 0]) == [64, 0]
    run(sim, 5)
    assert np.isfinite(scene.field("q", water).numpy()).all()


def test_fluid_structural_and_live(case):
    sim = load_case(case)
    fluid = find(sim, Fluid)
    run(sim, 2)
    fluid.relaxation.fill_(0.9)  # live: no recapture
    run(sim, 2)
    fluid.grid_dim = 64
    with pytest.raises(RuntimeError, match="structural"):
        sim.step()
    with pytest.raises(ValueError, match="grid_dim"):
        sim.prepare()


def test_emitter_rate_wrap_and_isolation(case):
    sim = load_case(case)
    scene = sim.scene
    water = scene.body("water")
    fluid, emitter = find(sim, Fluid), find(sim, Emitter)
    emitter.rate.assign(np.float32([200.0, 0.0]))  # env 1 silent
    run(sim, 60)  # one second
    alive = fluid.alive_count.numpy()
    assert abs(int(alive[0, 0]) - 200) <= 1 and alive[1, 0] == 0, alive
    assert int(emitter.spawned.numpy()[0]) in (3, 4) and emitter.spawned.numpy()[1] == 0
    q = alive_positions(scene, water, 0)
    assert np.isfinite(q).all() and q[:, 2].min() >= -0.5 * fluid.rest_distance[0] and q[:, 2].max() <= 0.26
    assert np.array_equal(scene.field("q", water).numpy()[1], scene.field("q0", water).numpy()[1])
    run(sim, 60)  # 400 spawned into 256 slots: the ring wrapped and overwrote the oldest (the cursor stays below count)
    cursor = fluid.cursor.numpy()
    assert 396 % water.count <= cursor[0, 0] <= 400 % water.count and cursor[1, 0] == 0
    assert fluid.alive_count.numpy()[0, 0] == water.count
    before = scene.field("q", water).numpy()[0].copy()
    sim.reset([1])
    assert np.array_equal(scene.field("q", water).numpy()[0], before)
    assert emitter.acc.numpy()[1] == 0.0 and emitter.serial.numpy()[1] == 0 and fluid.cursor.numpy()[1, 0] == 0
    sim.reset([0])
    assert (fluid.alive_count.numpy() == 0).all() and (fluid.cursor.numpy() == 0).all()
    run(sim, 30)
    assert abs(int(fluid.alive_count.numpy()[0, 0]) - 100) <= 1


def test_emitter_config_errors(case):
    cfg = read_case(case)
    entry = next(e for e in cfg["systems"] if e["type"] == "emitter")
    entry.pop("position")
    with pytest.raises(ValueError, match="exactly one"):
        load_case(cfg)
    entry["instrument"] = "probe"
    with pytest.raises(TypeError, match="instruments"):
        load_case(cfg)
    entry["instruments"] = "instrument_control"
    sim = load_case(cfg)  # a free (trocar-less) tool emits along the live direction, gated by its grip
    scene, control = sim.scene, find(sim, InstrumentControl)
    fluid, emitter = find(sim, Fluid), find(sim, Emitter)
    emitter.direction.assign(np.tile(np.float32([0.0, 0.0, 1.0]), (scene.envs, 1)))
    emitter.speed.fill_(0.5)
    actions = np.zeros((scene.envs, control.count, 10), np.float32)
    actions[:, :, :3] = control.rest[None]
    actions[:, :, 6] = 1.0
    actions[0, 0, 7] = 1.0  # env 0 squeezes the handle: full rate; env 1 does not
    run(sim, 30, actions)
    alive = fluid.alive_count.numpy()
    assert abs(int(alive[0, 0]) - 100) <= 1 and alive[1, 0] == 0
    q = alive_positions(scene, scene.body("water"), 0)
    assert q[:, 2].max() > control.rest[0, 2] + 0.05, "the jet left the probe upward"
    actions[0, 0, 8] = 1.0  # suction wins over irrigation (and the probe's suction system may swallow a drop)
    n0 = int(alive[0, 0])
    run(sim, 10, actions)
    assert int(fluid.alive_count.numpy()[0, 0]) <= n0 and int(emitter.spawned.numpy()[0]) == 0


def tet_depth(points, tets, q):
    """Deepest penetration (m) of any point into any tet: max over points of the min signed face distance."""
    a, b, c, d = (q[tets[:, i]] for i in range(4))
    deepest = 0.0
    for p in points:
        inside = np.full(len(tets), np.inf)
        for (p0, p1, p2, opposite) in ((a, b, c, d), (a, d, b, c), (a, c, d, b), (b, d, c, a)):
            n = np.cross(p1 - p0, p2 - p0)
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
            sign = np.sign(np.einsum("ij,ij->i", n, opposite - p0))  # +1 when n points inward
            inside = np.minimum(inside, sign * np.einsum("ij,ij->i", n, p - p0))
        deepest = max(deepest, float(inside.max()))
    return deepest


def test_contact_keeps_water_out_of_the_block(case):
    """The block hangs from its +y face and droops, so there is no flat top: the check is that no drop
    ends up inside any live tet deeper than the contact shell, while drops do collect on the slab."""
    sim = load_case(case)
    scene = sim.scene
    water, block = scene.body("water"), scene.body("block")
    find(sim, Emitter).rate.fill_(400.0)
    run(sim, 200)
    tet = find(sim, Tet)
    for env in range(scene.envs):
        q = alive_positions(scene, water, env)
        assert len(q) == water.count and np.isfinite(q).all()
        tets = block.tet.tets[block.variant_of_env[env]][tet.active.numpy()[env] == 1]
        qb = scene.field("q", block).numpy()[env]
        assert tet_depth(q, tets, qb) < 0.004, f"env {env}: a drop sits inside the block"
        assert (q[:, 2] > 0.02).sum() > 40, f"env {env}: drops collect on the slab instead of falling through"
    quiet_block = load_case(case)
    quiet(quiet_block)
    run(quiet_block, 200)
    assert np.allclose(scene.field("q", block).numpy(), quiet_block.scene.field("q", block).numpy(),
                       rtol=1e-4, atol=1e-3), "one-way contact: the water does not move the block"


def test_suction_removes_and_parks(case):
    sim = quiet(load_case(case))
    scene = sim.scene
    water = scene.body("water")
    fluid, suction, control = find(sim, Fluid), find(sim, Suction), find(sim, InstrumentControl)
    puddle = lattice((-0.32, -0.015, 0.0), (6, 6, 3), fluid.rest_distance[0])
    for env in (0, 1):
        activate(scene, water, env, puddle)
    suction.power.fill_(30.0)  # 2.0 (the chole value) cannot lift water against 9.81; here the pull must win
    actions = np.zeros((scene.envs, control.count, 10), np.float32)
    actions[:, :, :3] = (-0.305, 0.0, 0.02)  # the probe dips into the puddle in both envs
    actions[:, :, 6] = 1.0
    actions[0, 0, 8] = 1.0  # only env 0 sucks
    run(sim, 60, actions)
    alive, removed = fluid.alive_count.numpy()[:, 0], suction.removed.numpy()[:, 0]
    assert removed[0] > 20 and alive[0] + removed[0] == len(puddle), (alive, removed)
    assert removed[1] == 0 and alive[1] == len(puddle)
    flags = scene.field("flags", water).numpy()[0]
    q = scene.field("q", water).numpy()[0]
    dead = (flags & ACTIVE_INT) == 0
    assert np.array_equal(q[dead], scene.field("q0", water).numpy()[0][dead]), "swallowed particles are parked at rest"
    live_q = q[~dead]
    if len(live_q):
        assert np.linalg.norm(live_q - np.float32([-0.305, 0.0, 0.02]), axis=1).min() > 0.5 * float(suction.removal_radius.numpy()[0])
    sim.reset([0])
    assert (suction.removed.numpy() == 0).all()


def test_fluid_launches_are_batched_over_envs(case):
    counts = {}
    for envs in (1, 3):
        sim = load_case(case, envs=envs)
        sim.cuda_graph = False
        sim.step()
        launches, leading = [], set()
        real = wp.launch

        def counting(kernel, dim, *args, **kwargs):
            launches.append(kernel.key)
            leading.add((dim if isinstance(dim, (tuple, list)) else (dim,))[0])
            return real(kernel, dim, *args, **kwargs)

        wp.launch = counting
        try:
            sim.step()
        finally:
            wp.launch = real
        counts[envs] = len(launches)
        assert leading == {envs}, f"envs={envs}: launch leading dims {leading}"
        assert any("lambdas" in key for key in launches) and any("push_out" in key for key in launches)
    assert counts[1] == counts[3] > 0


def test_fluid_system_order(case):
    sim = load_case(case)
    order = [type(s) for s in sim.systems]
    assert order.index(Fluid) < order.index(Emitter) < order.index(FluidContact) < order.index(Suction) < order.index(Bounds)


@needs_chole
def test_suction_tool_loads_shaft_only():
    from i4h_standalone.render.instruments import piece_skins, skin_pieces
    from i4h_surgical.loaders.instrument_usd import CHAIN_SHAFT

    sim = load_case("chole_tet_med", envs=1)
    scene, control = sim.scene, find(sim, InstrumentControl)
    rec = scene.instruments["suction"]
    assert rec.kind == "suction" and [p.name for p in rec.pieces] == ["tool"] and rec.trocar is not None
    assert all(sp.chain == CHAIN_SHAFT for sp in rec.spheres) and [p[2] for p in rec.prims] == ["shaft"]
    k = control.names.index("suction")
    run(sim, 2, rest_actions(control, 1))
    skins = piece_skins(control, scene.device)
    tool = [s for s in skins if s.instrument == k][0]
    out = [wp.zeros(s.count, dtype=wp.vec3, device=scene.device) for s in skins]
    skin_pieces(control, 0, skins, out, [wp.zeros_like(o) for o in out], scene.device)
    pts = out[skins.index(tool)].numpy()
    root, trocar = control.root.numpy()[0, k], rec.trocar
    along = (pts - root) @ ((trocar - root) / np.linalg.norm(trocar - root))
    assert along.max() > 1.0 and along.min() > -0.05, "the tube runs from the tip at the root back toward the trocar"
    assert np.linalg.norm(pts - root, axis=1).min() < 0.05, "the tip sits at the controlled point"


# The artery cut by the cutting tool lives in test_scissors.py::test_scissors_cut_the_artery (M9b: the scissors cut).


@needs_chole
def test_bleeding_sources_and_seal():
    """omnisurg's rules on a hand-cut run: the two boundaries bleed; clips strictly before and after seal the vessel."""
    sim = load_case("chole_tet_med", envs=2)
    control, fluid, bleeding, clipping = find(sim, InstrumentControl), find(sim, Fluid), find(sim, Bleeding), find(sim, Clipping)
    actions = rest_actions(control, 2)
    node = (bleeding.node_lo + bleeding.node_hi) // 2
    cut = bleeding.cut.numpy()
    cut[0, node] = cut[0, node + 1] = 1  # a two-node cut run in env 0
    bleeding.cut.assign(cut)
    run(sim, 30, actions)
    assert list(bleeding.sources.numpy()) == [2, 0] and list(bleeding.sealed.numpy()) == [0, 0]
    assert list(np.flatnonzero(bleeding.source.numpy()[0])) == [node, node + 2]
    alive = fluid.alive_count.numpy()
    assert alive[0, fluid.index["blood"]] > 100 and alive[1].sum() == 0
    clipped = clipping.clipped.numpy()
    clipped[0, node + 2] = 1  # after the run only: not sealed yet
    clipping.clipped.assign(clipped)
    run(sim, 5, actions)
    assert bleeding.sealed.numpy()[0] == 0 and bleeding.sources.numpy()[0] == 2
    clipped[0, node - 1] = 1  # and before it: sealed, no sources, no more spawns
    clipping.clipped.assign(clipped)
    run(sim, 1, actions)
    n0 = int(fluid.alive_count.numpy()[0, 0])
    run(sim, 30, actions)
    assert bleeding.sealed.numpy()[0] == 1 and bleeding.sources.numpy()[0] == 0
    assert int(fluid.alive_count.numpy()[0, 0]) == n0
    sim.reset([0])
    assert (bleeding.cut.numpy() == 0).all() and (bleeding.sealed.numpy() == 0).all() and (fluid.alive_count.numpy() == 0).all()


@needs_chole
def test_irrigation_and_suction_from_the_tube():
    sim = load_case("chole_tet_med", envs=2)
    scene, control = sim.scene, find(sim, InstrumentControl)
    fluid, emitter = find(sim, Fluid), find(sim, Emitter)
    irrigation = scene.body("irrigation")
    f = fluid.index["irrigation"]
    k = control.names.index("suction")
    actions = rest_actions(control, 2)
    actions[0, k, 7] = 1.0  # env 0 squeezes the handle
    run(sim, 1, actions)  # one step of jet: every alive particle still flies along the tube (gravity bent it 0.16 m/s)
    rec = scene.instruments["suction"]
    root = control.root.numpy()[0, k]
    axis = (root - rec.trocar) / np.linalg.norm(root - rec.trocar)
    v = scene.field("qd", irrigation).numpy()[0][(scene.field("flags", irrigation).numpy()[0] & ACTIVE_INT) != 0]
    assert len(v) >= 60 and (v @ axis / np.linalg.norm(v, axis=1)).min() > 0.95, "the jet leaves along the tube"
    assert np.linalg.norm(alive_positions(scene, irrigation, 0).mean(0) - root) < 0.15
    run(sim, 19, actions)
    alive = fluid.alive_count.numpy()
    assert 1000 < alive[0, f] <= 4096 // 3 + 200 and alive[1, f] == 0, alive  # 4096/s, one third of a second
    actions[0, k, 8] = 1.0  # suction wins: no spawns while energised, and the tube swallows what is near
    run(sim, 1, actions)
    n0 = int(fluid.alive_count.numpy()[0, f])
    run(sim, 20, actions)
    assert int(fluid.alive_count.numpy()[0, f]) <= n0 and emitter.spawned.numpy()[0] == 0


@needs_chole
def test_chole_fluid_system_order_and_masked_reset():
    sim = load_case("chole_tet_med", envs=2)
    order = [type(s) for s in sim.systems]
    assert order.index(Clipping) < order.index(Fluid) < order.index(FluidContact) < order.index(Bleeding) < order.index(Emitter) < order.index(Suction)
    control, fluid = find(sim, InstrumentControl), find(sim, Fluid)
    actions = rest_actions(control, 2)
    actions[:, control.names.index("suction"), 7] = 1.0
    run(sim, 10, actions)
    before = [scene_view.numpy()[0].copy() for scene_view in (sim.scene.field("q", sim.scene.body("irrigation")),)]
    sim.reset([1])
    after = sim.scene.field("q", sim.scene.body("irrigation")).numpy()
    assert np.array_equal(after[0], before[0]) and (fluid.alive_count.numpy()[1] == 0).all() and fluid.alive_count.numpy()[0, 1] > 0
