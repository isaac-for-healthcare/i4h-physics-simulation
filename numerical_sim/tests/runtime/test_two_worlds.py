# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M0 checks on the two_worlds case: env isolation, masked reset, capture, live/structural settings."""


import newton
import numpy as np
import pytest
import warp as wp
from conftest import env_slice
from helpers import find, needs_chole, push_actions, run, tet_volumes
from i4h_isaaclab.composition import load_case, resolve_case
from i4h_physics import Scene, Stepper
from i4h_physics.debug import alias_checks
from i4h_physics.loaders import add_tet_grid
from i4h_physics.scene import ACTIVE_INT
from i4h_physics.systems.mechanics.bounds import Bounds
from i4h_physics.systems.mechanics.particles import Particles
from i4h_physics.systems.mechanics.tet import Tet
from i4h_standalone.app.instruments import scripted_actions
from i4h_surgical.loaders.instrument_usd import add_instrument
from i4h_surgical.systems.contact.instrument_contact import InstrumentContact
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
from i4h_surgical.systems.instruments.thermal import Thermal
from i4h_surgical.systems.mechanics.surface import Surface


def test_env_isolation(case):
    sim = load_case(case)
    scene, body = sim.scene, sim.scene.bodies[0]
    control = find(sim, InstrumentControl)

    run(sim, 100)
    baseline = scene.view(scene.q, body).numpy().copy()
    assert np.isfinite(baseline).all()
    assert baseline[:, :, 2].min() >= 0.0  # floor holds

    sim.reset()
    run(sim, 100, push_actions(control, scene.envs))
    q = scene.view(scene.q, body).numpy()
    assert np.isfinite(q).all()
    # Cross-run comparison: the tet constraints accumulate with float atomics, whose order is
    # not deterministic, so env 1 matches the baseline run to a tolerance, not bitwise. Measured
    # run-to-run noise on the hanging block is far below 1e-3; an indexing leak is centimetres.
    assert np.allclose(q[1], baseline[1], rtol=1e-4, atol=1e-3), "contact in env 0 leaked into env 1"
    assert np.abs(q[0] - baseline[0]).max() > 1e-2, "sphere push had no effect in env 0"


def test_masked_reset(case):
    sim = load_case(case)
    scene, body = sim.scene, sim.scene.bodies[0]
    control = find(sim, InstrumentControl)
    rest = control.rest

    run(sim, 50, lambda t: scripted_actions(rest, scene.envs, t))
    q_before = env_slice(scene, scene.q, body, 0)
    qd_before = env_slice(scene, scene.qd, body, 0)
    sphere_before = control.sphere_q.numpy()[0].copy()
    assert not np.array_equal(control.sphere_q.numpy()[1], rest)

    sim.reset([1])
    assert np.array_equal(env_slice(scene, scene.q, body, 0), q_before)
    assert np.array_equal(env_slice(scene, scene.qd, body, 0), qd_before)
    assert np.array_equal(control.sphere_q.numpy()[0], sphere_before)
    assert np.array_equal(env_slice(scene, scene.q, body, 1), env_slice(scene, scene.q0, body, 1))
    assert np.all(env_slice(scene, scene.qd, body, 1) == 0.0)
    assert np.array_equal(control.sphere_q.numpy()[1], rest)
    assert np.array_equal(control.actions.numpy()[1][:, :3], rest) and control.actions.numpy()[1][:, 7].max() == 0.0

    sim.reset(wp.array([1], dtype=wp.int32, device=scene.device))
    assert np.array_equal(env_slice(scene, scene.q, body, 0), q_before)

    # Topology masks are per-env state too: deactivate env 1, reset it, env 0 untouched.
    tet = find(sim, Tet)
    active_env0 = tet.active.numpy()[0].copy()
    tet.active[1].fill_(0)
    tet.edge_active[1].fill_(0)
    sim.reset([1])
    assert tet.active.numpy()[1].sum() == 5 * 27 and tet.edge_active.numpy()[1].sum() > 0
    assert np.array_equal(tet.active.numpy()[0], active_env0)


def test_captured_equals_uncaptured(case):
    captured = load_case(case)
    eager = load_case(case)
    eager.cuda_graph = False
    rest = find(captured, InstrumentControl).rest
    def actions(t):
        return scripted_actions(rest, captured.scene.envs, t)

    run(captured, 100, actions)
    run(eager, 100, actions)
    assert captured.graph is not None and eager.graph is None
    # Tolerance because of float atomics in the tet solve (see test_env_isolation).
    for b in captured.scene.bodies:
        if not b.integrate:
            continue  # splashing water is chaotic run to run; the fluid tests compare it in a quiet setting
        assert np.allclose(captured.scene.view(captured.scene.q, b).numpy(), eager.scene.view(eager.scene.q, b).numpy(), rtol=1e-4, atol=1e-3), b.name
        assert np.allclose(captured.scene.view(captured.scene.qd, b).numpy(), eager.scene.view(eager.scene.qd, b).numpy(), rtol=1e-3, atol=1e-2), b.name


def test_live_stiffness_no_recapture(case):
    sim = load_case(case)
    scene, body = sim.scene, sim.scene.bodies[0]
    control, contact = find(sim, InstrumentControl), find(sim, InstrumentContact)
    actions = push_actions(control, scene.envs)

    run(sim, 50, actions)
    graph = sim.graph
    stiff = env_slice(scene, scene.q, body, 0)

    sim.reset()
    contact.stiffness.fill_(0.0)
    run(sim, 50, actions)
    soft = env_slice(scene, scene.q, body, 0)

    assert sim.graph is graph
    assert not np.array_equal(stiff, soft)
    assert np.isfinite(soft).all()


def test_live_max_velocity_no_recapture(case):
    """The substep motion clip lives in particles.finalize as a live parameter; it bounds the velocity."""
    sim = load_case(case)
    scene, body = sim.scene, sim.scene.bodies[0]
    control, particles = find(sim, InstrumentControl), find(sim, Particles)
    actions = push_actions(control, scene.envs)

    run(sim, 50, actions)
    graph = sim.graph
    free = env_slice(scene, scene.q, body, 0)

    sim.reset()
    particles.max_velocity.fill_(0.05)  # the pushed block can barely move
    run(sim, 50, actions)
    limited = env_slice(scene, scene.q, body, 0)

    assert sim.graph is graph
    assert not np.array_equal(free, limited) and np.isfinite(limited).all()
    block_qd = scene.view(scene.qd, body).numpy().reshape(-1, 3)  # the fluid pool has its own bound (fluid.max_velocity)
    assert np.linalg.norm(block_qd, axis=1).max() <= 0.05 * (1.0 + 1e-4)


def test_velocity_clip_before_contact():
    """A small block shot at a sphere faster than max_velocity stops on top of it.

    Without the prediction clip the predicted block overshoots deep into the sphere (or past it), the
    contact pushes it out on the far side and the final displacement clip drags it back through the
    interior: measured 0.06 from the centre, then flying off. With the clip the closest points reach
    the surface on the first step (corner vertices lag at 0.16) and the block rests at 0.2."""
    scene = Scene("cuda:0", 1, (0.0, 0.0, 0.0))
    add_tet_grid(scene, "p", dim=(1, 1, 1), cell=0.02, origin=(-0.01, -0.01, 0.24), parts={"p": {"spring": 1.0, "volume": 1.0}})
    add_instrument(scene, "ball", "sphere", radius=0.2, initial={"position": (0.0, 0.0, 0.0)})
    scene.allocate()
    scene.qd.assign(np.tile(np.float32([0.0, 0.0, -20.0]), (scene.particle_count, 1)))  # 0.33 per step, past the centre
    particles = Particles(scene, damping=1.0, max_velocity=6.0)  # allows 0.1 per step
    control = InstrumentControl(scene)
    tet = Tet(scene, "p")
    tet.spring.fill_(0.0)  # a free cloud of particles: only the surface triangles matter here
    tet.volume.fill_(0.0)
    surface = Surface(scene, tet=tet)
    sim = Stepper.create(scene, [particles, tet, surface, control, InstrumentContact(scene, instruments=control, surface=surface, margin=0.0)],
                    dt=1.0 / 60.0, substeps=1, iterations=1)
    distances = []
    for _ in range(10):
        sim.step()
        distances.append(float(np.linalg.norm(scene.q.numpy(), axis=1).min()))
    assert min(distances) >= 0.15, distances
    assert abs(distances[-1] - 0.2) < 3e-3, distances  # resting on the sphere


def test_contact_events_capped_and_reset(case):
    """More contact events than solver iterations must not skip sweep intervals (the cap makes both configurations
    identical launches), and a masked reset clears only that env's contact records."""
    results = []
    for events in (3, 1):
        sim = load_case(case)
        sim.iterations = 1
        contact = find(sim, InstrumentContact)
        contact.iterations = events
        control = find(sim, InstrumentControl)
        rest = control.rest
        q0 = env_slice(sim.scene, sim.scene.q, sim.scene.bodies[0], 0)
        run(sim, 40, lambda t: scripted_actions(rest, sim.scene.envs, t))
        results.append(env_slice(sim.scene, sim.scene.q, sim.scene.bodies[0], 0))  # the block; water is chaotic run to run
        assert contact.record_count.numpy().max() > 0
        assert np.abs(results[-1] - q0).max() > 5e-3  # the probe pushed the block
    assert np.allclose(results[0], results[1], atol=1e-4)  # same launches up to atomic noise

    sim = load_case(case)
    contact, control = find(sim, InstrumentContact), find(sim, InstrumentControl)
    rest = control.rest
    run(sim, 40, lambda t: scripted_actions(rest, sim.scene.envs, t))
    counts = contact.record_count.numpy()
    assert (counts > 0).all(), counts  # both probes touch their block
    sim.reset([0])
    after = contact.record_count.numpy()
    assert after[0].max() == 0 and np.array_equal(after[1], counts[1])


@pytest.mark.parametrize("records", [1, 8])
def test_contact_record_overflow_readout(records):
    """record_overflow counts the contacts beyond the record slots each step (a readout, not an observation)."""
    from i4h_isaaclab.composition import read_case

    cfg = read_case("two_worlds")
    for entry in cfg["systems"]:
        if entry["type"] == "instrument_contact":
            entry["records"] = records
    sim = load_case(cfg)
    contact, control = find(sim, InstrumentContact), find(sim, InstrumentControl)
    assert set(contact.readouts()) == {"record_overflow"}
    rest, overflow, found = control.rest, 0, 0
    for step in range(40):
        control.actions.assign(scripted_actions(rest, sim.scene.envs, step * sim.dt))
        sim.step()
        overflow = max(overflow, int(contact.record_overflow.numpy().max()))
        found = max(found, int(contact.record_count.numpy().max()))
    assert found > 1, "the probe touched the block with several contacts per capsule"
    assert (overflow > 0) == (found > records), (records, found, overflow)
    sim.reset([0])
    wp.synchronize()
    assert contact.record_overflow.numpy()[0] == 0


def test_frozen_structural_fields_rejected(case):
    sim = load_case(case)
    sim.step()
    contact = find(sim, InstrumentContact)
    contact.records = 4  # sizes the record buffers: cannot change after construction
    with pytest.raises(RuntimeError, match="structural"):
        sim.step()
    with pytest.raises(ValueError, match="fixed at construction"):
        sim.prepare()
    contact.records = 8
    sim.prepare()
    sim.step()
    assert np.isfinite(sim.scene.q.numpy()).all()


def test_selection_fixed_and_bounds_recaptured(case):
    """A structural selection (thermal.instrument) derives host state at construction: rebinding it must fail in
    prepare() instead of recapturing stale launches. A structural value read at the launch site (bounds.lower) is
    picked up by the recapture."""
    sim = load_case(case)
    sim.step()
    thermal = find(sim, Thermal)
    thermal.instrument = "hook2"  # k and source_prim were derived from "probe"
    with pytest.raises(RuntimeError, match="structural"):
        sim.step()
    with pytest.raises(ValueError, match="thermal.instrument"):
        sim.prepare()
    thermal.instrument = "probe"
    sim.prepare()  # a failed prepare() leaves the simulator unprepared; step() would otherwise recapture on its own
    sim.step()

    bounds = find(sim, Bounds)
    bounds.lower = (-0.5, -0.5, 0.06)  # the floor rises into the block's free bottom layer (z 0.05)
    with pytest.raises(RuntimeError, match="structural"):
        sim.step()
    sim.prepare()
    sim.step()
    scene, block = sim.scene, sim.scene.body("block")
    q, flags, inv_mass = (scene.view(a, block).numpy() for a in (scene.q, scene.flags, scene.inv_mass))
    free = ((flags & ACTIVE_INT) != 0) & (inv_mass > 0.0)
    assert q[free][:, 2].min() >= 0.06 - 1e-5, "the recaptured graph projects onto the rebound floor"
    sim.reset([0])
    sim.step()
    assert np.isfinite(scene.q.numpy()).all()


def test_reset_ids_out_of_range(case):
    sim = load_case(case)
    sim.step()
    E = sim.scene.envs
    for bad in ([E], [-1], np.array([0, E])):
        with pytest.raises(ValueError, match="env_ids"):
            sim.reset(bad)
    before = sim.scene.q.numpy()
    sim.reset(wp.array([E, -1], dtype=wp.int32, device=sim.scene.device))  # device ids: guarded, no write
    assert np.array_equal(before, sim.scene.q.numpy())


def test_device_ids_are_range_checked_before_the_cast_in_debug(case):
    """4294967296 wraps to env 0 in the int32 cast: debug rejects it (and resets nothing); without debug, device ids
    out of range are still ignored without a sync."""
    import torch

    for debug in (True, False):
        sim = load_case(case, debug=debug)
        sim.step()
        before = sim.scene.q.numpy()
        wrapped = torch.tensor([4294967296], dtype=torch.int64, device="cuda:0")
        with pytest.raises(ValueError, match="env_ids must be integers"):  # any mode: a dtype check, no sync
            sim.reset(torch.tensor([0.5], device="cuda:0"))
        assert np.array_equal(before, sim.scene.q.numpy())
        if debug:
            with pytest.raises(ValueError, match="env_ids must be in"):
                sim.reset(wrapped)
            with pytest.raises(ValueError, match="env_ids must be in"):
                sim.reset(wp.array([sim.scene.envs], dtype=wp.int32, device=sim.scene.device))
            assert np.array_equal(before, sim.scene.q.numpy()), "nothing was reset"
        else:
            sim.reset(torch.tensor([sim.scene.envs], dtype=torch.int64, device="cuda:0"))
            assert np.array_equal(before, sim.scene.q.numpy())


def test_structural_rejected(case):
    sim = load_case(case)
    sim.step()

    sim.iterations += 1
    with pytest.raises(RuntimeError, match="prepare"):
        sim.step()
    sim.prepare()
    sim.step()

    find(sim, Bounds).enabled = False
    with pytest.raises(RuntimeError, match="prepare"):
        sim.step()
    sim.prepare()
    sim.step()

    sim.systems[0] = Particles(sim.scene)  # same settings, different instance and arrays
    with pytest.raises(RuntimeError, match="prepare"):
        sim.step()
    sim.prepare()
    sim.step()


@wp.kernel
def _copy_first(src: wp.array(dtype=wp.float32), dst: wp.array(dtype=wp.float32)):
    dst[0] = src[0]


def test_alias_check(case):
    sim = load_case(case, debug=True)
    run(sim, 5)
    a = wp.zeros(4, dtype=wp.float32, device=sim.scene.device)
    b = wp.zeros(4, dtype=wp.float32, device=sim.scene.device)
    with alias_checks():
        wp.launch(_copy_first, dim=1, inputs=[a], outputs=[b], device=sim.scene.device)
        with pytest.raises(RuntimeError, match="aliases"):
            wp.launch(_copy_first, dim=1, inputs=[a], outputs=[a[0:1]], device=sim.scene.device)


def test_launches_are_batched_over_envs(case):
    """One launch covers all envs: the launch count per step does not depend on env count, and
    every launch's leading dimension is the env count."""
    counts = {}
    for envs in (1, 3):
        sim = load_case(case, envs=envs)
        sim.cuda_graph = False
        sim.step()  # module load
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
    assert counts[1] == counts[3] > 0


def test_estimate_bytes(case):
    sim = load_case(case)
    scene = sim.scene
    n = scene.particle_count
    estimate = scene.estimate_bytes()
    # The main block is Newton's model and state (q0, inv_mass, radius, flags on the model; q, qd, f on the state) plus the
    # scene-owned q_prev and flags0; the pools are the private block.
    assert estimate["q_prev"] == n * 12 and estimate["flags0"] == n * 4
    assert estimate["newton_state"] == n * 36 and estimate["newton_model"] == n * 44 and "newton_grid" not in estimate
    assert scene.estimate_bytes(construction=True)["newton_grid"] > 0  # finalize allocates it; build_model drops it
    assert estimate["pool_q"] == scene.pool_count * 12 and "q" not in estimate
    # The preflight names the two largest arrays and fails before any allocation.
    with pytest.raises(MemoryError, match=r"largest: big .*, medium"):
        scene.check_memory({"small": 1, "medium": 10**14, "big": 10**15})


def test_active_bit_matches_newton():
    import newton
    from i4h_physics.scene import ACTIVE_INT

    assert ACTIVE_INT == int(newton.ParticleFlags.ACTIVE)


def test_reference_resolution(case, tmp_path):
    text = resolve_case(case).read_text()
    assert "instruments: instrument_control" in text and "rate: 200.0" in text

    bad = tmp_path / "missing.yaml"
    bad.write_text(text.replace("instruments: instrument_control", "instruments: nonexistent"))
    with pytest.raises(ValueError, match="must name an earlier system"):
        load_case(bad)

    bad.write_text(text.replace("instruments: instrument_control", "instruments: particles"))
    with pytest.raises(TypeError, match="instrument_control"):
        load_case(bad)

    # A plain value is never turned into a system, even when it spells a system id.
    bad.write_text(text.replace("rate: 200.0", "rate: instrument_control"))
    with pytest.raises(ValueError):
        load_case(bad)


def test_inactive_particles(case):
    sim = load_case(case)
    scene, body = sim.scene, sim.scene.bodies[0]
    control = find(sim, InstrumentControl)
    active = int(newton.ParticleFlags.ACTIVE)

    scene.view(scene.flags, body)[0].fill_(0)  # deactivate env 0
    before = env_slice(scene, scene.q, body, 0)
    run(sim, 20, push_actions(control, scene.envs))
    assert np.array_equal(env_slice(scene, scene.q, body, 0), before), "inactive particles moved"
    assert not np.array_equal(env_slice(scene, scene.q, body, 1), env_slice(scene, scene.q0, body, 1))

    sim.reset([0])
    assert np.all(scene.view(scene.flags, body).numpy()[0] == active)
    run(sim, 20, push_actions(control, scene.envs))
    assert not np.array_equal(env_slice(scene, scene.q, body, 0), before), "reactivated particles did not move"


def test_variants(case):
    """Env 0 runs the 4x4x4 grid, env 1 the 3x3x3 grid padded to the same capacity."""
    sim = load_case(case)
    scene, body = sim.scene, sim.scene.bodies[0]
    tet = find(sim, Tet)
    flags = scene.view(scene.flags, body).numpy()
    inv_mass = scene.view(scene.inv_mass, body).numpy()
    assert body.count == 125 and body.parts["block"].particle_count == 125
    assert (flags[0] != 0).sum() == 125 and (flags[1] != 0).sum() == 64
    assert np.all(inv_mass[1][flags[1] == 0] == 0.0)
    active = tet.active.numpy()
    assert active[0].sum() == 5 * 64 and active[1].sum() == 5 * 27  # five tets per grid cell

    # Immutable mesh data is stored once per variant; envs point at it.
    assert body.variants == 2 and body.variant_of_env.tolist() == [0, 1]
    assert tet.tets.shape == (2, 320) and tet.active.shape == (2, 320)

    run(sim, 100)
    q = scene.view(scene.q, body).numpy()
    assert np.isfinite(q).all()
    edges, edge_active, rest = tet.edges.numpy(), tet.edge_active.numpy(), tet.edge_rest.numpy()
    tets, tet_active, rest_volume = tet.tets.numpy(), tet.active.numpy(), tet.rest_volume.numpy()
    for env in range(scene.envs):
        v = body.variant_of_env[env]
        live = edge_active[env] != 0
        a, b = edges[v][live, 0], edges[v][live, 1]
        stretch = np.linalg.norm(q[env][a] - q[env][b], axis=1) / rest[v][live]
        assert np.abs(stretch - 1.0).max() < 0.3, f"env {env}: block lost its shape (max stretch {stretch.max():.2f})"
        # Volume constraint at work: measured median 0.995, worst 0.92 after 100 steps.
        live_t = tet_active[env] != 0
        ratio = tet_volumes(q[env], tets[v][live_t]) / rest_volume[v][live_t]
        assert abs(np.median(ratio) - 1.0) < 0.02 and np.abs(ratio - 1.0).max() < 0.15, f"env {env}: volume drift"
    assert scene.part_view(scene.q, body, "block").shape == (2, 125)


def test_padded_triangles_stay_in_their_part():
    """Smaller variants pad triangles with degenerate ones inside the part's particle range (gl subtracts the part offset)."""
    from i4h_physics.loaders.grid import grid_variant
    from i4h_physics.loaders.tet_bundle import pack_tet_body, parse_materials

    def big():
        return grid_variant((3, 3, 3), 0.02, (0.0, 0.0, 0.0), 1000.0, 0.005, False)
    def small():
        return grid_variant((2, 2, 2), 0.02, (0.0, 0.0, 0.0), 1000.0, 0.005, False)
    variants = [{"a": big(), "b": big()}, {"a": small(), "b": small()}]
    materials, visuals = parse_materials({"a": {"spring": 1.0, "volume": 1.0}, "b": {"spring": 1.0, "volume": 1.0}})
    scene = Scene("cuda:0", 2, (0.0, 0.0, -9.81))
    body = pack_tet_body(scene, "two", variants, materials, 0.005, visuals)
    tet = body.tet
    assert tet.uvs.shape == (2, body.count, 2) and not tet.uvs.any()
    for v, variant in enumerate(variants):
        for name, p in body.parts.items():
            rows = tet.tris[v, p.tri_start : p.tri_start + p.tri_count]
            assert rows.min() >= p.particle_start and rows.max() < p.particle_start + p.particle_count
            used = len(variant[name].tris)
            assert (rows[used:] == p.particle_start).all()  # degenerate padding
            assert (rows - p.particle_start).min() >= 0


def test_two_bodies_keep_their_offsets():
    """Two identical hanging blocks, each with its own tet system, deform identically relative
    to their origins. A forgotten body offset in any topology kernel breaks this."""
    scene = Scene("cuda:0", 2, (0.0, 0.0, -9.81))
    origin_a, origin_b = (-0.04, -0.04, 0.05), (0.26, -0.04, 0.05)
    a = add_tet_grid(scene, "a", [4, 4, 4], 0.02, origin=origin_a, fix_top=True)
    b = add_tet_grid(scene, "b", [4, 4, 4], 0.02, origin=origin_b, fix_top=True)
    scene.allocate()
    systems = [Particles(scene), Tet(scene, "a"), Tet(scene, "b"),
               Bounds(scene, lower=(-1.0, -1.0, 0.0), upper=(1.0, 1.0, 1.0))]
    sim = Stepper.create(scene, systems, dt=1.0 / 60.0, substeps=4, iterations=8)
    for _ in range(100):
        sim.step()
    qa = scene.view(scene.q, a).numpy() - np.float32(origin_a)
    qb = scene.view(scene.q, b).numpy() - np.float32(origin_b)
    assert np.isfinite(qa).all() and np.isfinite(qb).all()
    assert np.allclose(qa, qb, atol=1e-3), "the two bodies deformed differently"
    sag = (scene.view(scene.q0, a).numpy() - scene.view(scene.q, a).numpy())[..., 2].max()
    assert sag > 1e-3, "blocks did not deform at all"


def test_isaaclab_stub_loop(case):
    torch = pytest.importorskip("torch")
    from i4h_isaaclab.views import action_views, observation, observation_views, state_views, write_actions

    sim = load_case(case)
    scene, body = sim.scene, sim.scene.bodies[0]
    control = find(sim, InstrumentControl)
    E = scene.envs
    dev = wp.device_to_torch(scene.device)

    # Zero-copy views the env holds. Observations are per body (strided views of the world-major main block: the
    # block and the M5 rod; the M6 pool lives in the pool block and is observed as a count).
    acts, obs, state = action_views(sim), observation_views(sim), state_views(sim)
    assert len(acts) == 1 and acts[0].shape == (E, 10 * control.count) and acts[0].data_ptr() == control.actions.ptr
    obs_t = obs[0]
    assert obs_t.shape == (E, 3 * body.count) and obs_t.data_ptr() == scene.view(scene.q, body).ptr
    rod = scene.body("rod").count
    assert observation(obs, state).shape == (E, 3 * body.count + 3 * rod + 12)  # block, rod, then surface x2 / tool force (3) + count + jaw load / cutting / stretch / fluid / suction / smoke

    policy_action = torch.zeros((E, 10), device=dev)
    policy_action[:, 6] = 1.0  # identity rotation, jaw open
    policy_action[0, :3] = torch.tensor([0.0, 0.0, 0.02], device=dev)
    policy_action[1, :3] = torch.tensor([0.0, 0.0, 0.25], device=dev)
    decimation = 2
    for _ in range(5):  # DirectRLEnv.step order: apply action, physics step, per decimation substep
        for _ in range(decimation):
            write_actions(acts, policy_action)
            sim.step()
        assert torch.isfinite(observation(obs, state)).all()

    q_env0 = obs_t[0].clone()
    sim.reset(torch.tensor([1], dtype=torch.int32, device=dev))
    torch.cuda.synchronize()
    assert torch.equal(obs_t[0], q_env0)
    assert torch.equal(obs_t[1], wp.to_torch(scene.view(scene.q0, body)).reshape(E, -1)[1])


PEAK_PROBE = """
import sys, warp as wp
wp.init()
import i4h_physics.scene as S
from i4h_isaaclab.composition import load_case, read_case
from i4h_physics.newton_model import build_model, rebuild_particle_bvh
def allocate(self):
    dev = self.device
    wp.synchronize()
    base = wp.get_mempool_used_mem_current(dev)
    model = build_model(self)
    state = model.state()
    rebuild_particle_bvh(model, state)
    wp.synchronize()
    peak, steady = wp.get_mempool_used_mem_high(dev) - base, wp.get_mempool_used_mem_current(dev) - base
    est, est_steady = self.estimate_bytes(construction=True), self.estimate_bytes()
    newton = lambda e: sum(v for k, v in e.items() if k.startswith("newton"))
    print(peak, steady, newton(est), newton(est_steady))
    raise SystemExit(0)
S.Scene.allocate = allocate
load_case(read_case(sys.argv[1]), envs=int(sys.argv[2]))
"""


@pytest.mark.parametrize("case, envs", [("two_worlds", 64), pytest.param("chole_tet_med", 16, marks=needs_chole)])
def test_newton_memory_estimate_covers_construction(case, envs):
    """The construction estimate checked before finalize covers the measured peak of building the model (a fresh
    process: the mempool high-water mark is process-wide); the steady estimate covers what stays allocated."""
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-c", PEAK_PROBE, case, str(envs)], check=True, capture_output=True, text=True)
    peak, steady, est, est_steady = map(int, out.stdout.split()[-4:])
    assert peak <= est, f"construction peak {peak / 1e6:.1f} MB exceeds the estimate {est / 1e6:.1f} MB"
    assert steady <= est_steady, f"steady {steady / 1e6:.1f} MB exceeds the estimate {est_steady / 1e6:.1f} MB"
    assert est <= 1.25 * peak + 4e6, "the construction estimate should stay close to the measured peak"
