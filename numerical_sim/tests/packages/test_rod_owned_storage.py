# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""C3.2 ownership, history-complete continuation, command interop and accounting."""

import newton
import numpy as np
import pytest
import warp as wp
from i4h_endoluminal.loaders.centreline import add_centreline_tree
from i4h_endoluminal.systems.rods.catheter_control import CatheterControl
from i4h_endoluminal.systems.rods.control_layout import ControlLayout
from i4h_endoluminal.systems.rods.drag import Drag
from i4h_isaaclab.component import I4hComponent, I4hComponentCfg
from i4h_physics import Scene, Stepper, author_case
from i4h_physics.loaders import add_rod
from i4h_physics.rod_storage import declare_rod_direct, declare_rod_direct_material
from i4h_physics.storage import Owner
from i4h_physics.systems.mechanics.particles import Particles
from i4h_physics.systems.rods.material import RodDirectMaterial, RodPbdMaterial
from i4h_physics.systems.rods.rod_direct import RodDirect, cylinder_inv_inertia
from i4h_physics.systems.rods.rod_pbd import RodPbd
from ownership_helpers import MODES

SYSTEMS = dict(
    particles=Particles,
    rod_direct=RodDirect,
    rod_pbd=RodPbd,
    rod_direct_material=RodDirectMaterial,
    rod_pbd_material=RodPbdMaterial,
    catheter_control=CatheterControl,
    control_layout=ControlLayout,
    drag=Drag,
)


def chain(scene, name):
    return add_rod(scene, name, [0, 0, 0.2], [1, 0, 0], 7, 0.02, 0.003, mass=0.01)


def tree(scene, name, empty=False):
    positions = [[0, 0, 0], [0.1, 0, 0]] if empty else [[0, 0, 0], [0.1, 0, 0], [0.2, 0.1, 0], [0.2, -0.1, 0]]
    edges = [[0, 1]] if empty else [[0, 1], [1, 2], [1, 3]]
    b = add_centreline_tree(scene, name, positions, edges, np.full(len(positions), 0.01), root=0)
    b.anatomy.layouts = [
        [
            dict(
                pid=1,
                position=[0.1, 0, 0],
                rotation=[0, 0, 0, 1],
                stiffness=0.2,
                falloff=0.5,
                active=True,
                rest_bend=[0.1, 0],
                bend_stiffness_multiplier=0.6,
            )
        ]
    ]
    return b


def authored(device, backend="direct", regional=False, tree_body=False, empty=False, inertia="unit", bodies=1):
    assets = {
        f"b{i}": dict(type="test_tree" if tree_body else "test_chain", **(dict(empty=empty) if tree_body else {}))
        for i in range(bodies)
    }
    systems = [dict(type="particles")]
    for i in range(bodies):
        body, sid = f"b{i}", f"r{i}"
        if regional:
            systems.append(dict(type=f"rod_{backend}_material", id=f"m{i}", body=body))
        systems.append(
            dict(type=f"rod_{backend}", id=sid, body=body, **(dict(inertia=inertia) if backend == "direct" else {}))
        )
        if tree_body:
            systems.extend(
                [
                    dict(type="control_layout", id=f"l{i}", centreline=body, rod=sid),
                    dict(type="drag", id=f"d{i}", body=body),
                ]
            )
        else:
            systems.append(dict(type="catheter_control", id=f"c{i}", rod=sid, track_length=0.04, tip_edges=2))
    a = author_case(
        dict(sim=dict(device=device, envs=3, gravity=[0, 0, 0]), assets=assets, systems=systems),
        loaders=dict(test_chain=chain, test_tree=tree),
        systems=SYSTEMS,
    )
    b = newton.ModelBuilder()
    b.add_particle((9, 9, 9), (0, 0, 0), 1.0)
    for e in range(3):
        b.begin_world(gravity=(0, 0, 0))
        b.add_particle((8, 8, 8), (0, 0, 0), 1.0)
        a.add_world(b, e)
        b.end_world()
    a.complete(b)
    m = b.finalize(device=device)
    m.particle_grid = None
    return a, b, m


def runtime(a, m, captured=False, state=None, control=None):
    scene, systems = a.bind(m, m.state() if state is None else state, m.control() if control is None else control)
    sim = Stepper.create(scene, systems, cuda_graph=captured, dt=0.002, substeps=2, iterations=2)
    sim.prepare()
    return sim, {s.id: s for s in systems}


def snapshot(scene, owners=(Owner.STATE, Owner.CONTROL)):
    return {
        f.name: scene.attr(f.domain, f.key).numpy().tobytes()
        for f in scene.declarations.fields.values()
        if f.owner in owners
    }


def copy_container(dst, src, scene, owner):
    for f in scene.declarations.fields.values():
        if f.owner is owner:
            wp.copy(getattr(dst.i4h, f.name), getattr(src.i4h, f.name))
    if owner is Owner.STATE:
        wp.copy(dst.particle_q, src.particle_q)
        wp.copy(dst.particle_qd, src.particle_qd)


@pytest.mark.parametrize("device,captured", MODES)
@pytest.mark.parametrize(
    "backend,regional,inertia",
    [("direct", False, "unit"), ("direct", True, "physical"), ("pbd", False, "unit"), ("pbd", True, "unit")],
)
def test_rod_channels_commands_history_and_accounting(device, captured, backend, regional, inertia):
    a, b, m = authored(device, backend, regional, inertia=inertia, bodies=2)
    sim, s = runtime(a, m, captured)
    scene = sim.scene
    for i in range(2):
        rod, ctl = s[f"r{i}"], s[f"c{i}"]
        shared = scene.rod_body_domains[rod.body.name]
        assert rod.bend_offset is ctl.controls.bend_offset is rod.body.rod_controls.bend_offset
        assert rod.frame.size * 16 + rod.omega.size * 12 == 28 * 3 * rod.frame_count
        arrays = vars(rod.body.rod_controls)
        assert (
            sum(v.size * wp.types.type_size_in_bytes(v.dtype) for v in arrays.values())
            == 4 * 3 * shared.S + 16 * 3 * shared.K
        )
        assert ctl.targets.size * 12 + ctl.root.size * 12 == 24 * 3 and ctl.actions.size * 4 == 12 * 3
        for record in (scene.rod_domains[rod.id], shared, scene.endoluminal_domains[ctl.id]):
            for f in record.fields.values():
                backing = getattr(
                    {Owner.MODEL: m, Owner.STATE: scene.state, Owner.CONTROL: scene.control}[f.owner].i4h, f.name
                )
                assert scene.attr(f.domain, f.key).ptr == backing.ptr
                assert f.references is None
            for d in record.domains.values():
                worlds = getattr(m.i4h, d.frequency[4:] + "_world").numpy().reshape(d.shape)
                owners = getattr(m.i4h, d.frequency[4:] + "_owner").numpy().reshape(d.shape)
                assert np.all((owners == -1) | (owners >= 2))  # two foreign particles before world zero's body
                assert np.all(worlds == -1) or np.array_equal(
                    worlds, np.broadcast_to(np.arange(3).reshape((3,) + (1,) * (len(d.shape) - 1)), d.shape)
                )
        np.testing.assert_array_equal(rod.frames0.numpy()[0], rod.frame.numpy().reshape(3, rod.frame_count, 4)[0])
        if backend == "direct":
            np.testing.assert_array_equal(
                rod.inertia_local.numpy(), cylinder_inv_inertia(scene, rod.body, rod.rest_len.numpy()[0]).reshape(-1, 3)
            )
        if regional:
            provider = s[f"m{i}"]
            mat = rod.body.rod_material
            assert provider.controls is ctl.controls
            assert mat.rest_len is rod.rest_len if backend == "direct" else mat.pair is rod.pair
            rod.body.region_properties.edit(
                {"bend_multiplier" if backend == "direct" else "bend": 0.7}, "unassigned", [1]
            )
        v = sim.physics.command_views(ctl.id)
        assert sim.physics.command_views(ctl.id) is v
        assert v["actions"] is ctl.actions
        with pytest.raises(TypeError):
            v["new"] = ctl.actions
        component = I4hComponent(I4hComponentCfg(assets={}, systems=[]))
        component.sim = sim.physics
        t = component.command_views(ctl.id)
        assert component.command_views(ctl.id) is t
        for k, name in enumerate(("insertion", "rotation", "bend")):
            assert v[name].shape == (3, 1) and v[name].strides == (12, 4)
            assert v[name].ptr == ctl.actions.ptr + 4 * k
            assert t[name].stride() == (3, 1)
            t[name].fill_(0.1 * (k + 1))
        if device != "cpu":
            wp.synchronize_device(device)
        np.testing.assert_allclose(ctl.actions.numpy(), np.tile([0.1, 0.2, 0.3], (3, 1)))
        v["rotation"].fill_(0.4)
        assert np.all(wp.to_torch(ctl.actions).cpu().numpy()[:, 1] == np.float32(0.4))
    sim.step()
    before = snapshot(scene)
    for _ in range(2):
        sim.prepare()
    assert snapshot(scene) == before
    # Rebinding nondefault State and ordinary live Control must preserve every byte.
    rebound, rs = runtime(a, m, captured, state=scene.state, control=scene.control)
    assert snapshot(rebound.scene) == before
    assert rs["r0"].bend_offset.ptr == s["r0"].bend_offset.ptr
    # A history-complete rod/catheter fixture resumes bitwise from a copied checkpoint.
    state, control = m.state(), m.control()
    copy_container(state, scene.state, scene, Owner.STATE)
    copy_container(control, scene.control, scene, Owner.CONTROL)
    resumed, _ = runtime(a, m, captured, state=state, control=control)
    sim.step()
    resumed.step()
    assert snapshot(resumed.scene) == snapshot(scene)
    saved = snapshot(scene, (Owner.MODEL,))
    untouched = {k: v for k, v in snapshot(scene).items()}
    sim.reset([])
    assert snapshot(scene) == untouched
    sim.reset([1])
    once = snapshot(scene)
    sim.reset([1])
    assert snapshot(scene) == once
    assert snapshot(scene, (Owner.MODEL,)) == saved
    for i in range(2):
        rod, ctl = s[f"r{i}"], s[f"c{i}"]
        assert not ctl.actions.numpy()[1].any() and not ctl.targets.numpy()[1].any()
        assert not rod.omega.numpy().reshape(3, -1, 3)[1].any()
    sim.physics.invalidate()
    with pytest.raises(RuntimeError, match="invalidated"):
        sim.physics.command_views("c0")


@pytest.mark.parametrize("device,captured", MODES)
@pytest.mark.parametrize("empty", [False, True])
def test_tree_layout_drag_continuation_and_empty_pairs(device, captured, empty):
    a, b, m = authored(device, "pbd", True, tree_body=True, empty=empty)
    sim, s = runtime(a, m, captured)
    rod, layout, drag = s["r0"], s["l0"], s["d0"]
    assert layout.controls is rod.body.rod_controls is s["m0"].controls
    assert drag.weights.size * 4 + drag.baseline.size * 12 == 16 * 3 * drag.P
    assert drag.active.size * 4 + drag.node.size * 4 + drag.target.size * 12 == 20 * 3
    assert not drag.active.numpy().any() and not drag.weights.numpy().any()
    for key in ("pos_corr", "pos_count", "ori_corr", "ori_count"):
        assert not getattr(rod, key).numpy().any()
    if empty:
        assert rod.bend_count == 0
        f = sim.scene.rod_body_domains["b0"].fields["bend_offset"]
        assert getattr(m.i4h, f.name).size == 1 and rod.bend_offset.size == 0
    drag.begin(1, 1, [0.1, 0.02, 0])
    sim.step()
    before = snapshot(sim.scene)
    state, control = m.state(), m.control()
    copy_container(state, sim.scene.state, sim.scene, Owner.STATE)
    copy_container(control, sim.scene.control, sim.scene, Owner.CONTROL)
    other, os = runtime(a, m, captured, state=state, control=control)
    assert snapshot(other.scene) == before
    sim.prepare()
    assert snapshot(sim.scene) == before
    # Derived products and predicted frames are overwritten before use.
    os["r0"].frame_pred.fill_(wp.quat(9, 8, 7, 6))
    for key in ("stretch", "bend", "twist"):
        getattr(os["r0"].body.rod_material, key).fill_(123)
    os["d0"].ref.fill_(wp.vec3(8, 8, 8))
    sim.step()
    other.step()
    assert snapshot(sim.scene) == snapshot(other.scene)
    preserved = {k: getattr(drag, k).numpy().tobytes() for k in ("node", "target", "weights", "baseline")}
    sim.reset([1])
    sim.reset([1])
    assert not drag.active.numpy()[1]
    assert preserved == {k: getattr(drag, k).numpy().tobytes() for k in preserved}
    with pytest.raises(NotImplementedError):
        sim.physics.command_views("d0")


@pytest.mark.parametrize("device,captured", MODES)
def test_tip_recapture_is_transactional_and_selective(device, captured):
    a, b, m = authored(device)
    sim, s = runtime(a, m, captured)
    r, c = s["r0"], s["c0"]
    c.actions.fill_(0.31)
    c.targets.fill_(wp.vec3(0.1, 0.2, 0.3))
    r.omega.fill_(wp.vec3(0.4, 0.5, 0.6))
    r.bend_offset.assign(np.arange(3 * r.bend_count * 3, dtype=np.float32).reshape(3, r.bend_count, 3))
    before = snapshot(sim.scene)
    for invalid in (-1, r.bend_count, 1.5):
        c.tip_edges = invalid
        with pytest.raises((ValueError, TypeError)):
            sim.prepare()
        assert snapshot(sim.scene) == before
    expected = r.bend_offset.numpy().copy()
    for new, old in ((1, 2), (1, 1), (3, 1), (0, 3), (0, 0), (2, 0)):
        c.tip_edges = new
        if new < old:
            expected[:, r.bend_count - old : r.bend_count - new] = 0
        sim.prepare()
        np.testing.assert_array_equal(r.bend_offset.numpy(), expected)
        assert c.actions.numpy().tobytes() == np.full((3, 3), 0.31, np.float32).tobytes()
        assert np.all(c.targets.numpy() == np.float32([0.1, 0.2, 0.3]))
        sim.prepare()
        np.testing.assert_array_equal(r.bend_offset.numpy(), expected)


@pytest.mark.parametrize("device,captured", MODES)
def test_two_models_sharing_templates_and_replaced_aliases(device, captured):
    a, b, m = authored(device, regional=True, inertia="physical")
    sim, s = runtime(a, m, captured)
    s["c0"].actions.fill_(0.23)
    sim.step()
    before = snapshot(sim.scene, (Owner.MODEL, Owner.STATE, Owner.CONTROL))
    m2 = b.finalize(device=device)
    m2.particle_grid = None
    assert snapshot(sim.scene, (Owner.MODEL, Owner.STATE, Owner.CONTROL)) == before
    other, os = runtime(a, m2, captured)
    assert os["r0"].frame.ptr != s["r0"].frame.ptr and os["r0"].inertia_local.ptr != s["r0"].inertia_local.ptr
    np.testing.assert_array_equal(os["r0"].inertia_local.numpy(), s["r0"].inertia_local.numpy())
    with pytest.raises(ValueError, match="foreign"):
        a.bind(m, m2.state(), m.control())
    shared, ss = runtime(a, m, captured, control=sim.scene.control)
    assert ss["c0"].actions.ptr == s["c0"].actions.ptr
    assert ss["r0"].bend_offset.ptr == s["r0"].bend_offset.ptr
    assert ss["r0"].frame.ptr != s["r0"].frame.ptr
    ordinary = m.control()
    template = m.control(clone_variables=False)
    templ, ts = runtime(a, m, captured, control=template)
    ts["c0"].actions.fill_(0.76)
    clone = m.control()
    f = sim.scene.endoluminal_domains["c0"].fields["actions"]
    assert getattr(template.i4h, f.name).ptr == getattr(m.i4h, f.name).ptr
    np.testing.assert_array_equal(getattr(clone.i4h, f.name).numpy(), np.float32(0.76))
    assert not getattr(ordinary.i4h, f.name).numpy().any()
    # Nested holders and their arrays are validated before preparation can adopt them.
    original = s["r0"].body.rod_controls.bend_offset
    s["r0"].body.rod_controls.bend_offset = wp.clone(original)
    with pytest.raises(RuntimeError, match="replaced"):
        sim.prepare()
    s["r0"].body.rod_controls.bend_offset = original
    props = s["r0"].body.region_properties
    original = props.tables["bend_multiplier"]
    props.tables["bend_multiplier"] = wp.clone(original)
    with pytest.raises(RuntimeError, match="replaced"):
        sim.prepare()
    props.tables["bend_multiplier"] = original
    original = sim.scene.control.i4h
    sim.scene.control.i4h = ordinary.i4h
    with pytest.raises(RuntimeError, match="replaced"):
        sim.prepare()
    sim.scene.control.i4h = original


def test_missing_duplicate_late_and_incompatible_declarations():
    scene = Scene("cpu", 2, (0, 0, 0))
    chain(scene, "r")
    scene.allocate()
    with pytest.raises(RuntimeError, match="missing"):
        RodDirect(scene, body="r")
    with pytest.raises(RuntimeError, match="before allocation"):
        declare_rod_direct(scene, "r")
    scene = Scene("cpu", 2, (0, 0, 0))
    chain(scene, "r")
    declare_rod_direct(scene, "r")
    with pytest.raises(ValueError, match="duplicate"):
        declare_rod_direct(scene, "r")
    scene = Scene("cpu", 2, (0, 0, 0))
    chain(scene, "r")
    declare_rod_direct_material(scene, "r")
    with pytest.raises(ValueError, match="matching solver"):
        scene.allocate()


@pytest.mark.parametrize("device,captured", MODES)
def test_direct_scratch_overwrite_and_zero_accumulators(device, captured):
    a, b, m = authored(device, regional=True, inertia="physical")
    sim, s = runtime(a, m, captured)
    other, t = runtime(a, m, captured)
    for systems in (s, t):
        systems["c0"].actions.assign(np.tile(np.float32([0.3, -0.2, 0.1]), (3, 1)))
    rod = t["r0"]
    for key in ("pos_corr", "rot_corr"):
        assert not getattr(rod, key).numpy().any()
    for key in (
        "compliance",
        "lambda_sum",
        "rhs",
        "delta_lambda",
        "jacobian_rot",
        "diag_blocks",
        "offdiag_blocks",
        "c_blocks",
    ):
        getattr(rod, key).fill_(123)
    rod.frame_pred.fill_(wp.quat(1, 2, 3, 4))
    rod.edge_w.fill_(wp.vec3(123, 123, 123))
    rod.inv_inertia.fill_(wp.mat33(123))
    rod.body.rod_material.preferred_rest.fill_(wp.vec3(123, 123, 123))
    for key in ("young_multiplier", "torsion_multiplier", "bend_multiplier", "twist_multiplier"):
        getattr(rod.body.rod_material, key).fill_(123)
    sim.step()
    other.step()
    assert snapshot(sim.scene) == snapshot(other.scene)
    for key in ("pos_corr", "rot_corr"):
        assert not getattr(rod, key).numpy().any()
    # Model live defaults survive resets; root is deliberately not reset by catheter.
    t["c0"].track_length.fill_(0.03)
    root = t["c0"].root.numpy().copy()
    other.reset([1])
    np.testing.assert_array_equal(t["c0"].root.numpy(), root)
    np.testing.assert_array_equal(t["c0"].track_length.numpy(), np.float32(0.03))


def test_finalizer_uses_exact_custom_inverse_mass_of_each_model():
    a, b, m = authored("cpu", inertia="physical")
    # Native mass inversion can differ from exact component inverse-mass authoring.
    # Supply a different native mass for the second Model; inertia still uses its
    # declared i4h inverse masses, just as Scene.field did before this migration.
    b.particle_mass = [x * 1.137 for x in b.particle_mass]
    second = b.finalize(device="cpu")
    second.particle_grid = None
    sim, s = runtime(a, second)
    np.testing.assert_array_equal(
        s["r0"].inertia_local.numpy(), cylinder_inv_inertia(sim.scene, s["r0"].body, s["r0"].rest_len.numpy()[0]).reshape(-1, 3)
    )
    assert not np.array_equal(second.particle_inv_mass.numpy(), second.i4h.particle_inv_mass.numpy())


@pytest.mark.parametrize("device,captured", MODES)
@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize(
    "backend,key",
    [
        ("direct", "variant"),
        ("pbd", "variant"),
        ("pbd", "effective_stretch"),
        ("pbd", "effective_bend"),
        ("pbd", "effective_twist"),
    ],
)
def test_solver_alias_replacement_requires_fresh_binding(device, captured, prepared, backend, key):
    a, b, m = authored(device, backend)
    scene, systems = a.bind(m, m.state(), m.control())
    sim = Stepper.create(scene, systems, cuda_graph=captured)
    if prepared:
        sim.prepare()
    rod = next(s for s in systems if s.id == "r0")
    original = getattr(rod, key)
    setattr(rod, key, wp.clone(original))
    before = snapshot(scene)
    with pytest.raises(RuntimeError, match="replaced"):
        sim.prepare()
    assert snapshot(scene) == before
    setattr(rod, key, original)
    sim.prepare()
    assert snapshot(scene) == before


def test_catheter_recording_replay_and_host_named_view_lifetime(tmp_path):
    pytest.importorskip("isaaclab.envs")
    pytest.importorskip("h5py")
    from types import SimpleNamespace

    import torch
    from i4h_isaaclab.recording import Recorder, load_demo, replay_episode
    from isaaclab.sim import SimulationContext

    env = catheter_host_env(tmp_path)
    recorder = None
    component = I4hComponent(I4hComponentCfg(assets={}, systems=[]))
    try:
        env.reset()
        u = env.unwrapped
        component.sim = u.simulator
        named = component.command_views("control")
        assert component.command_views("control") is named
        teleop = SimpleNamespace(
            u=u, physical=torch.zeros((2, 3), device=u.device), devices=SimpleNamespace(states={}), reset=env.reset
        )
        recorder = Recorder(tmp_path / "rates", teleop)
        recorder.start(reset=False)
        rows = []
        for k in range(5):
            named["insertion"].fill_(0.2 + 0.03 * k)
            named["rotation"].fill_(-0.1)
            named["bend"].fill_(0.1)
            teleop.physical.copy_(named["actions"])
            rows.append(teleop.physical[0].clone())
            obs, *_ = env.step(teleop.physical)
            recorder.record(obs)
        assert recorder.end(True) == "demo_0"
        recorder.close()
        episode, args = load_demo(recorder.path, "demo_0")
        assert args["format_version"] == 1
        assert args["action_layout"] == [["control", 3, ["catheter"]]]
        torch.testing.assert_close(episode.data["actions"], torch.stack(rows).cpu(), rtol=0, atol=0)
        diffs = replay_episode(env, episode, args)
        np.testing.assert_array_equal(diffs, 0)
        previous = u.simulator
        u.sim.reset(soft=False)
        with pytest.raises(RuntimeError, match="invalidated"):
            component.command_views("control")
        component.sim = u.simulator
        rebuilt = component.command_views("control")
        assert rebuilt is not named and u.simulator is not previous
    finally:
        if recorder is not None:
            recorder.close()
        env.close()
        SimulationContext.clear_instance()
    with pytest.raises(RuntimeError, match="invalidated"):
        component.command_views("control")


def catheter_host_env(tmp_path):
    pytest.importorskip("isaaclab.envs")
    import yaml
    from i4h_isaaclab.env import I4hEnv, I4hEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationContext

    case = dict(
        sim=dict(device="cuda:0", envs=2, dt=1 / 60, substeps=2, iterations=2, gravity=[0, 0, 0]),
        assets=dict(
            catheter=dict(
                type="rod",
                start=[0, 0, 0.2],
                direction=[1, 0, 0],
                count=7,
                segment_length=0.02,
                radius=0.003,
                mass=0.01,
            )
        ),
        systems=[
            dict(type="particles"),
            dict(type="rod_direct", id="rod", body="catheter"),
            dict(type="catheter_control", id="control", rod="rod", track_length=0.04, tip_edges=2),
        ],
    )
    path = tmp_path / "catheter.yaml"
    path.write_text(yaml.safe_dump(case))
    if SimulationContext.instance() is not None:
        SimulationContext.clear_instance()
    cfg = I4hEnvCfg(
        case=str(path),
        backend="newton",
        action_mode="physical",
        seed=0,
        gravity_from_case=True,
        stagger_episodes=False,
        episode_length_s=10.0,
    )
    cfg.scene = InteractiveSceneCfg(num_envs=2, env_spacing=1.0)
    env = I4hEnv(cfg)
    return env


def test_catheter_host_named_view_lifetime_without_optional_recording(tmp_path):
    pytest.importorskip("isaaclab.envs")
    import torch
    from isaaclab.sim import SimulationContext

    env = catheter_host_env(tmp_path)
    component = I4hComponent(I4hComponentCfg(assets={}, systems=[]))
    try:
        env.reset()
        u = env.unwrapped
        component.sim = u.simulator
        old = component.command_views("control")
        assert component.command_views("control") is old
        old["insertion"].fill_(0.2)
        old["rotation"].fill_(0.1)
        old["bend"].fill_(0.3)
        obs, *_ = env.step(old["actions"].clone())
        assert torch.isfinite(obs["policy"]).all()
        previous = u.simulator
        u.sim.reset(soft=False)
        with pytest.raises(RuntimeError, match="invalidated"):
            component.command_views("control")
        component.sim = u.simulator
        new = component.command_views("control")
        assert new is not old and new["actions"].data_ptr() != old["actions"].data_ptr()
        assert u.simulator is not previous
        env.step(new["actions"].clone())
    finally:
        env.close()
        SimulationContext.clear_instance()
    with pytest.raises(RuntimeError, match="invalidated"):
        component.command_views("control")
