# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""C3.3 ownership, delayed coupling and continuation with fully exposed rigid history."""
from dataclasses import dataclass
from types import SimpleNamespace

import newton
import numpy as np
import pytest
import warp as wp
from i4h_physics import Scene, Stepper, author_case, declare_mounts
from i4h_physics.coupling import CoupledSolver, Mounts
from i4h_physics.mount_storage import find_body
from i4h_physics.storage import Owner
from i4h_physics.systems.base import System
from i4h_surgical.instrument_storage import declare_instruments
from i4h_surgical.loaders.instrument_usd import add_instrument
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
from ownership_helpers import MODES, ExposedRigid, copy_container, snapshot

DT = .01
XFORMS = [((.5 * e, -.1 * e, 0.), (0., 0., float(np.sin(e * .2)), float(np.cos(e * .2)))) for e in range(3)]
MOUNTS = {name: {"body": "holder", "offset": ((.01 * (i + 1), .02, 0.), (0., 0., .3, .7))}
          for i, name in enumerate(("a", "b"))}


def tool(scene, name, **kwargs):
    return add_instrument(scene, name, kind="sphere", radius=.02,
                          initial={"position": [.1, .2, -.3], "rotation": [0., 0., .2, .9]})


@wp.kernel
def source_values(root: wp.array2d(dtype=wp.vec3), impulse: wp.array2d(dtype=wp.vec3), moment: wp.array2d(dtype=wp.vec3)):
    e, k = wp.tid()
    impulse[e, k] = wp.vec3(.00001, -.00002, .00003) + root[e, k] * .00001
    moment[e, k] = wp.vec3(.000004, .000005, -.000006)


@dataclass(eq=False)
class Reaction(System):
    instruments: object = None
    refs = ("instruments",)

    def __post_init__(self):
        self.impulse = wp.zeros((self.scene.envs, self.instruments.count), dtype=wp.vec3, device=self.scene.device)
        self.moment = wp.zeros_like(self.impulse)

    def step_start(self, scene, dt):
        wp.launch(source_values, self.impulse.shape, [self.instruments.root], [self.impulse, self.moment], device=scene.device)


def authored(device="cpu", *, empty=False, two_way=True, mounts=None, catheter=False, duplicate=False, bad_label=False):
    specs = {} if empty else (MOUNTS if mounts is None else mounts)
    systems = [{"type": "instrument_control", "id": "tools"}]
    if not empty:
        systems += [{"type": "reaction", "id": f"source{i}", "instruments": "tools"} for i in range(2)]
    cfg = {"sim": {"device": device, "envs": 3, "gravity": [0, 0, 0]},
           "assets": {} if empty else {"a": {"type": "test_tool"}, "b": {"type": "test_tool"}}, "systems": systems}
    loaders = {"test_tool": tool}
    constructors = {"instrument_control": InstrumentControl, "reaction": Reaction}
    if catheter:
        from i4h_endoluminal.systems.rods.catheter_control import CatheterControl
        from i4h_physics.config import LOADERS, SYSTEMS
        loaders.update(LOADERS)
        constructors.update(SYSTEMS, catheter_control=CatheterControl)
        cfg["systems"].append(dict(type="particles"))
        specs = dict(specs)
        for i in range(2):
            name = f"cat{i}"
            cfg["assets"][name] = dict(type="rod", start=[.1, .2, .3], direction=[1, 0, 0], count=6,
                                      segment_length=.02, radius=.002, mass=.01)
            cfg["systems"] += [dict(type="rod_direct", id=f"rod{i}", body=name),
                               dict(type="catheter_control", id=f"driver{i}", rod=f"rod{i}", track_length=.02)]
            specs[name] = dict(body="holder")
    a = author_case(cfg, loaders, constructors, mounts=specs, coupling_mode="two_way" if two_way else "one_way")
    b = newton.ModelBuilder()
    b.add_body(label="foreign/holder")  # non-world prefix must never be selected
    for e, xform in enumerate(XFORMS):
        b.begin_world(gravity=(0, 0, 0))
        b.add_body(label="unrelated")
        a.add_world(b, e, xform=xform)
        # Bodies intentionally appear AFTER i4h's schema/world hook.
        b.add_body(label="robot/arm/holder" if not bad_label else "wrong", mass=1.,
                   xform=wp.transform(wp.vec3(*xform[0]), wp.quat(*xform[1])))
        if duplicate:
            b.add_body(label="other/holder")
        b.end_world()
    a.complete(b)
    m = b.finalize(device=device)
    m.particle_grid = None
    return a, b, m


def runtime(a, m, captured=False, state=None, control=None):
    state, control = (m.state() if state is None else state), (m.control() if control is None else control)
    scene, systems = a.bind(m, state, control)
    sim = Stepper.create(scene, systems, dt=DT, substeps=2, iterations=1, cuda_graph=False)
    sim.prepare()
    mounts = Mounts(m, scene, systems, a.mounts, a.coupling_mode == "two_way")
    coupled = CoupledSolver(m, ExposedRigid(m), sim.solver, mounts, rigid_substeps=2)
    def step():
        state.body_f.zero_()
        coupled.step(state, state, control, None, DT)
    if captured:
        wp.synchronize_device(m.device)
        with wp.ScopedCapture(device=m.device) as capture:
            step()
        def advance():
            sim.solver.check_binding()
            wp.capture_launch(capture.graph)
    else:
        advance = step
    return SimpleNamespace(sim=sim, scene=scene, systems=systems, mounts=mounts, coupled=coupled,
                           state=state, control=control, step=advance)


@pytest.mark.parametrize("device,captured", MODES)
@pytest.mark.parametrize("two_way", [False, True])
def test_defaults_declarations_aliases_and_unchanged_binding(device, captured, two_way, monkeypatch):
    a, b, m = authored(device, two_way=two_way, catheter=True)
    state, control = m.state(), m.control()
    control.i4h.tools_mount_pose.fill_(wp.transform((2., 3., 4.), wp.quat_identity()))
    state.i4h.tools_mount_force.fill_(wp.vec3(3., 4., 5.))
    state.i4h.tools_mount_torque.fill_(wp.vec3(6., 7., 8.))
    before = snapshot(m, state, control)
    def forbidden(*args):
        raise AssertionError("binding must not call control.mount")
    monkeypatch.setattr(InstrumentControl, "mount", forbidden)
    r = runtime(a, m, captured, state, control)
    assert snapshot(m, state, control) == before
    assert len(r.mounts.groups) == 3
    np.testing.assert_array_equal(r.systems[0].mounted.numpy(), [1, 1])
    np.testing.assert_array_equal(r.systems[0].root.numpy(), np.tile([[[.1, .2, -.3]]], (3, 2, 1)).astype(np.float32))
    for g, (sid, entries, fields) in zip(r.mounts.groups, r.scene.mount_domains["groups"]):
        for key, f in fields.items():
            value = getattr(g, key)
            owner = m if f.owner is Owner.MODEL else state
            assert value.ptr == getattr(owner.i4h, f.name).ptr
            assert f.references == (newton.Model.AttributeFrequency.BODY if key == "body" else None)
        d = fields["body"].domain
        np.testing.assert_array_equal(getattr(m.i4h, d.frequency[4:]+"_owner").numpy().reshape(d.shape), g.body.numpy())
        np.testing.assert_array_equal(g.body.numpy(), np.tile(np.array([2, 4, 6])[:, None], (1, g.count)))
        d = fields["target"].domain
        assert d.owner_reference == r.scene.mountable_domains[sid].target_domain.frequency
        np.testing.assert_array_equal(getattr(m.i4h, d.frequency[4:]+"_world").numpy(), -1)
        np.testing.assert_array_equal(getattr(m.i4h, d.frequency[4:]+"_owner").numpy(), g.target.numpy())
    r.sim.prepare()
    r.coupled.reset(state)
    assert snapshot(m, state, control) == before
    Mounts(m, r.scene, r.systems, a.mounts, two_way)
    assert snapshot(m, state, control) == before
    r.step()
    for group in r.mounts.groups[1:]:
        initial = np.tile(np.float32([[[.1, .2, .3]]]), (3, 1, 1))
        delta = group.control.mount_pose.numpy()[..., :3] - initial
        distance = np.linalg.norm(delta, axis=-1, keepdims=True)
        limit = r.sim.substeps * group.control.max_delta.numpy().reshape(3, 1, 1)
        expected = initial + delta * np.minimum(1., limit / np.maximum(distance, 1.e-12))
        np.testing.assert_allclose(group.control.root.numpy(), expected, atol=1e-6)
        root, pose = group.control.root.numpy().copy(), group.control.mount_pose.numpy().copy()
        r.coupled.restore_episode(state, control, wp.array([True, False, False, False], dtype=wp.bool, device=device))
        np.testing.assert_array_equal(group.control.root.numpy(), root)
        np.testing.assert_array_equal(group.control.mount_pose.numpy(), pose)


@pytest.mark.parametrize("device,captured", MODES)
def test_delayed_multiple_source_shared_body_and_resume(device, captured):
    a, b, m = authored(device)
    r = runtime(a, m, captured)
    r.step()
    g = r.mounts.groups[0]
    np.testing.assert_array_equal(r.state.body_f.numpy(), 0.)  # previous buffer was zero
    scale = 1. / ((DT / 2) ** 2 * 2)
    np.testing.assert_allclose(g.force.numpy(), -sum(s.impulse.numpy() for s in g.sources) * scale, rtol=1e-7)
    assert g.force.numpy().any()
    for _ in range(2):
        r.step()
    checkpoint, commands = m.state(), m.control()
    copy_container(r.state, checkpoint)
    copy_container(r.control, commands)
    before = snapshot(r.state, r.control)
    resumed = runtime(a, m, captured, checkpoint, commands)
    rebound = runtime(a, m, captured, r.state, r.control)
    assert snapshot(checkpoint, commands) == before == snapshot(r.state, r.control)
    r.sim.prepare()
    assert snapshot(r.state, r.control) == before
    for _ in range(4):
        r.step()
        resumed.step()
        assert snapshot(r.state, r.control) == snapshot(resumed.state, resumed.control)
    # Apply to a cleared force array independently: both targets sum atomically into one body/world.
    pending = g.force.numpy().copy()
    r.state.body_f.zero_()
    r.mounts.apply(r.state, m)
    applied = r.state.body_f.numpy().copy()
    for e, (_, quat) in enumerate(XFORMS):
        expected = np.array(wp.quat_to_matrix(wp.quat(*quat))).reshape(3, 3) @ pending[e].sum(axis=0)
        np.testing.assert_allclose(applied[g.body.numpy()[e, 0], :3], expected, atol=1e-6)
    r.mounts.apply(r.state, m)
    np.testing.assert_array_equal(r.state.body_f.numpy(), applied)
    assert not g.force.numpy().any()
    assert rebound.mounts.groups[0].force.ptr == g.force.ptr


@pytest.mark.parametrize("device,captured", MODES)
def test_masks_persistent_flags_commands_clones_and_second_model(device, captured):
    a, b, m = authored(device)
    r = runtime(a, m, captured)
    g, tool = r.mounts.groups[0], r.systems[0]
    tool.mounted.assign(np.array([0, 1], np.int32))
    tool.mount(0)
    tool.mount_pose.fill_(wp.transform((2., 3., 4.), wp.quat_identity()))
    g.force.fill_(wp.vec3(1., 2., 3.))
    g.torque.fill_(wp.vec3(4., 5., 6.))
    before = snapshot(m, r.state, r.control)
    empty = wp.zeros(4, dtype=wp.bool, device=device)
    r.coupled.restore_episode(r.state, r.control, empty)
    assert snapshot(m, r.state, r.control) == before
    mask = wp.array([False, True, False, False], dtype=wp.bool, device=device)
    for _ in range(2):
        r.coupled.restore_episode(r.state, r.control, mask)
    np.testing.assert_array_equal(g.force.numpy()[1], 0)
    np.testing.assert_array_equal(g.force.numpy()[[0, 2]], np.tile([[[1., 2., 3.]]], (2, 2, 1)))
    np.testing.assert_array_equal(tool.root.numpy()[1], [[2., 3., 4.]] * 2)
    np.testing.assert_array_equal(tool.mounted.numpy(), [1, 1])
    np.testing.assert_array_equal(tool.mount_pose.numpy()[..., :3], np.tile([[[2., 3., 4.]]], (3, 2, 1)))
    shared = runtime(a, m, captured, control=r.control)
    assert shared.systems[0].mount_pose.ptr == tool.mount_pose.ptr != m.i4h.tools_mount_pose.ptr
    assert shared.mounts.groups[0].force.ptr != g.force.ptr
    template_before = snapshot(m)
    later_model = b.finalize(device=device)
    assert snapshot(m) == template_before
    other = runtime(a, later_model, captured)
    assert other.systems[0].mounted.ptr != tool.mounted.ptr
    np.testing.assert_array_equal(other.systems[0].mounted.numpy(), [1, 1])
    np.testing.assert_array_equal(other.mounts.groups[0].force.numpy(), 0)
    alias = runtime(a, m, captured, control=m.control(clone_variables=False))
    alias.systems[0].mount_pose.fill_(wp.transform((7., 8., 9.), wp.quat_identity()))
    np.testing.assert_array_equal(m.control().i4h.tools_mount_pose.numpy()[:, :3], [[7., 8., 9.]] * 6)
    np.testing.assert_array_equal(tool.mount_pose.numpy()[..., :3], np.tile([[[2., 3., 4.]]], (3, 2, 1)))


@pytest.mark.parametrize("replacement", ["model", "state", "command", "native", "group", "groups", "source", "sources", "root", "control", "mode", "names", "layout"])
def test_replacement_rejected_before_any_coupling_write(replacement):
    a, _, m = authored()
    r = runtime(a, m)
    g = r.mounts.groups[0]
    g.force.fill_(wp.vec3(1, 2, 3))
    if replacement in ("model", "state", "command"):
        owner, key = {"model": (m, "tools_mount_body"), "state": (r.state, "tools_mount_force"),
                      "command": (r.control, "tools_mount_pose")}[replacement]
        setattr(owner.i4h, key, wp.clone(getattr(owner.i4h, key)))
    elif replacement == "native":
        r.state.body_q = wp.clone(r.state.body_q)
    elif replacement == "group":
        g.force = wp.clone(g.force)
    elif replacement == "groups":
        r.mounts.groups[0] = SimpleNamespace(**vars(g))
    elif replacement == "source":
        g.sources[0].impulse = wp.clone(g.sources[0].impulse)
    elif replacement == "sources":
        g.sources.pop()
    elif replacement == "root":
        g.control.root = wp.clone(g.control.root)
    elif replacement == "control":
        g.control = SimpleNamespace(**vars(g.control))
    elif replacement == "mode":
        r.mounts.two_way = False
    elif replacement == "names":
        g.control.names.reverse()
    else:
        g.body.shape = (g.envs * g.count,)
    before = snapshot(m, r.state, r.control)
    for operation in (lambda: r.coupled.step(r.state, r.state, r.control, None, DT), r.sim.prepare,
                      lambda: r.mounts.drive(r.state), lambda: r.mounts.fill(2, DT)):
        with pytest.raises((RuntimeError, ValueError)):
            operation()
        assert snapshot(m, r.state, r.control) == before


def test_foreign_containers_and_incompatible_declarations_rejected_before_writes():
    a, b, m = authored()
    r = runtime(a, m)
    g = r.mounts.groups[0]
    g.force.fill_(wp.vec3(1, 2, 3))
    other_model = b.finalize(device="cpu")
    before = snapshot(m, r.state, r.control)
    for state, control in ((m.state(), r.control), (r.state, m.control())):
        with pytest.raises(ValueError):
            r.coupled.step(state, state, control, None, DT)
    with pytest.raises(ValueError):
        r.mounts.apply(r.state, other_model)
    with pytest.raises(ValueError, match="foreign"):
        a.bind(other_model, r.state, r.control)
    with pytest.raises(ValueError, match="incompatible"):
        Mounts(m, r.scene, r.systems, {}, True)
    with pytest.raises(RuntimeError, match="before"):
        declare_mounts(r.scene, MOUNTS)
    assert snapshot(m, r.state, r.control) == before


@pytest.mark.parametrize("device,captured", MODES)
def test_empty_padding_and_manual_mount_without_mapping(device, captured):
    a, _, m = authored(device, empty=True)
    r = runtime(a, m, captured)
    assert r.mounts.groups == []
    assert r.systems[0].mounted.shape == (1,)
    assert r.systems[0].mount_pose.shape == (3, 0)
    assert m.i4h.tools_mount_pose.shape == (1,)
    np.testing.assert_array_equal(m.i4h.tools_mountable_rows_owner.numpy(), [-1])
    np.testing.assert_array_equal(m.i4h.tools_runtime_rows_owner.numpy(), [-1])
    r.step()
    a, _, m = authored(device, mounts={})
    r = runtime(a, m, captured)
    assert not r.mounts.groups
    np.testing.assert_array_equal(r.systems[0].mounted.numpy(), 0)
    r.systems[0].mount(1)
    np.testing.assert_array_equal(m.i4h.tools_mounted.numpy(), [0, 1])
    assert not r.mounts.groups
    r.systems[0].mount_pose.fill_(wp.transform((2., 3., 4.), wp.quat_identity()))
    r.step()  # the already captured graph reads the explicit Model edit
    np.testing.assert_array_equal(r.systems[0].root.numpy()[:, 1], [[2., 3., 4.]] * 3)


def test_body_label_and_target_errors():
    with pytest.raises(ValueError, match="0 bodies match"):
        authored(bad_label=True)
    with pytest.raises(ValueError, match="2 bodies match"):
        authored(duplicate=True)
    with pytest.raises(ValueError, match="no mount target"):
        authored(mounts={"missing": {"body": "holder"}})
    scene = Scene(device="cpu", envs=1)
    tool(scene, "a")
    declare_instruments(scene, system_id="first")
    declare_instruments(scene, system_id="second")
    declare_mounts(scene, {"a": {"body": "holder"}})
    with pytest.raises(ValueError, match="duplicate"):
        declare_mounts(scene, {})
    with pytest.raises(ValueError, match="2 controls"):
        scene.allocate()
    host = SimpleNamespace(body_world=[0, 0, 1], body_label=["a/holder", "b/holder", "holder"])
    assert find_body(host, "a/holder", 0) == 0
    assert find_body(host, "holder", 1) == 2


def test_two_way_pbd_and_late_configuration_rejected():
    from i4h_endoluminal.systems.rods.catheter_control import CatheterControl
    from i4h_physics.config import LOADERS, SYSTEMS
    cfg = {"sim": {"device": "cpu", "envs": 1},
           "assets": {"cat": dict(type="rod", start=[0, 0, 0], direction=[1, 0, 0], count=6, segment_length=.02, radius=.002)},
           "systems": [dict(type="rod_pbd", id="rod", body="cat"), dict(type="catheter_control", rod="rod", track_length=.01)]}
    a = author_case(cfg, LOADERS, SYSTEMS | {"catheter_control": CatheterControl}, mounts={"cat": {"body": "holder"}}, coupling_mode="two_way")
    with pytest.raises(ValueError, match="rod_direct only"):
        a.declare(newton.ModelBuilder())
    a, b, m = authored()
    a.mounts["a"]["body"] = "changed"
    with pytest.raises(RuntimeError, match="configuration changed"):
        a.bind(m, m.state(), m.control())
    with pytest.raises(RuntimeError, match="configuration changed"):
        a.complete(b)


def test_direct_scene_requires_matching_predeclared_mounts():
    scene = Scene(device="cpu", envs=1)
    tool(scene, "a")
    declare_instruments(scene)
    scene.allocate()
    control = InstrumentControl(scene)
    before = snapshot(scene.model, scene.state, scene.control)
    with pytest.raises(RuntimeError, match="mount storage missing"):
        Mounts(scene.model, scene, [control], {"a": {"body": "holder"}}, False)
    assert snapshot(scene.model, scene.state, scene.control) == before



def test_host_labels_cannot_change_after_completion():
    a, b, m = authored()
    b.body_label[2] = "changed"
    with pytest.raises(ValueError, match="0 bodies match"):
        b.finalize(device="cpu")


@pytest.mark.parametrize('device,captured', MODES)
@pytest.mark.parametrize('shared', [False, True])
def test_control_clear_rejects_before_writes_and_keeps_mount_poses(device, captured, shared):
    """Blanket Newton clear is unsafe for transform commands, even with a Model."""
    a, builder, model = authored(device, catheter=True)
    control = model.control(clone_variables=not shared)
    # The policy applies before first binding, including template-alias controls.
    control.i4h.tools_actions.fill_(.375)
    before = snapshot(model, control)
    for args in ((), (model,)):
        with pytest.raises(RuntimeError, match=r'Control.clear\(\).*mount-pose quaternions'):
            control.clear(*args)
        assert snapshot(model, control) == before
    run = runtime(a, model, captured=captured, control=control)
    run.step()
    before = snapshot(model, run.state, control)
    with pytest.raises(RuntimeError, match='packed/named action views'):
        control.clear(model=model)
    assert snapshot(model, run.state, control) == before
    run.scene.storage.check()
    # Clearing supported command views leaves mount transforms untouched.
    poses = [s.mount_pose for s in run.systems if hasattr(s, 'mount_pose')]
    expected = [p.numpy().tobytes() for p in poses]
    for system in run.systems:
        if hasattr(system, 'actions'):
            system.actions.zero_()
    assert [p.numpy().tobytes() for p in poses] == expected
    run.step()
    # A second finalization gets its own factory, and Newton itself is untouched.
    second = builder.finalize(device=device)
    for current in (model.control(), second.control(), second.control(clone_variables=False)):
        with pytest.raises(RuntimeError, match='mount-pose quaternions'):
            current.clear()
    newton.ModelBuilder().finalize(device=device).control().clear()
