# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""C3.1 history-complete control fixtures, ownership and command interop."""
from dataclasses import dataclass

import newton
import numpy as np
import pytest
import warp as wp
from i4h_isaaclab.component import I4hComponent, I4hComponentCfg
from i4h_physics import Scene, Stepper, author_case
from i4h_physics.storage import Domain, Field, Owner
from i4h_physics.systems.base import System
from i4h_surgical.instrument_storage import STATE_TYPES, declare_instruments
from i4h_surgical.loaders.instrument_usd import add_instrument
from i4h_surgical.scene import SphereSpec
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
from ownership_helpers import MODES


def tool(scene, name, **kwargs):
    record = add_instrument(scene, name, kind="sphere", radius=.02,
                            initial={"position": [.1, .2, -.3], "rotation": [0, 0, 0, 2]})
    record.spheres = [SphereSpec(2, sign, np.array([0, .02, .05], np.float32),
                               np.eye(4, dtype=np.float32), .02, 0, 0, 1) for sign in (1., -1.)]
    record.prims = [(0, 0, "jaw_left"), (1, 1, "jaw_right")]
    return record


def authored(device, empty=False):
    cfg = {"sim": {"device": device, "envs": 3, "gravity": [0, 0, 0]},
           "assets": {} if empty else {"a": {"type": "test_tool"}, "b": {"type": "test_tool"}},
           "systems": [{"type": "instrument_control", "id": sid} for sid in ("first", "second")]}
    a = author_case(cfg, loaders={"test_tool": tool}, systems={"instrument_control": InstrumentControl})
    b = newton.ModelBuilder()
    b.add_particle((9, 9, 9), (0, 0, 0), 1.)
    b.add_body()
    for world in range(3):
        b.begin_world(gravity=(0, 0, 0))
        b.add_particle((9, 9, 9), (0, 0, 0), 1.)
        b.add_body()
        a.add_world(b, world)
        b.end_world()
    a.complete(b)
    m = b.finalize(device=device)
    m.particle_grid = None
    return a, b, m


def runtime(a, m, captured=False, state=None, control=None):
    state = m.state() if state is None else state
    control = m.control() if control is None else control
    scene, systems = a.bind(m, state, control)
    sim = Stepper.create(scene, systems, cuda_graph=captured, dt=.01, substeps=1, iterations=1)
    sim.prepare()
    return sim, systems[0]


def data(system):
    return {k: getattr(system, k).numpy().tobytes() for k in (*STATE_TYPES, "actions")}


@pytest.mark.parametrize("device,captured", MODES)
@pytest.mark.parametrize("empty", [False, True])
def test_templates_domains_and_named_views(device, captured, empty):
    a, b, m = authored(device, empty)
    sim, s = runtime(a, m, captured)
    N, S = s.count, s.sphere_count
    assert sum(getattr(s, k).size * wp.types.type_size_in_bytes(t) for k, t in STATE_TYPES.items()) == 92*3*N + 28*3*S
    assert s.actions.size*4 == 40*3*N
    for sid, record in sim.scene.instrument_domains.items():
        for name, d in record["domains"].items():
            worlds = getattr(m.i4h, d.frequency[4:]+"_world").numpy()
            owners = getattr(m.i4h, d.frequency[4:]+"_owner").numpy()
            if not d.count:
                assert worlds.tolist() == owners.tolist() == [-1]
            elif "runtime" in name:
                np.testing.assert_array_equal(worlds.reshape(d.shape), np.broadcast_to(np.arange(3)[:, None], d.shape))
                np.testing.assert_array_equal(owners.reshape(d.shape), np.tile(np.arange(d.shape[1]), (3, 1)))
            else:
                np.testing.assert_array_equal(worlds, -1)
                if name == "instrument":
                    np.testing.assert_array_equal(owners, -1)
                else:
                    assert d.owner_reference == record["domains"]["instrument"].frequency
            assert d.frequency.startswith(f"i4h:{sid}_")
        for key, f in record["fields"].items():
            assert f.references is None
            assert getattr(s if sid == 'first' else sim.systems[1], key) is sim.scene.attr(f.domain, key)
    v = sim.physics.command_views("first")
    assert sim.physics.command_views("first") is v
    component = I4hComponent(I4hComponentCfg(assets={}, systems=[]))
    component.sim = sim.physics
    t = component.command_views("first")
    assert component.command_views("first") is t
    with pytest.raises(TypeError):
        v["new"] = s.actions
    with pytest.raises(TypeError):
        t["new"] = t["actions"]
    for key, (start, stop) in s.command_view_spec.items():
        assert v[key].shape == (3, N, stop-start)
        assert v[key].strides == s.actions.strides
        assert v[key].ptr == s.actions.ptr+start*4
        assert tuple(t[key].stride()) == tuple(n//4 for n in s.actions.strides)
    if empty:
        assert m.i4h.first_actions.shape == (1,)
        assert not v['position'].size
        sim.step()
        sim.reset([1])
        return
    np.testing.assert_array_equal(s.actions.numpy()[..., 3:7], np.tile([0, 0, 0, 1], (3, N, 1)))
    np.testing.assert_array_equal(s.sphere_q.numpy(), s.sphere_q_prev.numpy())
    assert not np.array_equal(s.sphere_q.numpy()[:, 0], s.sphere_q.numpy()[:, 1])
    for key, (start, stop) in s.command_view_spec.items():
        t[key].fill_(start + .25)
        component.order_inputs()
        np.testing.assert_array_equal(s.actions.numpy()[..., start:stop], start+.25)
        v[key].fill_(start+.5)
        component.sync_views()
        np.testing.assert_array_equal(t[key].cpu().numpy(), start+.5)
    np.testing.assert_array_equal(sim.systems[1].actions.numpy()[..., 7:], 0)
    # Model/template storage is distinct from the selected live container.
    assert m.i4h.first_actions.ptr != s.actions.ptr
    with pytest.raises(KeyError):
        sim.physics.command_views("missing")
    sim.physics.invalidate()
    with pytest.raises(RuntimeError, match="invalidated"):
        component.command_views("first")


@pytest.mark.parametrize("device,captured", MODES)
def test_history_preserved_rebind_resume_edges_and_masked_reset(device, captured):
    a, b, m = authored(device)
    sim, s = runtime(a, m, captured)
    commands = s.actions.numpy()
    commands[..., 0] += .2
    commands[..., 7:] = 1
    s.actions.assign(commands)
    sim.step()
    np.testing.assert_array_equal(s.fired.numpy(), 1)
    np.testing.assert_array_equal(s.grasp_pending.numpy(), 1)
    assert np.all(s.jaw_delta.numpy() > 0)
    assert np.all(s.jaw_angle.numpy() > s.jaw_closed.numpy())
    snapshot = data(s)
    checkpoint = m.state()
    for key in STATE_TYPES:
        wp.copy(getattr(checkpoint.i4h, 'first_'+key), getattr(sim.state.i4h, 'first_'+key))
    checkpoint_control = m.control()
    wp.copy(checkpoint_control.i4h.first_actions, sim.control.i4h.first_actions)
    resumed, r = runtime(a, m, captured, checkpoint, checkpoint_control)
    rebound, _ = runtime(a, m, captured, sim.state, sim.control)
    sim.prepare()
    assert data(s) == data(r) == snapshot
    for _ in range(3):
        sim.step()
        resumed.step()
        assert data(s) == data(r)
    np.testing.assert_array_equal(s.fired.numpy(), 0)
    np.testing.assert_array_equal(s.grasp_pending.numpy(), 0)
    # Ordinary Control sharing updates both runtimes, without touching templates.
    shared, sh = runtime(a, m, captured, m.state(), sim.control)
    assert sh.actions.ptr == s.actions.ptr != m.i4h.first_actions.ptr
    s.actions.fill_(.75)
    np.testing.assert_array_equal(sh.actions.numpy(), .75)
    np.testing.assert_array_equal(m.control().i4h.first_actions.numpy()[..., 7:], 0)
    # Defaults are live Model arrays; reset must read their current values.
    s.initial_position.fill_(wp.vec3(2, 3, 4))
    s.jaw_open.fill_(.8)
    before = data(s)
    sim.reset([])
    assert data(s) == before
    untouched = {k: getattr(s, k).numpy()[[0, 2]].tobytes() for k in (*STATE_TYPES, 'actions')}
    for _ in range(2):
        sim.reset([1])
    for k, value in untouched.items():
        assert getattr(s, k).numpy()[[0, 2]].tobytes() == value
    np.testing.assert_array_equal(s.actions.numpy()[1, :, :3], np.tile([2, 3, 4], (s.count, 1)))
    np.testing.assert_array_equal(s.jaw_angle.numpy()[1], np.float32(.8))
    np.testing.assert_array_equal(s.sphere_q.numpy()[1], s.sphere_q_prev.numpy()[1])
    # A second finalize cannot mutate either first-model templates or live state.
    old_live, old_template = data(s), m.i4h.first_actions.numpy().tobytes()
    other = b.finalize(device=device)
    assert data(s) == old_live and m.i4h.first_actions.numpy().tobytes() == old_template
    for key in (*STATE_TYPES, 'actions'):
        assert getattr(m.i4h, 'first_'+key).ptr != getattr(other.i4h, 'first_'+key).ptr
    other_sim, other_s = runtime(a, other, captured)
    np.testing.assert_array_equal(other_s.actions.numpy()[..., 7:], 0)
    with pytest.raises(ValueError, match="foreign"):
        a.bind(other, sim.state, sim.control)
    alias = m.control(clone_variables=False)
    alias_sim, alias_s = runtime(a, m, captured, control=alias)
    assert alias_s.actions.ptr == m.i4h.first_actions.ptr
    alias_s.actions.fill_(.125)
    np.testing.assert_array_equal(m.control().i4h.first_actions.numpy(), .125)
    assert data(s) == old_live


@pytest.mark.parametrize("replacement", ['model', 'state', 'control', 'system', 'nested', 'layout', 'host', 'tensor'])
def test_replacements_rejected_even_by_prepare(replacement):
    a, _, m = authored('cpu')
    sim, s = runtime(a, m)
    if replacement in ('model', 'state', 'control'):
        owner, key = {'model': (m, 'first_jaw_open'), 'state': (sim.state, 'first_sphere_q_prev'),
                      'control': (sim.control, 'first_actions')}[replacement]
        setattr(owner.i4h, key, wp.clone(getattr(owner.i4h, key)))
    elif replacement == 'system':
        s.actions = wp.clone(s.actions)
    elif replacement == 'nested':
        s.fields.root = wp.clone(s.root)
    elif replacement == 'layout':
        s.root.shape = (1, 6)
    elif replacement == 'host':
        sim.scene.instruments['a'].spheres.pop()
    else:
        sim.solver.command_view('first').torch_views()['position'].transpose_(0, 1)
    for f in (sim.prepare, sim.step, lambda: sim.physics.command_views('first')):
        with pytest.raises((RuntimeError, ValueError), match="changed|replaced|fresh runtime|reconstruct"):
            f()


def test_direct_declaration_and_missing_late_changed_settings():
    scene = Scene('cpu', 2)
    tool(scene, 'a')
    declare_instruments(scene, max_velocity=3.)
    with pytest.raises(ValueError, match='duplicate'):
        declare_instruments(scene)
    scene.allocate()
    with pytest.raises(RuntimeError, match='before allocation'):
        declare_instruments(scene, system_id='late')
    with pytest.raises(ValueError, match='settings changed'):
        InstrumentControl(scene)
    control = InstrumentControl(scene, max_velocity=3.)
    assert control.max_velocity.numpy().tolist() == [3., 3.]
    undeclared = Scene('cpu', 1)
    undeclared.allocate()
    with pytest.raises(RuntimeError, match='storage missing'):
        InstrumentControl(undeclared)


@dataclass(eq=False)
class ExternalControl(System):
    is_control = True
    command_view_spec = {'all': (0, 5), 'tail': (2, 5)}

    @classmethod
    def declare_storage(cls, authoring, declarations, spec, records):
        domain = Domain('system', spec['id'], newton.Model.AttributeFrequency.WORLD, (authoring.scene.envs,))
        declarations.domain(domain)
        declarations.add(Field(domain, 'actions', Owner.CONTROL, wp.float32, (authoring.scene.envs, 5),
                               'world, command', 'preserve', row_width=5), np.zeros((authoring.scene.envs, 5), np.float32))
        authoring.scene.external_domain = domain

    def __post_init__(self):
        self.actions = self.scene.attr(self.scene.external_domain, 'actions')


@pytest.mark.parametrize('bad', [None, {}, {'bad': (1, 6)}, {'bad': (2, 2)}, {'bad': (True, 3)}, {'bad': [0, 2]}, {'bad': (-1, 2)}])
def test_external_discovery_invalid_specifications_and_unsupported(bad):
    cfg = {'sim': {'device': 'cpu', 'envs': 2}, 'assets': {},
           'systems': [{'type': 'external', 'id': 'custom'}, {'type': 'plain', 'id': 'plain'}]}
    a = author_case(cfg, systems={'external': ExternalControl, 'plain': System})
    b = newton.ModelBuilder()
    for e in range(2):
        b.begin_world()
        a.add_world(b, e)
        b.end_world()
    a.complete(b)
    m = b.finalize(device='cpu')
    sim, s = runtime(a, m)
    assert sim.physics.command_views('custom')['tail'].shape == (2, 3)
    with pytest.raises(NotImplementedError):
        sim.physics.command_views('plain')
    scene, systems = a.bind(m, m.state(), m.control())
    if bad is None:
        systems[0].actions = wp.clone(systems[0].actions)
    else:
        systems[0].command_view_spec = bad
    with pytest.raises(ValueError):
        Stepper.create(scene, systems, cuda_graph=False)


def test_declaration_initializer_order_and_freeze():
    from i4h_physics.storage import Declarations
    d = Declarations()
    world = Domain('system', 'probe', newton.Model.AttributeFrequency.WORLD, (1,))
    d.domain(world)
    d.add(Field(world, 'history', Owner.STATE, wp.float32, (1,), 'world', 'default'), [2.])
    calls = []
    def first(model):
        assert model.i4h.probe_history.numpy().tolist() == [2.]
        model.i4h.probe_history.fill_(7.)
        calls.append('first')
    def second(model):
        assert model.i4h.probe_history.numpy().tolist() == [7.]
        calls.append('second')
    d.initializer(first)
    d.initializer(second)
    b = newton.ModelBuilder()
    d.register_schema(b)
    with pytest.raises(RuntimeError, match='frozen'):
        d.initializer(first)
    b.begin_world()
    b.end_world()
    d.author_rows(b)
    m = b.finalize(device='cpu')
    assert calls == ['first', 'second']
    assert m.state().i4h.probe_history.numpy().tolist() == [7.]


def test_direct_layout_freezes_at_declaration():
    scene = Scene('cpu', 1)
    tool(scene, 'a')
    declare_instruments(scene)
    scene.instruments['a'].spheres.pop()
    with pytest.raises(RuntimeError, match='layout changed'):
        scene.allocate()


# A device mismatch requires allocating the command array on CUDA.
@pytest.mark.parametrize('change', ['dtype', pytest.param('device', id='cuda_device'), 'shape', 'strides'])
def test_invalid_command_array_layout(change):
    from i4h_physics.commands import CommandViews
    a, _, m = authored('cpu')
    sim, s = runtime(a, m)
    array = s.actions
    if change == 'dtype':
        s.actions = wp.zeros(array.shape, dtype=wp.float64, device='cpu')
    elif change == 'device':
        s.actions = wp.zeros(array.shape, dtype=wp.float32, device='cuda:0')
    elif change == 'shape':
        s.actions = array.flatten()
    else:
        s.actions = wp.array(ptr=array.ptr, shape=array.shape, dtype=wp.float32, device='cpu',
                             strides=(4, 12, 24), copy=False)
    with pytest.raises(ValueError):
        CommandViews(sim.scene, s)
