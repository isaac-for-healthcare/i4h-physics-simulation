# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""F2: i4h physics stepped beside a host rigid solver in one Newton model (physics/component-contracts.md). Needs
MuJoCo-Warp (the Isaac Lab venv) and CUDA."""

import copy

import numpy as np
import pytest

pytest.importorskip("mujoco_warp")
newton = pytest.importorskip("newton")
import warp as wp  # noqa: E402
from i4h_isaaclab.coupling import check_rigid_solver_cfg  # noqa: E402
from i4h_physics.config import build_systems, load_case, load_scene, sim_settings  # noqa: E402
from i4h_physics.coupling import CoupledSolver, Mounts  # noqa: E402
from i4h_physics.mount_storage import declare_mounts  # noqa: E402
from i4h_physics.newton_model import add_env_particles, complete_scene  # noqa: E402
from i4h_physics.sim import I4hSolver  # noqa: E402
from i4h_physics.stepper import Stepper  # noqa: E402

ROD = {"type": "rod", "start": [0, 0, 0], "direction": [1, 0, 0], "count": 6, "segment_length": .02, "radius": .002}
CASE = {"sim": {"device": "cuda:0", "envs": 2, "dt": 1.0 / 120, "substeps": 2, "cuda_graph": False},
        "assets": {"rod": ROD}, "systems": [{"type": "particles"}, {"type": "rod_direct", "body": "rod"}]}
DT = 1.0 / 120


def pendulum(b, e):
    link = b.add_link(xform=wp.transform((float(e), 0.0, 1.0), wp.quat_identity()), mass=1.0)
    b.add_shape_capsule(link, radius=0.05, half_height=0.2)
    j = b.add_joint_revolute(-1, link, parent_xform=wp.transform((float(e), 0.0, 1.5), wp.quat_identity()),
                             child_xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()), axis=(1.0, 0.0, 0.0))
    b.add_articulation([j])


def host(with_rod=True):
    """Per world: a revolute pendulum (the host's rigid body), then (``with_rod``) the i4h rod through the hook."""
    scene = load_scene(CASE)
    from i4h_physics.rod_storage import declare_rod_direct
    declare_rod_direct(scene, "rod")
    b = newton.ModelBuilder()
    for e in range(2):
        b.begin_world()
        pendulum(b, e)
        if with_rod:
            add_env_particles(b, scene, e, xform=(np.float32([e, 0, 0]), np.float32([0, 0, 0, 1])))
        b.end_world()
    if with_rod:
        complete_scene(b, scene)
    model = b.finalize(device="cuda:0")
    state = model.state()
    state.joint_q.assign(np.full(model.joint_coord_count, 0.5, np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return scene, model, state


def test_coupled_step_integrates_each_once():
    scene, model, state = host()
    scene.allocate(model, state)
    ours = I4hSolver(model, scene, build_systems(scene, CASE), state)
    stepper = Stepper(ours, **sim_settings(CASE))
    stepper.prepare()  # configures the solver (dt, substeps)
    coupled = CoupledSolver(model, newton.solvers.SolverMuJoCo(model), ours)
    _, rmodel, rstate = host(with_rod=False)
    rigid_only = newton.solvers.SolverMuJoCo(rmodel)
    alone = load_case(CASE | {"sim": CASE["sim"] | {"gravity": [0.0, 0.0, -9.81]}})
    control, rcontrol = ours.control, rmodel.control()
    for _ in range(30):
        coupled.step(state, state, control, None, DT)
        rigid_only.step(rstate, rstate, rcontrol, None, DT)
        alone.step()
    wp.synchronize()
    rod = scene.field("q", scene.body("rod")).numpy()
    np.testing.assert_allclose(rod, alone.scene.field("q", alone.scene.body("rod")).numpy(), atol=1e-6)  # once
    np.testing.assert_allclose(state.body_q.numpy(), rstate.body_q.numpy(), atol=1e-6)  # the pendulum, unaffected
    assert not np.allclose(state.body_q.numpy()[0, :3], [0.0, 0.0, 1.0]), "the pendulum swings"
    with pytest.raises(RuntimeError, match="in place"):
        coupled.step(state, model.state(), control, None, DT)


def test_rigid_reset_leaves_i4h_particles():
    scene, model, state = host()
    scene.allocate(model, state)
    ours = I4hSolver(model, scene, build_systems(scene, CASE), state)
    Stepper(ours, **sim_settings(CASE)).prepare()
    coupled = CoupledSolver(model, newton.solvers.SolverMuJoCo(model), ours)
    for _ in range(10):
        coupled.step(state, state, ours.control, None, DT)
    before = state.particle_q.numpy().copy()
    mask = wp.array([True, False, False], dtype=wp.bool, device="cuda:0")
    coupled.reset(state, world_mask=mask)
    np.testing.assert_array_equal(state.particle_q.numpy(), before)


def test_unsupported_rigid_configurations_are_rejected():
    from types import SimpleNamespace

    check_rigid_solver_cfg(SimpleNamespace(solver_type="mujoco_warp"), "one_way")
    with pytest.raises(ValueError, match="use_mujoco_contacts=False"):
        check_rigid_solver_cfg(SimpleNamespace(solver_type="mujoco_warp", use_mujoco_contacts=False), "one_way")
    with pytest.raises(ValueError, match="Featherstone integrates every particle"):
        check_rigid_solver_cfg(SimpleNamespace(solver_type="featherstone"), "one_way")
    with pytest.raises(ValueError, match="coupling_mode"):
        check_rigid_solver_cfg(SimpleNamespace(solver_type="mujoco_warp"), "both")


# --- F2b: an instrument held by a host body ------------------------------------------------------------------------

ENV_XFORMS = [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
              ((1.0, 0.5, 0.0), (0.0, 0.0, float(np.sin(np.pi / 4)), float(np.cos(np.pi / 4))))]  # world 1: 90 deg about z
OFFSET = ((0.01, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))  # the probe 1 cm beside the holder's centre of mass
TARGET = 0.12  # the holder's slide (it sags 1 cm under gravity): probe root near z 0.06, into the block's bottom face (z 0.05)


def probe_case():
    """two_worlds' block and probe only (no rod, fluid or cautery)."""
    from i4h_host.composition import read_case

    cfg = copy.deepcopy(read_case("two_worlds"))
    cfg["sim"]["cuda_graph"] = False
    cfg["assets"] = {"block": cfg["assets"]["block"]}
    keep = ("particles", "tet", "surface", "instrument_control", "instrument_contact", "tool_force", "bounds")
    cfg["systems"] = [s for s in cfg["systems"] if s["type"] in keep]
    return cfg


def held_probe(mode):
    """Per world a holder sliding along z (a stiff drive towards TARGET) from z -0.05, then the block and probe."""
    from i4h_host.composition import build_systems as build
    from i4h_host.composition import load_scene as scene_of
    from i4h_physics import declare_tet

    cfg = probe_case()
    scene = scene_of(cfg)
    declare_tet(scene, "block")
    from i4h_physics import declare_surface
    declare_surface(scene)
    from i4h_surgical.interaction_storage import declare_appearance
    declare_appearance(scene)
    from i4h_surgical.instrument_storage import declare_instruments
    declare_instruments(scene)
    from i4h_surgical.interaction_storage import declare_instrument_contact
    contact = next(s for s in cfg["systems"] if s["type"] == "instrument_contact")
    declare_instrument_contact(scene, surface=contact["surface"], records=contact.get("records", 8))
    declare_mounts(scene, {"probe": {"body": "holder", "offset": OFFSET}}, two_way=mode == "two_way")
    b = newton.ModelBuilder()
    for e, (pos, quat) in enumerate(ENV_XFORMS):
        env = wp.transform(wp.vec3(*pos), wp.quat(*quat))
        b.begin_world()
        holder = b.add_link(xform=env * wp.transform((0.0, 0.0, -0.05), wp.quat_identity()), mass=1.0, label="holder")
        b.add_shape_sphere(holder, radius=0.01, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False))
        j = b.add_joint_prismatic(-1, holder, parent_xform=env * wp.transform((0.0, 0.0, -0.05), wp.quat_identity()),
                                  axis=(0.0, 0.0, 1.0), target_ke=1000.0, target_kd=100.0, armature=0.0)
        b.add_articulation([j])
        add_env_particles(b, scene, e, xform=(np.float32(pos), np.float32(quat)))
        b.end_world()
    complete_scene(b, scene)
    model = b.finalize(device="cuda:0")
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    scene.allocate(model, state)
    systems = build(scene, cfg)
    ours = I4hSolver(model, scene, systems, state)
    Stepper(ours, **sim_settings(cfg)).prepare()
    mounts = Mounts(model, scene, systems, {"probe": {"body": "holder", "offset": OFFSET}}, mode == "two_way")
    control = ours.control
    control.joint_target_q.assign(np.full(model.joint_dof_count, TARGET, np.float32))
    return scene, systems, model, state, control, CoupledSolver(model, newton.solvers.SolverMuJoCo(model), ours, mounts)


def by_type(systems, name):
    return next(s for s in systems if type(s).__name__ == name)


def run(mode, steps=60):
    scene, systems, model, state, control, coupled = held_probe(mode)
    dt = 1.0 / 60
    for _ in range(steps):
        state.body_f.zero_()  # Lab clears the forces after every step
        coupled.step(state, state, control, None, dt)
    wp.synchronize()
    return scene, systems, model, state, coupled


def test_mounted_probe_follows_its_body_in_the_component_frame():
    _, systems, _, state, coupled = run("one_way")
    ic, mounts = by_type(systems, "InstrumentControl"), coupled.mounts.groups[0]
    root, body_q, bodies = ic.root.numpy()[:, 0], state.body_q.numpy(), mounts.body.numpy()[:, 0]
    for e, (pos, quat) in enumerate(ENV_XFORMS):
        env = wp.transform(wp.vec3(*pos), wp.quat(*quat))
        body = wp.transform(wp.vec3(*body_q[bodies[e], :3]), wp.quat(*body_q[bodies[e], 3:]))
        held = wp.transform_inverse(env) * (body * wp.transform(wp.vec3(*OFFSET[0]), wp.quat(*OFFSET[1])))
        np.testing.assert_allclose(root[e], np.array(wp.transform_get_translation(held)), atol=1e-5)
        np.testing.assert_allclose(root[e, :2], OFFSET[0][:2], atol=1e-4)  # the slide is along the component's z
    assert (root[:, 2] > 0.03).all(), "the holder lifted the probe into the block"
    assert (by_type(systems, "ToolForce").count.numpy()[:, 0] > 0).all(), "the probe touches the block in both worlds"
    assert not state.body_f.numpy().any(), "one_way writes no reaction"


def test_two_way_reaction_is_the_tissue_force_on_the_holder():
    _, systems, model, state, coupled = run("two_way")
    tf, ic, mounts = by_type(systems, "ToolForce"), by_type(systems, "InstrumentControl"), coupled.mounts.groups[0]
    f, m = mounts.force.numpy()[:, 0], mounts.torque.numpy()[:, 0]
    np.testing.assert_allclose(f, tf.force.numpy()[:, 0], rtol=1e-5, atol=1e-6)  # ToolForce at gain 1, uncapped
    assert (f[:, 2] < 0.0).all(), "the block pushes the probe down"
    assert (np.linalg.norm(m, axis=1) < 0.2 * 0.015 * np.linalg.norm(f, axis=1)).all(), "a sphere's contact is radial"
    bodies, body_q, com, root = mounts.body.numpy()[:, 0], state.body_q.numpy(), model.body_com.numpy(), ic.root.numpy()[:, 0]
    state.body_f.zero_()
    coupled.mounts.apply(state, model)
    body_f = state.body_f.numpy()
    for e, (pos, quat) in enumerate(ENV_XFORMS):
        rot = np.array(wp.quat_to_matrix(wp.quat(*quat))).reshape(3, 3)
        fw = rot @ f[e]
        p_com = np.array(wp.transform_point(wp.transform(wp.vec3(*body_q[bodies[e], :3]), wp.quat(*body_q[bodies[e], 3:])),
                                            wp.vec3(*com[bodies[e]])))
        lever = rot @ root[e] + np.asarray(pos) - p_com
        np.testing.assert_allclose(lever, rot @ np.asarray(OFFSET[0]), atol=1e-4)  # mount offset from the centre of mass
        np.testing.assert_allclose(body_f[bodies[e], :3], fw, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(body_f[bodies[e], 3:], rot @ m[e] + np.cross(lever, fw), rtol=1e-4, atol=1e-6)
    assert not mounts.force.numpy().any() and not mounts.torque.numpy().any(), "the buffer is cleared once written"


def test_two_way_reaction_moves_the_holder_and_reset_clears_it():
    *_, one_state, _ = run("one_way")
    _, systems, _, state, coupled = run("two_way")
    z_one, z_two = one_state.body_q.numpy()[:, 2], state.body_q.numpy()[:, 2]
    assert (z_two < z_one - 1e-4).all(), "the tissue pushes the holder back down its slide"
    assert np.isfinite(state.body_q.numpy()).all()
    mounts = coupled.mounts.groups[0]
    assert mounts.force.numpy()[:, 0].any(axis=1).all()
    mask = wp.array([False, True, False], dtype=wp.bool, device="cuda:0")
    before_force, before_torque = mounts.force.numpy().copy(), mounts.torque.numpy().copy()
    coupled.reset(state, world_mask=mask)
    np.testing.assert_array_equal(mounts.force.numpy(), before_force)
    np.testing.assert_array_equal(mounts.torque.numpy(), before_torque)
    coupled.restore_episode(state, coupled.ours.control, world_mask=mask)
    assert mounts.force.numpy()[0].any() and not mounts.force.numpy()[1].any() and not mounts.torque.numpy()[1].any()


def test_mount_configuration_errors():
    scene, systems, model, *_ = held_probe("one_way")
    with pytest.raises(ValueError, match="no mount target 'grasper'"):
        Mounts(model, scene, systems, {"grasper": {"body": "holder"}}, False)
    with pytest.raises(ValueError, match="incompatible mount declaration"):
        Mounts(model, scene, systems, {"probe": {"body": "arm"}}, False)
    with pytest.raises(ValueError, match="must be"):
        Mounts(model, scene, systems, {"probe": {"link": "holder"}}, False)


# --- F2c: a catheter driver held by a host body --------------------------------------------------------------------

def held_catheter(mode):
    """Per world a holder on a vertical slide (a stiff drive at the rod's rest root, z 1) carrying the catheter driver
    of a horizontal 20-node rod under gravity; the driver frame is the rod's rest root frame."""
    from i4h_host.composition import build_systems as build
    from i4h_host.composition import load_scene as scene_of
    from test_catheter_mount import case as catheter_case

    cfg = catheter_case(gravity=(0.0, 0.0, -9.81), track=0.0)
    scene = scene_of(cfg)
    from i4h_endoluminal.rod_storage import declare_catheter_control
    from i4h_physics.rod_storage import declare_rod_direct
    declare_rod_direct(scene, 'catheter', system_id='rod')
    declare_catheter_control(scene, 'rod', track_length=0.0, tip_edges=2)
    rest_frame = tuple(float(x) for x in scene.body('catheter').rod.frames[0, 0])
    declared_mounts = {'catheter': {'body': 'driver', 'offset': ((0.0, 0.0, 0.0), rest_frame)}}
    declare_mounts(scene, declared_mounts, two_way=mode == 'two_way')
    b = newton.ModelBuilder()
    for e, (pos, quat) in enumerate(ENV_XFORMS):
        env = wp.transform(wp.vec3(*pos), wp.quat(*quat))
        b.begin_world()
        at = env * wp.transform((0.0, 0.0, 1.0), wp.quat_identity())
        holder = b.add_link(xform=at, mass=1.0, label="driver")
        b.add_shape_sphere(holder, radius=0.01, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False))
        j = b.add_joint_prismatic(-1, holder, parent_xform=at, axis=(0.0, 0.0, 1.0), target_ke=1000.0, target_kd=100.0,
                                  armature=0.0)
        b.add_articulation([j])
        add_env_particles(b, scene, e, xform=(np.float32(pos), np.float32(quat)))
        b.end_world()
    complete_scene(b, scene)
    model = b.finalize(device="cuda:0")
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    scene.allocate(model, state)
    systems = build(scene, cfg)
    ours = I4hSolver(model, scene, systems, state)
    stepper = Stepper(ours, **sim_settings(cfg))
    stepper.prepare()
    cc = by_type(systems, "CatheterControl")
    rest_frame = tuple(float(x) for x in cc.mount_pose.numpy()[0, 0, 3:])
    mounts = Mounts(model, scene, systems, {"catheter": {"body": "driver", "offset": ((0.0, 0.0, 0.0), rest_frame)}},
                    mode == "two_way")
    return systems, model, state, ours.control, CoupledSolver(model, newton.solvers.SolverMuJoCo(model), ours, mounts)


def test_catheter_held_by_a_body_two_way():
    from test_catheter_mount import MASS, P

    results = {}
    for mode in ("one_way", "two_way"):
        systems, model, state, control, coupled = held_catheter(mode)
        for _ in range(120):
            state.body_f.zero_()
            coupled.step(state, state, control, None, 1.0 / 60)
        wp.synchronize()
        group = coupled.mounts.groups[0]
        assert type(group.control).__name__ == "CatheterControl"
        results[mode] = state.body_q.numpy()[group.body.numpy()[:, 0], 2], group.force.numpy()[:, 0]
        root = group.control.root.numpy()[:, 0]
        np.testing.assert_allclose(root[:, :2], 0.0, atol=1e-4)  # the driver's frame, back in the component frame
        np.testing.assert_allclose(root[:, 2], state.body_q.numpy()[group.body.numpy()[:, 0], 2], atol=1e-4)  # z is shared
    (z_one, f_one), (z_two, f_two) = results["one_way"], results["two_way"]
    assert not f_one.any(), "one_way fills no reaction"
    weight = (P - 1) * MASS * 9.81
    np.testing.assert_allclose(f_two, np.tile([0.0, 0.0, -weight], (2, 1)), rtol=0.02, atol=1e-4)  # the rod hangs on it
    np.testing.assert_allclose(z_one - z_two, weight / 1000.0, rtol=0.1)  # the drive gives way by weight / stiffness


def test_rigid_substeps_match_a_substepped_rigid_solver():
    """The host's num_substeps run inside the coupled step: the pendulum equals MuJoCo-Warp stepped twice at dt / 2,
    and the rod still advances once per physics step."""
    scene, model, state = host()
    scene.allocate(model, state)
    ours = I4hSolver(model, scene, build_systems(scene, CASE), state)
    Stepper(ours, **sim_settings(CASE)).prepare()
    coupled = CoupledSolver(model, newton.solvers.SolverMuJoCo(model), ours, rigid_substeps=2)
    _, rmodel, rstate = host(with_rod=False)
    rigid = newton.solvers.SolverMuJoCo(rmodel)
    alone = load_case(CASE | {"sim": CASE["sim"] | {"gravity": [0.0, 0.0, -9.81]}})
    control, rcontrol = ours.control, rmodel.control()
    for _ in range(30):
        coupled.step(state, state, control, None, DT)
        for _ in range(2):
            rigid.step(rstate, rstate, rcontrol, None, DT / 2)
        alone.step()
    wp.synchronize()
    np.testing.assert_allclose(state.body_q.numpy(), rstate.body_q.numpy(), atol=1e-6)
    np.testing.assert_allclose(scene.field("q", scene.body("rod")).numpy(),
                               alone.scene.field("q", alone.scene.body("rod")).numpy(), atol=1e-6)
    with pytest.raises(ValueError, match="rigid_substeps"):
        CoupledSolver(model, coupled.rigid, ours, rigid_substeps=0)


def test_pending_reaction_delay_and_cache_reset():
    scene, systems, model, state, control, coupled = held_probe("two_way")
    ic = by_type(systems, "InstrumentControl")
    group = coupled.mounts.groups[0]
    np.testing.assert_array_equal(ic.root.numpy(), np.tile(ic.rest, (scene.envs, 1, 1)))
    pose = ic.mount_pose.numpy().copy()
    group.force.fill_(wp.vec3(1.0, 2.0, 3.0))
    group.torque.fill_(wp.vec3(4.0, 5.0, 6.0))
    coupled.reset(state)
    np.testing.assert_array_equal(ic.mount_pose.numpy(), pose)
    np.testing.assert_array_equal(group.force.numpy(), np.tile([[[1., 2., 3.]]], (2, 1, 1)))
    state.body_f.zero_()
    coupled.mounts.apply(state, model)
    first = state.body_f.numpy().copy()
    assert first.any()
    coupled.mounts.apply(state, model)
    np.testing.assert_array_equal(state.body_f.numpy(), first)
    for source in group.sources:
        source.impulse.fill_(wp.vec3(0.1, 0.2, 0.3))
        source.moment.fill_(wp.vec3(0.4, 0.5, 0.6))
    coupled.mounts.fill(coupled.ours.substeps, DT)
    np.testing.assert_array_equal(state.body_f.numpy(), first)  # filled for the next step only
    assert group.force.numpy().any()


@pytest.mark.parametrize("mode", ["one_way", "two_way"])
def test_catheter_mujoco_warp_captured(mode):
    systems, model, state, control, coupled = held_catheter(mode)
    wp.synchronize_device(model.device)
    with wp.ScopedCapture(device=model.device) as capture:
        state.body_f.zero_()
        coupled.step(state, state, control, None, 1.0 / 60)
    for _ in range(10):
        coupled.ours.check_binding()
        wp.capture_launch(capture.graph)
    wp.synchronize_device(model.device)
    group = coupled.mounts.groups[0]
    assert np.isfinite(state.body_q.numpy()).all()
    np.testing.assert_allclose(group.control.root.numpy(), group.control.mount_pose.numpy()[..., :3], atol=1e-6)
    assert bool(group.force.numpy().any()) == (mode == "two_way")

@pytest.mark.parametrize("captured", [False, True])
def test_a_disabled_catheter_driver_stops_its_reaction(captured):
    """Disabling catheter_control (then prepare) clears its published reaction, and the coupling no longer fills or
    applies a stale one: the holder feels no force on the following steps."""
    systems, model, state, control, coupled = held_catheter("two_way")
    for _ in range(30):
        state.body_f.zero_()
        coupled.step(state, state, control, None, 1.0 / 60)
    group = coupled.mounts.groups[0]
    wp.synchronize()
    assert np.abs(group.force.numpy()).max() > 0.1, "the rod's weight hangs on the holder"
    cc = by_type(systems, "CatheterControl")
    cc.enabled = False
    cc.prepare(coupled.ours.scene)
    assert not cc.impulse.numpy().any() and not cc.moment.numpy().any(), "prepare clears the disabled driver's products"
    for _ in range(3):
        state.body_f.zero_()
        coupled.step(state, state, control, None, 1.0 / 60)
        wp.synchronize()
        assert not group.force.numpy().any() and not group.torque.numpy().any(), "no stale reaction is filled"

    if captured:
        with wp.ScopedCapture(device=model.device) as capture:
            state.body_f.zero_()
            coupled.step(state, state, control, None, 1.0 / 60)
        wp.capture_launch(capture.graph)
        assert not group.force.numpy().any() and not group.torque.numpy().any()
        assert not state.body_f.numpy().any(), "disabled source contributes no force after recapture"
    cc.enabled = True
    for system in systems:
        system.prepare(coupled.ours.scene)
    if captured:
        with wp.ScopedCapture(device=model.device) as capture:
            state.body_f.zero_()
            coupled.step(state, state, control, None, 1.0 / 60)
        for _ in range(30):
            wp.capture_launch(capture.graph)
    else:
        for _ in range(30):
            state.body_f.zero_()
            coupled.step(state, state, control, None, 1.0 / 60)
    assert np.abs(group.force.numpy()).max() > 0.1, "re-enabled driver publishes its live reaction"
