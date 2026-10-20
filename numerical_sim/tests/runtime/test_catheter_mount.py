# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""F2c: a catheter driver held by a host body (``catheter_control.mount``) and its reaction products
(docs/component-contracts.md section 3), on a short synthetic rod_direct catheter."""

import numpy as np
import pytest
import warp as wp
from i4h_isaaclab.composition import load_case

P, MASS, SEG = 20, 0.01, 0.01
REST = ((0.0, 0.0, 1.0), None)  # the rest root: start [0, 0, 1], frame from the rod's rest frames


def case(gravity=(0.0, 0.0, 0.0), track=0.05, damping=1.0, rod="rod_direct", mass=MASS, seg=SEG, inertia=None):
    return {"sim": {"device": "cuda:0", "envs": 2, "dt": 1.0 / 60, "substeps": 8, "iterations": 1, "cuda_graph": False,
                    "gravity": list(gravity)},
            "assets": {"catheter": {"type": "rod", "start": list(REST[0]), "direction": [1.0, 0.0, 0.0], "count": P,
                                    "segment_length": seg, "radius": 0.002, "mass": mass}},
            "systems": [{"type": "particles", "damping": damping},
                        {"type": rod, "id": "rod", "body": "catheter", **({"inertia": inertia} if inertia else {})},
                        {"type": "catheter_control", "rod": "rod", "track_length": track, "tip_edges": 2}]}


def control(sim):
    return next(s for s in sim.systems if type(s).__name__ == "CatheterControl")


def rod_q(sim):
    return sim.scene.field("q", sim.scene.body("catheter")).numpy().copy()


def drive(sim, steps, actions=(0.5, 0.4, 0.3)):
    cc = control(sim)
    cc.actions.assign(np.tile(np.float32(actions), (sim.scene.envs, 1)))
    for _ in range(steps):
        sim.step()
    wp.synchronize()


def rest_pose(cc):
    return cc.mount_pose.numpy().copy()  # (E, 1, 7): initialised at the rest root frame


def place(sim, moves):
    """Every env's rod moved rigidly by ``moves[e]``: positions, previous positions and node frames."""
    body, rod = sim.scene.body("catheter"), control(sim).rod
    for name in ("q", "q_prev"):
        arr = sim.scene.field(name, body)
        rows = arr.numpy()
        arr.assign(np.float32([[list(wp.transform_point(g, wp.vec3(*p))) for p in rows[e]] for e, g in enumerate(moves)]))
    F = rod.frame_count
    for frames in (rod.frame, rod.frame_pred):
        rows = frames.numpy().reshape(len(moves), F, 4)
        frames.assign(np.float32([[list(wp.transform_get_rotation(g) * wp.quat(*f)) for f in rows[e]]
                                  for e, g in enumerate(moves)]).reshape(-1, 4))


def test_a_mount_at_the_rest_root_drives_as_unmounted():
    """Equal up to float rounding: the mounted path carries the rest tangent through the mount's frame."""
    free, held = load_case(case()), load_case(case())
    control(held).mount()
    drive(free, 30)
    drive(held, 30)
    np.testing.assert_allclose(rod_q(held), rod_q(free), atol=1e-6)
    np.testing.assert_allclose(control(held).root.numpy(), control(free).root.numpy(), atol=1e-6)


def test_a_moved_mount_carries_the_track_and_the_rod_rigidly():
    """Without gravity the mounted rod is the unmounted one moved by the rigid motion from the rest root frame to the
    mount (per env, both envs differently)."""
    free, held = load_case(case()), load_case(case())
    cc = control(held)
    pose = rest_pose(cc)
    rest_q = wp.quat(*pose[0, 0, 3:])
    moves = [wp.transform(wp.vec3(0.3, -0.2, 0.1), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.7)),
             wp.transform(wp.vec3(-0.1, 0.4, 0.0), wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 1.0, 0.0)), -0.5))]
    rest = wp.transform(wp.vec3(*REST[0]), rest_q)
    mounts = [m * rest for m in moves]  # the moved rest frame: start and rest orientation together
    cc.mount_pose.assign(np.float32([[list(m.p) + list(m.q)] for m in mounts]))
    cc.mount()
    place(held, moves)  # the rod starts at the moved rest (a host would mount before the first step)
    drive(free, 40)
    drive(held, 40)
    q_free, q_held = rod_q(free), rod_q(held)
    for e, g in enumerate(moves):
        moved = np.array([wp.transform_point(g, wp.vec3(*p)) for p in q_free[e]])
        np.testing.assert_allclose(q_held[e], moved, atol=1e-4)  # float32 rounding over 40 steps: < 0.1% of the rod
    assert np.abs(q_free[0, -1] - q_free[0, 0]).max() > 0.05, "the rod was driven (inserted, rotated, bent)"


def test_root_hold_carries_the_weight():
    """A horizontal cantilever at rest under gravity: the force on the holder is the weight of the free nodes."""
    sim = load_case(case(gravity=(0.0, 0.0, -9.81), track=0.0))
    cc = control(sim)
    for _ in range(200):
        sim.step()
    sdt = sim.dt / sim.substeps
    force = -cc.impulse.numpy()[:, 0] / (sdt * sdt * sim.substeps)
    np.testing.assert_allclose(force, np.tile([0.0, 0.0, -(P - 1) * MASS * 9.81], (2, 1)), rtol=2e-3, atol=1e-6)


def test_momentum_balance_with_root_and_guide():
    """Every step, the impulse the holder receives (root hold plus guide snaps) is minus the free nodes' momentum change:
    the PBD identity with every external correction counted (no gravity, no damping, nothing else touches the rod). The
    mount moves sideways each step and the shaft starts bowed off the guide, so both terms are large. Heavy nodes: with
    rod_direct's unit rotational inertia a light rod is effectively rigid and would come to rest within each step."""
    sim = load_case(case(track=0.3, mass=10.0, seg=0.05))
    cc = control(sim)
    body = sim.scene.body("catheter")
    inv_mass = sim.scene.field("inv_mass", body).numpy()
    mass = np.where(inv_mass > 0, 1.0 / np.where(inv_mass > 0, inv_mass, 1.0), 0.0)  # the locked root has none
    q = sim.scene.field("q", body)
    bowed = q.numpy()
    bowed[:, 2:7, 2] += np.float32(0.002)  # 2 mm above the guide
    q.assign(bowed)
    pose = rest_pose(cc)
    cc.mount()
    sdt = sim.dt / sim.substeps

    def momentum():
        return (mass[..., None] * sim.scene.field("qd", body).numpy()).sum(axis=1)

    largest, worst = 0.0, 0.0
    for k in range(8):
        pose[:, 0, 1] += np.float32([0.002, -0.003]) * (k % 2 * 2 - 1)  # sideways, back and forth
        cc.mount_pose.assign(pose)
        before = momentum()
        sim.step()
        received = -cc.impulse.numpy()[:, 0] / (sdt * sdt * sim.substeps) * sim.dt  # force on the holder x dt
        largest = max(largest, float(np.abs(received).max()))
        worst = max(worst, float(np.abs(received + (momentum() - before)).max()))
    assert largest > 1e-3, "the mount and the guide moved the rod"
    assert worst < 1e-3 * largest, f"momentum balance off by {worst:.2e} (largest step impulse {largest:.2e})"
    assert np.abs(cc.moment.numpy()).max() > 0.0, "the guide's snaps have a moment about the root"


def test_rod_pbd_measures_the_guide_only():
    sim = load_case(case(rod="rod_pbd"))
    assert control(sim).root_reaction is False
    assert control(load_case(case())).root_reaction is True
    with pytest.raises(IndexError):
        control(sim).mount(1)


def test_root_hold_carries_the_weights_moment():
    """A cantilever settled under gravity (physical inertia: with the unit one the light frames' inertia, not the root,
    carries the weight's torque): the holder carries the weight's moment about the root, sum of r_i x m g, from the
    first edge's rotational multipliers. Lightly damped, so the bound allows the damped gravity's 1%."""
    sim = load_case(case(gravity=(0.0, 0.0, -9.81), track=0.0, damping=0.99, inertia="physical"))
    cc = control(sim)
    for _ in range(1200):
        sim.step()
    sdt = sim.dt / sim.substeps
    moment = -cc.moment.numpy()[:, 0] / (sdt * sdt * sim.substeps)
    q = sim.scene.field("q", sim.scene.body("catheter")).numpy()
    arm = q[:, 1:] - q[:, :1]
    expected = np.cross(arm, np.float32([0.0, 0.0, -MASS * 9.81])).sum(axis=1)
    assert np.abs(expected[:, 1]).min() > 0.01, "the weight has a lever arm"
    np.testing.assert_allclose(moment, expected, rtol=0.03, atol=2e-4)
