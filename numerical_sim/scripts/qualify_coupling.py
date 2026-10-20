# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coupled-scene qualification (F2, docs/component-contracts.md section 3): i4h physics held by MuJoCo-Warp bodies.

Run with the Isaac Lab 6864dd1 venv's python (MuJoCo-Warp, CUDA)::

    I4H_DATA=$PWD/data $PY scripts/qualify_coupling.py [--json coupling.json]

Scenarios, two worlds each (world 1 translated and turned 90 degrees about z):
- ``probe``: a holder on a stiff vertical slide (ke 1000 N/m) lifts two_worlds' probe into the hanging block (XPBD
  tissue, ``tet.model: xpbd``), 1 cm beside its centre of mass, one_way and two_way, at dt and at dt / 2.
- ``catheter``: a holder on the same slide carries the driver of a horizontal 20-node rod_direct catheter under gravity.

Printed per row: the value, the check and the development value (2026-10-06, RTX PRO 6000; not a qualification).

Known at development:
- with two_worlds' default PBD tissue the probe's reaction changed by ~0.8 at dt / 2 (its per-iteration stiffness
  factors depend on the time step), so the scene uses XPBD tissue (converged: its settled sag agrees within 0.3% over
  dt, substeps and iterations) and frictionless contact (a frictional stick has no unique equilibrium);
- the probe's normal reaction then changes by 5.3% and 4.5% at dt / 2 (env 0 marginally over the bound) and the full
  vector by 13% and 8%: the lateral part comes from the PBD contact projection on the coarse 2 cm mesh (more contact
  iterations or a finer block did not converge it further). The coupling itself is exact: reaction == ToolForce.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys

import numpy as np
import warp as wp

ENV_XFORMS = [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
              ((1.0, 0.5, 0.0), (0.0, 0.0, float(np.sin(np.pi / 4)), float(np.cos(np.pi / 4))))]
KE = 1000.0
SPRING, VOLUME = 1000.0, 1.0e10  # the XPBD block: N/m per edge, N/m^5 per tet; hanging, its bottom settles near z 0
ITERATIONS = 32  # converged Jacobi XPBD: the settled sag agrees within 0.3% over dt, substeps and iterations


def build(cfg, holder_z, target, mounts_for, mode):
    """Per world a holder on a vertical slide driven to ``target`` from ``holder_z``, then the component's particles."""
    import newton
    from i4h_isaaclab.composition import build_systems, load_scene, sim_settings
    from i4h_isaaclab.coupling import CoupledSolver, Mounts
    from i4h_physics.newton_model import add_env_particles
    from i4h_physics.sim import I4hSolver
    from i4h_physics.stepper import Stepper

    scene = load_scene(cfg)
    b = newton.ModelBuilder()
    for e, (pos, quat) in enumerate(ENV_XFORMS):
        env = wp.transform(wp.vec3(*pos), wp.quat(*quat))
        at = env * wp.transform((0.0, 0.0, holder_z), wp.quat_identity())
        b.begin_world()
        link = b.add_link(xform=at, mass=1.0, label="holder")
        b.add_shape_sphere(link, radius=0.01, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False))
        joint = b.add_joint_prismatic(-1, link, parent_xform=at, axis=(0.0, 0.0, 1.0), target_ke=KE, target_kd=100.0,
                                      armature=0.0)
        b.add_articulation([joint])
        add_env_particles(b, scene, e, xform=(np.float32(pos), np.float32(quat)))
        b.end_world()
    model = b.finalize(device="cuda:0")
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    scene.allocate(model, state)
    systems = build_systems(scene, cfg)
    ours = I4hSolver(model, scene, systems, state)
    Stepper(ours, **sim_settings(cfg)).prepare()
    mounts = Mounts(model, scene, systems, mounts_for(systems), mode == "two_way")
    control = model.control()
    control.joint_target_q.assign(np.full(model.joint_dof_count, target, np.float32))
    return systems, model, state, control, CoupledSolver(model, newton.solvers.SolverMuJoCo(model), ours, mounts)


def run(setup, dt, seconds, average=0.0):
    """Step ``seconds``; the pending reaction (force, torque) averaged over the last ``average`` seconds is returned
    too (a settled contact still chatters frame to frame)."""
    systems, model, state, control, coupled = setup
    steps, tail = int(round(seconds / dt)), int(round(average / dt))
    total = None
    for k in range(steps):
        state.body_f.zero_()  # Lab clears the forces after every step
        coupled.step(state, state, control, None, dt)
        if k >= steps - tail and coupled.mounts is not None:
            group = coupled.mounts.groups[0]
            sample = np.concatenate([group.force.numpy()[:, 0], group.torque.numpy()[:, 0]], axis=1)
            total = sample if total is None else total + sample
    wp.synchronize()
    return systems, model, state, coupled, (None if total is None else total / tail)


def by_type(systems, name):
    return next(s for s in systems if type(s).__name__ == name)


def probe_case(dt):
    from i4h_isaaclab.composition import read_case

    cfg = copy.deepcopy(read_case("two_worlds"))
    cfg["sim"].update(cuda_graph=False, dt=dt, iterations=ITERATIONS)
    # XPBD tissue (tet.model: xpbd): its stiffness does not depend on dt, so the refinement row tests the coupling.
    cfg["assets"] = {"block": dict(cfg["assets"]["block"], parts={"block": {"spring": SPRING, "volume": VOLUME}})}
    keep = ("particles", "tet", "surface", "instrument_control", "instrument_contact", "tool_force")
    # Frictionless contact: a frictional stick has no unique equilibrium (where it sticks depends on the path), so the
    # refinement row would measure stick-slip history rather than the coupling.
    tweak = {"tet": {"model": "xpbd"}, "instrument_contact": {"static_friction": 0.0, "kinetic_friction": 0.0}}
    cfg["systems"] = [dict(s, **tweak.get(s["type"], {})) for s in cfg["systems"] if s["type"] in keep]
    return cfg


def catheter_case(dt):
    return {"sim": {"device": "cuda:0", "envs": 2, "dt": dt, "substeps": 8, "iterations": 1, "cuda_graph": False,
                    "gravity": [0.0, 0.0, -9.81]},
            "assets": {"catheter": {"type": "rod", "start": [0.0, 0.0, 1.0], "direction": [1.0, 0.0, 0.0], "count": 20,
                                    "segment_length": 0.01, "radius": 0.002, "mass": 0.01}},
            "systems": [{"type": "particles", "damping": 1.0}, {"type": "rod_direct", "id": "rod", "body": "catheter"},
                        {"type": "catheter_control", "rod": "rod", "track_length": 0.0, "tip_edges": 2}]}


def probe(dt, mode):
    offset = ((0.01, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    setup = build(probe_case(dt), -0.1, 0.14, lambda systems: {"probe": {"body": "holder", "offset": offset}}, mode)
    systems, model, state, coupled, mean = run(setup, dt, 4.0, average=1.0)  # the block settles, the probe lifts into it
    group = coupled.mounts.groups[0]
    bodies = group.body.numpy()[:, 0]
    return {"z": state.body_q.numpy()[bodies, 2], "force": group.force.numpy()[:, 0].copy(),
            "torque": group.torque.numpy()[:, 0].copy(), "tool": by_type(systems, "ToolForce").force.numpy()[:, 0].copy(),
            "mean_force": mean[:, :3]}


def catheter(dt, mode):
    def mounts_for(systems):
        frame = tuple(float(x) for x in by_type(systems, "CatheterControl").mount_pose.numpy()[0, 0, 3:])
        return {"catheter": {"body": "holder", "offset": ((0.0, 0.0, 0.0), frame)}}

    systems, model, state, coupled, _ = run(build(catheter_case(dt), 1.0, 0.0, mounts_for, mode), dt, 2.0)
    group = coupled.mounts.groups[0]
    return {"z": state.body_q.numpy()[group.body.numpy()[:, 0], 2], "force": group.force.numpy()[:, 0].copy()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    rows = []

    def row(name, value, ok, dev):
        rows.append({"row": name, "value": value, "pass": bool(ok), "development": dev})
        print(f"{'PASS' if ok else 'FAIL'}  {name:<58} {value:<26} dev {dev}")

    dt = 1.0 / 60
    one, two, two_half = probe(dt, "one_way"), probe(dt, "two_way"), probe(dt / 2, "two_way")
    f = two["force"]
    row("probe: reaction == ToolForce (gain 1, uncapped)", f"{np.abs(f - two['tool']).max():.2e} N",
        np.allclose(f, two["tool"], rtol=1e-5, atol=1e-6), "<1e-6 N")
    row("probe: the block pushes the holder down (f_z < 0)", f"{f[:, 2].round(3).tolist()} N", (f[:, 2] < 0).all(), "about -2.9, -1.6 N")
    sag = one["z"] - two["z"]
    row("probe: holder gives way by |f_z| / ke (within 10%)", f"{sag.round(4).tolist()} m",
        np.allclose(sag, np.abs(f[:, 2]) / KE, rtol=0.1), "about 2.9e-3, 1.6e-3 m")
    row("probe: sphere contact moment about the root < 0.2 r |f|",
        f"{np.linalg.norm(two['torque'], axis=1).round(4).tolist()} N m",
        (np.linalg.norm(two["torque"], axis=1) < 0.2 * 0.015 * np.linalg.norm(f, axis=1)).all(), "about 4e-3, 1e-3 N m")
    mean, mean_half = two["mean_force"], two_half["mean_force"]
    normal = np.abs(mean_half[:, 2] - mean[:, 2]) / np.abs(mean[:, 2])
    row("probe: normal reaction (last-second mean) change at dt / 2 <= 5%", f"{normal.round(3).tolist()}",
        (normal <= 0.05).all(), "0.053, 0.045")
    full = np.linalg.norm(mean_half - mean, axis=1) / np.linalg.norm(mean, axis=1)
    print(f"info  probe: full reaction change at dt / 2 (lateral: contact on the 2 cm mesh)   {str(full.round(3).tolist()):<26} dev 0.13, 0.084")

    weight = 19 * 0.01 * 9.81
    cone, ctwo, chalf = catheter(dt, "one_way"), catheter(dt, "two_way"), catheter(dt / 2, "two_way")
    row("catheter: hold == weight of the free nodes (within 2%)", f"{ctwo['force'][:, 2].round(4).tolist()} N",
        np.allclose(ctwo["force"], [[0.0, 0.0, -weight]] * 2, rtol=0.02, atol=1e-4), f"-{weight:.4f} N")
    csag = cone["z"] - ctwo["z"]
    row("catheter: holder gives way by weight / ke (within 10%)", f"{csag.round(5).tolist()} m",
        np.allclose(csag, weight / KE, rtol=0.1), f"{weight / KE:.5f} m")
    cchange = np.linalg.norm(chalf["force"] - ctwo["force"], axis=1) / np.linalg.norm(ctwo["force"], axis=1)
    row("catheter: hold change at dt / 2 <= 5%", f"{cchange.round(4).tolist()}", (cchange <= 0.05).all(), "0.0032, 0.0038")

    if args.json:
        with open(args.json, "w") as out:
            json.dump(rows, out, indent=2)
    return 0 if all(r["pass"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
