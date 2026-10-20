# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify authored demo targets using controls only; no writes to simulated positions.

Run from the package root: .venv/bin/python scripts/qualify_procedures.py
"""

import argparse

import numpy as np
from i4h_endoluminal.systems.rods.catheter_control import CatheterControl
from i4h_isaaclab.composition import build_procedure, load_case, read_case
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
from i4h_surgical.systems.instruments.volumetric_grasp import VolumetricGrasp

NAVIGATION = {
    "aorta_static_reach": (60, 0.0, 0.0),
    "aorta_shell_reach": (250, 0.15, 0.05),
}


def navigation_steps(case):
    return NAVIGATION[case][0] + 100


def navigation_action(case, step):
    stop, rotation, bend = NAVIGATION[case]
    return np.float32([0.5 if step < stop else 0.0,
                       rotation if step < 30 else 0.0, bend if step < 30 else 0.0])


def sequence_action(task, axis):
    """Control the executor's current goal; no controller-owned sequence state."""
    rows, progress = task.rows.numpy(), task.progress.numpy()
    actions = np.zeros((len(rows), 3), dtype=np.float32)
    actions[:, 0] = np.clip(2 * (rows[:, 3:6] @ axis), -0.5, 0.5)
    actions[(rows[:, 8] <= progress[:, 1] * 0.5) | (rows[:, 11] != 0)] = 0
    return actions


def qualify_sequence(sim, task, cfg, lifecycle):
    control = next(s for s in sim.systems if isinstance(s, CatheterControl))
    axis = np.asarray(cfg["assets"]["catheter"]["direction"], dtype=np.float32)
    axis /= np.linalg.norm(axis)
    for _ in range(int(np.ceil(task.timeout / sim.dt)) + 1):
        control.actions.assign(sequence_action(task, axis))
        sim.step()
        task.update(sim.dt)
        if task.results()[0]["result"] != "running":
            break
    result = task.results()[0]
    print("sequence", "captured" if sim.cuda_graph else "uncaptured", result, flush=True)
    assert result["result"] == "success", result
    if lifecycle:
        sim.reset()
        task.reset(mask=sim._mask)
        for _ in range(int(np.ceil(task.timeout / sim.dt)) + 1):
            sim.step()
            task.update(sim.dt)
        assert task.results()[0]["result"] == "timeout", task.results()
        assert not (task.history.numpy()[:, :, 1] >= 0).any()
        sim.reset()
        task.reset(mask=sim._mask)
        assert not task.path_length.numpy().any()
        print("sequence zero-control timeout / reset passed", flush=True)
    return result


def navigation_lifecycle(sim, task, control, case):
    # Replay until inside, withdraw through controls before the hold completes.
    sim.reset()
    task.reset(mask=sim._mask)
    for i in range(navigation_steps(case)):
        control.actions.assign(navigation_action(case, i)[None])
        sim.step()
        task.update(sim.dt)
        if task.rows.numpy()[0, 6] == 1:
            break
    assert 0 < task.rows.numpy()[0, 10] < task.hold
    for _ in range(100):
        control.actions.assign(np.float32([[-1, 0, 0]]))
        sim.step()
        task.update(sim.dt)
        if task.rows.numpy()[0, 6] == 0:
            break
    assert task.rows.numpy()[0, 10] == 0 and task.rows.numpy()[0, 11] == 0
    sim.reset()
    task.reset(mask=sim._mask)
    # Authored 120-second timeout, measured by real physics steps, with zero controls.
    for _ in range(int(np.ceil(task.timeout / sim.dt)) + 1):
        sim.step()
        task.update(sim.dt)
    assert task.results()[0]["result"] == "timeout", task.results()
    sim.reset()
    task.reset(mask=sim._mask)
    assert task.rows.numpy()[0, 9] == 0 and task.rows.numpy()[0, 11] == 0
    print(case, "hold interruption / zero-control timeout / reset passed", flush=True)


def qualify(case, graph=True, lifecycle=False):
    cfg = read_case(case)
    sim = load_case(cfg, envs=1)
    sim.cuda_graph = graph
    task = build_procedure(cfg, sim)
    sim.prepare()
    if task.sequence:
        return qualify_sequence(sim, task, cfg, lifecycle)
    if task.cfg["type"] == "reach_target":
        control = next(s for s in sim.systems if isinstance(s, CatheterControl))
        for i in range(navigation_steps(case)):
            control.actions.assign(navigation_action(case, i)[None])
            sim.step()
            task.update(sim.dt)
            if task.results()[0]["result"] == "success":
                break
    else:
        control = next(s for s in sim.systems if isinstance(s, InstrumentControl))
        grasp = next(s for s in sim.systems if isinstance(s, VolumetricGrasp))
        k = control.names.index(task.cfg["tool"])
        sphere = int(grasp.jaw_start.numpy()[k]) + 2
        actions = control.actions.numpy()
        for _ in range(30):
            sim.step()  # settle with the authored controls
            task.update(sim.dt)
        for i in range(160):
            target = task.target_q.numpy()[0, task.index]
            jaw = control.sphere_q.numpy()[0, sphere]
            actions[0, k, :3] = control.root.numpy()[0, k] + (target - jaw)
            actions[0, k, 7] = float(i >= 80)
            control.actions.assign(actions)
            sim.step()
            task.update(sim.dt)
    result = task.results()[0]
    print(case, "captured" if graph else "uncaptured", result, flush=True)
    assert result["result"] == "success", result
    if lifecycle and task.cfg["type"] == "reach_target":
        navigation_lifecycle(sim, task, control, case)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--case", choices=(*NAVIGATION, "chole_grasp"))
    parser.add_argument("--lifecycle", action="store_true", help="also check interruption, zero-control timeout and reset")
    args = parser.parse_args()
    for case in [args.case] if args.case else (*NAVIGATION, "chole_grasp"):
        qualify(case, not args.no_cuda_graph, args.lifecycle)
