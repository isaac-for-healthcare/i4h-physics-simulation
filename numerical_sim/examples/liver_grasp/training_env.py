# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experiment-local RSL-RL 5.5.1 adapter; no Isaac Lab/runtime dependency changes."""

import json
from pathlib import Path

import numpy as np
import torch
import warp as wp
from i4h_surgical.config import load_case, read_case
from i4h_surgical.grasp_probe import GraspTargetProbe
from i4h_surgical.grasp_snapshot import HarnessSnapshot, identity
from i4h_surgical.systems.instruments.instrument_control import (
    place_instruments,
    shaft_rotation,
    wrist_rotation,
)
from manual_teacher import ManualGeometryTeacher
from rsl_rl.env import VecEnv
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
PHASES = (
    "approach",
    "capture",
    "qualify",
    "retract",
    "hold",
    "release",
    "relax",
    "passed",
    "failed",
)
SETTINGS = dict(
    physics_hz=60,
    control_hz=30,
    lift_m=0.02,
    start_front_m=0.03,
    start_jitter_m=0.005,
    max_increment_m=0.001,
    deadline_s=20,
    initial_retention_s=0.5,
    hold_s=1.0,
    destination_tolerance_m=0.002,
    release_deadline_s=0.5,
    relaxation_s=1.0,
    reward_schema="valid_patch_capped_lift_v2",
)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def examples():
    path = (
        ROOT
        / "recordings/liver/20261119_141840/physics_replay/successful_examples.json"
    )
    return [e for e in json.loads(path.read_text()) if e["episode"] in (1, 4, 5, 6, 15)]


def schedule(count, seed):
    """Balanced complete cycles, with no repeated target at cycle boundaries."""
    rng = np.random.default_rng(seed)
    result, previous = [], -1
    while len(result) < count:
        cycle = rng.permutation(5)
        if cycle[0] == previous:
            cycle[[0, 1]] = cycle[[1, 0]]
        for target in cycle:
            episode_seed = int(rng.integers(0, 2**31 - 1))
            jitter = np.random.default_rng(episode_seed).uniform(-0.005, 0.005, 3)
            result.append(
                dict(
                    target_index=int(target),
                    seed=episode_seed,
                    jitter_m=jitter.tolist(),
                )
            )
        previous = int(cycle[-1])
    return result[:count]


def lift_progress_reward(previous, current):
    """Lift potential saturates at the requested destination; overshoot earns no bonus."""
    old = np.clip(previous, 0.0, SETTINGS["lift_m"])
    new = np.clip(current, 0.0, SETTINGS["lift_m"])
    return float(np.clip((new - old) / 0.001, -1.0, 1.0) * 0.2)


class Evaluator:
    """Physical scoring only: policy outputs never get corrected by this state machine."""

    def __init__(self):
        self.phase = "approach"
        self.step = self.phase_step = self.hold_streak = 0
        self.failure = None
        self.acquisition_step = self.release_step = None
        self.surface_clear_step = self.volume_clear_step = None
        self.destination = self.acquired_position = None
        self.max_lift = self.max_slip = self.max_drift = 0.0
        self.max_inverted = 0

    @property
    def done(self):
        return self.phase in ("passed", "failed")

    def fail(self, reason):
        self.failure, self.phase = reason, "failed"

    def transition(self, phase):
        self.phase, self.phase_step = phase, self.step

    def observe(self, m, grip, finite, workspace_ok):
        if self.done:
            return
        self.step += 1
        if not finite:
            self.fail("non_finite")
            return
        if not workspace_ok:
            self.fail("workspace_exit")
            return
        if self.step >= 1200:
            self.fail("timeout")
            return
        if m["latched"] and self.acquisition_step is None:
            self.acquisition_step = self.step
            self.acquired_position = m["geometry"][0].copy()
            self.destination = self.acquired_position + [0.0, 0.02, 0.0]
            self.transition("qualify")
        if self.destination is not None:
            self.max_lift = max(
                self.max_lift, float(m["geometry"][0, 1] - self.acquired_position[1])
            )
            if self.phase != "relax" and grip >= 0.5:
                self.max_slip = max(self.max_slip, float(m["stats"][0]))
        if self.acquisition_step is not None and (
            self.phase not in ("release", "relax")
            or (self.phase == "release" and self.release_step is None and grip >= 0.5)
        ):
            c = m["counts"]
            if c[5] or not c[1]:
                self.fail("wrong_patch_capture")
                return
            if c[3] < (int(c[1]) + 1) // 2:
                self.fail("retention_loss")
                return
        if self.phase == "approach" and grip >= 0.5:
            self.transition("capture")
        elif self.phase == "capture" and self.step - self.phase_step >= 60:
            self.fail("empty_capture")
        elif self.phase == "qualify" and self.step - self.phase_step >= 30:
            self.transition("retract")
        elif self.phase in ("retract", "hold"):
            error = np.linalg.norm(m["geometry"][0] - self.destination)
            self.hold_streak = self.hold_streak + 1 if error <= 0.002 else 0
            self.phase = "hold" if error <= 0.002 else "retract"
            if self.hold_streak >= 60:
                self.transition("release")
        elif self.phase == "release":
            if grip < 0.5 and self.release_step is None:
                self.release_step = self.step - 1
            if self.release_step is None:
                return
            c = m["mechanism_counts"]
            if not c[0] and not c[2] and self.surface_clear_step is None:
                self.surface_clear_step = self.step
            if not c[1] and not c[3] and self.volume_clear_step is None:
                self.volume_clear_step = self.step
            if (
                self.surface_clear_step is not None
                and self.volume_clear_step is not None
            ):
                self.transition("relax")
            elif self.step - self.release_step >= 30:
                self.fail("release_timeout")
        elif self.phase == "relax":
            if grip >= 0.5 or np.any(m["mechanism_counts"]):
                self.fail("reacquisition_during_relaxation")
            elif self.step - self.phase_step >= 60:
                self.transition("passed")

    def result(self, spec, target):
        cleared = [self.surface_clear_step, self.volume_clear_step]
        latency = (
            None
            if self.release_step is None or None in cleared
            else (max(cleared) - self.release_step) / 60
        )
        return dict(
            **spec,
            target=target,
            status=self.phase,
            failure=self.failure,
            steps=self.step,
            acquisition_step=self.acquisition_step,
            achieved_lift_m=None if self.acquisition_step is None else self.max_lift,
            radial_slip_m=None if self.acquisition_step is None else self.max_slip,
            release_latency_s=latency,
            attachment_max_drift_m=self.max_drift,
            inverted_tetrahedra_max=self.max_inverted,
        )


class Teacher(ManualGeometryTeacher):
    def observe(self, *args, **kwargs):
        first = self.acquisition_step is None
        super().observe(*args, **kwargs)
        if first and self.acquisition_step is not None:
            self.destination[1] += 0.01


@wp.kernel
def stage(
    mask: wp.array(dtype=wp.bool),
    actions: wp.array3d(dtype=float),
    trocar: wp.array(dtype=wp.vec3),
    has_trocar: wp.array(dtype=int),
    root: wp.array2d(dtype=wp.vec3),
    shaft: wp.array2d(dtype=wp.quat),
    wrist: wp.array2d(dtype=wp.quat),
    jaw: wp.array2d(dtype=wp.quat),
):
    e = wp.tid()
    if mask[e]:
        p = wp.vec3(actions[e, 0, 0], actions[e, 0, 1], actions[e, 0, 2])
        q = wp.normalize(
            wp.quat(
                actions[e, 0, 3], actions[e, 0, 4], actions[e, 0, 5], actions[e, 0, 6]
            )
        )
        s = shaft_rotation(p, q, trocar[0], has_trocar[0])
        root[e, 0] = p
        shaft[e, 0] = s
        wrist[e, 0] = wrist_rotation(s, q)
        jaw[e, 0] = q


class LiverEnv(VecEnv):
    num_actions = 4
    max_episode_length = 600
    device = (
        "cpu"  # Small policy and synchronous numerical observation are faster on CPU.
    )
    cfg = SETTINGS

    def __init__(self, data, worlds, directory, seed=7341):
        self.num_envs = worlds
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.examples = examples()
        if len(self.examples) != 5:
            raise ValueError("expected five demonstrated targets")
        self.specs = schedule(100000, seed)
        self.cursor = 0
        self.sim = load_case(read_case("liver_pinned_grasp_1x", data=data), envs=worlds)
        self.sim.prepare()
        self.probe = p = GraspTargetProbe(self.sim.physics)
        cfg = read_case("liver_pinned_grasp_1x", data=data)
        ident = identity(
            self.sim.physics, dict(case=cfg, device="cuda:0", worlds=worlds), data
        )
        snap_path = self.directory / f"settled_{worlds}.npz"
        if snap_path.exists():
            self.snapshot = HarnessSnapshot(self.sim.physics, ident, {})
            self.snapshot.load(snap_path)
            self.snapshot.restore({}, p)
        else:
            previous = self.sim.scene.field("q", p.body).numpy()
            streak = np.zeros(worlds, int)
            for step in range(1, 1801):
                self.sim.step()
                q = self.sim.scene.field("q", p.body).numpy()
                speed = np.linalg.norm(q - previous, axis=-1).max(axis=1) * 60
                streak = np.where(speed < 0.005, streak + 1, 0)
                previous = q
                if step >= 180 and np.all(streak >= 60):
                    break
            if not np.all(streak >= 60):
                raise RuntimeError("canonical settling failed")
            self.snapshot = HarnessSnapshot(
                self.sim.physics,
                ident,
                dict(settling_steps=step, streak=streak.tolist()),
            )
            self.snapshot.write(snap_path)
            self.snapshot.load(snap_path)
        write_json(self.directory / f"identity_{worlds}.json", self.snapshot.metadata())
        self.tasks = [None] * worlds
        self.teachers = [None] * worlds
        self.current_specs = [None] * worlds
        self.workspaces = np.zeros((worlds, 2, 3))
        self.previous_action = np.zeros((worlds, 4), np.float32)
        self.previous_target = np.zeros((worlds, 3), np.float32)
        self.previous_root = np.zeros((worlds, 3), np.float32)
        self.episode_length_buf = torch.zeros(worlds, dtype=torch.long)
        self.completed = []
        self.diagnostics_every_physics_step = False
        self.reset(list(range(worlds)))

    def reset(self, ids, specs=None):
        p = self.probe
        self.snapshot.restore({}, p, ids)
        actions = p.ic.actions.numpy()
        for i, e in enumerate(ids):
            spec = specs[i] if specs is not None else self.specs[self.cursor]
            if specs is None:
                self.cursor += 1
            self.current_specs[e] = dict(spec)
            ex = self.examples[spec["target_index"]]
            target = ex["particle_ids"][0]
            p.select(target, e)
            q = self.sim.scene.field("q", p.body).numpy()[e, target]
            offset = np.array(ex["capture_root"]) - ex["capture_particle_positions"][0]
            goal = q + offset
            actions[e, 0, :3] = goal + [0.0, 0.0, 0.03] + np.array(spec["jitter_m"])
            actions[e, 0, 3:7] = ex["capture_quaternion_xyzw"]
            actions[e, 0, 7:] = 0
            points = np.array([actions[e, 0, :3], goal, goal + [0, 0.02, 0]])
            self.workspaces[e] = [points.min(0) - 0.02, points.max(0) + 0.02]
            self.tasks[e] = Evaluator()
            self.teachers[e] = Teacher(actions[e, 0], self.workspaces[e], ex)
            self.previous_action[e] = 0
            self.episode_length_buf[e] = 0
            p.arm(
                e
            )  # Always observe first physical acquisition, independent of controller phase.
        p.ic.actions.assign(actions)
        mask = wp.array(
            np.isin(np.arange(self.num_envs), ids),
            dtype=wp.bool,
            device=self.sim.scene.device,
        )
        wp.launch(
            stage,
            dim=self.num_envs,
            inputs=[mask, p.ic.actions, p.ic.trocar, p.ic.has_trocar],
            outputs=[p.ic.root, p.ic.shaft_q, p.ic.wrist_q, p.ic.jaw_q],
            device=self.sim.scene.device,
        )
        place_instruments(p.ic, self.sim.scene, mask, 1)
        p.sample()
        m = p.read()
        self.previous_target[ids] = m["geometry"][ids, 0]
        self.previous_root[ids] = p.ic.root.numpy()[ids, 0]
        self.measurement = m
        return self.get_observations()

    def get_observations(self):
        p, m = self.probe, self.measurement
        root = p.ic.root.numpy()[:, 0]
        action = p.ic.actions.numpy()[:, 0]
        geometry = m["geometry"]
        destination = np.array(
            [
                t.destination
                if t.destination is not None
                else geometry[e, 0] + [0, 0.02, 0]
                for e, t in enumerate(self.tasks)
            ]
        )
        phase = np.eye(len(PHASES))[[PHASES.index(t.phase) for t in self.tasks]]
        timers = np.array(
            [
                [
                    t.step / 1200,
                    (t.step - t.phase_step) / 1200,
                    t.hold_streak / 60,
                    0
                    if t.acquisition_step is None
                    else (t.step - t.acquisition_step) / 1200,
                ]
                for t in self.tasks
            ]
        )
        # Only current simulator state and past action. No teacher offset, command or future samples.
        obs = np.concatenate(
            [
                (geometry[:, 0] - root) * 100,
                geometry[:, 1],
                (geometry[:, 2] - root) * 100,
                (geometry[:, 3] - root) * 100,
                (geometry[:, 0] - self.previous_target) * 3000,
                (root - self.previous_root) * 3000,
                (destination - geometry[:, 0]) * 100,
                action[:, 3:8],
                m["jaw_state"],
                m["counts"] / 64,
                m["stats"] * 1000,
                m["mechanism_counts"] / 64,
                self.previous_action,
                phase,
                timers,
            ],
            axis=1,
        )
        return TensorDict(
            {"policy": torch.tensor(obs, dtype=torch.float32)},
            batch_size=[self.num_envs],
        )

    def teacher_actions(self):
        result = np.zeros((self.num_envs, 4), np.float32)
        actions = self.probe.ic.actions.numpy()[:, 0]
        radii = self.sim.scene.field("radius", self.probe.body).numpy()
        for e, teacher in enumerate(self.teachers):
            m = {k: v[e] for k, v in self.measurement.items()}
            command = teacher.command(
                m,
                float(m["nearest_radius"]),
                float(
                    radii[
                        e,
                        self.examples[self.current_specs[e]["target_index"]][
                            "particle_ids"
                        ][0],
                    ]
                ),
                float(m["jaw_state"][0]),
            )
            result[e, :3] = (command[:3] - actions[e, :3]) / 0.001
            result[e, 3] = command[7] * 2 - 1
        return torch.tensor(result)

    def step(self, actions, auto_reset=True):
        requested = actions.detach().cpu().numpy().copy()
        if requested.shape != (self.num_envs, 4) or not np.isfinite(requested).all():
            raise ValueError("actions must be finite (E,4)")
        applied = np.clip(requested, -1, 1)
        # Cap total Cartesian distance at 1 mm, including diagonal actions.
        applied[:, :3] /= np.maximum(
            1.0, np.linalg.norm(applied[:, :3], axis=1, keepdims=True)
        )
        p = self.probe
        physical = p.ic.actions.numpy()
        physical[:, 0, :3] += applied[:, :3] * 0.001
        physical[:, 0, 7] = (applied[:, 3] + 1) / 2
        physical[:, 0, 8:] = 0
        self.previous_target[:] = self.measurement["geometry"][:, 0]
        self.previous_root[:] = p.ic.root.numpy()[:, 0]
        before = [t.phase for t in self.tasks]
        was_done = [t.done for t in self.tasks]
        old_distance = np.linalg.norm(
            self.measurement["geometry"][:, 0] - self.measurement["geometry"][:, 2],
            axis=1,
        )
        old_lift = np.array([t.max_lift for t in self.tasks])
        p.ic.actions.assign(physical)
        for _ in range(2):
            self.sim.step()
            p.sample()
            m = p.read()
            root = p.ic.root.numpy()[:, 0]
            q = self.sim.scene.field("q", p.body).numpy()
            for e, task in enumerate(self.tasks):
                sample = {k: v[e] for k, v in m.items()}
                task.observe(
                    sample,
                    float(physical[e, 0, 7]),
                    bool(np.isfinite(q[e]).all()),
                    bool(
                        np.all(root[e] >= self.workspaces[e, 0])
                        and np.all(root[e] <= self.workspaces[e, 1])
                    ),
                )
                teacher = self.teachers[e]
                if teacher.phase == "release" and teacher.release_step is None:
                    teacher.release_step = teacher.step
                c = sample["mechanism_counts"]
                teacher.observe(
                    sample, root[e], *map(int, c), bool(np.isfinite(q[e]).all())
                )
            self.measurement = m
            if self.diagnostics_every_physics_step:
                self.record_diagnostics()
        self.previous_action[:] = applied
        self.episode_length_buf += 1
        distance = np.linalg.norm(m["geometry"][:, 0] - m["geometry"][:, 2], axis=1)
        rewards = np.clip((old_distance - distance) / 0.001, -1, 1) * 0.1
        for e, t in enumerate(self.tasks):
            rewards[e] += lift_progress_reward(old_lift[e], t.max_lift)
            if t.acquisition_step is not None and t.phase in (
                "qualify",
                "retract",
                "hold",
            ):
                rewards[e] += 0.02
            if (
                before[e] in ("approach", "capture")
                and t.acquisition_step is not None
                and m["counts"][e, 1] > 0
                and not m["counts"][e, 5]
            ):
                rewards[e] += 2
            if t.phase == "release" and before[e] != "release":
                rewards[e] += 3
            if t.done:
                rewards[e] += 10 if t.phase == "passed" else -3
        dones = torch.tensor([t.done for t in self.tasks])
        # The attempt deadline is a physical failure, not an artificial truncation to bootstrap through.
        extras = dict(
            time_outs=torch.zeros(self.num_envs, dtype=torch.bool),
            requested=requested,
            applied=applied,
            terminal_observation=self.get_observations().clone(),
            results=[],
        )
        ids = torch.where(dones & ~torch.tensor(was_done))[0].tolist()
        if ids:
            self.record_diagnostics()
        for e in ids:
            t = self.tasks[e]
            r = t.result(
                self.current_specs[e],
                self.examples[self.current_specs[e]["target_index"]]["particle_ids"][0],
            )
            extras["results"].append((e, r))
            self.completed.append(r)
        if auto_reset and ids:
            self.reset(ids)
        return (
            self.get_observations(),
            torch.tensor(rewards, dtype=torch.float32),
            dones,
            extras,
        )

    def record_diagnostics(self):
        from i4h_surgical.grasp_qualification import diagnostics

        q = self.sim.scene.field("q", self.probe.body).numpy()
        for e, t in enumerate(self.tasks):
            d = diagnostics(q[e : e + 1], self.probe.body, self.probe.rest)
            t.max_drift = max(t.max_drift, d["attachment_max_drift_m"])
            t.max_inverted = max(t.max_inverted, d["inverted"])
