# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional assessment over live physics products. No physics writes or Torch dependency.

Rows: target xyz, approach xyz (target - tool), qualifying, hold/required,
      distance, elapsed seconds, consecutive hold seconds, result code.
Result: 0 running, 1 success, 2 timeout, 3 invalid_target. Reward is a one-step pulse;
action_reward accumulates unweighted completion counts until begin_action().
Only Procedure owns ordering; TargetMeasurement instances are stateless samplers.
"""

import math

import numpy as np
import warp as wp
from i4h_physics.scene import ACTIVE, ACTIVE_INT, fill_rows
from i4h_physics.stepper import select_envs

RESULTS = ("running", "success", "timeout", "invalid_target")


def _check_keys(value, allowed, where):
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected a mapping")
    unknown = value.keys() - set(allowed)
    if unknown:
        raise ValueError(f"{where}: unsupported fields {sorted(unknown)}")


@wp.kernel
def measure(
    target_q: wp.array2d(dtype=wp.vec3),
    target_flags: wp.array2d(dtype=wp.int32),
    index: int,
    fixed: wp.vec3,
    tool_q: wp.array2d(dtype=wp.vec3),
    tool_flags: wp.array2d(dtype=wp.int32),
    tool_index: int,
    reach: int,
    radius: float,
    mask: wp.array(dtype=wp.bool),
    masked: int,
    rows: wp.array2d(dtype=float),
):
    e = wp.tid()
    if masked == 1 and not mask[e]:
        return
    rows[e, 11] = 0.0
    target = fixed
    if index >= 0:
        if (target_flags[e, index] & ACTIVE) == 0:
            rows[e, 11] = 3.0
        target = target_q[e, index]
    approach = target - tool_q[e, tool_index]
    distance = wp.length(approach)
    for j in range(3):
        rows[e, j] = target[j]
        rows[e, j + 3] = approach[j]
    rows[e, 8] = distance
    rows[e, 6] = 0.0
    if (
        reach == 1
        and rows[e, 11] == 0.0
        and (tool_flags[e, tool_index] & ACTIVE) != 0
        and distance <= radius
    ):
        rows[e, 6] = 1.0


class TargetMeasurement:
    extra_keys = ()
    is_reach = False
    def __init__(self, cfg):
        allowed = ("type", "tool", "target", "radius", "hold", "timeout") + self.extra_keys
        _check_keys(cfg, allowed, "procedure")
        required = ("type", "tool", "target", "radius") + self.extra_keys
        if set(required) - cfg.keys():
            raise ValueError(f"procedure requires {required}")
        self.cfg = cfg
        self.radius, self.hold, self.timeout = (
            float(cfg.get(k, d))
            for k, d in [("radius", 0), ("hold", 0.5), ("timeout", 120)]
        )
        if not all(
            math.isfinite(v) and 0 < v <= float(np.finfo(np.float32).max)
            for v in (self.radius, self.hold, self.timeout)
        ):
            raise ValueError(
                "procedure radius, hold and timeout must be finite, positive and supported by float32"
            )

    def bind(self, sim, legacy=False, physics_binding=None):
        self.sim, self.scene = sim, sim.scene
        self.physics_binding = physics_binding
        scene = self.scene
        self.rows = wp.zeros((scene.envs, 12), dtype=float, device=scene.device)
        self.mask = wp.zeros(scene.envs, dtype=wp.bool, device=scene.device)
        self._bind_tool()
        target = self.cfg["target"]
        _check_keys(target, ("position", "body", "index"), "procedure.target")
        self.index, self.fixed = -1, wp.vec3(0.0)
        self.target_body = self.tool_body
        if "position" in target:
            if (
                len(target) != 1
                or np.asarray(target["position"]).shape != (3,)
                or not np.isfinite(target["position"]).all()
                or np.max(np.abs(target["position"])) > float(np.finfo(np.float32).max)
            ):
                raise ValueError("fixed target requires only a finite position xyz")
            self.fixed = wp.vec3(*target["position"])
        else:
            if set(target) != {"body", "index"}:
                raise ValueError("attached target requires body and index")
            self.target_body, start, end = self._selection(target["body"])
            index = target["index"]
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < end - start
            ):
                raise ValueError("target particle index is outside its body/part")
            self.index = start + index
            if not np.all(self.target_body.rest.flags[:, self.index] & ACTIVE_INT) or (
                legacy
                and not np.all(
                    scene.field("flags", self.target_body).numpy()[:, self.index]
                    & ACTIVE_INT
                )
            ):
                raise ValueError("target particle is inactive or padded")
        self.target_q = scene.field("q", self.target_body)
        self.target_flags = scene.field("flags", self.target_body)
        self._measure()
        return self

    def _selection(self, name):
        if not isinstance(name, str) or not name:
            raise ValueError("procedure body must be a nonempty name")
        body_name, _, part = name.partition(".")
        body = self.scene.body(body_name)
        if part:
            if part not in body.parts:
                raise ValueError(f"unknown procedure body part {name!r}")
            p = body.parts[part]
            return body, p.particle_start, p.particle_start + p.particle_count
        return body, 0, body.count

    def _measure(self, masked=False, evidence=None, slot=0):
        wp.launch(
            measure,
            self.scene.envs,
            inputs=[
                self.target_q,
                self.target_flags,
                self.index,
                self.fixed,
                self.tool_q if evidence is None else evidence.positions,
                self.tool_flags if evidence is None else evidence.active,
                self.tool_index if evidence is None else slot,
                int(self.is_reach),
                self.radius,
                self.mask,
                int(masked),
            ],
            outputs=[self.rows],
            device=self.scene.device,
        )


@wp.kernel
def execute(
    samples: wp.array3d(dtype=float),
    holds: wp.array(dtype=float),
    dt: float,
    timeout: wp.float64,
    tip: wp.array2d(dtype=wp.vec3),
    tip_index: int,
    metric: int,
    mask: wp.array(dtype=wp.bool),
    masked: int,
    active: wp.array(dtype=int),
    elapsed: wp.array(dtype=wp.float64),
    hold: wp.array(dtype=wp.float64),
    result: wp.array(dtype=int),
    history: wp.array3d(dtype=wp.float64),
    current: wp.array2d(dtype=float),
    path: wp.array(dtype=float),
    previous: wp.array(dtype=wp.vec3),
    pulses: wp.array2d(dtype=float),
    completions: wp.array2d(dtype=float),
):
    e = wp.tid()
    if masked == 1 and not mask[e]:
        return
    for s in range(holds.shape[0]):
        pulses[e, s] = 0.0
    if result[e] != 0:
        return
    s = active[e]
    used = wp.min(wp.float64(dt), wp.max(wp.float64(0.0), timeout - elapsed[e]))
    elapsed[e] += used
    if metric == 1:
        point = tip[e, tip_index]
        path[e] += wp.length(point - previous[e]) * float(used / wp.float64(dt))
        previous[e] = point
    for j in range(9):
        current[e, j] = samples[s, e, j]
    if samples[s, e, 11] == 3.0:
        result[e] = 3
        return
    if current[e, 6] != 0.0:
        hold[e] += used
    else:
        hold[e] = wp.float64(0.0)
    if hold[e] + wp.float64(1.0e-6) >= wp.float64(holds[s]):
        history[e, s, 1] = elapsed[e]
        pulses[e, s] = 1.0
        completions[e, s] += 1.0
        if s == holds.shape[0] - 1:
            result[e] = 1
        elif elapsed[e] + wp.float64(1.0e-6) >= timeout:
            result[e] = 2
        else:
            s += 1
            active[e] = s
            history[e, s, 0] = elapsed[e]
            hold[e] = wp.float64(0.0)
            for j in range(9):
                current[e, j] = samples[s, e, j]
            if samples[s, e, 11] == 3.0:
                result[e] = 3
    elif elapsed[e] + wp.float64(1.0e-6) >= timeout:
        result[e] = 2


@wp.kernel
def reset_executor(
    mask: wp.array(dtype=wp.bool),
    samples: wp.array3d(dtype=float),
    tip: wp.array2d(dtype=wp.vec3),
    tip_index: int,
    active: wp.array(dtype=int),
    elapsed: wp.array(dtype=wp.float64),
    hold: wp.array(dtype=wp.float64),
    result: wp.array(dtype=int),
    history: wp.array3d(dtype=wp.float64),
    current: wp.array2d(dtype=float),
    path: wp.array(dtype=float),
    previous: wp.array(dtype=wp.vec3),
    pulses: wp.array2d(dtype=float),
    completions: wp.array2d(dtype=float),
):
    e = wp.tid()
    if not mask[e]:
        return
    active[e] = 0
    elapsed[e] = wp.float64(0.0)
    hold[e] = wp.float64(0.0)
    result[e] = int(samples[0, e, 11])
    path[e] = 0.0
    previous[e] = tip[e, tip_index]
    for s in range(history.shape[1]):
        history[e, s, 0] = wp.float64(-1.0)
        history[e, s, 1] = wp.float64(-1.0)
        pulses[e, s] = 0.0
        completions[e, s] = 0.0
    history[e, 0, 0] = wp.float64(0.0)
    for j in range(9):
        current[e, j] = samples[0, e, j]


@wp.kernel
def present(
    active: wp.array(dtype=int),
    elapsed: wp.array(dtype=wp.float64),
    hold: wp.array(dtype=wp.float64),
    result: wp.array(dtype=int),
    current: wp.array2d(dtype=float),
    path: wp.array(dtype=float),
    holds: wp.array(dtype=float),
    radii: wp.array(dtype=float),
    pulses: wp.array2d(dtype=float),
    completions: wp.array2d(dtype=float),
    rows: wp.array2d(dtype=float),
    progress: wp.array2d(dtype=float),
    reward: wp.array(dtype=float),
    action_reward: wp.array(dtype=float),
):
    e = wp.tid()
    s = active[e]
    for j in range(9):
        rows[e, j] = current[e, j]
    rows[e, 7] = wp.min(1.0, float(hold[e]) / holds[s])
    rows[e, 9] = float(elapsed[e])
    rows[e, 10] = float(hold[e])
    rows[e, 11] = float(result[e])
    progress[e, 0] = float(s)
    progress[e, 1] = radii[s]
    progress[e, 2] = holds[s]
    progress[e, 3] = path[e]
    reward[e] = 0.0
    action_reward[e] = 0.0
    for k in range(holds.shape[0]):
        reward[e] += pulses[e, k]
        action_reward[e] += completions[e, k]


@wp.kernel
def select_sample(count: wp.array(dtype=int), slot: int,
                  consumed: wp.array(dtype=int), mask: wp.array(dtype=wp.bool)):
    e = wp.tid()
    mask[e] = slot < count[e] and slot >= consumed[e]
    if mask[e]:
        consumed[e] = slot + 1


class Procedure:
    """One ordered executor for legacy and sequence configurations.

    History columns are activation/completion seconds, -1 means not occurred.
    Rows and progress are derived presentation buffers; accessors return copies.
    Path length is a sampled world-space estimate, linearly apportioned at timeout.
    """

    def __init__(self, spec, sim, *, measurements, physics_binding=None, horizon_s=None):
        _check_keys(spec, spec.keys() if isinstance(spec, dict) else (), "procedure")
        self.cfg, self.sim, self.scene = spec, sim, sim.scene
        self.physics_binding = physics_binding
        self.sequence = "steps" in spec
        if self.sequence:
            _check_keys(spec, ("steps", "timeout", "metrics"), "procedure")
            authored = spec["steps"]
            if not isinstance(authored, list) or not authored:
                raise ValueError("procedure.steps must be a nonempty list")
        else:
            authored = [spec]
        self.timeout = float(spec.get("timeout", 120))
        if not math.isfinite(self.timeout) or not 0 < self.timeout <= float(
            np.finfo(np.float32).max
        ):
            raise ValueError("procedure timeout must be finite and positive")
        self.timeout_source = "procedure"
        if horizon_s is not None:
            horizon_s = float(horizon_s)
            if not math.isfinite(horizon_s) or horizon_s <= 0:
                raise ValueError("host horizon must be finite and positive")
            if horizon_s < self.timeout:
                self.timeout, self.timeout_source = horizon_s, "host_horizon"
        self.steps, self.ids = [], []
        for cfg in authored:
            if not isinstance(cfg, dict):
                raise ValueError("procedure step must be a mapping")
            sid = cfg.get("id") if self.sequence else "target"
            if not isinstance(sid, str) or not sid.strip() or sid in self.ids:
                raise ValueError("procedure step IDs must be unique nonempty strings")
            if self.sequence and "timeout" in cfg:
                raise ValueError("independent step timeouts are unsupported")
            params = (
                {k: v for k, v in cfg.items() if k != "id"} if self.sequence else cfg
            )
            kind = params.get("type")
            if kind not in measurements:
                raise ValueError(f"unknown procedure type {kind!r}")
            step = measurements[kind](params)
            step.bind(sim, legacy=not self.sequence, physics_binding=physics_binding)
            self.steps.append(step)
            self.ids.append(sid)
        self.step_count = len(self.steps)
        self.observation_size = 10 if self.sequence else 8
        self.radius, self.hold = (
            self.steps[0].radius,
            self.steps[0].hold,
        )  # legacy metadata
        self.target_q, self.index = self.steps[0].target_q, self.steps[0].index
        metrics = spec.get("metrics", {}) if self.sequence else {}
        _check_keys(metrics, ("tip_path_length",), "procedure.metrics")
        self.has_path = "tip_path_length" in metrics
        self.tip_q, self.tip_index = self.steps[0].tool_q, self.steps[0].tool_index
        if self.has_path:
            _check_keys(
                metrics["tip_path_length"],
                ("tool",),
                "procedure.metrics.tip_path_length",
            )
            name = metrics["tip_path_length"].get("tool")
            body = self.scene.body(name)
            if body.rod is None:
                raise ValueError("tip_path_length tool must be a rod")
            self.tip_q, self.tip_index = self.scene.field("q", body), body.count - 1
            if not np.all(body.rest.flags[:, self.tip_index] & ACTIVE_INT):
                raise ValueError("tip_path_length tool tip is inactive or padded")
        e, n, dev = self.scene.envs, self.step_count, self.scene.device
        self.samples = wp.zeros((n, e, 12), dtype=float, device=dev)
        self.holds = wp.array([s.hold for s in self.steps], dtype=float, device=dev)
        self.radii = wp.array([s.radius for s in self.steps], dtype=float, device=dev)
        self.active_step = wp.zeros(e, dtype=int, device=dev)
        self.elapsed = wp.zeros(e, dtype=wp.float64, device=dev)
        self.active_hold = wp.zeros_like(self.elapsed)
        self.result = wp.zeros(e, dtype=int, device=dev)
        self.history = wp.full((e, n, 2), -1.0, dtype=wp.float64, device=dev)
        self.current = wp.zeros((e, 9), dtype=float, device=dev)
        self.path_length = wp.zeros(e, dtype=float, device=dev)
        self.previous_tip = wp.zeros(e, dtype=wp.vec3, device=dev)
        self.completion_pulses = wp.zeros((e, n), dtype=float, device=dev)
        self.action_completions = wp.zeros_like(self.completion_pulses)
        self.rows = wp.zeros((e, 12), dtype=float, device=dev)
        self.progress = wp.zeros((e, 4), dtype=float, device=dev)
        self.reward = wp.zeros(e, dtype=float, device=dev)
        self.action_reward = wp.zeros_like(self.reward)
        self.mask = wp.zeros(e, dtype=wp.bool, device=dev)
        self.consumed = wp.zeros(e, dtype=int, device=dev)
        self.reset()

    # Legacy diagnostic tests can narrow a grasp's measured body selection.
    @property
    def start(self):
        return self.steps[0].start

    @start.setter
    def start(self, value):
        self.steps[0].start = value

    @property
    def end(self):
        return self.steps[0].end

    @end.setter
    def end(self, value):
        self.steps[0].end = value

    def _sample(self, evidence=None, slot=0):
        for i, step in enumerate(self.steps):
            step._measure(evidence=evidence, slot=slot)
            step._qualify()
            wp.copy(
                self.samples,
                step.rows,
                dest_offset=i * self.scene.envs * 12,
                count=self.scene.envs * 12,
            )

    def _state(self):
        return [
            self.active_step,
            self.elapsed,
            self.active_hold,
            self.result,
            self.history,
            self.current,
            self.path_length,
            self.previous_tip,
            self.completion_pulses,
            self.action_completions,
        ]

    def _present(self):
        wp.launch(
            present,
            self.scene.envs,
            inputs=[
                self.active_step,
                self.elapsed,
                self.active_hold,
                self.result,
                self.current,
                self.path_length,
                self.holds,
                self.radii,
                self.completion_pulses,
                self.action_completions,
            ],
            outputs=[self.rows, self.progress, self.reward, self.action_reward],
            device=self.scene.device,
        )

    def update(self, dt):
        if not math.isfinite(dt) or not 0 < dt <= float(np.finfo(np.float32).max):
            raise ValueError("procedure dt must be finite and positive")
        self._check_valid()
        self._sample()
        self._advance(dt, self.tip_q, self.tip_index)
        self._present()

    def _advance(self, dt, tip_q, tip_index, masked=False):
        wp.launch(
            execute,
            self.scene.envs,
            inputs=[
                self.samples,
                self.holds,
                dt,
                self.timeout,
                tip_q,
                tip_index,
                int(self.has_path),
                self.mask,
                int(masked),
            ],
            outputs=self._state(),
            device=self.scene.device,
        )

    def _check_valid(self):
        if self.physics_binding is not None:
            self.physics_binding.owner.check_valid()

    def consume(self, evidence, dt, *, validate=False):
        """Replay each recorded physics sample once, in order, using the shared executor.

        Single fixed rod targets only. begin_action() resets the replay cursor;
        publication never schedules assessment. count/overflow remain readable.
        Validation reads device counters and synchronizes before any state changes.
        The default path performs no device-to-host reads.
        """
        self._check_valid()
        evidence.owner.check_valid()
        if (self.physics_binding is None or evidence.owner is not self.physics_binding.owner
                or evidence.body is not self.physics_binding.body or self.sequence
                or self.index != -1):
            raise ValueError("tip evidence requires a single fixed target on the bound catheter")
        if not math.isfinite(dt) or not 0 < dt <= float(np.finfo(np.float32).max):
            raise ValueError("procedure dt must be finite and positive")
        if validate:
            counts, overflow = evidence.count.numpy(), evidence.overflow.numpy()
            invalid = np.flatnonzero((counts != evidence.capacity) | (overflow != 0))
            if invalid.size:
                details = "; ".join(
                    f"env {e}: count={counts[e]}, overflow={overflow[e]}" for e in invalid
                )
                raise RuntimeError(f"invalid tip evidence (capacity={evidence.capacity}): {details}")
        for slot in range(evidence.capacity):
            wp.launch(select_sample, self.scene.envs, inputs=[evidence.count, slot],
                      outputs=[self.consumed, self.mask], device=self.scene.device)
            self._sample(evidence, slot)
            self._advance(dt, evidence.positions, slot, masked=True)
        self._present()

    def begin_action(self):
        self._check_valid()
        self.consumed.zero_()
        self.action_completions.zero_()
        self.action_reward.zero_()

    def reset(self, env_ids=None, mask=None):
        self._check_valid()
        if select_envs(self.mask, env_ids, mask, debug=bool(getattr(self.sim, "debug", False))):
            wp.launch(fill_rows, (self.scene.envs, 1), inputs=[self.mask, 0],
                      outputs=[self.consumed.reshape((self.scene.envs, 1))], device=self.scene.device)
            self._sample()
            wp.launch(
                reset_executor,
                self.scene.envs,
                inputs=[self.mask, self.samples, self.tip_q, self.tip_index],
                outputs=self._state(),
                device=self.scene.device,
            )
            self._present()

    def measurements(self):
        return self.current.numpy().copy()

    def progress_state(self):
        return self.progress.numpy().copy()

    def metrics(self):
        return (
            {"tip_path_length": self.path_length.numpy().copy()}
            if self.has_path
            else {}
        )

    def step_history(self):
        h, elapsed = self.history.numpy(), self.elapsed.numpy()
        return [
            [
                dict(
                    id=sid,
                    complete=bool(c >= 0),
                    activated=None if a < 0 else float(a),
                    completed=None if c < 0 else float(c),
                    duration=None
                    if a < 0
                    else float((c if c >= 0 else elapsed[e]) - a),
                )
                for sid, (a, c) in zip(self.ids, row)
            ]
            for e, row in enumerate(h)
        ]

    def results(self):
        history, progress = self.step_history(), self.progress.numpy()
        return [
            dict(
                tool=self.steps[int(p[0])].cfg["tool"],
                target=r[:3].tolist(),
                distance=float(r[8]),
                hold=float(r[10]),
                elapsed=float(r[9]),
                result=RESULTS[int(r[11])],
                timeout_source=self.timeout_source,
                timeout_s=self.timeout,
                active_step=self.ids[int(p[0])],
                steps=h,
                metrics={"tip_path_length": float(p[3])} if self.has_path else {},
            )
            for r, p, h in zip(self.rows.numpy(), progress, history)
        ]
