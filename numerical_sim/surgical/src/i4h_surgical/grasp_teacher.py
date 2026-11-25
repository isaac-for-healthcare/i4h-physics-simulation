# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded standalone procedural reference, with per-physics-step qualification."""

import numpy as np
from i4h_surgical.grasp_geometry import approach_increment


class GraspTeacher:
    def __init__(self, initial, workspace, dt=1 / 60):
        self.action = np.array(initial, np.float32).copy()
        assert self.action.shape == (10,)
        self.action[7:] = 0
        self.workspace = np.asarray(workspace)
        self.dt = dt
        self.at_pregrasp = False
        self.phase = "approach"
        self.step = self.phase_step = self.hold_streak = 0
        self.destination = None
        self.failure = None
        self.acquisition_step = self.release_step = None
        self.surface_clear_step = self.volume_clear_step = None
        self.closure = None

    @property
    def done(self):
        return self.phase in ("passed", "failed")

    def fail(self, reason):
        self.failure = reason
        self.phase = "failed"

    def transition(self, phase):
        self.phase = phase
        self.phase_step = self.step

    def command(self, measurement, sphere_radius, particle_radius, jaw_angle):
        """Called exactly once per two physics samples. Returns physical ten-column action."""
        g, gap = measurement["geometry"], float(measurement["gap"])
        target, normal, _, sphere = g
        if self.phase in ("approach", "reposition"):
            if np.linalg.norm(normal) < 0.5:
                self.fail("invalid_live_normal")
                return self.action
            envelope = sphere_radius + particle_radius
            if gap < -0.0005:
                self.transition("reposition")
            if self.phase == "reposition":
                goal = target + normal * (envelope + 0.005)
                delta = goal - sphere
                if np.linalg.norm(delta) <= 0.0001:
                    self.transition("approach")
            elif -0.0005 <= gap <= 0 and np.dot(sphere - target, normal) > 0:
                self.action[7] = 1
                self.closure = dict(
                    step=self.step,
                    jaw_angle=float(jaw_angle),
                    gap=gap,
                    sphere=sphere.tolist(),
                    target=target.tolist(),
                    radius=float(sphere_radius),
                )
                self.transition("capture")
                return self.action
            else:
                # First reach +5 mm outside the envelope, then reduce clearance.
                goal_gap = (
                    0.005 if not getattr(self, "at_pregrasp", False) else -0.00025
                )
                goal = target + normal * (envelope + goal_gap)
                delta = goal - sphere
                if goal_gap > 0 and np.linalg.norm(delta) <= 0.0001:
                    self.at_pregrasp = True
                if self.at_pregrasp:
                    delta = approach_increment(target, normal, sphere, envelope, gap)
            limit = 0.0001 if gap < 0.002 else 0.001
            delta *= min(1.0, limit / max(np.linalg.norm(delta), 1.0e-20))
            self.action[:3] += delta
        elif self.phase == "retract":
            delta = self.destination - target
            self.action[:3] += delta * min(
                1.0, 0.001 / max(np.linalg.norm(delta), 1.0e-20)
            )
        elif self.phase in ("release", "relax"):
            self.action[7] = 0
        if np.any(self.action[:3] < self.workspace[0]) or np.any(
            self.action[:3] > self.workspace[1]
        ):
            self.fail("workspace_command_exit")
        return self.action

    def observe(
        self,
        measurement,
        root,
        surface_count,
        volume_count,
        surface_active,
        volume_active,
        finite=True,
    ):
        """Every physics step, including BOTH samples between teacher commands."""
        self.step += 1
        if self.done:
            return
        if not finite:
            self.fail("non_finite")
            return
        if np.any(root < self.workspace[0]) or np.any(root > self.workspace[1]):
            self.fail("workspace_exit")
            return
        if self.phase != "relax" and self.step * self.dt > 20:
            self.fail("timeout")
            return
        counts = measurement["counts"]
        if measurement["latched"] and self.acquisition_step is None:
            self.acquisition_step = self.step
            self.destination = measurement["geometry"][0].copy() + [0, 0.01, 0]
        if (
            self.phase in ("capture", "qualify", "retract", "hold")
            and measurement["latched"]
        ):
            if counts[5] or counts[1] == 0:
                self.fail("wrong_patch_capture")
                return
            if counts[3] < (int(counts[1]) + 1) // 2:
                self.fail("retention_loss")
                return
        if self.phase == "capture":
            if measurement["latched"]:
                self.transition("qualify")
            elif (self.step - self.phase_step) * self.dt >= 1:
                self.fail("empty_capture")
        elif self.phase == "qualify" and (self.step - self.phase_step) * self.dt >= 0.5:
            self.transition("retract")
        elif self.phase in ("retract", "hold"):
            error = np.linalg.norm(measurement["geometry"][0] - self.destination)
            self.hold_streak = self.hold_streak + 1 if error <= 0.002 else 0
            if error <= 0.002:
                self.phase = "hold"
            else:
                self.phase = "retract"
            if self.hold_streak * self.dt >= 1:
                self.transition("release")
        elif self.phase == "release":
            # Set by harness at the control tick that actually sends open, not at the phase transition.
            if self.release_step is None:
                return
            if (
                not surface_count
                and not surface_active
                and self.surface_clear_step is None
            ):
                self.surface_clear_step = self.step
            if (
                not volume_count
                and not volume_active
                and self.volume_clear_step is None
            ):
                self.volume_clear_step = self.step
            if (
                self.surface_clear_step is not None
                and self.volume_clear_step is not None
            ):
                self.transition("relax")
            elif (self.step - self.release_step) * self.dt >= 0.5:
                self.fail("release_timeout")
        elif self.phase == "relax" and (self.step - self.phase_step) * self.dt >= 1:
            self.transition("passed")
