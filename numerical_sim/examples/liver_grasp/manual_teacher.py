# SPDX-License-Identifier: Apache-2.0
"""Live-target teacher initialized from manual capture geometry, not action replay."""

import numpy as np
from i4h_surgical.grasp_teacher import GraspTeacher


class ManualGeometryTeacher(GraspTeacher):
    def __init__(self, initial, workspace, example, dt=1 / 60):
        super().__init__(initial, workspace, dt)
        self.offset = np.array(example["capture_root"]) - np.array(
            example["capture_particle_positions"][0]
        )
        self.action[3:7] = example["capture_quaternion_xyzw"]
        self.capture_grip = max(0.52, example["capture_grip"])
        self.phase = "approach"
        self.arrival_ticks = 0

    def command(self, measurement, sphere_radius, particle_radius, jaw_angle):
        if self.phase == "approach":
            goal = measurement["geometry"][0] + self.offset
            delta = goal - self.action[:3]
            length = np.linalg.norm(delta)
            self.action[:3] += delta * min(
                1.0, (0.0001 if length < 0.002 else 0.001) / max(length, 1.0e-20)
            )
            # Approach with partially open jaws, remaining below the capture threshold.
            self.action[7] = min(0.45, self.action[7] + 0.015)
            self.arrival_ticks = self.arrival_ticks + 1 if length < 0.0003 else 0
            if self.arrival_ticks >= 3:
                self.transition("capture")
                self.closure = dict(
                    step=self.step,
                    jaw_angle=float(jaw_angle),
                    gap=float(measurement["gap"]),
                    root=self.action[:3].tolist(),
                    source="manual live-target geometry",
                )
        elif self.phase == "capture":
            self.action[7] = min(self.capture_grip, self.action[7] + 0.01)
        else:
            return super().command(
                measurement, sphere_radius, particle_radius, jaw_angle
            )
        if np.any(self.action[:3] < self.workspace[0]) or np.any(
            self.action[:3] > self.workspace[1]
        ):
            self.fail("workspace_command_exit")
        return self.action
