# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Three-point bending and single-guide pull-through with the i4h RodDirect solver.

Run with the project's .venv/bin/python. SI units internally.
"""

import argparse
import csv

import numpy as np
import warp as wp
from harrison_inputs import SLOPES
from i4h_physics import Scene, Stepper
from i4h_physics.loaders import add_rod
from i4h_physics.rod_storage import declare_rod_direct
from i4h_physics.systems.base import System
from i4h_physics.systems.mechanics.particles import Particles
from i4h_physics.systems.rods.rod_direct import RodDirect


@wp.kernel
def project_guides(
    q: wp.array(dtype=wp.vec3),
    weights: wp.array(dtype=float),
    guides: wp.array(dtype=wp.vec3),
    dt: float,
    reaction: wp.array(dtype=wp.vec3),
    segment: wp.array(dtype=int),
):
    # Serial for this single free rod, so adjacent guides cannot race on nodes.
    for g in range(guides.shape[0]):
        best = float(1.0e20)
        index = int(0)
        fraction = float(0.0)
        for i in range(q.shape[0] - 1):
            edge = q[i + 1] - q[i]
            u = wp.clamp(
                wp.dot(guides[g] - q[i], edge) / wp.max(wp.length_sq(edge), 1.0e-20),
                0.0,
                1.0,
            )
            d2 = wp.length_sq(guides[g] - (q[i] + u * edge))
            if d2 < best:
                best = d2
                index = i
                fraction = u
        a = 1.0 - fraction
        b = fraction
        edge = q[index + 1] - q[index]
        normal_error = guides[g] - (q[index] + b * edge)
        tangent = wp.normalize(edge)
        normal_error -= wp.dot(normal_error, tangent) * tangent
        impulse = normal_error / (a * a * weights[index] + b * b * weights[index + 1])
        q[index] += weights[index] * a * impulse
        q[index + 1] += weights[index + 1] * b * impulse
        reaction[g] += impulse / (dt * dt)
        segment[g] = index


@wp.kernel
def press(
    q: wp.array(dtype=wp.vec3),
    weights: wp.array(dtype=float),
    target: wp.array(dtype=float),
    dt: float,
    force: wp.array(dtype=float),
):
    i = q.shape[0] // 2
    dy = target[0] - q[i][1]
    q[i] += wp.vec3(0.0, dy, 0.0)
    force[0] -= dy / (weights[i] * dt * dt)


@wp.kernel
def advance_nose(
    frame_start: wp.array(dtype=float),
    target: wp.array(dtype=float),
    elapsed: float,
    limit: float,
):
    target[0] = wp.max(-limit, wp.min(0.0, frame_start[0] - elapsed / 3000.0))


@wp.kernel
def apply_load(
    q: wp.array(dtype=wp.vec3),
    weights: wp.array(dtype=float),
    load: wp.array(dtype=float),
    dt: float,
):
    i = q.shape[0] // 2
    q[i] -= wp.vec3(0.0, weights[i] * load[0] * dt * dt, 0.0)


class Fixture(System):
    def __init__(self, scene, pull, load_control=False):
        super().__init__(scene)
        locations = [[0.0, 0.0, 0.0]] if pull else [[-0.02, 0.0, 0.0], [0.02, 0.0, 0.0]]
        self.guides = wp.array(locations, dtype=wp.vec3, device=scene.device)
        self.pull = pull
        self.load_control = load_control
        self.target = wp.zeros(1, device=scene.device)
        self.frame_start = wp.zeros_like(self.target)
        self.stroke_limit = 0.005
        self.force = wp.zeros_like(self.target)
        self.reaction = wp.zeros_like(self.guides)
        self.segment = wp.zeros(len(locations), dtype=int, device=scene.device)

    def substep_start(self, scene, dt, s, substeps):
        if self.load_control:
            wp.launch(
                apply_load,
                1,
                [scene.q, scene.inv_mass, self.target, dt],
                device=scene.device,
            )
        elif not self.pull and substeps > 1:
            wp.launch(
                advance_nose,
                1,
                [self.frame_start, self.target, (s + 1) * dt, self.stroke_limit],
                device=scene.device,
            )

    def iteration(self, scene, dt, k, iterations):
        if k == 0:
            self.force.zero_()
            self.reaction.zero_()
        wp.launch(
            project_guides,
            1,
            [scene.q, scene.inv_mass, self.guides, dt, self.reaction, self.segment],
            device=scene.device,
        )
        if not self.pull and not self.load_control:
            wp.launch(
                press,
                1,
                [scene.q, scene.inv_mass, self.target, dt, self.force],
                device=scene.device,
            )


def run_load_test(sim, fixture, writer, hold_seconds, sample_stride=1):
    writer.writerow(
        [
            "time_s",
            "load_N",
            "deflection_mm",
            "left_y_N",
            "right_y_N",
            "max_speed_mm_s",
            "max_guide_gap_mm",
        ]
    )
    guides = fixture.guides.numpy()
    steps = round((1.0 + hold_seconds) / sim.dt)
    for step in range(1, steps + 1):
        time = step * sim.dt
        load = 0.1 * min(1.0, time)  # ramp for one second, then hold
        fixture.target.fill_(load)
        sim.step()
        if step % sample_stride and step != steps:
            continue
        q = sim.scene.q.numpy().astype(float)
        edges = np.diff(q, axis=0)
        lengths_sq = np.sum(edges**2, axis=1)
        gap = 0.0
        for guide in guides:
            u = np.clip(np.sum((guide - q[:-1]) * edges, axis=1) / lengths_sq, 0.0, 1.0)
            gap = max(
                gap, np.min(np.linalg.norm(q[:-1] + u[:, None] * edges - guide, axis=1))
            )
        reactions = fixture.reaction.numpy()[:, 1]
        speed = np.linalg.norm(sim.scene.qd.numpy(), axis=1).max()
        row = [
            time,
            load,
            -1000 * q[len(q) // 2, 1],
            *reactions,
            1000 * speed,
            1000 * gap,
        ]
        if not np.isfinite(row).all():
            raise RuntimeError(f"Nonfinite result at {time} s")
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ei-n-mm2", type=float)
    parser.add_argument("--radius-mm", type=float, default=0.4445)
    parser.add_argument("--edges", type=int, default=24)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--wire", choices=SLOPES, default="amplatz-extra-stiff")
    parser.add_argument(
        "--stiffness-scale",
        type=float,
        default=1.0,
        help="multiply the shaft bending rigidity in the numerical fixture",
    )
    parser.add_argument("--stroke-mm", type=float, default=5.0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument(
        "--damping-reference-dt",
        type=float,
        help="preserve the original 0.5 velocity retention over this interval (s)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pull-through", action="store_true")
    mode.add_argument(
        "--load-test", action="store_true", help="ramp to 0.10 N and hold"
    )
    parser.add_argument(
        "--hold-seconds", type=float, default=4.0, help="hold time for --load-test"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not (np.isfinite(args.radius_mm) and args.radius_mm > 0):
        parser.error("Radius must be finite and positive.")
    if args.ei_n_mm2 is not None and not (
        np.isfinite(args.ei_n_mm2) and args.ei_n_mm2 > 0
    ):
        parser.error("Bending rigidity must be finite and positive.")
    if not np.isfinite(args.stiffness_scale) or args.stiffness_scale <= 0:
        parser.error("Stiffness scale must be finite and positive.")
    if args.damping_reference_dt is not None and not args.damping_reference_dt > 0:
        parser.error("Damping reference timestep must be positive.")
    if not 0.3 <= args.stroke_mm <= 5 or args.sample_stride < 1:
        parser.error("Stroke must be between 0.3 and 5 mm and stride positive.")
    if not np.isfinite(args.hold_seconds) or args.hold_seconds <= 0:
        parser.error("Hold time must be finite and positive.")
    if (
        args.edges < 12
        or args.edges % 2
        or args.dt <= 0
        or args.iterations < 1
        or args.substeps < 1
    ):
        parser.error(
            "Use an even edge count >=12, positive dt and positive iterations."
        )
    length = 0.06 / args.edges
    scene = Scene(args.device, 1, (0.0, 0.0, 0.0))
    # Numerical mass for damped quasi-static loading, independent of subdivision.
    add_rod(
        scene,
        "wire",
        [-0.03, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        args.edges + 1,
        length,
        args.radius_mm / 1000,
        mass=0.006 / (args.edges + 1),
        lock_root=False,
    )
    rigidity = (
        1000 * SLOPES[args.wire] * 0.04**3 / 48
        if args.ei_n_mm2 is None
        else args.ei_n_mm2 * 1e-6
    )
    rigidity *= args.stiffness_scale
    # The local bending constraint is sin(angle/2), with stiffness young*bend*L.
    retention = (
        0.5 ** (args.dt / args.substeps / args.damping_reference_dt)
        if args.damping_reference_dt
        else 0.5
    )
    settings = dict(
        young=4 * rigidity / length**2,
        inertia="physical",
        angular_damping=1 - retention,
    )
    declare_rod_direct(scene, "wire", **settings)
    scene.allocate()
    rod = RodDirect(scene, body="wire", **settings)
    fixture = Fixture(scene, args.pull_through, args.load_test)
    particles = Particles(scene, damping=1.0 if args.pull_through else retention)
    fixture.stroke_limit = args.stroke_mm / 1000
    sim = Stepper.create(
        scene,
        [particles, rod, fixture],
        dt=args.dt,
        substeps=args.substeps,
        iterations=args.iterations,
    )
    with open(args.output, "w", newline="") as stream:
        writer = csv.writer(stream)
        if args.load_test:
            run_load_test(sim, fixture, writer, args.hold_seconds, args.sample_stride)
        elif args.pull_through:
            writer.writerow(
                ["travel_mm", "segment", "guide_error_mm", "axial_reaction_N"]
            )
            segments = []
            peak_error = peak_axial = 0.0
            for direction in (1.0, -1.0):
                velocity = np.zeros((args.edges + 1, 3), np.float32)
                velocity[:, 0] = direction * length / (8 * args.dt)
                scene.qd.assign(velocity)
                for _ in range(48):
                    sim.step()
                    q = scene.q.numpy()
                    i = int(fixture.segment.numpy()[0])
                    edge = q[i + 1] - q[i]
                    u = np.clip(-q[i] @ edge / (edge @ edge), 0.0, 1.0)
                    error = 1000 * np.linalg.norm(q[i] + u * edge)
                    axial = fixture.reaction.numpy()[0] @ (edge / np.linalg.norm(edge))
                    writer.writerow([1000 * (q[0, 0] + 0.03), i, error, axial])
                    segments.append(i)
                    peak_error = max(peak_error, error)
                    peak_axial = max(peak_axial, abs(axial))
            print(
                "Segment range, peak guide error (mm), peak axial reaction (N):",
                min(segments),
                max(segments),
                peak_error,
                peak_axial,
            )
            if (
                max(segments) - min(segments) < 5
                or np.max(np.abs(np.diff(segments))) > 1
                or peak_error > 1e-4
                or peak_axial > 1e-4
            ):
                raise RuntimeError("RodDirect guide pull-through failed")
        else:
            writer.writerow(
                ["displacement_mm", "force_N", "left_y_N", "right_y_N", "guide_gap_mm"]
            )
            rows = []
            last_step = int(np.ceil(3 * args.stroke_mm / args.dt))
            for step in range(last_step + 1):
                displacement = min(args.stroke_mm, step * args.dt / 3.0)
                if args.substeps == 1:
                    fixture.target.fill_(-displacement / 1000)
                else:
                    fixture.frame_start.fill_(-(step - 1) * args.dt / 3000)
                sim.step()
                force = float(fixture.force.numpy()[0])
                if not np.isfinite(force):
                    raise RuntimeError("Nonfinite nose reaction")
                if step % args.sample_stride == 0 or step == last_step:
                    q = scene.q.numpy().astype(float)
                    edges = np.diff(q, axis=0)
                    gaps = []
                    for guide in fixture.guides.numpy():
                        u = np.clip(
                            np.sum((guide - q[:-1]) * edges, axis=1)
                            / np.sum(edges**2, axis=1),
                            0,
                            1,
                        )
                        gaps.append(
                            np.min(
                                np.linalg.norm(
                                    q[:-1] + u[:, None] * edges - guide, axis=1
                                )
                            )
                        )
                    writer.writerow(
                        [
                            displacement,
                            force,
                            *fixture.reaction.numpy()[:, 1],
                            1000 * max(gaps),
                        ]
                    )
                if displacement <= 0.3:
                    rows.append((displacement, force))
            print(
                "Initial slope (N/mm), final force (N):",
                np.polyfit(*np.array(rows).T, 1)[0],
                force,
            )


if __name__ == "__main__":
    main()
