# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the device-side coupled fixture with the PR #22 RodDirect model."""

import argparse
import csv
import math

import kernels
import numpy as np
import warp as wp
from guidewire_solver_bench import Fixture
from harrison_inputs import SLOPES
from i4h_physics import Scene, Stepper
from i4h_physics.loaders import add_rod
from i4h_physics.rod_storage import declare_rod_direct
from i4h_physics.systems.mechanics.particles import Particles
from i4h_physics.systems.rods.rod_direct import RodDirect
from i4h_physics.systems.rods.rod_direct_kernels import (
    apply_corrections,
    edge_constraints,
    world_inv_inertia,
)


class RodTangent(RodDirect):
    """Populate the internal rows without moving the rod."""

    def iteration(self, scene, dt, k, iterations):
        wp.launch(
            edge_constraints,
            dim=(1, self.Ed),
            inputs=[
                scene.field("q", self.body), self.frame_pred, scene.field("inv_mass", self.body),
                scene.field("flags", self.body), self.rest_len, self.rest_darboux, self.preferred_rest,
                int(self.regional), self.bend_offset, self.variant, self.compliance, self.lambda_sum,
                int(k == 0), self.P, self.Ed,
            ],
            outputs=[self.edge_w, self.jacobian_rot, self.rhs],
            device=scene.device,
        )
        wp.launch(
            world_inv_inertia,
            dim=(1, self.P),
            inputs=[self.frame_pred, self.inv_mass_q, self.variant, self.inertia_local, self.P],
            outputs=[self.inv_inertia],
            device=scene.device,
        )


class DeviceFixture(Fixture):
    """Single-rod displacement fixture, with device-resident solve buffers."""

    def __init__(self, scene, rod):
        super().__init__(scene, False)
        self.rod = rod
        device = scene.device
        self.rows = wp.empty((5, rod.P), dtype=wp.vec3d, device=device)
        self.normals = wp.empty(4, dtype=wp.vec3d, device=device)
        self.residual = wp.empty(5, dtype=wp.float64, device=device)
        self.diag = wp.empty(rod.Ed, dtype=kernels.Matrix6, device=device)
        self.upper = wp.empty_like(self.diag)
        self.rhs = wp.empty_like(self.diag)
        self.transfer = wp.empty_like(self.diag)
        self.answer = wp.empty_like(self.diag)
        self.fixture_impulse = wp.empty(1, dtype=kernels.Vector5, device=device)
        self.rod_impulse = wp.empty(rod.Ed * 6, dtype=wp.float64, device=device)
        self.failure = wp.zeros(1, dtype=int, device=device)
        self.total_force = wp.empty(1, dtype=wp.float64, device=device)
        self.total_reaction = wp.empty(2, dtype=wp.vec3d, device=device)

    def iteration(self, scene, dt, k, iterations):
        rod = self.rod
        device = scene.device
        first = int(k == 0)
        wp.launch(
            kernels.build_fixture_rows, dim=1,
            inputs=[scene.q, self.guides, self.target],
            outputs=[self.rows, self.residual, self.normals], device=device,
        )
        wp.launch(
            kernels.assemble_blocks, dim=rod.Ed,
            inputs=[rod.jacobian_rot, rod.inv_inertia, scene.inv_mass, rod.compliance, rod.rhs, self.rows],
            outputs=[self.diag, self.upper, self.rhs], device=device,
        )
        wp.launch(
            kernels.thomas_fixture, dim=1, inputs=[self.diag, self.upper, self.rhs],
            outputs=[self.transfer, self.answer, self.failure], device=device,
        )
        wp.launch(
            kernels.solve_fixture, dim=1,
            inputs=[scene.inv_mass, self.rows, self.residual, self.rhs, self.answer],
            outputs=[self.fixture_impulse, self.failure], device=device,
        )
        wp.launch(
            kernels.recover_rod_impulses, dim=rod.Ed,
            inputs=[self.answer, self.fixture_impulse, self.failure, first],
            outputs=[self.rod_impulse, rod.lambda_sum], device=device,
        )
        wp.launch(
            kernels.compute_fixture_corrections, dim=rod.P,
            inputs=[
                rod.jacobian_rot, rod.inv_inertia, scene.inv_mass, self.rows,
                self.rod_impulse, self.fixture_impulse, self.failure,
            ],
            outputs=[rod.pos_corr, rod.rot_corr], device=device,
        )
        wp.launch(
            apply_corrections, dim=(1, rod.P), inputs=[scene.field("flags", rod.body), rod.P],
            outputs=[scene.field("q", rod.body), rod.frame_pred, rod.pos_corr, rod.rot_corr], device=device,
        )
        wp.launch(
            kernels.record_reactions, dim=1,
            inputs=[self.fixture_impulse, self.normals, self.failure, dt, first],
            outputs=[self.total_force, self.total_reaction, self.force, self.reaction], device=device,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--wire", default="lunderquist-extra-stiff", choices=SLOPES)
    parser.add_argument("--ei-n-mm2", type=float)
    parser.add_argument("--radius-mm", type=float, default=.4445)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--stroke-mm", type=float, default=5.)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not (args.iterations >= 1 and .3 <= args.stroke_mm <= 5.0
            and math.isfinite(args.radius_mm) and args.radius_mm > 0.0
            and (args.ei_n_mm2 is None or math.isfinite(args.ei_n_mm2) and args.ei_n_mm2 > 0.0)):
        parser.error("Use positive iterations, radius and EI, and a stroke in [0.3, 5] mm.")

    edges = 24
    length = .06 / edges
    dt = .016
    scene = Scene(args.device, 1, (0., 0., 0.))
    add_rod(scene, "wire", [-.03, 0., 0.], [1., 0., 0.], edges + 1, length,
            args.radius_mm / 1000, mass=.006 / (edges + 1), lock_root=False)
    rigidity = args.ei_n_mm2 * 1e-6 if args.ei_n_mm2 is not None else 1000 * SLOPES[args.wire] * .04**3 / 48
    material = dict(young=4 * rigidity / length**2, inertia="physical", angular_damping=.5)
    declare_rod_direct(scene, "wire", **material)
    scene.allocate()
    rod = RodTangent(scene, body="wire", **material)
    fixture = DeviceFixture(scene, rod)
    fixture.stroke_limit = args.stroke_mm / 1000
    sim = Stepper.create(
        scene, [Particles(scene, damping=.5), rod, fixture],
        dt=dt, substeps=16, iterations=args.iterations, cuda_graph=not args.no_cuda_graph,
    )

    guides = fixture.guides.numpy().astype(np.float64)
    initial = []
    with open(args.output, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["displacement_mm", "force_N", "left_y_N", "right_y_N", "guide_gap_mm"])
        for step in range(math.ceil(3 * args.stroke_mm / dt) + 1):
            displacement = min(args.stroke_mm, step * dt / 3)
            fixture.frame_start.fill_(-(step - 1) * dt / 3000)
            sim.step()
            force = float(fixture.force.numpy()[0])
            if not math.isfinite(force):
                code = int(fixture.failure.numpy()[0])
                raise ArithmeticError(f"Coupled fixture failed at {displacement:g} mm (factorization status {code})")
            q = scene.q.numpy().astype(np.float64)
            edges_vec = np.diff(q, axis=0)
            offset = guides[:, None, :] - q[None, :-1, :]
            fraction = np.clip(np.sum(offset * edges_vec, axis=2) / np.sum(edges_vec**2, axis=1), 0., 1.)
            gaps = np.linalg.norm(offset - fraction[..., None] * edges_vec, axis=2)
            guide_gap = 1000 * np.max(np.min(gaps, axis=1))
            writer.writerow([displacement, force, *fixture.reaction.numpy()[:, 1], guide_gap])
            if displacement <= .3:
                initial.append((displacement, force))
    print("Initial slope (N/mm), final force (N):", np.polyfit(*np.array(initial).T, 1)[0], force)


if __name__ == "__main__":
    main()
