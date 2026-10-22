# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct (block-Thomas) XPBD Cosserat rod: the catheter backend omniendo's xcath scenes run.

Ported from omniendo/xpbd_rod/solver_xpbd_rod.py:1577-1682 (``_step_batched``) and :1684-1820
(``_project_direct_batched``) with the kernels in ``rod_direct_kernels.py``. The rod's node
positions are a ``Body`` in ``scene.q``: ``scene.q`` plays omniendo's ``predicted_positions`` and
``scene.q_prev`` its ``positions``, and ``particles`` does the position predict and finalize (its
``damping`` factor is omniendo's ``1 - linear_damping``), so ``_warp_predict_positions_batched``
and ``_warp_integrate_positions_batched`` are not ported. This system owns the rotational state
(``frame``, ``frame_pred``, ``omega``, one node frame per node) and the solver scratch. One
constraint projection per ``iteration``, five launches: edge constraints + Jacobian + right-hand
side, block assembly, Thomas sweep (one thread per env: 72% of the aorta step, a serial
191-edge recurrence), corrections, self-clearing apply. omniendo ran exactly one projection per
substep, so ``substeps: 8, iterations: 1`` reproduces its numerics. Compliance depends on the live
stiffness and the substep dt, so it is recomputed on the first substep of every step, inside the
graph (measured 0.004 ms, not worth moving). Inactive nodes (``scene.flags``) hold their position
and frame; edges touching one drop out of the solve but stay in the sweep. One mesh variant only.

Products read by ``catheter_control``: ``frame`` / ``frame_pred`` (the root frame of env ``e`` is
index ``e * P``), ``frame_count``, ``bend_count`` (= edges) and the live input
``bend_offset (E, edges) vec3`` that the constraint kernel adds to the rest Darboux vector of each
edge (the steerable tip; zero by default, written by ``catheter_control`` every ``step_start``).

Dropped from omniendo: ``prev_orientations`` (written, never read), ``torques``, the
``_delta_lambda_max`` / ``_correction_max`` diagnostics, floor collision, the per-edge stiffness
array (live ``(E,)`` ``bend`` and ``twist`` instead), the split/Jacobi/banded/tiled solvers, and
the constant arrays ``jacobian_pos``, ``inv_inertia`` (identity), ``constraint_values``, ``d_prime``
(see ``rod_direct_kernels``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
from i4h_physics.scene import restore_variant_rows
from i4h_physics.systems.base import System
from i4h_physics.systems.rods.rod_direct_kernels import (
    apply_corrections,
    assemble_jmjt,
    compute_corrections,
    edge_constraints,
    integrate_rot,
    predict_rot,
    prepare_compliance,
    thomas,
    world_inv_inertia,
)


@dataclass(eq=False)
class RodDirect(System):
    body: str  # rod asset id in YAML; the Body instance after construction
    young: float = 1.0e6  # live, per env: Young's modulus, scales the bend compliance 1 / (E k L dt^2)
    torsion: float = 1.0e6  # live, per env: torsion modulus, scales the twist compliance
    bend: float = 1.0  # live, per env: bend stiffness factor (both bending directions)
    twist: float = 1.0  # live, per env: twist stiffness factor
    angular_damping: float = 0.01  # live, per env: omega *= (1 - d) per substep (omniendo angular_damping)
    inertia: str = "unit"  # structural: unit (omniendo: identity rotational inertia) or physical (each node's cylinder)

    structural = System.structural + ("inertia",)

    def __post_init__(self):
        scene = self.scene
        self.body = b = scene.body(self.body)
        if b.rod is None:
            raise ValueError(f"rod_direct: body {b.name!r} has no rod record")
        m = b.rod
        if m.frames.shape[0] != 1:
            raise ValueError("rod_direct: rod systems support a single mesh variant")
        E, dev = scene.envs, scene.device
        P = b.count
        Ed = P - 1
        self.P, self.Ed = P, Ed
        self.frame_count = P  # node frames
        self.bend_count = Ed  # one Darboux constraint per edge
        n, ne = E * P, E * Ed
        scene.check_memory({
            "variants": m.frames.nbytes + m.rest_len.nbytes + m.rest_darboux.nbytes + m.inv_mass_q.nbytes,
            "frames": n * (16 * 2 + 12) + ne * 12,
            "dofs": ne * 6 * 4 * 4,
            "blocks": ne * 36 * 4 * 4,
            "corrections": n * 12 * 2 + ne * 12,
        })
        self.variant = b.variant  # (E,) int32, shared by the body's systems
        self.frames0 = wp.array(m.frames, dtype=wp.quat, device=dev)  # (V, P)
        material = b.rod_material
        self.rest_len = material.rest_len if material is not None and material.backend == 'direct' else wp.array(m.rest_len, dtype=wp.float32, device=dev)
        self.rest_darboux = material.rest_darboux if material is not None and material.backend == 'direct' else wp.array(m.rest_darboux, dtype=wp.vec3, device=dev)
        self.inv_mass_q = wp.array(m.inv_mass_q, dtype=wp.float32, device=dev)  # (V, P); 0 at a locked root
        self.frame = wp.empty(n, dtype=wp.quat, device=dev)  # body-local: index env * P + node
        self.frame_pred = wp.empty(n, dtype=wp.quat, device=dev)
        self.omega = wp.zeros(n, dtype=wp.vec3, device=dev)
        from i4h_physics.regions import RodControls
        material = b.rod_material
        if b.regions is not None and material is None:
            raise ValueError("rod_direct: regional body requires rod_direct_material before the solver")
        if material is not None and material.backend != "direct":
            raise ValueError("rod_direct: mismatched material backend")
        if b.rod_controls is None:
            b.rod_controls = RodControls(scene, Ed, Ed)
        self.bend_offset = b.rod_controls.bend_offset
        self.regional = material is not None and material.active
        self.preferred_rest = material.preferred_rest if self.regional else self.rest_darboux
        self._live("young", "torsion", "bend", "twist", "angular_damping")
        # Solver scratch (omniendo's _BatchedRodWorkspace, solver_xpbd_rod.py:287-316, minus the constant arrays).
        # Everything is rewritten before it is read within an iteration; lambda_sum is overwritten on each
        # substep's first iteration; the correction accumulators are cleared by the apply kernel. No masked reset.
        def f32(k):
            return wp.zeros(k, dtype=wp.float32, device=dev)
        self.compliance, self.lambda_sum, self.rhs, self.delta_lambda = f32(ne * 6), f32(ne * 6), f32(ne * 6), f32(ne * 6)
        self.jacobian_rot, self.diag_blocks, self.offdiag_blocks, self.c_blocks = f32(ne * 36), f32(ne * 36), f32(ne * 36), f32(ne * 36)
        self.edge_w = wp.zeros(ne, dtype=wp.vec3, device=dev)  # (inv_m0, inv_m1, live) per edge
        self.pos_corr = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.rot_corr = wp.zeros(n, dtype=wp.vec3, device=dev)
        if self.inertia not in ("unit", "physical"):
            raise ValueError(f"rod_direct.inertia must be 'unit' or 'physical', got {self.inertia!r}")
        # (E * P,) local inverse inertia from each env's own masses and radii, refreshed by prepare() (physical only)
        self.inertia_local = wp.array(cylinder_inv_inertia(scene, b, m.rest_len[0]).reshape(-1, 3), dtype=wp.vec3,
                                      device=dev)
        self.inv_inertia = wp.zeros(n if self.inertia == "physical" else 1, dtype=wp.mat33, device=dev)  # world, per node
        self.freeze("inertia")
        self.reset(scene, wp.ones(E, dtype=wp.bool, device=dev))

    def prepare(self, scene):
        super().prepare(scene)
        if self.inertia == "physical":  # per-env masses may have changed since construction (in place: graphs keep it)
            self.inertia_local.assign(cylinder_inv_inertia(scene, self.body, self.rest_len.numpy()[0]).reshape(-1, 3))
        material = self.body.rod_material
        self.regional = material is not None and material.active
        self.preferred_rest = material.preferred_rest if self.regional else self.rest_darboux


    def substep_start(self, scene, dt, s, substeps):
        E, dev, P, Ed = scene.envs, scene.device, self.P, self.Ed
        wp.launch(predict_rot, dim=(E, P),
                  inputs=[self.frame, self.inv_mass_q, self.variant, scene.field("flags", self.body), self.angular_damping, P, dt],
                  outputs=[self.omega, self.frame_pred], device=dev)
        if s == 0 and self.regional:
            from i4h_physics.systems.rods.rod_direct_kernels import prepare_regional_compliance
            material = self.body.rod_material
            wp.launch(prepare_regional_compliance, dim=(E, Ed),
                      inputs=[self.rest_len, self.young, self.torsion, self.bend, self.twist,
                              material.young_multiplier, material.torsion_multiplier,
                              material.bend_multiplier, material.twist_multiplier, Ed, dt],
                      outputs=[self.compliance], device=dev)
        elif s == 0:  # live stiffness: recompute inside the graph, once per step
            wp.launch(prepare_compliance, dim=(E, Ed),
                      inputs=[self.rest_len, self.variant, self.young, self.torsion, self.bend, self.twist, Ed, dt],
                      outputs=[self.compliance], device=dev)

    def iteration(self, scene, dt, k, iterations):
        b, E, dev, P, Ed = self.body, scene.envs, scene.device, self.P, self.Ed
        first = int(k == 0)  # the substep boundary of the XPBD multipliers
        wp.launch(edge_constraints, dim=(E, Ed),
                  inputs=[scene.field("q", b), self.frame_pred, scene.field("inv_mass", b), scene.field("flags", b), self.rest_len,
                          self.rest_darboux, self.preferred_rest, int(self.regional), self.bend_offset, self.variant, self.compliance, self.lambda_sum, first, P, Ed],
                  outputs=[self.edge_w, self.jacobian_rot, self.rhs], device=dev)
        physical = int(self.inertia == "physical")
        if physical:
            wp.launch(world_inv_inertia, dim=(E, P), inputs=[self.frame_pred, self.inv_mass_q, self.variant, self.inertia_local, P],
                      outputs=[self.inv_inertia], device=dev)
        wp.launch(assemble_jmjt, dim=(E, Ed),
                  inputs=[self.edge_w, self.jacobian_rot, self.compliance, self.inv_mass_q, self.variant, Ed, physical,
                          self.inv_inertia, P],
                  outputs=[self.diag_blocks, self.offdiag_blocks], device=dev)
        wp.launch(thomas, dim=E,  # one thread per env: a serial sweep over the rod's edges
                  inputs=[self.diag_blocks, self.offdiag_blocks, self.rhs, Ed],
                  outputs=[self.c_blocks, self.delta_lambda], device=dev)
        wp.launch(compute_corrections, dim=(E, Ed),
                  inputs=[self.edge_w, self.inv_mass_q, self.variant, self.jacobian_rot, self.delta_lambda, first, P, Ed,
                          physical, self.inv_inertia],
                  outputs=[self.lambda_sum, self.pos_corr, self.rot_corr], device=dev)
        wp.launch(apply_corrections, dim=(E, P),
                  inputs=[scene.field("flags", b), P],
                  outputs=[scene.field("q", b), self.frame_pred, self.pos_corr, self.rot_corr], device=dev)

    def substep_end(self, scene, dt):
        E, dev, P = scene.envs, scene.device, self.P
        wp.launch(integrate_rot, dim=(E, P),
                  inputs=[self.frame_pred, self.inv_mass_q, self.variant, scene.field("flags", self.body), P, dt],
                  outputs=[self.frame, self.omega], device=dev)

    def reset(self, scene, mask):
        """Selected envs: frames back to the variant's rest frames, angular velocity and tip offset to zero."""
        E, dev, P = scene.envs, scene.device, self.P
        for dst in (self.frame, self.frame_pred):
            wp.launch(restore_variant_rows, dim=(E, P), inputs=[mask, self.variant, self.frames0],
                      outputs=[dst.reshape((E, P))], device=dev)
        self.clear(mask, (self.omega, wp.vec3(0.0, 0.0, 0.0)))
        self.clear(mask, (self.bend_offset, wp.vec3(0.0, 0.0, 0.0)))


def cylinder_inv_inertia(scene, body, rest_len) -> np.ndarray:
    """``(E, P, 3)`` float32 local inverse inertia per env and node for ``inertia: physical``: a cylinder of the node's
    mass, radius and segment length, ``12 / (m (3 r² + L²))`` about the transverse axes and ``2 / (m r²)`` about the
    tangent (z), as the workflow's catheter sets it (newton_catheter_physics._apply_physical_rotational_inertia). Each
    env's own rows: one geometry variant may carry different masses or radii per env. Zero for a node without mass
    (the locked root)."""
    inv_mass = scene.field("inv_mass", body).numpy().astype(np.float64)  # (E, P)
    radius = scene.field("radius", body).numpy().astype(np.float64)
    seg = np.asarray(rest_len, np.float64)
    seg = np.broadcast_to(np.concatenate([seg, seg[-1:]]), inv_mass.shape)  # node e: edge e's length (last: the last)
    out = np.zeros(inv_mass.shape + (3,), np.float32)
    free = inv_mass > 0.0
    m = 1.0 / np.where(free, inv_mass, 1.0)
    out[free, 0] = out[free, 1] = (12.0 / (m * (3.0 * radius**2 + seg**2)))[free]
    out[free, 2] = (2.0 / (m * radius**2))[free]
    return out
