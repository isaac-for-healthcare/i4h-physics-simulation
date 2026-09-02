# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""XPBD solver for Cosserat elastic rods with direct block-tridiagonal solve."""

from __future__ import annotations

import warnings

import numpy as np
import warp as wp

from newton import Contacts, Control, Model, ModelBuilder, State
from newton.solvers import SolverBase

try:  # ``typing.override`` is 3.12+; this package supports 3.10.
    from typing import override
except ImportError:  # pragma: no cover - exercised only on Python < 3.12

    def override(method):
        return method


from .constants import (
    BAND_LDAB,
    BLOCK_DIM,
    DIRECT_SOLVE_BACKENDS,
    DIRECT_SOLVE_BANDED_CHOLESKY,
    DIRECT_SOLVE_BLOCK_JACOBI,
    DIRECT_SOLVE_BLOCK_THOMAS,
    DIRECT_SOLVE_SPLIT_THOMAS,
    TILE,
)
from .kernels_assembly import (
    _warp_assemble_darboux_blocks,
    _warp_assemble_jmjt_banded,
    _warp_assemble_jmjt_blocks,
    _warp_assemble_jmjt_blocks_batched,
    _warp_assemble_jmjt_dense,
    _warp_assemble_stretch_blocks,
    _warp_compute_inv_inertia_world_batched,
    _warp_pad_diagonal,
)
from .kernels_collision import (
    _warp_apply_accumulated_corrections,
    _warp_apply_floor_collisions,
    _warp_compute_corrections_parallel,
    _warp_compute_corrections_parallel_batched,
    _warp_compute_inv_inertia_world,
    _warp_merge_delta_lambda,
    _warp_set_root_orientation,
    _warp_zero_2d,
    _warp_zero_float,
    _warp_zero_vec3,
)
from .kernels_constraints import (
    _warp_build_rhs,
    _warp_build_rhs_darboux,
    _warp_build_rhs_stretch,
    _warp_compute_jacobians_batched,
    _warp_compute_jacobians_direct,
    _warp_prepare_compliance,
    _warp_prepare_compliance_batched,
    _warp_update_constraints_batched_v2,
    _warp_update_constraints_direct,
)
from .kernels_deformable import (
    _warp_apply_deformable_corrections,
    _warp_gather_deformable,
    _warp_predict_deformable,
    _warp_project_dihedral_gs,
    _warp_project_dihedral_jacobi,
    _warp_project_distance_gs,
    _warp_project_distance_jacobi,
    _warp_project_volume_gs,
    _warp_project_volume_jacobi,
    _warp_scatter_deformable,
)
from .kernels_integration import (
    _warp_integrate_positions,
    _warp_integrate_positions_batched,
    _warp_integrate_rotations,
    _warp_integrate_rotations_batched,
    _warp_predict_positions,
    _warp_predict_positions_batched,
    _warp_predict_rotations,
    _warp_predict_rotations_batched,
)
from .kernels_solvers import (
    _warp_block_thomas_solve,
    _warp_block_thomas_solve_3x3,
    _warp_block_thomas_solve_batched,
    _warp_cholesky_solve_tile,
    _warp_solve_blocks_jacobi,
    _warp_spbsv_u11_1rhs,
)


class _RodWorkspace:
    """GPU workspace arrays for a single rod within the solver."""

    def __init__(self, num_points: int, num_edges: int, device: wp.Device):
        n_dofs = num_edges * 6
        alloc_dofs = max(1, n_dofs)
        alloc_edges = max(1, num_edges)
        alloc_points = max(1, num_points)

        self.num_points = num_points
        self.num_edges = num_edges
        self.n_dofs = n_dofs
        self.device = device

        # Per-particle state arrays
        self.positions_wp = wp.zeros(alloc_points, dtype=wp.vec3, device=device)
        self.predicted_positions_wp = wp.zeros(
            alloc_points, dtype=wp.vec3, device=device
        )
        self.velocities_wp = wp.zeros(alloc_points, dtype=wp.vec3, device=device)
        self.forces_wp = wp.zeros(alloc_points, dtype=wp.vec3, device=device)

        self.orientations_wp = wp.zeros(alloc_points, dtype=wp.quat, device=device)
        self.predicted_orientations_wp = wp.zeros(
            alloc_points, dtype=wp.quat, device=device
        )
        self.prev_orientations_wp = wp.zeros(alloc_points, dtype=wp.quat, device=device)
        self.angular_velocities_wp = wp.zeros(
            alloc_points, dtype=wp.vec3, device=device
        )
        self.torques_wp = wp.zeros(alloc_points, dtype=wp.vec3, device=device)

        self.inv_masses_wp = wp.zeros(alloc_points, dtype=wp.float32, device=device)
        self.quat_inv_masses_wp = wp.zeros(
            alloc_points, dtype=wp.float32, device=device
        )

        # Per-edge arrays
        self.rest_lengths_wp = wp.zeros(alloc_edges, dtype=wp.float32, device=device)
        self.rest_darboux_wp = wp.zeros(alloc_edges, dtype=wp.vec3, device=device)
        self.bend_stiffness_wp = wp.zeros(alloc_edges, dtype=wp.vec3, device=device)

        # Constraint workspace
        self.constraint_values_wp = wp.zeros(
            alloc_dofs, dtype=wp.float32, device=device
        )
        self.compliance_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=device)
        self.lambda_sum_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=device)
        self.jacobian_pos_wp = wp.zeros(
            alloc_edges * 36, dtype=wp.float32, device=device
        )
        self.jacobian_rot_wp = wp.zeros(
            alloc_edges * 36, dtype=wp.float32, device=device
        )

        # Solver workspace
        self.rhs_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=device)
        self.delta_lambda_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=device)
        self.diag_blocks_wp = wp.zeros(
            alloc_edges * 36, dtype=wp.float32, device=device
        )
        self.offdiag_blocks_wp = wp.zeros(
            alloc_edges * 36, dtype=wp.float32, device=device
        )
        self.c_blocks_wp = wp.zeros(alloc_edges * 36, dtype=wp.float32, device=device)
        self.d_prime_wp = wp.zeros(alloc_edges * 6, dtype=wp.float32, device=device)

        # Dense/tiled solver workspace
        self.A_wp = wp.zeros((TILE, TILE), dtype=wp.float32, device=device)
        self.rhs_tile_wp = wp.zeros(TILE, dtype=wp.float32, device=device)
        self.delta_lambda_tile_wp = wp.zeros(TILE, dtype=wp.float32, device=device)

        # Banded solver workspace
        self.ab_wp = wp.zeros((BAND_LDAB, alloc_dofs), dtype=wp.float32, device=device)

        # Inverse inertia
        self.inv_inertia_wp = wp.zeros(
            alloc_points * 9, dtype=wp.float32, device=device
        )
        self.inv_inertia_local_diag = wp.vec3(1.0, 1.0, 1.0)

        # Parallel correction workspace
        self.pos_corrections_wp = wp.zeros(alloc_points, dtype=wp.vec3, device=device)
        self.rot_corrections_wp = wp.zeros(alloc_points, dtype=wp.vec3, device=device)

        # Diagnostics
        self._constraint_max_wp = wp.zeros(1, dtype=wp.float32, device=device)
        self._delta_lambda_max_wp = wp.zeros(1, dtype=wp.float32, device=device)
        self._correction_max_wp = wp.zeros(1, dtype=wp.float32, device=device)

        # Split Thomas solver arrays (lazily allocated)
        self._split_stretch_diag_wp = None

        # Material properties
        self.young_modulus = 1.0e6
        self.torsion_modulus = 1.0e6
        self.gravity = wp.vec3(0.0, 0.0, -9.81)


class _BatchedRodWorkspace:
    """Concatenated GPU workspace for all rods, enabling batched kernel launches."""

    def __init__(self, rods: list[_RodWorkspace], device: wp.Device):
        n_rods = len(rods)
        self.n_rods = n_rods

        # Build offset arrays on CPU
        rod_offsets_cpu = [0]
        edge_offsets_cpu = [0]
        for ws in rods:
            rod_offsets_cpu.append(rod_offsets_cpu[-1] + ws.num_points)
            edge_offsets_cpu.append(edge_offsets_cpu[-1] + ws.num_edges)

        self.rod_offsets_cpu = rod_offsets_cpu
        self.edge_offsets_cpu = edge_offsets_cpu
        self.total_particles = rod_offsets_cpu[-1]
        self.total_edges = edge_offsets_cpu[-1]
        self.total_dofs = self.total_edges * 6

        # Upload index arrays
        self.rod_offsets = wp.array(rod_offsets_cpu, dtype=wp.int32, device=device)
        self.edge_offsets = wp.array(edge_offsets_cpu, dtype=wp.int32, device=device)

        # Build particle_rod_id and edge_rod_id
        particle_rod_id_cpu = []
        edge_rod_id_cpu = []
        for i, ws in enumerate(rods):
            particle_rod_id_cpu.extend([i] * ws.num_points)
            edge_rod_id_cpu.extend([i] * ws.num_edges)
        self.particle_rod_id = wp.array(
            particle_rod_id_cpu, dtype=wp.int32, device=device
        )
        self.edge_rod_id = wp.array(edge_rod_id_cpu, dtype=wp.int32, device=device)

        # Per-rod property arrays
        gravity_cpu = [list(ws.gravity) for ws in rods]
        self.gravity = wp.array(gravity_cpu, dtype=wp.vec3, device=device)
        self.young_modulus = wp.array(
            [ws.young_modulus for ws in rods], dtype=wp.float32, device=device
        )
        self.torsion_modulus = wp.array(
            [ws.torsion_modulus for ws in rods], dtype=wp.float32, device=device
        )
        self.inv_inertia_local_diag = wp.array(
            [list(ws.inv_inertia_local_diag) for ws in rods],
            dtype=wp.vec3,
            device=device,
        )

        tp = self.total_particles
        te = self.total_edges
        td = self.total_dofs

        # Concatenated per-particle arrays
        self.positions = wp.zeros(tp, dtype=wp.vec3, device=device)
        self.predicted_positions = wp.zeros(tp, dtype=wp.vec3, device=device)
        self.velocities = wp.zeros(tp, dtype=wp.vec3, device=device)
        self.forces = wp.zeros(tp, dtype=wp.vec3, device=device)
        self.orientations = wp.zeros(tp, dtype=wp.quat, device=device)
        self.predicted_orientations = wp.zeros(tp, dtype=wp.quat, device=device)
        self.prev_orientations = wp.zeros(tp, dtype=wp.quat, device=device)
        self.angular_velocities = wp.zeros(tp, dtype=wp.vec3, device=device)
        self.torques = wp.zeros(tp, dtype=wp.vec3, device=device)
        self.inv_masses = wp.zeros(tp, dtype=wp.float32, device=device)
        self.quat_inv_masses = wp.zeros(tp, dtype=wp.float32, device=device)
        self.inv_inertia = wp.zeros(tp * 9, dtype=wp.float32, device=device)
        self.pos_corrections = wp.zeros(tp, dtype=wp.vec3, device=device)
        self.rot_corrections = wp.zeros(tp, dtype=wp.vec3, device=device)

        # Concatenated per-edge arrays
        self.rest_lengths = wp.zeros(te, dtype=wp.float32, device=device)
        self.rest_darboux = wp.zeros(te, dtype=wp.vec3, device=device)
        self.bend_stiffness = wp.zeros(te, dtype=wp.vec3, device=device)
        self.constraint_values = wp.zeros(td, dtype=wp.float32, device=device)
        self.compliance = wp.zeros(td, dtype=wp.float32, device=device)
        self.lambda_sum = wp.zeros(td, dtype=wp.float32, device=device)
        self.jacobian_pos = wp.zeros(te * 36, dtype=wp.float32, device=device)
        self.jacobian_rot = wp.zeros(te * 36, dtype=wp.float32, device=device)

        # Solver workspace
        self.rhs = wp.zeros(td, dtype=wp.float32, device=device)
        self.delta_lambda = wp.zeros(td, dtype=wp.float32, device=device)
        self.diag_blocks = wp.zeros(te * 36, dtype=wp.float32, device=device)
        self.offdiag_blocks = wp.zeros(te * 36, dtype=wp.float32, device=device)
        self.c_blocks = wp.zeros(te * 36, dtype=wp.float32, device=device)
        self.d_prime = wp.zeros(te * 6, dtype=wp.float32, device=device)

        # Diagnostics
        self._delta_lambda_max = wp.zeros(1, dtype=wp.float32, device=device)
        self._correction_max = wp.zeros(1, dtype=wp.float32, device=device)

        # Copy data from individual workspaces into concatenated arrays
        for i, ws in enumerate(rods):
            po = rod_offsets_cpu[i]
            eo = edge_offsets_cpu[i]
            np_ = ws.num_points
            ne = ws.num_edges

            wp.copy(
                dest=self.positions,
                src=ws.positions_wp,
                dest_offset=po,
                src_offset=0,
                count=np_,
            )
            wp.copy(
                dest=self.predicted_positions,
                src=ws.predicted_positions_wp,
                dest_offset=po,
                src_offset=0,
                count=np_,
            )
            wp.copy(
                dest=self.velocities,
                src=ws.velocities_wp,
                dest_offset=po,
                src_offset=0,
                count=np_,
            )
            wp.copy(
                dest=self.forces,
                src=ws.forces_wp,
                dest_offset=po,
                src_offset=0,
                count=np_,
            )
            wp.copy(
                dest=self.orientations,
                src=ws.orientations_wp,
                dest_offset=po,
                src_offset=0,
                count=np_,
            )
            wp.copy(
                dest=self.predicted_orientations,
                src=ws.predicted_orientations_wp,
                dest_offset=po,
                src_offset=0,
                count=np_,
            )
            wp.copy(
                dest=self.prev_orientations,
                src=ws.prev_orientations_wp,
                dest_offset=po,
                src_offset=0,
                count=np_,
            )
            wp.copy(
                dest=self.angular_velocities,
                src=ws.angular_velocities_wp,
                dest_offset=po,
                src_offset=0,
                count=np_,
            )
            wp.copy(
                dest=self.torques,
                src=ws.torques_wp,
                dest_offset=po,
                src_offset=0,
                count=np_,
            )
            wp.copy(
                dest=self.inv_masses,
                src=ws.inv_masses_wp,
                dest_offset=po,
                src_offset=0,
                count=np_,
            )
            wp.copy(
                dest=self.quat_inv_masses,
                src=ws.quat_inv_masses_wp,
                dest_offset=po,
                src_offset=0,
                count=np_,
            )

            wp.copy(
                dest=self.rest_lengths,
                src=ws.rest_lengths_wp,
                dest_offset=eo,
                src_offset=0,
                count=ne,
            )
            wp.copy(
                dest=self.rest_darboux,
                src=ws.rest_darboux_wp,
                dest_offset=eo,
                src_offset=0,
                count=ne,
            )
            wp.copy(
                dest=self.bend_stiffness,
                src=ws.bend_stiffness_wp,
                dest_offset=eo,
                src_offset=0,
                count=ne,
            )


def _greedy_color(constraints: list[tuple[int, ...]]) -> tuple[list[int], list[int]]:
    """Return a color-grouped permutation and cumulative color offsets."""
    occupied: list[set[int]] = []
    groups: list[list[int]] = []
    for ci, vertices in enumerate(constraints):
        for color, used in enumerate(occupied):
            if used.isdisjoint(vertices):
                used.update(vertices)
                groups[color].append(ci)
                break
        else:
            occupied.append(set(vertices))
            groups.append([ci])
    permutation = [ci for group in groups for ci in group]
    offsets = [0]
    for group in groups:
        offsets.append(offsets[-1] + len(group))
    return permutation, offsets


class _DistanceConstraintFamily:
    def __init__(
        self,
        constraints: list[tuple[int, int]],
        stiffnesses: list[float],
        rest_positions: np.ndarray,
        device: wp.Device,
    ):
        permutation, self.color_offsets = _greedy_color(constraints)
        ordered = [constraints[i] for i in permutation]
        ordered_stiffness = np.asarray(
            [stiffnesses[i] for i in permutation], dtype=np.float32
        )
        rest = np.asarray(
            [np.linalg.norm(rest_positions[j] - rest_positions[i]) for i, j in ordered],
            dtype=np.float32,
        )
        self.num_constraints = len(ordered)
        self.indices = wp.array(ordered, dtype=wp.vec2i, device=device)
        self.rest_lengths = wp.array(rest, dtype=wp.float32, device=device)
        self.compliance = wp.array(
            1.0 / ordered_stiffness, dtype=wp.float32, device=device
        )
        self.lambdas = wp.zeros(self.num_constraints, dtype=wp.float32, device=device)


class _VolumeConstraintFamily:
    def __init__(
        self,
        constraints: list[tuple[int, int, int, int]],
        stiffnesses: list[float],
        rest_positions: np.ndarray,
        device: wp.Device,
    ):
        permutation, self.color_offsets = _greedy_color(constraints)
        ordered = [constraints[i] for i in permutation]
        ordered_stiffness = np.asarray(
            [stiffnesses[i] for i in permutation], dtype=np.float32
        )
        volumes = []
        for i, j, k, p3 in ordered:
            volume = float(
                np.dot(
                    rest_positions[j] - rest_positions[i],
                    np.cross(
                        rest_positions[k] - rest_positions[i],
                        rest_positions[p3] - rest_positions[i],
                    ),
                )
                / 6.0
            )
            # Volume is scale-dependent. Small anatomical meshes can have
            # perfectly valid rest volumes far below an absolute 1e-12 cutoff.
            if volume == 0.0:
                raise ValueError(
                    "Zero-volume tetrahedron cannot be used by SolverXPBDRod"
                )
            volumes.append(volume)
        self.num_constraints = len(ordered)
        self.indices = wp.array(ordered, dtype=wp.vec4i, device=device)
        self.rest_volumes = wp.array(volumes, dtype=wp.float32, device=device)
        self.compliance = wp.array(
            1.0 / ordered_stiffness, dtype=wp.float32, device=device
        )
        self.lambdas = wp.zeros(self.num_constraints, dtype=wp.float32, device=device)


class _DihedralConstraintFamily:
    def __init__(
        self,
        constraints: list[tuple[int, int, int, int]],
        stiffnesses: list[float],
        rest_angles: list[float],
        rest_positions: np.ndarray,
        device: wp.Device,
    ):
        permutation, self.color_offsets = _greedy_color(constraints)
        ordered = [constraints[i] for i in permutation]
        ordered_stiffness = np.asarray(
            [stiffnesses[i] for i in permutation], dtype=np.float32
        )
        ordered_angles = np.asarray(
            [rest_angles[i] for i in permutation], dtype=np.float32
        )
        edge_lengths = np.asarray(
            [
                np.linalg.norm(rest_positions[p3] - rest_positions[k])
                for _, _, k, p3 in ordered
            ],
            dtype=np.float32,
        )
        self.num_constraints = len(ordered)
        self.indices = wp.array(ordered, dtype=wp.vec4i, device=device)
        self.rest_angles = wp.array(ordered_angles, dtype=wp.float32, device=device)
        self.compliance = wp.array(
            1.0 / (ordered_stiffness * edge_lengths),
            dtype=wp.float32,
            device=device,
        )
        self.lambdas = wp.zeros(self.num_constraints, dtype=wp.float32, device=device)


class _DeformableWorkspace:
    """Compact device workspace and preprocessed topology for deformable particles."""

    def __init__(
        self,
        model: Model,
        enable_cloth: bool,
        enable_tets: bool,
        rod_particles: set[int],
        cloth_bending_mode: str,
    ):
        device = model.device
        q = np.asarray(model.particle_q.numpy(), dtype=np.float32)

        def array(name: str, columns: int, dtype):
            value = getattr(model, name, None)
            if value is None:
                return np.empty((0, columns), dtype=dtype)
            return np.asarray(value.numpy(), dtype=dtype).reshape(-1, columns)

        triangles = array("tri_indices", 3, np.int32)
        tri_materials = array("tri_materials", 5, np.float32)
        bending_edges = array("edge_indices", 4, np.int32)
        bending_materials = array("edge_bending_properties", 2, np.float32)
        bending_rest_angles = np.asarray(
            model.edge_rest_angle.numpy()
            if getattr(model, "edge_rest_angle", None) is not None
            else np.empty(0),
            dtype=np.float32,
        )
        tets = array("tet_indices", 4, np.int32)
        tet_materials = array("tet_materials", 3, np.float32)

        def validate(values: np.ndarray, label: str):
            bad = ~np.isfinite(values) | (values < 0.0)
            if np.any(bad):
                raise ValueError(f"{label} stiffness must be finite and non-negative")

        if enable_cloth:
            validate(tri_materials[:, 0], "tri_ke")
            validate(bending_materials[:, 0], "edge_ke")
        if enable_tets:
            validate(tet_materials[:, 0], "k_mu")
            validate(tet_materials[:, 1], "k_lambda")

        cloth_edge_stiffness: dict[tuple[int, int], list[float]] = {}
        if enable_cloth:
            for tri, material in zip(triangles, tri_materials, strict=True):
                stiffness = float(material[0])
                if stiffness == 0.0:
                    continue
                for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                    key = tuple(sorted((int(a), int(b))))
                    cloth_edge_stiffness.setdefault(key, []).append(stiffness)

        cloth_bends_global: list[tuple[int, int]] = []
        cloth_dihedrals_global: list[tuple[int, int, int, int]] = []
        cloth_bend_stiffness: list[float] = []
        cloth_bend_rest_angles: list[float] = []
        if enable_cloth:
            for edge_index, (edge, material) in enumerate(
                zip(bending_edges, bending_materials, strict=True)
            ):
                stiffness = float(material[0])
                if stiffness > 0.0 and edge[0] >= 0 and edge[1] >= 0:
                    cloth_bends_global.append((int(edge[0]), int(edge[1])))
                    cloth_dihedrals_global.append(tuple(int(value) for value in edge))
                    cloth_bend_stiffness.append(stiffness)
                    cloth_bend_rest_angles.append(float(bending_rest_angles[edge_index]))

        tet_edge_stiffness: dict[tuple[int, int], list[float]] = {}
        tet_volumes_global: list[tuple[int, int, int, int]] = []
        tet_volume_stiffness: list[float] = []
        if enable_tets:
            for tet, material in zip(tets, tet_materials, strict=True):
                mu, lam = float(material[0]), float(material[1])
                if mu > 0.0 or lam > 0.0:
                    x0, x1, x2, x3 = (q[int(p)] for p in tet)
                    volume = float(np.dot(x1 - x0, np.cross(x2 - x0, x3 - x0)) / 6.0)
                    if volume == 0.0:
                        raise ValueError(
                            "Zero-volume tetrahedron cannot be used by SolverXPBDRod"
                        )
                if mu > 0.0:
                    for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
                        key = tuple(sorted((int(tet[a]), int(tet[b]))))
                        tet_edge_stiffness.setdefault(key, []).append(mu)
                if lam > 0.0:
                    tet_volumes_global.append(tuple(int(x) for x in tet))
                    tet_volume_stiffness.append(lam)

        referenced = set()
        for edge in (
            *cloth_edge_stiffness.keys(),
            *cloth_bends_global,
            *cloth_dihedrals_global,
            *tet_edge_stiffness.keys(),
        ):
            referenced.update(edge)
        for tet in tet_volumes_global:
            referenced.update(tet)
        overlap = referenced.intersection(rod_particles)
        if overlap:
            raise ValueError(
                "Rod and deformable constraints reference the same particle; double integration is undefined"
            )

        self.num_particles = len(referenced)
        self.empty = self.num_particles == 0
        self.cloth_stretch = None
        self.cloth_bend = None
        self.cloth_bend_distance = None
        self.cloth_bend_dihedral = None
        self.tet_edges = None
        self.tet_volume = None
        if self.empty:
            return

        global_indices = sorted(referenced)
        local = {p: i for i, p in enumerate(global_indices)}
        local_q = q[global_indices]
        zero_length_skipped = False

        def distances(edge_map, stiffness_values=None):
            nonlocal zero_length_skipped
            edges, stiffness = [], []
            items = (
                edge_map.items()
                if stiffness_values is None
                else zip(edge_map, stiffness_values, strict=True)
            )
            for edge, values in items:
                i, j = local[edge[0]], local[edge[1]]
                if np.linalg.norm(local_q[j] - local_q[i]) <= 1.0e-12:
                    zero_length_skipped = True
                    continue
                edges.append((i, j))
                stiffness.append(
                    float(np.mean(values))
                    if stiffness_values is None
                    else float(values)
                )
            return edges, stiffness

        cloth_edges, cloth_ke = distances(cloth_edge_stiffness)
        cloth_bends = {
            edge: value
            for edge, value in zip(
                cloth_bends_global, cloth_bend_stiffness, strict=True
            )
        }
        bend_edges, bend_ke = distances(cloth_bends)
        local_dihedrals = [
            tuple(local[p] for p in bend) for bend in cloth_dihedrals_global
        ]
        tet_edges, tet_ke = distances(tet_edge_stiffness)
        if zero_length_skipped:
            warnings.warn(
                "Skipping zero-length deformable distance constraint(s)",
                RuntimeWarning,
                stacklevel=2,
            )

        local_tets = [tuple(local[p] for p in tet) for tet in tet_volumes_global]
        if cloth_edges:
            self.cloth_stretch = _DistanceConstraintFamily(
                cloth_edges, cloth_ke, local_q, device
            )
        if bend_edges:
            self.cloth_bend_distance = _DistanceConstraintFamily(
                bend_edges, bend_ke, local_q, device
            )
        if local_dihedrals:
            self.cloth_bend_dihedral = _DihedralConstraintFamily(
                local_dihedrals,
                cloth_bend_stiffness,
                cloth_bend_rest_angles,
                local_q,
                device,
            )
        if tet_edges:
            self.tet_edges = _DistanceConstraintFamily(
                tet_edges, tet_ke, local_q, device
            )
        if local_tets:
            self.tet_volume = _VolumeConstraintFamily(
                local_tets, tet_volume_stiffness, local_q, device
            )

        incidence_base = np.zeros(self.num_particles, dtype=np.int32)
        for edges in (cloth_edges, tet_edges):
            for edge in edges:
                incidence_base[list(edge)] += 1
        for tet in local_tets:
            incidence_base[list(tet)] += 1
        incidence_distance = incidence_base.copy()
        incidence_dihedral = incidence_base.copy()
        for edge in bend_edges:
            incidence_distance[list(edge)] += 1
        for bend in local_dihedrals:
            incidence_dihedral[list(bend)] += 1

        self.particle_indices = wp.array(global_indices, dtype=wp.int32, device=device)
        self.positions = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)
        self.predicted = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)
        self.velocities = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)
        self.inv_masses = wp.array(
            model.particle_inv_mass.numpy()[global_indices],
            dtype=wp.float32,
            device=device,
        )
        particle_world = getattr(model, "particle_world", None)
        if particle_world is None:
            world_indices = np.zeros(self.num_particles, dtype=np.int32)
        else:
            world_indices = np.asarray(particle_world.numpy(), dtype=np.int32)[
                global_indices
            ]
        self.particle_world = wp.array(world_indices, dtype=wp.int32, device=device)
        self.incidence_counts_distance = wp.array(
            incidence_distance, dtype=wp.int32, device=device
        )
        self.incidence_counts_dihedral = wp.array(
            incidence_dihedral, dtype=wp.int32, device=device
        )
        self.incidence_counts = self.incidence_counts_distance
        self.corrections = wp.zeros(self.num_particles, dtype=wp.vec3, device=device)
        self.set_cloth_bending_mode(cloth_bending_mode)

    @property
    def families(self):
        return (self.cloth_stretch, self.cloth_bend, self.tet_edges, self.tet_volume)

    def set_cloth_bending_mode(self, mode: str) -> None:
        self.cloth_bend = (
            self.cloth_bend_dihedral if mode == "dihedral" else self.cloth_bend_distance
        )
        if hasattr(self, "incidence_counts_distance"):
            self.incidence_counts = (
                self.incidence_counts_dihedral
                if mode == "dihedral"
                else self.incidence_counts_distance
            )


class SolverXPBDRod(SolverBase):
    """XPBD solver for Cosserat rods, triangle cloth, and tetrahedral solids.

    This solver implements Extended Position-Based Dynamics (XPBD) for
    Cosserat elastic rods. It supports stretch and bend/twist constraints
    solved via block-tridiagonal direct solvers on GPU.

    Multiple solver backends are available:

    - ``"block_thomas"``: Block Thomas algorithm for 6x6 block-tridiagonal systems (default).
    - ``"split_thomas"``: Split into two 3x3 block-tridiagonal systems (stretch + darboux).
    - ``"block_jacobi"``: Block-diagonal Jacobi (ignores coupling between edges).
    - ``"banded_cholesky"``: Dense banded Cholesky for banded JMJT matrix.

    Args:
        model: The Newton model containing rod data.
        linear_damping: Linear velocity damping factor.
        angular_damping: Angular velocity damping factor.
        solver_backend: Solver backend to use.
        floor_z: Z coordinate of the floor plane, or ``None`` to disable.
        enable_cloth: Build constraints from positive-stiffness triangle and bending elements.
        enable_tets: Build edge and volume constraints from positive-stiffness tetrahedra.
        deformable_backend: ``"colored_gauss_seidel"`` or ``"jacobi"``.
        cloth_bending_mode: ``"distance"`` or true four-particle ``"dihedral"`` bending.
        deformable_iterations: Number of deformable projection iterations per step.
        deformable_relaxation: Position-correction relaxation in the interval ``(0, 1]``.
    """

    def __init__(
        self,
        model: Model,
        linear_damping: float = 0.0,
        angular_damping: float = 0.0,
        solver_backend: str = DIRECT_SOLVE_BLOCK_THOMAS,
        floor_z: float | None = 0.0,
        enable_cloth: bool = True,
        enable_tets: bool = True,
        deformable_backend: str = "colored_gauss_seidel",
        cloth_bending_mode: str = "distance",
        deformable_iterations: int = 8,
        deformable_relaxation: float = 1.0,
    ):
        super().__init__(model)
        if solver_backend not in DIRECT_SOLVE_BACKENDS:
            raise ValueError(
                f"Unknown solver backend {solver_backend!r}. "
                f"Expected one of {DIRECT_SOLVE_BACKENDS}"
            )
        if deformable_backend not in ("colored_gauss_seidel", "jacobi"):
            raise ValueError(
                f"Unknown deformable backend {deformable_backend!r}. "
                "Expected 'colored_gauss_seidel' or 'jacobi'"
            )
        if cloth_bending_mode not in ("distance", "dihedral"):
            raise ValueError(
                f"Unknown cloth bending mode {cloth_bending_mode!r}. "
                "Expected 'distance' or 'dihedral'"
            )
        if deformable_iterations < 1:
            raise ValueError("deformable_iterations must be at least 1")
        if (
            not np.isfinite(deformable_relaxation)
            or not 0.0 < deformable_relaxation <= 1.0
        ):
            raise ValueError("deformable_relaxation must be finite and in (0, 1]")

        self.linear_damping = linear_damping
        self.angular_damping = angular_damping
        self.solver_backend = solver_backend
        self.floor_z = floor_z
        self.enable_cloth = enable_cloth
        self.enable_tets = enable_tets
        self.deformable_backend = deformable_backend
        self._cloth_bending_mode = cloth_bending_mode
        self.deformable_iterations = int(deformable_iterations)
        self.deformable_relaxation = float(deformable_relaxation)

        device = model.device

        # Build rod workspaces from model data stored during build
        self._rods: list[_RodWorkspace] = []
        self._rod_particle_starts: list[int] = []
        self._batched_ws: _BatchedRodWorkspace | None = None

        rod_data = getattr(
            model,
            "xpbd_rod",
            {
                "rod_num_points": [],
                "rod_particle_start": [],
                "rod_young_modulus": [],
                "rod_torsion_modulus": [],
                "orientations": [],
                "quat_inv_masses": [],
                "rest_lengths": [],
                "rest_darboux": [],
                "bend_stiffness": [],
            },
        )
        rod_num_points = rod_data["rod_num_points"]
        rod_particle_starts = rod_data["rod_particle_start"]
        rod_young_moduli = rod_data["rod_young_modulus"]
        rod_torsion_moduli = rod_data["rod_torsion_modulus"]

        all_orientations = rod_data["orientations"]
        all_quat_inv_masses = rod_data["quat_inv_masses"]
        all_rest_lengths = rod_data["rest_lengths"]
        all_rest_darboux = rod_data["rest_darboux"]
        all_bend_stiffness = rod_data["bend_stiffness"]

        orient_cursor = 0
        edge_cursor = 0

        for rod_idx in range(len(rod_num_points)):
            np_ = rod_num_points[rod_idx]
            ne = np_ - 1
            ps = rod_particle_starts[rod_idx]

            ws = _RodWorkspace(np_, ne, device)
            ws.young_modulus = rod_young_moduli[rod_idx]
            ws.torsion_modulus = rod_torsion_moduli[rod_idx]

            # Copy particle positions/masses from model (GPU→GPU, no CPU roundtrip)
            wp.copy(
                dest=ws.positions_wp,
                src=model.particle_q,
                dest_offset=0,
                src_offset=ps,
                count=np_,
            )
            wp.copy(
                dest=ws.predicted_positions_wp,
                src=model.particle_q,
                dest_offset=0,
                src_offset=ps,
                count=np_,
            )
            wp.copy(
                dest=ws.inv_masses_wp,
                src=model.particle_inv_mass,
                dest_offset=0,
                src_offset=ps,
                count=np_,
            )

            # Copy rod-specific data
            orient_slice = np.array(
                all_orientations[orient_cursor : orient_cursor + np_], dtype=np.float32
            )
            ws.orientations_wp.assign(
                wp.array(orient_slice, dtype=wp.quat, device=device)
            )
            ws.predicted_orientations_wp.assign(
                wp.array(orient_slice, dtype=wp.quat, device=device)
            )
            ws.prev_orientations_wp.assign(
                wp.array(orient_slice, dtype=wp.quat, device=device)
            )

            qim_slice = np.array(
                all_quat_inv_masses[orient_cursor : orient_cursor + np_],
                dtype=np.float32,
            )
            ws.quat_inv_masses_wp.assign(
                wp.array(qim_slice, dtype=wp.float32, device=device)
            )

            rl_slice = np.array(
                all_rest_lengths[edge_cursor : edge_cursor + ne], dtype=np.float32
            )
            ws.rest_lengths_wp.assign(
                wp.array(rl_slice, dtype=wp.float32, device=device)
            )

            rd_slice = np.array(
                all_rest_darboux[edge_cursor : edge_cursor + ne], dtype=np.float32
            )
            ws.rest_darboux_wp.assign(wp.array(rd_slice, dtype=wp.vec3, device=device))

            bs_slice = np.array(
                all_bend_stiffness[edge_cursor : edge_cursor + ne], dtype=np.float32
            )
            ws.bend_stiffness_wp.assign(
                wp.array(bs_slice, dtype=wp.vec3, device=device)
            )

            # Gravity from model
            if model.gravity is not None:
                g = model.gravity.numpy()
                ws.gravity = wp.vec3(float(g[0][0]), float(g[0][1]), float(g[0][2]))

            orient_cursor += np_
            edge_cursor += ne

            self._rods.append(ws)

        # Store particle start indices for syncing back
        self._rod_particle_starts = list(rod_particle_starts) if rod_num_points else []

        # Build batched workspace for multi-rod parallelism
        self._batched_ws = None
        if len(self._rods) > 1 and self.solver_backend == DIRECT_SOLVE_BLOCK_THOMAS:
            self._batched_ws = _BatchedRodWorkspace(self._rods, device)

        rod_particles: set[int] = set()
        for start, ws in zip(self._rod_particle_starts, self._rods, strict=True):
            rod_particles.update(range(start, start + ws.num_points))
        self._deformable = _DeformableWorkspace(
            model, enable_cloth, enable_tets, rod_particles, cloth_bending_mode
        )

    @property
    def cloth_bending_mode(self) -> str:
        return self._cloth_bending_mode

    @cloth_bending_mode.setter
    def cloth_bending_mode(self, mode: str) -> None:
        if mode not in ("distance", "dihedral"):
            raise ValueError("cloth_bending_mode must be 'distance' or 'dihedral'")
        self._cloth_bending_mode = mode
        workspace = getattr(self, "_deformable", None)
        if workspace is not None and not workspace.empty:
            workspace.set_cloth_bending_mode(mode)

    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Register rod-specific data storage on the builder.

        Must be called before adding rods and before
        :meth:`~newton.ModelBuilder.finalize`.
        """
        builder._xpbd_rod_data = {
            "rod_num_points": [],
            "rod_particle_start": [],
            "rod_young_modulus": [],
            "rod_torsion_modulus": [],
            "orientations": [],
            "quat_inv_masses": [],
            "rest_lengths": [],
            "rest_darboux": [],
            "bend_stiffness": [],
        }

        # Wrap finalize to transfer rod data to the model
        original_finalize = builder.finalize

        def _finalize_with_rod_data(*args, **kwargs):
            model = original_finalize(*args, **kwargs)
            model.xpbd_rod = builder._xpbd_rod_data
            return model

        builder.finalize = _finalize_with_rod_data

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ):
        device = self.model.device
        timings_enabled = bool(getattr(self, "stage_timings_enabled", False))

        # Preserve all state not owned by this solver, including particles used only
        # for collision geometry or by another solver component.
        with wp.ScopedTimer(
            "solver.state_copy_in",
            active=timings_enabled,
            synchronize=True,
        ):
            if state_in.particle_q is not None:
                wp.copy(
                    dest=state_out.particle_q,
                    src=state_in.particle_q,
                    count=self.model.particle_count,
                )
                wp.copy(
                    dest=state_out.particle_qd,
                    src=state_in.particle_qd,
                    count=self.model.particle_count,
                )

        if not self._deformable.empty:
            self._step_deformable(state_in, state_out, dt, device)

        if self._batched_ws is not None:
            bws = self._batched_ws
            self._step_batched(bws, dt, device)
            # Sync positions back to state_out using precomputed offsets
            for rod_idx, ws in enumerate(self._rods):
                ps = self._rod_particle_starts[rod_idx]
                wp.copy(
                    dest=state_out.particle_q,
                    src=bws.positions,
                    dest_offset=ps,
                    src_offset=bws.rod_offsets_cpu[rod_idx],
                    count=ws.num_points,
                )
                wp.copy(
                    dest=state_out.particle_qd,
                    src=bws.velocities,
                    dest_offset=ps,
                    src_offset=bws.rod_offsets_cpu[rod_idx],
                    count=ws.num_points,
                )
        else:
            for rod_idx, ws in enumerate(self._rods):
                if ws.num_edges == 0:
                    continue

                self._step_rod(rod_idx, ws, dt, device)

                # Sync positions and velocities back to the model particle range.
                with wp.ScopedTimer(
                    "solver.state_copy_out",
                    active=timings_enabled,
                    synchronize=True,
                ):
                    ps = self._rod_particle_starts[rod_idx]
                    wp.copy(
                        dest=state_out.particle_q,
                        src=ws.positions_wp,
                        dest_offset=ps,
                        src_offset=0,
                        count=ws.num_points,
                    )
                    wp.copy(
                        dest=state_out.particle_qd,
                        src=ws.velocities_wp,
                        dest_offset=ps,
                        src_offset=0,
                        count=ws.num_points,
                    )

    def _step_deformable(
        self, state_in: State, state_out: State, dt: float, device: wp.Device
    ) -> None:
        ws = self._deformable
        wp.launch(
            _warp_gather_deformable,
            dim=ws.num_particles,
            inputs=[
                ws.particle_indices,
                state_in.particle_q,
                state_in.particle_qd,
                ws.positions,
                ws.velocities,
            ],
            device=device,
        )
        wp.launch(
            _warp_predict_deformable,
            dim=ws.num_particles,
            inputs=[
                ws.particle_indices,
                ws.positions,
                ws.velocities,
                state_in.particle_f,
                ws.inv_masses,
                ws.particle_world,
                self.model.gravity,
                float(dt),
                float(self.linear_damping),
                ws.predicted,
            ],
            device=device,
        )
        for family in ws.families:
            if family is not None:
                family.lambdas.zero_()

        if self.deformable_backend == "colored_gauss_seidel":
            for iteration in range(self.deformable_iterations):
                self._project_during_deformable_iteration(
                    ws, state_in, state_out, dt, device, iteration
                )
                for family in ws.families:
                    if family is None:
                        continue
                    if isinstance(family, _VolumeConstraintFamily):
                        kernel = _warp_project_volume_gs
                        values = family.rest_volumes
                    elif isinstance(family, _DihedralConstraintFamily):
                        kernel = _warp_project_dihedral_gs
                        values = family.rest_angles
                    else:
                        kernel = _warp_project_distance_gs
                        values = family.rest_lengths
                    for begin, end in zip(
                        family.color_offsets[:-1], family.color_offsets[1:], strict=True
                    ):
                        wp.launch(
                            kernel,
                            dim=end - begin,
                            inputs=[
                                family.indices,
                                values,
                                family.compliance,
                                family.lambdas,
                                ws.inv_masses,
                                ws.predicted,
                                begin,
                                float(dt),
                                self.deformable_relaxation,
                            ],
                            device=device,
                        )
                self._project_after_deformable_iteration(
                    ws, state_in, state_out, dt, device, iteration
                )
        else:
            for iteration in range(self.deformable_iterations):
                self._project_during_deformable_iteration(
                    ws, state_in, state_out, dt, device, iteration
                )
                ws.corrections.zero_()
                for family in ws.families:
                    if family is None:
                        continue
                    if isinstance(family, _VolumeConstraintFamily):
                        kernel = _warp_project_volume_jacobi
                        values = family.rest_volumes
                    elif isinstance(family, _DihedralConstraintFamily):
                        kernel = _warp_project_dihedral_jacobi
                        values = family.rest_angles
                    else:
                        kernel = _warp_project_distance_jacobi
                        values = family.rest_lengths
                    wp.launch(
                        kernel,
                        dim=family.num_constraints,
                        inputs=[
                            family.indices,
                            values,
                            family.compliance,
                            family.lambdas,
                            ws.inv_masses,
                            ws.predicted,
                            ws.corrections,
                            float(dt),
                            self.deformable_relaxation,
                        ],
                        device=device,
                    )
                wp.launch(
                    _warp_apply_deformable_corrections,
                    dim=ws.num_particles,
                    inputs=[
                        ws.predicted,
                        ws.corrections,
                        ws.incidence_counts,
                        ws.inv_masses,
                    ],
                    device=device,
                )
                self._project_after_deformable_iteration(
                    ws, state_in, state_out, dt, device, iteration
                )

        self._finish_deformable_iterations(ws, state_in, state_out, dt, device)

        wp.launch(
            _warp_scatter_deformable,
            dim=ws.num_particles,
            inputs=[
                ws.particle_indices,
                ws.positions,
                ws.predicted,
                ws.inv_masses,
                float(dt),
                state_out.particle_q,
                state_out.particle_qd,
            ],
            device=device,
        )

    def _project_during_deformable_iteration(
        self,
        ws: _DeformableWorkspace,
        state_in: State,
        state_out: State,
        dt: float,
        device: wp.Device,
        iteration: int,
    ) -> None:
        """Extension point run before each deformable constraint iteration."""
        del ws, state_in, state_out, dt, device, iteration

    def _finish_deformable_iterations(
        self,
        ws: _DeformableWorkspace,
        state_in: State,
        state_out: State,
        dt: float,
        device: wp.Device,
    ) -> None:
        """Extension point run after the deformable constraint loop."""
        del ws, state_in, state_out, dt, device

    def _project_after_deformable_iteration(
        self,
        ws: _DeformableWorkspace,
        state_in: State,
        state_out: State,
        dt: float,
        device: wp.Device,
        iteration: int,
    ) -> None:
        """Extension point run after each deformable constraint iteration."""
        del ws, state_in, state_out, dt, device, iteration

    def set_root_orientation(self, rod_idx: int, q: wp.quat) -> None:
        """Set the root particle orientation for a rod directly on GPU.

        Args:
            rod_idx: Index of the rod in ``self._rods``.
            q: New root orientation as a ``wp.quat``.
        """
        ws = self._rods[rod_idx]
        wp.launch(
            _warp_set_root_orientation,
            dim=1,
            inputs=[
                ws.orientations_wp,
                ws.predicted_orientations_wp,
                ws.prev_orientations_wp,
                q,
            ],
            device=ws.device,
        )

    def _project_predicted_positions(
        self,
        rod_idx: int,
        ws: _RodWorkspace,
        dt: float,
        device: wp.Device,
    ) -> None:
        """Private post-constraint extension point for predicted-position projection."""
        del rod_idx, ws, dt, device

    def _project_predicted_positions_pre_constraints(
        self,
        rod_idx: int,
        ws: _RodWorkspace,
        dt: float,
        device: wp.Device,
    ) -> None:
        """Private extension point before rod constraint projection."""
        del rod_idx, ws, dt, device

    def _project_predicted_positions_post_constraints(
        self,
        rod_idx: int,
        ws: _RodWorkspace,
        dt: float,
        device: wp.Device,
    ) -> None:
        """Private extension point after rod constraint projection."""
        self._project_predicted_positions(rod_idx, ws, dt, device)

    def _step_rod(self, rod_idx: int, ws: _RodWorkspace, dt: float, device: wp.Device):
        """Run one XPBD step for a single rod."""
        timings_enabled = bool(getattr(self, "stage_timings_enabled", False))

        # 1. Predict positions & rotations
        with wp.ScopedTimer(
            "catheter.predict",
            active=timings_enabled,
            synchronize=True,
        ):
            gravity = ws.gravity
            wp.launch(
                _warp_predict_positions,
                dim=ws.num_points,
                inputs=[
                    ws.positions_wp,
                    ws.velocities_wp,
                    ws.forces_wp,
                    ws.inv_masses_wp,
                    gravity,
                    float(dt),
                    float(self.linear_damping),
                    ws.predicted_positions_wp,
                ],
                device=device,
            )
            wp.launch(
                _warp_predict_rotations,
                dim=ws.num_points,
                inputs=[
                    ws.orientations_wp,
                    ws.angular_velocities_wp,
                    ws.torques_wp,
                    ws.quat_inv_masses_wp,
                    float(dt),
                    float(self.angular_damping),
                    ws.predicted_orientations_wp,
                ],
                device=device,
            )

        # 2. Private sample hook before rod constraint projection.
        with wp.ScopedTimer(
            "catheter.pre_collision",
            active=timings_enabled,
            synchronize=True,
        ):
            self._project_predicted_positions_pre_constraints(rod_idx, ws, dt, device)

        # 3. Prepare constraints
        with wp.ScopedTimer(
            "catheter.prepare_constraints",
            active=timings_enabled,
            synchronize=True,
        ):
            wp.launch(
                _warp_zero_float,
                dim=ws.n_dofs,
                inputs=[ws.lambda_sum_wp],
                device=device,
            )
            wp.launch(
                _warp_prepare_compliance,
                dim=ws.num_edges,
                inputs=[
                    ws.rest_lengths_wp,
                    ws.bend_stiffness_wp,
                    float(ws.young_modulus),
                    float(ws.torsion_modulus),
                    float(dt),
                    ws.compliance_wp,
                ],
                device=device,
            )

        # 4. Project constraints
        with wp.ScopedTimer(
            "catheter.direct_solve",
            active=timings_enabled,
            synchronize=True,
        ):
            self._project_direct(ws, device)

        # 5. Private sample hook after rod constraint projection.
        with wp.ScopedTimer(
            "catheter.post_collision_and_track",
            active=timings_enabled,
            synchronize=True,
        ):
            self._project_predicted_positions_post_constraints(rod_idx, ws, dt, device)

        # 6. Floor collision (optional)
        with wp.ScopedTimer(
            "catheter.floor_collision",
            active=timings_enabled and self.floor_z is not None,
            synchronize=True,
        ):
            if self.floor_z is not None:
                min_z = float(self.floor_z)
                wp.launch(
                    _warp_apply_floor_collisions,
                    dim=ws.num_points,
                    inputs=[ws.predicted_positions_wp, ws.velocities_wp, min_z, 0.0],
                    device=device,
                )

        # 7. Integrate
        with wp.ScopedTimer(
            "catheter.integrate",
            active=timings_enabled,
            synchronize=True,
        ):
            wp.launch(
                _warp_integrate_positions,
                dim=ws.num_points,
                inputs=[
                    ws.positions_wp,
                    ws.predicted_positions_wp,
                    ws.velocities_wp,
                    ws.inv_masses_wp,
                    float(dt),
                ],
                device=device,
            )
            wp.launch(
                _warp_integrate_rotations,
                dim=ws.num_points,
                inputs=[
                    ws.orientations_wp,
                    ws.predicted_orientations_wp,
                    ws.prev_orientations_wp,
                    ws.angular_velocities_wp,
                    ws.quat_inv_masses_wp,
                    float(dt),
                ],
                device=device,
            )

    def _step_batched(self, bws: _BatchedRodWorkspace, dt: float, device: wp.Device):
        """Run one XPBD step for all rods using batched kernel launches."""
        tp = bws.total_particles
        te = bws.total_edges
        td = bws.total_dofs

        # 1. Predict positions & rotations
        wp.launch(
            _warp_predict_positions_batched,
            dim=tp,
            inputs=[
                bws.positions,
                bws.velocities,
                bws.forces,
                bws.inv_masses,
                bws.gravity,
                bws.particle_rod_id,
                float(dt),
                float(self.linear_damping),
                bws.predicted_positions,
            ],
            device=device,
        )
        wp.launch(
            _warp_predict_rotations_batched,
            dim=tp,
            inputs=[
                bws.orientations,
                bws.angular_velocities,
                bws.torques,
                bws.quat_inv_masses,
                float(dt),
                float(self.angular_damping),
                bws.predicted_orientations,
            ],
            device=device,
        )

        # 2. Prepare constraints
        wp.launch(_warp_zero_float, dim=td, inputs=[bws.lambda_sum], device=device)
        wp.launch(
            _warp_prepare_compliance_batched,
            dim=te,
            inputs=[
                bws.rest_lengths,
                bws.bend_stiffness,
                bws.edge_rod_id,
                bws.young_modulus,
                bws.torsion_modulus,
                float(dt),
                bws.compliance,
            ],
            device=device,
        )

        # 3. Project constraints
        self._project_direct_batched(bws, device)

        # 4. Floor collision (optional)
        if self.floor_z is not None:
            wp.launch(
                _warp_apply_floor_collisions,
                dim=tp,
                inputs=[
                    bws.predicted_positions,
                    bws.velocities,
                    float(self.floor_z),
                    0.0,
                ],
                device=device,
            )

        # 5. Integrate
        wp.launch(
            _warp_integrate_positions_batched,
            dim=tp,
            inputs=[
                bws.positions,
                bws.predicted_positions,
                bws.velocities,
                bws.inv_masses,
                float(dt),
            ],
            device=device,
        )
        wp.launch(
            _warp_integrate_rotations_batched,
            dim=tp,
            inputs=[
                bws.orientations,
                bws.predicted_orientations,
                bws.prev_orientations,
                bws.angular_velocities,
                bws.quat_inv_masses,
                float(dt),
            ],
            device=device,
        )

    def _project_direct_batched(self, bws: _BatchedRodWorkspace, device: wp.Device):
        """Project constraints using batched block Thomas solver."""
        tp = bws.total_particles
        te = bws.total_edges
        td = bws.total_dofs

        # Update constraints
        wp.launch(
            _warp_update_constraints_batched_v2,
            dim=te,
            inputs=[
                bws.predicted_positions,
                bws.predicted_orientations,
                bws.rest_lengths,
                bws.rest_darboux,
                bws.rod_offsets,
                bws.edge_offsets,
                bws.edge_rod_id,
                bws.constraint_values,
            ],
            device=device,
        )

        # Compute Jacobians
        wp.launch(
            _warp_compute_jacobians_batched,
            dim=te,
            inputs=[
                bws.predicted_orientations,
                bws.rest_lengths,
                bws.rod_offsets,
                bws.edge_offsets,
                bws.edge_rod_id,
                bws.jacobian_pos,
                bws.jacobian_rot,
            ],
            device=device,
        )

        # Update inverse inertia
        wp.launch(
            _warp_compute_inv_inertia_world_batched,
            dim=tp,
            inputs=[
                bws.predicted_orientations,
                bws.quat_inv_masses,
                bws.inv_inertia_local_diag,
                bws.particle_rod_id,
                bws.inv_inertia,
            ],
            device=device,
        )

        # Assemble JMJT blocks
        wp.launch(
            _warp_assemble_jmjt_blocks_batched,
            dim=te,
            inputs=[
                bws.jacobian_pos,
                bws.jacobian_rot,
                bws.compliance,
                bws.inv_masses,
                bws.inv_inertia,
                bws.rod_offsets,
                bws.edge_offsets,
                bws.edge_rod_id,
                bws.diag_blocks,
                bws.offdiag_blocks,
            ],
            device=device,
        )

        # Build RHS
        wp.launch(
            _warp_build_rhs,
            dim=td,
            inputs=[
                bws.constraint_values,
                bws.compliance,
                bws.lambda_sum,
                int(td),
                bws.rhs,
            ],
            device=device,
        )

        # Batched Thomas solve (one thread per rod)
        wp.launch(
            _warp_block_thomas_solve_batched,
            dim=bws.n_rods,
            inputs=[
                bws.diag_blocks,
                bws.offdiag_blocks,
                bws.rhs,
                bws.edge_offsets,
                int(bws.n_rods),
                bws.c_blocks,
                bws.d_prime,
                bws.delta_lambda,
            ],
            device=device,
        )

        # Apply corrections (parallel two-phase)
        wp.launch(_warp_zero_vec3, dim=tp, inputs=[bws.pos_corrections], device=device)
        wp.launch(_warp_zero_vec3, dim=tp, inputs=[bws.rot_corrections], device=device)
        wp.launch(
            _warp_zero_float, dim=1, inputs=[bws._delta_lambda_max], device=device
        )
        wp.launch(_warp_zero_float, dim=1, inputs=[bws._correction_max], device=device)

        wp.launch(
            _warp_compute_corrections_parallel_batched,
            dim=te,
            inputs=[
                bws.predicted_positions,
                bws.inv_masses,
                bws.quat_inv_masses,
                bws.inv_inertia,
                bws.jacobian_pos,
                bws.jacobian_rot,
                bws.delta_lambda,
                bws.lambda_sum,
                bws.rod_offsets,
                bws.edge_offsets,
                bws.edge_rod_id,
                bws.pos_corrections,
                bws.rot_corrections,
                bws._delta_lambda_max,
                bws._correction_max,
            ],
            device=device,
        )
        wp.launch(
            _warp_apply_accumulated_corrections,
            dim=tp,
            inputs=[
                bws.predicted_positions,
                bws.predicted_orientations,
                bws.pos_corrections,
                bws.rot_corrections,
                int(tp),
            ],
            device=device,
        )

    def _project_direct(self, ws: _RodWorkspace, device: wp.Device):
        """Project constraints using the configured direct solver backend."""
        if ws.num_edges == 0:
            return

        # Update constraints
        wp.launch(
            _warp_update_constraints_direct,
            dim=ws.num_edges,
            inputs=[
                ws.predicted_positions_wp,
                ws.predicted_orientations_wp,
                ws.rest_lengths_wp,
                ws.rest_darboux_wp,
                ws.constraint_values_wp,
            ],
            device=device,
        )

        # Compute Jacobians
        wp.launch(
            _warp_compute_jacobians_direct,
            dim=ws.num_edges,
            inputs=[
                ws.predicted_orientations_wp,
                ws.rest_lengths_wp,
                ws.jacobian_pos_wp,
                ws.jacobian_rot_wp,
            ],
            device=device,
        )

        # Update inverse inertia
        inv_inertia_local = ws.inv_inertia_local_diag
        wp.launch(
            _warp_compute_inv_inertia_world,
            dim=ws.num_points,
            inputs=[
                ws.predicted_orientations_wp,
                ws.quat_inv_masses_wp,
                inv_inertia_local,
                ws.inv_inertia_wp,
            ],
            device=device,
        )

        n_dofs = ws.n_dofs
        delta_lambda = self._solve_system(ws, n_dofs, device)

        # Apply corrections (parallel two-phase)
        wp.launch(
            _warp_zero_vec3,
            dim=ws.num_points,
            inputs=[ws.pos_corrections_wp],
            device=device,
        )
        wp.launch(
            _warp_zero_vec3,
            dim=ws.num_points,
            inputs=[ws.rot_corrections_wp],
            device=device,
        )
        wp.launch(
            _warp_zero_float, dim=1, inputs=[ws._delta_lambda_max_wp], device=device
        )
        wp.launch(
            _warp_zero_float, dim=1, inputs=[ws._correction_max_wp], device=device
        )

        wp.launch(
            _warp_compute_corrections_parallel,
            dim=ws.num_edges,
            inputs=[
                ws.predicted_positions_wp,
                ws.inv_masses_wp,
                ws.quat_inv_masses_wp,
                ws.inv_inertia_wp,
                ws.jacobian_pos_wp,
                ws.jacobian_rot_wp,
                delta_lambda,
                ws.lambda_sum_wp,
                int(ws.num_edges),
                ws.pos_corrections_wp,
                ws.rot_corrections_wp,
                ws._delta_lambda_max_wp,
                ws._correction_max_wp,
            ],
            device=device,
        )
        wp.launch(
            _warp_apply_accumulated_corrections,
            dim=ws.num_points,
            inputs=[
                ws.predicted_positions_wp,
                ws.predicted_orientations_wp,
                ws.pos_corrections_wp,
                ws.rot_corrections_wp,
                int(ws.num_points),
            ],
            device=device,
        )

    def _solve_system(
        self, ws: _RodWorkspace, n_dofs: int, device: wp.Device
    ) -> wp.array:
        """Assemble and solve the linear system based on the chosen backend."""
        if self.solver_backend == DIRECT_SOLVE_SPLIT_THOMAS:
            return self._solve_split_thomas(ws, device)

        if self.solver_backend == DIRECT_SOLVE_BLOCK_JACOBI:
            wp.launch(
                _warp_assemble_jmjt_blocks,
                dim=ws.num_edges,
                inputs=[
                    ws.jacobian_pos_wp,
                    ws.jacobian_rot_wp,
                    ws.compliance_wp,
                    ws.inv_masses_wp,
                    ws.inv_inertia_wp,
                    int(ws.num_edges),
                    ws.diag_blocks_wp,
                    ws.offdiag_blocks_wp,
                ],
                device=device,
            )
            wp.launch(
                _warp_build_rhs,
                dim=n_dofs,
                inputs=[
                    ws.constraint_values_wp,
                    ws.compliance_wp,
                    ws.lambda_sum_wp,
                    int(n_dofs),
                    ws.rhs_wp,
                ],
                device=device,
            )
            wp.launch(
                _warp_solve_blocks_jacobi,
                dim=ws.num_edges,
                inputs=[
                    ws.diag_blocks_wp,
                    ws.rhs_wp,
                    ws.delta_lambda_wp,
                    int(ws.num_edges),
                ],
                device=device,
            )
            return ws.delta_lambda_wp

        if self.solver_backend == DIRECT_SOLVE_BANDED_CHOLESKY:
            wp.launch(
                _warp_zero_2d,
                dim=BAND_LDAB * max(1, n_dofs),
                inputs=[ws.ab_wp, int(BAND_LDAB), int(max(1, n_dofs))],
                device=device,
            )
            wp.launch(
                _warp_assemble_jmjt_banded,
                dim=ws.num_edges,
                inputs=[
                    ws.jacobian_pos_wp,
                    ws.jacobian_rot_wp,
                    ws.compliance_wp,
                    ws.inv_masses_wp,
                    ws.inv_inertia_wp,
                    int(n_dofs),
                    ws.ab_wp,
                ],
                device=device,
            )
            wp.launch(
                _warp_build_rhs,
                dim=n_dofs,
                inputs=[
                    ws.constraint_values_wp,
                    ws.compliance_wp,
                    ws.lambda_sum_wp,
                    int(n_dofs),
                    ws.rhs_wp,
                ],
                device=device,
            )
            wp.launch(
                _warp_spbsv_u11_1rhs,
                dim=1,
                inputs=[int(n_dofs), ws.ab_wp, ws.rhs_wp],
                device=device,
            )
            return ws.rhs_wp

        # Default: Block Thomas (or tiled Cholesky for small systems)
        if n_dofs <= TILE:
            wp.launch(
                _warp_zero_2d,
                dim=TILE * TILE,
                inputs=[ws.A_wp, int(TILE), int(TILE)],
                device=device,
            )
            wp.launch(
                _warp_assemble_jmjt_dense,
                dim=ws.num_edges,
                inputs=[
                    ws.jacobian_pos_wp,
                    ws.jacobian_rot_wp,
                    ws.compliance_wp,
                    ws.inv_masses_wp,
                    ws.inv_inertia_wp,
                    int(n_dofs),
                    ws.A_wp,
                ],
                device=device,
            )
            wp.launch(
                _warp_build_rhs,
                dim=TILE,
                inputs=[
                    ws.constraint_values_wp,
                    ws.compliance_wp,
                    ws.lambda_sum_wp,
                    int(n_dofs),
                    ws.rhs_tile_wp,
                ],
                device=device,
            )
            if n_dofs < TILE:
                wp.launch(
                    _warp_pad_diagonal,
                    dim=TILE,
                    inputs=[ws.A_wp, int(n_dofs), int(TILE)],
                    device=device,
                )
            wp.launch_tiled(
                _warp_cholesky_solve_tile,
                dim=[1, 1],
                inputs=[ws.A_wp, ws.rhs_tile_wp],
                outputs=[ws.delta_lambda_tile_wp],
                block_dim=BLOCK_DIM,
                device=device,
            )
            return ws.delta_lambda_tile_wp

        # Block Thomas for larger systems
        wp.launch(
            _warp_assemble_jmjt_blocks,
            dim=ws.num_edges,
            inputs=[
                ws.jacobian_pos_wp,
                ws.jacobian_rot_wp,
                ws.compliance_wp,
                ws.inv_masses_wp,
                ws.inv_inertia_wp,
                int(ws.num_edges),
                ws.diag_blocks_wp,
                ws.offdiag_blocks_wp,
            ],
            device=device,
        )
        wp.launch(
            _warp_build_rhs,
            dim=n_dofs,
            inputs=[
                ws.constraint_values_wp,
                ws.compliance_wp,
                ws.lambda_sum_wp,
                int(n_dofs),
                ws.rhs_wp,
            ],
            device=device,
        )
        wp.launch(
            _warp_block_thomas_solve,
            dim=1,
            inputs=[
                ws.diag_blocks_wp,
                ws.offdiag_blocks_wp,
                ws.rhs_wp,
                int(ws.num_edges),
                ws.c_blocks_wp,
                ws.d_prime_wp,
                ws.delta_lambda_wp,
            ],
            device=device,
        )
        return ws.delta_lambda_wp

    def _solve_split_thomas(self, ws: _RodWorkspace, device: wp.Device) -> wp.array:
        """Solve using split 3x3 block Thomas for stretch and darboux."""
        n = ws.num_edges

        # Lazily allocate split arrays
        if ws._split_stretch_diag_wp is None:
            ws._split_stretch_diag_wp = wp.zeros(n * 9, dtype=wp.float32, device=device)
            ws._split_stretch_offdiag_wp = wp.zeros(
                n * 9, dtype=wp.float32, device=device
            )
            ws._split_stretch_rhs_wp = wp.zeros(n * 3, dtype=wp.float32, device=device)
            ws._split_stretch_c_blocks_wp = wp.zeros(
                n * 9, dtype=wp.float32, device=device
            )
            ws._split_stretch_d_prime_wp = wp.zeros(
                n * 3, dtype=wp.float32, device=device
            )
            ws._split_stretch_delta_lambda_wp = wp.zeros(
                n * 3, dtype=wp.float32, device=device
            )
            ws._split_darboux_diag_wp = wp.zeros(n * 9, dtype=wp.float32, device=device)
            ws._split_darboux_offdiag_wp = wp.zeros(
                n * 9, dtype=wp.float32, device=device
            )
            ws._split_darboux_rhs_wp = wp.zeros(n * 3, dtype=wp.float32, device=device)
            ws._split_darboux_c_blocks_wp = wp.zeros(
                n * 9, dtype=wp.float32, device=device
            )
            ws._split_darboux_d_prime_wp = wp.zeros(
                n * 3, dtype=wp.float32, device=device
            )
            ws._split_darboux_delta_lambda_wp = wp.zeros(
                n * 3, dtype=wp.float32, device=device
            )

        # Assemble
        wp.launch(
            _warp_assemble_stretch_blocks,
            dim=n,
            inputs=[
                ws.jacobian_pos_wp,
                ws.jacobian_rot_wp,
                ws.compliance_wp,
                ws.inv_masses_wp,
                ws.inv_inertia_wp,
                int(n),
                ws._split_stretch_diag_wp,
                ws._split_stretch_offdiag_wp,
            ],
            device=device,
        )
        wp.launch(
            _warp_assemble_darboux_blocks,
            dim=n,
            inputs=[
                ws.jacobian_rot_wp,
                ws.compliance_wp,
                ws.inv_inertia_wp,
                int(n),
                ws._split_darboux_diag_wp,
                ws._split_darboux_offdiag_wp,
            ],
            device=device,
        )

        # Build RHS
        wp.launch(
            _warp_build_rhs_stretch,
            dim=n,
            inputs=[
                ws.constraint_values_wp,
                ws.compliance_wp,
                ws.lambda_sum_wp,
                int(n),
                ws._split_stretch_rhs_wp,
            ],
            device=device,
        )
        wp.launch(
            _warp_build_rhs_darboux,
            dim=n,
            inputs=[
                ws.constraint_values_wp,
                ws.compliance_wp,
                ws.lambda_sum_wp,
                int(n),
                ws._split_darboux_rhs_wp,
            ],
            device=device,
        )

        # Solve
        wp.launch(
            _warp_block_thomas_solve_3x3,
            dim=1,
            inputs=[
                ws._split_stretch_diag_wp,
                ws._split_stretch_offdiag_wp,
                ws._split_stretch_rhs_wp,
                int(n),
                ws._split_stretch_c_blocks_wp,
                ws._split_stretch_d_prime_wp,
                ws._split_stretch_delta_lambda_wp,
            ],
            device=device,
        )
        wp.launch(
            _warp_block_thomas_solve_3x3,
            dim=1,
            inputs=[
                ws._split_darboux_diag_wp,
                ws._split_darboux_offdiag_wp,
                ws._split_darboux_rhs_wp,
                int(n),
                ws._split_darboux_c_blocks_wp,
                ws._split_darboux_d_prime_wp,
                ws._split_darboux_delta_lambda_wp,
            ],
            device=device,
        )

        # Merge
        wp.launch(
            _warp_merge_delta_lambda,
            dim=n,
            inputs=[
                ws._split_stretch_delta_lambda_wp,
                ws._split_darboux_delta_lambda_wp,
                ws.delta_lambda_wp,
                int(n),
            ],
            device=device,
        )
        return ws.delta_lambda_wp

    @override
    def update_contacts(self, contacts: Contacts) -> None:
        pass
