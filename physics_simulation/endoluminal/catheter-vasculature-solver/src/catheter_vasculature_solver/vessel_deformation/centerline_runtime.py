# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Centerline vessel runtime: Cosserat predict/project/finalize + tapered-tube
# containment, orchestrated for the standalone XPBDRodSolver API.

"""Runtime that owns a deforming Cosserat centerline vessel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .centerline_containment import (
    CENTERLINE_EDGE_SAMPLES,
    apply_centerline_corrections_kernel,
    clamp_positions_delta_kernel,
    project_centerline_containment_kernel,
)
from .centerline_tree import CenterlineTree
from .pbd_cosserat import CosseratRod, finalize, predict, project
from .vessel_skinning import VesselSkinner


@dataclass
class CenterlineDynamicsParams:
    """Material / BC knobs for the centerline Cosserat vessel."""

    node_mass: float = 1.0
    segment_inertia: float = 1.0
    bend_stiffness: float = 1.0
    twist_stiffness: float = 1.0
    root_locked: bool = True
    endpoints_locked: bool = False
    stretch_stiffness: float = 1.0
    iterations: int = 4
    linear_damping: float = 0.0
    angular_damping: float = 0.0
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)


class CenterlineVesselRuntime:
    """Deforming centerline vessel + live tapered-tube containment.

    Step order::

        snapshot → predict_cosserat → project/clamp iterations
        → (catheter XPBD substeps, with containment in post-constraint hook)
        → finalize_cosserat
    """

    def __init__(
        self,
        rod: CosseratRod,
        tree: CenterlineTree,
        *,
        edges: "wp.array",
        radii: "wp.array",
        params: CenterlineDynamicsParams | None = None,
        catheter_radius: float = 0.002,
        max_distance: float = 0.05,
        two_way: bool = False,
        vessel_response: float = 0.5,
        catheter_max_delta: float = 1.0e30,
        vessel_max_delta: float = 1.0e30,
        collision_iterations: int = 2,
        open_root: int | None = None,
        open_root_neighbor: int | None = None,
    ):
        self.rod = rod
        self.tree = tree
        self.edges = edges
        self.radii = radii
        self.params = params or CenterlineDynamicsParams()
        self.catheter_radius = float(catheter_radius)
        self.max_distance = float(max_distance)
        self.two_way = bool(two_way)
        self.vessel_response = float(vessel_response)
        self.catheter_max_delta = float(catheter_max_delta)
        self.vessel_max_delta = float(vessel_max_delta)
        self.collision_iterations = max(1, int(collision_iterations))

        root = int(tree.root) if open_root is None else int(open_root)
        if open_root_neighbor is None:
            neighbor = -1
            for a, b in np.asarray(tree.edges, dtype=np.int32):
                if int(a) == root:
                    neighbor = int(b)
                    break
                if int(b) == root:
                    neighbor = int(a)
                    break
            open_root_neighbor = neighbor
        self.open_root = int(root)
        self.open_root_neighbor = int(open_root_neighbor)

        self.skinner: "VesselSkinner | None" = None

        device = rod.device
        self._step_start = wp.zeros(rod.n_nodes, dtype=wp.vec3, device=device)
        self._vessel_corrections = wp.zeros(rod.n_nodes, dtype=wp.vec3, device=device)
        self._vessel_counts = wp.zeros(rod.n_nodes, dtype=wp.float32, device=device)
        self._catheter_corrections: wp.array | None = None
        self._catheter_counts: wp.array | None = None
        self._catheter_step_start: wp.array | None = None

    # ------------------------------------------------------------------ factory
    @classmethod
    def from_tree(
        cls,
        tree: CenterlineTree,
        device: str = "cuda:0",
        *,
        params: CenterlineDynamicsParams | None = None,
        collision_radius_scale: float = 1.0,
        **kwargs,
    ) -> "CenterlineVesselRuntime":
        """Build a Cosserat centerline vessel from a :class:`CenterlineTree`."""
        params = params or CenterlineDynamicsParams()
        positions = np.asarray(tree.positions, dtype=np.float32)
        edges = [tuple(map(int, e)) for e in np.asarray(tree.edges, dtype=np.int32)]

        fixed_nodes: set[int] = set()
        if params.root_locked:
            fixed_nodes.add(int(tree.root))
        if params.endpoints_locked:
            degree = np.bincount(
                np.asarray(tree.edges, dtype=np.int32).reshape(-1),
                minlength=len(positions),
            )
            fixed_nodes.update(np.flatnonzero(degree == 1).tolist())
        fixed_segments = (
            [int(tree.root_segment)]
            if params.root_locked and tree.root_segment is not None
            else []
        )

        rod = CosseratRod(
            positions,
            device,
            edges=edges,
            node_mass=params.node_mass,
            seg_inertia=params.segment_inertia,
            bend_stiffness=(params.bend_stiffness, params.bend_stiffness),
            twist_stiffness=params.twist_stiffness,
            fixed_nodes=sorted(fixed_nodes),
            fixed_segments=fixed_segments,
            fix_root_pos=False,
            fix_root_orient=False,
            orientation_mode="pbd",
        )
        radii_np = np.maximum(
            np.asarray(tree.radii, dtype=np.float32) * float(collision_radius_scale),
            1.0e-6,
        )
        edges_wp = wp.array(np.asarray(tree.edges, dtype=np.int32), dtype=wp.vec2i, device=device)
        radii_wp = wp.array(radii_np, dtype=wp.float32, device=device)
        return cls(
            rod,
            tree,
            edges=edges_wp,
            radii=radii_wp,
            params=params,
            **kwargs,
        )

    # --------------------------------------------------------------- buffers
    def _ensure_catheter_buffers(self, num_points: int, device: str) -> None:
        if (
            self._catheter_corrections is not None
            and self._catheter_corrections.shape[0] >= num_points
            and str(self._catheter_corrections.device) == str(device)
        ):
            return
        self._catheter_corrections = wp.zeros(num_points, dtype=wp.vec3, device=device)
        self._catheter_counts = wp.zeros(num_points, dtype=wp.float32, device=device)
        self._catheter_step_start = wp.zeros(num_points, dtype=wp.vec3, device=device)

    # ---------------------------------------------------------- orchestration
    def begin_step(self, catheter_positions: "wp.array | None" = None) -> None:
        """Snapshot centerline (and optional catheter) for delta clamps."""
        wp.copy(self._step_start, self.rod.x, count=self.rod.n_nodes)
        if catheter_positions is not None:
            self._ensure_catheter_buffers(catheter_positions.shape[0], str(catheter_positions.device))
            wp.copy(self._catheter_step_start, catheter_positions, count=catheter_positions.shape[0])

    def predict(self, dt: float) -> None:
        predict(self.rod, dt, gravity=self.params.gravity)

    def project_constraints(self) -> None:
        params = self.params
        for _ in range(params.iterations):
            project(self.rod, 1, params.stretch_stiffness)
            self.clamp_vessel_delta()

    def finalize(self, dt: float) -> None:
        self.clamp_vessel_delta()
        finalize(
            self.rod,
            dt,
            self.params.linear_damping,
            self.params.angular_damping,
        )
        self.update_surface()

    # ------------------------------------------------------------------ surface
    def attach_surface(self, vertices, triangles) -> VesselSkinner:
        """Bind a render surface to the centerline and return the skinner.

        Call once with the vessel mesh in the same frame as the centerline tree.
        Afterwards every :meth:`finalize` refreshes ``surface_positions`` /
        ``surface_normals`` for the viewer or fluoro renderer to read.
        """
        skinner = VesselSkinner(vertices, triangles, self.edges.numpy(), device=self.rod.device)
        # Bind and capture rest state against the committed (rest) centerline.
        skinner.bind(self.rod.x)
        skinner.compute_rest_coordinates(None, None, self.rod.x, self.rod.q)
        self.skinner = skinner
        self.update_surface()
        return skinner

    def update_surface(self) -> None:
        """Re-skin the attached surface from the committed centerline state."""
        if self.skinner is not None:
            self.skinner.update(self.rod.x, self.rod.q)

    @property
    def surface_positions(self) -> np.ndarray | None:
        return None if self.skinner is None else self.skinner.positions

    @property
    def surface_normals(self) -> np.ndarray | None:
        return None if self.skinner is None else self.skinner.normals

    def clamp_vessel_delta(self) -> None:
        wp.launch(
            clamp_positions_delta_kernel,
            dim=self.rod.n_nodes,
            inputs=[
                self.rod.p,
                self._step_start,
                self.rod.inv_mass,
                float(self.vessel_max_delta),
            ],
            device=self.rod.device,
        )

    def clamp_catheter_delta(self, predicted: "wp.array", inv_masses: "wp.array") -> None:
        if self._catheter_step_start is None:
            return
        wp.launch(
            clamp_positions_delta_kernel,
            dim=predicted.shape[0],
            inputs=[
                predicted,
                self._catheter_step_start,
                inv_masses,
                float(self.catheter_max_delta),
            ],
            device=predicted.device,
        )

    def project_containment(
        self,
        catheter_predicted: "wp.array",
        catheter_inv_mass: "wp.array",
        num_points: int,
        num_edges: int,
    ) -> None:
        """Contain catheter predicted positions against live centerline tubes."""
        self._ensure_catheter_buffers(num_points, str(catheter_predicted.device))
        sample_dim = num_points + CENTERLINE_EDGE_SAMPLES * max(num_edges, 0)
        device = catheter_predicted.device

        for _ in range(self.collision_iterations):
            self._catheter_corrections.zero_()
            self._catheter_counts.zero_()
            self._vessel_corrections.zero_()
            self._vessel_counts.zero_()
            wp.launch(
                project_centerline_containment_kernel,
                dim=sample_dim,
                inputs=[
                    catheter_predicted,
                    catheter_inv_mass,
                    int(num_points),
                    self.rod.p,
                    self.edges,
                    self.radii,
                    self.rod.inv_mass,
                    int(self.open_root),
                    int(self.open_root_neighbor),
                    float(self.catheter_radius),
                    float(self.max_distance),
                    int(1 if self.two_way else 0),
                    float(self.vessel_response),
                    self._catheter_corrections,
                    self._catheter_counts,
                    self._vessel_corrections,
                    self._vessel_counts,
                ],
                device=device,
            )
            wp.launch(
                apply_centerline_corrections_kernel,
                dim=num_points,
                inputs=[
                    catheter_predicted,
                    catheter_inv_mass,
                    self._catheter_corrections,
                    self._catheter_counts,
                ],
                device=device,
            )
            self.clamp_catheter_delta(catheter_predicted, catheter_inv_mass)
            if self.two_way:
                wp.launch(
                    apply_centerline_corrections_kernel,
                    dim=self.rod.n_nodes,
                    inputs=[
                        self.rod.p,
                        self.rod.inv_mass,
                        self._vessel_corrections,
                        self._vessel_counts,
                    ],
                    device=self.rod.device,
                )
                self.clamp_vessel_delta()

    @property
    def positions(self) -> np.ndarray:
        return self.rod.x.numpy()

    @property
    def predicted_positions(self) -> np.ndarray:
        return self.rod.p.numpy()


__all__ = ["CenterlineDynamicsParams", "CenterlineVesselRuntime", "VesselSkinner"]
