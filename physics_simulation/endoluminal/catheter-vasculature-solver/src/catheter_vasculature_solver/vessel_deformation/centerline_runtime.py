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

from ..host_arrays import to_host_ids
from .centerline_containment import (
    CENTERLINE_EDGE_SAMPLES,
    apply_centerline_corrections_kernel,
    clamp_positions_delta_kernel,
    project_centerline_containment_batched_kernel,
    project_centerline_containment_kernel,
    project_centerline_containment_per_env_kernel,
    reset_centerline_contact_kernel,
    reset_centerline_nodes_kernel,
    reset_centerline_segments_kernel,
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


def _root_neighbor(edges, root: int) -> int:
    """The node across the first edge touching ``root``, or ``-1``."""
    for a, b in np.asarray(edges, dtype=np.int32).reshape((-1, 2)):
        if int(a) == root:
            return int(b)
        if int(b) == root:
            return int(a)
    return -1


def _tree_boundary(tree: CenterlineTree, params: CenterlineDynamicsParams) -> tuple[list[int], list[int]]:
    """Fixed nodes (sorted, so the first roots the rod) and fixed segments of one tree."""
    n_nodes = int(np.asarray(tree.positions).shape[0])
    fixed_nodes: set[int] = set()
    if params.root_locked:
        fixed_nodes.add(int(tree.root))
    if params.endpoints_locked:
        degree = np.bincount(np.asarray(tree.edges, dtype=np.int32).reshape(-1), minlength=n_nodes)
        fixed_nodes.update(np.flatnonzero(degree == 1).tolist())
    fixed_segments = (
        [int(tree.root_segment)] if params.root_locked and tree.root_segment is not None else []
    )
    return sorted(fixed_nodes), fixed_segments


def _rod_material(params: CenterlineDynamicsParams) -> dict:
    """The CosseratRod settings a centerline vessel is built with."""
    return dict(
        node_mass=params.node_mass,
        seg_inertia=params.segment_inertia,
        bend_stiffness=(params.bend_stiffness, params.bend_stiffness),
        twist_stiffness=params.twist_stiffness,
        fix_root_pos=False,
        fix_root_orient=False,
        orientation_mode="pbd",
    )


class CenterlineVesselRuntime:
    """Deforming centerline vessel + live tapered-tube containment.

    Step order::

        snapshot → predict_cosserat → project/clamp iterations
        → (catheter XPBD substeps, with containment in post-constraint hook)
        → finalize_cosserat

    Multi-env: with ``num_envs > 1`` each env owns an independent vessel, so
    two-way contact deforms only the calling env's wall and envs cannot
    contaminate one another. Vessel memory therefore scales with ``num_envs``.
    Containment still runs as a single batched launch per collision iteration,
    so cost tracks total samples rather than env count.

    Envs may also hold different vessels (:meth:`from_trees`). Each env's
    vessel then sits in a window padded to the largest one, and containment
    searches only that env's ``edge_counts[env]`` real edges from its own entry
    (``open_roots[env]``, ``open_root_neighbors[env]``).
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
        interior_deadband: float = 1.0,
        interior_stiffness: float = 0.0,
        two_way: bool = False,
        vessel_response: float = 0.5,
        catheter_max_delta: float = 1.0e30,
        vessel_max_delta: float = 1.0e30,
        collision_iterations: int = 2,
        open_root: int | None = None,
        open_root_neighbor: int | None = None,
        edge_counts: np.ndarray | None = None,
        open_roots: np.ndarray | None = None,
        open_root_neighbors: np.ndarray | None = None,
    ):
        self.rod = rod
        self.tree = tree
        self.trees = [tree]
        self.env_tree = np.zeros(int(getattr(rod, "num_envs", 1)), dtype=np.int32)
        self.edges = edges
        self.radii = radii
        self.params = params or CenterlineDynamicsParams()
        self.catheter_radius = float(catheter_radius)
        self.max_distance = float(max_distance)
        # Where the interior pull starts, as a fraction of the radius the wire is
        # free to occupy, and how hard it pulls. The defaults are inert -- zero
        # stiffness, deadband at the wall -- so containment stays the one-sided
        # test it has always been until a caller asks for otherwise.
        self.interior_deadband = min(max(float(interior_deadband), 0.0), 1.0)
        self.interior_stiffness = min(max(float(interior_stiffness), 0.0), 1.0)
        self.two_way = bool(two_way)
        self.vessel_response = float(vessel_response)
        self.catheter_max_delta = float(catheter_max_delta)
        self.vessel_max_delta = float(vessel_max_delta)
        self.collision_iterations = max(1, int(collision_iterations))

        root = int(tree.root) if open_root is None else int(open_root)
        if open_root_neighbor is None:
            open_root_neighbor = _root_neighbor(tree.edges, root)
        self.open_root = int(root)
        self.open_root_neighbor = int(open_root_neighbor)

        self.skinner: "VesselSkinner | None" = None
        self._skin_env = 0
        self._skin_x: "wp.array | None" = None
        self._skin_q: "wp.array | None" = None

        # Batch geometry. Node/segment counts come from the rod so a caller
        # cannot describe a batch the vessel state does not actually have.
        self.num_envs = int(getattr(rod, "num_envs", 1))
        self.nodes_per_env = int(getattr(rod, "n_nodes_per_env", rod.n_nodes))
        self.segments_per_env = int(getattr(rod, "n_seg_per_env", rod.n_seg))
        total_edges = int(edges.shape[0])
        if total_edges % self.num_envs != 0:
            raise ValueError(
                f"edges length {total_edges} is not divisible by num_envs {self.num_envs}"
            )
        self.edges_per_env = total_edges // self.num_envs
        if int(radii.shape[0]) != self.nodes_per_env * self.num_envs:
            raise ValueError(
                f"radii length {int(radii.shape[0])} does not match "
                f"nodes_per_env {self.nodes_per_env} x num_envs {self.num_envs}"
            )

        # Per-env vessel extents. A uniform batch has every edge of every window
        # real and one shared entry; a padded batch says otherwise per env.
        self.per_env_vessels = (
            edge_counts is not None
            or open_roots is not None
            or open_root_neighbors is not None
            or not bool(getattr(rod, "uniform_topology", True))
        )
        self.edge_counts = self._per_env_ints(edge_counts, self.edges_per_env, "edge_counts")
        if np.any(self.edge_counts < 0) or np.any(self.edge_counts > self.edges_per_env):
            raise ValueError(f"edge_counts must lie in [0, {self.edges_per_env}]")
        self.open_roots = self._per_env_ints(open_roots, self.open_root, "open_roots")
        self.open_root_neighbors = self._per_env_ints(
            open_root_neighbors, self.open_root_neighbor, "open_root_neighbors"
        )

        device = rod.device
        self._edge_counts_wp = wp.array(self.edge_counts, dtype=wp.int32, device=device)
        self._open_roots_wp = wp.array(self.open_roots, dtype=wp.int32, device=device)
        self._open_root_neighbors_wp = wp.array(self.open_root_neighbors, dtype=wp.int32, device=device)
        self._step_start = wp.zeros(rod.n_nodes, dtype=wp.vec3, device=device)
        self._vessel_corrections = wp.zeros(rod.n_nodes, dtype=wp.vec3, device=device)
        self._vessel_counts = wp.zeros(rod.n_nodes, dtype=wp.float32, device=device)
        self._catheter_corrections: wp.array | None = None
        self._catheter_counts: wp.array | None = None
        self._catheter_step_start: wp.array | None = None
        self._contact_depth = wp.zeros(self.num_envs, dtype=wp.float32, device=device)
        self._contact_count = wp.zeros(self.num_envs, dtype=wp.int32, device=device)

        # Rest state for reset(), captured before any stepping.
        self._rest_x = wp.clone(rod.x)
        self._rest_q = wp.clone(rod.q)
        # Env-id scratch, staged on the host and copied, so reset() allocates
        # nothing on the hot path.
        self._env_ids_scratch = wp.zeros(max(self.num_envs, 1), dtype=wp.int32, device=device)
        self._env_ids_host = wp.zeros(max(self.num_envs, 1), dtype=wp.int32, device="cpu")

    # ------------------------------------------------------------------ factory
    @classmethod
    def from_tree(
        cls,
        tree: CenterlineTree,
        device: str = "cuda:0",
        *,
        params: CenterlineDynamicsParams | None = None,
        collision_radius_scale: float = 1.0,
        num_envs: int = 1,
        env_positions: np.ndarray | None = None,
        env_radii: np.ndarray | None = None,
        **kwargs,
    ) -> "CenterlineVesselRuntime":
        """Build a Cosserat centerline vessel from a :class:`CenterlineTree`.

        Args:
            tree: Centerline topology and rest geometry.
            device: Warp device string.
            params: Material / boundary-condition knobs.
            collision_radius_scale: Multiplier on the tree's lumen radii.
            num_envs: Independent vessel replicas. Each env gets its own state,
                so two-way contact in one env cannot move another env's wall.
            env_positions: Optional ``(num_envs, n_nodes, 3)`` per-env node
                positions, for anatomy that varies across envs. Defaults to
                replicating ``tree.positions``. Topology is always shared; use
                :meth:`from_trees` for envs whose vessel trees differ.
            env_radii: Optional ``(num_envs, n_nodes)`` per-env lumen radii,
                before ``collision_radius_scale``. Defaults to replicating
                ``tree.radii``.
        """
        params = params or CenterlineDynamicsParams()
        num_envs = int(num_envs)
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        positions = np.asarray(tree.positions, dtype=np.float32)
        if env_positions is not None:
            env_positions = np.asarray(env_positions, dtype=np.float32).reshape((num_envs, -1, 3))
            if env_positions.shape[1] != positions.shape[0]:
                raise ValueError(
                    f"env_positions has {env_positions.shape[1]} nodes but the tree has "
                    f"{positions.shape[0]}; topology is shared across envs"
                )
            positions = env_positions
        edges = [tuple(map(int, e)) for e in np.asarray(tree.edges, dtype=np.int32)]

        n_nodes = int(np.asarray(tree.positions).shape[0])
        fixed_nodes, fixed_segments = _tree_boundary(tree, params)

        rod = CosseratRod(
            positions,
            device,
            edges=edges,
            fixed_nodes=fixed_nodes,
            fixed_segments=fixed_segments,
            num_envs=num_envs,
            **_rod_material(params),
        )
        if env_radii is None:
            radii_np = np.tile(np.asarray(tree.radii, dtype=np.float32), num_envs)
        else:
            radii_np = np.asarray(env_radii, dtype=np.float32)
            if radii_np.shape != (num_envs, n_nodes):
                raise ValueError(
                    f"env_radii must have shape ({num_envs}, {n_nodes}), got {radii_np.shape}"
                )
            radii_np = radii_np.reshape(-1)
        radii_np = np.maximum(radii_np * float(collision_radius_scale), 1.0e-6)
        # Edges carry global node indices and radii are per global node, so the
        # batched containment kernel can address each env's vessel directly.
        edges_np = np.asarray(tree.edges, dtype=np.int32)
        if num_envs > 1:
            node_off = (np.arange(num_envs, dtype=np.int32) * n_nodes)[:, None, None]
            edges_np = (edges_np[None, :, :] + node_off).reshape((-1, 2))
        edges_wp = wp.array(edges_np, dtype=wp.vec2i, device=device)
        radii_wp = wp.array(radii_np, dtype=wp.float32, device=device)
        return cls(
            rod,
            tree,
            edges=edges_wp,
            radii=radii_wp,
            params=params,
            **kwargs,
        )

    @classmethod
    def from_trees(
        cls,
        trees,
        device: str = "cuda:0",
        *,
        env_trees=None,
        params: CenterlineDynamicsParams | None = None,
        collision_radius_scale: float = 1.0,
        **kwargs,
    ) -> "CenterlineVesselRuntime":
        """Build a batch in which envs hold different vessel trees.

        Each env's vessel is padded to the largest tree's node and edge counts,
        so the batch keeps one fixed-stride layout and a single containment
        launch per collision iteration. Padding is inert: padded nodes carry
        zero inverse mass and padded edges are never searched.

        Args:
            trees: The distinct :class:`CenterlineTree` anatomies.
            device: Warp device string.
            env_trees: Index into ``trees`` for each env, which also sets the
                env count. Defaults to one env per tree.
            params: Material / boundary-condition knobs shared by every env.
            collision_radius_scale: Multiplier on every tree's lumen radii.
            **kwargs: Forwarded to the constructor (contact settings). Per-env
                entries are derived from each tree, so ``open_root`` and
                ``open_root_neighbor`` are not accepted.
        """
        trees = list(trees)
        if not trees:
            raise ValueError("from_trees needs at least one tree")
        for key in ("num_envs", "env_positions", "env_radii", "open_root", "open_root_neighbor",
                    "edge_counts", "open_roots", "open_root_neighbors"):
            if key in kwargs:
                raise TypeError(f"from_trees derives {key!r} from the trees; do not pass it")
        env_trees = (
            np.arange(len(trees), dtype=np.int32)
            if env_trees is None
            else np.asarray(env_trees, dtype=np.int32).reshape(-1)
        )
        if env_trees.size == 0 or env_trees.min() < 0 or env_trees.max() >= len(trees):
            raise ValueError(f"env_trees must index into the {len(trees)} trees")
        params = params or CenterlineDynamicsParams()

        env_list = [trees[int(i)] for i in env_trees]
        boundaries = [_tree_boundary(tree, params) for tree in trees]
        rod = CosseratRod.from_envs(
            [np.asarray(tree.positions, dtype=np.float32) for tree in env_list],
            device,
            env_edges=[[tuple(map(int, e)) for e in np.asarray(tree.edges, dtype=np.int32)] for tree in env_list],
            env_fixed_nodes=[boundaries[int(i)][0] for i in env_trees],
            env_fixed_segments=[boundaries[int(i)][1] for i in env_trees],
            **_rod_material(params),
        )

        # Vessel edges are the rod's segments in the same order, so the rod's
        # padded window (degenerate edges on the env's root) is reused as is.
        nodes, segments = rod.n_nodes_per_env, rod.n_seg_per_env
        edges_np = np.stack([rod.seg_n0.numpy(), rod.seg_n1.numpy()], axis=1).astype(np.int32)
        radii_np = np.empty((len(env_list), nodes), dtype=np.float32)
        for env, tree in enumerate(env_list):
            radii = np.asarray(tree.radii, dtype=np.float32).reshape(-1)
            radii_np[env, : len(radii)] = radii
            radii_np[env, len(radii):] = radii[int(rod.env_roots[env])]
        radii_np = np.maximum(radii_np.reshape(-1) * float(collision_radius_scale), 1.0e-6)
        if edges_np.shape[0] != segments * len(env_list):
            raise ValueError("vessel trees without edges cannot be batched")

        runtime = cls(
            rod,
            env_list[0],
            edges=wp.array(edges_np, dtype=wp.vec2i, device=device),
            radii=wp.array(radii_np, dtype=wp.float32, device=device),
            params=params,
            edge_counts=rod.env_segment_counts,
            open_roots=np.asarray([int(tree.root) for tree in env_list], dtype=np.int32),
            open_root_neighbors=np.asarray(
                [_root_neighbor(tree.edges, int(tree.root)) for tree in env_list], dtype=np.int32
            ),
            **kwargs,
        )
        runtime.trees = trees
        runtime.env_tree = env_trees
        return runtime

    def _per_env_ints(self, values, default: int, name: str) -> np.ndarray:
        """``values`` as one int32 per env, or ``default`` replicated."""
        if values is None:
            return np.full(self.num_envs, int(default), dtype=np.int32)
        values = np.asarray(values, dtype=np.int32).reshape(-1)
        if values.shape != (self.num_envs,):
            raise ValueError(f"{name} must have one entry per env ({self.num_envs}), got {values.shape}")
        return values.copy()

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
    def attach_surface(self, vertices, triangles, env: int = 0) -> VesselSkinner:
        """Bind a render surface to the centerline and return the skinner.

        Call once with the vessel mesh in the same frame as the centerline tree.
        Afterwards every :meth:`finalize` refreshes ``surface_positions`` /
        ``surface_normals`` for the viewer or fluoro renderer to read.

        Args:
            vertices: Surface vertices in the centerline's frame.
            triangles: Triangle indices into *vertices*.
            env: Which env's vessel drives the surface. Skinning follows one env
                because it exists to be rendered, and a viewer shows one scene;
                training batches normally attach no surface at all. Choosing
                explicitly avoids silently skinning env 0 of a large batch.

        Raises:
            IndexError: If *env* is outside ``[0, num_envs)``.
        """
        if not 0 <= int(env) < self.num_envs:
            raise IndexError(f"env {env} out of range for num_envs={self.num_envs}")
        self._skin_env = int(env)

        edges_np = self.edges.numpy()
        if self.num_envs > 1:
            begin = self._skin_env * self.edges_per_env
            count = int(self.edge_counts[self._skin_env])
            edges_np = edges_np[begin : begin + count] - self._skin_env * self.nodes_per_env
            device = self.rod.device
            self._skin_x = wp.zeros(self.nodes_per_env, dtype=wp.vec3, device=device)
            self._skin_q = wp.zeros(max(self.segments_per_env, 1), dtype=wp.quat, device=device)
        self._sync_skin_views()

        skin_x, skin_q = self._skin_state()
        skinner = VesselSkinner(vertices, triangles, edges_np, device=self.rod.device)
        # Bind and capture rest state against the committed (rest) centerline.
        skinner.bind(skin_x)
        skinner.compute_rest_coordinates(None, None, skin_x, skin_q)
        self.skinner = skinner
        self.update_surface()
        return skinner

    def _skin_state(self) -> tuple["wp.array", "wp.array"]:
        """Centerline state feeding the skinner: the whole rod, or one env's slice."""
        if self.num_envs == 1:
            return self.rod.x, self.rod.q
        return self._skin_x, self._skin_q

    def _sync_skin_views(self) -> None:
        if self.num_envs == 1 or getattr(self, "_skin_x", None) is None:
            return
        env = self._skin_env
        wp.copy(
            self._skin_x, self.rod.x,
            dest_offset=0, src_offset=env * self.nodes_per_env, count=self.nodes_per_env,
        )
        if self.segments_per_env > 0:
            wp.copy(
                self._skin_q, self.rod.q,
                dest_offset=0, src_offset=env * self.segments_per_env,
                count=self.segments_per_env,
            )

    def update_surface(self) -> None:
        """Re-skin the attached surface from the committed centerline state."""
        if self.skinner is None:
            return
        self._sync_skin_views()
        skin_x, skin_q = self._skin_state()
        self.skinner.update(skin_x, skin_q)

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
        """Contain catheter predicted positions against live centerline tubes.

        Also refreshes :attr:`contact_depth` and :attr:`contact_count`, which
        describe contact during this call only and accumulate across its
        collision iterations. A caller wanting a wider window, such as a whole
        policy step, aggregates across calls itself.
        """
        self._ensure_catheter_buffers(num_points, str(catheter_predicted.device))
        device = catheter_predicted.device
        self._contact_depth.zero_()
        self._contact_count.zero_()

        num_envs = self.num_envs
        if num_envs > 1:
            if num_points % num_envs != 0:
                raise ValueError(
                    f"catheter num_points {num_points} is not divisible by the vessel's "
                    f"num_envs {num_envs}"
                )
            points_per_env = num_points // num_envs
            # Catheter edges, not vessel edges (self.edges_per_env): the sample grid
            # is one slot per catheter point plus interior samples along each of its
            # edges, while self.edges_per_env sizes the vessel loop inside the kernel.
            cath_edges_per_env = max(num_edges, 0) // num_envs
            sample_dim = (num_envs, points_per_env + CENTERLINE_EDGE_SAMPLES * cath_edges_per_env)
        else:
            sample_dim = num_points + CENTERLINE_EDGE_SAMPLES * max(num_edges, 0)

        for _ in range(self.collision_iterations):
            self._catheter_corrections.zero_()
            self._catheter_counts.zero_()
            self._vessel_corrections.zero_()
            self._vessel_counts.zero_()
            if num_envs > 1:
                per_env = self.per_env_vessels
                wp.launch(
                    project_centerline_containment_per_env_kernel
                    if per_env
                    else project_centerline_containment_batched_kernel,
                    dim=sample_dim,
                    inputs=[
                        catheter_predicted,
                        catheter_inv_mass,
                        int(points_per_env),
                        self.rod.p,
                        self.edges,
                        self.radii,
                        self.rod.inv_mass,
                        int(self.nodes_per_env),
                        int(self.edges_per_env),
                    ]
                    + (
                        [self._edge_counts_wp, self._open_roots_wp, self._open_root_neighbors_wp]
                        if per_env
                        else [int(self.open_root), int(self.open_root_neighbor)]
                    )
                    + [
                        float(self.catheter_radius),
                        float(self.max_distance),
                        float(self.interior_deadband),
                        float(self.interior_stiffness),
                        int(1 if self.two_way else 0),
                        float(self.vessel_response),
                        self._catheter_corrections,
                        self._catheter_counts,
                        self._vessel_corrections,
                        self._vessel_counts,
                        self._contact_depth,
                        self._contact_count,
                    ],
                    device=device,
                )
            else:
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
                        float(self.interior_deadband),
                        float(self.interior_stiffness),
                        int(1 if self.two_way else 0),
                        float(self.vessel_response),
                        self._catheter_corrections,
                        self._catheter_counts,
                        self._vessel_corrections,
                        self._vessel_counts,
                        self._contact_depth,
                        self._contact_count,
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
    def positions_per_env(self) -> np.ndarray:
        """Vessel node positions shaped ``(num_envs, nodes_per_env, 3)``."""
        return self.rod.x.numpy().reshape((self.num_envs, self.nodes_per_env, 3))

    @property
    def predicted_positions(self) -> np.ndarray:
        return self.rod.p.numpy()

    # -------------------------------------------------------------------- reset
    def reset(self, env_ids=None) -> None:
        """Restore the listed envs' vessels to rest, at rest.

        Node positions and segment frames return to their construction-time rest
        state, all vessel velocity is cleared, and the listed envs' contact
        accumulators are zeroed. Envs not listed keep their current deformation
        *and* their contact history, which is what lets an RL trainer recycle one
        env without disturbing its neighbours' observations.

        Args:
            env_ids: Env indices to reset, in any container the rod solver
                accepts, since it forwards the caller's ids here unchanged.
                ``None`` resets every env.

        Raises:
            IndexError: If any id is outside ``[0, num_envs)``.
        """
        if env_ids is None:
            ids = np.arange(self.num_envs, dtype=np.int32)
        else:
            raw = to_host_ids(env_ids)
            if raw.size == 0:
                return
            if raw.min() < 0 or raw.max() >= self.num_envs:
                raise IndexError(
                    f"env_ids out of range for num_envs={self.num_envs}: "
                    f"min={int(raw.min())}, max={int(raw.max())}"
                )
            ids = np.unique(raw).astype(np.int32)

        device = self.rod.device
        self._env_ids_host.numpy()[: ids.size] = ids
        wp.copy(self._env_ids_scratch, self._env_ids_host, count=ids.size)
        wp.launch(
            reset_centerline_nodes_kernel,
            dim=(ids.size, self.nodes_per_env),
            inputs=[
                self._env_ids_scratch,
                int(self.nodes_per_env),
                self._rest_x,
                self.rod.x,
                self.rod.p,
                self.rod.v,
            ],
            device=device,
        )
        if self.segments_per_env > 0:
            wp.launch(
                reset_centerline_segments_kernel,
                dim=(ids.size, self.segments_per_env),
                inputs=[
                    self._env_ids_scratch,
                    int(self.segments_per_env),
                    self._rest_q,
                    self.rod.q,
                    self.rod.u,
                    self.rod.omega,
                ],
                device=device,
            )
        # A reset env has no contact history, but envs that were *not* reset keep
        # theirs: a trainer reads observations between reset() and the next step,
        # so wiping the whole accumulator would report zero wall contact for
        # episodes that are still touching the wall.
        wp.launch(
            reset_centerline_contact_kernel,
            dim=ids.size,
            inputs=[self._env_ids_scratch, self._contact_depth, self._contact_count],
            device=device,
        )
        self.update_surface()

    # ------------------------------------------------------------ contact state
    @property
    def contact_depth(self) -> float:
        """Deepest wall penetration in metres during the last containment call.

        Across the batch; see :attr:`contact_depth_per_env` to attribute it.
        """
        return float(self._contact_depth.numpy().max())

    @property
    def contact_depth_per_env(self) -> np.ndarray:
        """Deepest wall penetration per env, shape ``(num_envs,)``."""
        return self._contact_depth.numpy().copy()

    @property
    def contact_count(self) -> int:
        """Contacting samples during the last containment call.

        Samples are counted once per collision iteration, so this measures how
        widely and how persistently the tool met the wall rather than how many
        distinct nodes touched it. Summed over the batch.
        """
        return int(self._contact_count.numpy().sum())

    @property
    def contact_count_per_env(self) -> np.ndarray:
        """Contacting samples per env, shape ``(num_envs,)``."""
        return self._contact_count.numpy().copy()

    @property
    def contact_buffers(self) -> tuple["wp.array", "wp.array"]:
        """Depth and count accumulators for callers building tensor views.

        Both hold one entry per env, so a trainer can map them straight onto a
        per-env observation or termination signal without a host round-trip.
        """
        return self._contact_depth, self._contact_count


__all__ = ["CenterlineDynamicsParams", "CenterlineVesselRuntime", "VesselSkinner"]
