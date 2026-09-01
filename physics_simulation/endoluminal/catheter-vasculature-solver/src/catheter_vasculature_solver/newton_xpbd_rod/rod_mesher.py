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

"""Rod tube mesh generator using Hermite interpolation and parallel transport frames.

Generates smooth tube meshes from rod centerline positions. Supports both
single-rod and batched multi-rod updates in a single GPU kernel launch.

The mesh topology (triangles, UVs) is created once; only vertex positions
and normals are updated each frame.
"""

from __future__ import annotations

import numpy as np
import warp as wp


# ---------------------------------------------------------------------------
# Warp helper functions
# ---------------------------------------------------------------------------

@wp.func
def _basis_from_direction(w: wp.vec3) -> wp.mat33:
    wn = wp.normalize(w)
    if wp.abs(wn[0]) > wp.abs(wn[1]):
        inv = 1.0 / wp.sqrt(wn[0] * wn[0] + wn[2] * wn[2] + 1e-10)
        u = wp.vec3(-wn[2] * inv, 0.0, wn[0] * inv)
    else:
        inv = 1.0 / wp.sqrt(wn[1] * wn[1] + wn[2] * wn[2] + 1e-10)
        u = wp.vec3(0.0, wn[2] * inv, -wn[1] * inv)
    v = wp.cross(wn, u)
    return wp.mat33(u[0], v[0], wn[0],
                    u[1], v[1], wn[1],
                    u[2], v[2], wn[2])


@wp.func
def _axis_angle_rotation(angle: float, axis: wp.vec3) -> wp.mat33:
    a = wp.normalize(axis)
    s = wp.sin(angle)
    c = wp.cos(angle)
    return wp.mat33(
        a[0]*a[0]+(1.0-a[0]*a[0])*c, a[0]*a[1]*(1.0-c)-a[2]*s, a[0]*a[2]*(1.0-c)+a[1]*s,
        a[0]*a[1]*(1.0-c)+a[2]*s,    a[1]*a[1]+(1.0-a[1]*a[1])*c, a[1]*a[2]*(1.0-c)-a[0]*s,
        a[0]*a[2]*(1.0-c)-a[1]*s,    a[1]*a[2]*(1.0-c)+a[0]*s, a[2]*a[2]+(1.0-a[2]*a[2])*c,
    )


@wp.func
def _hermite_pos(p1: wp.vec3, p2: wp.vec3, m1: wp.vec3, m2: wp.vec3, t: float) -> wp.vec3:
    t2 = t * t
    t3 = t2 * t
    return p1*(1.0 - 3.0*t2 + 2.0*t3) + p2*t2*(3.0 - 2.0*t) + m1*(t3 - 2.0*t2 + t) + m2*t2*(t - 1.0)


@wp.func
def _hermite_tan(p1: wp.vec3, p2: wp.vec3, m1: wp.vec3, m2: wp.vec3, t: float) -> wp.vec3:
    t2 = t * t
    return p1*(6.0*t2 - 6.0*t) + p2*(-6.0*t2 + 6.0*t) + m1*(3.0*t2 - 4.0*t + 1.0) + m2*(3.0*t2 - 2.0*t)


@wp.func
def _mesh_one_rod(
    positions: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.float32),
    vertices: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    frames: wp.array(dtype=wp.float32),
    pos_offset: int,
    radius_offset: int,
    vert_offset: int,
    frame_offset: int,
    num_points: int,
    resolution: int,
    smoothing: int,
    radius: float,
    use_point_radii: bool,
):
    """Generate tube mesh for one rod. Shared by single and batched kernels."""
    if num_points < 2:
        return

    # Initialise Bishop frame from first edge
    w = wp.normalize(positions[pos_offset + 1] - positions[pos_offset])
    fm = _basis_from_direction(w)
    # Store frame columns at frame_offset (9 floats per rod)
    fo = frame_offset
    frames[fo + 0] = fm[0, 0]
    frames[fo + 1] = fm[1, 0]
    frames[fo + 2] = fm[2, 0]
    frames[fo + 3] = fm[0, 1]
    frames[fo + 4] = fm[1, 1]
    frames[fo + 5] = fm[2, 1]
    frames[fo + 6] = fm[0, 2]
    frames[fo + 7] = fm[1, 2]
    frames[fo + 8] = fm[2, 2]

    v_id = vert_offset
    for i in range(num_points - 1):
        a = wp.max(i - 1, 0)
        b = i
        c = wp.min(i + 1, num_points - 1)
        d = wp.min(i + 2, num_points - 1)

        p1 = positions[pos_offset + b]
        p2 = positions[pos_offset + c]
        m1 = 0.5 * (positions[pos_offset + c] - positions[pos_offset + a])
        m2 = 0.5 * (positions[pos_offset + d] - positions[pos_offset + b])

        segs = smoothing
        if i >= num_points - 2:
            segs = smoothing + 1

        for s in range(segs):
            t = float(s) / float(smoothing)
            pos = _hermite_pos(p1, p2, m1, m2, t)
            tan = wp.normalize(_hermite_tan(p1, p2, m1, m2, t))
            ring_radius = radius
            if use_point_radii:
                ring_radius = (
                    (1.0 - t) * radii[radius_offset + b]
                    + t * radii[radius_offset + c]
                )

            cur = wp.vec3(frames[fo + 6], frames[fo + 7], frames[fo + 8])
            dot_val = wp.clamp(wp.dot(cur, tan), -1.0, 1.0)
            ang = wp.acos(dot_val)
            if wp.abs(ang) > 0.001:
                ax = wp.cross(cur, tan)
                ax_len = wp.length(ax)
                if ax_len > 1e-10:
                    rot = _axis_angle_rotation(ang, ax / ax_len)
                    u_f = wp.vec3(frames[fo + 0], frames[fo + 1], frames[fo + 2])
                    v_f = wp.vec3(frames[fo + 3], frames[fo + 4], frames[fo + 5])
                    w_f = wp.vec3(frames[fo + 6], frames[fo + 7], frames[fo + 8])
                    u_n = rot * u_f
                    v_n = rot * v_f
                    w_n = rot * w_f
                    frames[fo + 0] = u_n[0]
                    frames[fo + 1] = u_n[1]
                    frames[fo + 2] = u_n[2]
                    frames[fo + 3] = v_n[0]
                    frames[fo + 4] = v_n[1]
                    frames[fo + 5] = v_n[2]
                    frames[fo + 6] = w_n[0]
                    frames[fo + 7] = w_n[1]
                    frames[fo + 8] = w_n[2]

            for cc in range(resolution):
                theta = 2.0 * 3.14159265359 * float(cc) / float(resolution)
                u_ax = wp.vec3(frames[fo + 0], frames[fo + 1], frames[fo + 2])
                v_ax = wp.vec3(frames[fo + 3], frames[fo + 4], frames[fo + 5])
                wd = u_ax * wp.cos(theta) + v_ax * wp.sin(theta)
                vertices[v_id] = wd * ring_radius + pos
                normals[v_id] = wd
                v_id = v_id + 1


# ---------------------------------------------------------------------------
# Single-rod kernel (backward compat)
# ---------------------------------------------------------------------------

@wp.kernel
def _update_mesh_kernel(
    positions: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.float32),
    vertices: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    frame: wp.array(dtype=wp.float32),
    num_points: int,
    resolution: int,
    smoothing: int,
    radius: float,
    use_point_radii: bool,
):
    tid = wp.tid()
    if tid != 0:
        return
    _mesh_one_rod(
        positions, radii, vertices, normals, frame,
        0, 0, 0, 0, num_points, resolution, smoothing, radius, use_point_radii,
    )


# ---------------------------------------------------------------------------
# Batched kernel — one thread per rod
# ---------------------------------------------------------------------------

@wp.kernel
def _update_mesh_batched_kernel(
    positions: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.float32),
    vertices: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    frames: wp.array(dtype=wp.float32),
    pos_offsets: wp.array(dtype=wp.int32),
    radius_offsets: wp.array(dtype=wp.int32),
    vert_offsets: wp.array(dtype=wp.int32),
    num_points_arr: wp.array(dtype=wp.int32),
    resolution: int,
    smoothing: int,
    radius: float,
    use_point_radii: bool,
):
    rod_idx = wp.tid()
    _mesh_one_rod(
        positions, radii, vertices, normals, frames,
        pos_offsets[rod_idx],
        radius_offsets[rod_idx],
        vert_offsets[rod_idx],
        rod_idx * 9,
        num_points_arr[rod_idx],
        resolution, smoothing, radius, use_point_radii,
    )


# ---------------------------------------------------------------------------
# Public classes
# ---------------------------------------------------------------------------

def _build_topology(num_rings: int, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Build static index + UV arrays for one tube."""
    num_verts = num_rings * resolution
    num_tris = (num_rings - 1) * resolution * 2 if num_rings > 1 else 0

    indices = np.zeros(num_tris * 3, dtype=np.int32)
    uvs = np.zeros((num_verts, 2), dtype=np.float32)

    for r in range(num_rings):
        for v in range(resolution):
            vid = r * resolution + v
            uvs[vid, 0] = v / float(resolution)
            uvs[vid, 1] = r / float(num_rings - 1) if num_rings > 1 else 0.0

    tri = 0
    for r in range(1, num_rings):
        base = r * resolution
        for v in range(resolution):
            cur = base + v
            nxt = base + (v + 1) % resolution
            indices[tri * 3] = cur
            indices[tri * 3 + 1] = cur - resolution
            indices[tri * 3 + 2] = nxt - resolution
            tri += 1
            indices[tri * 3] = nxt - resolution
            indices[tri * 3 + 1] = nxt
            indices[tri * 3 + 2] = cur
            tri += 1

    return indices, uvs


class RodMesher:
    """Generate a tube mesh from a single rod's particle positions.

    Args:
        num_points: Number of rod control points (particles).
        radius: Tube radius [m].
        resolution: Vertices around the circumference.
        smoothing: Hermite subdivisions between consecutive control points.
        device: Warp device for GPU arrays.
        point_radii: Optional radius at each control point. Values are linearly
            interpolated across the generated rings.
    """

    def __init__(
        self,
        num_points: int,
        radius: float = 0.02,
        resolution: int = 8,
        smoothing: int = 3,
        device: wp.Device | None = None,
        point_radii: list[float] | np.ndarray | None = None,
    ):
        self.num_points = num_points
        self.radius = radius
        self.resolution = resolution
        self.smoothing = smoothing
        self.device = device or wp.get_device()
        if point_radii is not None:
            point_radii = np.asarray(point_radii, dtype=np.float32)
            if point_radii.shape != (num_points,) or not np.all(np.isfinite(point_radii)):
                raise ValueError(f"point_radii must contain {num_points} finite values")
            if np.any(point_radii <= 0.0):
                raise ValueError("point_radii values must be greater than zero")
        self._use_point_radii = point_radii is not None

        self.num_rings = (num_points - 1) * smoothing + 1 if num_points >= 2 else 0
        self.num_vertices = self.num_rings * resolution
        self.num_triangles = (self.num_rings - 1) * resolution * 2 if self.num_rings > 1 else 0

        indices, uvs = _build_topology(self.num_rings, resolution)

        self._vertices = wp.zeros(self.num_vertices, dtype=wp.vec3, device=self.device)
        self._normals = wp.zeros(self.num_vertices, dtype=wp.vec3, device=self.device)
        self._indices = wp.array(indices, dtype=wp.int32, device=self.device)
        self._uvs = wp.array(uvs, dtype=wp.vec2, device=self.device)
        self._frame = wp.zeros(9, dtype=wp.float32, device=self.device)
        self._radii = wp.array(
            np.zeros(num_points, dtype=np.float32) if point_radii is None else point_radii,
            dtype=wp.float32,
            device=self.device,
        )

    def update(self, positions_wp: wp.array) -> None:
        """Recompute vertices/normals from current rod positions (GPU)."""
        wp.launch(
            _update_mesh_kernel,
            dim=1,
            inputs=[
                positions_wp,
                self._radii,
                self._vertices,
                self._normals,
                self._frame,
                self.num_points,
                self.resolution,
                self.smoothing,
                self.radius,
                self._use_point_radii,
            ],
            device=self.device,
        )

    @property
    def vertices(self) -> wp.array:
        return self._vertices

    @property
    def normals(self) -> wp.array:
        return self._normals

    @property
    def indices(self) -> wp.array:
        return self._indices

    @property
    def uvs(self) -> wp.array:
        return self._uvs


class BatchedRodMesher:
    """Generate tube meshes for multiple rods in a single GPU kernel launch.

    All rods share the same resolution and smoothing. ``num_points`` preserves
    the original uniform-rod API, while ``point_counts`` allows each rod/path
    to contain a different number of control points. Rods may use one shared
    radius or per-control-point radii.
    Positions are read from a single contiguous array (e.g. ``state.particle_q``)
    using per-rod offsets.

    Args:
        num_rods: Number of rods.
        num_points: Number of control points per rod when ``point_counts`` is
            omitted.
        radius: Tube radius [m].
        resolution: Vertices around the circumference.
        smoothing: Hermite subdivisions between consecutive control points.
        particle_offsets: Per-rod start index into the positions array.
        point_counts: Optional per-rod control-point counts.
        device: Warp device for GPU arrays.
        point_radii: Optional flattened per-rod control-point radii.
    """

    def __init__(
        self,
        num_rods: int,
        num_points: int,
        radius: float = 0.02,
        resolution: int = 8,
        smoothing: int = 3,
        particle_offsets: list[int] | np.ndarray | None = None,
        point_counts: list[int] | np.ndarray | None = None,
        device: wp.Device | None = None,
        point_radii: list[float] | np.ndarray | None = None,
    ):
        self.num_rods = num_rods
        if point_counts is None:
            point_counts = np.full(num_rods, num_points, dtype=np.int32)
        else:
            point_counts = np.asarray(point_counts, dtype=np.int32)
            if point_counts.shape != (num_rods,):
                raise ValueError(f"point_counts must contain {num_rods} values")
        if np.any(point_counts < 2):
            raise ValueError("point_counts values must be at least two")
        self.point_counts = point_counts
        self.num_points = num_points
        self.radius = radius
        self.resolution = resolution
        self.smoothing = smoothing
        self.device = device or wp.get_device()
        total_points = int(np.sum(point_counts))
        if point_radii is not None:
            point_radii = np.asarray(point_radii, dtype=np.float32)
            if point_radii.shape != (total_points,) or not np.all(np.isfinite(point_radii)):
                raise ValueError(f"point_radii must contain {total_points} finite values")
            if np.any(point_radii <= 0.0):
                raise ValueError("point_radii values must be greater than zero")
        self._use_point_radii = point_radii is not None

        num_rings = (point_counts - 1) * smoothing + 1
        verts_per_rod = num_rings * resolution
        uniform_topology = bool(num_rods and np.all(num_rings == num_rings[0]))
        self.num_rings = int(num_rings[0]) if uniform_topology else None
        self._verts_per_rod = int(verts_per_rod[0]) if uniform_topology else None
        vert_offsets = (
            np.concatenate(
                (
                    np.zeros(1, dtype=np.int32),
                    np.cumsum(verts_per_rod[:-1], dtype=np.int32),
                )
            )
            if num_rods
            else np.empty(0, dtype=np.int32)
        )
        total_verts = int(np.sum(verts_per_rod))

        self._vertices = wp.zeros(total_verts, dtype=wp.vec3, device=self.device)
        self._normals = wp.zeros(total_verts, dtype=wp.vec3, device=self.device)
        per_rod_topology = [
            _build_topology(int(rings), resolution) for rings in num_rings
        ]
        self._base_indices_per_rod = [
            wp.array(indices, dtype=wp.int32, device=self.device)
            for indices, _ in per_rod_topology
        ]
        self._base_uvs_per_rod = [
            wp.array(uvs, dtype=wp.vec2, device=self.device)
            for _, uvs in per_rod_topology
        ]
        combined_indices = (
            np.concatenate(
                [
                    indices + int(vert_offsets[rod])
                    for rod, (indices, _) in enumerate(per_rod_topology)
                ]
            )
            if num_rods
            else np.empty(0, dtype=np.int32)
        )
        combined_uvs = (
            np.concatenate([uvs for _, uvs in per_rod_topology], axis=0)
            if num_rods
            else np.empty((0, 2), dtype=np.float32)
        )
        self._indices = wp.array(combined_indices, dtype=wp.int32, device=self.device)
        self._uvs = wp.array(combined_uvs, dtype=wp.vec2, device=self.device)
        self._frames = wp.zeros(num_rods * 9, dtype=wp.float32, device=self.device)
        self._radii = wp.array(
            np.zeros(total_points, dtype=np.float32) if point_radii is None else point_radii,
            dtype=wp.float32,
            device=self.device,
        )

        # Per-rod offsets into positions and vertex arrays
        if particle_offsets is None:
            particle_offsets = (
                np.concatenate(
                    (
                        np.zeros(1, dtype=np.int32),
                        np.cumsum(point_counts[:-1], dtype=np.int32),
                    )
                )
                if num_rods
                else np.empty(0, dtype=np.int32)
            )
        particle_offsets = np.asarray(particle_offsets, dtype=np.int32)
        if particle_offsets.shape != (num_rods,):
            raise ValueError(f"particle_offsets must contain {num_rods} values")
        self._pos_offsets = wp.array(
            particle_offsets, dtype=wp.int32, device=self.device
        )
        radius_offsets = (
            np.concatenate(
                (
                    np.zeros(1, dtype=np.int32),
                    np.cumsum(point_counts[:-1], dtype=np.int32),
                )
            )
            if num_rods
            else np.empty(0, dtype=np.int32)
        )
        self._radius_offsets = wp.array(
            radius_offsets, dtype=wp.int32, device=self.device
        )
        self._vert_offsets = wp.array(vert_offsets, dtype=wp.int32, device=self.device)
        self._num_points_arr = wp.array(
            point_counts, dtype=wp.int32, device=self.device
        )

        # Per-rod vertex/normal views (zero-copy slices into the big arrays)
        self._per_rod_vertices = []
        self._per_rod_normals = []
        for r in range(num_rods):
            v0 = int(vert_offsets[r])
            v1 = v0 + int(verts_per_rod[r])
            self._per_rod_vertices.append(self._vertices[v0:v1])
            self._per_rod_normals.append(self._normals[v0:v1])

    def update(self, positions_wp: wp.array) -> None:
        """Recompute all rod meshes from positions in one kernel launch."""
        wp.launch(
            _update_mesh_batched_kernel,
            dim=self.num_rods,
            inputs=[
                positions_wp,
                self._radii,
                self._vertices,
                self._normals,
                self._frames,
                self._pos_offsets,
                self._radius_offsets,
                self._vert_offsets,
                self._num_points_arr,
                self.resolution,
                self.smoothing,
                self.radius,
                self._use_point_radii,
            ],
            device=self.device,
        )

    def set_point_radii(self, point_radii: list[float] | np.ndarray) -> None:
        """Update the per-control-point radii used by subsequent updates."""
        total_points = int(np.sum(self.point_counts))
        point_radii = np.asarray(point_radii, dtype=np.float32)
        if point_radii.shape != (total_points,) or not np.all(np.isfinite(point_radii)):
            raise ValueError(f"point_radii must contain {total_points} finite values")
        if np.any(point_radii <= 0.0):
            raise ValueError("point_radii values must be greater than zero")
        self._radii.assign(point_radii)
        self._use_point_radii = True

    def rod_vertices(self, rod_idx: int) -> wp.array:
        return self._per_rod_vertices[rod_idx]

    def rod_normals(self, rod_idx: int) -> wp.array:
        return self._per_rod_normals[rod_idx]

    def rod_indices(self, rod_idx: int = 0) -> wp.array:
        """Return the rod-local, zero-based index buffer."""
        return self._base_indices_per_rod[rod_idx]

    def rod_uvs(self, rod_idx: int = 0) -> wp.array:
        """Return the rod-local UV buffer."""
        return self._base_uvs_per_rod[rod_idx]

    @property
    def vertices(self) -> wp.array:
        return self._vertices

    @property
    def normals(self) -> wp.array:
        return self._normals

    @property
    def indices(self) -> wp.array:
        return self._indices

    @property
    def uvs(self) -> wp.array:
        return self._uvs
