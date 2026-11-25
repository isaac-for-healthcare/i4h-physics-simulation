# Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Ported from Newton Physics Engine — upstream catheter solver.
#
# Modifications for Isaac Lab integration:
#   Copyright (c) 2022-2025, The Isaac Lab Project Developers
#   SPDX-License-Identifier: BSD-3-Clause
#
# Vendored/adapted from the Isaac Lab Newton rod solver stack for
# catheter_navigation integration.

"""XPBD rod solver with catheter-in-vessel mesh collision.

This module provides :class:`CathRodSolver`, a subclass of
:class:`~catheter_vasculature_solver.XPBDRodSolver` that adds three capabilities
needed for realistic catheter-in-vessel simulation:

**Vessel containment (two collision paths)**

*SDF path (default, cheapest):*
Each rod particle is queried against a :class:`warp.Mesh` BVH via
``wp.mesh_query_point_sign_normal``.  Particles that penetrate the vessel
surface are projected back to a configurable clearance offset
``target_phi`` (typically ``-particle_radius``).

*AABB / mesh-edge path (more robust for tight curvature):*
An AABB broadphase finds all nearby triangles; vertex-vs-triangle and
rod-segment-vs-mesh-edge contacts are resolved into a shared correction
buffer and flushed to ``predicted_positions`` after each Gauss-Seidel
iteration.

**Track-guided insertion**
Non-tip rod particles are gently projected onto a fixed linear insertion
axis (``track_start``, ``track_dir``, ``track_length``), mimicking the
rigid catheter shaft sliding through a guide sheath while only the
steerable distal tip deforms freely.

**Configurable collision ordering**
Vessel containment can run *before* the XPBD constraint solve
(``collision_pre_constraints_enabled``), *after* (default), or both.
``collision_iterations`` controls how many Gauss-Seidel passes are done
per substep.

Usage::

    import numpy as np
    import warp as wp
    from catheter_vasculature_solver import CathRodSolver, RodConfig

    # Build a wp.Mesh from your vessel geometry
    verts = np.load("aorta_verts.npy")   # (N, 3) float32
    faces = np.load("aorta_faces.npy")   # (M, 3) int32
    vessel_mesh = wp.Mesh(
        points=wp.array(verts, dtype=wp.vec3, device="cuda"),
        indices=wp.array(faces.flatten(), dtype=int, device="cuda"),
    )

    cfg = RodConfig()
    solver = CathRodSolver(
        cfg,
        collision_mesh=vessel_mesh,
        track_start=np.array([0.0, 0.0, 0.0]),
        track_dir=np.array([1.0, 0.0, 0.0]),
        track_length=0.5,
        tip_num_edges=10,
        particle_radius=0.002,
        segment_length=cfg.geometry.segment_length,
    )
    for _ in range(300):
        solver.step(cfg.solver.dt)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from .host_arrays import to_host
from .xpbd_rod_solver import XPBDRodSolver, _Workspace

try:
    from .vessel_deformation.centerline_runtime import CenterlineVesselRuntime
except ImportError:  # pragma: no cover - optional deformation package
    CenterlineVesselRuntime = None  # type: ignore[misc, assignment]

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

COLLISION_PROJECTION_STAGE_PRE = "pre"
COLLISION_PROJECTION_STAGE_POST = "post"
_COLLISION_PROJECTION_STAGES = (
    COLLISION_PROJECTION_STAGE_PRE,
    COLLISION_PROJECTION_STAGE_POST,
)
DEFAULT_MESH_EDGE_MAX_TRIANGLES = 64


# ──────────────────────────────────────────────────────────────────────────────
# NumPy utility
# ──────────────────────────────────────────────────────────────────────────────


def compute_smooth_vertex_normals(vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Compute area-weighted vertex normals from triangle topology.

    Args:
        vertices: ``(N, 3)`` float32 array of vertex positions.
        indices:  ``(M, 3)`` int32 array of triangle indices.

    Returns:
        ``(N, 3)`` float32 array of unit vertex normals.
    """
    vertices = np.asarray(vertices, dtype=np.float32).reshape((-1, 3))
    triangles = np.asarray(indices, dtype=np.int32).reshape((-1, 3))

    normals = np.zeros_like(vertices, dtype=np.float32)
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0).astype(np.float32)

    np.add.at(normals, triangles[:, 0], face_normals)
    np.add.at(normals, triangles[:, 1], face_normals)
    np.add.at(normals, triangles[:, 2], face_normals)

    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1.0e-12
    normals[valid] /= lengths[valid, None]
    normals[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return normals


# ──────────────────────────────────────────────────────────────────────────────
# Warp helper functions
# ──────────────────────────────────────────────────────────────────────────────


@wp.func
def _mesh_face_normal(mesh_id: wp.uint64, face_index: int) -> wp.vec3:
    """Return the outward face normal for a triangle in *mesh_id*."""
    mesh = wp.mesh_get(mesh_id)
    i0 = mesh.indices[face_index * 3 + 0]
    i1 = mesh.indices[face_index * 3 + 1]
    i2 = mesh.indices[face_index * 3 + 2]
    v0 = mesh.points[i0]
    v1 = mesh.points[i1]
    v2 = mesh.points[i2]
    return wp.normalize(wp.cross(v1 - v0, v2 - v0))


@wp.func
def _safe_normalize(v: wp.vec3, fallback: wp.vec3) -> wp.vec3:
    length = wp.length(v)
    if length > 1.0e-8:
        return v / length
    return fallback


@wp.func
def _closest_point_triangle(point: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.vec3:
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = point - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b

    cp = point - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + ab * v

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + ac * w

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + (c - b) * w

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w


@wp.func
def _closest_point_triangle_barycentric(point: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.vec3:
    """Return barycentric coords ``(wa, wb, wc)`` of the closest point."""
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return wp.vec3(1.0, 0.0, 0.0)

    bp = point - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return wp.vec3(0.0, 1.0, 0.0)

    cp = point - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return wp.vec3(0.0, 0.0, 1.0)

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return wp.vec3(1.0 - v, v, 0.0)

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return wp.vec3(1.0 - w, 0.0, w)

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return wp.vec3(0.0, 1.0 - w, w)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return wp.vec3(1.0 - v - w, v, w)


@wp.func
def _mesh_barycentric_normal(
    mesh_id: wp.uint64,
    smooth_normals: wp.array(dtype=wp.vec3),
    face_index: int,
    weights: wp.vec3,
    fallback: wp.vec3,
) -> wp.vec3:
    mesh = wp.mesh_get(mesh_id)
    i0 = mesh.indices[face_index * 3 + 0]
    i1 = mesh.indices[face_index * 3 + 1]
    i2 = mesh.indices[face_index * 3 + 2]
    n = smooth_normals[i0] * weights.x + smooth_normals[i1] * weights.y + smooth_normals[i2] * weights.z
    length = wp.length(n)
    if length > 1.0e-8:
        return n / length
    return fallback


@wp.func
def _segment_segment_barycentric(a0: wp.vec3, a1: wp.vec3, b0: wp.vec3, b1: wp.vec3) -> wp.vec4:
    """Return ``(sa, 1-sa, sb, 1-sb)`` barycentric parameters for the closest
    points between segment ``[a0,a1]`` and segment ``[b0,b1]``."""
    da = a1 - a0
    db = b1 - b0
    r = a0 - b0
    a = wp.dot(da, da)
    e = wp.dot(db, db)
    f = wp.dot(db, r)

    sa = float(0.0)
    sb = float(0.0)

    if a <= 1.0e-10 and e <= 1.0e-10:
        return wp.vec4(1.0, 0.0, 1.0, 0.0)

    if a <= 1.0e-10:
        sa = 0.0
        sb = wp.clamp(f / e, 0.0, 1.0)
    else:
        c = wp.dot(da, r)
        if e <= 1.0e-10:
            sb = 0.0
            sa = wp.clamp(-c / a, 0.0, 1.0)
        else:
            b_coeff = wp.dot(da, db)
            denom = a * e - b_coeff * b_coeff
            if wp.abs(denom) > 1.0e-10:
                sa = wp.clamp((b_coeff * f - c * e) / denom, 0.0, 1.0)
            else:
                sa = 0.0
            sb = (b_coeff * sa + f) / e
            if sb < 0.0:
                sb = 0.0
                sa = wp.clamp(-c / a, 0.0, 1.0)
            elif sb > 1.0:
                sb = 1.0
                sa = wp.clamp((b_coeff - c) / a, 0.0, 1.0)

    return wp.vec4(sa, 1.0 - sa, sb, 1.0 - sb)


# ──────────────────────────────────────────────────────────────────────────────
# Warp kernels
# ──────────────────────────────────────────────────────────────────────────────


@wp.kernel
def _track_sliding_kernel(
    predicted_positions: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    track_starts: wp.array(dtype=wp.vec3),
    track_dirs: wp.array(dtype=wp.vec3),
    track_length: float,
    track_stiffness: float,
    end_idx: int,
    points_per_env: int,
):
    """Project non-tip rod particles toward the insertion axis (track).

    ``points_per_env`` makes the non-tip test env-local, so one launch covers a
    whole batch. Single-env callers pass ``num_points``, for which the modulo is
    the identity and behaviour is unchanged. ``track_starts`` and ``track_dirs``
    hold one axis per env, or a single axis every env shares.
    """
    i = wp.tid()
    if i % points_per_env >= end_idx:
        return
    if inv_masses[i] <= 0.0:
        return

    track = wp.min(i // points_per_env, track_starts.shape[0] - 1)
    track_start = track_starts[track]
    track_dir = track_dirs[track]
    p = predicted_positions[i]
    t = wp.clamp(wp.dot(p - track_start, track_dir), 0.0, track_length)
    proj = track_start + track_dir * t
    predicted_positions[i] = p + (proj - p) * track_stiffness


@wp.kernel
def _track_polyline_kernel(
    predicted_positions: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=float),
    track_path: wp.array(dtype=wp.vec3),
    path_offsets: wp.array(dtype=wp.int32),
    track_stiffness: float,
    end_idx: int,
    points_per_env: int,
):
    """Project non-tip rod particles toward a curved insertion path.

    The curved counterpart of :func:`_track_sliding_kernel`, for anatomy rather
    than a bench rail. A straight track is only usable where the vessel is
    straight; projecting onto a ray through a curved lumen would pull the wire
    out through the wall, which is why guidance is otherwise left off there.

    Why this is what stops the shaft bunching: containment is a one-sided radial
    test, so a particle anywhere inside the lumen is corrected by exactly
    nothing and is free to slide along the vessel and pile up against its
    neighbours. Removing the two lateral degrees of freedom leaves arc length as
    the only way the rod can move, and the distance constraints already resist
    compression along it -- so spacing is recovered by the existing solve rather
    than prescribed here. That only holds when this runs *before* the solve; run
    afterwards it is the last word on position, and the chords never re-equalize.

    The projection is a blend rather than a snap, so it argues with the elastic
    solve instead of overruling it, and ``end_idx`` leaves the distal tip free to
    deform and steer.

    ``track_path`` holds every route back to back: route ``r`` is the points
    ``[path_offsets[r], path_offsets[r + 1])``. There is one route per env, or a
    single route every env shares. ``points_per_env`` makes the non-tip test and
    the route lookup env-local so one launch covers a whole batch.
    """
    i = wp.tid()
    if i % points_per_env >= end_idx:
        return
    # Skips the root while a mount is prescribing it, whose inverse mass is
    # zeroed. Guidance must not fight the pose the holder asked for.
    if inv_masses[i] <= 0.0:
        return
    route = wp.min(i // points_per_env, path_offsets.shape[0] - 2)
    first = path_offsets[route]
    last = path_offsets[route + 1]
    if last - first < 2:
        return

    p = predicted_positions[i]
    closest = track_path[first]
    best = wp.dot(p - closest, p - closest)
    # Brute force over segments, as the containment kernel does over centerline
    # edges: a centerline resampled for a catheter is hundreds of points, and a
    # BVH would have to be refit every step once the vessel deforms.
    for segment in range(first, last - 1):
        a = track_path[segment]
        ab = track_path[segment + 1] - a
        span = wp.dot(ab, ab)
        t = float(0.0)
        if span > 1.0e-18:
            t = wp.clamp(wp.dot(p - a, ab) / span, 0.0, 1.0)
        q = a + ab * t
        d2 = wp.dot(p - q, p - q)
        if d2 < best:
            best = d2
            closest = q

    predicted_positions[i] = p + (closest - p) * track_stiffness


@wp.kernel
def _clear_vec3_kernel(values: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    values[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _set_root_position_kernel(
    positions: wp.array(dtype=wp.vec3),
    predicted_positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    new_pos: wp.vec3,
):
    """Kinematically write the root particle to both position buffers and zero its velocity.

    Zeroing velocities[0] prevents any residual angular-momentum ghost from
    persisting when the root is stationary between W presses.
    """
    tid = wp.tid()
    if tid != 0:
        return
    positions[0] = new_pos
    predicted_positions[0] = new_pos
    velocities[0] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _distance_cleanup_kernel(
    predicted_positions: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=float),
    rest_lengths: wp.array(dtype=float),
    points_per_env: int,
    edges_per_env: int,
    color: int,
    relaxation: float,
):
    """Restore edge lengths after a containment projection has disturbed them.

    Containment runs after the constraint solve because that is the only side of
    it where a projection persists, but it is then also the last word on
    position and overrides the stretch constraints the solve had satisfied.
    Particles slide radially and pile up along the vessel: measured chords ran
    from 6% to 362% of rest, which is the bunching and the long gaps seen in the
    viewport. This is the missing third step -- re-satisfy the distances *after*
    containment, so both get to hold.

    Containment corrections are radial and small, so restoring length mostly
    moves particles along the vessel rather than back through its wall, which is
    what makes this recoverable rather than a second tug of war.

    Runs as two-colour Gauss-Seidel: even edges then odd, so no two threads write
    the same particle while each sweep still sees its neighbour's result. Jacobi
    would be race-free too but propagates a correction one edge per iteration,
    and a 40-edge chain does not have that many to spare.

    Args:
        predicted_positions: Particle positions to correct, modified in place.
        inv_masses: Per-particle inverse mass; zero pins a particle, which is how
            a prescribed root stays where its mount put it.
        rest_lengths: Per-edge target length.
        points_per_env: Particles per rod, to index within an environment.
        edges_per_env: Edges per rod.
        color: ``0`` for even edges, ``1`` for odd.
        relaxation: Fraction of each correction to apply, in ``(0, 1]``.
    """
    env, e = wp.tid()
    if e % 2 != color:
        return
    i = env * points_per_env + e
    j = i + 1
    w1 = inv_masses[i]
    w2 = inv_masses[j]
    total = w1 + w2
    # Both ends pinned: no correction is possible and the constraint is the
    # caller's to have made satisfiable.
    if total <= 0.0:
        return
    delta = predicted_positions[j] - predicted_positions[i]
    dist = wp.length(delta)
    # A fully collapsed edge has no direction to separate along. Leaving it is
    # safe because its neighbours still have one, and the next sweep sees a
    # non-degenerate edge again.
    if dist < 1.0e-12:
        return
    direction = delta / dist
    correction = relaxation * (dist - rest_lengths[env * edges_per_env + e]) / total
    predicted_positions[i] = predicted_positions[i] + direction * (w1 * correction)
    predicted_positions[j] = predicted_positions[j] - direction * (w2 * correction)


@wp.kernel
def _update_tip_rest_darboux_kernel(
    rest_darboux: wp.array(dtype=wp.vec3),
    baseline: wp.array(dtype=wp.vec3),
    num_edges: int,
    tip_num_edges: int,
    bend_angle: wp.array(dtype=wp.float32),
):
    """Encode a steerable pre-bend into the bend constraint's REST shape.

    The bend is added to *baseline* rather than written over the buffer, so
    whatever rest shape was authored for the body survives being steered. See
    :meth:`CathRodSolver.set_tip_bend` for the ``2n - 1`` and for why the bend
    is added rather than assigned.
    """
    e = wp.tid()
    if e >= num_edges:
        return
    tip_start = num_edges - tip_num_edges
    if e >= tip_start:
        per_edge = wp.sin(bend_angle[0] / float(2 * tip_num_edges - 1))
        rest_darboux[e] = baseline[e] + wp.vec3(per_edge, 0.0, 0.0)
    else:
        rest_darboux[e] = baseline[e]


@wp.kernel
def _update_tip_rest_darboux_batched_kernel(
    rest_darboux: wp.array(dtype=wp.vec3),
    baseline: wp.array(dtype=wp.vec3),
    edges_per_env: int,
    tip_num_edges: int,
    bend_angle: wp.array(dtype=wp.float32),
):
    """Per-env counterpart: each env steers its own tip.

    ``bend_angle`` is indexed by env, so a batched rollout can hold a different
    tip shape per environment. A shared angle is the degenerate case of filling
    the array.
    """
    env, e = wp.tid()
    tip_start = edges_per_env - tip_num_edges
    i = env * edges_per_env + e
    if e >= tip_start:
        per_edge = wp.sin(bend_angle[env] / float(2 * tip_num_edges - 1))
        rest_darboux[i] = baseline[i] + wp.vec3(per_edge, 0.0, 0.0)
    else:
        rest_darboux[i] = baseline[i]


@wp.kernel
def _set_root_position_batched_kernel(
    positions: wp.array(dtype=wp.vec3),
    predicted_positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    points_per_env: int,
    env_ids: wp.array(dtype=wp.int32),
    new_pos: wp.array(dtype=wp.vec3),
):
    """Kinematically write each listed env's root particle and zero its velocity."""
    slot = wp.tid()
    i = env_ids[slot] * points_per_env
    p = new_pos[slot]
    positions[i] = p
    predicted_positions[i] = p
    velocities[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _project_vessel_containment_kernel(
    current_positions: wp.array(dtype=wp.vec3),
    predicted_positions: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    smooth_normals: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    sign_scale: float,
    use_smooth_normals: bool,
    target_phi: float,
    max_dist: float,
):
    """Project predicted positions to the vessel interior signed offset.

    Uses ``wp.mesh_query_point_sign_normal`` (BVH SDF query) to determine
    whether each particle is inside the vessel and how far from the wall it is.
    Particles that violate the ``target_phi`` clearance are projected back.
    The velocity clamp at the end ensures the derived velocity (computed as
    ``(predicted - current) / dt`` during integration) does not point outward.
    """
    i = wp.tid()
    if inv_masses[i] <= 0.0:
        return

    pos = predicted_positions[i]

    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)

    if not wp.mesh_query_point_sign_normal(mesh_id, pos, max_dist, sign, face_index, face_u, face_v):
        return

    closest = wp.mesh_eval_position(mesh_id, face_index, face_u, face_v)
    diff = pos - closest
    dist = wp.length(diff)

    side = sign * sign_scale
    if wp.abs(side) < 1.0e-6:
        side = sign_scale

    grad = wp.vec3(0.0, 0.0, 0.0)
    if use_smooth_normals:
        weights = wp.vec3(1.0 - face_u - face_v, face_u, face_v)
        face_normal = _mesh_face_normal(mesh_id, face_index)
        grad = _mesh_barycentric_normal(mesh_id, smooth_normals, face_index, weights, face_normal) * sign_scale
        if dist > 1.0e-8 and wp.dot(grad, diff) * side < 0.0:
            grad = -grad
    elif dist > 1.0e-8:
        grad = (diff / dist) * side
    else:
        grad = _mesh_face_normal(mesh_id, face_index) * side

    phi = side * dist
    if phi <= target_phi:
        return

    projected = pos - grad * (phi - target_phi)
    disp = projected - current_positions[i]
    outward_disp = wp.dot(disp, grad)
    if outward_disp > 0.0:
        projected = projected - grad * outward_disp
    predicted_positions[i] = projected


@wp.kernel
def _apply_wall_contact_friction_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    smooth_normals: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    sign_scale: float,
    use_smooth_normals: bool,
    target_phi: float,
    max_dist: float,
    friction_coeff: float,
    contact_band: float,
):
    """Damp tangential velocity at wall contacts (proximal push transmission, stuck tip)."""
    i = wp.tid()
    if inv_masses[i] <= 0.0 or friction_coeff <= 0.0:
        return

    pos = positions[i]
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)

    if not wp.mesh_query_point_sign_normal(mesh_id, pos, max_dist, sign, face_index, face_u, face_v):
        return

    closest = wp.mesh_eval_position(mesh_id, face_index, face_u, face_v)
    diff = pos - closest
    dist = wp.length(diff)

    side = sign * sign_scale
    if wp.abs(side) < 1.0e-6:
        side = sign_scale

    grad = wp.vec3(0.0, 0.0, 0.0)
    if use_smooth_normals:
        weights = wp.vec3(1.0 - face_u - face_v, face_u, face_v)
        face_normal = _mesh_face_normal(mesh_id, face_index)
        grad = _mesh_barycentric_normal(mesh_id, smooth_normals, face_index, weights, face_normal) * sign_scale
        if dist > 1.0e-8 and wp.dot(grad, diff) * side < 0.0:
            grad = -grad
    elif dist > 1.0e-8:
        grad = (diff / dist) * side
    else:
        grad = _mesh_face_normal(mesh_id, face_index) * side

    grad_len = wp.length(grad)
    if grad_len < 1.0e-8:
        return
    n = grad / grad_len

    phi = side * dist
    gap = phi - target_phi
    band = wp.max(contact_band, 1.0e-6)
    if gap > band:
        return
    w = 1.0 - wp.clamp(gap / band, 0.0, 1.0)

    v = velocities[i]
    v_t = v - n * wp.dot(v, n)
    velocities[i] = v - friction_coeff * w * v_t


@wp.kernel
def _project_mesh_vertex_collision_kernel(
    predicted_positions: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    smooth_normals: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    radius: float,
    query_radius: float,
    sign_scale: float,
    use_smooth_normals: bool,
    max_triangles: int,
    corrections: wp.array(dtype=wp.vec3),
):
    """Per-particle vertex-vs-triangle containment (max-penetration variant).

    Picks the single nearby triangle with maximum penetration depth and applies
    that correction — summing all overlapping triangles over-corrects on curved
    surfaces where many nearly-tangent triangles report similar penetrations.
    """
    i = wp.tid()
    if inv_masses[i] <= 0.0:
        return

    p = predicted_positions[i]
    lower = wp.vec3(p.x - query_radius, p.y - query_radius, p.z - query_radius)
    upper = wp.vec3(p.x + query_radius, p.y + query_radius, p.z + query_radius)

    mesh = wp.mesh_get(mesh_id)
    query = wp.mesh_query_aabb(mesh_id, lower, upper)
    face_index = wp.int32(0)
    tri_count = int(0)
    best_pen = float(0.0)
    best_n = wp.vec3(0.0, 0.0, 0.0)

    while wp.mesh_query_aabb_next(query, face_index):
        if tri_count >= max_triangles:
            break
        tri_count = tri_count + 1

        i0 = mesh.indices[face_index * 3 + 0]
        i1 = mesh.indices[face_index * 3 + 1]
        i2 = mesh.indices[face_index * 3 + 2]
        v0 = mesh.points[i0]
        v1 = mesh.points[i1]
        v2 = mesh.points[i2]

        face_normal = _safe_normalize(wp.cross(v1 - v0, v2 - v0), wp.vec3(0.0, 0.0, 1.0))
        bar = _closest_point_triangle_barycentric(p, v0, v1, v2)
        n_outward = face_normal * sign_scale
        if use_smooth_normals:
            n_outward = _mesh_barycentric_normal(mesh_id, smooth_normals, face_index, bar, face_normal) * sign_scale

        cp = v0 * bar.x + v1 * bar.y + v2 * bar.z
        signed = wp.dot(p - cp, n_outward)
        penetration = signed + radius
        if penetration > best_pen:
            best_pen = penetration
            best_n = n_outward

    if best_pen > 0.0:
        wp.atomic_add(corrections, i, -best_n * best_pen)


@wp.kernel
def _project_mesh_vertex_collision_kernel_averaged(
    predicted_positions: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    smooth_normals: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    radius: float,
    query_radius: float,
    sign_scale: float,
    use_smooth_normals: bool,
    max_triangles: int,
    corrections: wp.array(dtype=wp.vec3),
):
    """Per-particle vertex-vs-triangle containment (averaged variant).

    Averages corrections from all penetrating nearby triangles.  Less
    aggressive than the max-penetration variant; useful when the SDF path
    causes ringing on highly curved surfaces.
    """
    i = wp.tid()
    if inv_masses[i] <= 0.0:
        return

    p = predicted_positions[i]
    lower = wp.vec3(p.x - query_radius, p.y - query_radius, p.z - query_radius)
    upper = wp.vec3(p.x + query_radius, p.y + query_radius, p.z + query_radius)

    mesh = wp.mesh_get(mesh_id)
    query = wp.mesh_query_aabb(mesh_id, lower, upper)
    face_index = wp.int32(0)
    tri_count = int(0)
    col_count = int(0)
    delta = wp.vec3(0.0, 0.0, 0.0)

    while wp.mesh_query_aabb_next(query, face_index):
        if tri_count >= max_triangles:
            break
        tri_count = tri_count + 1

        i0 = mesh.indices[face_index * 3 + 0]
        i1 = mesh.indices[face_index * 3 + 1]
        i2 = mesh.indices[face_index * 3 + 2]
        v0 = mesh.points[i0]
        v1 = mesh.points[i1]
        v2 = mesh.points[i2]

        face_normal = _safe_normalize(wp.cross(v1 - v0, v2 - v0), wp.vec3(0.0, 0.0, 1.0))
        bar = _closest_point_triangle_barycentric(p, v0, v1, v2)
        n_outward = face_normal * sign_scale
        if use_smooth_normals:
            n_outward = _mesh_barycentric_normal(mesh_id, smooth_normals, face_index, bar, face_normal) * sign_scale

        cp = v0 * bar.x + v1 * bar.y + v2 * bar.z
        signed = wp.dot(p - cp, n_outward)
        penetration = signed + radius
        if penetration > 0.0:
            delta = delta - n_outward * penetration
            col_count += 1

    if col_count > 0:
        wp.atomic_add(corrections, i, delta / float(col_count))


@wp.kernel
def _project_mesh_edge_collision_kernel(
    predicted_positions: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    smooth_normals: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    radius: float,
    query_radius: float,
    sign_scale: float,
    use_smooth_normals: bool,
    max_triangles: int,
    corrections: wp.array(dtype=wp.vec3),
    points_per_env: int,
    edges_per_env: int,
):
    """Per-rod-edge: rod-segment vs. mesh-edge collision (max-penetration variant).

    Iterates over the three edges of each nearby mesh triangle (deduped by
    ``ia < ib`` so each undirected edge is visited once), finds the single
    mesh edge with maximum rod-segment penetration, and accumulates the
    correction into the shared buffer.

    ``points_per_env`` / ``edges_per_env`` map a global edge to its own env's
    particle window, so one launch covers a batch. Single-env callers pass
    ``(num_points, num_edges)``, for which ``pi == edge`` and behaviour is
    unchanged. The floor division is required: envs have one more particle than
    edge, so a fractional quotient would land ``pi`` in the wrong env window.
    """
    edge = wp.tid()
    pi = (edge // edges_per_env) * points_per_env + (edge % edges_per_env)

    p0 = predicted_positions[pi]
    p1 = predicted_positions[pi + 1]
    inv0 = inv_masses[pi]
    inv1 = inv_masses[pi + 1]
    if inv0 <= 0.0 and inv1 <= 0.0:
        return

    lower = wp.vec3(
        wp.min(p0.x, p1.x) - query_radius,
        wp.min(p0.y, p1.y) - query_radius,
        wp.min(p0.z, p1.z) - query_radius,
    )
    upper = wp.vec3(
        wp.max(p0.x, p1.x) + query_radius,
        wp.max(p0.y, p1.y) + query_radius,
        wp.max(p0.z, p1.z) + query_radius,
    )

    mesh = wp.mesh_get(mesh_id)
    query = wp.mesh_query_aabb(mesh_id, lower, upper)
    face_index = wp.int32(0)
    tri_count = int(0)
    best_pen = float(0.0)
    best_n = wp.vec3(0.0, 0.0, 0.0)
    best_bar = wp.vec4(0.0, 0.0, 0.0, 0.0)

    while wp.mesh_query_aabb_next(query, face_index):
        if tri_count >= max_triangles:
            break
        tri_count = tri_count + 1

        i0 = mesh.indices[face_index * 3 + 0]
        i1 = mesh.indices[face_index * 3 + 1]
        i2 = mesh.indices[face_index * 3 + 2]
        v0 = mesh.points[i0]
        v1 = mesh.points[i1]
        v2 = mesh.points[i2]
        face_normal = _safe_normalize(wp.cross(v1 - v0, v2 - v0), wp.vec3(0.0, 0.0, 1.0))

        for mesh_edge in range(3):
            ia = i0
            ib = i1
            ea = v0
            eb = v1
            mesh_normal_a = wp.vec3(1.0, 0.0, 0.0)
            mesh_normal_b = wp.vec3(0.0, 1.0, 0.0)
            if mesh_edge == 1:
                ia = i1
                ib = i2
                ea = v1
                eb = v2
                mesh_normal_a = wp.vec3(0.0, 1.0, 0.0)
                mesh_normal_b = wp.vec3(0.0, 0.0, 1.0)
            elif mesh_edge == 2:
                ia = i2
                ib = i0
                ea = v2
                eb = v0
                mesh_normal_a = wp.vec3(0.0, 0.0, 1.0)
                mesh_normal_b = wp.vec3(1.0, 0.0, 0.0)
            if ia >= ib:
                continue

            bar = _segment_segment_barycentric(p0, p1, ea, eb)
            rod_point = p0 * bar.x + p1 * bar.y
            mesh_point = ea * bar.z + eb * bar.w
            n_outward = face_normal * sign_scale
            if use_smooth_normals:
                weights = mesh_normal_a * bar.z + mesh_normal_b * bar.w
                n_outward = (
                    _mesh_barycentric_normal(mesh_id, smooth_normals, face_index, weights, face_normal) * sign_scale
                )
            signed = wp.dot(rod_point - mesh_point, n_outward)
            penetration = signed + radius
            if penetration > best_pen:
                best_pen = penetration
                best_n = n_outward
                best_bar = bar

    if best_pen > 0.0:
        denom = inv0 * best_bar.x * best_bar.x + inv1 * best_bar.y * best_bar.y
        if denom > 1.0e-8:
            correction = -best_n * (best_pen / denom)
            if inv0 > 0.0:
                wp.atomic_add(corrections, pi, correction * best_bar.x * inv0)
            if inv1 > 0.0:
                wp.atomic_add(corrections, pi + 1, correction * best_bar.y * inv1)


@wp.kernel
def _project_mesh_edge_collision_kernel_averaged(
    predicted_positions: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    smooth_normals: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    radius: float,
    query_radius: float,
    sign_scale: float,
    use_smooth_normals: bool,
    max_triangles: int,
    corrections: wp.array(dtype=wp.vec3),
    points_per_env: int,
    edges_per_env: int,
):
    """Per-rod-edge: rod-segment vs. mesh-edge collision (averaged variant).

    ``points_per_env`` / ``edges_per_env`` are the batch mapping described on
    :func:`_project_mesh_edge_collision_kernel`.
    """
    edge = wp.tid()
    pi = (edge // edges_per_env) * points_per_env + (edge % edges_per_env)

    p0 = predicted_positions[pi]
    p1 = predicted_positions[pi + 1]
    inv0 = inv_masses[pi]
    inv1 = inv_masses[pi + 1]
    if inv0 <= 0.0 and inv1 <= 0.0:
        return

    lower = wp.vec3(
        wp.min(p0.x, p1.x) - query_radius,
        wp.min(p0.y, p1.y) - query_radius,
        wp.min(p0.z, p1.z) - query_radius,
    )
    upper = wp.vec3(
        wp.max(p0.x, p1.x) + query_radius,
        wp.max(p0.y, p1.y) + query_radius,
        wp.max(p0.z, p1.z) + query_radius,
    )

    mesh = wp.mesh_get(mesh_id)
    query = wp.mesh_query_aabb(mesh_id, lower, upper)
    face_index = wp.int32(0)
    tri_count = int(0)
    col_count = int(0)
    delta0 = wp.vec3(0.0, 0.0, 0.0)
    delta1 = wp.vec3(0.0, 0.0, 0.0)

    while wp.mesh_query_aabb_next(query, face_index):
        if tri_count >= max_triangles:
            break
        tri_count = tri_count + 1

        i0 = mesh.indices[face_index * 3 + 0]
        i1 = mesh.indices[face_index * 3 + 1]
        i2 = mesh.indices[face_index * 3 + 2]
        v0 = mesh.points[i0]
        v1 = mesh.points[i1]
        v2 = mesh.points[i2]
        face_normal = _safe_normalize(wp.cross(v1 - v0, v2 - v0), wp.vec3(0.0, 0.0, 1.0))

        for mesh_edge in range(3):
            ia = i0
            ib = i1
            ea = v0
            eb = v1
            mesh_normal_a = wp.vec3(1.0, 0.0, 0.0)
            mesh_normal_b = wp.vec3(0.0, 1.0, 0.0)
            if mesh_edge == 1:
                ia = i1
                ib = i2
                ea = v1
                eb = v2
                mesh_normal_a = wp.vec3(0.0, 1.0, 0.0)
                mesh_normal_b = wp.vec3(0.0, 0.0, 1.0)
            elif mesh_edge == 2:
                ia = i2
                ib = i0
                ea = v2
                eb = v0
                mesh_normal_a = wp.vec3(0.0, 0.0, 1.0)
                mesh_normal_b = wp.vec3(1.0, 0.0, 0.0)
            if ia >= ib:
                continue

            bar = _segment_segment_barycentric(p0, p1, ea, eb)
            rod_point = p0 * bar.x + p1 * bar.y
            mesh_point = ea * bar.z + eb * bar.w
            n_outward = face_normal * sign_scale
            if use_smooth_normals:
                weights = mesh_normal_a * bar.z + mesh_normal_b * bar.w
                n_outward = (
                    _mesh_barycentric_normal(mesh_id, smooth_normals, face_index, weights, face_normal) * sign_scale
                )
            signed = wp.dot(rod_point - mesh_point, n_outward)
            penetration = signed + radius
            if penetration > 0.0:
                denom = inv0 * bar.x * bar.x + inv1 * bar.y * bar.y
                if denom > 1.0e-8:
                    correction = -n_outward * (penetration / denom)
                    if inv0 > 0.0:
                        delta0 = delta0 + correction * bar.x * inv0
                    if inv1 > 0.0:
                        delta1 = delta1 + correction * bar.y * inv1
                    col_count += 1

    if col_count > 0:
        inv_count = 1.0 / float(col_count)
        if inv0 > 0.0:
            wp.atomic_add(corrections, pi, delta0 * inv_count)
        if inv1 > 0.0:
            wp.atomic_add(corrections, pi + 1, delta1 * inv_count)


@wp.kernel
def _apply_mesh_edge_collision_corrections_kernel(
    predicted_positions: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    corrections: wp.array(dtype=wp.vec3),
):
    """Flush the accumulated edge-collision correction buffer to predicted_positions."""
    i = wp.tid()
    if inv_masses[i] > 0.0:
        predicted_positions[i] = predicted_positions[i] + corrections[i]


# ──────────────────────────────────────────────────────────────────────────────
# Signed-distance sampling (diagnostic / offline use)
# ──────────────────────────────────────────────────────────────────────────────


@wp.kernel
def _sample_signed_distance_kernel(
    points: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    max_dist: float,
    sign_scale: float,
    distances: wp.array(dtype=wp.float32),
):
    """Sample the signed distance from each point to the mesh surface."""
    i = wp.tid()
    pos = points[i]
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    if wp.mesh_query_point_sign_normal(mesh_id, pos, max_dist, sign, face_index, face_u, face_v):
        closest = wp.mesh_eval_position(mesh_id, face_index, face_u, face_v)
        dist = wp.length(pos - closest)
        distances[i] = sign * sign_scale * dist
    else:
        distances[i] = max_dist


def compute_signed_distances(
    points: np.ndarray,
    mesh: "wp.Mesh",
    max_dist: float = 1.0,
    sign_scale: float = 1.0,
    device: str = "cuda",
) -> np.ndarray:
    """Return the signed distance from each point to *mesh*.

    Args:
        points:     ``(N, 3)`` float32 positions.
        mesh:       A :class:`warp.Mesh` whose BVH has been built.
        max_dist:   Search radius; points farther than this receive
                    ``max_dist`` as their distance.
        sign_scale: Flip sign convention (``+1`` for interior-positive,
                    ``-1`` for interior-negative).
        device:     Warp device string.

    Returns:
        ``(N,)`` float32 signed distances.
    """
    pts = np.asarray(points, dtype=np.float32).reshape((-1, 3))
    n = pts.shape[0]
    pts_wp = wp.array(pts, dtype=wp.vec3, device=device)
    out_wp = wp.zeros(n, dtype=wp.float32, device=device)
    wp.launch(
        _sample_signed_distance_kernel,
        dim=n,
        inputs=[pts_wp, mesh.id, float(max_dist), float(sign_scale), out_wp],
        device=device,
    )
    return out_wp.numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Stage normalisation helper
# ──────────────────────────────────────────────────────────────────────────────


def _normalize_collision_projection_stage(stage: str) -> str:
    s = stage.strip().lower()
    if s not in _COLLISION_PROJECTION_STAGES:
        raise ValueError(
            f"Invalid collision_projection_stage {stage!r}. " f"Choose one of {_COLLISION_PROJECTION_STAGES}."
        )
    return s


# ──────────────────────────────────────────────────────────────────────────────
# CathRodSolver
# ──────────────────────────────────────────────────────────────────────────────


class CathRodSolver(XPBDRodSolver):
    """XPBD rod solver with track-guided insertion and vessel mesh containment.

    Extends :class:`~catheter_vasculature_solver.XPBDRodSolver` with three
    sample-local projections:

    * **Track guidance** — non-tip particles slide along a linear axis.
    * **Vessel containment (SDF)** — BVH signed-distance query per particle.
    * **Vessel containment (AABB/edge)** — vertex + edge contacts via AABB
      broadphase; more robust in tight curvature regions.

    Args:
        *args: Positional arguments forwarded to :class:`XPBDRodSolver`.
        collision_mesh: Closed triangle mesh defining the vessel interior.
            Pass ``None`` to disable collision (track guidance still runs).
        track_start: World-space origin of the insertion axis, shape ``(3,)``
            shared by every env, or ``(num_envs, 3)`` with one origin per env.
        track_dir: Direction of the insertion axis, ``(3,)`` or
            ``(num_envs, 3)`` like ``track_start``; normalized here.
        track_length: Maximum insertion distance along the axis (metres).
        tip_num_edges: Number of distal rod edges treated as the free tip;
            particles beyond index ``num_points - tip_num_edges`` are *not*
            snapped to the track.
        particle_radius: Effective rod radius for collision detection (metres).
        segment_length: Rest length of a single rod segment (metres).  Used
            to set the default ``max_dist`` BVH search radius, together with
            the longest of ``env_segment_lengths`` when that is given.
        track_stiffness: Blend factor for track snapping (``0`` = free,
            ``1`` = perfectly locked to axis).
        track_enabled: Toggle track guidance on/off.
        track_path: Optional ordered route, proximal to distal, that replaces the
            straight axis: one ``(N, 3)`` route shared by every env, a
            ``(num_envs, N, 3)`` array, or a sequence of ``(N_e, 3)`` routes,
            one per env, whose lengths may differ.
        proximal_feed_span: How many nodes ahead of the root supply the
            insertion direction. One is the rod's own first-segment tangent,
            which a fold at the root turns -- and the push then drives further
            into the fold, so it grows on its own. A span of several nodes
            reads the same direction over a baseline long enough that the fold
            barely moves it. Must be set before the first step, since the index
            array is built once with the workspace.
        track_free_distal_edges: How many distal edges guidance leaves alone,
            when that should differ from ``tip_num_edges``. Unset, the two are
            the same and guidance holds everything but the steerable tip, which
            prescribes the whole route the wire takes. Set larger, guidance
            becomes a proximal rail: the shaft behind the working length is
            held on the path it already traversed, which is where it physically
            lies, while the distal window stays free to be steered. Make that
            window longer than the route still to be navigated and guidance
            constrains nothing the operator has yet to choose.
        collision_enabled: Toggle vessel containment on/off.
        collision_mesh_normals: Optional pre-computed smooth vertex normals
            on the *same* device as the solver.  Pass the output of
            :func:`compute_smooth_vertex_normals` wrapped in a
            :class:`warp.array`.
        smooth_collision_normals_enabled: Use area-weighted vertex normals
            instead of flat face normals (reduces facet-aligned ringing).
        collision_iterations: Gauss-Seidel iterations per substep.
        collision_pre_constraints_enabled: Run vessel containment *before*
            the XPBD constraint solve.
        collision_post_constraints_enabled: Run vessel containment *after*
            the XPBD constraint solve (default ``True``, recommended).
        mesh_edge_collision_enabled: Use the AABB / edge collision path
            instead of the SDF path.
        mesh_edge_collision_radius: Effective rod radius for the edge path
            (defaults to ``particle_radius``).
        mesh_edge_collision_query_radius: AABB query radius for the edge path
            (defaults to ``mesh_edge_collision_radius``).
        mesh_edge_collision_max_triangles: Triangle budget per AABB query
            (default ``64``).
        collision_projection_stage: Shorthand for enabling exactly one stage
            (``"pre"`` or ``"post"``); overrides the two boolean flags when set.
        sign_scale: Sign convention flip for the mesh (``+1`` if mesh normals
            point inward, ``-1`` if they point outward).
        target_phi: Target signed-distance clearance from the vessel wall.
            Defaults to ``-particle_radius`` (keep the rod surface just inside).
        max_dist: BVH search radius.  Defaults to
            ``2 * particle_radius + segment_length``.
        centerline_runtime: Optional :class:`CenterlineVesselRuntime` — a
            deforming Cosserat centerline vessel co-simulated with the catheter.
            Supports independent vessels in single and batched solves.
        centerline_containment_enabled: Contain the catheter inside the live
            lumen. Defaults to ``True`` whenever ``centerline_runtime`` is given.
        centerline_containment_stage: ``"pre"`` or ``"post"`` — when centerline
            containment runs relative to the XPBD constraint solve
            (default ``"post"``). Independent of ``collision_projection_stage``,
            which governs the static mesh path.
        contact_coupling_iterations: Alternate live-wall projections with
            global elastic solves, updating position and orientation.
            Each edge block solves all six stretch/shear and bend/twist
            constraints. Zero retains the global direct solve and legacy
            pre/post containment. When enabled, this replaces centerline
            stage ordering and position-only cleanup; use physical rotational
            inertia for the material frames.
        **kwargs: Keyword arguments forwarded to :class:`XPBDRodSolver`.
    """

    def __init__(
        self,
        *args,
        collision_mesh: "wp.Mesh | None",
        track_start: np.ndarray,
        track_dir: np.ndarray,
        track_length: float,
        tip_num_edges: int,
        particle_radius: float,
        segment_length: float,
        track_stiffness: float = 1.0,
        track_enabled: bool = True,
        track_free_distal_edges: int | None = None,
        proximal_feed_span: int = 1,
        track_path: np.ndarray | None = None,
        track_stage: str | None = None,
        containment_cleanup_iterations: int = 0,
        containment_cleanup_relaxation: float = 1.0,
        containment_cleanup_rounds: int = 1,
        contact_coupling_iterations: int = 0,
        collision_enabled: bool = True,
        collision_mesh_normals: "wp.array | None" = None,
        smooth_collision_normals_enabled: bool = False,
        collision_iterations: int = 2,
        collision_pre_constraints_enabled: bool = False,
        collision_post_constraints_enabled: bool = True,
        mesh_edge_collision_enabled: bool = False,
        mesh_edge_collision_radius: float | None = None,
        mesh_edge_collision_query_radius: float | None = None,
        mesh_edge_collision_max_triangles: int = DEFAULT_MESH_EDGE_MAX_TRIANGLES,
        collision_projection_stage: str | None = None,
        sign_scale: float = 1.0,
        target_phi: float | None = None,
        max_dist: float | None = None,
        centerline_runtime: "CenterlineVesselRuntime | None" = None,
        centerline_containment_enabled: bool | None = None,
        centerline_containment_stage: str = COLLISION_PROJECTION_STAGE_POST,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.set_track_axes(track_start, track_dir)

        self.collision_mesh = collision_mesh
        self.collision_mesh_normals = collision_mesh_normals
        self.smooth_collision_normals_enabled = bool(smooth_collision_normals_enabled)
        self.track_length = float(track_length)
        self.tip_num_edges = int(tip_num_edges)
        # Rest curvature as authored, snapshotted the first time a tip bend is
        # applied and re-read on every one after that. Steering has to be
        # absolute rather than cumulative -- an action supplies an angle, not a
        # delta -- which means each update has to know the unsteered shape. One
        # slot per workspace because both can exist and only one is read.
        self._tip_bend_baseline_ws: "wp.array | None" = None
        self._tip_bend_baseline_bws: "wp.array | None" = None
        self._tip_bend_angle_device: "wp.array | None" = None
        self.track_stiffness = float(track_stiffness)
        self.track_enabled = bool(track_enabled)
        if track_free_distal_edges is not None and int(track_free_distal_edges) < 0:
            raise ValueError(f"track_free_distal_edges must be non-negative, got {track_free_distal_edges}")
        self.track_free_distal_edges = None if track_free_distal_edges is None else int(track_free_distal_edges)
        if int(proximal_feed_span) < 1:
            raise ValueError(f"proximal_feed_span must be at least 1, got {proximal_feed_span}")
        self.proximal_feed_span = int(proximal_feed_span)
        # Kept on the host and uploaded per device on first use, matching how the
        # collision-normal buffers are handled: the workspace device is not known
        # until a step runs.
        self.set_track_path(track_path)
        # Defaults to post for the straight rail, which is where it has always
        # run. A curved track wants pre instead, so that the distance
        # constraints are what finally set the chord lengths.
        self.track_stage = _normalize_collision_projection_stage(
            track_stage if track_stage is not None else COLLISION_PROJECTION_STAGE_POST
        )
        # Off by default: it costs two launches per iteration and only earns them
        # where a post-solve containment projection is disturbing edge lengths.
        self.containment_cleanup_iterations = max(int(containment_cleanup_iterations), 0)
        if not 0.0 < float(containment_cleanup_relaxation) <= 1.0:
            raise ValueError(
                "containment_cleanup_relaxation must be in (0, 1], got "
                f"{containment_cleanup_relaxation}"
            )
        self.containment_cleanup_relaxation = float(containment_cleanup_relaxation)
        # How many times to alternate containment with the cleanup sweeps. One
        # runs containment once and spends the whole sweep budget after it, which
        # is the original sequencing and lets the sweeps overrule containment.
        self.containment_cleanup_rounds = max(1, int(containment_cleanup_rounds))
        if int(contact_coupling_iterations) != contact_coupling_iterations or contact_coupling_iterations < 0:
            raise ValueError("contact_coupling_iterations must be a nonnegative integer")
        self.contact_coupling_iterations = int(contact_coupling_iterations)
        self.collision_enabled = bool(collision_enabled)
        self.collision_iterations = max(1, int(collision_iterations))
        self.collision_pre_constraints_enabled = bool(collision_pre_constraints_enabled)
        self.collision_post_constraints_enabled = bool(collision_post_constraints_enabled)
        self.mesh_edge_collision_enabled = bool(mesh_edge_collision_enabled)
        self.mesh_edge_collision_radius = (
            float(mesh_edge_collision_radius) if mesh_edge_collision_radius is not None else float(particle_radius)
        )
        self.mesh_edge_collision_query_radius = (
            float(mesh_edge_collision_query_radius)
            if mesh_edge_collision_query_radius is not None
            else self.mesh_edge_collision_radius
        )
        self.mesh_edge_collision_max_triangles = max(1, int(mesh_edge_collision_max_triangles))
        if collision_projection_stage is not None:
            self.set_collision_projection_stage(collision_projection_stage)
        self.sign_scale = float(sign_scale)
        self.target_phi = float(target_phi) if target_phi is not None else -float(particle_radius)
        longest_segment = max(float(segment_length), *self.env_segment_lengths)
        self.max_dist = float(max_dist) if max_dist is not None else 2.0 * float(particle_radius) + longest_segment

        # Centerline Cosserat vessel (optional). The centerline is its own
        # collision path, so it carries its own stage rather than reusing the
        # static-mesh ``collision_{pre,post}_constraints_enabled`` flags —
        # those may both be set, which would double-apply the correction.
        self.centerline_runtime = centerline_runtime
        if centerline_containment_enabled is None:
            centerline_containment_enabled = centerline_runtime is not None
        self.centerline_containment_enabled = bool(centerline_containment_enabled) and (
            centerline_runtime is not None
        )
        self.centerline_containment_stage = _normalize_collision_projection_stage(centerline_containment_stage)

        # Dedicated per-rod correction buffer for the mesh-edge collision path.
        # Decoupled from ws.pos_corrections (which the constraint solver owns).
        self._mesh_edge_corrections: "wp.array | None" = None

        # Per-device dummy normals array for when smooth normals are disabled.
        self._dummy_collision_mesh_normals: dict[str, "wp.array"] = {}

        # Tangential damping at wall contacts (fluoro-nav realism).
        self.collision_friction_enabled = False
        self.collision_friction_coeff = 0.0
        self.collision_friction_contact_band = 0.004

        self._validate_centerline_batch(self.centerline_runtime)

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def set_collision_mesh(
        self,
        collision_mesh: "wp.Mesh | None",
        collision_mesh_normals: "wp.array | None" = None,
    ) -> None:
        """Swap the vessel mesh at runtime (also rebuilds the BVH on next step).

        Args:
            collision_mesh: New vessel mesh, or ``None`` to disable collision.
            collision_mesh_normals: Optional pre-computed smooth vertex normals
                on the solver device.
        """
        self._require_single_anatomy_for_mesh(mesh=collision_mesh)
        self.collision_mesh = collision_mesh
        self.collision_mesh_normals = collision_mesh_normals

    @property
    def track_starts(self) -> np.ndarray:
        """Insertion-axis origins, ``(T, 3)`` with ``T`` 1 or ``num_envs``. Read-only."""
        return self._track_starts

    @property
    def track_dirs(self) -> np.ndarray:
        """Unit insertion directions matching :attr:`track_starts`. Read-only."""
        return self._track_dirs

    @property
    def track_start(self) -> wp.vec3:
        """The first env's insertion origin, for callers that read a single track."""
        return wp.vec3(*(float(v) for v in self._track_starts[0]))

    @property
    def track_dir(self) -> wp.vec3:
        """The first env's insertion direction, for callers that read a single track."""
        return wp.vec3(*(float(v) for v in self._track_dirs[0]))

    def set_track_axes(self, track_start, track_dir) -> None:
        """Replace the insertion axes, one shared or one per env.

        The axes are uploaded to each device once, so they are read-only
        attributes and change only through here, which drops the uploads.
        """
        starts, dirs = self._validated_track_axes(track_start, track_dir)
        starts = starts.copy()
        starts.flags.writeable = False
        dirs.flags.writeable = False
        self._track_starts, self._track_dirs = starts, dirs
        self._track_axes_device: dict[str, tuple[wp.array, wp.array]] = {}

    def set_track_path(self, track_path) -> None:
        """Replace the guidance routes, or pass ``None`` for the straight rail.

        Accepts the same forms as the ``track_path`` constructor argument and
        drops the per-device uploads of the previous routes.
        """
        path = self._validated_track_path(track_path)
        self._require_single_anatomy_for_mesh(path=path)
        self._track_path_host = path
        self._track_path_device: dict[str, tuple[wp.array, wp.array]] = {}

    @staticmethod
    def _routes_differ(points: np.ndarray, offsets: np.ndarray) -> bool:
        routes = [points[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]
        return any(route.shape != routes[0].shape or not np.array_equal(route, routes[0]) for route in routes[1:])

    _UNSET = object()

    def _require_single_anatomy_for_mesh(self, *, mesh=_UNSET, runtime=_UNSET, path=_UNSET) -> None:
        """Refuse a vessel mesh alongside per-env anatomy.

        ``collision_mesh`` is one ``wp.Mesh`` queried by every env, so it can
        only describe a batch that shares one anatomy. Combined with per-env
        vessels or routes, each env would be contained by the wrong patient.
        Arguments default to the solver's current state.
        """
        if mesh is self._UNSET:
            mesh = getattr(self, "collision_mesh", None)
        if mesh is None:
            return
        if runtime is self._UNSET:
            runtime = getattr(self, "centerline_runtime", None)
        if path is self._UNSET:
            path = getattr(self, "_track_path_host", None)
        if runtime is not None and bool(getattr(runtime, "per_env_vessels", False)):
            source = "a centerline runtime with per-env vessels"
        elif path is not None and self._routes_differ(*path):
            source = "different track_path routes per env"
        else:
            return
        raise ValueError(
            f"collision_mesh supports a single anatomy shared by every env, but the solver has {source}. "
            "Pass collision_mesh=None and rely on centerline containment for per-env anatomy."
        )

    def set_collision_projection_stage(self, stage: str) -> None:
        """Configure whether containment runs before or after rod constraints.

        Args:
            stage: ``"pre"`` or ``"post"``.
        """
        normalized = _normalize_collision_projection_stage(stage)
        self.collision_pre_constraints_enabled = normalized == COLLISION_PROJECTION_STAGE_PRE
        self.collision_post_constraints_enabled = normalized == COLLISION_PROJECTION_STAGE_POST

    def set_root_position(self, world_pos, env_ids=None) -> None:
        """Kinematically write root particle positions (fixed-rail advance).

        Args:
            world_pos: ``(3,)`` applied to every selected env, or ``(N, 3)`` giving
                one position per selected env.
            env_ids: Envs to write. ``None`` means all. Ignored on the
                single-env path.

        Raises:
            ValueError: If *world_pos* does not broadcast to the selected envs.
            IndexError: If any env id is out of range.
        """
        if self._bws is None:
            ws = self._ws
            if ws is None:
                return
            p = to_host(world_pos, np.float32).reshape(-1)
            if p.size != 3:
                raise ValueError(f"world_pos must be (3,), got shape {p.shape}")
            wp.launch(
                _set_root_position_kernel,
                dim=1,
                inputs=[
                    ws.positions,
                    ws.predicted_positions,
                    ws.velocities,
                    wp.vec3(float(p[0]), float(p[1]), float(p[2])),
                ],
                device=self._device,
            )
            return

        bws = self._bws
        ids = self._resolve_env_ids(env_ids)
        if ids.size == 0:
            return
        pos = to_host(world_pos, np.float32).reshape((-1, 3))
        if pos.shape[0] == 1:
            pos = np.repeat(pos, ids.size, axis=0)
        elif pos.shape[0] != ids.size:
            raise ValueError(
                f"world_pos has {pos.shape[0]} rows but {ids.size} envs were selected"
            )
        env_id_view = self._stage_env_ids(bws, ids)
        wp.launch(
            _set_root_position_batched_kernel,
            dim=ids.size,
            inputs=[
                bws.positions,
                bws.predicted_positions,
                bws.velocities,
                int(bws.num_points_per_rod),
                env_id_view,
                wp.array(pos, dtype=wp.vec3, device=self._device),
            ],
            device=self._device,
        )

    def capture_tip_bend_baseline(self) -> None:
        """Re-read the unsteered rest curvature that :meth:`set_tip_bend` adds to.

        Runs by itself before the first steer, so the common case needs nothing.
        Call it again after writing rest curvature straight into the solver --
        :func:`i4h_arena.medical.newton_catheter_physics.seed_rest_curvature_from_path`
        is the case that exists -- or that shape counts as part of the bend and
        the next steer undoes it.

        It snapshots the buffer as it stands, so calling it while a bend is
        applied bakes that bend in as the new rest shape.
        """
        ws = self._ws
        if ws is not None and ws.num_edges > 0:
            self._tip_bend_baseline_ws = wp.clone(ws.rest_darboux)
        bws = self._bws
        if bws is not None and bws.num_edges_per_rod > 0:
            self._tip_bend_baseline_bws = wp.clone(bws.rest_darboux)

    def _tip_bend_angle_buffer(self, bend_angle: Any, count: int) -> "wp.array":
        """Coerce a steering command into a device float array of *count* angles.

        A device-resident tensor is handed through as it stands, so steering the
        tip every step from an action costs no host transfer. A scalar is
        broadcast into a buffer that is reused across calls.
        """
        if isinstance(bend_angle, wp.array):
            if bend_angle.dtype != wp.float32:
                raise TypeError(f"bend_angle array must be float32, got {bend_angle.dtype}")
            if bend_angle.shape[0] < count:
                raise ValueError(f"bend_angle holds {bend_angle.shape[0]} angles, need {count}")
            return bend_angle
        if hasattr(bend_angle, "detach") and hasattr(bend_angle, "contiguous"):
            flat = bend_angle.detach().reshape(-1).float().contiguous()
            if flat.numel() < count:
                raise ValueError(f"bend_angle holds {flat.numel()} angles, need {count}")
            return wp.from_torch(flat, dtype=wp.float32)
        values = np.asarray(bend_angle, dtype=np.float32).reshape(-1)
        if values.size == 1:
            values = np.full(count, float(values[0]), dtype=np.float32)
        elif values.size < count:
            raise ValueError(f"bend_angle holds {values.size} angles, need {count}")
        buffer = self._tip_bend_angle_device
        if buffer is None or buffer.shape[0] < count:
            buffer = wp.zeros(count, dtype=wp.float32, device=self._device)
            self._tip_bend_angle_device = buffer
        buffer.assign(wp.array(values[:count], dtype=wp.float32, device=self._device))
        return buffer

    def set_tip_bend(self, bend_angle: Any, tip_num_edges: int | None = None) -> None:
        """Steer the tip by rewriting the tip edges' rest Darboux vector.

        Safe to call every step: the angle is absolute, measured from the
        authored rest shape rather than from wherever the last call left the
        buffer, so a controller supplies an angle and not a delta.

        Args:
            bend_angle: Turn of the tip *polyline*, in radians, either one value
                shared by every env or one per env. A device tensor is used
                where it lies.
            tip_num_edges: Trailing edges to spread the bend over. Defaults to
                the solver's own ``tip_num_edges``.

        The bend is about the tip's local X axis. Pointing it at a particular
        branch is the axial rotation command's job, which is also how a
        pre-shaped wire is aimed in a real procedure.

        The kernels write ``sin(angle / (2n - 1))`` into each of the ``n`` tip
        edges, which is what makes the argument an angle rather than a solver
        coordinate. The derivation, from Mosaic's measurement of this same
        entry point:

        ``rest_darboux`` holds the imaginary part of the rotation between an
        edge's two end-frames, and a quaternion's imaginary part is
        ``sin(phi / 2)``, so a per-edge value ``d`` turns that edge's frame by
        ``phi = 2 asin(d)`` and the ``n`` tip frames turn by ``n phi`` in total.
        The polyline turns *less* than the frames do: a segment points along the
        average of the frames at its two ends, so the tip segments point at
        ``phi/2, 3phi/2, ..., (n - 1/2) phi`` and the last one reaches only
        ``(n - 1/2) phi``. The final half-hinge of frame rotation lives past the
        last segment. Setting ``(n - 1/2) phi`` equal to the requested angle and
        solving gives ``d = sin(angle / (2n - 1))``.

        That matters because the node positions are what a camera sees and what
        a policy is scored against, not the frame ladder. Writing ``angle / n``
        per edge -- the obvious reading, and what this did before -- over-bends
        by very nearly ``(2n - 1) / n``, so a requested 1.5 rad came out at
        2.86 rad with the shipped ``tip_num_edges = 10``.

        Two consequences worth knowing. Because ``sin`` is bounded, a large
        request can no longer drive ``asin`` out of domain; the old form went
        undefined once the argument passed ``n``, which a one-edge tip band
        reached at 1 rad -- inside the shipped 1.5 rad ceiling. And the mapping
        is exact only while ``angle <= (2n - 1) pi / 2``, beyond which
        ``asin(sin(x))`` folds back and a larger request realizes a smaller
        turn. That is 29.8 rad at ten edges and ``pi / 2`` at one, so the
        shipped ceiling fits either way, with 0.07 rad to spare in the
        single-edge case.

        The derivation assumes the tip's baseline curvature is zero, which is
        the shipped rest shape. Steering a tip that was authored curved adds to
        that curvature, and the realized turn is then the sum rather than the
        request.
        """
        n_tip = int(tip_num_edges if tip_num_edges is not None else self.tip_num_edges)
        if n_tip <= 0:
            return
        if self._tip_bend_baseline_ws is None and self._tip_bend_baseline_bws is None:
            self.capture_tip_bend_baseline()
        # Both workspaces exist in batched mode but only the batched buffers are
        # read by the batched solve, so writing just _ws would silently do
        # nothing. Keep them consistent by writing whichever exists.
        ws = self._ws
        bws = self._bws
        # Sized once for the widest consumer: the single-env kernel reads entry
        # zero, so one buffer serves both and a batched step does not reallocate.
        needed = 1
        if bws is not None and bws.num_edges_per_rod > 0:
            needed = max(needed, int(bws.num_envs))
        angles = self._tip_bend_angle_buffer(bend_angle, needed)
        if ws is not None and ws.num_edges > 0 and self._tip_bend_baseline_ws is not None:
            wp.launch(
                _update_tip_rest_darboux_kernel,
                dim=ws.num_edges,
                inputs=[
                    ws.rest_darboux,
                    self._tip_bend_baseline_ws,
                    int(ws.num_edges),
                    n_tip,
                    angles,
                ],
                device=self._device,
            )
        if bws is not None and bws.num_edges_per_rod > 0 and self._tip_bend_baseline_bws is not None:
            wp.launch(
                _update_tip_rest_darboux_batched_kernel,
                dim=(bws.num_envs, bws.num_edges_per_rod),
                inputs=[
                    bws.rest_darboux,
                    self._tip_bend_baseline_bws,
                    int(bws.num_edges_per_rod),
                    n_tip,
                    angles,
                ],
                device=self._device,
            )

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _collision_normals_for_device(self, dev: str) -> "tuple[wp.array, bool]":
        """Return ``(normals_array, use_smooth)`` for the given device."""
        if self.smooth_collision_normals_enabled and self.collision_mesh_normals is not None:
            return self.collision_mesh_normals, True
        if dev not in self._dummy_collision_mesh_normals:
            self._dummy_collision_mesh_normals[dev] = wp.zeros(1, dtype=wp.vec3, device=dev)
        return self._dummy_collision_mesh_normals[dev], False

    def _ensure_edge_corrections(
        self, ws: _Workspace, dev: str, *, num_points: int | None = None
    ) -> "wp.array":
        """Lazily allocate (or reuse) the per-rod edge-collision buffer."""
        n_points = int(num_points if num_points is not None else ws.num_points)
        if self._mesh_edge_corrections is None or self._mesh_edge_corrections.shape[0] < n_points:
            self._mesh_edge_corrections = wp.zeros(max(1, n_points), dtype=wp.vec3, device=dev)
        return self._mesh_edge_corrections

    def _track_count_ok(self, count: int) -> bool:
        """A per-env track input must have one entry, or one per env."""
        return count == 1 or count == int(self._num_envs)

    def _validated_track_axes(self, track_start, track_dir) -> tuple[np.ndarray, np.ndarray]:
        """Insertion axes as ``(T, 3)`` origins and unit directions, ``T`` 1 or ``num_envs``."""
        starts = np.asarray(track_start, dtype=np.float32).reshape(-1, 3)
        dirs = np.asarray(track_dir, dtype=np.float32).reshape(-1, 3)
        for name, values in (("track_start", starts), ("track_dir", dirs)):
            if not self._track_count_ok(len(values)):
                raise ValueError(
                    f"{name} needs one axis or one per env ({self._num_envs}), got {len(values)}"
                )
        if len(starts) != len(dirs):
            if len(starts) == 1:
                starts = np.repeat(starts, len(dirs), axis=0)
            else:
                dirs = np.repeat(dirs, len(starts), axis=0)
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        if np.any(norms <= 0.0):
            raise ValueError("track_dir must be non-zero")
        return starts, (dirs / norms).astype(np.float32)

    def _validated_track_path(self, track_path) -> "tuple[np.ndarray, np.ndarray] | None":
        """Normalize ordered guidance routes, or ``None`` for the straight rail.

        Accepts one ``(N, 3)`` route shared by every env, a ``(num_envs, N, 3)``
        array, or a sequence of ``(N_e, 3)`` routes, one per env, whose lengths
        may differ. Returns the routes back to back with their start offsets.

        Ordering is the caller's responsibility and cannot be checked here: the
        points have to run proximal to distal along the route the wire takes,
        because consecutive pairs are treated as segments. A shuffled path is a
        valid polyline that happens to zig-zag across the anatomy.
        """
        if track_path is None:
            return None
        if isinstance(track_path, (list, tuple)) and track_path and all(
            np.asarray(route).ndim == 2 for route in track_path
        ):
            routes = [np.asarray(route, dtype=np.float32).reshape(-1, 3) for route in track_path]
        else:
            path = np.asarray(track_path, dtype=np.float32)
            routes = list(path) if path.ndim == 3 else [path.reshape(-1, 3)]
        if not self._track_count_ok(len(routes)):
            raise ValueError(f"track_path needs one route or one per env ({self._num_envs}), got {len(routes)}")
        for route in routes:
            if len(route) < 2:
                raise ValueError(f"track_path needs at least 2 points to form a segment, got {len(route)}")
            if not np.all(np.isfinite(route)):
                raise ValueError("track_path contains non-finite coordinates")
        offsets = np.cumsum([0] + [len(route) for route in routes]).astype(np.int32)
        return np.concatenate(routes, axis=0), offsets

    def _track_axes_for_device(self, dev: str) -> "tuple[wp.array, wp.array]":
        """Upload the insertion axes to ``dev`` once and reuse them thereafter."""
        cached = self._track_axes_device.get(dev)
        if cached is None:
            cached = (
                wp.array(self.track_starts, dtype=wp.vec3, device=dev),
                wp.array(self.track_dirs, dtype=wp.vec3, device=dev),
            )
            self._track_axes_device[dev] = cached
        return cached

    def _track_path_for_device(self, dev: str) -> "tuple[wp.array, wp.array] | None":
        """Upload the guidance routes and their offsets to ``dev`` once."""
        if self._track_path_host is None:
            return None
        cached = self._track_path_device.get(dev)
        if cached is None:
            points, offsets = self._track_path_host
            cached = (
                wp.array(points, dtype=wp.vec3, device=dev),
                wp.array(offsets, dtype=wp.int32, device=dev),
            )
            self._track_path_device[dev] = cached
        return cached

    def _project_containment_and_cleanup(
        self,
        project_containment,
        predicted_positions,
        inv_masses,
        rest_lengths,
        *,
        num_envs: int,
        points_per_env: int,
        edges_per_env: int,
        dev: str,
    ) -> None:
        """Alternate containment and length cleanup until both roughly hold.

        These two projections disagree, and running one after the other lets
        whichever went last win outright. Containment shoves a particle radially
        without regard for its neighbours, which mangles edge lengths; the
        cleanup sweeps equalize edge lengths with no idea where the vessel is,
        which shoves particles laterally through the wall. Measured on the s0011
        route with one containment followed by 128 sweeps: chords reach a
        best-ever 100-104% while 13 of 41 particles sit outside the lumen, the
        worst containment recorded. The sweeps simply had the last word.

        Alternating in ``containment_cleanup_rounds`` rounds is the standard
        answer -- projecting onto each constraint set in turn converges toward a
        point in both, instead of to whichever was applied last. The sweep budget
        is divided across rounds rather than multiplied by them, so raising the
        round count trades sweep depth per round for agreement between the two
        and leaves the cleanup cost alone. Containment itself is re-run per
        round, which is the added cost.

        One round reproduces the original sequencing exactly.
        """
        rounds = max(1, int(self.containment_cleanup_rounds))
        total = max(int(self.containment_cleanup_iterations), 0)
        for round_index in range(rounds):
            if project_containment is not None:
                project_containment()
            # Spread the remainder over the earliest rounds so the totals match
            # the single-round budget exactly and a sweep is never silently lost.
            sweeps = total // rounds + (1 if round_index < total % rounds else 0)
            self._project_distance_cleanup(
                predicted_positions,
                inv_masses,
                rest_lengths,
                num_envs=num_envs,
                points_per_env=points_per_env,
                edges_per_env=edges_per_env,
                dev=dev,
                iterations=sweeps,
            )

    def _project_distance_cleanup(
        self,
        predicted_positions,
        inv_masses,
        rest_lengths,
        *,
        num_envs: int,
        points_per_env: int,
        edges_per_env: int,
        dev: str,
        iterations: int | None = None,
    ) -> None:
        """Re-satisfy edge lengths that a post-solve projection has disturbed.

        Ordered after containment rather than before it: running first would just
        be more constraint solve, and containment would overwrite the result as
        it already overwrites the solve's. ``iterations`` overrides the configured
        budget so a caller alternating the two can spend it a slice at a time.
        """
        sweeps = self.containment_cleanup_iterations if iterations is None else int(iterations)
        if sweeps <= 0 or edges_per_env <= 0:
            return
        for _ in range(sweeps):
            # Even edges then odd. Splitting the sweep is what keeps two threads
            # off the same shared particle.
            for color in (0, 1):
                wp.launch(
                    _distance_cleanup_kernel,
                    dim=(num_envs, edges_per_env),
                    inputs=[
                        predicted_positions,
                        inv_masses,
                        rest_lengths,
                        int(points_per_env),
                        int(edges_per_env),
                        int(color),
                        float(self.containment_cleanup_relaxation),
                    ],
                    device=dev,
                )

    def _track_guidance_active(self, stage: str) -> bool:
        """Is guidance enabled and scheduled for this side of the solve?"""
        return self.track_enabled and self.track_stage == stage

    def _track_end_index(self, points_per_env: int) -> int:
        """First particle index guidance leaves free.

        At least one particle stays guided even when the free window is asked
        to cover the whole rod, since a guidance pass that touches nothing is
        better expressed by disabling it.
        """
        free_edges = self.tip_num_edges if self.track_free_distal_edges is None else self.track_free_distal_edges
        return max(1, int(points_per_env) - int(free_edges))

    def _project_track_guidance(self, ws: _Workspace, dev: str) -> None:
        """Project non-tip particles onto the insertion axis or path."""
        self._launch_track_guidance(ws.predicted_positions, ws.inv_masses, int(ws.num_points),
                                    int(ws.num_points), dev)

    def _project_track_guidance_batched(self, bws, dev: str) -> None:
        """Track guidance for every env in one launch, each on its own track."""
        self._launch_track_guidance(bws.predicted_positions, bws.inv_masses, int(bws.np_total),
                                    int(bws.num_points_per_rod), dev)

    def _launch_track_guidance(self, predicted_positions, inv_masses, total_points: int,
                               points_per_env: int, dev: str) -> None:
        end_idx = self._track_end_index(points_per_env)
        path = self._track_path_for_device(dev)
        if path is not None:
            points, offsets = path
            wp.launch(
                _track_polyline_kernel,
                dim=total_points,
                inputs=[
                    predicted_positions,
                    inv_masses,
                    points,
                    offsets,
                    float(self.track_stiffness),
                    int(end_idx),
                    points_per_env,
                ],
                device=dev,
            )
            return
        starts, dirs = self._track_axes_for_device(dev)
        wp.launch(
            _track_sliding_kernel,
            dim=total_points,
            inputs=[
                predicted_positions,
                inv_masses,
                starts,
                dirs,
                float(self.track_length),
                float(self.track_stiffness),
                int(end_idx),
                points_per_env,
            ],
            device=dev,
        )

    def _project_vessel_containment(
        self,
        ws: _Workspace,
        dev: str,
        *,
        num_points: int | None = None,
        num_edges: int | None = None,
        points_per_env: int | None = None,
        edges_per_env: int | None = None,
    ) -> None:
        """Run vessel containment for the configured number of iterations.

        Every kernel here is per-particle or per-rod-edge against a *shared*
        static mesh, so the batched path is the same launches over the whole
        concatenated buffer. The counts default to this workspace's own, which
        is what the single-env path passes.

        Args:
            num_points: Total particles to cover (default ``ws.num_points``).
            num_edges: Total rod edges to cover (default ``ws.num_edges``).
            points_per_env: Particles per env, for mapping an edge to its own
                env's particle window. Defaults to *num_points* (single env).
            edges_per_env: Rod edges per env. Defaults to *num_edges*.
        """
        if not self.collision_enabled or self.collision_mesh is None:
            return

        n_points = int(num_points if num_points is not None else ws.num_points)
        n_edges = int(num_edges if num_edges is not None else ws.num_edges)
        ppe = int(points_per_env if points_per_env is not None else n_points)
        epe = int(edges_per_env if edges_per_env is not None else n_edges)

        smooth_normals, use_smooth = self._collision_normals_for_device(dev)
        mesh_args = [
            self.collision_mesh.id,
            smooth_normals,
            ws.inv_masses,
            float(self.mesh_edge_collision_radius),
            float(self.mesh_edge_collision_query_radius),
            float(self.sign_scale),
            bool(use_smooth),
            int(self.mesh_edge_collision_max_triangles),
        ]
        sdf_args = [
            self.collision_mesh.id,
            smooth_normals,
            ws.inv_masses,
            float(self.sign_scale),
            bool(use_smooth),
            float(self.target_phi),
            float(self.max_dist),
        ]

        if self.mesh_edge_collision_enabled:
            corrections = self._ensure_edge_corrections(ws, dev, num_points=n_points)
            for _ in range(self.collision_iterations):
                wp.launch(_clear_vec3_kernel, dim=n_points, inputs=[corrections], device=dev)
                wp.launch(
                    _project_mesh_vertex_collision_kernel_averaged,
                    dim=n_points,
                    inputs=[ws.predicted_positions, *mesh_args, corrections],
                    device=dev,
                )
                wp.launch(
                    _project_mesh_edge_collision_kernel_averaged,
                    dim=n_edges,
                    inputs=[ws.predicted_positions, *mesh_args, corrections, ppe, epe],
                    device=dev,
                )
                # Robustness pass: add max-penetration contacts to catch deep
                # local tunneling at sharp bends where averaged contacts can
                # under-correct. Keep this in the same correction buffer so
                # one flush applies the combined displacement.
                wp.launch(
                    _project_mesh_vertex_collision_kernel,
                    dim=n_points,
                    inputs=[ws.predicted_positions, *mesh_args, corrections],
                    device=dev,
                )
                wp.launch(
                    _project_mesh_edge_collision_kernel,
                    dim=n_edges,
                    inputs=[ws.predicted_positions, *mesh_args, corrections, ppe, epe],
                    device=dev,
                )
                wp.launch(
                    _apply_mesh_edge_collision_corrections_kernel,
                    dim=n_points,
                    inputs=[ws.predicted_positions, ws.inv_masses, corrections],
                    device=dev,
                )
                # Final SDF clamp guarantees particle-level lumen containment
                # even if edge contacts miss a high-curvature corner.
                wp.launch(
                    _project_vessel_containment_kernel,
                    dim=n_points,
                    inputs=[ws.positions, ws.predicted_positions, *sdf_args],
                    device=dev,
                )
        else:
            for _ in range(self.collision_iterations):
                wp.launch(
                    _project_vessel_containment_kernel,
                    dim=n_points,
                    inputs=[ws.positions, ws.predicted_positions, *sdf_args],
                    device=dev,
                )

        if getattr(self, "_apply_friction_after_containment", True):
            self._apply_collision_friction(ws, dev, num_points=n_points)

    def _project_vessel_containment_batched(self, bws, dev: str) -> None:
        """Static-mesh containment for every env in one launch per kernel."""
        self._project_vessel_containment(
            bws,
            dev,
            num_points=bws.np_total,
            num_edges=bws.ne_total,
            points_per_env=bws.num_points_per_rod,
            edges_per_env=bws.num_edges_per_rod,
        )

    def _apply_collision_friction(
        self, ws: _Workspace, dev: str, *, num_points: int | None = None
    ) -> None:
        """Damp tangential velocity where the rod contacts the vessel wall."""
        if (
            not self.collision_friction_enabled
            or self.collision_friction_coeff <= 0.0
            or not self.collision_enabled
            or self.collision_mesh is None
        ):
            return

        smooth_normals, use_smooth = self._collision_normals_for_device(dev)
        wp.launch(
            _apply_wall_contact_friction_kernel,
            dim=int(num_points if num_points is not None else ws.num_points),
            inputs=[
                ws.positions,
                ws.velocities,
                self.collision_mesh.id,
                smooth_normals,
                ws.inv_masses,
                float(self.sign_scale),
                bool(use_smooth),
                float(self.target_phi),
                float(self.max_dist),
                float(self.collision_friction_coeff),
                float(self.collision_friction_contact_band),
            ],
            device=dev,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Hooks wired into XPBDRodSolver._substep
    # ──────────────────────────────────────────────────────────────────────

    def set_centerline_runtime(
        self,
        runtime: "CenterlineVesselRuntime | None",
        *,
        containment_enabled: bool | None = None,
        stage: str | None = None,
    ) -> None:
        """Attach or clear a deforming centerline vessel runtime.

        Args:
            runtime: Centerline vessel runtime, or ``None`` to detach.
            containment_enabled: Whether the catheter is contained by the live
                lumen. Defaults to ``True`` whenever *runtime* is given.
            stage: ``"pre"`` or ``"post"`` — when containment runs relative to
                the XPBD constraint solve. Unchanged when omitted.
        """
        self._validate_centerline_batch(runtime)
        self.centerline_runtime = runtime
        if stage is not None:
            self.centerline_containment_stage = _normalize_collision_projection_stage(stage)
        if containment_enabled is None:
            containment_enabled = runtime is not None
        self.centerline_containment_enabled = bool(containment_enabled) and runtime is not None
        # Centerline predict/project/finalize is outside the captured rod graph.
        self.reset_cuda_graph()

    @property
    def contact_depth(self) -> float:
        """Deepest lumen penetration in metres during the last substep.

        Covers centerline containment only, so static mesh collision and the
        floor plane do not contribute. Zero without an attached centerline
        runtime.
        """
        runtime = self.centerline_runtime
        return 0.0 if runtime is None else runtime.contact_depth

    @property
    def contact_count(self) -> int:
        """Contacting centerline samples during the last substep."""
        runtime = self.centerline_runtime
        return 0 if runtime is None else runtime.contact_count

    @property
    def contact_depth_per_env(self) -> np.ndarray:
        """Deepest lumen penetration per env, shape ``(num_envs,)``.

        A per-env signal is what a trainer needs in order to penalise or
        terminate the env that perforated, rather than the whole batch.
        """
        runtime = self.centerline_runtime
        if runtime is None:
            return np.zeros(self._num_envs, dtype=np.float32)
        return runtime.contact_depth_per_env

    @property
    def contact_count_per_env(self) -> np.ndarray:
        """Contacting centerline samples per env, shape ``(num_envs,)``."""
        runtime = self.centerline_runtime
        if runtime is None:
            return np.zeros(self._num_envs, dtype=np.int32)
        return runtime.contact_count_per_env

    def reset(self, env_ids=None) -> None:
        """Reset the listed envs' catheter *and* their vessels.

        Extends :meth:`XPBDRodSolver.reset` so a reset env starts against an
        undeformed wall. Without this the rod would return to rest inside a
        lumen still carrying the previous episode's deformation, which would
        leak state across episode boundaries.
        """
        super().reset(env_ids)
        runtime = self.centerline_runtime
        if runtime is not None:
            runtime.reset(env_ids)

    def _can_use_cuda_graph(self, sub_dt: float) -> bool:
        # The vessel launches are inside _substep, so they *are* capturable — a
        # forced capture runs ~3.7x faster. Capture stays off because the runtime
        # feeds its tunables (two_way, vessel_response, catheter_radius,
        # max_distance, the delta clamps) as Python scalars, which a graph bakes
        # in at capture time: mutating them afterwards would silently no-op.
        # Enabling capture means making those device-resident first.
        if self.centerline_runtime is not None:
            return False
        return super()._can_use_cuda_graph(sub_dt)

    def _substep(self, dt: float) -> None:
        """Advance catheter and (when attached) the centerline vessel by *dt*.

        Wrapping the substep rather than :meth:`step` keeps substep counting,
        batched dispatch and CUDA-graph selection in the base class. With a
        :class:`CenterlineVesselRuntime` attached the order is::

            predict_cosserat → project/clamp → catheter XPBD → finalize_cosserat

        Catheter containment against the live lumen runs inside the base
        substep via the constraint hooks, i.e. against predicted positions and
        before ``finalize`` commits the vessel.
        """
        runtime = self.centerline_runtime
        if runtime is None:
            super()._substep(dt)
            return

        runtime.begin_step(self._ws.positions)
        runtime.predict(dt)
        runtime.project_constraints()
        super()._substep(dt)
        runtime.finalize(dt)

    def _substep_batched(self, dt: float) -> None:
        """Batched counterpart of :meth:`_substep`, driving the per-env vessels.

        The base :meth:`step` dispatches straight to this method when batched, so
        the vessel lifecycle has to be driven here as well — otherwise the walls
        would stay frozen while every env's rod advanced against them.
        """
        runtime = self.centerline_runtime
        if runtime is None:
            super()._substep_batched(dt)
            return

        runtime.begin_step(self._bws.positions)
        runtime.predict(dt)
        runtime.project_constraints()
        super()._substep_batched(dt)
        runtime.finalize(dt)

    def _validate_centerline_batch(self, runtime) -> None:
        """Require one independent vessel per env, so no env shares a wall.

        A mismatch would either index out of bounds in the batched containment
        kernel or, worse, quietly couple envs through a single vessel — which
        would make every env's trajectory depend on its neighbours.
        """
        if runtime is None:
            return
        self._require_single_anatomy_for_mesh(runtime=runtime)
        vessel_envs = int(getattr(runtime, "num_envs", 1))
        if vessel_envs != self._num_envs:
            raise ValueError(
                f"centerline vessel has num_envs={vessel_envs} but this solver has "
                f"num_envs={self._num_envs}. Build the runtime with "
                f"CenterlineVesselRuntime.from_tree(..., num_envs={self._num_envs}) so each "
                "env owns its own deformable vessel."
            )

    def _centerline_containment_active(self, stage: str) -> bool:
        return self.centerline_containment_enabled and self.centerline_containment_stage == stage

    def _coupled_centerline_active(self) -> bool:
        return self.centerline_containment_enabled and self.contact_coupling_iterations > 0

    def _project_direct(self, ws: _Workspace, dt: float, dev: str) -> None:
        """Reconcile live contact with positions AND material frames.

        Compliance and elastic multipliers are prepared once by the base
        substep. Keep those multipliers across nonlinear iterations, then let
        the base integrator derive velocities from the final solved state.
        No position-only cleanup follows this solve.
        """
        if not self._coupled_centerline_active():
            return super()._project_direct(ws, dt, dev)
        for _ in range(self.contact_coupling_iterations):
            if self.centerline_runtime.two_way:
                self.centerline_runtime.project_constraints()
            self._project_centerline_containment(ws)
            super()._project_direct(ws, dt, dev)

    def _project_direct_batched(self, bws, dt: float, dev: str) -> None:
        """Independent contact/elastic iterations for each rod in the batch."""
        if not self._coupled_centerline_active():
            return super()._project_direct_batched(bws, dt, dev)
        for _ in range(self.contact_coupling_iterations):
            if self.centerline_runtime.two_way:
                self.centerline_runtime.project_constraints()
            self._project_centerline_containment_batched(bws)
            super()._project_direct_batched(bws, dt, dev)

    def _pre_constraints_hook(self, ws: _Workspace, dt: float, dev: str) -> None:
        """Apply guidance and vessel containment *before* XPBD constraint solve."""
        del dt
        # Guidance leads, so containment sees a rod already on its route and has
        # only residual penetration left to answer for.
        if self._track_guidance_active(COLLISION_PROJECTION_STAGE_PRE):
            self._project_track_guidance(ws, dev)
        if self.collision_pre_constraints_enabled:
            self._project_vessel_containment(ws, dev)
        if (
            not self._coupled_centerline_active()
            and self._centerline_containment_active(COLLISION_PROJECTION_STAGE_PRE)
        ):
            self._project_centerline_containment(ws)

    def _post_constraints_hook(self, ws: _Workspace, dt: float, dev: str) -> None:
        """Apply track guidance and vessel containment *after* XPBD constraint solve."""
        del dt
        if self._track_guidance_active(COLLISION_PROJECTION_STAGE_POST):
            self._project_track_guidance(ws, dev)
        if self.collision_post_constraints_enabled:
            self._project_vessel_containment(ws, dev)
        if self._coupled_centerline_active():
            return
        containment = None
        if self._centerline_containment_active(COLLISION_PROJECTION_STAGE_POST):
            containment = lambda: self._project_centerline_containment(ws)  # noqa: E731
        self._project_containment_and_cleanup(
            containment,
            ws.predicted_positions,
            ws.inv_masses,
            ws.rest_lengths,
            num_envs=1,
            points_per_env=int(ws.num_points),
            edges_per_env=int(ws.num_edges),
            dev=dev,
        )

    def _pre_constraints_hook_batched(self, bws, dt: float, dev: str) -> None:
        """Batched counterpart of :meth:`_pre_constraints_hook`."""
        del dt
        if self._track_guidance_active(COLLISION_PROJECTION_STAGE_PRE):
            self._project_track_guidance_batched(bws, dev)
        if self.collision_pre_constraints_enabled:
            self._project_vessel_containment_batched(bws, dev)
        if (
            not self._coupled_centerline_active()
            and self._centerline_containment_active(COLLISION_PROJECTION_STAGE_PRE)
        ):
            self._project_centerline_containment_batched(bws)

    def _post_constraints_hook_batched(self, bws, dt: float, dev: str) -> None:
        """Batched counterpart of :meth:`_post_constraints_hook`."""
        del dt
        if self._track_guidance_active(COLLISION_PROJECTION_STAGE_POST):
            self._project_track_guidance_batched(bws, dev)
        if self.collision_post_constraints_enabled:
            self._project_vessel_containment_batched(bws, dev)
        if self._coupled_centerline_active():
            return
        containment = None
        if self._centerline_containment_active(COLLISION_PROJECTION_STAGE_POST):
            containment = lambda: self._project_centerline_containment_batched(bws)  # noqa: E731
        self._project_containment_and_cleanup(
            containment,
            bws.predicted_positions,
            bws.inv_masses,
            bws.rest_lengths,
            num_envs=int(bws.num_envs),
            points_per_env=int(bws.num_points_per_rod),
            edges_per_env=int(bws.num_edges_per_rod),
            dev=dev,
        )

    def _project_centerline_containment(self, ws: _Workspace) -> None:
        """Project catheter predicted positions into the live centerline lumen."""
        runtime = self.centerline_runtime
        if runtime is None:
            return
        runtime.project_containment(
            ws.predicted_positions,
            ws.inv_masses,
            ws.num_points,
            ws.num_edges,
        )

    def _project_centerline_containment_batched(self, bws) -> None:
        """Contain every env's catheter against its own live lumen in one launch."""
        runtime = self.centerline_runtime
        if runtime is None:
            return
        runtime.project_containment(
            bws.predicted_positions,
            bws.inv_masses,
            bws.np_total,
            bws.ne_total,
        )


__all__ = [
    "CathRodSolver",
    "COLLISION_PROJECTION_STAGE_POST",
    "COLLISION_PROJECTION_STAGE_PRE",
    "compute_smooth_vertex_normals",
    "compute_signed_distances",
    "_apply_mesh_edge_collision_corrections_kernel",
    "_project_mesh_edge_collision_kernel",
    "_project_mesh_edge_collision_kernel_averaged",
    "_project_mesh_vertex_collision_kernel",
    "_project_mesh_vertex_collision_kernel_averaged",
    "_project_vessel_containment_kernel",
    "_sample_signed_distance_kernel",
    "_set_root_position_batched_kernel",
    "_set_root_position_kernel",
    "_update_tip_rest_darboux_batched_kernel",
    "_update_tip_rest_darboux_kernel",
    "_apply_wall_contact_friction_kernel",
]
