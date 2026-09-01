# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Primary vessel-deformation backend (branching iterative local Cosserat rods).

"""Iterative per-segment PBD Cosserat rod (Kugelstadt & Schoemer, SCA 2016).

Canonical Cosserat discretization used by the reference literature:
  * N centerline nodes carry POSITIONS,
  * S SEGMENTS each carry one orientation QUATERNION (frame at the edge center).

Topology is EXPLICIT (edge -> two node indices, bend pair -> two segment indices), so
the same solver handles a single chain OR a branching tree/graph. Two vector
constraints, projected with colored Gauss-Seidel (colors from a greedy graph coloring
so no two constraints in a color share a node/segment):
  * stretch/shear (single director):  C = (x_b - x_a)/l - R(q) e3
  * bend/twist (Darboux):             C = Im(conj(q_i) q_j) - Omega_rest

Projection formulas are ported verbatim from PositionBasedDynamics'
``PositionBasedCosseratRods`` (PositionBasedElasticRods.cpp:20-79). Quaternion "mass"
is a single scalar per segment; quaternions are renormalized after each projection.
Shared foundation for the direct (Deul 2018) and Stable Cosserat Rods (Hsu 2025) modes.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np
import warp as wp

E3 = wp.constant(wp.vec3(0.0, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Quaternion helpers -- Hamilton product, {x, y, z, w}
# --------------------------------------------------------------------------- #
@wp.func
def qmul(a: wp.quat, b: wp.quat) -> wp.quat:
    av = wp.vec3(a[0], a[1], a[2])
    bv = wp.vec3(b[0], b[1], b[2])
    aw = a[3]
    bw = b[3]
    v = aw * bv + bw * av + wp.cross(av, bv)
    return wp.quat(v[0], v[1], v[2], aw * bw - wp.dot(av, bv))


@wp.func
def qconj(a: wp.quat) -> wp.quat:
    return wp.quat(-a[0], -a[1], -a[2], a[3])


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
@wp.kernel
def predict_positions(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=wp.float32),
    gravity: wp.vec3,
    dt: float,
    p: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    if inv_mass[i] > 0.0:
        p[i] = x[i] + v[i] * dt + gravity * (dt * dt)
    else:
        p[i] = x[i]


@wp.kernel
def predict_orientations(
    q: wp.array(dtype=wp.quat),
    omega: wp.array(dtype=wp.vec3),
    inv_mass_q: wp.array(dtype=wp.float32),
    dt: float,
    u: wp.array(dtype=wp.quat),
):
    i = wp.tid()
    if inv_mass_q[i] > 0.0:
        w = omega[i]
        # Body-space angular velocity: q_dot = 1/2 q (x) (0, omega). This matches the
        # body-space omega recovered in finalize_orientations (omega = Im(2 conj(q) u)/dt).
        dq = qmul(q[i], wp.quat(w[0], w[1], w[2], 0.0))
        n = wp.normalize(wp.vec4(q[i][0] + 0.5 * dt * dq[0], q[i][1] + 0.5 * dt * dq[1],
                                 q[i][2] + 0.5 * dt * dq[2], q[i][3] + 0.5 * dt * dq[3]))
        u[i] = wp.quat(n[0], n[1], n[2], n[3])
    else:
        u[i] = q[i]


@wp.kernel
def stretch_shear(
    group: wp.array(dtype=wp.int32),
    p: wp.array(dtype=wp.vec3),
    u: wp.array(dtype=wp.quat),
    inv_mass: wp.array(dtype=wp.float32),
    inv_mass_q: wp.array(dtype=wp.float32),
    seg_n0: wp.array(dtype=wp.int32),
    seg_n1: wp.array(dtype=wp.int32),
    rest_len: wp.array(dtype=wp.float32),
    stretch_multiplier: wp.array(dtype=wp.float32),
    ks: float,
    orient: int,
    n_group: int,
):
    """C = (x_b - x_a)/l - R(q) e3, colored Gauss-Seidel (no two edges in a color share
    a node, so writes never race). ``orient=0`` projects positions only (SCR mode, where
    the closed-form pass owns orientation); ``orient!=0`` also corrects the quaternion (PBD)."""
    t = wp.tid()
    if t >= n_group:
        return
    e = group[t]
    i0 = seg_n0[e]
    i1 = seg_n1[e]
    w0 = inv_mass[i0]
    w1 = inv_mass[i1]
    wq = inv_mass_q[e]
    rest_length = rest_len[e]
    q0 = u[e]

    d3 = wp.quat_rotate(q0, E3)
    gamma = (p[i1] - p[i0]) / rest_length - d3
    # In positions-only (SCR) mode the quaternion is not corrected here, so its compliance
    # must not appear in the effective-mass denominator (else positions are over-damped).
    denom = (w0 + w1) / rest_length + float(orient) * wq * 4.0 * rest_length + 1.0e-6
    gamma = (gamma / denom) * ks * stretch_multiplier[e]

    if w0 > 0.0:
        p[i0] = p[i0] + w0 * gamma
    if w1 > 0.0:
        p[i1] = p[i1] - w1 * gamma
    if orient != 0 and wq > 0.0:
        qe3 = wp.quat(-q0[1], q0[0], -q0[3], q0[2])  # q0 * conj(e3)
        cq = qmul(wp.quat(gamma[0], gamma[1], gamma[2], 0.0), qe3)
        s = 2.0 * wq * rest_length
        n = wp.normalize(wp.vec4(q0[0] + s * cq[0], q0[1] + s * cq[1],
                                 q0[2] + s * cq[2], q0[3] + s * cq[3]))
        u[e] = wp.quat(n[0], n[1], n[2], n[3])


@wp.kernel
def bend_twist(
    group: wp.array(dtype=wp.int32),
    u: wp.array(dtype=wp.quat),
    inv_mass_q: wp.array(dtype=wp.float32),
    bend_a: wp.array(dtype=wp.int32),
    bend_b: wp.array(dtype=wp.int32),
    rest_darboux: wp.array(dtype=wp.quat),
    kbt: wp.array(dtype=wp.vec3),
    n_group: int,
):
    """C = Im(conj(q_i) q_j) - Omega_rest, colored Gauss-Seidel."""
    t = wp.tid()
    if t >= n_group:
        return
    k = group[t]
    ia = bend_a[k]
    ib = bend_b[k]
    q0 = u[ia]
    q1 = u[ib]
    wq0 = inv_mass_q[ia]
    wq1 = inv_mass_q[ib]
    if wq0 <= 0.0 and wq1 <= 0.0:
        return

    omega = qmul(qconj(q0), q1)
    rd = rest_darboux[k]
    om_minus = wp.vec4(omega[0] - rd[0], omega[1] - rd[1], omega[2] - rd[2], omega[3] - rd[3])
    om_plus = wp.vec4(omega[0] + rd[0], omega[1] + rd[1], omega[2] + rd[2], omega[3] + rd[3])
    d = om_minus
    if wp.dot(om_minus, om_minus) > wp.dot(om_plus, om_plus):  # quaternion double-cover fix
        d = om_plus

    kb = kbt[k]
    denom = wq0 + wq1 + 1.0e-6
    om_s = wp.quat(d[0] * kb[0] / denom, d[1] * kb[1] / denom, d[2] * kb[2] / denom, 0.0)
    cq0 = qmul(q1, om_s)
    cq1 = qmul(q0, om_s)
    if wq0 > 0.0:
        n0 = wp.normalize(wp.vec4(q0[0] + wq0 * cq0[0], q0[1] + wq0 * cq0[1],
                                  q0[2] + wq0 * cq0[2], q0[3] + wq0 * cq0[3]))
        u[ia] = wp.quat(n0[0], n0[1], n0[2], n0[3])
    if wq1 > 0.0:
        n1 = wp.normalize(wp.vec4(q1[0] - wq1 * cq1[0], q1[1] - wq1 * cq1[1],
                                  q1[2] - wq1 * cq1[2], q1[3] - wq1 * cq1[3]))
        u[ib] = wp.quat(n1[0], n1[1], n1[2], n1[3])


@wp.kernel
def scr_orientation(
    group: wp.array(dtype=wp.int32),
    u: wp.array(dtype=wp.quat),
    p: wp.array(dtype=wp.vec3),
    inv_mass_q: wp.array(dtype=wp.float32),
    seg_n0: wp.array(dtype=wp.int32),
    seg_n1: wp.array(dtype=wp.int32),
    rest_len: wp.array(dtype=wp.float32),
    pair_off: wp.array(dtype=wp.int32),
    pair_nb: wp.array(dtype=wp.int32),
    pair_rest: wp.array(dtype=wp.quat),
    pair_role: wp.array(dtype=wp.int32),
    pair_scale: wp.array(dtype=wp.float32),
    scr_stretch: float,
    scr_bend: float,
    n_group: int,
):
    """Stable Cosserat Rods closed-form orientation (Hsu 2025), colored Gauss-Seidel.

    v = -2 k_ss (x_b - x_a) aligns the director to the edge (Eq. 15; k_ss folds in the l);
    b sums the length-scaled (4 k_bt / l) bend/twist targets from every incident bend pair
    (so junctions work). q = normalize((Q(v) Q(b) e3) + lam b).
    """
    t = wp.tid()
    if t >= n_group:
        return
    e = group[t]
    if inv_mass_q[e] <= 0.0:
        return
    q0 = u[e]
    v = (p[seg_n1[e]] - p[seg_n0[e]]) * (-2.0 * scr_stretch)

    b = wp.vec4(0.0, 0.0, 0.0, 0.0)
    lo = pair_off[e]
    hi = pair_off[e + 1]
    for j in range(lo, hi):
        qn = u[pair_nb[j]]
        qr = pair_rest[j]
        if pair_role[j] != 0:  # e is the pair's first segment ("next" neighbor): (q_nb * qr^-1)
            rel = qmul(qconj(q0), qn)
            tq = qmul(qn, qconj(qr))
        else:                  # e is the pair's second segment ("prev" neighbor): (q_nb * qr)
            rel = qmul(qconj(qn), q0)
            tq = qmul(qn, qr)
        sgn = 1.0
        if wp.dot(wp.vec4(rel[0], rel[1], rel[2], rel[3]),
                  wp.vec4(qr[0], qr[1], qr[2], qr[3])) < 0.0:
            sgn = -1.0
        b = b + (sgn * scr_bend * pair_scale[j]) * wp.vec4(tq[0], tq[1], tq[2], tq[3])

    lam = wp.length(v) + wp.length(b)
    prod = qmul(qmul(wp.quat(v[0], v[1], v[2], 0.0), wp.quat(b[0], b[1], b[2], b[3])),
                wp.quat(0.0, 0.0, 1.0, 0.0))
    num = wp.vec4(prod[0] + lam * b[0], prod[1] + lam * b[1],
                  prod[2] + lam * b[2], prod[3] + lam * b[3])
    if wp.length(num) < 1.0e-9:  # no bend target -> keep current orientation
        return
    num = wp.normalize(num)
    u[e] = wp.quat(num[0], num[1], num[2], num[3])


@wp.kernel
def floor_collision(
    p: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=wp.float32),
    floor_z: float,
    radius: float,
):
    """Unilateral PBD bounds constraint: clamp each free node to z >= floor_z + radius."""
    i = wp.tid()
    if inv_mass[i] > 0.0:
        pi = p[i]
        minz = floor_z + radius
        if pi[2] < minz:
            p[i] = wp.vec3(pi[0], pi[1], minz)


@wp.kernel
def finalize_positions(
    x: wp.array(dtype=wp.vec3),
    p: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=wp.float32),
    dt: float,
    damping: float,
):
    i = wp.tid()
    if inv_mass[i] > 0.0:
        v[i] = (p[i] - x[i]) / dt * (1.0 - damping)
        x[i] = p[i]
    else:
        v[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def finalize_orientations(
    q: wp.array(dtype=wp.quat),
    u: wp.array(dtype=wp.quat),
    omega: wp.array(dtype=wp.vec3),
    inv_mass_q: wp.array(dtype=wp.float32),
    dt: float,
    damping: float,
):
    i = wp.tid()
    if inv_mass_q[i] > 0.0:
        rel = qmul(qconj(q[i]), u[i])  # omega = Im(2 conj(q^t) u) / dt
        omega[i] = wp.vec3(2.0 * rel[0] / dt, 2.0 * rel[1] / dt, 2.0 * rel[2] / dt) * (1.0 - damping)
        q[i] = u[i]
    else:
        omega[i] = wp.vec3(0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# NumPy quaternion helpers (build time)
# --------------------------------------------------------------------------- #
def _qmul_np(a, b):
    av, aw = a[:3], a[3]
    bv, bw = b[:3], b[3]
    v = aw * bv + bw * av + np.cross(av, bv)
    return np.array([v[0], v[1], v[2], aw * bw - float(np.dot(av, bv))], dtype=np.float64)


def _qconj_np(a):
    return np.array([-a[0], -a[1], -a[2], a[3]], dtype=np.float64)


def _quat_axis_angle(axis, angle):
    h = 0.5 * angle
    s = math.sin(h)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(h)], dtype=np.float64)


def _quat_rotate_np(q, v):
    u = q[:3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + q[3] * v)


def _quat_from_matrix(m):
    """Quaternion (x,y,z,w) from a 3x3 rotation matrix whose columns are the frame axes."""
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)


def _frame_from_tangent(z, ref_x):
    """Deterministic orthonormal frame with local z = tangent; ref_x seeds local x."""
    z = z / np.linalg.norm(z)
    ref = np.array([0.0, 0.0, -1.0]) if abs(z[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    if ref_x is not None:
        ref = ref_x
    x = ref - np.dot(ref, z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return _quat_from_matrix(np.column_stack((x, y, z)))


def _chain_frames(positions):
    """Twist-minimizing per-segment frames (parallel transport) for a single chain."""
    P = np.asarray(positions, dtype=np.float64)
    tang = np.diff(P, axis=0)
    lengths = np.linalg.norm(tang, axis=1, keepdims=True)
    if np.any(lengths <= 1.0e-9):
        raise ValueError("coincident consecutive rod points")
    tang = tang / lengths
    z0 = tang[0]
    ref = np.array([0.0, 0.0, -1.0]) if abs(z0[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = ref - np.dot(ref, z0) * z0
    x /= np.linalg.norm(x)
    frames, prev_z = [], z0
    for z in tang:
        c = np.cross(prev_z, z)
        cn = float(np.linalg.norm(c))
        d = float(np.clip(np.dot(prev_z, z), -1.0, 1.0))
        if cn > 1.0e-10:
            x = _quat_rotate_np(_quat_axis_angle(c / cn, math.atan2(cn, d)), x)
        x -= np.dot(x, z) * z
        x /= np.linalg.norm(x)
        frames.append(_quat_from_matrix(np.column_stack((x, np.cross(z, x), z))))
        prev_z = z
    return np.asarray(frames, dtype=np.float64)


def _quat_between(t_from, t_to):
    """Minimal-rotation quaternion taking unit vector t_from to t_to."""
    c = np.cross(t_from, t_to)
    s = float(np.linalg.norm(c))
    d = float(np.dot(t_from, t_to))
    if s < 1.0e-8:
        if d > 0.0:
            return np.array([0.0, 0.0, 0.0, 1.0])
        perp = np.cross(t_from, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(perp) < 1.0e-6:
            perp = np.cross(t_from, np.array([0.0, 1.0, 0.0]))
        perp /= np.linalg.norm(perp)
        return _quat_axis_angle(perp, math.pi)
    return _quat_axis_angle(c / s, math.atan2(s, d))


def _tree_frames(positions, edges, root=0):
    """Parallel-transport per-segment frames over a BFS spanning tree from ``root``.

    Each edge's frame is the parent edge's frame rotated by the minimal rotation between
    their tangents, so twist is propagated consistently across the whole tree. For a
    symmetric fork this yields mirror-symmetric branch frames (unlike an independent
    per-segment reference frame, which bakes in an arbitrary per-branch twist)."""
    P = np.asarray(positions, dtype=np.float64)
    tang = [(P[b] - P[a]) / np.linalg.norm(P[b] - P[a]) for a, b in edges]
    node_edges = defaultdict(list)
    for e, (a, b) in enumerate(edges):
        node_edges[a].append(e)
        node_edges[b].append(e)
    frames = [None] * len(edges)
    visited = set()

    def bfs(seed):
        frames[seed] = _frame_from_tangent(tang[seed], None)
        visited.add(seed)
        dq = deque([seed])
        while dq:
            e = dq.popleft()
            for node in edges[e]:
                for f in node_edges[node]:
                    if f in visited:
                        continue
                    qf = _qmul_np(_quat_between(tang[e], tang[f]), frames[e])
                    frames[f] = qf / np.linalg.norm(qf)
                    visited.add(f)
                    dq.append(f)

    incident = node_edges[root]
    if incident:  # root the primary component at `root`
        bfs(min(incident))
    for e in range(len(edges)):  # each remaining connected component gets its own consistent seed
        if e not in visited:
            bfs(e)
    return np.asarray(frames, dtype=np.float64)


def _rest_darboux(q, bend_pairs):
    rd = np.zeros((max(len(bend_pairs), 1), 4), dtype=np.float64)
    for k, (a, b) in enumerate(bend_pairs):
        rel = _qmul_np(_qconj_np(q[a]), q[b])
        if rel[3] < 0.0:
            rel = -rel
        rd[k] = rel
    return rd


def derive_bend_pairs(edges, root=0):
    """Bend/twist pairs as a spanning tree (parent edge -> child edge), rooted at ``root``.

    Each edge is bend-coupled only to its PARENT edge in a BFS spanning tree, so a
    junction couples every outgoing branch to the incoming (trunk) segment -- never the
    branches directly to each other. For a chain this reduces to consecutive pairs; for
    a symmetric fork it keeps the two branches symmetric. At the root (or any junction
    with no incoming edge) the lowest-index incident edge acts as the local parent.
    """
    adj = defaultdict(list)
    for e, (a, b) in enumerate(edges):
        adj[a].append((e, b))
        adj[b].append((e, a))

    parent_edge = {root: -1}
    visited = {root}
    pairs = set()
    dq = deque([root])
    while dq:
        node = dq.popleft()
        pe = parent_edge[node]
        incident = [e for e, _ in adj[node]]
        local_parent = pe if pe != -1 else (min(incident) if incident else -1)
        for e, nb in adj[node]:
            if e == pe:
                continue
            if local_parent != -1 and e != local_parent:
                pairs.add((min(local_parent, e), max(local_parent, e)))
            if nb not in visited:
                visited.add(nb)
                parent_edge[nb] = e
                dq.append(nb)
    return sorted(pairs)


def _greedy_color(n_items, conflicts):
    """Greedy graph coloring; returns a list of int32 arrays (index groups per color)."""
    color = [-1] * n_items
    for i in range(n_items):
        used = {color[j] for j in conflicts[i] if color[j] >= 0}
        c = 0
        while c in used:
            c += 1
        color[i] = c
    groups = defaultdict(list)
    for i, c in enumerate(color):
        groups[c].append(i)
    return [np.asarray(groups[c], dtype=np.int32) for c in sorted(groups)]


def _color_edges(edges, n_seg):
    """Two segments conflict (must differ in color) iff they share a node."""
    node_segs = defaultdict(list)
    for e, (a, b) in enumerate(edges):
        node_segs[a].append(e)
        node_segs[b].append(e)
    conflicts = [set() for _ in range(n_seg)]
    for segs in node_segs.values():
        for i in segs:
            for j in segs:
                if i != j:
                    conflicts[i].add(j)
    return _greedy_color(n_seg, conflicts)


def _color_bend_pairs(bend_pairs):
    """Two bend pairs conflict iff they share a segment."""
    n = len(bend_pairs)
    seg_pairs = defaultdict(list)
    for k, (a, b) in enumerate(bend_pairs):
        seg_pairs[a].append(k)
        seg_pairs[b].append(k)
    conflicts = [set() for _ in range(n)]
    for ks in seg_pairs.values():
        for i in ks:
            for j in ks:
                if i != j:
                    conflicts[i].add(j)
    return _greedy_color(n, conflicts)


def _color_segments(bend_pairs, n_seg):
    """Two segments conflict (SCR orientation sweep) iff they share a bend pair."""
    conflicts = [set() for _ in range(n_seg)]
    for a, b in bend_pairs:
        conflicts[a].add(b)
        conflicts[b].add(a)
    return _greedy_color(n_seg, conflicts)


def _segment_pair_csr(bend_pairs, rest_darboux, seg_len, n_seg):
    """Per-segment incident bend pairs as CSR arrays for the SCR orientation kernel.

    Returns (offsets[n_seg+1], neighbor_seg[E], rest_quat[E], role[E], scale[E]) where
    role=1 marks the segment as the pair's first member (uses q_rest^-1) and role=0 the
    second (uses q_rest). scale = 4/l_pair is the beam-theory bend length scaling
    (l_pair = the lower-index segment's rest length), so k_bt = scr_bend * scale.
    """
    inc = defaultdict(list)
    for k, (a, b) in enumerate(bend_pairs):
        lp = float(seg_len[min(a, b)])
        s = 4.0 / lp if lp > 1e-12 else 0.0
        inc[a].append((b, k, 1, s))
        inc[b].append((a, k, 0, s))
    offsets = [0]
    nb, rest, role, scale = [], [], [], []
    for e in range(n_seg):
        for (neighbor, k, r, s) in inc.get(e, ()):
            nb.append(neighbor)
            rest.append(rest_darboux[k])
            role.append(r)
            scale.append(s)
        offsets.append(len(nb))
    rest = np.asarray(rest, dtype=np.float32) if rest else np.zeros((1, 4), np.float32)
    return (np.asarray(offsets, np.int32), np.asarray(nb or [0], np.int32), rest,
            np.asarray(role or [0], np.int32), np.asarray(scale or [0.0], np.float32))


def _lumped_node_mass(edges, seg_len, n_nodes, density):
    """Node mass lumped from adjacent segment lengths: m_i = 0.5 density Sum(adjacent l)."""
    mass = np.zeros(n_nodes, dtype=np.float64)
    for (a, b), length in zip(edges, seg_len):
        mass[a] += 0.5 * density * length
        mass[b] += 0.5 * density * length
    return mass


# --------------------------------------------------------------------------- #
# Rod object + step driver
# --------------------------------------------------------------------------- #
class CosseratRod:
    """Per-segment PBD Cosserat rod. Chain by default; pass ``edges`` for a tree/graph."""

    def __init__(self, positions, device, edges=None, bend_pairs=None,
                 node_mass=1.0, seg_inertia=1.0,
                 bend_stiffness=(1.0, 1.0), twist_stiffness=1.0,
                 fix_root_pos=True, fix_root_orient=True, fix_tip_orient=False,
                 fixed_nodes=None, fixed_segments=None, freeze_positions=False,
                 orientation_mode="pbd", scr_stretch_stiffness=1.0, scr_bend_stiffness=0.5,
                 density=None):
        positions = np.asarray(positions, dtype=np.float32).reshape((-1, 3))
        self.device = device
        self.n_nodes = len(positions)

        is_chain = edges is None
        if is_chain:
            edges = [(i, i + 1) for i in range(self.n_nodes - 1)]
        edges = [(int(a), int(b)) for a, b in edges]
        self.edges = edges
        self.n_seg = len(edges)
        # Root the bend-pair spanning tree and the frame transport at the first fixed node.
        root = int(fixed_nodes[0]) if fixed_nodes else 0
        if bend_pairs is None:
            bend_pairs = derive_bend_pairs(edges, root=root)
        bend_pairs = [(int(a), int(b)) for a, b in bend_pairs]
        self.bend_pairs = bend_pairs
        self.n_pair = len(bend_pairs)

        q = _chain_frames(positions) if is_chain else _tree_frames(positions, edges, root=root)
        rd = _rest_darboux(q, bend_pairs)
        seg_len = np.asarray([np.linalg.norm(positions[b] - positions[a]) for a, b in edges], dtype=np.float32)

        # Boundary conditions. Mass is either uniform (node_mass) or length-lumped from
        # adjacent segments (density) so behavior is resolution-consistent.
        if freeze_positions:
            inv_mass = np.zeros(self.n_nodes, dtype=np.float32)
        elif density is not None:
            m = _lumped_node_mass(edges, seg_len, self.n_nodes, float(density))
            inv_mass = np.where(m > 0.0, 1.0 / np.maximum(m, 1e-12), 0.0).astype(np.float32)
        else:
            inv_mass = np.full(self.n_nodes, 1.0 / node_mass, dtype=np.float32)
        if not freeze_positions:
            if fixed_nodes is not None:
                for i in fixed_nodes:
                    inv_mass[int(i)] = 0.0
            elif fix_root_pos:
                inv_mass[0] = 0.0
        inv_mass_q = np.full(self.n_seg, 1.0 / seg_inertia, dtype=np.float32)
        if fixed_segments is not None:
            for s in fixed_segments:
                inv_mass_q[int(s)] = 0.0
        else:
            if fix_root_orient:
                inv_mass_q[0] = 0.0
            if fix_tip_orient:
                inv_mass_q[-1] = 0.0

        kbt = np.tile(np.array([bend_stiffness[0], bend_stiffness[1], twist_stiffness], np.float32),
                      (max(self.n_pair, 1), 1))

        seg_n0 = np.asarray([a for a, _ in edges], dtype=np.int32)
        seg_n1 = np.asarray([b for _, b in edges], dtype=np.int32)
        bend_a = np.asarray([a for a, _ in bend_pairs] or [0], dtype=np.int32)
        bend_b = np.asarray([b for _, b in bend_pairs] or [0], dtype=np.int32)

        # Warp arrays.
        self.x = wp.array(positions, dtype=wp.vec3, device=device)
        self.p = wp.zeros(self.n_nodes, dtype=wp.vec3, device=device)
        self.v = wp.zeros(self.n_nodes, dtype=wp.vec3, device=device)
        self.q = wp.array(q.astype(np.float32), dtype=wp.quat, device=device)
        self.u = wp.array(q.astype(np.float32).copy(), dtype=wp.quat, device=device)
        self.omega = wp.zeros(self.n_seg, dtype=wp.vec3, device=device)
        self.rest_darboux = wp.array(rd.astype(np.float32), dtype=wp.quat, device=device)
        self.rest_len = wp.array(seg_len, dtype=wp.float32, device=device)
        self.stretch_multiplier = wp.array(
            np.ones(max(self.n_seg, 1), dtype=np.float32),
            dtype=wp.float32,
            device=device,
        )
        self.kbt = wp.array(kbt, dtype=wp.vec3, device=device)
        self.seg_n0 = wp.array(seg_n0, device=device)
        self.seg_n1 = wp.array(seg_n1, device=device)
        self.bend_a = wp.array(bend_a, device=device)
        self.bend_b = wp.array(bend_b, device=device)
        self.inv_mass = wp.array(inv_mass, dtype=wp.float32, device=device)
        self.inv_mass_q = wp.array(inv_mass_q, dtype=wp.float32, device=device)

        # Color groups (device arrays of item indices) for race-free colored Gauss-Seidel:
        # no two edges in a stretch color share a node; no two bend pairs in a bend color
        # share a segment. Symmetric branching relies on symmetric frames (see _tree_frames),
        # not on the sweep order.
        self.stretch_groups = [wp.array(g, device=device) for g in _color_edges(edges, self.n_seg)]
        self.bend_groups = ([wp.array(g, device=device) for g in _color_bend_pairs(bend_pairs)]
                            if self.n_pair > 0 else [])
        self._rest_len_np = seg_len

        # SCR closed-form orientation mode: per-segment coloring + incident-bend-pair CSR.
        if orientation_mode not in ("pbd", "scr"):
            raise ValueError(f"orientation_mode must be 'pbd' or 'scr', got {orientation_mode!r}")
        self.orientation_mode = orientation_mode
        self.scr_stretch_stiffness = float(scr_stretch_stiffness)
        self.scr_bend_stiffness = float(scr_bend_stiffness)
        self.seg_groups = [wp.array(g, device=device) for g in _color_segments(bend_pairs, self.n_seg)]
        off, nb, rest, role, scale = _segment_pair_csr(bend_pairs, rd, seg_len, self.n_seg)
        self.seg_pair_off = wp.array(off, dtype=wp.int32, device=device)
        self.seg_pair_nb = wp.array(nb, dtype=wp.int32, device=device)
        self.seg_pair_rest = wp.array(rest, dtype=wp.quat, device=device)
        self.seg_pair_role = wp.array(role, dtype=wp.int32, device=device)
        self.seg_pair_scale = wp.array(scale, dtype=wp.float32, device=device)

    def positions(self):
        return self.x.numpy()

    def quaternions(self):
        return self.q.numpy()

    def total_length(self):
        p = self.positions()
        return float(np.sum([np.linalg.norm(p[b] - p[a]) for a, b in self.edges]))

    def rest_length(self):
        return float(np.sum(self._rest_len_np))


def predict(rod, dt, gravity=(0.0, 0.0, 0.0)):
    """Predict node positions and segment orientations for one substep."""
    dev, N, S = rod.device, rod.n_nodes, rod.n_seg
    scr = rod.orientation_mode == "scr"
    wp.launch(predict_positions, dim=N, inputs=[rod.x, rod.v, rod.inv_mass, wp.vec3(*gravity), dt, rod.p], device=dev)
    if scr:
        wp.copy(rod.u, rod.q)  # SCR orientations are quasi-static (J=0): no rotational inertia/prediction
    else:
        wp.launch(predict_orientations, dim=S, inputs=[rod.q, rod.omega, rod.inv_mass_q, dt, rod.u], device=dev)


def project(rod, iterations=10, ks=1.0, floor_enabled=False, floor_z=0.0, floor_radius=0.0):
    """Project internal Cosserat constraints on the current prediction."""
    dev, N = rod.device, rod.n_nodes
    scr = rod.orientation_mode == "scr"
    orient = 0 if scr else 1  # SCR projects positions only; the closed-form pass owns orientation
    for _ in range(iterations):
        for g in rod.stretch_groups:
            wp.launch(stretch_shear, dim=int(g.shape[0]),
                      inputs=[g, rod.p, rod.u, rod.inv_mass, rod.inv_mass_q, rod.seg_n0, rod.seg_n1,
                              rod.rest_len, rod.stretch_multiplier, ks, orient, int(g.shape[0])], device=dev)
        if scr:
            for g in rod.seg_groups:
                wp.launch(scr_orientation, dim=int(g.shape[0]),
                          inputs=[g, rod.u, rod.p, rod.inv_mass_q, rod.seg_n0, rod.seg_n1, rod.rest_len,
                                  rod.seg_pair_off, rod.seg_pair_nb, rod.seg_pair_rest, rod.seg_pair_role,
                                  rod.seg_pair_scale, rod.scr_stretch_stiffness, rod.scr_bend_stiffness,
                                  int(g.shape[0])], device=dev)
        else:
            for g in rod.bend_groups:
                wp.launch(bend_twist, dim=int(g.shape[0]),
                          inputs=[g, rod.u, rod.inv_mass_q, rod.bend_a, rod.bend_b,
                                  rod.rest_darboux, rod.kbt, int(g.shape[0])], device=dev)
        if floor_enabled:
            wp.launch(floor_collision, dim=N,
                      inputs=[rod.p, rod.inv_mass, float(floor_z), float(floor_radius)], device=dev)


def finalize(rod, dt, lin_damping=0.0, ang_damping=0.0):
    """Commit a projected prediction and update linear/angular velocities."""
    dev, N, S = rod.device, rod.n_nodes, rod.n_seg
    wp.launch(finalize_positions, dim=N, inputs=[rod.x, rod.p, rod.v, rod.inv_mass, dt, lin_damping], device=dev)
    wp.launch(finalize_orientations, dim=S, inputs=[rod.q, rod.u, rod.omega, rod.inv_mass_q, dt, ang_damping], device=dev)


def step(rod, dt, gravity=(0.0, 0.0, 0.0), iterations=10, ks=1.0,
         lin_damping=0.0, ang_damping=0.0, floor_enabled=False, floor_z=0.0, floor_radius=0.0):
    """One PBD substep, preserving the original convenience API."""
    predict(rod, dt, gravity)
    project(rod, iterations, ks, floor_enabled, floor_z, floor_radius)
    finalize(rod, dt, lin_damping, ang_damping)


def run(rod, steps, dt, gravity=(0.0, 0.0, 0.0), iterations=10, ks=1.0,
        lin_damping=0.0, ang_damping=0.0, floor_enabled=False, floor_z=0.0, floor_radius=0.0):
    for _ in range(steps):
        step(rod, dt, gravity, iterations, ks, lin_damping, ang_damping,
             floor_enabled, floor_z, floor_radius)
    wp.synchronize()
