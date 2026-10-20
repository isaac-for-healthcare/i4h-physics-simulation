# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Iterative PBD Cosserat rod (Kugelstadt & Schoemer 2016), the Jacobi path of omniendo's pbd_cosserat.py, for a
chain (a ``rod`` asset) or an anatomical centreline graph (a ``vessel_mesh`` asset with a ``centreline`` block).

Ported from omniendo/xpbd_rod/pbd_cosserat.py: ``qmul`` / ``qconj`` (:33-43), ``predict_orientations``
(:69-88), ``stretch_shear_jacobi`` (:186-234), ``bend_twist_jacobi`` (:235-313), the two self-clearing apply
kernels (:314-366) and ``finalize_orientations`` (:460-475); the rest Darboux quaternions from ``_rest_darboux``
(:617-624). The topology is explicit as in the source: ``seg (V, S) vec2i`` node pairs and ``pair (V, K) vec2i``
segment pairs, so a chain (``seg = (i, i+1)``, ``pair = (k, k+1)``) and a tree with junctions and cycles run the
same kernels. Positions live in ``scene.q`` (``p``) with ``scene.q_prev`` as ``x``; ``particles`` predicts and
finalizes them (``predict_positions`` and ``finalize_positions`` are not ported), this system owns the segment
frames, their predicted copies and body-space angular velocities. Plain PBD: ``stretch``, ``bend`` and ``twist``
are multipliers, not compliances; ``stretch_scale (E, S)`` and ``bend_scale (E, K)`` are per-element live
multipliers (omniendo's ``stretch_multiplier`` and the control layouts' bend multipliers), ones by default. The
colored Gauss-Seidel path is not ported: its per-colour launches have ragged dims that do not batch over envs, the
Jacobi path is four launches of fixed dims; ``sweeps`` repeats them within one runner iteration (omniendo's
``iterations`` of the centreline solve).

Inactive nodes (``scene.flags``) hold their position and frame; a segment or pair touching one is skipped (and
its angular velocity is zeroed at finalize, as for a locked segment). One mesh variant only. Products read by
``catheter_control``, the renderers and the snapshot: ``frame`` / ``frame_pred`` (the frames of env ``e`` start
at index ``e * S``; for a chain segment k is node k's frame), ``frame_count`` (= S), ``bend_count`` (= K) and the
live input ``bend_offset (E, K) vec3`` folded into the imaginary part of each pair's rest Darboux quaternion
(the steerable tip on a chain, the control layouts' rest bend on a tree).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
from i4h_physics.checks import check_int
from i4h_physics.scene import ACTIVE, restore_variant_rows
from i4h_physics.systems.base import System

E3 = wp.constant(wp.vec3(0.0, 0.0, 1.0))


# pbd_cosserat.py:479-485 numpy quaternion helpers. Edits: none.
def _qmul_np(a, b):
    av, aw = a[:3], a[3]
    bv, bw = b[:3], b[3]
    v = aw * bv + bw * av + np.cross(av, bv)
    return np.array([v[0], v[1], v[2], aw * bw - float(np.dot(av, bv))], dtype=np.float64)


def _qconj_np(a):
    return np.array([-a[0], -a[1], -a[2], a[3]], dtype=np.float64)


def rest_darboux_quats(frames: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """pbd_cosserat.py:617-624 (_rest_darboux): ``(K, 4)`` unit quaternions ``conj(q_a) q_b`` with ``w >= 0`` for the
    segment pairs ``pairs (K, 2)`` of one variant's ``(S, 4)`` segment frames."""
    rd = np.zeros((len(pairs), 4), dtype=np.float64)
    for k, (a, b) in enumerate(pairs):
        rel = _qmul_np(_qconj_np(frames[a]), frames[b])
        if rel[3] < 0.0:
            rel = -rel
        rd[k] = rel
    return rd.astype(np.float32)


def chain_topology(P: int) -> tuple[np.ndarray, np.ndarray]:
    """Segments ``(i, i+1)`` and bend pairs ``(k, k+1)`` of a chain of ``P`` nodes (pbd_cosserat.py:773-775)."""
    seg = np.stack([np.arange(P - 1), np.arange(1, P)], axis=1).astype(np.int32)
    pair = np.stack([np.arange(P - 2), np.arange(1, P - 1)], axis=1).astype(np.int32)
    return seg, pair


@dataclass(eq=False)
class RodPbd(System):
    body: str  # rod or centreline-tree asset id in YAML; the Body instance after construction
    stretch: float = 1.0  # live, per env: stretch/shear multiplier (omniendo ks)
    bend: float = 1.0  # live, per env: bend multiplier (kbt.x = kbt.y)
    twist: float = 1.0  # live, per env: twist multiplier (kbt.z)
    angular_damping: float = 0.0  # live, per env: omega *= (1 - d) at finalize (omniendo ang_damping)
    sweeps: int = 1  # structural: Jacobi sweeps per runner iteration (omniendo centreline ``iterations``)

    structural = System.structural + ("sweeps",)

    def __post_init__(self):
        scene = self.scene
        self.body = b = scene.body(self.body)
        E, dev = scene.envs, scene.device
        P = b.count
        self.sweeps = check_int("rod_pbd.sweeps", self.sweeps, 1)
        if b.rod is not None:
            m = b.rod
            if m.frames.shape[0] != 1:
                raise ValueError("rod_pbd: rod systems support a single mesh variant")
            seg, pair = chain_topology(P)  # the loader guarantees P >= 3
            frames = np.ascontiguousarray(m.frames[:, : P - 1])  # segment k's frame is node k's (the tangent of edge k)
            rest_len, inv_mass_q = m.rest_len, np.ascontiguousarray(m.inv_mass_q[:, : P - 1])
        elif b.tree is not None:
            t = b.tree
            if t.frames.shape[0] != 1:
                raise ValueError("rod_pbd: rod systems support a single mesh variant")
            seg, pair = t.edges[0], t.bend_pairs[0]
            frames, rest_len, inv_mass_q = t.frames, t.rest_len, t.inv_mass_q
        else:
            raise ValueError(f"rod_pbd: body {b.name!r} has neither a rod nor a centreline tree record")
        S, K = len(seg), len(pair)
        self.P, self.S, self.K = P, S, K
        self.frame_count = S  # segment frames
        self.bend_count = K  # bend pairs
        n, ns = E * P, E * S
        rd = np.stack([rest_darboux_quats(f, pair) for f in frames])
        scene.check_memory({
            "variants": frames.nbytes + rd.nbytes + rest_len.nbytes + inv_mass_q.nbytes + (S + K) * 8,
            "frames": ns * (16 * 2 + 12) + E * K * 12,
            "scales": E * (S + K) * 4,
            "accumulators": n * 16 + ns * 20,
        })
        self.variant = b.variant
        self.seg = wp.array(seg[None], dtype=wp.vec2i, device=dev)  # (V, S) node pairs
        material = b.rod_material
        self.pair = material.pair if material is not None and material.backend == 'pbd' else wp.array(pair[None], dtype=wp.vec2i, device=dev)  # (V, K)
        self.frames0 = wp.array(frames, dtype=wp.quat, device=dev)  # (V, S)
        self.rest_darboux = wp.array(rd, dtype=wp.quat, device=dev)  # (V, K)
        self.rest_len = wp.array(rest_len, dtype=wp.float32, device=dev)  # (V, S)
        self.inv_mass_q = wp.array(inv_mass_q, dtype=wp.float32, device=dev)  # (V, S)
        self.frame = wp.empty(ns, dtype=wp.quat, device=dev)  # body-local: index env * S + segment
        self.frame_pred = wp.empty(ns, dtype=wp.quat, device=dev)
        self.omega = wp.zeros(ns, dtype=wp.vec3, device=dev)
        from i4h_physics.regions import RodControls
        material = b.rod_material
        if b.regions is not None and material is None:
            raise ValueError("rod_pbd: regional body requires rod_pbd_material before the solver")
        if material is not None and material.backend != "pbd":
            raise ValueError("rod_pbd: mismatched material backend")
        if b.rod_controls is None:
            b.rod_controls = RodControls(scene, S, K)
        controls = b.rod_controls
        self.bend_offset = controls.bend_offset
        self.stretch_scale = controls.stretch_scale
        self.bend_scale = controls.bend_scale
        self.unit_twist = wp.ones((E, K), dtype=float, device=dev)
        self.effective_stretch, self.effective_bend = self.stretch_scale, self.bend_scale
        self.effective_twist = self.unit_twist
        self._live("stretch", "bend", "twist", "angular_damping")
        # Jacobi accumulators: the apply kernels clear them, so they are zero between launches.
        self.pos_corr = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.pos_count = wp.zeros(n, dtype=wp.int32, device=dev)
        self.ori_corr = wp.zeros(ns, dtype=wp.vec4, device=dev)
        self.ori_count = wp.zeros(ns, dtype=wp.int32, device=dev)
        self.reset(scene, wp.ones(E, dtype=wp.bool, device=dev))

    def prepare(self, scene):
        super().prepare(scene)
        material = self.body.rod_material
        self.regional = material is not None and material.active
        self.effective_stretch = material.stretch if self.regional else self.stretch_scale
        self.effective_bend = material.bend if self.regional else self.bend_scale
        self.effective_twist = material.twist if self.regional else self.unit_twist


    def substep_start(self, scene, dt, s, substeps):
        E, dev, S = scene.envs, scene.device, self.S
        wp.launch(predict_orientations, dim=(E, S),
                  inputs=[self.frame, self.omega, self.inv_mass_q, self.variant, self.seg, scene.field("flags", self.body), S, dt],
                  outputs=[self.frame_pred], device=dev)

    def iteration(self, scene, dt, k, iterations):
        b, E, dev, P, S, K = self.body, scene.envs, scene.device, self.P, self.S, self.K
        for _ in range(self.sweeps):
            wp.launch(cosserat_jacobi, dim=(E, S + K),
                      inputs=[scene.field("q", b), self.frame_pred, scene.field("inv_mass", b), scene.field("flags", b), self.inv_mass_q,
                              self.rest_len, self.rest_darboux, self.bend_offset, self.effective_bend, self.effective_twist, self.variant, self.seg, self.pair,
                              self.stretch, self.effective_stretch, self.bend, self.twist, P, S],
                      outputs=[self.pos_corr, self.pos_count, self.ori_corr, self.ori_count], device=dev)
            wp.launch(apply_corrections, dim=(E, max(P, S)),
                      inputs=[scene.field("inv_mass", b), scene.field("flags", b), self.inv_mass_q, self.variant, self.seg, P, S],
                      outputs=[scene.field("q", b), self.pos_corr, self.pos_count, self.frame_pred, self.ori_corr, self.ori_count], device=dev)

    def substep_end(self, scene, dt):
        E, dev, S = scene.envs, scene.device, self.S
        wp.launch(finalize_orientations, dim=(E, S),
                  inputs=[self.frame_pred, self.inv_mass_q, self.variant, self.seg, scene.field("flags", self.body), self.angular_damping,
                          S, dt],
                  outputs=[self.frame, self.omega], device=dev)

    def reset(self, scene, mask):
        """Selected envs: segment frames back to the variant's rest frames, angular velocity, tip offset and the
        per-element scales to their defaults."""
        E, dev, S, K = scene.envs, scene.device, self.S, self.K
        for dst in (self.frame, self.frame_pred):
            wp.launch(restore_variant_rows, dim=(E, S), inputs=[mask, self.variant, self.frames0],
                      outputs=[dst.reshape((E, S))], device=dev)
        self.clear(mask, (self.omega, wp.vec3(0.0, 0.0, 0.0)))
        self.clear(mask, (self.stretch_scale, wp.float32(1.0)))
        if K:
            self.clear(mask, (self.bend_offset, wp.vec3(0.0, 0.0, 0.0)))
            self.clear(mask, (self.bend_scale, wp.float32(1.0)))


# --- kernels -----------------------------------------------------------------------------------------
# pbd_cosserat.py:33-43 quaternion helpers -- Hamilton product, {x, y, z, w}. Edits: none.


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


@wp.func
def segment_active(flags: wp.array2d(dtype=wp.int32), env: wp.int32, e: wp.vec2i) -> bool:
    """A segment spanning nodes ``e[0]`` and ``e[1]`` of env ``env`` is live when both are active."""
    return (flags[env, e[0]] & ACTIVE) != 0 and (flags[env, e[1]] & ACTIVE) != 0


# Ported from pbd_cosserat.py:61-79 (predict_orientations). Edits: (env, segment) prologue; inverse
# rotational mass per variant; inactive segments keep their frame.
@wp.kernel
def predict_orientations(
    q: wp.array(dtype=wp.quat),
    omega: wp.array(dtype=wp.vec3),
    inv_mass_q: wp.array2d(dtype=wp.float32),
    variant: wp.array(dtype=wp.int32),
    seg: wp.array2d(dtype=wp.vec2i),
    flags: wp.array2d(dtype=wp.int32),
    S: wp.int32,
    dt: wp.float32,
    u: wp.array(dtype=wp.quat),
):
    env, e = wp.tid()
    v = variant[env]
    i = env * S + e
    if inv_mass_q[v, e] > 0.0 and segment_active(flags, env, seg[v, e]):
        w = omega[i]
        # Body-space angular velocity: q_dot = 1/2 q (x) (0, omega). This matches the
        # body-space omega recovered in finalize_orientations (omega = Im(2 conj(q) u)/dt).
        dq = qmul(q[i], wp.quat(w[0], w[1], w[2], 0.0))
        n = wp.normalize(wp.vec4(q[i][0] + 0.5 * dt * dq[0], q[i][1] + 0.5 * dt * dq[1],
                                 q[i][2] + 0.5 * dt * dq[2], q[i][3] + 0.5 * dt * dq[3]))
        u[i] = wp.quat(n[0], n[1], n[2], n[3])
    else:
        u[i] = q[i]


# Ported from pbd_cosserat.py:186-234 (stretch_shear_jacobi). Edits: (env, segment) prologue; the segment's
# nodes come from ``seg`` (omniendo seg_n0 / seg_n1), read from the body's (E, P) views at (env, node);
# ks * stretch_multiplier[e] becomes the live stretch[env] * stretch_scale[env, e]; rest length and inverse
# rotational mass per variant; accumulators are body-local; a segment with an inactive node is skipped;
# ``orient`` is gone (omniendo's 0 selected the unported SCR mode, so it was always 1).
@wp.func
def stretch_shear_jacobi(
    env: wp.int32,
    e: wp.int32,
    p: wp.array2d(dtype=wp.vec3),
    u: wp.array(dtype=wp.quat),
    inv_mass: wp.array2d(dtype=wp.float32),
    flags: wp.array2d(dtype=wp.int32),
    inv_mass_q: wp.array2d(dtype=wp.float32),
    rest_len: wp.array2d(dtype=wp.float32),
    variant: wp.array(dtype=wp.int32),
    seg: wp.array2d(dtype=wp.vec2i),
    stretch: wp.array(dtype=wp.float32),
    stretch_scale: wp.array2d(dtype=wp.float32),
    P: wp.int32,
    S: wp.int32,
    position_corrections: wp.array(dtype=wp.vec3),
    position_counts: wp.array(dtype=wp.int32),
    orientation_corrections: wp.array(dtype=wp.vec4),
    orientation_counts: wp.array(dtype=wp.int32),
):
    """Accumulate one stretch/shear projection from an immutable iteration snapshot."""
    v = variant[env]
    ends = seg[v, e]
    multiplier = stretch[env] * stretch_scale[env, e]
    if wp.abs(multiplier) <= 1.0e-12 or not segment_active(flags, env, ends):
        return

    n0 = env * P + ends[0]
    n1 = env * P + ends[1]
    i0 = ends[0]  # the views take (env, node), the accumulators n0 / n1
    i1 = ends[1]
    s_index = env * S + e
    w0 = inv_mass[env, i0]
    w1 = inv_mass[env, i1]
    wq = inv_mass_q[v, e]
    rest_length = rest_len[v, e]
    q0 = u[s_index]

    d3 = wp.quat_rotate(q0, E3)
    gamma = (p[env, i1] - p[env, i0]) / rest_length - d3
    denom = (w0 + w1) / rest_length + wq * 4.0 * rest_length + 1.0e-6
    gamma = (gamma / denom) * multiplier

    if w0 > 0.0:
        wp.atomic_add(position_corrections, n0, w0 * gamma)
        wp.atomic_add(position_counts, n0, 1)
    if w1 > 0.0:
        wp.atomic_add(position_corrections, n1, -w1 * gamma)
        wp.atomic_add(position_counts, n1, 1)
    if wq > 0.0:
        qe3 = wp.quat(-q0[1], q0[0], -q0[3], q0[2])
        cq = qmul(wp.quat(gamma[0], gamma[1], gamma[2], 0.0), qe3)
        s = 2.0 * wq * rest_length
        wp.atomic_add(orientation_corrections, s_index, wp.vec4(s * cq[0], s * cq[1], s * cq[2], s * cq[3]))
        wp.atomic_add(orientation_counts, s_index, 1)


# Ported from pbd_cosserat.py:235-313 (bend_twist_jacobi). Edits: (env, pair) prologue; the pair's segments
# come from ``pair`` (omniendo bend_a / bend_b); kbt[k] becomes (bend, bend, twist)[env] with the live per-pair
# bend_scale on the two bend components; the rest Darboux is per variant with the live bend_offset added to its
# imaginary part and w recomputed (the +-rd selection below is unchanged); inverse rotational mass per variant;
# accumulators body-local; a pair with an inactive segment is skipped.
@wp.func
def bend_twist_jacobi(
    env: wp.int32,
    k: wp.int32,
    u: wp.array(dtype=wp.quat),
    inv_mass_q: wp.array2d(dtype=wp.float32),
    rest_darboux: wp.array2d(dtype=wp.quat),
    bend_offset: wp.array2d(dtype=wp.vec3),
    bend_scale: wp.array2d(dtype=wp.float32),
    twist_scale: wp.array2d(dtype=wp.float32),
    variant: wp.array(dtype=wp.int32),
    seg: wp.array2d(dtype=wp.vec2i),
    pair: wp.array2d(dtype=wp.vec2i),
    flags: wp.array2d(dtype=wp.int32),
    bend: wp.array(dtype=wp.float32),
    twist: wp.array(dtype=wp.float32),
    S: wp.int32,
    orientation_corrections: wp.array(dtype=wp.vec4),
    orientation_counts: wp.array(dtype=wp.int32),
):
    """Accumulate one bend/twist projection from the same iteration snapshot."""
    v = variant[env]
    pr = pair[v, k]
    if not segment_active(flags, env, seg[v, pr[0]]) or not segment_active(flags, env, seg[v, pr[1]]):
        return
    ia = env * S + pr[0]
    ib = env * S + pr[1]
    q0 = u[ia]
    q1 = u[ib]
    wq0 = inv_mass_q[v, pr[0]]
    wq1 = inv_mass_q[v, pr[1]]
    kbend = bend[env] * bend_scale[env, k]
    kb = wp.vec3(kbend, kbend, twist[env] * twist_scale[env, k])
    if (
        (wq0 <= 0.0 and wq1 <= 0.0)
        or (
            wp.abs(kb[0]) <= 1.0e-12
            and wp.abs(kb[1]) <= 1.0e-12
            and wp.abs(kb[2]) <= 1.0e-12
        )
    ):
        return

    omega = qmul(qconj(q0), q1)
    rd0 = rest_darboux[v, k]
    o = bend_offset[env, k]
    vx = rd0[0] + o[0]
    vy = rd0[1] + o[1]
    vz = rd0[2] + o[2]
    v2 = vx * vx + vy * vy + vz * vz
    rd = wp.quat(vx, vy, vz, wp.sqrt(wp.max(0.0, 1.0 - v2)))
    if v2 > 1.0:  # an offset beyond the unit ball (a large tip bend) would leave rd non-unit; inside it rd is exact
        rd = wp.normalize(rd)
    om_minus = wp.vec4(omega[0] - rd[0], omega[1] - rd[1], omega[2] - rd[2], omega[3] - rd[3])
    om_plus = wp.vec4(omega[0] + rd[0], omega[1] + rd[1], omega[2] + rd[2], omega[3] + rd[3])
    d = om_minus
    if wp.dot(om_minus, om_minus) > wp.dot(om_plus, om_plus):
        d = om_plus

    denom = wq0 + wq1 + 1.0e-6
    om_s = wp.quat(d[0] * kb[0] / denom, d[1] * kb[1] / denom, d[2] * kb[2] / denom, 0.0)
    cq0 = qmul(q1, om_s)
    cq1 = qmul(q0, om_s)
    if wq0 > 0.0:
        wp.atomic_add(orientation_corrections, ia, wp.vec4(wq0 * cq0[0], wq0 * cq0[1], wq0 * cq0[2], wq0 * cq0[3]))
        wp.atomic_add(orientation_counts, ia, 1)
    if wq1 > 0.0:
        wp.atomic_add(orientation_corrections, ib, wp.vec4(-wq1 * cq1[0], -wq1 * cq1[1], -wq1 * cq1[2], -wq1 * cq1[3]))
        wp.atomic_add(orientation_counts, ib, 1)


# One launch per sweep for both projections (they accumulate into the correction buffers and read only the positions
# and the iteration's frame snapshot, which the apply writes afterwards): threads [0, S) are segments, [S, S + K) pairs.
@wp.kernel
def cosserat_jacobi(
    p: wp.array2d(dtype=wp.vec3),
    u: wp.array(dtype=wp.quat),
    inv_mass: wp.array2d(dtype=wp.float32),
    flags: wp.array2d(dtype=wp.int32),
    inv_mass_q: wp.array2d(dtype=wp.float32),
    rest_len: wp.array2d(dtype=wp.float32),
    rest_darboux: wp.array2d(dtype=wp.quat),
    bend_offset: wp.array2d(dtype=wp.vec3),
    bend_scale: wp.array2d(dtype=wp.float32),
    twist_scale: wp.array2d(dtype=wp.float32),
    variant: wp.array(dtype=wp.int32),
    seg: wp.array2d(dtype=wp.vec2i),
    pair: wp.array2d(dtype=wp.vec2i),
    stretch: wp.array(dtype=wp.float32),
    stretch_scale: wp.array2d(dtype=wp.float32),
    bend: wp.array(dtype=wp.float32),
    twist: wp.array(dtype=wp.float32),
    P: wp.int32,
    S: wp.int32,
    position_corrections: wp.array(dtype=wp.vec3),
    position_counts: wp.array(dtype=wp.int32),
    orientation_corrections: wp.array(dtype=wp.vec4),
    orientation_counts: wp.array(dtype=wp.int32),
):
    env, i = wp.tid()
    if i < S:
        stretch_shear_jacobi(env, i, p, u, inv_mass, flags, inv_mass_q, rest_len, variant, seg, stretch, stretch_scale, P, S,
                             position_corrections, position_counts, orientation_corrections, orientation_counts)
    else:
        bend_twist_jacobi(env, i - S, u, inv_mass_q, rest_darboux, bend_offset, bend_scale, twist_scale, variant, seg, pair, flags, bend,
                          twist, S, orientation_corrections, orientation_counts)


# Ported from pbd_cosserat.py:314-327 (apply_cosserat_jacobi_positions). Edits: (env, node) prologue;
# positions are the body's (E, P) view, accumulators body-local; inactive nodes skipped.
@wp.kernel
def apply_corrections(
    inv_mass: wp.array2d(dtype=wp.float32),
    flags: wp.array2d(dtype=wp.int32),
    inv_mass_q: wp.array2d(dtype=wp.float32),
    variant: wp.array(dtype=wp.int32),
    seg: wp.array2d(dtype=wp.vec2i),
    P: wp.int32,
    S: wp.int32,
    p: wp.array2d(dtype=wp.vec3),
    corrections: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=wp.int32),
    u: wp.array(dtype=wp.quat),
    ori_corrections: wp.array(dtype=wp.vec4),
    ori_counts: wp.array(dtype=wp.int32),
):
    """Averaged node corrections on the first P threads of each env, and on the first S threads the averaged quaternion
    corrections of the segments (pbd_cosserat.py:341-366, apply_cosserat_jacobi_orientations); scratch left clear. One
    launch over (E, max(P, S)) instead of two: the two writes touch disjoint slots (a graph with cycles has S > P)."""
    env, e = wp.tid()
    if e < P:
        i = env * P + e
        count = counts[i]
        if inv_mass[env, e] > 0.0 and (flags[env, e] & ACTIVE) != 0 and count > 0:
            p[env, e] = p[env, e] + corrections[i] / float(count)
        corrections[i] = wp.vec3(0.0)
        counts[i] = 0
    if e >= S:
        return
    v = variant[env]
    j = env * S + e
    ori_count = ori_counts[j]
    if inv_mass_q[v, e] > 0.0 and ori_count > 0 and segment_active(flags, env, seg[v, e]):
        q = u[j]
        correction = ori_corrections[j] / float(ori_count)
        normalized = wp.normalize(wp.vec4(q[0] + correction[0], q[1] + correction[1], q[2] + correction[2], q[3] + correction[3]))
        u[j] = wp.quat(normalized[0], normalized[1], normalized[2], normalized[3])
    ori_corrections[j] = wp.vec4(0.0)
    ori_counts[j] = 0


# Ported from pbd_cosserat.py:460-475 (finalize_orientations). Edits: (env, segment) prologue; damping
# becomes the live angular_damping[env]; inverse rotational mass per variant; inactive segments hold.
@wp.kernel
def finalize_orientations(
    u: wp.array(dtype=wp.quat),
    inv_mass_q: wp.array2d(dtype=wp.float32),
    variant: wp.array(dtype=wp.int32),
    seg: wp.array2d(dtype=wp.vec2i),
    flags: wp.array2d(dtype=wp.int32),
    damping: wp.array(dtype=wp.float32),
    S: wp.int32,
    dt: wp.float32,
    q: wp.array(dtype=wp.quat),
    omega: wp.array(dtype=wp.vec3),
):
    env, e = wp.tid()
    v = variant[env]
    i = env * S + e
    if inv_mass_q[v, e] > 0.0 and segment_active(flags, env, seg[v, e]):
        rel = qmul(qconj(q[i]), u[i])  # omega = Im(2 conj(q^t) u) / dt
        omega[i] = wp.vec3(2.0 * rel[0] / dt, 2.0 * rel[1] / dt, 2.0 * rel[2] / dt) * (1.0 - damping[env])
        q[i] = u[i]
    else:
        omega[i] = wp.vec3(0.0, 0.0, 0.0)
