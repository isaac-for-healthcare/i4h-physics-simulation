# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regional rod products, refreshed inside the captured step from persistent tables."""
import warp as wp


@wp.func
def harmonic(a: float, b: float):
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)


@wp.kernel
def pbd_regions(
    edge: wp.array(dtype=int), pair: wp.array2d(dtype=wp.vec2i),
    stretch: wp.array2d(dtype=float), bend: wp.array2d(dtype=float), twist: wp.array2d(dtype=float),
    radius: wp.array(dtype=float), weighted: int,
    layout_stretch: wp.array2d(dtype=float), layout_bend: wp.array2d(dtype=float), S: int, K: int,
    out_stretch: wp.array2d(dtype=float), out_bend: wp.array2d(dtype=float), out_twist: wp.array2d(dtype=float),
):
    env, i = wp.tid()
    if i < S:
        out_stretch[env, i] = stretch[env, edge[i]] * layout_stretch[env, i]
    if i < K:
        pr = pair[0, i]
        a, b = pr[0], pr[1]
        wa, wb = float(1.0), float(1.0)
        if weighted != 0:
            wa = radius[a]
            wb = radius[b]
        out_bend[env, i] = harmonic(bend[env, edge[a]] * wa, bend[env, edge[b]] * wb) * layout_bend[env, i]
        out_twist[env, i] = harmonic(twist[env, edge[a]] * wa, twist[env, edge[b]] * wb)


@wp.kernel
def direct_regions(
    edge: wp.array(dtype=int), rest_len: wp.array2d(dtype=float), rest: wp.array2d(dtype=wp.vec3),
    ry: wp.array2d(dtype=float), rt: wp.array2d(dtype=float), rb: wp.array2d(dtype=float), rw: wp.array2d(dtype=float),
    cx: wp.array2d(dtype=float), cy: wp.array2d(dtype=float),
    out_young: wp.array2d(dtype=float), out_torsion: wp.array2d(dtype=float),
    out_bend: wp.array2d(dtype=float), out_twist: wp.array2d(dtype=float), preferred: wp.array2d(dtype=wp.vec3),
):
    env, e = wp.tid()
    r = edge[e]
    L = rest_len[0, e]
    out_young[env, e] = ry[env, r]
    out_torsion[env, e] = rt[env, r]
    out_bend[env, e] = rb[env, r]
    out_twist[env, e] = rw[env, r]
    v = wp.vec3(cx[env, r] * L, cy[env, r] * L, 0.0)
    angle = wp.length(v)
    rd = rest[0, e]
    if angle == 0.0:
        preferred[env, e] = rd
    else:
        q = wp.quat(rd[0], rd[1], rd[2], wp.sqrt(wp.max(0.0, 1.0 - wp.dot(rd, rd))))
        composed = q * wp.quat_from_axis_angle(v / angle, angle)
        sign = float(1.0)
        if composed[3] < 0.0:
            sign = -1.0
        preferred[env, e] = wp.vec3(composed[0], composed[1], composed[2]) * sign
