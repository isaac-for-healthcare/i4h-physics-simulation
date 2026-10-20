# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closest-point geometry shared by the contact systems (points, segments and triangles). Ported from omnisurg
instruments/kernels.py:576-660 and 1702-1827; lives here so a contact system needing a closest triangle point
(fluid contact, shell containment) does not import the instrument systems."""

from __future__ import annotations

import warp as wp

# --- closest points ---------------------------------------------------------


@wp.func
def clamp01(value: wp.float32):
    return wp.min(wp.max(value, 0.0), 1.0)


@wp.func
def closest_point_segment_parameter(point: wp.vec3, start: wp.vec3, end: wp.vec3):
    delta = end - start
    length_sq = wp.length_sq(delta)
    if length_sq <= 1.0e-20:
        return wp.float32(0.0)
    return clamp01(wp.dot(point - start, delta) / length_sq)


@wp.func
def closest_degenerate_triangle_barycentric(point: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3):
    ab_t = closest_point_segment_parameter(point, a, b)
    best_distance = wp.length_sq(point - (a + (b - a) * ab_t))
    best = wp.vec3(1.0 - ab_t, ab_t, 0.0)
    bc_t = closest_point_segment_parameter(point, b, c)
    bc_distance = wp.length_sq(point - (b + (c - b) * bc_t))
    if bc_distance < best_distance:
        best_distance = bc_distance
        best = wp.vec3(0.0, 1.0 - bc_t, bc_t)
    ca_t = closest_point_segment_parameter(point, c, a)
    if wp.length_sq(point - (c + (a - c) * ca_t)) < best_distance:
        best = wp.vec3(ca_t, 0.0, 1.0 - ca_t)
    return best


@wp.func
def closest_triangle_barycentric(point: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3):
    ab = b - a
    ac = c - a
    if wp.length_sq(wp.cross(ab, ac)) <= 1.0e-20:
        return closest_degenerate_triangle_barycentric(point, a, b, c)
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
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return wp.vec3(1.0 - v, v, 0.0)
    cp = point - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return wp.vec3(0.0, 0.0, 1.0)
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
def closest_segment_segment_params(p1: wp.vec3, q1: wp.vec3, p2: wp.vec3, q2: wp.vec3):
    """(s on segment 1, t on segment 2, squared distance)."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = wp.length_sq(d1)
    e = wp.length_sq(d2)
    f = wp.dot(d2, r)
    s = wp.float32(0.0)
    t = wp.float32(0.0)
    if a <= 1.0e-12 and e <= 1.0e-12:
        return wp.vec3(0.0, 0.0, wp.length_sq(p1 - p2))
    if a <= 1.0e-12:
        t = clamp01(f / e)
    else:
        c = wp.dot(d1, r)
        if e <= 1.0e-12:
            s = clamp01(-c / a)
        else:
            b = wp.dot(d1, d2)
            denom = a * e - b * b
            if denom != 0.0:
                s = clamp01((b * f - c * e) / denom)
            t_nom = b * s + f
            if t_nom < 0.0:
                t = 0.0
                s = clamp01(-c / a)
            elif t_nom > e:
                t = 1.0
                s = clamp01((b - c) / a)
            else:
                t = t_nom / e
    return wp.vec3(s, t, wp.length_sq((p1 + d1 * s) - (p2 + d2 * t)))


@wp.func
def closest_segment_triangle(segment_start: wp.vec3, segment_end: wp.vec3, p0: wp.vec3, p1: wp.vec3, p2: wp.vec3):
    """Barycentric of the closest triangle point and the segment parameter of the closest capsule point."""
    segment_delta = segment_end - segment_start
    best_barycentric = wp.vec3(1.0, 0.0, 0.0)
    best_alpha = wp.float32(0.0)
    best_distance_sq = wp.float32(3.4028234663852886e38)
    triangle_normal = wp.cross(p1 - p0, p2 - p0)
    if wp.length_sq(triangle_normal) > 1.0e-16:
        denom = wp.dot(segment_delta, triangle_normal)
        if wp.abs(denom) > 1.0e-12:
            plane_alpha = wp.dot(p0 - segment_start, triangle_normal) / denom
            if plane_alpha >= 0.0 and plane_alpha <= 1.0:
                plane_point = segment_start + segment_delta * plane_alpha
                barycentric = closest_triangle_barycentric(plane_point, p0, p1, p2)
                closest = p0 * barycentric[0] + p1 * barycentric[1] + p2 * barycentric[2]
                distance_sq = wp.length_sq(closest - plane_point)
                if distance_sq < best_distance_sq:
                    best_distance_sq = distance_sq
                    best_barycentric = barycentric
                    best_alpha = plane_alpha
    start_barycentric = closest_triangle_barycentric(segment_start, p0, p1, p2)
    start_closest = p0 * start_barycentric[0] + p1 * start_barycentric[1] + p2 * start_barycentric[2]
    start_distance_sq = wp.length_sq(start_closest - segment_start)
    if start_distance_sq < best_distance_sq:
        best_distance_sq = start_distance_sq
        best_barycentric = start_barycentric
        best_alpha = 0.0
    end_barycentric = closest_triangle_barycentric(segment_end, p0, p1, p2)
    end_closest = p0 * end_barycentric[0] + p1 * end_barycentric[1] + p2 * end_barycentric[2]
    end_distance_sq = wp.length_sq(end_closest - segment_end)
    if end_distance_sq < best_distance_sq:
        best_distance_sq = end_distance_sq
        best_barycentric = end_barycentric
        best_alpha = 1.0
    edge = closest_segment_segment_params(segment_start, segment_end, p0, p1)
    if edge[2] < best_distance_sq:
        best_distance_sq = edge[2]
        best_barycentric = wp.vec3(1.0 - edge[1], edge[1], 0.0)
        best_alpha = edge[0]
    edge = closest_segment_segment_params(segment_start, segment_end, p1, p2)
    if edge[2] < best_distance_sq:
        best_distance_sq = edge[2]
        best_barycentric = wp.vec3(0.0, 1.0 - edge[1], edge[1])
        best_alpha = edge[0]
    edge = closest_segment_segment_params(segment_start, segment_end, p2, p0)
    if edge[2] < best_distance_sq:
        best_barycentric = wp.vec3(edge[1], 0.0, 1.0 - edge[1])
        best_alpha = edge[0]
    return wp.vec4f(best_barycentric[0], best_barycentric[1], best_barycentric[2], best_alpha)


@wp.func
def contact_normal(offset: wp.vec3, distance: wp.float32, triangle_normal: wp.vec3, triangle_normal_length: wp.float32):
    """kernels.py:1840-1883 without side persistence: the tool -> tissue offset, else the triangle normal."""
    if distance > 1.0e-8:
        return offset / distance
    if triangle_normal_length > 1.0e-8:
        return triangle_normal / triangle_normal_length
    return wp.vec3(0.0, 1.0, 0.0)
