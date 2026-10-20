# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tet bundle loader: omnisurg's ASCII part directories into a padded, per-variant Body.

Layout after omnisurg/tet/bundle/{mesh_io,loader,connectors}.py: one directory per part
holding ``model.vertices`` (n,3), ``model.tetras`` (T,4), ``model.edges`` (S,2), ``model.tris``
(F,3) and optionally ``model.uvs`` (n,2) with part-local indices; connector files with 8 columns ``pid rest t0 t1 t2 b0 b1 b2``. Parts
are concatenated in YAML order. ``path`` may list several bundle roots (mesh variants),
stored once each and cycled over envs; every part is padded to its largest variant.
``pack_tet_body`` also builds the face and edge tables cutting needs
(omnisurg/tet/bundle/topology.py:116-144, 233-256) and reserves ``cut_faces`` triangle
slots per deletable part for faces exposed by cutting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from i4h_physics.scene import ACTIVE_INT, Body, Part, ParticleRest, Scene, TetMesh

MATERIAL_KEYS = ("spring", "volume", "spring_damping", "volume_damping")
FLAG_KEYS = ("heat", "deletable")  # per-part booleans read by thermal, cutting and stretch_breaking
# Outward face corners and edge corner pairs of a tet (omnisurg/tet/bundle/centrelines.py:34-47).
TET_FACES = np.array(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), np.int64)
TET_EDGES = np.array(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), np.int64)


@dataclass
class PartMesh:
    q0: np.ndarray  # (n, 3) float32, world space
    inv_mass: np.ndarray  # (n,) float32
    tets: np.ndarray  # (t, 4) int32, part-local
    edges: np.ndarray  # (s, 2) int32, part-local
    tris: np.ndarray  # (f, 3) int32, part-local surface triangles
    uvs: np.ndarray | None = None  # (n, 2) float32


def read_rows(path: Path, width: int, dtype) -> np.ndarray:
    rows = np.loadtxt(path, dtype=dtype, ndmin=2)
    if rows.size == 0:
        return np.zeros((0, width), dtype)
    if rows.shape[1] != width:
        raise ValueError(f"{path}: expected {width} columns, got {rows.shape[1]}")
    return rows


def read_part(directory: Path, translation, mass: float) -> PartMesh:
    q0 = read_rows(directory / "model.vertices", 3, np.float32) + np.asarray(translation, np.float32)
    tets = read_rows(directory / "model.tetras", 4, np.int32)
    edges = read_rows(directory / "model.edges", 2, np.int32)
    tris = read_rows(directory / "model.tris", 3, np.int32)
    n = len(q0)
    for label, idx in (("model.tetras", tets), ("model.edges", edges), ("model.tris", tris)):
        if idx.size and (idx.min() < 0 or idx.max() >= n):
            raise ValueError(f"{directory / label}: index outside [0, {n})")
    uvs = None
    if (directory / "model.uvs").is_file():
        uvs = read_rows(directory / "model.uvs", 2, np.float32)
        if len(uvs) != n:
            raise ValueError(f"{directory / 'model.uvs'}: {len(uvs)} rows for {n} vertices")
    return PartMesh(q0, np.full(n, 1.0 / float(mass), np.float32), tets, edges, tris, uvs)


def parse_materials(parts: dict[str, dict]) -> tuple[dict[str, np.ndarray], list]:
    """Per-part vectors ``(P,)`` in part order plus the per-part ``visual`` dicts.

    ``spring`` and ``volume`` are required; ``heat`` and ``deletable`` are flags (off by default);
    ``cut_faces`` (deletable parts only) is the number of triangle slots reserved for faces exposed
    by cutting, default the part's authored triangle count; ``visual`` holds texture paths."""
    out = {key: np.zeros(len(parts), np.float32) for key in MATERIAL_KEYS}
    out.update({key: np.zeros(len(parts), np.int32) for key in FLAG_KEYS})
    out["cut_faces"] = np.full(len(parts), -1, np.int32)  # -1: default
    allowed = MATERIAL_KEYS + FLAG_KEYS + ("cut_faces", "visual")
    visuals = []
    for index, (name, spec) in enumerate(parts.items()):
        spec = dict(spec or {})
        visuals.append(spec.pop("visual", None))
        for key in ("spring", "volume"):
            if key not in spec:
                raise ValueError(f"part {name!r}: missing {key!r}")
        unknown = sorted(set(spec) - set(allowed))
        if unknown:
            raise ValueError(f"part {name!r}: unknown keys {unknown}; allowed {list(allowed)}")
        for key in MATERIAL_KEYS:
            out[key][index] = float(spec.get(key, 0.0))
        for key in FLAG_KEYS:
            out[key][index] = int(bool(spec.get(key, False)))
        if "cut_faces" in spec:
            if not out["deletable"][index]:
                raise ValueError(f"part {name!r}: cut_faces needs deletable: true")
            out["cut_faces"][index] = int(spec["cut_faces"])
            if out["cut_faces"][index] < 0:
                raise ValueError(f"part {name!r}: cut_faces must be non-negative")
    return out, visuals


def tet_topology(tets, active, edges, edge_active, tris, authored, count, tet_edges, face_neighbor, face_tri, tri_tet,
                 support, edge_support) -> None:
    """Face and edge tables of one variant, written into the given rows: face neighbours by hashing
    sorted corner triples, authored triangle -> owning tet (first face with that triple), tet ->
    edge ids, and the live-tet counts per particle and edge that ``deactivate_tet`` decrements
    (omnisurg/tet/bundle/topology.py:116-144 and 233-256, vectorised)."""
    live = np.flatnonzero(active)
    if live.size == 0:
        return
    n = np.int64(count)
    T = tets[live].astype(np.int64)
    faces = np.sort(T[:, TET_FACES], axis=2)  # (t, 4, 3)
    code = ((faces[..., 0] * n + faces[..., 1]) * n + faces[..., 2]).reshape(-1)  # one integer per triple
    order = np.argsort(code, kind="stable")
    sorted_code = code[order]
    first = np.flatnonzero(sorted_code[:-1] == sorted_code[1:])  # a triple shared by two slots: interior face
    a, b = order[first], order[first + 1]
    ta, tb = live[a // 4], live[b // 4]
    face_neighbor[ta, a % 4] = tb
    face_neighbor[tb, b % 4] = ta
    f_idx = np.flatnonzero(authored)
    if f_idx.size:
        tri = np.sort(tris[f_idx].astype(np.int64), axis=1)
        tcode = (tri[:, 0] * n + tri[:, 1]) * n + tri[:, 2]
        pos = np.minimum(np.searchsorted(sorted_code, tcode), len(sorted_code) - 1)
        hit = sorted_code[pos] == tcode
        slot = order[pos[hit]]
        owner = live[slot // 4]
        tri_tet[f_idx[hit]] = owner
        face_tri[owner, slot % 4] = f_idx[hit]
    support[:] = np.bincount(T.reshape(-1), minlength=count)[:count]
    live_e = np.flatnonzero(edge_active)
    if live_e.size:
        pair = np.sort(edges[live_e].astype(np.int64), axis=1)
        ecode = pair[:, 0] * n + pair[:, 1]
        eorder = np.argsort(ecode, kind="stable")
        esorted = ecode[eorder]
        corners = np.sort(T[:, TET_EDGES], axis=2)  # (t, 6, 2)
        pcode = corners[..., 0] * n + corners[..., 1]
        pos = np.minimum(np.searchsorted(esorted, pcode), len(esorted) - 1)
        ids = np.where(esorted[pos] == pcode, live_e[eorder[pos]], -1).astype(np.int32)
        tet_edges[live] = ids
        edge_support[:] = np.bincount(ids[ids >= 0], minlength=len(edge_support))[: len(edge_support)]


# Convention from omnisurg/tet/bundle/topology.py:50-60 (_compute_rest_volumes), vectorised.
def rest_volumes(q0: np.ndarray, tets: np.ndarray, label: str) -> np.ndarray:
    if len(tets) == 0:
        return np.zeros(0, np.float32)
    p0, p1, p2, p3 = (q0[tets[:, k]].astype(np.float64) for k in range(4))
    volumes = np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0
    bad = np.flatnonzero(volumes <= 0.0)
    if bad.size:
        raise ValueError(f"{label}: tetrahedron {int(bad[0])} has non-positive rest volume {volumes[bad[0]]:.3e}")
    return volumes.astype(np.float32)


def pack_tet_body(scene: Scene, name: str, variants: list[dict[str, PartMesh]], materials: dict[str, np.ndarray],
                  radius: float, visuals: list) -> Body:
    """Lay out parts with per-part capacity (max over variants); one row per variant, envs cycle."""
    names = list(variants[0])
    E, V = scene.envs, len(variants)
    visuals = list(visuals)
    parts: dict[str, Part] = {}
    pc = tc = ec = fc = 0
    tet_range: dict[str, tuple[int, int]] = {}  # part -> (start, capacity) in the body's tet and edge tables
    edge_range: dict[str, tuple[int, int]] = {}
    cut_start = np.zeros(len(names), np.int32)
    cut_capacity = np.zeros(len(names), np.int32)
    for index, pname in enumerate(names):
        meshes = [v[pname] for v in variants]
        cap_p, cap_t, cap_e, cap_f = (max(len(getattr(m, attr)) for m in meshes) for attr in ("q0", "tets", "edges", "tris"))
        extra = 0  # slots for faces exposed by cutting, after the authored skin of the part
        if materials["deletable"][index]:
            extra = int(materials["cut_faces"][index]) if materials["cut_faces"][index] >= 0 else cap_f
        cut_start[index], cut_capacity[index] = fc + cap_f, extra
        parts[pname] = Part(index, pc, cap_p, fc, cap_f + extra)
        tet_range[pname], edge_range[pname] = (tc, cap_t), (ec, cap_e)
        pc, tc, ec, fc = pc + cap_p, tc + cap_t, ec + cap_e, fc + cap_f + extra

    q0 = np.zeros((V, pc, 3), np.float32)
    inv_mass = np.zeros((V, pc), np.float32)
    flags = np.zeros((V, pc), np.int32)
    uvs = np.zeros((V, pc, 2), np.float32)
    tets = np.zeros((V, tc, 4), np.int32)
    active0 = np.zeros((V, tc), np.int32)
    rest_volume = np.zeros((V, tc), np.float32)
    edges = np.zeros((V, ec, 2), np.int32)
    edge_active0 = np.zeros((V, ec), np.int32)
    edge_rest = np.zeros((V, ec), np.float32)
    tris = np.zeros((V, fc, 3), np.int32)
    authored = np.zeros((V, fc), bool)
    tet_part = np.zeros(tc, np.int32)
    edge_part = np.zeros(ec, np.int32)
    tri_part = np.zeros(fc, np.int32)
    for pname, p in parts.items():
        (ts, tn), (es, en) = tet_range[pname], edge_range[pname]
        tet_part[ts : ts + tn] = p.index
        edge_part[es : es + en] = p.index
        tri_part[p.tri_start : p.tri_start + p.tri_count] = p.index

    for v, variant in enumerate(variants):
        for pname, p in parts.items():
            m = variant[pname]
            n, t, s, f = len(m.q0), len(m.tets), len(m.edges), len(m.tris)
            ps, ts, es, fs = p.particle_start, tet_range[pname][0], edge_range[pname][0], p.tri_start
            q0[v, ps : ps + p.particle_count] = m.q0.mean(axis=0) if n else 0.0  # parked padding
            q0[v, ps : ps + n] = m.q0
            inv_mass[v, ps : ps + n] = m.inv_mass
            flags[v, ps : ps + n] = ACTIVE_INT
            if m.uvs is not None:
                uvs[v, ps : ps + n] = m.uvs
            tets[v, ts : ts + t] = m.tets + ps
            active0[v, ts : ts + t] = 1
            rest_volume[v, ts : ts + t] = rest_volumes(m.q0, m.tets, f"{name}/{pname} variant {v}")
            edges[v, es : es + s] = m.edges + ps
            edge_active0[v, es : es + s] = 1
            if s:
                edge_rest[v, es : es + s] = np.linalg.norm(m.q0[m.edges[:, 0]] - m.q0[m.edges[:, 1]], axis=1)
            tris[v, fs : fs + p.tri_count] = ps  # padding and cut slots: degenerate triangles inside the part's range
            tris[v, fs : fs + f] = m.tris + ps
            authored[v, fs : fs + f] = True

    tet_edges = np.full((V, tc, 6), -1, np.int32)
    face_neighbor = np.full((V, tc, 4), -1, np.int32)
    face_tri = np.full((V, tc, 4), -1, np.int32)
    tri_tet = np.full((V, fc), -1, np.int32)
    support0 = np.zeros((V, pc), np.int32)
    edge_support0 = np.zeros((V, ec), np.int32)
    for v in range(V):
        tet_topology(tets[v], active0[v], edges[v], edge_active0[v], tris[v], authored[v], pc, tet_edges[v],
                     face_neighbor[v], face_tri[v], tri_tet[v], support0[v], edge_support0[v])

    variant_of_env = np.arange(E, dtype=np.int32) % V
    rest = ParticleRest(q0[variant_of_env], inv_mass[variant_of_env], np.full((E, pc), float(radius), np.float32),
                        flags[variant_of_env])
    body = scene.add_body(name, rest, parts, variant_of_env, variants=V)
    body.tet = TetMesh(
        tets=tets, active0=active0, rest_volume=rest_volume, part=tet_part,
        edges=edges, edge_active0=edge_active0, edge_rest=edge_rest, edge_part=edge_part,
        spring=materials["spring"], spring_damping=materials["spring_damping"],
        volume=materials["volume"], volume_damping=materials["volume_damping"],
        tris=tris, tri_part=tri_part, uvs=uvs,
        heat=materials["heat"], deletable=materials["deletable"], cut_start=cut_start, cut_capacity=cut_capacity,
        tet_edges=tet_edges, face_neighbor=face_neighbor, face_tri=face_tri, tri_tet=tri_tet,
        support0=support0, edge_support0=edge_support0,
    )
    return body


def read_connectors(path: Path, particle_part: str, n_particle: int, triangle_part: str, n_triangle: int):
    """Rows ``pid rest t0 t1 t2 b0 b1 b2`` with part-local indices, validated strictly (no role swapping)."""
    rows = read_rows(path, 8, np.float64)
    ids = rows[:, [0, 2, 3, 4]]
    if not np.array_equal(ids, np.floor(ids)):
        row = int(np.flatnonzero((ids != np.floor(ids)).any(axis=1))[0])
        raise ValueError(f"{path} row {row}: particle and triangle indices must be integers")
    pid = rows[:, 0].astype(np.int32)
    tri = rows[:, 2:5].astype(np.int32)
    for label, idx, n, part in (("particle", pid, n_particle, particle_part), ("triangle", tri, n_triangle, triangle_part)):
        bad = np.flatnonzero((idx < 0) | (idx >= n))
        if bad.size:
            row = int(np.unravel_index(bad[0], idx.shape)[0])
            raise ValueError(
                f"{path} row {row}: {label} index {int(idx.flat[bad[0]])} outside part {part!r} "
                f"({n} particles); check particle_part/triangle_part"
            )
    return pid, tri, rows[:, 5:8].astype(np.float32), rows[:, 1].astype(np.float32)
