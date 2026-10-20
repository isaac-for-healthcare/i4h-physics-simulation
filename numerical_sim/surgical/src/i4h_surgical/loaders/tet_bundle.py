# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
from i4h_physics.loaders.tet_bundle import pack_tet_body, parse_materials, read_connectors, read_part
from i4h_physics.scene import Body, ConnectorSet, PinSet, Scene
from i4h_surgical.scene import CentrelineSet


def read_centreline(path: Path, particle_count: int) -> list[np.ndarray]:
    """One ``.cntr`` file: per row (node) ``n (pid dist)*n m (eid)*m stiffness rest_mul radius``; the
    part-local clamp particle ids are kept, the distances, edge ids and material values are not
    (omnisurg/tet/bundle/centrelines.py:50-178 parsed them; no kernel ever read them)."""
    nodes = []
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        tokens = line.split()
        if not tokens:
            continue
        n = int(tokens[0])
        if n < 1 or len(tokens) < 2 + 2 * n:  # a node is the mean of its clamp particles: it needs one
            raise ValueError(f"{path} row {number}: bad clamp count {n}")
        m = int(tokens[1 + 2 * n])
        if m < 0 or len(tokens) != 2 + 2 * n + m + 3:
            raise ValueError(f"{path} row {number}: expected {2 + 2 * n + max(m, 0) + 3} values, got {len(tokens)}")
        ids = np.array(tokens[1 : 1 + 2 * n : 2], np.int32)
        if ids.size and (ids.min() < 0 or ids.max() >= particle_count):
            raise ValueError(f"{path} row {number}: clamp particle outside [0, {particle_count})")
        nodes.append(ids)
    return nodes


def add_tet_bundle(scene: Scene, name: str, path, parts: dict[str, dict], translation=(0.0, 0.0, 0.0), connectors=(),
                   pin=None, centrelines=(), mass=0.1, radius=0.01) -> Body:
    roots = [Path(p).expanduser() for p in ([path] if isinstance(path, (str, Path)) else path)]
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"tet bundle {name!r}: {root} is not a directory")
    materials, visuals = parse_materials(parts)
    variants = [{pname: read_part(root / pname, translation, mass) for pname in parts} for root in roots]
    body = pack_tet_body(scene, name, variants, materials, radius, visuals)
    from types import SimpleNamespace
    for part, visual in zip(body.parts.values(), visuals):
        part.appearance = SimpleNamespace(visual=visual)
    V = len(roots)

    if connectors:
        specs = [dict(c) for c in connectors]
        for c in specs:
            missing = [k for k in ("path", "particle_part", "triangle_part", "stiffness") if k not in c]
            if missing:
                raise ValueError(f"tet bundle {name!r}: connector {c} is missing {missing}")
        parsed = [[read_connectors(root / c["path"], c["particle_part"], len(variant[c["particle_part"]].q0),
                                   c["triangle_part"], len(variant[c["triangle_part"]].q0))
                   for c in specs] for root, variant in zip(roots, variants)]
        caps = [max(len(parsed[v][g][0]) for v in range(V)) for g in range(len(specs))]
        starts = np.concatenate([[0], np.cumsum(caps)[:-1]]).astype(int)
        C = int(sum(caps))
        particle = np.full((V, C), -1, np.int32)
        tri = np.zeros((V, C, 3), np.int32)
        bar = np.zeros((V, C, 3), np.float32)
        rest = np.zeros((V, C), np.float32)
        group = np.concatenate([np.full(cap, g, np.int32) for g, cap in enumerate(caps)]) if C else np.zeros(0, np.int32)
        for v in range(V):
            for g, c in enumerate(specs):
                pid, t, b, r = parsed[v][g]
                s, n = starts[g], len(pid)
                particle[v, s : s + n] = pid + body.parts[c["particle_part"]].particle_start
                tri[v, s : s + n] = t + body.parts[c["triangle_part"]].particle_start
                bar[v, s : s + n] = b
                rest[v, s : s + n] = r
        body.connectors = ConnectorSet(
            particle, tri, bar, rest, group,
            np.array([float(c["stiffness"]) for c in specs], np.float32),
            np.array([float(c.get("damping", 0.0)) for c in specs], np.float32),
            np.array([float(c.get("weight_ratio", 1.0)) for c in specs], np.float32),  # omnisurg default
        )

    if pin is not None:
        center = np.asarray(pin["center"], np.float32)
        r = float(pin["radius"])
        selected = []  # per variant: (body-local index, rest position) of every particle inside the sphere
        for variant in variants:
            index, rest = [], []
            for pname, p in body.parts.items():
                m = variant[pname]
                hit = np.flatnonzero(np.linalg.norm(m.q0 - center, axis=1) < r)
                index.append(hit + p.particle_start)
                rest.append(m.q0[hit])
            selected.append((np.concatenate(index), np.concatenate(rest)))
        A = max(len(sel[0]) for sel in selected)
        index = np.full((V, A), -1, np.int32)
        rest = np.zeros((V, A, 3), np.float32)
        for v, (idx, pos) in enumerate(selected):
            index[v, : len(idx)] = idx
            rest[v, : len(idx)] = pos
        body.pin = PinSet(index, rest)

    if centrelines:
        specs = [dict(c) for c in centrelines]
        for c in specs:
            missing = [k for k in ("name", "path", "part") if k not in c]
            if missing:
                raise ValueError(f"tet bundle {name!r}: centreline {c} is missing {missing}")
            if c["part"] not in body.parts:
                raise ValueError(f"tet bundle {name!r}: centreline {c['name']!r} names unknown part {c['part']!r}")
        names = [c["name"] for c in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"tet bundle {name!r}: duplicate centreline names {names}")
        # nodes[v][g] = list of part-local clamp id arrays of line g in variant v
        nodes = [[read_centreline(root / c["path"], len(variant[c["part"]].q0)) for c in specs] for root, variant in zip(roots, variants)]
        per_line = [max(len(nodes[v][g]) for v in range(V)) for g in range(len(specs))]
        K = int(sum(per_line))
        line = np.concatenate([np.full(k, g, np.int32) for g, k in enumerate(per_line)]) if K else np.zeros(0, np.int32)
        C = max(sum(len(ids) for g in range(len(specs)) for ids in nodes[v][g]) for v in range(V))
        clamp_start = np.zeros((V, K), np.int32)
        clamp_count = np.zeros((V, K), np.int32)
        clamp = np.full((V, C), -1, np.int32)
        rest = np.zeros((V, K, 3), np.float32)
        for v in range(V):
            cursor, k = 0, 0
            for g, c in enumerate(specs):
                ps = body.parts[c["part"]].particle_start
                for j in range(per_line[g]):
                    if j < len(nodes[v][g]):
                        ids = nodes[v][g][j] + ps
                        clamp_start[v, k], clamp_count[v, k] = cursor, len(ids)
                        clamp[v, cursor : cursor + len(ids)] = ids
                        cursor += len(ids)
                        rest[v, k] = variants[v][c["part"]].q0[nodes[v][g][j]].mean(axis=0)
                    k += 1
        body.centrelines = CentrelineSet(names, line, clamp_start, clamp_count, clamp, rest)

    return body
