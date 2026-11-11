# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Host-only declaration of graph containment's candidate cache and its validity record.

The cache is history: a projection reads candidate lists built steps earlier. One row per world and catheter
sample holds the candidate edges, their count and the sample's reference position; one row per world and tree
node holds the tree's reference position; one row per world holds the rebuild request and the build settings.
"""
import numpy as np
import warp as wp
from i4h_physics.history_storage import Layout
from i4h_physics.storage import defer_declaration, scene_record

EDGE_SAMPLES = 3  # samples inside each catheter edge (graph_containment.EDGE_SAMPLES)


def publish_graph_containment(scene, declarations, system_id, *, body, centreline, capacity=64):
    b = scene.body(body) if isinstance(body, str) else body
    t = scene.body(centreline) if isinstance(centreline, str) else centreline
    if t is b or getattr(t, "tree", None) is None or b.count < 1:
        raise ValueError("graph_containment: needs a source body with nodes and a different body with a centreline tree")
    E, Pc, Pt, cap = scene.envs, b.count, t.count, int(capacity)
    n = Pc + EDGE_SAMPLES * (Pc - 1)
    # A sample is a catheter node, or lies inside the edge that starts at its anchor node.
    anchor = np.concatenate([np.arange(Pc), np.repeat(np.arange(Pc - 1), EDGE_SAMPLES)]).astype(np.int64)

    def owners(part, local):
        return lambda: np.asarray([scene_record(scene).starts[e] + part.offset for e in range(E)], np.int64)[:, None] + local[None]

    layout = Layout(scene, declarations, system_id, "graph_containment")
    sample = layout.domain("sample", (E, n), "particle", owners(b, anchor))
    tree = layout.domain("tree", (E, Pt), "particle", owners(t, np.arange(Pt, dtype=np.int64)))
    layout.add("cand_idx", sample, wp.int32, default=-1, flat=True, width=cap)  # body-local tree edge ids
    for key in ("cand_count", "cand_overflow"):
        layout.add(key, sample, wp.int32, default=0, flat=True)
    layout.add("ref_samples", sample, wp.vec3, default=0, flat=True)
    layout.add("ref_tree", tree, wp.vec3, default=0, flat=True)
    world = layout.world()
    layout.add("rebuild", world, wp.int32, default=1)
    layout.add("built", world, wp.vec2, default=0)
    return layout.publish(body=b, centreline=t, capacity=cap, samples=n)


def declare_graph_containment(scene, *, body, centreline, capacity=64, system_id="graph_containment"):
    """For a GraphContainment built on a Scene directly; call before allocation."""
    defer_declaration(scene, system_id, publish_graph_containment, dict(body=body, centreline=centreline, capacity=capacity))
