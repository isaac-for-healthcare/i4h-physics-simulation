# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Host-only declaration of the fluid system's per-pool history.

Pool particles are scene storage. What a Fluid keeps across steps is one ring cursor and one neighbour-overflow
count per world and pool; systems that count per pool (suction) add their history to the same rows.
"""
import numpy as np
import warp as wp
from i4h_physics.history_storage import Layout
from i4h_physics.storage import defer_declaration


def publish_fluid(scene, declarations, system_id):
    E, nf = scene.envs, len([b for b in scene.pools if b.fluid is not None])
    layout = Layout(scene, declarations, system_id, "fluid")
    pool = layout.domain("pool", (E, nf), "world", np.full((E, nf), -1, np.int32))
    for key in ("cursor", "overflow"):
        layout.add(key, pool, wp.int32, default=0)
    return layout.publish(pools=nf)


def declare_fluid(scene, *, system_id="fluid"):
    """For a Fluid built on a Scene directly; call before allocation."""
    defer_declaration(scene, system_id, publish_fluid, {})
