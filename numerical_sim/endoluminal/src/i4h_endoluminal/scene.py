# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(eq=False, repr=False)
class SkinMesh:
    """The surface mesh of a centreline-tree vessel, bound to the tree's segments for dual-quaternion skinning
    (omniendo xcath/vessel_skinning.py). Rendering only: containment uses the tree. Built by the loader after
    binding on the device; the renderers place the vertices from the tree body's positions and its rod system's
    segment frames every frame."""

    points: np.ndarray  # (n, 3) float32 rest vertices, scene coordinates
    indices: np.ndarray  # (f, 3) int32, winding as authored
    visual: dict  # {color: rgb, cull: back} as a static mesh
    bound_edge: np.ndarray  # (n,) int32 nearest tree segment at rest
    bound_t: np.ndarray  # (n,) float32 parameter along it
    rest_center: np.ndarray  # (n, 3) float32 the bound point at rest
    rest_orientation: np.ndarray  # (n, 4) float32 xyzw the bound segment's rest frame
    adj_offsets: np.ndarray  # (n + 1,) int32 one-ring CSR offsets (the dual-quaternion smoothing)
    adj_vertices: np.ndarray  # (m,) int32 sorted neighbours


@dataclass
class GraphMetadata:
    edge_sources: tuple = ()
    layouts: list | None = None
