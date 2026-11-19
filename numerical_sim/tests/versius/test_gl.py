# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device publication uses the same rigid pose as measurements without modifying physics."""
from types import SimpleNamespace

import numpy as np
import warp as wp
from i4h_standalone.render.gl import GlRenderer, part_indices


def test_persistent_vertex_transform_and_live_triangle_indices():
    renderer = GlRenderer.__new__(GlRenderer)
    renderer.scene = SimpleNamespace(device='cpu')
    renderer.component_pose = wp.transform((1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0))
    renderer.transformed = {}
    points = wp.array([[0, 1, 2], [1, 2, 3]], dtype=wp.vec3, device='cpu')
    out = renderer.transform(points)
    np.testing.assert_allclose(out.numpy(), [[1, 1, 1], [2, 0, 0]])
    np.testing.assert_allclose(points.numpy(), [[0, 1, 2], [1, 2, 3]])
    points.assign(np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32))
    assert renderer.transform(points) is out
    np.testing.assert_allclose(out.numpy(), [[1, 2, 3], [2, 1, 2]])
    tris = wp.array([[4, 5, 6], [6, 6, 6]], dtype=wp.vec3i, device='cpu')
    indices = wp.empty(6, dtype=wp.int32, device='cpu')
    wp.launch(part_indices, dim=2, inputs=[tris, 0, 4], outputs=[indices], device='cpu')
    np.testing.assert_array_equal(indices.numpy(), [0, 1, 2, 2, 2, 2])
