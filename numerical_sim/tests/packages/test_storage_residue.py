# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""A conditional clear is not sufficient proof for excluding physics history."""

import numpy as np
import pytest
import warp as wp
from i4h_physics.scene import ACTIVE_INT
from i4h_surgical.systems.mechanics.tool_delta import apply_average_tools


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("value,retained", [(1e-30, True), (1e-3, False)])
def test_per_tool_correction_residue_requires_state(device, value, retained):
    wp.init()
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    with wp.ScopedDevice(device):
        flags = wp.full((1, 1), ACTIVE_INT, dtype=wp.int32)
        inv_mass = wp.ones((1, 1), dtype=wp.float32)
        root = wp.zeros((1, 1), dtype=wp.vec3)
        q = wp.zeros_like(root)
        delta = wp.zeros(1, dtype=wp.vec3)
        count = wp.ones(1, dtype=wp.int32)
        authored = np.array([[[value, 0., 0.]]], np.float32)
        delta_k = wp.array(authored, dtype=wp.vec3)
        delta_jaw = wp.array(authored, dtype=wp.vec3)
        impulse, jaw, moment = (wp.zeros_like(root) for _ in range(3))
        wp.launch(apply_average_tools, dim=(1, 1), inputs=[flags, inv_mass, 1, 1, root],
                  outputs=[q, delta, count, delta_k, delta_jaw, impulse, jaw, moment])
        expected = authored if retained else np.zeros_like(authored)
        assert delta_k.numpy().tobytes() == expected.tobytes()
        assert delta_jaw.numpy().tobytes() == expected.tobytes()
        assert count.numpy()[0] == 0
