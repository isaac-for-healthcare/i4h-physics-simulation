# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp

wp.config.quiet = True
wp.init()


def pytest_collection_modifyitems(config, items):
    if wp.is_cuda_available():
        return
    skip = pytest.mark.skip(reason="CUDA device required")
    for item in items:
        item.add_marker(skip)


@pytest.fixture
def case():
    return "two_worlds"


def env_slice(scene, arr, body, env) -> np.ndarray:
    """Host copy of one env's slice of a per-particle array."""
    return scene.view(arr, body).numpy()[env].copy()
