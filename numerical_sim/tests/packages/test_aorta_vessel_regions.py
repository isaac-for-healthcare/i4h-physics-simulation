# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shipped aorta_centreline case declares its vessel regions completely."""

import numpy as np
from tissue_helpers import aorta


def test_shipped_aorta_regions_assign_every_edge():
    cfg, authoring, _ = aorta("cuda:0", envs=1)
    regions = authoring.scene.body("aorta").regions
    assert regions.names[0] == "unassigned"
    assert set(regions.names[1:]) == {"aortic_trunk", "right_iliac_femoral", "left_iliac_femoral", "arch_outlets"}
    edges = np.asarray(regions.edge)
    assert len(edges) == 254, "the case declares 254 original edges"
    assert not (edges == regions.names.index("unassigned")).any(), "no edge may remain unassigned"
    assert set(edges.tolist()) == {regions.names.index(n) for n in regions.names[1:]}, "every region owns an edge"
