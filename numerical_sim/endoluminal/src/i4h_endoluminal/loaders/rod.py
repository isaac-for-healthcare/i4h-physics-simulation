# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from i4h_endoluminal.regions import instrument_regions
from i4h_physics.loaders.rod import add_rod as physical_rod


def add_rod(scene, name, *args, regions=None, **kwargs):
    body = physical_rod(scene, name, *args, **kwargs)
    if regions is not None:
        body.regions = instrument_regions(regions, body.rod.rest_len[0])
    return body
