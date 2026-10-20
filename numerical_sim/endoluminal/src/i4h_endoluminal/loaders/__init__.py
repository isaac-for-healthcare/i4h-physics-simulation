# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from i4h_endoluminal.loaders.rod import add_rod
from i4h_endoluminal.loaders.vessel_mesh import add_vessel_mesh
from i4h_physics.loaders import LOADERS as GENERIC_LOADERS

from .presentation import with_presentation

LOADERS = {
    "rod": add_rod,
    "vessel_mesh": add_vessel_mesh,
}



LOADERS = {key: with_presentation(loader) for key, loader in {**GENERIC_LOADERS, **LOADERS}.items()}
