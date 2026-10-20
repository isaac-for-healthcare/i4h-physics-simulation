# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from i4h_physics.loaders.fluid import add_fluid
from i4h_physics.loaders.grid import add_tet_grid
from i4h_physics.loaders.rod import add_rod
from i4h_physics.loaders.static_mesh import add_static_mesh
from i4h_physics.loaders.tet_vtu import add_tet_vtu

LOADERS = {
    "tet_grid": add_tet_grid,
    "tet_vtu": add_tet_vtu,
    "static_mesh": add_static_mesh,
    "rod": add_rod,
    "fluid": add_fluid,
}
