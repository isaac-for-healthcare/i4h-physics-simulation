# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Host-only declarations for surgical persistent per-tet fields."""

import warp as wp
from i4h_physics.storage import Field, Owner
from i4h_physics.tet_storage import tet_domain


def declare_thermal_tet_fields(declarations, body, system_id, particle_starts):
    domain = tet_domain(declarations, body, system_id, particle_starts, variants=False)
    for key in ("heat", "burn"):
        declarations.add(Field(domain, key, Owner.STATE, wp.float32, domain.shape,
                               "world, body-local tet", "zero selected worlds"), default=0.0)
    # heat_next and Cutting.cut_tets are overwritten solver scratch, not history.
    # Thermal.coag is indexed by particles and lives on the surface's particle rows (interaction_storage).
    return domain
