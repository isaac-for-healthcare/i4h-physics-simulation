# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from i4h_physics.systems.contact.fluid_contact import FluidContact
from i4h_physics.systems.contact.graph_containment import GraphContainment
from i4h_physics.systems.contact.part_contact import PartContact
from i4h_physics.systems.contact.rod_containment import RodContainment
from i4h_physics.systems.contact.shell_containment import ShellContainment
from i4h_physics.systems.contact.static_contact import StaticContact
from i4h_physics.systems.contact.tet_containment import TetContainment
from i4h_physics.systems.fluids.fluid import Fluid
from i4h_physics.systems.mechanics.attachments import Attachments
from i4h_physics.systems.mechanics.bounds import Bounds
from i4h_physics.systems.mechanics.connectors import Connectors
from i4h_physics.systems.mechanics.particles import Particles
from i4h_physics.systems.mechanics.shell import Shell
from i4h_physics.systems.mechanics.stretch_breaking import StretchBreaking
from i4h_physics.systems.mechanics.surface import Surface
from i4h_physics.systems.mechanics.tet import Tet
from i4h_physics.systems.rods.material import RodDirectMaterial, RodPbdMaterial
from i4h_physics.systems.rods.rod_direct import RodDirect
from i4h_physics.systems.rods.rod_pbd import RodPbd

from .base import System as System

SYSTEMS = {
    "particles": Particles,
    "tet": Tet,
    "part_contact": PartContact,
    "surface": Surface,
    "attachments": Attachments,
    "connectors": Connectors,
    "bounds": Bounds,
    "static_contact": StaticContact,
    "stretch_breaking": StretchBreaking,
    "rod_direct_material": RodDirectMaterial,
    "rod_pbd_material": RodPbdMaterial,
    "rod_direct": RodDirect,
    "rod_pbd": RodPbd,
    "rod_containment": RodContainment,
    "shell": Shell,
    "shell_containment": ShellContainment,
    "tet_containment": TetContainment,
    "graph_containment": GraphContainment,
    "fluid": Fluid,
    "fluid_contact": FluidContact,
}
