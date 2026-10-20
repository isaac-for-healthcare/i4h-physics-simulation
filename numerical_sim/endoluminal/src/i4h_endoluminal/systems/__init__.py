# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from i4h_endoluminal.systems.rods.catheter_control import CatheterControl
from i4h_endoluminal.systems.rods.control_layout import ControlLayout
from i4h_endoluminal.systems.rods.drag import Drag
from i4h_physics.systems.contact.graph_containment import GraphContainment
from i4h_physics.systems.contact.shell_containment import ShellContainment
from i4h_physics.systems.contact.tet_containment import TetContainment


def compatible(constructor):
    """Domain YAML aliases; the physical constructor uses neutral parameter names."""
    def factory(scene, **kwargs):
        for old, new in (("vessel_response", "target_response"), ("catheter_max_delta", "source_max_delta"),
                         ("vessel_max_delta", "target_max_delta")):
            if old in kwargs:
                if new in kwargs:
                    raise ValueError(f"duplicate containment parameter {old}/{new}")
                kwargs[new] = kwargs.pop(old)
        return constructor(scene, **kwargs)
    factory.refs = constructor.refs
    return factory

SYSTEMS = {
    "lumen_containment": compatible(GraphContainment),
    "shell_containment": compatible(ShellContainment),
    "tet_containment": compatible(TetContainment),
    "catheter_control": CatheterControl,
    "drag": Drag,
    "control_layout": ControlLayout,
}
