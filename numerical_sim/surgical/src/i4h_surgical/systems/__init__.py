# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from i4h_physics.systems.contact.part_contact import PartContact
from i4h_surgical.systems.contact.instrument_contact import InstrumentContact
from i4h_surgical.systems.fluids.bleeding import Bleeding
from i4h_surgical.systems.fluids.emitter import Emitter
from i4h_surgical.systems.fluids.smoke import Smoke
from i4h_surgical.systems.fluids.suction import Suction
from i4h_surgical.systems.instruments.clipping import Clipping
from i4h_surgical.systems.instruments.contact_paint import ContactPaint
from i4h_surgical.systems.instruments.cutting import Cutting
from i4h_surgical.systems.instruments.grasp import Grasp
from i4h_surgical.systems.instruments.hook import Hook
from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
from i4h_surgical.systems.instruments.thermal import Thermal
from i4h_surgical.systems.instruments.tool_force import ToolForce
from i4h_surgical.systems.instruments.volumetric_grasp import VolumetricGrasp
from i4h_surgical.systems.mechanics.centreline import Centreline
from i4h_surgical.systems.mechanics.stretch_breaking import StretchBreaking
from i4h_surgical.systems.mechanics.surface import Surface

SYSTEMS = {
    "tissue_contact": PartContact,
    "instrument_contact": InstrumentContact,
    "instrument_control": InstrumentControl,
    "contact_paint": ContactPaint,
    "grasp": Grasp,
    "volumetric_grasp": VolumetricGrasp,
    "hook": Hook,
    "tool_force": ToolForce,
    "cutting": Cutting,
    "thermal": Thermal,
    "centreline": Centreline,
    "clipping": Clipping,
    "emitter": Emitter,
    "suction": Suction,
    "bleeding": Bleeding,
    "smoke": Smoke,
    "surface": Surface,
    "stretch_breaking": StretchBreaking,
}
