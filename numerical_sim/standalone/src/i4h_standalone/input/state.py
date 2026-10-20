# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input state shared with the app's instrument driver: pose, grip and buttons of one device."""

from __future__ import annotations

from dataclasses import dataclass, field

INACTIVE = {"active": False}  # the sample of a device with nothing this frame (shared by every source: never written)


@dataclass
class InputState:
    input_id: str
    active: bool = False
    position: tuple = (0.0, 0.0, 0.0)
    rotation: tuple = (0.0, 0.0, 0.0, 1.0)  # xyzw
    grip: float = 0.0  # 0 open .. 1 closed
    buttons: dict = field(default_factory=dict)  # grab, secondary (energy), fire
