# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Control systems in their authored order."""
def controls(sim) -> list:
    """The control systems with a non-empty actions array, in system-list order."""
    return [s for s in sim.systems if getattr(s, "is_control", False) and s.actions.size]
