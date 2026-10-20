# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Domain-owned presentation metadata; physical records retain only physical fields."""
from types import SimpleNamespace


def with_presentation(loader):
    def factory(scene, name, *args, **kwargs):
        visual = kwargs.get("visual")
        parts = kwargs.get("parts") or {}
        record = loader(scene, name, *args, **kwargs)
        if hasattr(record, "parts"):
            for part_name, part in record.parts.items():
                authored = (parts.get(part_name) or {}).get("visual", visual)
                part.appearance = SimpleNamespace(visual=authored)
            if record.fluid is not None:
                record.fluid.appearance = SimpleNamespace(visual=dict(visual or {}))
        elif hasattr(record, "points"):
            record.appearance = SimpleNamespace(visual=dict(visual or {}))
        return record
    return factory
