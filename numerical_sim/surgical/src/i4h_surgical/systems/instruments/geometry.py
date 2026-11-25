# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Immutable instrument geometry preparation, before device allocation."""

PRIM_KINDS = {"sphere": 0, "jaw_left": 1, "jaw_right": 2, "wrist": 3, "shaft": 4}


class InstrumentLayout:
    def __init__(self, records):
        # Flat sphere layout: instrument by instrument, each [fixed chains][wrist][shaft].
        specs, owner, starts = [], [], []
        prim_a, prim_b, prim_owner, prim_kind, prim_starts = [], [], [], [], []
        jaw_root_a, jaw_root_b = [], []
        for k, r in enumerate(records):
            starts.append(len(specs))
            prim_starts.append(len(prim_a))
            specs += r.spheres
            owner += [k] * len(r.spheres)
            for a, b, name in r.prims:
                prim_a.append(starts[k] + a)
                prim_b.append(starts[k] + b)
                prim_owner.append(k)
                prim_kind.append(PRIM_KINDS[name])
            roots = [starts[k] + a for a, b, name in r.prims if name in ("jaw_left", "jaw_right")]
            jaw_root_a.append(roots[0] if len(roots) >= 2 else -1)
            jaw_root_b.append(roots[1] if len(roots) >= 2 else -1)
        self.specs, self.owner, self.starts = specs, owner, starts
        self.prim_a, self.prim_b = prim_a, prim_b
        self.prim_owner, self.prim_kind, self.prim_starts = prim_owner, prim_kind, prim_starts
        self.jaw_root_a, self.jaw_root_b = jaw_root_a, jaw_root_b
        self.sphere_count, self.prim_count = len(specs), len(prim_a)
