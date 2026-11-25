# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve camera targets without renderer or host dependencies."""
try:
    from i4h_surgical.systems.instruments.instrument_control import InstrumentControl
except ModuleNotFoundError as error:
    if error.name != "i4h_surgical":
        raise
    InstrumentControl = ()
from i4h_physics.systems.rods.rod_direct import RodDirect
from i4h_physics.systems.rods.rod_pbd import RodPbd

TIP_DEFAULTS = dict(body=None, instrument=None, distance=0.04, lookahead=0.05, fov=100.0, near=0.005)


def resolve_tip(sim, cfg: dict | None):
    """``(kind, target, settings, names)``: kind ``"rod"`` with ``(body, rod_system)`` or ``"instrument"`` with the tool index;
    ``names`` lists every possible target (``rod:<body>``, ``instrument:<name>``) for the panel."""
    cfg = {**TIP_DEFAULTS, **dict(cfg or {})}
    unknown = set(cfg) - set(TIP_DEFAULTS)
    if unknown:
        raise ValueError(f"render.tip: unknown keys {sorted(unknown)}; known {sorted(TIP_DEFAULTS)}")
    rods = {s.body.name: (s.body, s) for s in sim.systems if isinstance(s, (RodDirect, RodPbd)) and s.body.rod is not None}
    control = next((s for s in sim.systems if isinstance(s, InstrumentControl)), None)
    names = [f"rod:{n}" for n in rods] + [f"instrument:{n}" for n in (control.names if control is not None else [])]
    if not names:
        raise ValueError("render.tip: the case has no chain rod and no instrument to attach the camera to")
    if cfg["body"] is not None and cfg["instrument"] is not None:
        raise ValueError("render.tip: give body or instrument, not both")
    if cfg["body"] is not None:
        if cfg["body"] not in rods:
            raise ValueError(f"render.tip.body {cfg['body']!r} is not a chain rod body; choose from {names}")
        kind, target = "rod", rods[cfg["body"]]
    elif cfg["instrument"] is not None:
        if control is None or cfg["instrument"] not in control.names:
            raise ValueError(f"render.tip.instrument {cfg['instrument']!r} is not an instrument; choose from {names}")
        kind, target = "instrument", control.names.index(cfg["instrument"])
    elif rods:
        kind, target = "rod", next(iter(rods.values()))
    else:
        kind, target = "instrument", 0
    settings = {k: float(cfg[k]) for k in ("distance", "lookahead", "fov", "near")}
    settings["fov"] = min(max(settings["fov"], 30.0), 150.0)  # xcath.py:1334
    return kind, target, settings, names
