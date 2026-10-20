# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(eq=False, repr=False)
class Piece:
    """One rigid piece of an instrument mesh in bind space (``bind`` applied by the skinning kernel)."""

    name: str
    role: int  # 0 shaft (rigid to the trocar-constrained shaft), 1 wrist (body roll), 2 jaw (hinge)
    jaw_sign: float  # +1 / -1 for the two jaws, 0 otherwise
    points: np.ndarray  # (n, 3) float32 local (scaled) points
    indices: np.ndarray  # (f, 3) int32
    bind: np.ndarray  # (4, 4) float32
    color: tuple


@dataclass(eq=False, repr=False)
class SphereSpec:
    """One collision sphere: skinned like a piece (``chain`` 0) or generated along a chain per frame."""

    role: int
    jaw_sign: float
    local: np.ndarray  # (3,) float32, before the jaw hinge and ``bind``
    bind: np.ndarray  # (4, 4) float32
    radius: float
    chain: int  # 0 fixed, 1 wrist chain (root -> jaw roots / wrist end), 2 shaft chain (root -> trocar), 3 cutting chain (skinned like 0, no capsule)
    chain_index: int
    chain_count: int


@dataclass(eq=False, repr=False)
class Instrument:
    """Kinematic instrument: pieces to draw, spheres and capsule primitives to collide, trocar and jaw settings."""

    name: str
    kind: str  # grasper | scissors | clip_applier | hook | suction | sphere
    input: str | None  # input device id bound by the app
    trocar: np.ndarray | None  # (3,) float32 remote centre of motion; None = free
    initial_position: np.ndarray  # (3,)
    initial_rotation: np.ndarray  # (4,) xyzw
    pieces: list  # [Piece]
    spheres: list  # [SphereSpec]
    prims: list  # [(a, b, name)] local sphere indices of each capsule (a == b: sphere)
    wrist_pivot: np.ndarray  # (3,) bind space
    jaw_anchor: np.ndarray  # (3,) bind space
    wrist_end: np.ndarray | None  # (3,) local point on the wrist piece ending the wrist chain
    wrist_end_bind: np.ndarray  # (4, 4)
    jaw_open: float
    jaw_closed: float
    jaw_response: float
    grasp_threshold: float
    grasp_stiffness: float
    shaft_spacing: float


@dataclass(eq=False, repr=False)
class CentrelineSet:
    """Per-variant vessel centrelines. A node is a cross-section clamped to a list of particles whose
    mean is its position (omnisurg ``.cntr`` rows); nodes of all lines are concatenated in YAML
    order, ``clamp_count`` is 0 for the padding nodes of a smaller variant."""

    names: list  # line names, in order
    line: np.ndarray  # (K,) int32 line of each node
    clamp_start: np.ndarray  # (V, K) int32 offset into clamp
    clamp_count: np.ndarray  # (V, K) int32
    clamp: np.ndarray  # (V, C) int32 body-local particle ids, -1 padding
    rest: np.ndarray  # (V, K, 3) float32 node positions at rest
