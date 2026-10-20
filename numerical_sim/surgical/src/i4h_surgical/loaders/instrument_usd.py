# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Instrument loader: USD pieces, collision-sphere templates and capsule primitives.

Ported from omnisurg/instruments/grasper.py:425-770 (USD walk, roles, jaw chains, static
hook chains) and instruments/system.py:379-398 (sphere range layout: jaw chains, then a
32-sphere wrist chain, then a 32-sphere shaft chain when a trocar is set). The record is
host data on ``Scene.instruments``; ``instrument_control`` uploads it.

Kinds: ``grasper`` (jaw pieces hinge with the jaw angle), ``scissors`` (the grasper family plus a
separate cutting sphere chain per jaw from the ``cutting`` block, grasper.py:600-619: skinned like the
jaw chains, no capsules, so contact never sees it; the ``cutting`` system cuts with it while the jaw
closes), ``clip_applier`` (the grasper family in its own colours; it never grasps and its fire
applies a clip — ours, omnisurg had no clip kind), ``hook`` (static sphere chains on the hook piece,
an explicit wrist end), ``suction`` (omnisurg ``irrigation_suction``: one ``tool`` prim, no fixed
chains, the shaft chain alone when a trocar is set; grip irrigates, energy sucks), ``sphere`` (one
free sphere, no mesh: the probe of ``two_worlds`` and a debug tool).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from i4h_physics.mesh_utils import triangulate
from i4h_physics.quat import axis_angle_quat, euler_xyz_degrees_quat, normalize_quat, quat_to_matrix
from i4h_physics.scene import Scene
from i4h_physics.usd import usd
from i4h_surgical.scene import Instrument, Piece, SphereSpec

ROLE_SHAFT, ROLE_WRIST, ROLE_JAW = 0, 1, 2
CHAIN_FIXED, CHAIN_WRIST, CHAIN_SHAFT, CHAIN_CUT = 0, 1, 2, 3  # 3: a scissors cutting chain, skinned like 0, no capsule
CHAIN_CAPACITY = 32  # omnisurg _WRIST_SPHERE_CAPACITY and _SHAFT_SPHERE_COUNT
KINDS = ("grasper", "scissors", "clip_applier", "hook", "suction", "sphere")
PIECE_COLORS = {  # grasper.py:410-417
    "shaft": (0.44, 0.47, 0.50),
    "body": (0.55, 0.58, 0.62),
    "hook": (0.78, 0.80, 0.84),
    "tool": (0.78, 0.80, 0.84),
    "jaw_left": (0.78, 0.80, 0.84),
    "jaw_right": (0.70, 0.73, 0.78),
}
KIND_COLORS = {  # per-kind overrides by role: the applier and the scissors share the grasper mesh family (ours; omnisurg coloured by role)
    "clip_applier": {"body": (0.36, 0.46, 0.64), "jaw_left": (0.42, 0.62, 0.92), "jaw_right": (0.34, 0.54, 0.84)},  # blue steel
    "scissors": {"body": (0.60, 0.48, 0.40), "jaw_left": (0.92, 0.60, 0.34), "jaw_right": (0.84, 0.52, 0.28)},  # copper
}
CUTTING_DEFAULTS = {"count": 8, "radius": 0.01, "distance": 0.01, "offset": (0.0, -0.005, 0.0)}  # the chole scissors as authored


def transform_role(role: str) -> int:  # grasper.py:843-848
    if role in ("shaft", "tool"):
        return ROLE_SHAFT
    if role in ("jaw_left", "jaw_right"):
        return ROLE_JAW
    return ROLE_WRIST




def read_usd_pieces(path, prims: dict, scale: float, colors: dict = PIECE_COLORS) -> list[Piece]:
    """Meshes named in ``prims`` (role -> prim path) as pieces in bind space (grasper.py:467-548)."""
    Usd, UsdGeom = usd()

    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"instrument mesh not found: {path}")
    stage = Usd.Stage.Open(str(path))
    roles_by_prim = {str(prim_path): role for role, prim_path in prims.items()}
    model_fix = np.eye(4, dtype=np.float32)
    model_fix[:3, :3] = quat_to_matrix(axis_angle_quat((0.0, 1.0, 0.0), math.pi))  # 180 degrees about Y
    pieces: list[Piece] = []

    def collect(prim, parent_world):
        local = np.eye(4, dtype=np.float32)
        if prim.IsA(UsdGeom.Xformable):
            local = np.array(UsdGeom.Xformable(prim).GetLocalTransformation(), np.float32).T
            local[:3, 3] *= float(scale)
        world = parent_world @ local
        if prim.IsA(UsdGeom.Mesh) and str(prim.GetPath()) in roles_by_prim:
            mesh = UsdGeom.Mesh(prim)
            role = roles_by_prim[str(prim.GetPath())]
            points = np.array(mesh.GetPointsAttr().Get(), np.float32) * float(scale)
            indices = triangulate(np.array(mesh.GetFaceVertexIndicesAttr().Get(), np.int32),
                                  np.array(mesh.GetFaceVertexCountsAttr().Get(), np.int32))
            jaw_sign = 1.0 if role == "jaw_left" else -1.0 if role == "jaw_right" else 0.0
            pieces.append(Piece(role, transform_role(role), jaw_sign, np.ascontiguousarray(points),
                                np.ascontiguousarray(indices), (model_fix @ world).astype(np.float32),
                                colors.get(role, (0.6, 0.6, 0.6))))
        for child in prim.GetChildren():
            collect(child, world)

    for child in stage.GetPseudoRoot().GetChildren():
        collect(child, np.eye(4, dtype=np.float32))
    missing = set(prims) - {p.name for p in pieces}
    if missing:
        raise ValueError(f"instrument mesh {path}: prims not found for roles {sorted(missing)}")
    return pieces


def chain_spheres(local_points: np.ndarray, bind, role: int, jaw_sign: float, radius: float) -> list:
    return [SphereSpec(role, jaw_sign, np.asarray(p, np.float32), bind, float(radius), CHAIN_FIXED, i, len(local_points))
            for i, p in enumerate(local_points)]


def spaced_chain(chain: int, radius: float, count: int = CHAIN_CAPACITY) -> list:
    zero, eye = np.zeros(3, np.float32), np.eye(4, dtype=np.float32)
    return [SphereSpec(ROLE_SHAFT, 0.0, zero, eye, float(radius), chain, i, count) for i in range(count)]


JAW_OPEN, JAW_CLOSED, JAW_RESPONSE = 0.5, 0.0, 20.0  # schema.py:1899-1911 jaw defaults; no case tunes them


def add_instrument(scene: Scene, name: str, kind: str, mesh=None, prims=None, scale=1.0, trocar=None, initial=None,
                   jaw_spheres=None, shaft_spheres=None, wrist_end=None, grasp=None, radius=None, input=None, cutting=None) -> Instrument:
    """``initial: {position, rotation}`` is the pose the instrument starts and resets at (rotation xyzw, identity default).
    ``cutting: {count, radius, distance, offset}`` (scissors only) shapes the cutting chains; the defaults are the chole values."""
    if not hasattr(scene, "instruments"):
        scene.instruments = {}  # owned by surgical loaders and control systems
    if name in scene.instruments:
        raise ValueError(f"duplicate instrument {name!r}")
    if kind not in KINDS:
        raise ValueError(f"instrument {name!r}: kind must be one of {', '.join(KINDS)}, got {kind!r}")
    if cutting is not None and kind != "scissors":
        raise ValueError(f"instrument {name!r}: only scissors take a cutting block")
    cut = {**CUTTING_DEFAULTS, **dict(cutting or {})}
    initial = dict(initial or {})
    initial_position = np.asarray(initial.get("position", (0.0, 0.0, 0.0)), np.float32)
    initial_rotation = normalize_quat(initial.get("rotation", (0.0, 0.0, 0.0, 1.0)))
    grasp = {"threshold": 0.5, "stiffness": 1.0, **dict(grasp or {})}
    shaft_spheres = {"radius": 0.04, "distance": 0.02, **dict(shaft_spheres or {})}
    js = {"count": 13, "radius": 0.015, "distance": None, "offset": (0.0, 0.0, 0.0), "chains": None, **dict(jaw_spheres or {})}
    for label, given, known in (("initial", initial, {"position", "rotation"}), ("grasp", grasp, {"threshold", "stiffness"}),
                                ("shaft_spheres", shaft_spheres, {"radius", "distance"}),
                                ("jaw_spheres", js, {"count", "radius", "distance", "offset", "chains"}),
                                ("cutting", cut, set(CUTTING_DEFAULTS))):
        unknown = set(given) - known
        if unknown:
            raise ValueError(f"instrument {name!r}: unknown {label} keys {sorted(unknown)}; known {sorted(known)}")
    eye = np.eye(4, dtype=np.float32)

    if kind == "sphere":
        if radius is None:
            raise ValueError(f"instrument {name!r}: kind sphere needs a radius")
        spheres = [SphereSpec(ROLE_SHAFT, 0.0, np.zeros(3, np.float32), eye, float(radius), CHAIN_FIXED, 0, 1)]
        record = Instrument(name, kind, input, None, initial_position, initial_rotation, [], spheres, [(0, 0, "sphere")],
                            np.zeros(3, np.float32), np.zeros(3, np.float32), None, eye, JAW_OPEN, JAW_CLOSED,
                            JAW_RESPONSE, grasp["threshold"], grasp["stiffness"], shaft_spheres["distance"])
        scene.instruments[name] = record
        return record

    if mesh is None or not prims:
        raise ValueError(f"instrument {name!r}: kind {kind} needs mesh and prims")
    pieces = read_usd_pieces(mesh, dict(prims), float(scale), {**PIECE_COLORS, **KIND_COLORS.get(kind, {})})
    count = int(js["count"])
    spheres: list = []
    prims_out: list = []
    wrist_end_local, wrist_end_bind = None, eye

    def distances_along(tip):  # sphere spacing along a chain: explicit distance, else spread to the piece's tip
        if js["distance"] is not None:
            return np.arange(count, dtype=np.float32) * float(js["distance"])
        return np.linspace(0.0, max(float(tip), 0.25), count, dtype=np.float32)

    if kind in ("grasper", "scissors", "clip_applier"):  # grasper.py:632-680: one hinged chain per jaw piece along local +Z
        jaw_pieces = [p for p in pieces if p.jaw_sign != 0.0]
        if len(jaw_pieces) != 2:
            raise ValueError(f"instrument {name!r}: a {kind} needs jaw_left and jaw_right prims")
        tip = max(float(p.points[:, 2].max()) for p in jaw_pieces)
        distances = distances_along(tip)
        offset = np.asarray(js["offset"], np.float32)
        for piece in sorted(jaw_pieces, key=lambda p: -p.jaw_sign):  # jaw_left first
            template = np.repeat((offset * piece.jaw_sign)[None, :], count, axis=0)
            template[:, 2] += distances
            start = len(spheres)
            spheres += chain_spheres(template, piece.bind, ROLE_JAW, piece.jaw_sign, js["radius"])
            prims_out.append((start, start + count - 1, piece.name))
        if kind == "scissors":  # grasper.py:600-619: the cutting chains, the same construction with the cutting block, no capsules
            n_cut = int(cut["count"])
            cut_distances = (np.arange(n_cut, dtype=np.float32) * float(cut["distance"]) if cut["distance"] is not None
                             else np.linspace(0.0, max(tip, 0.25), n_cut, dtype=np.float32))
            cut_offset = np.asarray(cut["offset"], np.float32)
            for piece in sorted(jaw_pieces, key=lambda p: -p.jaw_sign):
                template = np.repeat((cut_offset * piece.jaw_sign)[None, :], n_cut, axis=0)
                template[:, 2] += cut_distances
                spheres += [SphereSpec(ROLE_JAW, piece.jaw_sign, np.asarray(q, np.float32), piece.bind, float(cut["radius"]), CHAIN_CUT, i, n_cut)
                            for i, q in enumerate(template)]
    elif kind == "suction":  # one rigid tube, authored along local -z with its tip at the origin: through the 180-degree
        # model fix that is bind-space +z, the shaft direction (toward the trocar) the other tools' shaft prims share
        if not any(p.name == "tool" for p in pieces):
            raise ValueError(f"instrument {name!r}: a suction tool needs a 'tool' prim")
    else:  # hook, grasper.py:683-769: two static chains on the hook piece, offset + euler per chain
        hook_piece = next((p for p in pieces if p.name == "hook"), None)
        if hook_piece is None:
            raise ValueError(f"instrument {name!r}: a hook needs a 'hook' prim")
        distances = distances_along(hook_piece.points[:, 2].max())
        chains = dict(js["chains"] or {})
        for chain_name, sign in (("jaw_left", 1.0), ("jaw_right", -1.0)):
            template = np.zeros((count, 3), np.float32)
            template[:, 2] = distances
            if chains:
                spec = dict(chains[chain_name])
                rotation = euler_xyz_degrees_quat(spec.get("euler", (0.0, 0.0, 0.0)))
                template = template @ quat_to_matrix(rotation).T + np.asarray(spec["offset"], np.float32)[None, :]
            else:
                template += (np.asarray(js["offset"], np.float32) * sign)[None, :]
            start = len(spheres)
            spheres += chain_spheres(template, hook_piece.bind, ROLE_WRIST, 0.0, js["radius"])
            prims_out.append((start, start + count - 1, chain_name))
        wrist_end_local = None if wrist_end is None else np.asarray(wrist_end, np.float32)
        wrist_end_bind = hook_piece.bind

    if trocar is not None:  # system.py:379-398: wrist chain between the jaw roots, shaft chain toward the trocar
        chains_out = ((CHAIN_SHAFT, "shaft"),) if kind == "suction" else ((CHAIN_WRIST, "wrist"), (CHAIN_SHAFT, "shaft"))
        for chain, chain_name in chains_out:  # a suction tool has no jaw roots for a wrist chain to span
            start = len(spheres)
            spheres += spaced_chain(chain, shaft_spheres["radius"])
            prims_out.append((start, start + CHAIN_CAPACITY - 1, chain_name))

    shaft_binds = [p.bind for p in pieces if p.role == ROLE_SHAFT]
    jaw_binds = [p.bind for p in pieces if p.role == ROLE_JAW]
    wrist_pivot = np.mean([b[:3, 3] for b in shaft_binds], axis=0).astype(np.float32) if shaft_binds else np.zeros(3, np.float32)
    jaw_anchor = np.mean([b[:3, 3] for b in jaw_binds], axis=0).astype(np.float32) if jaw_binds else np.zeros(3, np.float32)
    record = Instrument(
        name, kind, input, None if trocar is None else np.asarray(trocar, np.float32),
        initial_position, initial_rotation, pieces, spheres, prims_out, wrist_pivot, jaw_anchor,
        wrist_end_local, wrist_end_bind, JAW_OPEN, JAW_CLOSED, JAW_RESPONSE,
        float(grasp["threshold"]), float(grasp["stiffness"]), float(shaft_spheres["distance"]),
    )
    scene.instruments[name] = record
    return record
