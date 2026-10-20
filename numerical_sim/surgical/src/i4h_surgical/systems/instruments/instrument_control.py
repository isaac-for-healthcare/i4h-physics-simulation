# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Instrument control: per-env actions (pose + grip) -> poses, jaw angles and collision spheres.

Ported from omnisurg/instruments/runtime_state.py:29-95 (RCM shaft orientation),
grasper.py:149-166 and 900-910 (wrist hinge), kernels.py:96-186 (jaw relaxation, three-link
skinning) and kernels.py:19-93 (spaced wrist and shaft chains), system.py:1053-1074 (grasp
edges). Edits: (env, instrument) and (env, sphere) prologues over one flat sphere layout for
all instruments; the frame motion clamp is applied to the root translation (omnisurg clipped
each sphere's position per substep); actions carry ``[px, py, pz, qx, qy, qz, qw, grip, energy,
fire]``: grip closes the jaw, energy is the cautery level (omnisurg energised the hook on
``grip >= diathermy_activation_threshold``; here it is its own channel), fire is the discrete
trigger (scissors cut, a clip applier places a clip).

Products read by other systems and the renderer: ``sphere_q``/``sphere_q_prev`` ``(E, S)``,
``sphere_radius`` ``(S,)``, ``sphere_active`` ``(E, S)``, capsule primitives ``prim_a/prim_b``
``(P,)`` with ``prim_instrument`` and ``prim_kind``, poses ``root/shaft_q/wrist_q/jaw_q/jaw_angle``
``(E, n)``, the grasp edge flags ``grasp_pending``/``released`` ``(E, n)``, the energy level
``energized`` and the fire level and rising edge ``fire_on``/``fired`` ``(E, n)``, and ``jaw_delta``
``(E, n)``: the relaxed jaw angle's decrease this step (positive while closing; the scissors' cut
trigger, omnisurg system.py:1016-1032 compared the rate-limited jaw closure's per-frame increment).
Sphere chains: 0 fixed and 3 cutting are skinned by the jaw hinge (a cutting chain has no capsule,
so contact never sees it), 1 wrist and 2 shaft are spaced from the root.

Mounted instruments (``mount(k)``, a host rigid body holds the tool, i4h_isaaclab.coupling): the pose comes from
``mount_pose (E, n)`` (component frame, written by the host before each step) instead of ``actions[..., 0:7]``, without
the velocity limit or the trocar; grip, energy and fire still come from the actions. Unmounted tools are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
from i4h_physics.systems.base import System

PRIM_KINDS = {"sphere": 0, "jaw_left": 1, "jaw_right": 2, "wrist": 3, "shaft": 4}
EPS = wp.constant(1.0e-6)


@dataclass(eq=False)
class InstrumentControl(System):
    max_velocity: float = 2.5  # live, per env; the root moves at most max_velocity * dt per frame
    energy_threshold: float = 0.5  # live, per env; actions[..., 8] at or above it energises the instrument
    fire_threshold: float = 0.5  # live, per env; actions[..., 9] at or above it is "fire" (fired = rising edge)

    is_control = True

    def __post_init__(self):
        scene = self.scene
        E, dev = scene.envs, scene.device
        records = list(scene.instruments.values())
        self.names = [r.name for r in records]
        self.count = n = len(records)
        self.rest = np.array([r.initial_position for r in records], np.float32).reshape(n, 3)  # for scripts and tests
        self._live("max_velocity")
        self._live("energy_threshold")
        self._live("fire_threshold")
        self._all = wp.ones(E, dtype=wp.bool, device=dev)

        def floats(values, dtype, tail=()):  # (n, *tail) float constants
            return wp.array(np.asarray(values, np.float32).reshape((n,) + tail), dtype=dtype, device=dev)

        def ints(values):
            return wp.array(np.asarray(values, np.int32).reshape(n), dtype=wp.int32, device=dev)

        zero3 = np.zeros(3, np.float32)
        self.trocar = floats([r.trocar if r.trocar is not None else zero3 for r in records], wp.vec3, (3,))
        self.has_trocar = ints([r.trocar is not None for r in records])
        self.initial_position = floats(self.rest, wp.vec3, (3,))
        self.initial_rotation = floats([r.initial_rotation for r in records], wp.quat, (4,))
        self.jaw_open = floats([r.jaw_open for r in records], wp.float32)
        self.jaw_closed = floats([r.jaw_closed for r in records], wp.float32)
        self.jaw_response = floats([r.jaw_response for r in records], wp.float32)
        self.grasp_threshold = floats([r.grasp_threshold for r in records], wp.float32)
        self.wrist_pivot = floats([r.wrist_pivot for r in records], wp.vec3, (3,))
        self.jaw_anchor = floats([r.jaw_anchor for r in records], wp.vec3, (3,))
        self.wrist_end = floats([r.wrist_end if r.wrist_end is not None else zero3 for r in records], wp.vec3, (3,))
        self.has_wrist_end = ints([r.wrist_end is not None for r in records])
        self.wrist_end_bind = floats([r.wrist_end_bind for r in records], wp.mat44, (4, 4))
        self.shaft_spacing = floats([r.shaft_spacing for r in records], wp.float32)

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
        S, P = len(specs), len(prim_a)
        self.sphere_count = S
        self.prim_count = P
        # Host copies of the capsule layout for the systems that select capsules at construction.
        self.host_prim_kind = np.asarray(prim_kind, np.int32)
        self.host_prim_instrument = np.asarray(prim_owner, np.int32)
        self.host_prim_start = np.asarray(prim_starts, np.int32)
        self.host_prim_counts = np.asarray([len(r.prims) for r in records], np.int32)
        self.host_sphere_start = np.asarray(starts, np.int32)  # sphere layout for systems that select spheres at construction
        self.host_sphere_counts = np.asarray([len(r.spheres) for r in records], np.int32)
        self.host_sphere_chain = np.asarray([s.chain for s in specs], np.int32)
        self.prim_start = ints(prim_starts)
        self.prim_counts = ints([len(r.prims) for r in records])
        self.jaw_root_a = ints(jaw_root_a)
        self.jaw_root_b = ints(jaw_root_b)

        def sfloats(values, dtype, tail=()):  # (S, *tail) float constants
            return wp.array(np.asarray(values, np.float32).reshape((S,) + tail), dtype=dtype, device=dev)

        def sints(values, count=S):
            return wp.array(np.asarray(values, np.int32).reshape(count), dtype=wp.int32, device=dev)

        self.sphere_instrument = sints(owner)
        self.sphere_role = sints([s.role for s in specs])
        self.sphere_jaw_sign = sfloats([s.jaw_sign for s in specs], wp.float32)
        self.sphere_local = sfloats([s.local for s in specs], wp.vec3, (3,))
        self.sphere_bind = sfloats([s.bind for s in specs], wp.mat44, (4, 4))
        self.sphere_radius = sfloats([s.radius for s in specs], wp.float32)
        self.sphere_chain = sints([s.chain for s in specs])
        self.sphere_chain_index = sints([s.chain_index for s in specs])
        self.sphere_chain_count = sints([s.chain_count for s in specs])
        self.prim_a = sints(prim_a, P)
        self.prim_b = sints(prim_b, P)
        self.prim_instrument = sints(prim_owner, P)
        self.prim_kind = sints(prim_kind, P)

        # Per-env state. actions is written by the app or Isaac Lab between steps.
        self.actions = wp.zeros((E, n, 10), dtype=wp.float32, device=dev)
        self.root = wp.zeros((E, n), dtype=wp.vec3, device=dev)
        self.shaft_q = wp.zeros((E, n), dtype=wp.quat, device=dev)
        self.wrist_q = wp.zeros((E, n), dtype=wp.quat, device=dev)
        self.jaw_q = wp.zeros((E, n), dtype=wp.quat, device=dev)
        self.jaw_angle = wp.zeros((E, n), dtype=wp.float32, device=dev)
        self.jaw_delta = wp.zeros((E, n), dtype=wp.float32, device=dev)  # angle decrease this step (> 0 while closing)
        self.held = wp.zeros((E, n), dtype=wp.int32, device=dev)
        self.grasp_pending = wp.zeros((E, n), dtype=wp.int32, device=dev)
        self.released = wp.zeros((E, n), dtype=wp.int32, device=dev)
        self.energized = wp.zeros((E, n), dtype=wp.int32, device=dev)
        self.fire_on = wp.zeros((E, n), dtype=wp.int32, device=dev)
        self.fired = wp.zeros((E, n), dtype=wp.int32, device=dev)
        self.sphere_q = wp.zeros((E, S), dtype=wp.vec3, device=dev)
        self.sphere_q_prev = wp.zeros((E, S), dtype=wp.vec3, device=dev)
        self.sphere_active = wp.zeros((E, S), dtype=wp.int32, device=dev)
        self.mounted = wp.zeros(max(n, 1), dtype=wp.int32, device=dev)  # 1: a host body drives the pose (mount_pose)
        self.mount_pose = wp.array(np.tile(np.concatenate([self.rest, np.asarray([r.initial_rotation for r in records],
                                   np.float32).reshape(n, 4)], axis=1), (E, 1)).reshape(E, n, 7), dtype=wp.transform, device=dev)
        self.reset(scene, self._all)

    def action_bounds(self, workspace: float) -> tuple[list, list]:
        """Per instrument: position within ``workspace`` of its initial position, quaternion within ±1 of its initial
        rotation (the kernel normalises it), grip, energy and fire in [-1, 1] (clamped to [0, 1]: zero or less is open
        and off). The zero normalised action is the reset pose."""
        low, high = [], []
        for k, record in enumerate(self.scene.instruments.values()):
            rotation = np.asarray(record.initial_rotation, np.float32)
            low += [*(self.rest[k] - workspace), *(rotation - 1.0), -1.0, -1.0, -1.0]
            high += [*(self.rest[k] + workspace), *(rotation + 1.0), 1.0, 1.0, 1.0]
        return low, high

    def recorded_actions(self, base) -> list:
        """The app's recording (``<base>_<device>.npy``, one ``[px, py, pz, qx, qy, qz, qw, grip, energy, fire]`` row per
        physics step) of each instrument, ``None`` where it has no file."""
        from pathlib import Path

        base, blocks = Path(base), []
        for record in self.scene.instruments.values():
            path = base.with_name(f"{base.stem}_{record.input or record.name}.npy")
            blocks.append((10, np.load(path).astype(np.float32).reshape(-1, 10) if path.is_file() else None))
        return blocks

    def mount(self, k: int) -> None:
        """A host body drives instrument ``k`` from the next step on, through ``mount_pose`` (a device flag: no recapture)."""
        if not 0 <= k < self.count:
            raise IndexError(f"instrument_control: no instrument {k} (count {self.count})")
        flags = self.mounted.numpy()
        flags[k] = 1
        self.mounted.assign(flags)


    def step_start(self, scene, dt):
        if self.count == 0:
            return
        E, dev = scene.envs, scene.device
        wp.launch(
            update_pose,
            dim=(E, self.count),
            inputs=[self.actions, self.max_velocity, self.trocar, self.has_trocar, self.jaw_open, self.jaw_closed,
                    self.jaw_response, self.grasp_threshold, self.energy_threshold, self.fire_threshold, self.mounted,
                    self.mount_pose, dt],
            outputs=[self.root, self.shaft_q, self.wrist_q, self.jaw_q, self.jaw_angle, self.jaw_delta, self.held,
                     self.grasp_pending, self.released, self.energized, self.fire_on, self.fired],
            device=dev,
        )
        self._place(scene, self._all, 0)

    def reset(self, scene, mask):
        if self.count == 0:
            return
        wp.launch(
            reset_pose,
            dim=(scene.envs, self.count),
            inputs=[mask, self.initial_position, self.initial_rotation, self.trocar, self.has_trocar, self.jaw_open,
                    self.mounted, self.mount_pose],
            outputs=[self.actions, self.root, self.shaft_q, self.wrist_q, self.jaw_q, self.jaw_angle, self.jaw_delta,
                     self.held, self.grasp_pending, self.released, self.energized, self.fire_on, self.fired],
            device=scene.device,
        )
        self._place(scene, mask, 1)

    def _place(self, scene, mask, copy_prev: int):
        wp.launch(
            place_spheres,
            dim=(scene.envs, self.sphere_count),
            inputs=[mask, copy_prev, self.sphere_instrument, self.sphere_role, self.sphere_jaw_sign, self.sphere_local,
                    self.sphere_bind, self.sphere_chain, self.sphere_chain_index, self.sphere_chain_count,
                    self.jaw_root_a, self.jaw_root_b, self.wrist_end, self.has_wrist_end, self.wrist_end_bind,
                    self.trocar, self.mounted, self.shaft_spacing, self.wrist_pivot, self.jaw_anchor,
                    self.root, self.shaft_q, self.wrist_q, self.jaw_q, self.jaw_angle],
            outputs=[self.sphere_q, self.sphere_q_prev, self.sphere_active],
            device=scene.device,
        )


# --- pose math (runtime_state.py:29-95, grasper.py:900-910), as device functions ---------------------------


@wp.func
def unit_or_zero(v: wp.vec3):
    n = wp.length(v)
    if n > EPS:
        return v / n
    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def orthogonal_axis(axis: wp.vec3):
    reference = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(wp.dot(axis, reference)) > 0.9:
        reference = wp.vec3(0.0, 1.0, 0.0)
    result = unit_or_zero(reference - wp.dot(reference, axis) * axis)
    if wp.length(result) <= EPS:
        return wp.vec3(1.0, 0.0, 0.0)
    return result


@wp.func
def shaft_rotation(root: wp.vec3, q_in: wp.quat, trocar: wp.vec3, has_trocar: wp.int32):
    """Trocar-constrained shaft orientation: z points from the root back to the pivot, x follows the input."""
    if has_trocar == 0:
        return q_in
    direction = unit_or_zero(root - trocar)
    if wp.length(direction) <= EPS:
        direction = unit_or_zero(wp.quat_rotate(q_in, wp.vec3(0.0, 0.0, -1.0)))
        if wp.length(direction) <= EPS:
            direction = wp.vec3(0.0, 0.0, -1.0)
    z = -direction
    x_in = wp.quat_rotate(q_in, wp.vec3(1.0, 0.0, 0.0))
    x = unit_or_zero(x_in - wp.dot(x_in, direction) * direction)
    if wp.length(x) <= EPS:
        y_in = wp.quat_rotate(q_in, wp.vec3(0.0, 1.0, 0.0))
        y_proj = unit_or_zero(y_in - wp.dot(y_in, direction) * direction)
        if wp.length(y_proj) > EPS:
            x = unit_or_zero(wp.cross(y_proj, z))
    if wp.length(x) <= EPS:
        x = orthogonal_axis(z)
    y = unit_or_zero(wp.cross(z, x))
    if wp.length(y) <= EPS:
        y = orthogonal_axis(x)
    x = unit_or_zero(wp.cross(y, z))
    if wp.length(x) <= EPS:
        x = orthogonal_axis(z)
    return wp.normalize(wp.quat_from_matrix(wp.matrix_from_cols(x, y, z)))


@wp.func
def wrist_rotation(shaft_q: wp.quat, jaw_q: wp.quat):
    """shaft * hinge about the shaft's local Y that aims the wrist at the input's forward (-Z)."""
    target_forward = wp.quat_rotate(jaw_q, wp.vec3(0.0, 0.0, -1.0))
    local_forward = wp.quat_rotate(wp.quat_inverse(shaft_q), target_forward)
    n = wp.sqrt(local_forward[0] * local_forward[0] + local_forward[2] * local_forward[2])
    if n < 1.0e-8:
        return shaft_q
    angle = wp.atan2(-local_forward[0] / n, -local_forward[2] / n)
    return wp.normalize(shaft_q * wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle))


@wp.func
def skin_point(local: wp.vec3, jaw_sign: wp.float32, jaw_angle: wp.float32, bind: wp.mat44, role: wp.int32,
               root: wp.vec3, shaft_q: wp.quat, wrist_q: wp.quat, jaw_q: wp.quat, wrist_pivot: wp.vec3, jaw_anchor: wp.vec3):
    """kernels.py:114-153: jaw hinge about local X, bind matrix, then the shaft / wrist / jaw link transform."""
    p = local
    if jaw_sign != 0.0:
        p = wp.quat_rotate(wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), jaw_sign * jaw_angle), p)
    bind_point = wp.transform_point(bind, p)
    if role == 0:
        return root + wp.quat_rotate(shaft_q, bind_point)
    pivot_world = root + wp.quat_rotate(shaft_q, wrist_pivot)
    if role == 1:
        return pivot_world + wp.quat_rotate(wrist_q, bind_point - wrist_pivot)
    anchor_world = pivot_world + wp.quat_rotate(wrist_q, jaw_anchor - wrist_pivot)
    return anchor_world + wp.quat_rotate(jaw_q, bind_point - jaw_anchor)


# --- kernels -------------------------------------------------------------------------------------------


@wp.kernel
def update_pose(
    actions: wp.array3d(dtype=wp.float32),
    max_velocity: wp.array(dtype=wp.float32),
    trocar: wp.array(dtype=wp.vec3),
    has_trocar: wp.array(dtype=wp.int32),
    jaw_open: wp.array(dtype=wp.float32),
    jaw_closed: wp.array(dtype=wp.float32),
    jaw_response: wp.array(dtype=wp.float32),
    grasp_threshold: wp.array(dtype=wp.float32),
    energy_threshold: wp.array(dtype=wp.float32),
    fire_threshold: wp.array(dtype=wp.float32),
    mounted: wp.array(dtype=wp.int32),
    mount_pose: wp.array2d(dtype=wp.transform),
    dt: wp.float32,
    root: wp.array2d(dtype=wp.vec3),
    shaft_q: wp.array2d(dtype=wp.quat),
    wrist_q: wp.array2d(dtype=wp.quat),
    jaw_q: wp.array2d(dtype=wp.quat),
    jaw_angle: wp.array2d(dtype=wp.float32),
    jaw_delta: wp.array2d(dtype=wp.float32),
    held: wp.array2d(dtype=wp.int32),
    grasp_pending: wp.array2d(dtype=wp.int32),
    released: wp.array2d(dtype=wp.int32),
    energized: wp.array2d(dtype=wp.int32),
    fire_on: wp.array2d(dtype=wp.int32),
    fired: wp.array2d(dtype=wp.int32),
):
    env, k = wp.tid()
    target = wp.vec3(actions[env, k, 0], actions[env, k, 1], actions[env, k, 2])
    q_in = wp.quat(actions[env, k, 3], actions[env, k, 4], actions[env, k, 5], actions[env, k, 6])
    if wp.length(q_in) > EPS:
        q_in = wp.normalize(q_in)
    else:
        q_in = wp.quat_identity()
    grip = wp.clamp(actions[env, k, 7], 0.0, 1.0)

    shaft = q_in
    if mounted[k] != 0:  # the host body's pose: no velocity limit, no trocar
        target = wp.transform_get_translation(mount_pose[env, k])
        q_in = wp.normalize(wp.transform_get_rotation(mount_pose[env, k]))
        shaft = q_in
    else:
        previous = root[env, k]
        step = target - previous
        distance = wp.length(step)
        max_delta = max_velocity[env] * dt  # collision.py:360-392 translation limit, on the root; 0 = unlimited
        if max_delta > 0.0 and distance > max_delta:
            target = previous + step * (max_delta / distance)
        shaft = shaft_rotation(target, q_in, trocar[k], has_trocar[k])
    root[env, k] = target

    shaft_q[env, k] = shaft
    jaw_q[env, k] = q_in
    wrist_q[env, k] = wrist_rotation(shaft, q_in)

    # kernels.py:96-111 jaw relaxation toward the commanded closure
    angle = jaw_angle[env, k]
    target_angle = jaw_open[k] + grip * (jaw_closed[k] - jaw_open[k])
    new_angle = angle + (target_angle - angle) * wp.min(1.0, dt * jaw_response[k])
    jaw_angle[env, k] = new_angle
    jaw_delta[env, k] = angle - new_angle  # open 0.5 -> closed 0.0: positive while closing

    # system.py:1053-1074 grasp edges on the commanded closure
    was = held[env, k]
    now = wp.int32(0)
    if grip >= grasp_threshold[k]:
        now = wp.int32(1)
    held[env, k] = now
    grasp_pending[env, k] = wp.int32(0)
    released[env, k] = wp.int32(0)
    if now == 1 and was == 0:
        grasp_pending[env, k] = wp.int32(1)
    if now == 0 and was == 1:
        released[env, k] = wp.int32(1)

    # energy is a level (thermal.py:869-874 thresholded the grip); fire is a level plus its rising edge
    energized[env, k] = wp.int32(0)
    energy = wp.clamp(actions[env, k, 8], 0.0, 1.0)
    if energy > 0.0 and energy >= energy_threshold[env]:  # a zero command is off whatever the threshold
        energized[env, k] = wp.int32(1)
    was_fire = fire_on[env, k]
    now_fire = wp.int32(0)
    fire = wp.clamp(actions[env, k, 9], 0.0, 1.0)
    if fire > 0.0 and fire >= fire_threshold[env]:
        now_fire = wp.int32(1)
    fire_on[env, k] = now_fire
    fired[env, k] = wp.int32(0)
    if now_fire == 1 and was_fire == 0:
        fired[env, k] = wp.int32(1)


@wp.kernel
def reset_pose(
    mask: wp.array(dtype=wp.bool),
    initial_position: wp.array(dtype=wp.vec3),
    initial_rotation: wp.array(dtype=wp.quat),
    trocar: wp.array(dtype=wp.vec3),
    has_trocar: wp.array(dtype=wp.int32),
    jaw_open: wp.array(dtype=wp.float32),
    mounted: wp.array(dtype=wp.int32),
    mount_pose: wp.array2d(dtype=wp.transform),
    actions: wp.array3d(dtype=wp.float32),
    root: wp.array2d(dtype=wp.vec3),
    shaft_q: wp.array2d(dtype=wp.quat),
    wrist_q: wp.array2d(dtype=wp.quat),
    jaw_q: wp.array2d(dtype=wp.quat),
    jaw_angle: wp.array2d(dtype=wp.float32),
    jaw_delta: wp.array2d(dtype=wp.float32),
    held: wp.array2d(dtype=wp.int32),
    grasp_pending: wp.array2d(dtype=wp.int32),
    released: wp.array2d(dtype=wp.int32),
    energized: wp.array2d(dtype=wp.int32),
    fire_on: wp.array2d(dtype=wp.int32),
    fired: wp.array2d(dtype=wp.int32),
):
    env, k = wp.tid()
    if not mask[env]:
        return
    p = initial_position[k]
    q = initial_rotation[k]
    shaft = shaft_rotation(p, q, trocar[k], has_trocar[k])
    if mounted[k] != 0:  # where the host body holds it now
        p = wp.transform_get_translation(mount_pose[env, k])
        q = wp.normalize(wp.transform_get_rotation(mount_pose[env, k]))
        shaft = q
    actions[env, k, 0] = p[0]
    actions[env, k, 1] = p[1]
    actions[env, k, 2] = p[2]
    actions[env, k, 3] = q[0]
    actions[env, k, 4] = q[1]
    actions[env, k, 5] = q[2]
    actions[env, k, 6] = q[3]
    actions[env, k, 7] = 0.0
    actions[env, k, 8] = 0.0
    actions[env, k, 9] = 0.0
    root[env, k] = p
    shaft_q[env, k] = shaft
    jaw_q[env, k] = q
    wrist_q[env, k] = wrist_rotation(shaft, q)
    jaw_angle[env, k] = jaw_open[k]
    jaw_delta[env, k] = 0.0
    held[env, k] = wp.int32(0)
    grasp_pending[env, k] = wp.int32(0)
    released[env, k] = wp.int32(0)
    energized[env, k] = wp.int32(0)
    fire_on[env, k] = wp.int32(0)
    fired[env, k] = wp.int32(0)


@wp.kernel
def place_spheres(
    mask: wp.array(dtype=wp.bool),
    copy_prev: wp.int32,
    sphere_instrument: wp.array(dtype=wp.int32),
    sphere_role: wp.array(dtype=wp.int32),
    sphere_jaw_sign: wp.array(dtype=wp.float32),
    sphere_local: wp.array(dtype=wp.vec3),
    sphere_bind: wp.array(dtype=wp.mat44),
    sphere_chain: wp.array(dtype=wp.int32),
    sphere_chain_index: wp.array(dtype=wp.int32),
    sphere_chain_count: wp.array(dtype=wp.int32),
    jaw_root_a: wp.array(dtype=wp.int32),
    jaw_root_b: wp.array(dtype=wp.int32),
    wrist_end: wp.array(dtype=wp.vec3),
    has_wrist_end: wp.array(dtype=wp.int32),
    wrist_end_bind: wp.array(dtype=wp.mat44),
    trocar: wp.array(dtype=wp.vec3),
    mounted: wp.array(dtype=wp.int32),
    shaft_spacing: wp.array(dtype=wp.float32),
    wrist_pivot: wp.array(dtype=wp.vec3),
    jaw_anchor: wp.array(dtype=wp.vec3),
    root: wp.array2d(dtype=wp.vec3),
    shaft_q: wp.array2d(dtype=wp.quat),
    wrist_q: wp.array2d(dtype=wp.quat),
    jaw_q: wp.array2d(dtype=wp.quat),
    jaw_angle: wp.array2d(dtype=wp.float32),
    sphere_q: wp.array2d(dtype=wp.vec3),
    sphere_q_prev: wp.array2d(dtype=wp.vec3),
    sphere_active: wp.array2d(dtype=wp.int32),
):
    """Fixed (0) and cutting (3) spheres are skinned (kernels.py:156-186); wrist and shaft chains are spaced from
    the root (kernels.py:19-93). Chain endpoints re-skin the jaw roots inline so one launch has no ordering."""
    env, s = wp.tid()
    if not mask[env]:
        return
    k = sphere_instrument[s]
    p = root[env, k]
    sq = shaft_q[env, k]
    wq = wrist_q[env, k]
    jq = jaw_q[env, k]
    angle = jaw_angle[env, k]
    pivot = wrist_pivot[k]
    anchor = jaw_anchor[k]
    chain = sphere_chain[s]
    active = wp.int32(1)
    if chain == 0 or chain == 3:
        point = skin_point(sphere_local[s], sphere_jaw_sign[s], angle, sphere_bind[s], sphere_role[s], p, sq, wq, jq, pivot, anchor)
    else:
        i = wp.float32(sphere_chain_index[s])
        count = sphere_chain_count[s]
        spacing = shaft_spacing[k]
        if chain == 1:
            end = wp.vec3(0.0, 0.0, 0.0)
            if has_wrist_end[k] != 0:
                end = skin_point(wrist_end[k], 0.0, angle, wrist_end_bind[k], 1, p, sq, wq, jq, pivot, anchor)
            else:
                a = jaw_root_a[k]
                b = jaw_root_b[k]
                if a >= 0 and b >= 0:  # an instrument without two jaw chains has no wrist target (unreachable today)
                    end_a = skin_point(sphere_local[a], sphere_jaw_sign[a], angle, sphere_bind[a], sphere_role[a], p, sq, wq, jq, pivot, anchor)
                    end_b = skin_point(sphere_local[b], sphere_jaw_sign[b], angle, sphere_bind[b], sphere_role[b], p, sq, wq, jq, pivot, anchor)
                    end = (end_a + end_b) * 0.5
            direction = end - p
            length = wp.length(direction)
            if length > 1.0e-8:
                direction = direction / length
            else:
                direction = wp.vec3(0.0, 0.0, 0.0)
            point = p + direction * wp.min(spacing * i, length)
            if sphere_chain_index[s] == count - 1:
                point = end
            active = wp.int32(0)  # kernels.py:74-93 wrist activation
            if length > 1.0e-8:
                if sphere_chain_index[s] == 0 or sphere_chain_index[s] == count - 1 or spacing * i < length:
                    active = wp.int32(1)
        else:
            direction = unit_or_zero(trocar[k] - p)
            if mounted[k] != 0:  # no trocar: along the shaft's +z, which points back toward the pivot when there is one
                direction = wp.quat_rotate(sq, wp.vec3(0.0, 0.0, 1.0))
            point = p + direction * spacing * i
    if copy_prev != 0:
        sphere_q_prev[env, s] = point
    else:
        sphere_q_prev[env, s] = sphere_q[env, s]
    sphere_q[env, s] = point
    sphere_active[env, s] = active
