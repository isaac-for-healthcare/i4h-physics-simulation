# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catheter controls: insertion along a fixed track, axial rotation and a steerable tip, batched per env.

omniendo's ``catheter_root``, ``catheter_track`` and ``catheter_tip_bend`` constraints
(xcath/scenes/*.yaml) collapse into this one system, kernels ported from
omniendo/xpbd_rod/batched_xcath.py:68-146. Actions are omniendo's Isaac Lab contract
(omniendo/isaaclab/actions.py:56-61): ``actions (E, 3)`` are normalised rates in ``[-1, 1]`` for
insertion, rotation and tip bend; every ``step_start`` integrates them on device into
``targets (E,) vec3`` = (insertion distance, rotation angle, tip angle) scaled by ``rates`` and
clamped to ``[lower, upper]``. Isaac Lab writes ``actions`` in ``_apply_action``, the app's
``CatheterDriver`` otherwise.

Per substep, after the rod predicted its frames, the root node is teleported to
``track_start + direction * insertion`` (slewed by ``max_delta`` per substep) with its frame set to
the rest frame twisted about the local tangent by the rotation angle: insertion drags the whole
rod, nothing is fed or activated. After the rod's constraint projection every iteration, the
nodes whose axial coordinate is inside ``[0, track_length]`` are snapped onto the track line
(``track_stiffness`` 1 = hard), except the root. The tip is guided too while it is on the track, like a
steerable tip inside its introducer: it bends once it has left the guide (omniendo never guided the last
``tip_edges`` nodes; a catheter that starts on its track outside a deformable vessel could then be bent at
rest and miss the vessel's entry when inserted, NVBug 6940614). The tip bend is written once per step into
the rod's live ``bend_offset``: ``tip_axis * angle / tip_edges`` on the last ``tip_edges`` bend constraints.
The track starts at the rod's rest root along its rest tangent; ``rod_containment`` must be listed after this
system so the vessel projection sees the track result (omniendo POST hook order). An env whose root node is
inactive (``scene.flags``) is not driven.

Mounted (``mount(0)``, a host body holds the catheter's driver, i4h_isaaclab.coupling): the track's start and the root's
rest frame come from ``mount_pose (E, 1)`` (component frame, written by the host before each step) instead of the rest
root: the start is the mount's position, the rest frame its rotation, and the direction the rest tangent carried by
it. A mount at the rest root frame drives exactly as unmounted. Products for the coupling, per env, as the tools'
(``impulse``: the sum of m * dx the driver imposed on the rod over the step; ``moment``: its moment about the root):
the guide's snaps, and with ``rod_direct`` the root's hold: the negated stretch multipliers of the first edge (the
impulse the rod's constraints would have given the locked root) and their rotational Jacobian contraction (the
angular impulse they would have given its frame: the bending and twisting moment, as catheter_vasculature_solver's
``proximal_wrench``). ``names`` and ``root (E, 1)`` (the root's target) complete the mount interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
from i4h_physics.checks import check_int
from i4h_physics.scene import ACTIVE
from i4h_physics.systems.base import System
from i4h_physics.systems.rods.rod_direct import RodDirect
from i4h_physics.systems.rods.rod_pbd import RodPbd


@dataclass(eq=False)
class CatheterControl(System):
    state_products = ("targets",)  # per-env products a host observes (System.state_products)
    rod: object  # rod_direct or rod_pbd system id in YAML; the instance after construction
    track_length: float  # live, per env: the guide ends here (from the rod's rest root along its direction)
    tip_edges: int = 0  # structural: distal bend constraints that carry the tip angle
    tip_axis: tuple = (1.0, 0.0, 0.0)  # structural: material-frame axis the tip bends about
    track_stiffness: float = 1.0  # live, per env: fraction of the snap onto the track per iteration
    rates: tuple = (0.5, 1.5, 1.0)  # live (E, 3): m/s, rad/s, rad/s at action 1 (omniendo XCathVelocityActionCfg)
    lower: tuple = (0.0, -1.0e9, -1.5)  # live (E, 3): target bounds (insertion >= 0, tip in [-1.5, 1.5] as xcath.py)
    upper: tuple = (1.0e9, 1.0e9, 1.5)  # live (E, 3); the case caps insertion at its track length
    max_delta: float = 0.01  # live, per env: root motion per substep (omniendo catheter_max_delta)
    tip_mapping: str = "per_edge"  # structural: per_edge (omniendo: angle / tip_edges per edge) or polyline (the tip
    # polyline turns by the angle: sin(angle / (2 tip_edges - 1)) per edge, the workflow's tip_bend_rest_component)

    is_control = True

    refs = ("rod",)
    structural = System.structural + ("tip_edges", "tip_axis", "tip_mapping")

    def __post_init__(self):
        scene = self.scene
        rod = self.rod
        if not isinstance(rod, (RodDirect, RodPbd)):
            raise TypeError("catheter_control.rod must name a rod_direct or rod_pbd system")
        self.body = b = rod.body
        self.controls = b.rod_controls
        m = b.rod
        if m is None:
            raise TypeError("catheter_control drives a straight rod asset (a chain with an insertion track), not a centreline tree")
        E, dev = scene.envs, scene.device
        self.P, self.F, self.B = b.count, rod.frame_count, rod.bend_count
        self.tip_edges = check_int("catheter_control.tip_edges", self.tip_edges)
        self.tip_axis = tuple(float(x) for x in self.tip_axis)
        if not 0 <= self.tip_edges < self.B:
            raise ValueError(f"catheter_control.tip_edges must be in [0, {self.B}) for this rod, got {self.tip_edges}")
        if len(self.tip_axis) != 3:
            raise ValueError("catheter_control.tip_axis is a 3-vector")
        if self.tip_mapping not in ("per_edge", "polyline"):
            raise ValueError(f"catheter_control.tip_mapping must be 'per_edge' or 'polyline', got {self.tip_mapping!r}")
        self._live("track_length", "track_stiffness", "max_delta", "rates", "lower", "upper")
        for name in ("rates", "lower", "upper"):
            if getattr(self, name).shape != (E, 3):
                raise ValueError(f"catheter_control.{name} must have 3 entries")
        self.track_start = wp.array(m.start, dtype=wp.vec3, device=dev)  # (V,)
        self.track_direction = wp.array(m.direction, dtype=wp.vec3, device=dev)  # (V,) unit
        self.base_orientation = wp.array(np.ascontiguousarray(m.frames[:, 0]), dtype=wp.quat, device=dev)  # (V,)
        self.actions = wp.zeros((E, 3), dtype=wp.float32, device=dev)  # normalised rates, written from outside step()
        self.targets = wp.zeros(E, dtype=wp.vec3, device=dev)  # (insertion, rotation, tip angle)
        # Mount interface (i4h_isaaclab.coupling): one "instrument", the catheter's driver.
        self.names, self.count = [b.name], 1
        self.mounted = wp.zeros(1, dtype=wp.int32, device=dev)
        v = rod.variant.numpy()
        self.mount_pose = wp.array(np.concatenate([m.start[v], m.frames[v, 0]], axis=1).reshape(E, 1, 7).astype(np.float32),
                                   dtype=wp.transform, device=dev)
        self.root = wp.array(m.start[v].reshape(E, 1, 3).astype(np.float32), dtype=wp.vec3, device=dev)
        self.impulse = wp.zeros((E, 1), dtype=wp.vec3, device=dev)  # product: sum of m * dx the driver imposed this step
        self.moment = wp.zeros((E, 1), dtype=wp.vec3, device=dev)  # product: its moment about the root
        self._lambda = getattr(rod, "lambda_sum", None)  # rod_direct's multipliers: the root's hold
        self._jacobian = getattr(rod, "jacobian_rot", None)
        self.reset(scene, wp.ones(E, dtype=wp.bool, device=dev))

    def action_bounds(self, workspace: float) -> tuple[list, list]:
        """Normalised insertion, rotation and tip-bend rates: [-1, 1] each (``rates`` scales them)."""
        return [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]

    def mount(self, k: int = 0) -> None:
        """A host body holds the driver from the next step on, through ``mount_pose`` (a device flag: no recapture)."""
        if k != 0:
            raise IndexError(f"catheter_control: the mount index is 0, got {k}")
        self.mounted.fill_(1)

    @property
    def root_reaction(self) -> bool:
        """The root's hold is measured (``rod_direct``); with ``rod_pbd`` only the guide's snaps are."""
        return self._lambda is not None


    def prepare(self, scene):
        self.controls.bend_offset.zero_()  # only the last ``tip_edges`` entries are written from here on (or none when disabled)

    def step_start(self, scene, dt):
        self.impulse.zero_()
        self.moment.zero_()
        wp.launch(advance_targets, dim=scene.envs,
                  inputs=[self.actions, self.rates, self.lower, self.upper, dt, wp.vec3(*self.tip_axis), self.tip_edges, self.B,
                          int(self.tip_mapping == "polyline")],
                  outputs=[self.targets, self.controls.bend_offset], device=scene.device)

    def substep_start(self, scene, dt, s, substeps):
        b, rod = self.body, self.rod
        wp.launch(root_controls, dim=scene.envs,
                  inputs=[self.targets, self.track_start, self.track_direction, self.base_orientation, rod.variant,
                          scene.field("flags", b), self.max_delta, self.F, self.mounted, self.mount_pose],
                  outputs=[scene.field("q", b), scene.field("q_prev", b), rod.frame, rod.frame_pred, self.root],
                  device=scene.device)

    def iteration(self, scene, dt, k, iterations):
        b = self.body
        wp.launch(track_guidance, dim=(scene.envs, self.P),
                  inputs=[scene.field("inv_mass", b), scene.field("flags", b), self.track_start, self.track_direction,
                          self.base_orientation, self.rod.variant, self.track_length, self.track_stiffness,
                          self.mounted, self.mount_pose, self.root],
                  outputs=[scene.field("q", b), self.impulse, self.moment], device=scene.device)

    def substep_end(self, scene, dt):
        if self._lambda is not None:
            wp.launch(root_hold, dim=scene.envs, inputs=[self._lambda, self._jacobian, self.B],
                      outputs=[self.impulse, self.moment], device=scene.device)

    def reset(self, scene, mask):
        """Selected envs: actions and targets to zero, matching ``Scene.reset`` putting the root back at rest and
        the rod's own reset zeroing ``bend_offset``."""
        self.clear(mask, (self.actions, wp.float32(0.0)))
        self.clear(mask, (self.targets, wp.vec3(0.0, 0.0, 0.0)))
        self.zero(mask, "impulse", "moment")


# --- kernels -----------------------------------------------------------------------------------------


# Device form of omniendo/isaaclab/actions.py:56-61 (XCathVelocityAction.process_actions) and
# examples/xcath_parallel.py:53-60: clamp the rate, integrate over dt, clamp the target; then
# batched_xcath.py:97-115 (_apply_tip_controls): the tip angle spread over the last ``tip_edges`` bend
# constraints as the offset the rod adds to its immutable rest Darboux (the other entries stay zero).
@wp.kernel
def advance_targets(
    actions: wp.array2d(dtype=wp.float32),
    rates: wp.array2d(dtype=wp.float32),
    lower: wp.array2d(dtype=wp.float32),
    upper: wp.array2d(dtype=wp.float32),
    dt: wp.float32,
    tip_axis: wp.vec3,
    tip_edges: wp.int32,
    B: wp.int32,
    polyline: wp.int32,
    targets: wp.array(dtype=wp.vec3),
    bend_offset: wp.array2d(dtype=wp.vec3),
):
    env = wp.tid()
    t = targets[env]
    out = wp.vec3(0.0, 0.0, 0.0)
    for k in range(3):
        a = wp.clamp(actions[env, k], -1.0, 1.0)
        out[k] = wp.clamp(t[k] + a * rates[env, k] * dt, lower[env, k], upper[env, k])
    targets[env] = out
    per_edge = out[2] / wp.float32(tip_edges)
    if polyline != 0:  # rest Darboux turns frames, the polyline follows midpoints: 2n - 1 half-turns (newton_catheter_physics.py:157-176)
        per_edge = wp.sin(out[2] / wp.float32(2 * tip_edges - 1))
    for e in range(B - tip_edges, B):
        bend_offset[env, e] = tip_axis * per_edge


# Ported from batched_xcath.py:68-96 (_apply_root_controls). Edits: one thread per env; positions and
# predicted positions are the body's (E, P) views of scene.q_prev and scene.q at (env, 0) (both written, so the root's
# velocity is zero); orientations/predicted are the rod's frame/frame_pred at env*F; previous_orientations
# dropped; track and base orientation per variant; an inactive root is left alone.
@wp.kernel
def root_controls(
    targets: wp.array(dtype=wp.vec3),
    track_start: wp.array(dtype=wp.vec3),
    track_direction: wp.array(dtype=wp.vec3),
    base_orientation: wp.array(dtype=wp.quat),
    variant: wp.array(dtype=wp.int32),
    flags: wp.array2d(dtype=wp.int32),
    max_delta: wp.array(dtype=wp.float32),
    F: wp.int32,
    mounted: wp.array(dtype=wp.int32),
    mount_pose: wp.array2d(dtype=wp.transform),
    q: wp.array2d(dtype=wp.vec3),
    q_prev: wp.array2d(dtype=wp.vec3),
    frame: wp.array(dtype=wp.quat),
    frame_pred: wp.array(dtype=wp.quat),
    root_target: wp.array2d(dtype=wp.vec3),
):
    env = wp.tid()
    v = variant[env]
    root = 0  # the root node of the env's (E, P) view
    if (flags[env, root] & ACTIVE) == 0:
        return
    start, direction, base = track_frame(track_start[v], track_direction[v], base_orientation[v], mounted[0], mount_pose[env, 0])
    target = start + direction * targets[env][0]
    delta = target - q[env, root]
    length = wp.length(delta)
    limit = max_delta[env]
    if length > limit and length > 1.0e-12:
        target = q[env, root] + delta * (limit / length)
    q[env, root] = target
    q_prev[env, root] = target
    root_target[env, 0] = target

    twist = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), targets[env][1])
    orientation = base * twist
    frame[env * F] = orientation
    frame_pred[env * F] = orientation


# Ported from batched_xcath.py:116-146 (_project_track_guidance). Edits: (env, node) prologue; predicted
# positions are the body's (E, P) view of scene.q; track_enabled dropped (disable the system instead);
# the last tip_edge_count nodes are guided as well while on the track (see the module docstring); track per
# variant, length and stiffness live per env; inactive nodes skipped.
@wp.kernel
def track_guidance(
    inv_mass: wp.array2d(dtype=wp.float32),
    flags: wp.array2d(dtype=wp.int32),
    track_start: wp.array(dtype=wp.vec3),
    track_direction: wp.array(dtype=wp.vec3),
    base_orientation: wp.array(dtype=wp.quat),
    variant: wp.array(dtype=wp.int32),
    track_length: wp.array(dtype=wp.float32),
    track_stiffness: wp.array(dtype=wp.float32),
    mounted: wp.array(dtype=wp.int32),
    mount_pose: wp.array2d(dtype=wp.transform),
    root_target: wp.array2d(dtype=wp.vec3),
    q: wp.array2d(dtype=wp.vec3),
    impulse: wp.array2d(dtype=wp.vec3),
    moment: wp.array2d(dtype=wp.vec3),
):
    env, local = wp.tid()
    if local == 0:
        return
    particle = local
    if inv_mass[env, particle] <= 0.0 or (flags[env, particle] & ACTIVE) == 0:
        return
    v = variant[env]
    position = q[env, particle]
    start, direction, base = track_frame(track_start[v], track_direction[v], base_orientation[v], mounted[0], mount_pose[env, 0])
    t = wp.dot(position - start, direction)
    if t < 0.0 or t > track_length[env]:
        return
    closest = start + direction * t
    dx = (closest - position) * track_stiffness[env]
    q[env, particle] = position + dx
    snap = dx / inv_mass[env, particle]  # m * dx the guide imposed (the coupling's products)
    wp.atomic_add(impulse, env, 0, snap)
    wp.atomic_add(moment, env, 0, wp.cross(position - root_target[env, 0], snap))


@wp.func
def track_frame(start: wp.vec3, direction: wp.vec3, base: wp.quat, mounted: wp.int32, mount: wp.transform):
    """The track's start, direction and the root's rest frame: the rest root's, or the mount's (the rest tangent carried
    from the rest frame into the mount's)."""
    if mounted == 0:
        return start, direction, base
    frame = wp.normalize(wp.transform_get_rotation(mount))
    return wp.transform_get_translation(mount), wp.quat_rotate(frame, wp.quat_rotate_inv(base, direction)), frame


# The root's hold (rod_direct): the first edge's stretch multipliers are the impulse its constraint would have given the
# locked root (compute_corrections moves node 0 by inv_m0 * lambda[0:3]), and the contraction of node 0's rotational
# Jacobian columns with all six multipliers the angular impulse it would have given the root's frame (compute_corrections'
# dtheta0; xpbd_rod_solver.py _xr_proximal_wrench); the driver imposed their negatives. The Jacobian is the substep's
# last iteration's.
@wp.kernel
def root_hold(lambda_sum: wp.array(dtype=wp.float32), jacobian_rot: wp.array(dtype=wp.float32), Ed: wp.int32,
              impulse: wp.array2d(dtype=wp.vec3), moment: wp.array2d(dtype=wp.vec3)):
    env = wp.tid()
    edge = env * Ed
    base = edge * 6
    impulse[env, 0] = impulse[env, 0] - wp.vec3(lambda_sum[base], lambda_sum[base + 1], lambda_sum[base + 2])
    turn = wp.vec3(0.0, 0.0, 0.0)
    for axis in range(3):
        value = float(0.0)
        for row in range(6):
            value += jacobian_rot[edge * 36 + row * 6 + axis] * lambda_sum[base + row]
        turn[axis] = value
    moment[env, 0] = moment[env, 0] - turn
