# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""I4h physics beside the host's rigid solver in one Newton model (docs/component-contracts.md).

``CoupledSolver`` is what Isaac Lab's NewtonManager steps when ``I4hSolverCfg.rigid_solver_cfg`` is set. One physics
step (contract section 3):
1. ``two_way``: the coupling wrenches of the previous step are added to ``body_f`` (after Lab's own force callbacks);
2. the rigid solver steps ``state_0`` in place (``rigid_substeps`` times at ``dt / rigid_substeps``: the host's own
   ``NewtonCfg.num_substeps``, which Lab runs at 1 here so the coupling stays once per physics step);
3. mounted instruments read their pose from the new ``body_q`` (``InstrumentControl.mount_pose``, component frame);
4. ``I4hSolver`` steps the same state in place;
5. ``two_way``: the coupling buffer is filled from the tools' ``impulse`` and ``moment`` products.

MuJoCo-Warp is the supported rigid solver: it never reads or writes particles. Featherstone integrates every particle
and applies particle contact forces to bodies, so it is rejected rather than integrating our particles twice.

Mounts (``I4hSolverCfg.mounts``): ``{name: {"body": label, "offset": ((x, y, z), (qx, qy, qz, qw))}}``. ``name`` is an
instrument, or a catheter (its rod body's name); ``label`` names one body per world (equal to ``model.body_label``, or
its last path element); ``offset`` is the mount's pose in the body frame (identity by default). A mounted instrument
follows its body; its pose actions, velocity limit and trocar are ignored, grip, energy and fire still come from the
actions. A mounted catheter's track and root rest frame follow the body (``catheter_control``); insertion, rotation and
tip bend still come from the actions.

The wrench, per mount: the deformable's force on the mount ``f = −Σ impulse / (sdt² · substeps)`` (the frame average, as
``tool_force`` computes it without its gains and cap) and its moment about the mount root ``m = −Σ moment / (sdt² ·
substeps)``, both in the component frame and summed over every system that publishes them for the mount (instruments:
contact, grasps, hook; a catheter: its driver's root hold and guide). On the body, in the world frame, as Newton's ``body_f`` (force, torque about the centre of
mass): ``R_e f`` and ``R_e m + (p_root − p_com) × R_e f``."""

from __future__ import annotations

import numpy as np
import warp as wp
from newton import ModelFlags
from newton.solvers import SolverBase

COUPLING_MODES = ("one_way", "two_way")
IDENTITY = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def check_rigid_solver_cfg(rigid_cfg, coupling_mode: str) -> None:
    """MuJoCo-Warp with its own contacts only (it leaves particles alone); a known coupling mode."""
    if coupling_mode not in COUPLING_MODES:
        raise ValueError(f"I4hSolverCfg.coupling_mode must be one of {COUPLING_MODES}, got {coupling_mode!r}")
    kind = str(getattr(rigid_cfg, "solver_type", ""))
    if kind != "mujoco_warp":
        raise ValueError(f"I4hSolverCfg.rigid_solver_cfg: the coupled step supports MuJoCo-Warp (MJWarpSolverCfg), got "
                         f"{kind or type(rigid_cfg).__name__!r}. Featherstone integrates every particle and applies "
                         "particle contact forces to bodies, so it would integrate i4h particles a second time.")
    if not getattr(rigid_cfg, "use_mujoco_contacts", True):
        raise ValueError("I4hSolverCfg.rigid_solver_cfg: use_mujoco_contacts=False needs Newton's collision pipeline, which "
                         "the coupled step does not run; keep MuJoCo's contacts")


def find_body(model, label: str, world: int) -> int:
    """The one body of ``world`` whose label is ``label`` or ends in ``/label``."""
    worlds = model.body_world.numpy()
    hits = [b for b in np.flatnonzero(worlds == world)
            if model.body_label[b] == label or str(model.body_label[b]).endswith("/" + label)]
    if len(hits) != 1:
        raise ValueError(f"mount body {label!r}: {len(hits)} bodies match in world {world}, expected one")
    return int(hits[0])


class MountGroup:
    """The mounts of one control system (``instrument_control``, ``catheter_control``): its mounted indices, their bodies
    per world and offsets, the systems publishing ``impulse``/``moment`` for it, and the coupling buffer."""

    def __init__(self, model, scene, systems, control, entries, two_way):
        E, dev = scene.envs, scene.device
        self.control, self.envs, self.count = control, E, len(entries)
        index = [control.names.index(name) for name, _ in entries]
        offsets = [wp.transform(wp.vec3(*map(float, spec.get("offset", IDENTITY)[0])),
                                wp.normalize(wp.quat(*map(float, spec.get("offset", IDENTITY)[1])))) for _, spec in entries]
        bodies = [[find_body(model, spec["body"], e) for e in range(E)] for _, spec in entries]
        self.instrument = wp.array(np.int32(index), dtype=wp.int32, device=dev)
        self.offset = wp.array(offsets, dtype=wp.transform, device=dev)
        self.body = wp.array(np.int32(bodies).T.reshape(E, self.count), dtype=wp.int32, device=dev)  # (E, M)
        self.sources = [s for s in systems if (s is control or getattr(s, "instruments", None) is control)
                        and getattr(s, "impulse", None) is not None and getattr(s, "moment", None) is not None]
        if two_way and getattr(control, "root_reaction", True) is False:
            raise ValueError(f"{type(control).__name__}: two_way needs the root's hold, measured with rod_direct only")
        self.force = wp.zeros((E, self.count), dtype=wp.vec3, device=dev)  # on the body, component frame, pending
        self.torque = wp.zeros((E, self.count), dtype=wp.vec3, device=dev)  # about the mount root, component frame
        for k in index:
            control.mount(k)


class Mounts:
    """Instruments and catheter drivers held by host bodies (``I4hSolverCfg.mounts``), grouped per control system."""

    def __init__(self, model, scene, systems, mounts: dict, two_way: bool):
        controls = [s for s in systems if hasattr(s, "mount_pose") and hasattr(s, "names")]
        per_control = {id(c): [] for c in controls}
        for name, spec in mounts.items():
            owners = [c for c in controls if name in c.names]
            if len(owners) != 1:
                known = [n for c in controls for n in c.names]
                raise ValueError(f"I4hSolverCfg.mounts: no instrument {name!r} (mountable: {known})" if not owners
                                 else f"I4hSolverCfg.mounts: {name!r} names {len(owners)} controls")
            if not isinstance(spec, dict) or set(spec) - {"body", "offset"} or "body" not in spec:
                raise ValueError(f"I4hSolverCfg.mounts[{name!r}] must be {{'body': label, 'offset': ((x, y, z), (qx, qy, qz, qw))}}")
            per_control[id(owners[0])].append((name, spec))
        self.groups = [MountGroup(model, scene, systems, c, per_control[id(c)], two_way) for c in controls if per_control[id(c)]]
        self.two_way = two_way
        xforms = scene.env_xforms or [IDENTITY] * scene.envs
        self.env_xform = wp.array([wp.transform(wp.vec3(*map(float, p)), wp.quat(*map(float, q))) for p, q in xforms],
                                  dtype=wp.transform, device=scene.device)

    def drive(self, state) -> None:
        """Step 3: each mount's pose in the component frame, ``T_e⁻¹ · body_q · offset``."""
        for g in self.groups:
            wp.launch(drive_mounts, dim=(g.envs, g.count), inputs=[state.body_q, g.body, g.offset, g.instrument, self.env_xform],
                      outputs=[g.control.mount_pose], device=g.force.device)

    def fill(self, substeps: int, dt: float) -> None:
        """Step 5: the frame-averaged wrench of every source, accumulated (cleared when it is written)."""
        if not self.two_way:
            return
        sdt = dt / float(substeps)
        for g in self.groups:
            for s in g.sources:
                wp.launch(fill_reaction, dim=(g.envs, g.count), inputs=[s.impulse, s.moment, g.instrument, 1.0 / (sdt * sdt * float(substeps))],
                          outputs=[g.force, g.torque], device=g.force.device)

    def apply(self, state, model) -> None:
        """Step 1: the buffer onto the bodies (world frame, torque about the centre of mass), then cleared."""
        if not self.two_way:
            return
        for g in self.groups:
            wp.launch(apply_reaction, dim=(g.envs, g.count),
                      inputs=[state.body_q, model.body_com, g.body, g.instrument, self.env_xform, g.control.root],
                      outputs=[g.force, g.torque, state.body_f], device=g.force.device)

    def reset(self, world_mask) -> None:
        """The masked worlds' pending reactions (a reset world has none)."""
        for g in self.groups:
            wp.launch(clear_reaction, dim=(g.envs, g.count), inputs=[world_mask], outputs=[g.force, g.torque], device=g.force.device)


class CoupledSolver(SolverBase):
    """The rigid solver, then ``I4hSolver``, in place on one state per physics step, with optional ``mounts``."""

    def __init__(self, model, rigid: SolverBase, ours: SolverBase, mounts: Mounts | None = None, rigid_substeps: int = 1):
        super().__init__(model)
        if int(rigid_substeps) < 1:
            raise ValueError(f"rigid_substeps must be at least 1, got {rigid_substeps}")
        self.rigid, self.ours, self.mounts, self.rigid_substeps = rigid, ours, mounts, int(rigid_substeps)

    def step(self, state_in, state_out, control, contacts, dt: float) -> None:
        if state_out is not state_in:
            raise RuntimeError("CoupledSolver steps one state in place (NewtonManager._use_single_state)")
        if self.mounts is not None:
            self.mounts.apply(state_in, self.model)
        for _ in range(self.rigid_substeps):  # MuJoCo-Warp: its own contacts; particles untouched; body_f held
            self.rigid.step(state_in, state_in, control, None, dt / self.rigid_substeps)
        if self.mounts is not None:
            self.mounts.drive(state_in)
        self.ours.step(state_in, state_in, control, None, dt)
        if self.mounts is not None:
            self.mounts.fill(self.ours.substeps, dt)

    def notify_model_changed(self, flags) -> None:
        self.rigid.notify_model_changed(flags)
        if int(flags) & int(ModelFlags.MODEL_PROPERTIES):  # gravity; body, joint and shape changes are the host's
            self.ours.notify_model_changed(ModelFlags.MODEL_PROPERTIES)

    def reset(self, state, world_mask=None, flags=None) -> None:
        """The rigid solver's internals and the masked worlds' pending reactions. Our physics resets through
        ``NewtonSim.reset_physics`` (masked, immediately)."""
        self.rigid.reset(state, world_mask=world_mask, flags=0 if flags is None else flags)
        if self.mounts is not None and world_mask is not None:
            self.mounts.reset(world_mask)


@wp.kernel
def drive_mounts(
    body_q: wp.array(dtype=wp.transform),
    body: wp.array2d(dtype=wp.int32),
    offset: wp.array(dtype=wp.transform),
    instrument: wp.array(dtype=wp.int32),
    env_xform: wp.array(dtype=wp.transform),
    mount_pose: wp.array2d(dtype=wp.transform),
):
    env, m = wp.tid()
    mount_pose[env, instrument[m]] = wp.transform_inverse(env_xform[env]) * (body_q[body[env, m]] * offset[m])


@wp.kernel
def fill_reaction(
    impulse: wp.array2d(dtype=wp.vec3),
    moment: wp.array2d(dtype=wp.vec3),
    instrument: wp.array(dtype=wp.int32),
    per_mass_dx: wp.float32,
    force: wp.array2d(dtype=wp.vec3),
    torque: wp.array2d(dtype=wp.vec3),
):
    env, m = wp.tid()
    k = instrument[m]
    force[env, m] = force[env, m] - impulse[env, k] * per_mass_dx  # the tissue pushes back on the tool
    torque[env, m] = torque[env, m] - moment[env, k] * per_mass_dx


@wp.kernel
def apply_reaction(
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body: wp.array2d(dtype=wp.int32),
    instrument: wp.array(dtype=wp.int32),
    env_xform: wp.array(dtype=wp.transform),
    root: wp.array2d(dtype=wp.vec3),
    force: wp.array2d(dtype=wp.vec3),
    torque: wp.array2d(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    env, m = wp.tid()
    b = body[env, m]
    xf = env_xform[env]
    f = wp.transform_vector(xf, force[env, m])
    tau = wp.transform_vector(xf, torque[env, m]) + wp.cross(
        wp.transform_point(xf, root[env, instrument[m]]) - wp.transform_point(body_q[b], body_com[b]), f)
    wp.atomic_add(body_f, b, wp.spatial_vector(f, tau))  # added to Lab's own forces; two tools may share a body
    force[env, m] = wp.vec3(0.0, 0.0, 0.0)
    torque[env, m] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def clear_reaction(world_mask: wp.array(dtype=wp.bool), force: wp.array2d(dtype=wp.vec3), torque: wp.array2d(dtype=wp.vec3)):
    env, m = wp.tid()
    if world_mask[env]:
        force[env, m] = wp.vec3(0.0, 0.0, 0.0)
        torque[env, m] = wp.vec3(0.0, 0.0, 0.0)
