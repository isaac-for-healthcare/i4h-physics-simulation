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

import math
from dataclasses import dataclass, field

import warp as wp
from i4h_physics.mount_storage import configuration, find_body, group_entries  # noqa: F401
from i4h_physics.observables import KINDS, ObservableKind, Observables, check_identities, identities
from i4h_physics.storage import scene_record
from newton import ModelFlags
from newton.solvers import SolverBase

COUPLING_MODES = ("one_way", "two_way")
IDENTITY = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))





def _remember(storage, owner, name):
    storage.register_package(identities=((owner, name, getattr(owner, name)),))


class MountGroup:
    """Checked aliases for one declared control's host mappings and pending reactions."""

    def __init__(self, scene, systems, control, entries, fields):
        self.control, self.envs, self.count = control, scene.envs, len(entries)
        for name, f in fields.items():
            setattr(self, name, scene.attr(f.domain, f.key))
        self.sources = [s for s in systems if (s is control or getattr(s, "instruments", None) is control)
                        and getattr(s, "impulse", None) is not None and getattr(s, "moment", None) is not None]
        self._sources = tuple(self.sources)
        self._layout = self.envs, self.count
        for name in (*fields, "control", "sources"):
            _remember(scene.storage, self, name)
        for name in ("mounted", "mount_pose", "root"):
            _remember(scene.storage, control, name)
        for source in self.sources:
            for name in ("impulse", "moment"):
                _remember(scene.storage, source, name)
            if source is not control:
                _remember(scene.storage, source, "instruments")

    def check(self):
        if (self.envs, self.count) != self._layout or len(self.sources) != len(self._sources) or any(
                a is not b for a, b in zip(self.sources, self._sources)):
            raise RuntimeError("mount group or sources changed; bind a fresh runtime")


class Mounts:
    """Validate and bind declared mounts without allocating or initializing storage."""

    def __init__(self, model, scene, systems, mounts: dict, two_way: bool):
        from i4h_physics.storage import Owner
        scene.storage.check()
        if model is not scene.model or model is not scene.storage.container(Owner.MODEL):
            raise ValueError("mounts require the selected Model")
        config = configuration(mounts, two_way)
        descriptors = getattr(scene, "mountable_domains", {})
        group_entries(descriptors, config)  # reject invalid target/spec before binding
        record = getattr(scene, "mount_domains", None)
        if record is None:
            raise RuntimeError("mount storage missing; call declare_mounts before schema registration")
        if config != record["config"]:
            raise ValueError("incompatible mount declaration; reconstruct the model")
        controls = {}
        for sid, entries, fields in record["groups"]:
            found = [s for s in systems if s.id == sid]
            if len(found) != 1 or tuple(getattr(found[0], "names", ())) != descriptors[sid].names:
                raise ValueError(f"mount control {sid!r} missing, duplicate or incompatible")
            control = controls[sid] = found[0]
            descriptor = descriptors[sid]
            if bool(getattr(control, "root_reaction", True)) != descriptor.root_reaction:
                raise ValueError("mount root-reaction capability differs from declaration")
            for name in ("mounted", "mount_pose", "root"):
                f = getattr(descriptor, name)
                if getattr(control, name) is not scene.attr(f.domain, f.key):
                    raise ValueError(f"mount control {name} does not alias its declaration")
        self.scene = scene
        self.groups = [MountGroup(scene, systems, controls[sid], entries, fields) for sid, entries, fields in record["groups"]]
        self.two_way = two_way
        f = record["env_xform"]
        self.env_xform = scene.attr(f.domain, f.key)
        self._groups, self._config = tuple(self.groups), config
        self._controls = tuple((controls[sid], descriptors[sid]) for sid in controls)
        self._selected = tuple(scene.storage.container(o) for o in (Owner.MODEL, Owner.STATE, Owner.CONTROL))
        for owner, names in ((self, ("scene", "groups", "env_xform")),
                             (scene, ("model", "state", "control"))):
            for name in names:
                _remember(scene.storage, owner, name)
        # Native buffers used by coupling and the rigid step belong to the selected containers.
        for owner in self._selected:
            for name, value in vars(owner).items():
                if isinstance(value, wp.array):
                    _remember(scene.storage, owner, name)
        scene.storage.register_package(checks=(self._check_layout,))

    def _check_layout(self):
        if self.two_way != self._config[1] or scene_record(self.scene).mounts != self._config:
            raise RuntimeError("mount configuration changed; reconstruct the model")
        if len(self.groups) != len(self._groups) or any(a is not b for a, b in zip(self.groups, self._groups)):
            raise RuntimeError("mount groups changed; bind a fresh runtime")
        for control, descriptor in self._controls:
            if control.id != descriptor.system_id or tuple(control.names) != descriptor.names or bool(
                    getattr(control, "root_reaction", True)) != descriptor.root_reaction:
                raise RuntimeError("mount control layout changed; reconstruct the model")
        for group in self.groups:
            group.check()

    def check(self, state=None, model=None, control=None):
        for selected, actual, name in zip(self._selected, (model, state, control), ("Model", "State", "Control")):
            if actual is not None and actual is not selected:
                raise ValueError(f"mounts require the selected {name}")
        self.scene.storage.check()

    def drive(self, state) -> None:
        """Step 3: each mount's pose in the component frame, ``T_e⁻¹ · body_q · offset``."""
        self.check(state=state)
        for g in self.groups:
            wp.launch(drive_mounts, dim=(g.envs, g.count), inputs=[state.body_q, g.body, g.offset, g.target, self.env_xform],
                      outputs=[g.control.mount_pose], device=g.force.device)

    def fill(self, substeps: int, dt: float) -> None:
        """Step 5: the frame-averaged wrench of every source, accumulated (cleared when it is written)."""
        self.check()
        if not self.two_way:
            return
        sdt = dt / float(substeps)
        for g in self.groups:
            for s in g.sources:
                wp.launch(fill_reaction, dim=(g.envs, g.count), inputs=[s.impulse, s.moment, g.target, 1.0 / (sdt * sdt * float(substeps))],
                          outputs=[g.force, g.torque], device=g.force.device)

    def apply(self, state, model) -> None:
        """Step 1: the buffer onto the bodies (world frame, torque about the centre of mass), then cleared."""
        self.check(state=state, model=model)
        if not self.two_way:
            return
        for g in self.groups:
            wp.launch(apply_reaction, dim=(g.envs, g.count),
                      inputs=[state.body_q, model.body_com, g.body, g.target, self.env_xform, g.control.root],
                      outputs=[g.force, g.torque, state.body_f], device=g.force.device)

    def reset(self, world_mask) -> None:
        """The masked worlds' pending reactions (a reset world has none)."""
        self.check()
        for g in self.groups:
            wp.launch(clear_reaction, dim=(g.envs, g.count), inputs=[world_mask], outputs=[g.force, g.torque], device=g.force.device)


class CoupledSolver(SolverBase):
    """The rigid solver, then ``I4hSolver``, in place on one state per physics step, with optional ``mounts``."""

    ObservableKind = ObservableKind
    SUPPORTED_OBSERVABLES = KINDS | frozenset({SolverBase.ObservableKind.CONTACT_F})

    @dataclass(eq=False)
    class Observables(Observables):
        _rigid: object = field(default=None, repr=False)
        _ours: object = field(default=None, repr=False)

        def select(self, kinds):
            selected = super().select(kinds)
            selected._rigid = self._rigid.select(selected.kinds - KINDS)
            selected._ours = self._ours.select(selected.kinds & KINDS)
            selected._identities = identities(selected)
            return selected

    @property
    def supported_observables(self):
        return (self.SUPPORTED_OBSERVABLES & self.rigid.supported_observables) | self.ours.supported_observables

    def observables(self, kinds, *, contacts=None, requires_grad=None):
        result = super().observables(kinds, contacts=contacts, requires_grad=requires_grad)
        result._rigid = self.rigid.observables(result.kinds - KINDS, contacts=contacts, requires_grad=requires_grad)
        result._ours = self.ours.observables(result.kinds & KINDS, requires_grad=requires_grad)
        for kind in result.kinds & KINDS:
            setattr(result, kind, getattr(result._ours, kind))
        result.row_keys = result._ours.row_keys
        result._identities = identities(result)
        return result

    def __init__(self, model, rigid: SolverBase, ours: SolverBase, mounts: Mounts | None = None, rigid_substeps: int = 1):
        super().__init__(model)
        if int(rigid_substeps) < 1:
            raise ValueError(f"rigid_substeps must be at least 1, got {rigid_substeps}")
        self.rigid, self.ours, self.mounts, self.rigid_substeps = rigid, ours, mounts, int(rigid_substeps)

    def step(self, state_in, state_out, control, contacts, dt: float, observables=None) -> None:
        self.validate_observables(observables, contacts)
        if observables is not None:
            check_identities(observables._identities)
        if state_out is not state_in:
            raise RuntimeError("CoupledSolver steps one state in place (NewtonManager._use_single_state)")
        self.ours._check_state(state_in, "coupled step")
        if control is not self.ours.control:
            raise ValueError("CoupledSolver: only the bound control is supported")
        self.ours.check_binding()
        if self.ours.dt is None or not math.isfinite(float(dt)) or abs(float(dt) - self.ours.dt) > 1.0e-9 * max(1.0, self.ours.dt):
            raise ValueError("CoupledSolver dt differs from the prepared dt")
        if self.mounts is not None:
            self.mounts.check(state_in, self.model, control)
            self.mounts.apply(state_in, self.model)
        for _ in range(self.rigid_substeps):  # MuJoCo-Warp: its own contacts; particles untouched; body_f held
            self.rigid.step(state_in, state_in, control, contacts if observables is not None else None,
                            dt / self.rigid_substeps, observables=None if observables is None else observables._rigid)
        if self.mounts is not None:
            self.mounts.drive(state_in)
        self.ours.step(state_in, state_in, control, None, dt,
                       observables=None if observables is None else observables._ours)
        if observables is not None and observables.contact_f is not None:
            wp.copy(observables.contact_f, observables._rigid.contact_f)
        if self.mounts is not None:
            self.mounts.fill(self.ours.substeps, dt)

    def get_max_contact_count(self):
        return self.rigid.get_max_contact_count()

    def update_contacts(self, contacts, state=None):
        return self.rigid.update_contacts(contacts, state)

    def notify_model_changed(self, flags) -> None:
        self.rigid.notify_model_changed(flags)
        if int(flags) & int(ModelFlags.MODEL_PROPERTIES):  # gravity; body, joint and shape changes are the host's
            self.ours.notify_model_changed(ModelFlags.MODEL_PROPERTIES)

    def reset(self, state, world_mask=None, flags=0) -> None:
        """Refresh child caches; pending reactions belong to the current episode."""
        from newton import StateFlags

        self.ours._check_state(state, "coupled reset")
        self.ours.check_binding()
        if self.mounts is not None:
            self.mounts.check(state, self.model, self.ours.control)
        particle = int(StateFlags.PARTICLE_Q | StateFlags.PARTICLE_QD | StateFlags.PARTICLE_F)
        selected = int(StateFlags.ALL) if flags is None else int(flags)
        self.rigid.reset(state, world_mask=world_mask, flags=selected & ~particle)
        self.ours.reset(state, world_mask=world_mask, flags=selected & particle)

    def save_i4h(self, dst_state, dst_control, mask=None) -> None:
        """i4h rows only (pending mount reactions included); rigid body state is the host's to copy."""
        self.ours.save_i4h(dst_state, dst_control, mask)

    def restore_i4h(self, src_state, src_control, mask=None) -> None:
        self.ours.restore_i4h(src_state, src_control, mask)

    def restore_episode(self, state, control, world_mask=None) -> None:
        if self.mounts is not None:
            self.mounts.check(state, self.model, control)
        self.ours.restore_episode(state, control, world_mask)
        self.rigid.reset(state, world_mask=world_mask, flags=0)
        if self.mounts is not None:
            self.mounts.reset(self.ours._all if world_mask is None else world_mask)


@wp.kernel
def drive_mounts(
    body_q: wp.array(dtype=wp.transform),
    body: wp.array2d(dtype=wp.int32),
    offset: wp.array(dtype=wp.transform),
    target: wp.array(dtype=wp.int32),
    env_xform: wp.array(dtype=wp.transform),
    mount_pose: wp.array2d(dtype=wp.transform),
):
    env, m = wp.tid()
    mount_pose[env, target[m]] = wp.transform_inverse(env_xform[env]) * (body_q[body[env, m]] * offset[m])


@wp.kernel
def fill_reaction(
    impulse: wp.array2d(dtype=wp.vec3),
    moment: wp.array2d(dtype=wp.vec3),
    target: wp.array(dtype=wp.int32),
    per_mass_dx: wp.float32,
    force: wp.array2d(dtype=wp.vec3),
    torque: wp.array2d(dtype=wp.vec3),
):
    env, m = wp.tid()
    k = target[m]
    force[env, m] = force[env, m] - impulse[env, k] * per_mass_dx  # the tissue pushes back on the tool
    torque[env, m] = torque[env, m] - moment[env, k] * per_mass_dx


@wp.kernel
def apply_reaction(
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body: wp.array2d(dtype=wp.int32),
    target: wp.array(dtype=wp.int32),
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
        wp.transform_point(xf, root[env, target[m]]) - wp.transform_point(body_q[b], body_com[b]), f)
    wp.atomic_add(body_f, b, wp.spatial_vector(f, tau))  # added to Lab's own forces; two tools may share a body
    force[env, m] = wp.vec3(0.0, 0.0, 0.0)
    torque[env, m] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def clear_reaction(world_mask: wp.array(dtype=wp.bool), force: wp.array2d(dtype=wp.vec3), torque: wp.array2d(dtype=wp.vec3)):
    env, m = wp.tid()
    if world_mask[env]:
        force[env, m] = wp.vec3(0.0, 0.0, 0.0)
        torque[env, m] = wp.vec3(0.0, 0.0, 0.0)
