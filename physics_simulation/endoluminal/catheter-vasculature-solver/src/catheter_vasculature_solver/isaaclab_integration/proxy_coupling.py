# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Proxy coupling primitives, following Newton's coupled-solver framework.

Newton grew an experimental coupled-solver framework in 1.4.0 under
``newton.solvers.experimental.coupled``. Its proxy path is a relaxed
fixed-point iteration: synchronise the source pose into the destination solver,
cancel whatever the source already accounted for so it is not counted twice,
harvest the destination's reaction, then blend that reaction against the
previous iterate before feeding it back. This module carries that recipe onto
the Newton revision this package pins, which predates the framework.

:class:`DriveRootFollower` is the pose synchronisation: it puts the rod's root
on the drive mount inside every substep. :class:`DriveReactionCoupler` is the
harvest and the blend. Three parts of the upstream recipe are not carried over:

* There is no rewind. Upstream re-steps the source from the start of the step
  on each iteration, so removing the previous iterate's feedback leaves the
  destination solving from the free motion. With a single pass the rigid solver
  keeps the lagged reaction in the body's state, and rewinding it out of the
  pose the rod sees makes the coupling unstable; see :class:`DriveRootFollower`.
* The proxy has no inertia. The root keeps zero inverse mass, so the rod sees
  its holder as infinitely heavy within a substep. A finite root mass would
  also expose the root to the vessel containment, track guidance and wall
  friction kernels, which identify the root by its zero inverse mass.
* Each substep makes one pass, so ``relaxation`` filters the reaction across
  substeps rather than relaxing an iteration, and there is no Aitken update.

What is harvested differs from upstream. Upstream reads a proxy *body*'s
momentum change, which works because their sub-solvers move the proxy during
their own solve.
:class:`~catheter_vasculature_solver.newton_rod_solver.NewtonCathRodSolver`
never writes body state, and the rod's proximal particle carries zero mass
because ``lock_root`` pins it, so neither a body-momentum nor a
particle-momentum harvest can see anything: both would report exactly zero and
leave two-way coupling silently behaving as one-way. The reaction is taken from
the rod's own constraints instead, through
:meth:`~catheter_vasculature_solver.xpbd_rod_solver.XPBDRodSolver.proximal_wrench`,
which reports the force and moment transmitted to its proximal holder.
That keeps solver internals inside the solver, the way upstream keeps its
``CouplingInterface`` hooks on each sub-solver rather than in the coupler.

Upstream's gravity cancellation has no counterpart here for the same reason.
It exists to stop a proxy body being pulled by gravity twice; our feedback
comes from a constraint force that legitimately carries the weight of the rod
hanging off the drive point, and the driving body's own weight stays with the
rigid solver.
"""

from __future__ import annotations

import os

import warp as wp

from ..host_arrays import env_ids_from_world_mask

#: Steps between wrench reports when the debug flag is on but no interval given.
DEFAULT_DEBUG_INTERVAL = 60

#: Env var enabling the coupling diagnostic. See :func:`debug_log_interval`.
DEBUG_ENV_VAR = "I4H_COUPLING_DEBUG"

_DEBUG_OFF = frozenset({"", "0", "false", "no", "off"})
_DEBUG_ON = frozenset({"1", "true", "yes", "on"})


def debug_log_interval(environ=None) -> int | None:
    """How often to report coupling forces, or ``None`` when reporting is off.

    Reading the harvested wrench means copying it off the device, so the
    diagnostic is opt-in: two-way coupling runs silently unless
    ``I4H_COUPLING_DEBUG`` asks for it. A truthy word turns it on at
    :data:`DEFAULT_DEBUG_INTERVAL`, and a positive integer sets the step
    interval directly, so ``=1`` reports every step and ``=200`` occasionally.

    Args:
        environ: Mapping to read instead of ``os.environ``, for tests.

    Returns:
        Step interval, or ``None`` if the diagnostic is disabled.

    Raises:
        ValueError: If the value is neither a recognised word nor a positive
            integer, since silently ignoring it would look like the flag had no
            effect.
    """
    source = os.environ if environ is None else environ
    raw = source.get(DEBUG_ENV_VAR)
    if raw is None:
        return None

    value = str(raw).strip().lower()
    if value in _DEBUG_OFF:
        return None
    if value in _DEBUG_ON:
        return DEFAULT_DEBUG_INTERVAL

    try:
        interval = int(value)
    except ValueError:
        raise ValueError(
            f"{DEBUG_ENV_VAR}={raw!r} is not understood; use a positive integer step "
            f"interval, or one of {sorted(_DEBUG_ON)} to report every "
            f"{DEFAULT_DEBUG_INTERVAL} steps."
        ) from None
    if interval <= 0:
        raise ValueError(
            f"{DEBUG_ENV_VAR}={raw!r} must be positive; use one of {sorted(_DEBUG_OFF)} to "
            "turn the diagnostic off."
        )
    return interval


# ---------------------------------------------------------------------------
# 0. Name the drive body
# ---------------------------------------------------------------------------


def resolve_drive_body_index(labels, name: str, bodies_per_env: int) -> int:
    """Find the body called *name* and return its index within one env.

    Newton's ``Model`` carries no body names, but the ``ModelBuilder`` that
    produced it keeps ``body_label``, and IsaacLab is still holding that builder
    when the solver is constructed. Those labels are full USD paths, so a
    scene naming its flange ``"TCP"`` is matched against the last path
    component of ``/World/envs/env_0/Robot/TCP``. A caller that wants to be
    unambiguous can pass the whole path instead.

    Only the first env's slice of the labels is searched, because
    ``drive_body_index`` is defined within a single env and the manager
    replicates it across the rest.

    Args:
        labels: ``body_label`` from the builder, ordered by body index.
        name: Body leaf name, or a full USD path.
        bodies_per_env: Body count in one env, bounding the search.

    Returns:
        Index of the named body within one env.

    Raises:
        ValueError: If *name* matches no body, or more than one in an env.
    """
    if bodies_per_env <= 0:
        raise ValueError(f"bodies_per_env must be positive, got {bodies_per_env}")

    within_env = [str(label) for label in list(labels)[:bodies_per_env]]
    matches = [
        index
        for index, label in enumerate(within_env)
        if label == name or label.rsplit("/", 1)[-1] == name
    ]

    if not matches:
        raise ValueError(
            f"no body named {name!r} in the Newton model; one env holds "
            f"{[label.rsplit('/', 1)[-1] for label in within_env]}"
        )
    if len(matches) > 1:
        ambiguous = [within_env[index] for index in matches]
        raise ValueError(
            f"{name!r} matches {len(matches)} bodies in a single env ({ambiguous}); "
            "pass the full USD path to disambiguate"
        )
    return matches[0]


# ---------------------------------------------------------------------------
# 1. Harvest the reaction as a body wrench
# ---------------------------------------------------------------------------


@wp.func
def _drive_reaction_wrench(wrench: wp.spatial_vector, arm: wp.vec3):
    force = wp.spatial_top(wrench)
    return wp.spatial_vector(force, wp.spatial_bottom(wrench) + wp.cross(arm, force))


@wp.kernel
def harvest_drive_reaction_kernel(
    proximal_wrench: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    mount_local: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    drive_body: wp.array(dtype=wp.int32),
    out_coupling_forces: wp.array(dtype=wp.spatial_vector),
):
    """Place each rod's proximal reaction on the body holding it.

    The force and intrinsic moment arrive resolved by the rod solver; this
    carries them onto the right body and adds the moment produced by *where on that body*
    the wire is held. Accumulation is atomic because several rods may be held by
    the same body.

    The moment arm is the mount point, fixed in the body's frame, and not the
    rod's proximal particle. The wire is gripped by hardware bolted to the body
    -- a hemostatic valve on a flange -- so that mount is where the reaction
    is transmitted, and it does not move relative to the body no matter how much
    wire has been fed. Using the particle instead ties the arm to how far the
    wire has been inserted: under a roller drive the proximal particle travels
    away up the vessel, so the arm grows without bound and invents a torque that
    scales with insertion depth. Measured on the s0011 twin, the flange drifted
    off the access site by 28 mm after one second and 34 mm after two, growing
    with fed depth.

    Args:
        proximal_wrench: Per-env world force and moment about the rod root.
        body_com: Per-body center of mass in each body's local frame.
        mount_local: Per-env grip point in the drive body's local frame.
        body_q: Body transforms, used to orient the moment arm into world.
        drive_body: Per-env body index holding the rod. Negative disables.
        out_coupling_forces: Per-body wrench accumulator.
    """
    env = wp.tid()
    body = drive_body[env]
    if body < 0:
        return

    arm = wp.transform_vector(body_q[body], mount_local[env] - body_com[body])
    wp.atomic_add(out_coupling_forces, body, _drive_reaction_wrench(proximal_wrench[env], arm))


@wp.kernel
def _harvest_env_reactions(
    proximal_wrench: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    mount_local: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    drive_body: wp.array(dtype=wp.int32),
    relaxation: float,
    env_forces: wp.array(dtype=wp.spatial_vector),
    env_previous: wp.array(dtype=wp.spatial_vector),
):
    """Retain each rod's share so resetting one cannot erase another's load."""
    env = wp.tid()
    body = drive_body[env]
    if body < 0:
        return
    arm = wp.transform_vector(body_q[body], mount_local[env] - body_com[body])
    wrench = _drive_reaction_wrench(proximal_wrench[env], arm)
    env_previous[env] = env_forces[env]
    if relaxation != 1.0:
        wrench = relaxation * wrench + (1.0 - relaxation) * env_previous[env]
    env_forces[env] = wrench


@wp.kernel
def _accumulate_env_reactions(
    drive_body: wp.array(dtype=wp.int32),
    env_forces: wp.array(dtype=wp.spatial_vector),
    env_previous: wp.array(dtype=wp.spatial_vector),
    body_forces: wp.array(dtype=wp.spatial_vector),
    body_previous: wp.array(dtype=wp.spatial_vector),
):
    env = wp.tid()
    body = drive_body[env]
    if body >= 0:
        wp.atomic_add(body_forces, body, env_forces[env])
        wp.atomic_add(body_previous, body, env_previous[env])


@wp.kernel
def _reset_env_reactions(
    env_ids: wp.array(dtype=wp.int32),
    env_forces: wp.array(dtype=wp.spatial_vector),
    env_previous: wp.array(dtype=wp.spatial_vector),
    proximal_wrench: wp.array(dtype=wp.spatial_vector),
):
    env = env_ids[wp.tid()]
    zero = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    env_forces[env] = zero
    env_previous[env] = zero
    proximal_wrench[env] = zero


# ---------------------------------------------------------------------------
# 2. Relaxation blend against the previous substep's reaction
# ---------------------------------------------------------------------------


@wp.kernel
def stash_coupling_forces_kernel(
    coupling_forces: wp.array(dtype=wp.spatial_vector),
    out_previous: wp.array(dtype=wp.spatial_vector),
):
    """Save the current feedback so the next harvest can be under-relaxed."""
    i = wp.tid()
    out_previous[i] = coupling_forces[i]


@wp.kernel
def blend_coupling_forces_kernel(
    relaxation: float,
    previous: wp.array(dtype=wp.spatial_vector),
    coupling_forces: wp.array(dtype=wp.spatial_vector),
):
    """Blend freshly harvested feedback with the saved lagged value.

    Below ``1`` under-relaxes the update, ``1`` keeps the harvest unchanged and
    above ``1`` over-relaxes it, matching upstream's ``proxy_relaxation``.
    """
    i = wp.tid()
    coupling_forces[i] = relaxation * coupling_forces[i] + (1.0 - relaxation) * previous[i]


# ---------------------------------------------------------------------------
# 3. Inject the lagged feedback into the rigid solver
# ---------------------------------------------------------------------------


@wp.kernel
def apply_coupling_forces_kernel(
    coupling_forces: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    """Add the accumulated coupling wrench onto Newton's body force input."""
    i = wp.tid()
    body_f[i] = body_f[i] + coupling_forces[i]


class DriveReactionCoupler:
    """Feed the catheter's proximal reaction back to the body holding it.

    Stores current and previous reactions per environment, then accumulates
    them onto bodies. This preserves surviving reactions during selective reset,
    including when several rods share a holder. One instance spans every env,
    because Newton keeps all bodies and particles in single flat arrays.

    The call order over a substep mirrors upstream's ``LAGGED`` proxy mode with
    one pass::

        apply(state_0)                  # feedback harvested one substep ago
        ... rigid solve ...
        follower.sync(...)              # root onto the solved mount, if following
        ... rod solve ...
        harvest(state, rod)             # reaction the rod just transmitted

    There is no pre-step snapshot, unlike upstream: the reaction comes from the
    rod's constraint state after its solve rather than from a velocity
    difference across it, and the rod supplies its own substep scaling.
    Because there is one pass per substep, ``relaxation`` below 1 low-pass
    filters the reaction across substeps; it does not relax an iteration.

    Args:
        body_count: Number of bodies in the Newton model.
        num_envs: Number of rods in the particle range.
        root_offset: Index of the first rod's proximal particle.
        root_stride: Particle count per rod.
        drive_bodies: Per-env body index holding each rod's proximal end.
        mount_local: Per-env grip point in the drive body's frame. Defaults to
            the body origin. Intrinsic bending and twisting moments are
            transmitted even when the mount coincides with the center of mass.
        body_com: Per-body local centers of mass from the Newton model.
            Defaults to zero for models whose body origin is their COM.
        relaxation: Blend weight against the previous iterate.
        device: Warp device for the persistent buffers.
    """

    def __init__(
        self,
        *,
        body_count: int,
        num_envs: int,
        root_offset: int,
        root_stride: int,
        drive_bodies,
        mount_local=None,
        body_com=None,
        relaxation: float = 1.0,
        device=None,
    ) -> None:
        if body_count <= 0:
            raise ValueError(f"body_count must be positive, got {body_count}")
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if root_stride <= 0:
            raise ValueError(f"root_stride must be positive, got {root_stride}")
        if root_offset < 0:
            raise ValueError(f"root_offset must be non-negative, got {root_offset}")
        if relaxation <= 0.0:
            raise ValueError(
                f"relaxation must be positive, got {relaxation}. Zero would freeze the "
                "feedback at its initial value and never accept a harvested update."
            )

        drive = [int(value) for value in drive_bodies]
        if len(drive) != num_envs:
            raise ValueError(f"drive_bodies has {len(drive)} entries but num_envs is {num_envs}")
        for body in drive:
            if body >= body_count:
                raise IndexError(f"drive body {body} is out of range for {body_count} bodies")

        self.body_count = int(body_count)
        self.num_envs = int(num_envs)
        self.root_offset = int(root_offset)
        self.root_stride = int(root_stride)
        self.relaxation = float(relaxation)
        self._device = device

        mounts = [(0.0, 0.0, 0.0)] * num_envs if mount_local is None else [tuple(m) for m in mount_local]
        if len(mounts) != num_envs:
            raise ValueError(f"mount_local has {len(mounts)} entries but num_envs is {num_envs}")
        for mount in mounts:
            if len(mount) != 3:
                raise ValueError(f"each mount_local entry must be a 3-vector, got {mount!r}")

        self._drive_bodies = wp.array(drive, dtype=wp.int32, device=device)
        self._mount_local = wp.array(mounts, dtype=wp.vec3, device=device)
        self._coupling_forces = wp.zeros(self.body_count, dtype=wp.spatial_vector, device=device)
        self._previous = wp.zeros(self.body_count, dtype=wp.spatial_vector, device=device)
        self._env_forces = wp.zeros(self.num_envs, dtype=wp.spatial_vector, device=device)
        self._env_previous = wp.zeros(self.num_envs, dtype=wp.spatial_vector, device=device)
        self._reset_env_ids = wp.empty(self.num_envs, dtype=wp.int32, device=device)
        self._reset_env_ids_host = wp.empty(
            self.num_envs, dtype=wp.int32, device="cpu", pinned=wp.get_device(device).is_cuda,
        )
        # No proximal-particle buffer: the moment arm is the mount on the drive
        # body, so where the particle has fed to no longer enters the wrench.
        self._proximal_wrench = wp.zeros(self.num_envs, dtype=wp.spatial_vector, device=device)
        self._body_com = wp.zeros(self.body_count, dtype=wp.vec3, device=device) if body_com is None else body_com

    @property
    def coupling_forces(self):
        """Per-body wrench most recently harvested and blended."""
        return self._coupling_forces

    @property
    def drive_bodies(self):
        """Per-env index of the body each rod's reaction is applied to."""
        return self._drive_bodies

    @property
    def mount_local(self):
        """Per-env point in the drive body's frame where the wire is gripped."""
        return self._mount_local

    def report(self, env: int = 0):
        """Copy back what the catheter is doing to its holder, for diagnostics.

        Synchronises with the device, so this is for the opt-in debug path
        rather than the step loop. Safe to call there because a two-way scene
        already runs without CUDA graph capture.

        Args:
            env: Which env to report.

        Returns:
            ``(body_index, force, torque)``; the force is the rod's own
            proximal reaction, and torque includes both the intrinsic rod
            moment and the mount's lever arm about the body's center of mass.

        Raises:
            IndexError: If *env* is out of range.
        """
        if not 0 <= env < self.num_envs:
            raise IndexError(f"env {env} is out of range for {self.num_envs} envs")

        body = int(self._drive_bodies.numpy()[env])
        wrench = self._coupling_forces.numpy()[body]
        return body, wrench[:3].copy(), wrench[3:].copy()

    def reset(self, world_mask=None) -> None:
        """Drop feedback only for the selected worlds; ``None`` resets all.

        A reset env restarts from its rest pose, so a wrench harvested from the
        previous episode would load the driving body for contact the catheter
        is no longer making.
        """
        env_ids = env_ids_from_world_mask(world_mask, self.num_envs)
        if env_ids is None:
            for buffer in (self._coupling_forces, self._previous, self._env_forces,
                           self._env_previous, self._proximal_wrench):
                buffer.zero_()
            return
        if len(env_ids) == 0:
            return
        self._reset_env_ids_host.numpy()[:len(env_ids)] = env_ids
        wp.copy(self._reset_env_ids, self._reset_env_ids_host, count=len(env_ids))
        wp.launch(
            _reset_env_reactions, dim=len(env_ids),
            inputs=[self._reset_env_ids, self._env_forces, self._env_previous, self._proximal_wrench],
            device=self._device,
        )
        self._publish_env_reactions()

    def _publish_env_reactions(self) -> None:
        """Rebuild body loads from surviving per-environment contributions."""
        self._coupling_forces.zero_()
        self._previous.zero_()
        wp.launch(
            _accumulate_env_reactions, dim=self.num_envs,
            inputs=[self._drive_bodies, self._env_forces, self._env_previous,
                    self._coupling_forces, self._previous],
            device=self._device,
        )

    def apply(self, state) -> None:
        """Add the lagged feedback onto ``state.body_f``.

        Call after the force buffers are cleared and before the rigid solve, so
        the rigid solver sees the reaction the rod reported one substep ago.
        """
        body_f = getattr(state, "body_f", None)
        if body_f is None:
            return
        wp.launch(
            apply_coupling_forces_kernel,
            dim=self.body_count,
            inputs=[self._coupling_forces, body_f],
            device=self._device,
        )

    def harvest(self, state, rod_solver) -> None:
        """Harvest the rod's proximal reaction and blend it with the previous iterate."""
        particle_q = getattr(state, "particle_q", None)
        body_q = getattr(state, "body_q", None)
        if particle_q is None or body_q is None:
            raise ValueError(
                "Two-way coupling needs particle_q and body_q in State; got "
                f"particle_q={particle_q is not None}, body_q={body_q is not None}."
            )

        rod_solver.proximal_wrench(out=self._proximal_wrench)

        wp.launch(
            _harvest_env_reactions,
            dim=self.num_envs,
            inputs=[
                self._proximal_wrench,
                self._body_com,
                self._mount_local,
                body_q,
                self._drive_bodies,
                self.relaxation,
                self._env_forces,
                self._env_previous,
            ],
            device=self._device,
        )
        self._publish_env_reactions()


# ---------------------------------------------------------------------------
# 4. Hold the root on the drive mount
# ---------------------------------------------------------------------------


@wp.kernel
def _follow_drive_mount_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_q_begin: wp.array(dtype=wp.transform),
    drive_body: wp.array(dtype=wp.int32),
    mount_local: wp.array(dtype=wp.vec3),
    rod_orientations: wp.array(dtype=wp.quat),
    root_stride: int,
    bound: wp.array(dtype=wp.int32),
    root_rotation_local: wp.array(dtype=wp.quat),
    out_positions: wp.array(dtype=wp.vec3),
    out_orientations: wp.array(dtype=wp.quat),
):
    """Place each rod's root on its drive mount at the pose the rigid solve reached.

    The root's material frame keeps the rotation it had relative to the body
    when the env was bound, captured against the body's pose at the start of
    the substep, so a mount does not impose the body's own axes on the wire.
    """
    env = wp.tid()
    body = drive_body[env]
    root = env * root_stride
    rotation = wp.transform_get_rotation(body_q[body])

    if bound[env] == 0:
        begin_rotation = wp.transform_get_rotation(body_q_begin[body])
        root_rotation_local[env] = wp.quat_inverse(begin_rotation) * rod_orientations[root]
        bound[env] = 1

    out_positions[env] = wp.transform_point(body_q[body], mount_local[env])
    out_orientations[env] = wp.normalize(rotation * root_rotation_local[env])


@wp.kernel
def _unbind_envs_kernel(env_ids: wp.array(dtype=wp.int32), bound: wp.array(dtype=wp.int32)):
    bound[env_ids[wp.tid()]] = 0


class DriveRootFollower:
    """Hold each rod's proximal end on the body that grips it, inside the substep.

    :class:`DriveReactionCoupler` lets the body feel the rod; this is the other
    half of the loop, the rod being held where the body is. It runs between the
    rigid and rod solves of every substep, so the root follows the body with no
    lag, and it reads the body pose from the rigid solver's output rather than
    from whatever a task commanded once per environment step.

    The root is placed at ``mount_local`` on the body, the same grip point the
    coupler uses for the moment arm, so the force and the lever it acts through
    describe one mount.

    The root goes where the rigid solve put the body, lagged reaction included,
    and is not rewound to the body's free pose. The rigid solver keeps that
    reaction in the body's velocity, so a root placed as though it had not been
    applied lets the rod respond to the holder's position one substep late. In a
    holder-against-spring model that extra delay diverges from
    ``k·dt²/M ≈ 0.1``, while the root on the solved pose is a staggered
    symplectic Euler step, stable up to ``k·dt²/M < 4``.

    The follower owns the root while it is active. A rate or pose command latched
    on the rod by anything else would silently lose to it, so :meth:`sync` refuses
    to run over one; tasks command the drive body instead.

    Args:
        num_envs: Number of rods.
        root_stride: Particles per rod in the rod solver's own arrays.
        drive_bodies: Per-env body index holding each rod; all non-negative.
        mount_local: Per-env grip point in the drive body's frame. Defaults to
            the body origin.
        device: Warp device for the persistent buffers.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        root_stride: int,
        drive_bodies,
        mount_local=None,
        device=None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if root_stride <= 0:
            raise ValueError(f"root_stride must be positive, got {root_stride}")

        drive = [int(value) for value in drive_bodies]
        if len(drive) != num_envs:
            raise ValueError(f"drive_bodies has {len(drive)} entries but num_envs is {num_envs}")
        if any(body < 0 for body in drive):
            raise ValueError(
                f"drive_bodies {drive} include a negative index; the root has to follow a body"
            )

        mounts = [(0.0, 0.0, 0.0)] * num_envs if mount_local is None else [tuple(m) for m in mount_local]
        if len(mounts) != num_envs:
            raise ValueError(f"mount_local has {len(mounts)} entries but num_envs is {num_envs}")
        for mount in mounts:
            if len(mount) != 3:
                raise ValueError(f"each mount_local entry must be a 3-vector, got {mount!r}")

        self.num_envs = int(num_envs)
        self.root_stride = int(root_stride)
        self._device = device

        self._drive_bodies = wp.array(drive, dtype=wp.int32, device=device)
        self._mount_local = wp.array(mounts, dtype=wp.vec3, device=device)
        self._bound = wp.zeros(self.num_envs, dtype=wp.int32, device=device)
        self._root_rotation_local = wp.zeros(self.num_envs, dtype=wp.quat, device=device)
        self._root_positions = wp.zeros(self.num_envs, dtype=wp.vec3, device=device)
        self._root_orientations = wp.zeros(self.num_envs, dtype=wp.quat, device=device)
        self._reset_env_ids = wp.empty(self.num_envs, dtype=wp.int32, device=device)
        self._reset_env_ids_host = wp.empty(
            self.num_envs, dtype=wp.int32, device="cpu", pinned=wp.get_device(device).is_cuda,
        )

    @property
    def root_positions(self):
        """Per-env root positions placed by the most recent :meth:`sync`."""
        return self._root_positions

    @property
    def root_orientations(self):
        """Per-env root orientations (xyzw) placed by the most recent :meth:`sync`."""
        return self._root_orientations

    def sync(self, state_in, state_out, rod_solver) -> None:
        """Latch each root onto its mount for the rod step that follows.

        Call after the rigid solve and before the rod solve of the same substep.

        Args:
            state_in: State the rigid solve started from; its body poses anchor
                the root's rotation when an env is first bound.
            state_out: State the rigid solve wrote; the body pose the root follows.
            rod_solver: The :class:`~catheter_vasculature_solver.XPBDRodSolver`
                about to step.

        Raises:
            RuntimeError: If the rod already has a command latched, which this
                would silently override.
        """
        if rod_solver.has_pending_control:
            raise RuntimeError(
                "drive_root_sync holds the catheter's root on its drive body, but a rate or "
                "pose command is already latched on the rod and would be silently overridden. "
                "Command the drive body instead, or turn drive_root_sync off."
            )
        wp.launch(
            _follow_drive_mount_kernel,
            dim=self.num_envs,
            inputs=[
                state_out.body_q,
                state_in.body_q,
                self._drive_bodies,
                self._mount_local,
                rod_solver.orientation_array,
                self.root_stride,
                self._bound,
                self._root_rotation_local,
                self._root_positions,
                self._root_orientations,
            ],
            device=self._device,
        )
        rod_solver.set_root_pose_gpu(self._root_positions, self._root_orientations)

    def reset(self, world_mask=None) -> None:
        """Re-bind the selected worlds' root rotation at their next sync; ``None`` re-binds all."""
        env_ids = env_ids_from_world_mask(world_mask, self.num_envs)
        if env_ids is None:
            self._bound.zero_()
            return
        if len(env_ids) == 0:
            return
        self._reset_env_ids_host.numpy()[:len(env_ids)] = env_ids
        wp.copy(self._reset_env_ids, self._reset_env_ids_host, count=len(env_ids))
        wp.launch(
            _unbind_envs_kernel, dim=len(env_ids), inputs=[self._reset_env_ids, self._bound],
            device=self._device,
        )


__all__ = [
    "DEBUG_ENV_VAR",
    "DEFAULT_DEBUG_INTERVAL",
    "DriveReactionCoupler",
    "DriveRootFollower",
    "apply_coupling_forces_kernel",
    "blend_coupling_forces_kernel",
    "debug_log_interval",
    "harvest_drive_reaction_kernel",
    "resolve_drive_body_index",
    "stash_coupling_forces_kernel",
]
