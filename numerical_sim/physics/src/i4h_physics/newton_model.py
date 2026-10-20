# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Newton model behind the scene's main particle block.

``build_model(scene)`` builds one Newton world per env from the loaders' host records: one template world with the
main block's ``W`` particles (``ModelBuilder.add_particles``), ``replicate`` over the envs, ``finalize``; then the
exact per-env rows are assigned (mesh variants differ per env): ``particle_q`` (the rest positions), a consistent
``particle_mass`` / ``particle_inv_mass`` pair (0 = pinned or padding), ``particle_radius`` and its true maximum,
``particle_flags`` (i4h uses bit 0 alone, which is Newton's ``ParticleFlags.ACTIVE``), and per-world gravity. The
particle BVH ``finalize`` built from the template is rebuilt once the state exists (``rebuild_particle_bvh``), so no
Newton structure describes template values. Newton's particle order is world-grouped, the main block's layout, so
``Scene.allocate`` aliases these arrays instead of copying them.

Measured (chole_tet_med, 64 envs, 359k particles): template 0.15 s, replicate 0.06 s, finalize 0.15 s; the model and
state particle arrays with the particle BVH hold 71 MB, 129 MB at the construction peak. ``finalize`` also allocates
a 128^3 particle hash grid (~17 MB) that nothing here queries: ``build_model`` drops it at once, but it is part of the
peak together with the finalize temporaries. Per particle that is about 196 bytes steady and up to ~340 at the peak
plus the grid (two_worlds, aorta and chole at 16, 64 and 128 envs): ``estimate_bytes`` budgets 200 steady, and the
grid plus 150 more per particle during construction; ``Scene.allocate`` checks the construction figure before
``finalize``.
"""

from __future__ import annotations

import numpy as np
from i4h_physics.scene import _F32, _I32, _VEC3

HASH_GRID_BYTES = 2 * 128**3 * 4  # Newton's model.particle_grid (a 128^3 wp.HashGrid) whenever there are particles


def main_rows(scene, field: str) -> np.ndarray:
    """``(E, W, ...)`` rows of a ``ParticleRest`` field over the main block's bodies, in layout order."""
    rows = [getattr(b.rest, field) for b in scene.bodies]
    E = scene.envs
    if not rows:
        return np.zeros((E, 0, 3) if field == "q0" else (E, 0), np.float32 if field != "flags" else np.int32)
    return np.concatenate(rows, axis=1)


def env_rows(scene, field: str, env: int) -> np.ndarray:
    """``env``'s ``(W, ...)`` rows of a ``ParticleRest`` field, ``main_rows(scene, field)[env]`` without copying the
    other envs' rows (each body's rest rows are already the env's own mesh variant)."""
    rows = [getattr(b.rest, field)[env] for b in scene.bodies]
    if not rows:
        return np.zeros((0, 3) if field == "q0" else (0,), np.float32 if field != "flags" else np.int32)
    return np.concatenate(rows, axis=0)


BVH_BYTES_PER_PARTICLE = 120  # lowers, uppers, groups and nodes of the particle BVH (measured ~116)
CONSTRUCTION_BYTES_PER_PARTICLE = 150  # finalize temporaries and the template BVH during the rebuild (~140 measured)


def estimate_bytes(particles: int, construction: bool = False) -> dict[str, int]:
    """Device bytes of the Newton model and state for ``particles`` main-block particles: model q, qd, mass, inv_mass,
    radius, flags, world; state q, qd, f; hash grid; particle BVH. ``construction`` adds the transient peak of
    ``build_model`` and the BVH rebuild (the figure to check before ``finalize``; the steady one is for reporting)."""
    if particles == 0:
        return {}
    n = particles
    out = {
        "newton_model": n * (2 * _VEC3 + 3 * _F32 + 2 * _I32),
        "newton_state": n * 3 * _VEC3,
        "newton_bvh": n * BVH_BYTES_PER_PARTICLE,
    }
    if construction:  # finalize allocates the hash grid too; build_model drops it right away
        out["newton_grid"] = HASH_GRID_BYTES
        out["newton_construction"] = n * CONSTRUCTION_BYTES_PER_PARTICLE
    return out


def _masses(inv_mass: np.ndarray) -> np.ndarray:
    return np.where(inv_mass > 0.0, 1.0 / np.maximum(inv_mass, 1.0e-30), 0.0).astype(np.float32)


def add_env_particles(builder, scene, env: int, xform=None) -> None:
    """Add the main block's ``W`` particles of ``env`` (its own mesh-variant rows) to the builder's current world, in the
    main block's layout order. The per-world form: Isaac Lab's per-world builder hooks call it inside each replicated
    world (a scene with robots), passing the world's ``(position, xyzw quaternion)``; the rows' start and the transform
    are recorded so ``place_main`` finds our rows among the host's. ``build_builder`` uses one template world instead."""
    if xform is not None:
        if scene.world_starts is None:
            scene.world_starts, scene.env_xforms = [None] * scene.envs, [None] * scene.envs
        scene.world_starts[env] = builder.particle_count
        scene.env_xforms[env] = (np.asarray(xform[0], np.float64).copy(), np.asarray(xform[1], np.float64).copy())
    if not scene.main.width:
        return
    q0, inv_mass, radius, flags = (env_rows(scene, f, env) for f in ("q0", "inv_mass", "radius", "flags"))
    builder.add_particles(q0.tolist(), [(0.0, 0.0, 0.0)] * len(q0), _masses(inv_mass).tolist(), radius.tolist(),
                          [int(f) for f in flags])


def build_builder(scene, builder=None):
    """The unfinalized builder of the main block: env 0's particles as a template world, replicated over the envs (the
    fast path: ``replicate`` tiles with numpy). ``builder`` is the owner's (Isaac Lab's ``create_builder()`` applies its
    solver attributes and defaults), else a plain one. ``finish_model`` writes every env's own rows after finalize."""
    import newton

    template = newton.ModelBuilder()
    add_env_particles(template, scene, 0)
    builder = builder if builder is not None else newton.ModelBuilder()
    builder.replicate(template, scene.envs)
    return builder


def place_main(model, scene) -> None:
    """Find the main block's rows in a host's finalized model: every world must hold ours at the same column of
    equally sized worlds (``model.particle_world_start``), giving the block's ``(base, stride, col)``. Without recorded
    starts the model must hold our particles only."""
    E, W, main = scene.envs, scene.main.width, scene.main
    if not W:
        return
    if scene.world_starts is None or any(s is None for s in scene.world_starts):
        if model.particle_count != E * W:
            raise RuntimeError(f"newton model has {model.particle_count} particles, the main block {E * W}: add the "
                               "block through add_env_particles(..., xform=...) so its rows are found among the host's")
        main.placement = None
        return
    pws = np.asarray(model.particle_world_start.numpy(), np.int64)
    starts = np.asarray(scene.world_starts, np.int64)
    stride, col = np.diff(pws[: E + 1]), starts - pws[:E]
    if (stride != stride[0]).any() or (col != col[0]).any() or col[0] < 0 or col[0] + W > stride[0]:
        raise RuntimeError(f"the main block's rows must sit at one column of equally sized worlds; world starts "
                           f"{pws[:E + 1].tolist()}, ours {starts.tolist()} ({W} each)")
    base, S, c = int(pws[0]), int(stride[0]), int(col[0])
    main.placement = None if (base, S, c) == (0, W, 0) else (base, S, c)


def component_gravity(model, scene) -> np.ndarray:
    """``(E, 3)``: each world's gravity in the component frame, ``R_e^T g_e`` (``R_e`` the world's orientation)."""
    from i4h_physics.quat import quat_to_matrix

    g = model.gravity.numpy()[: scene.envs].astype(np.float64)
    if scene.env_xforms is None:
        return g.astype(np.float32)
    rot = [quat_to_matrix(np.asarray(q, np.float64)) for _, q in scene.env_xforms]
    return np.stack([r.T @ v for r, v in zip(rot, g)]).astype(np.float32)


def finish_model(model, scene, state=None, *, host=False) -> None:
    """After any finalize: every env's exact rows (mesh variants differ per env: positions, a consistent mass/inv_mass
    pair, radius and its maximum, flags), per-world gravity, and no hash grid. A ``state`` created before this (Isaac
    Lab's ``state_0``) gets the rest positions and zero velocities too.

    Gravity: a model built here (standalone) gets the case's. A ``host`` model keeps the host's: a case that declares
    ``sim.gravity`` must agree with it (raises otherwise, like ``check_dt``); one that does not adopts it."""
    E, W = scene.envs, scene.main.width
    if model.world_count != E:
        raise RuntimeError(f"newton model has {model.world_count} worlds, the scene {E} envs")
    base, S, col = scene.main.place(E)
    alone = model.particle_count == E * W  # no host particles share the arrays
    if alone:
        model.particle_grid = None  # finalize's 128^3 hash grid (~17 MB): no i4h system or renderer queries it
    if host:
        check_host_gravity(model, scene)
    else:
        model.set_gravity(tuple(float(g) for g in scene.gravity_vector))
    if W:
        q0, inv_mass, radius, flags = (main_rows(scene, f) for f in ("q0", "inv_mass", "radius", "flags"))
        rows = (base + np.arange(E)[:, None] * S + col + np.arange(W)[None, :]).reshape(-1)  # our flat indices

        def put(arr, values):  # only our rows: host particles keep theirs
            full = arr.numpy()
            full[rows] = values.reshape((rows.size,) + full.shape[1:]).astype(full.dtype)
            arr.assign(full)

        put(model.particle_q, q0)
        put(model.particle_mass, _masses(inv_mass))
        put(model.particle_inv_mass, inv_mass)
        put(model.particle_radius, radius)
        put(model.particle_flags, flags)
        ours = float(radius.max())
        model.particle_max_radius = ours if alone else max(ours, float(model.particle_max_radius))
        if state is not None:
            put(state.particle_q, q0)
            put(state.particle_qd, np.zeros_like(q0))


def check_host_gravity(model, scene) -> None:
    """The host owns gravity. A declared case gravity must match every env's world within 1e-6 (relative to 1 m/s^2
    or the magnitude); without one the scene adopts world 0's, so ``scene.down`` follows the host."""
    host = component_gravity(model, scene).astype(np.float64)
    if not scene.gravity_declared:
        scene._gravity = tuple(float(g) for g in host[0])
        return
    case = np.asarray(scene.gravity_vector, np.float64)
    if np.abs(host - case).max() > 1.0e-6 * max(1.0, float(np.linalg.norm(case))):
        raise ValueError(f"the host's gravity {host[0].tolist()} differs from the case's sim.gravity {case.tolist()}; "
                         "set the host's gravity to the case's, or drop sim.gravity from the case to use the host's")


def build_model(scene):
    """One Newton world per env holding the main block's particles, with the exact per-env rest rows (standalone)."""
    model = build_builder(scene).finalize(device=scene.device)
    finish_model(model, scene)
    return model


def rebuild_particle_bvh(model, state) -> None:
    """Rebuild Newton's particle BVH from the assigned rows (``finalize`` built it from the template)."""
    if model.particle_count:
        model.bvh_build_particles(state)
