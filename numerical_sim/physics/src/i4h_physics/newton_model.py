# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Author exact per-world particle rows before Newton finalization.

Particle positions remain in each component's mechanics frame. A host's world
transform is recorded separately; this module does not normalize asset units or
author native surgical tet, triangle or bending-edge topology.

Newton owns native rest/live positions, velocities and radii. Custom particle
attributes retain exact inverse-mass/default-flag bytes on Model and mutable
history/flags on State, avoiding reciprocal round trips through native masses.
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
    radius, flags, world; native state q, qd, f; declared exact defaults, mutable
    history/flags and their initialization templates; hash grid; particle BVH. ``construction`` adds the transient peak of
    ``build_model`` and the BVH rebuild (the figure to check before ``finalize``; the steady one is for reporting)."""
    # Empty custom native fields retain one inaccessible backing element.
    declared = max(1, particles)
    out = {
        "newton_particle_defaults": declared * (_F32 + _I32),
        "newton_particle_history": declared * (_VEC3 + _I32),
        "newton_particle_templates": declared * (_VEC3 + _I32),
    }
    if particles == 0:
        return out
    n = particles
    out.update({
        "newton_model": n * (2 * _VEC3 + 3 * _F32 + 2 * _I32),
        "newton_state": n * 3 * _VEC3,
        "newton_bvh": n * BVH_BYTES_PER_PARTICLE,
    })
    if construction:  # finalize allocates the hash grid too; build_model drops it right away
        # build_model releases the grid before model.state() allocates native
        # q/qd/f and declared history/flags. The steady budget already includes
        # those later State buffers, so only the grid's excess can increase the
        # peak. Keep the qualified temporary/BVH allowance for either phase.
        live_state = out["newton_state"] + out["newton_particle_history"]
        out["newton_grid"] = max(0, HASH_GRID_BYTES - live_state)
        out["newton_construction"] = n * CONSTRUCTION_BYTES_PER_PARTICLE
    return out


def _masses(inv_mass: np.ndarray) -> np.ndarray:
    return np.where(inv_mass > 0.0, 1.0 / np.maximum(inv_mass, 1.0e-30), 0.0).astype(np.float32)


def add_env_particles(builder, scene, env: int, xform=None, custom_attributes=None) -> None:
    """Add the main block's ``W`` particles of ``env`` (its own mesh-variant rows) to the builder's current world, in the
    main block's layout order. The per-world form: Isaac Lab's per-world builder hooks call it inside each replicated
    world (a scene with robots), passing the world's ``(position, xyzw quaternion)``; the rows' start and the transform
    are recorded so ``place_main`` finds our rows among the host's."""
    if custom_attributes is None:
        if not hasattr(scene, "declarations"):
            declare_scene_particles(builder, scene)
        custom_attributes = scene.declarations.entity_values(scene.particle_domain, env * scene.main.width, scene.main.width)
    if hasattr(scene, "_storage_worlds"):
        if env in scene._storage_worlds:
            raise ValueError(f"duplicate Scene world: {env}")
        scene._storage_worlds.add(env)
        scene._storage_starts[env] = builder.particle_count
    if xform is not None:
        if scene.world_starts is None:
            scene.world_starts, scene.env_xforms = [None] * scene.envs, [None] * scene.envs
        scene.world_starts[env] = builder.particle_count
        scene.env_xforms[env] = (np.asarray(xform[0], np.float64).copy(), np.asarray(xform[1], np.float64).copy())
    if not scene.main.width:
        return
    q0, inv_mass, radius, flags = (env_rows(scene, f, env) for f in ("q0", "inv_mass", "radius", "flags"))
    builder.add_particles(q0.tolist(), [(0.0, 0.0, 0.0)] * len(q0), _masses(inv_mass).tolist(), radius.tolist(),
                          [int(f) for f in flags], custom_attributes=custom_attributes)


def declare_scene_particles(builder, scene):
    """Scene-only construction uses the same native schema as case authoring."""
    from i4h_physics.scene_storage import declare_scene, register_scene_finalizers
    from i4h_physics.storage import Declarations
    scene.declarations = Declarations()
    declare_scene(scene.declarations, scene)
    from i4h_physics.mount_storage import publish_mounts
    publish_mounts(scene, scene.declarations, builder)
    scene.declarations.register_schema(builder)
    register_scene_finalizers(builder, scene)


def build_builder(scene, builder=None):
    """Author every world's exact variant rows into an unfinalized builder."""
    import newton

    builder = builder if builder is not None else newton.ModelBuilder()
    declare_scene_particles(builder, scene)
    for world in range(scene.envs):
        builder.begin_world(gravity=scene.gravity_vector)
        add_env_particles(builder, scene, world, custom_attributes=scene.declarations.entity_values(
            scene.particle_domain, world * scene.main.width, scene.main.width))
        builder.end_world()
    complete_scene(builder, scene)
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
    """Validate world count and host gravity; never overwrite finalized particle arrays.

    The standalone owner can release the unused particle hash grid. A host retains
    all of its allocation and chooses its gravity before binding.
    """
    if model.world_count != scene.envs:
        raise RuntimeError(f"newton model has {model.world_count} worlds, the scene {scene.envs} envs")
    if host:
        check_host_gravity(model, scene)
    else:
        model.particle_grid = None


def check_host_gravity(model, scene) -> None:
    """The host owns gravity. A declared case gravity is the world's (``I4hComponent.configure`` copies it to the host)
    and must match every world's within 1e-6 (relative to 1 m/s^2 or the magnitude), before any env rotation; a
    rotated component then sees ``R_e^T g`` (``component_gravity``). Without one the scene adopts world 0's in the
    component frame, so ``scene.down`` follows the host."""
    if not scene.gravity_declared:
        scene._gravity = tuple(float(g) for g in component_gravity(model, scene).astype(np.float64)[0])
        return
    world = model.gravity.numpy()[: scene.envs].astype(np.float64)
    case = np.asarray(scene.gravity_vector, np.float64)
    bad = np.abs(world - case).max(axis=1) > 1.0e-6 * max(1.0, float(np.linalg.norm(case)))
    if bad.any():
        raise ValueError(f"the host's gravity {world[int(np.argmax(bad))].tolist()} differs from the case's sim.gravity "
                         f"{case.tolist()} (both world frame); set the host's gravity to the case's, or drop sim.gravity "
                         "from the case to use the host's")


def build_model(scene):
    """One Newton world per env holding the main block's particles, with the exact per-env rest rows (standalone)."""
    model = build_builder(scene).finalize(device=scene.device)
    finish_model(model, scene)
    return model


def rebuild_particle_bvh(model, state) -> None:
    """Build Newton's particle BVH for the selected standalone state."""
    if model.particle_count:
        model.bvh_build_particles(state)


def complete_scene(builder, scene):
    """Complete direct Scene declarations after world authoring, before finalize."""
    from i4h_physics.scene_storage import complete_scene as complete
    complete(builder, scene)
