# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fluid pool loader: ``count`` parked, inactive particles per env that emitters bring to life.

The pool is a capacity. ``emitter`` and ``bleeding`` activate slots (ACTIVE flag, position, velocity),
``suction`` kills them (flag cleared, parked back at ``q0``). Rest state *is* the parked pool, so
``Scene.reset`` alone restores it; the pool lives in the scene's private pool block (``pool=True``), not
in ``scene.q``; ``inv_mass`` is constant, the ACTIVE bit is the one alive flag.
Physical parameters are asset properties like tet materials (omnisurg ``fluid.pbf`` ``particles`` and
``solver`` blocks, fluids/systems.py:764-996 defaults); the ``fluid`` system reads them from ``Body.fluid``.
The body does not integrate with ``particles`` (``Body.integrate`` False): the fluid system runs the same
kernels with its own damping and velocity bound.
"""

from __future__ import annotations

import numpy as np
from i4h_physics.scene import Body, FluidRest, Part, ParticleRest, Scene


def add_fluid(scene: Scene, name: str, count: int, mass: float = 0.02, radius: float = 0.005,
              interaction_radius: float = 0.01, rest_distance: float | None = None, iterations: int = 3,
              max_neighbors: int = 64, density_epsilon: float = 1.0e-6, relaxation: float = 1.0,
              viscosity: float = 0.0, cohesion: float = 0.0, visual=None) -> Body:
    """``radius`` is the collision radius (contact shell); ``interaction_radius`` the smoothing length;
    ``rest_distance`` defaults to ``0.6 * interaction_radius``; the pool is parked 1 km along gravity."""
    n = int(count)
    if n < 1:
        raise ValueError(f"fluid {name!r}: count must be at least 1")
    if float(mass) <= 0.0 or float(radius) <= 0.0 or float(interaction_radius) <= 0.0:
        raise ValueError(f"fluid {name!r}: mass, radius and interaction_radius must be positive")
    h = float(interaction_radius)
    rest_distance = 0.6 * h if rest_distance is None else float(rest_distance)
    if not 0.0 < rest_distance <= h:
        raise ValueError(f"fluid {name!r}: rest_distance must be in (0, interaction_radius]")
    if int(iterations) < 1 or int(max_neighbors) < 1:
        raise ValueError(f"fluid {name!r}: iterations and max_neighbors must be at least 1")
    if not 0.0 <= float(relaxation) <= 2.0:
        raise ValueError(f"fluid {name!r}: relaxation must be in [0, 2]")
    park = 1000.0 * scene.down
    E = scene.envs
    rest = ParticleRest(
        q0=np.tile(np.asarray(park, np.float32)[None, None], (E, n, 1)),
        inv_mass=np.full((E, n), 1.0 / float(mass), np.float32),
        radius=np.full((E, n), float(radius), np.float32),
        flags=np.zeros((E, n), np.int32),
    )
    visual = dict(visual or {})  # read from FluidRest by the renderers (pool bodies are not drawn as parts)
    body = scene.add_body(name, rest, {name: Part(0, 0, n)}, integrate=False, pool=True)
    body.fluid = FluidRest(h, rest_distance, int(iterations), int(max_neighbors), float(density_epsilon),
                           float(relaxation), float(viscosity), float(cohesion))
    return body
