# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""The chole procedure case as an explicit authoring, for interaction-history ownership tests."""
import newton
import numpy as np
import warp as wp
from i4h_physics import author_case
from i4h_physics import config as physical
from i4h_physics.stepper import Stepper
from i4h_surgical import read_case
from i4h_surgical.config import LOADERS, SYSTEMS

MODES = [("cpu", False), ("cuda:0", False), ("cuda:0", True)]
# Systems whose history C4.1 owns, and the arrays each one keeps across steps.
HISTORY = {
    "Centreline": ("positions", "active_count"),
    "InstrumentContact": ("delta_k", "delta_jaw"),
    "Hook": ("verts", "tet", "bary", "friction_coord", "radial", "active"),
    "Grasp": ("prim", "tet", "t", "radial", "prev_axis", "verts", "bary", "gap", "active", "count", "pending", "delta_k", "delta_jaw"),
    "VolumetricGrasp": ("particle", "sphere", "rest", "active", "count", "pending", "claim", "delta_k", "delta_jaw"),
    "Clipping": ("node", "active", "count", "clipped"),
}


def chole_cfg(disable=()):
    cfg = read_case("chole_tet_med")
    cfg["assets"].update({name: {"type": "instrument", **record} for name, record in cfg.get("instruments", {}).items()})
    for spec in cfg["systems"]:
        if spec["type"] in disable:
            spec["enabled"] = False
    return cfg


def authored(device, envs=2, disable=()):
    cfg = chole_cfg(disable)
    a = author_case(cfg, loaders={**physical.LOADERS, **LOADERS}, systems={**physical.SYSTEMS, **SYSTEMS}, device=device, envs=envs)
    b = newton.ModelBuilder()
    a.declare(b)
    for world in range(envs):
        b.begin_world(gravity=a.scene.gravity_vector)
        a.add_world(b, world)
        b.end_world()
    a.complete(b)
    m = b.finalize(device=device)
    m.particle_grid = None
    return cfg, a, m


def runtime(cfg, a, m, captured=False, state=None, control=None, prepare=True):
    state = m.state() if state is None else state
    control = m.control() if control is None else control
    scene, systems = a.bind(m, state, control)
    sim = Stepper.create(scene, systems, control=control, **{**physical.sim_settings(cfg, False), "cuda_graph": captured})
    if prepare:
        sim.prepare()
    return sim


def system(sim, name):
    return next(s for s in sim.systems if type(s).__name__ == name)


def history(sim, names=HISTORY):
    """Bytes of every interaction-history array, keyed by (system, array)."""
    wp.synchronize()
    return {(name, key): getattr(system(sim, name), key).numpy().tobytes() for name, keys in names.items() for key in keys}


def poison(sim, seed=7):
    """Deterministic non-default bytes in every interaction-history array (no physical meaning)."""
    rng = np.random.default_rng(seed)
    for name, keys in HISTORY.items():
        s = system(sim, name)
        for key in keys:
            a = getattr(s, key)
            host = a.numpy()
            values = rng.integers(1, 5, host.shape).astype(host.dtype) if host.dtype.kind in "iu" else rng.uniform(0.5, 1.5, host.shape).astype(host.dtype)
            wp.copy(a, wp.array(values, dtype=a.dtype, device=a.device))
    wp.synchronize()
