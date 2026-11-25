# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tissue, fluid and containment history: the arrays C4.2 owns and the two cases that hold them."""
import newton
import numpy as np
import warp as wp
from i4h_physics import author_case
from i4h_physics import config as physical
from interaction_helpers import Episode, system

# Arrays kept across steps, by system. ``Surface`` is the surgical surface; its colour lives on ``appearance``.
TISSUE = {
    "Thermal": ("heat", "burn", "coag"),
    "Surface": ("appearance.visual",),
    "Fluid": ("cursor", "overflow"),
    "Bleeding": ("cut", "source", "acc", "serial"),
    "Emitter": ("acc", "serial"),
    "Smoke": ("pos_age", "vel_life", "cursor", "acc", "serial"),
    "Suction": ("removed",),
}
CONTAINMENT = {"GraphContainment": ("cand_idx", "cand_count", "cand_overflow", "ref_samples", "ref_tree", "rebuild", "built")}
# What an episode reset writes into the selected worlds (everything not listed becomes zero).
RESET = {("GraphContainment", "cand_idx"): -1, ("GraphContainment", "rebuild"): 1}


def field(sim, name, key):
    value = system(sim, name)
    for part in key.split("."):
        value = getattr(value, part)
    return value


def history(sim, names):
    wp.synchronize()
    return {(name, key): field(sim, name, key).numpy().tobytes() for name, keys in names.items() for key in keys}


def poison(sim, names, seed=11):
    """Deterministic non-default bytes in every listed array (no physical meaning)."""
    rng = np.random.default_rng(seed)
    for name, keys in names.items():
        for key in keys:
            a = field(sim, name, key)
            host = a.numpy()
            values = rng.integers(2, 6, host.shape).astype(host.dtype) if host.dtype.kind in "iu" else rng.uniform(0.5, 1.5, host.shape).astype(host.dtype)
            wp.copy(a, wp.array(values, dtype=a.dtype, device=a.device))
    wp.synchronize()


def world_rows(sim, name, key, env):
    """Every component of one world's rows (NumPy exposes vector dtypes as a trailing axis)."""
    return field(sim, name, key).numpy().reshape(sim.scene.envs, -1)[env]


def aorta(device, envs=2):
    """The aorta_centreline case (catheter in a centreline tree, graph containment) as an explicit authoring."""
    from i4h_endoluminal.config import LOADERS, SYSTEMS, read_case
    cfg = read_case("aorta_centreline")
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


class TissueEpisode(Episode):
    """The scripted chole episode with the hook sent to the nearest fat and energised while the jaws are closed.
    Closed jaws also start irrigation, cut tissue and paint contacts."""

    def __init__(self, sim):
        super().__init__(sim)
        scene, body = sim.scene, sim.scene.body("tissue")
        self.hook = self.control.names.index("hook")
        tris, tri_part, tri_tet = body.tet.tris[0], body.tet.tri_part, body.tet.tri_tet[0]
        fat = np.flatnonzero((tri_part == body.parts["fat"].index) & (tri_tet >= 0))
        centres = scene.field("q", body).numpy()[0][tris[fat]].mean(axis=1)
        start = self.control.actions.numpy()[0, self.hook, :3]
        self.targets[:, self.hook, :3] = centres[np.argmin(np.linalg.norm(centres - start, axis=1))]

    def actions(self, closed):
        act = super().actions(closed)
        act[:, self.hook, 8] = float(closed)
        return act


def cut_artery(sim):
    """Mark one node of the bleeding vessel as cut in every world (the state a cutting tool leaves at step end)."""
    bleeding = system(sim, "Bleeding")
    cut = bleeding.cut.numpy()
    cut[:, bleeding.node_lo + 1] = 1
    wp.copy(bleeding.cut, wp.array(cut, dtype=wp.int32, device=bleeding.cut.device))
    wp.synchronize()


def write(array, host):
    wp.copy(array, wp.array(host, dtype=array.dtype, device=array.device))


def inject(sim):
    """Valid history the scripted episode does not reach by stepping, written into the bound State: hot fat tets
    (between the burn and the removal thresholds), live smoke particles, a cut artery node, and non-zero values in
    the counters only suction, neighbour overflow and smoke spawning advance."""
    scene, thermal, smoke = sim.scene, system(sim, "Thermal"), system(sim, "Smoke")
    body, E = thermal.body, sim.scene.envs
    heat, active = thermal.heat.numpy(), thermal.surface.tet_fields.active.numpy()
    for env in range(E):
        fat = np.flatnonzero((body.tet.part == body.parts["fat"].index) & (active[env] != 0))
        heat[env, fat[:: max(1, len(fat) // 8)][:8]] = 1.5
    write(thermal.heat, heat)
    q = scene.field("q", body).numpy()
    pos_age, vel_life = (x.numpy().reshape(E, smoke.count, 4) for x in (smoke.pos_age, smoke.vel_life))
    pos_age[:, :4, :3], pos_age[:, :4, 3] = q[:, :4], 0.01
    vel_life[:, :4] = (0.0, 0.05, 0.0, 0.5)
    write(smoke.pos_age, pos_age.reshape(-1, 4))
    write(smoke.vel_life, vel_life.reshape(-1, 4))
    for array, value in ((smoke.cursor, 4), (system(sim, "Suction").removed, 3), (system(sim, "Fluid").overflow, 1)):
        array.fill_(value)
    cut_artery(sim)
