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
from ownership_helpers import (  # noqa: F401 - re-exported for the interaction and tissue test modules
    MODES,
    ExposedRigid,
    arrays,
    copy_container,
    first_difference,
    snapshot,
)

# Systems whose history C4.1 owns, and the arrays each one keeps across steps.
HISTORY = {
    "Centreline": ("positions", "active_count"),
    "InstrumentContact": ("delta_k", "delta_jaw"),
    "Hook": ("verts", "tet", "bary", "friction_coord", "radial", "active"),
    "Grasp": ("prim", "tet", "t", "radial", "prev_axis", "verts", "bary", "gap", "active", "count", "pending", "delta_k", "delta_jaw"),
    "VolumetricGrasp": ("particle", "sphere", "rest", "active", "count", "pending", "claim", "delta_k", "delta_jaw"),
    "Clipping": ("node", "active", "count", "clipped"),
}


def chole_cfg(disable=(), enable=()):
    cfg = read_case("chole_tet_med")
    cfg["assets"].update({name: {"type": "instrument", **record} for name, record in cfg.get("instruments", {}).items()})
    for spec in cfg["systems"]:
        if spec["type"] in disable:
            spec["enabled"] = False
        if spec["type"] in enable:
            spec["enabled"] = True
    return cfg


def authored(device, envs=2, disable=(), mount=None, enable=()):
    """``mount``: an instrument name to mount two-way on a free rigid body ``holder`` added to every world.
    ``enable``: system types the case ships disabled (its liver-gallbladder ``tissue_contact``)."""
    cfg = chole_cfg(disable, enable)
    a = author_case(cfg, loaders={**physical.LOADERS, **LOADERS}, systems={**physical.SYSTEMS, **SYSTEMS}, device=device, envs=envs,
                    mounts={mount: {"body": "holder"}} if mount else None, coupling_mode="two_way" if mount else "one_way")
    b = newton.ModelBuilder()
    a.declare(b)
    for world in range(envs):
        b.begin_world(gravity=a.scene.gravity_vector)
        a.add_world(b, world)
        if mount:
            b.add_body(label="holder", mass=1.0)
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


# Tissue and fluid systems; the interaction resume fixtures keep them out of the step (test_tissue_owned_storage
# resumes with them).
LATER = ("thermal", "bleeding", "emitter", "suction", "smoke", "fluid", "fluid_contact")


class Episode:
    """A scripted chole episode: tools travel to the tissue, jaws close, the applier fires a clip."""

    def __init__(self, sim):
        self.sim = sim
        scene, self.control, self.clip = sim.scene, system(sim, "InstrumentControl"), system(sim, "Clipping")
        body = scene.body("tissue")
        c = body.centrelines
        q = scene.field("q", body).numpy()[0]
        live = np.flatnonzero(scene.field("flags", body).numpy()[0] & 1)
        self.targets = self.control.actions.numpy().copy()  # on the CPU numpy() aliases the Control
        for k in range(len(self.control.names)):
            self.targets[:, k, :3] = c.rest[0, int(np.flatnonzero(c.line == 1)[0])] if k == self.clip.k else q[live[(k * 997) % len(live)]]
        self.clip.reach.fill_(10.0)  # live parameter: every runtime of the comparison sets it

    def actions(self, closed):
        act = self.targets.copy()
        act[:, :, 7] = float(closed)
        act[:, self.clip.k, 9] = float(closed)
        return act

    def run(self, steps, closed, *others):
        for _ in range(steps):
            for sim in (self.sim, *others):
                system(sim, "InstrumentControl").actions.assign(self.actions(closed))
                sim.step()


def set_hook(sim):
    """Engage the hook on the fat triangle nearest its limb in every world (a valid anchor; the test_instruments
    hook test reaches the same state through an injected contact record)."""
    scene, hook, control = sim.scene, system(sim, "Hook"), system(sim, "InstrumentControl")
    body = hook.body
    tris, tri_part, tri_tet = body.tet.tris[0], body.tet.tri_part, body.tet.tri_tet[0]
    q, spheres = scene.field("q", body).numpy(), control.sphere_q.numpy()
    a, b = int(control.prim_a.numpy()[hook.prim]), int(control.prim_b.numpy()[hook.prim])
    candidates = np.flatnonzero((tri_part == body.parts["fat"].index) & (tri_tet >= 0))
    verts, tet, bary, active = (x.numpy() for x in (hook.verts, hook.tet, hook.bary, hook.active))
    for env in range(scene.envs):
        limb = spheres[env, a] + 0.8 * (spheres[env, b] - spheres[env, a])
        f = int(candidates[np.argmin(np.linalg.norm(q[env][tris[candidates]].mean(axis=1) - limb, axis=1))])
        verts[env, 0], tet[env, 0], bary[env, 0], active[env, 0] = tris[f], tri_tet[f], 1.0 / 3.0, 1
    for dst, host in ((hook.verts, verts), (hook.tet, tet), (hook.bary, bary), (hook.active, active)):
        wp.copy(dst, wp.array(host, dtype=dst.dtype, device=dst.device))
    wp.synchronize()


class Coupled:
    """A chole runtime whose mounted tool is carried by a rigid body through ``CoupledSolver`` (two-way)."""

    def __init__(self, cfg, a, m, state, control, rigid=lambda m: ExposedRigid(m, inverse_mass=1.0e-6)):
        from i4h_physics.coupling import CoupledSolver, Mounts

        self.inner = runtime(cfg, a, m, state=state, control=control)
        self.scene, self.systems, self.state, self.control = self.inner.scene, self.inner.systems, state, control
        self.mounts = Mounts(m, self.scene, self.systems, a.mounts, a.coupling_mode == "two_way")
        self.solver = CoupledSolver(m, rigid(m), self.inner.solver, self.mounts, rigid_substeps=2)
        self.dt = self.inner.dt

    def step(self):
        self.state.body_f.zero_()
        self.solver.step(self.state, self.state, self.control, None, self.dt)
