# SPDX-License-Identifier: Apache-2.0
"""Read-only physics observer. Call sample after EVERY completed physics step."""

import numpy as np
import warp as wp
from i4h_surgical.grasp_geometry import patches
from i4h_surgical.loaders.instrument_usd import CHAIN_FIXED


@wp.kernel
def measure(
    q: wp.array2d(dtype=wp.vec3),
    tris: wp.array(dtype=wp.vec3i),
    ids: wp.array2d(dtype=wp.int32),
    sizes: wp.array(dtype=wp.int32),
    target: wp.array(dtype=wp.int32),
    jaws: wp.array(dtype=wp.int32),
    sq: wp.array2d(dtype=wp.vec3),
    sr: wp.array(dtype=wp.float32),
    sa: wp.array2d(dtype=wp.int32),
    radius: wp.array2d(dtype=wp.float32),
    k: int,
    active: wp.array3d(dtype=wp.int32),
    particle: wp.array3d(dtype=wp.int32),
    sphere: wp.array3d(dtype=wp.int32),
    rest: wp.array3d(dtype=wp.vec3),
    released: wp.array2d(dtype=wp.int32),
    armed: wp.array(dtype=wp.int32),
    latched: wp.array(dtype=wp.int32),
    original_particle: wp.array2d(dtype=wp.int32),
    original_sphere: wp.array2d(dtype=wp.int32),
    original_rest: wp.array2d(dtype=wp.vec3),
    membership: wp.array2d(dtype=wp.int32),
    lost: wp.array2d(dtype=wp.int32),
    displacement: wp.array2d(dtype=wp.vec3),
    residual: wp.array2d(dtype=wp.float32),
    geometry: wp.array2d(dtype=wp.vec3),
    nearest: wp.array(dtype=wp.int32),
    gap: wp.array(dtype=wp.float32),
    counts: wp.array2d(dtype=wp.int32),
    stats: wp.array2d(dtype=wp.float32),
):
    e = wp.tid()
    t = target[e]
    if t < 0:
        return
    pos = q[e, t]
    normal = wp.vec3(0.0)
    for f in range(tris.shape[0]):
        v = tris[f]
        if v[0] == t or v[1] == t or v[2] == t:
            normal += wp.cross(q[e, v[1]] - q[e, v[0]], q[e, v[2]] - q[e, v[0]])
    if wp.length(normal) > 1.0e-12:
        normal = wp.normalize(normal)
    centre = wp.vec3(0.0)
    n = int(0)
    best = int(-1)
    distance = float(1.0e20)
    for j in range(jaws.shape[0]):
        s = jaws[j]
        if sa[e, s] != 0:
            centre += sq[e, s]
            n += 1
            d = wp.length(pos - sq[e, s])
            if d < distance:
                distance = d
                best = s
    geometry[e, 0] = pos
    geometry[e, 1] = normal
    geometry[e, 2] = centre / float(wp.max(n, 1))
    nearest[e] = best
    gap[e] = 1.0e20
    if best >= 0:
        geometry[e, 3] = sq[e, best]
        # Signed radial clearance from the particle+sphere capture envelope.
        gap[e] = wp.length(sq[e, best] - pos) - sr[best] - radius[e, t]
    G = active.shape[2]
    total = int(0)
    for g in range(G):
        total += active[e, k, g]
    if armed[e] != 0 and latched[e] == 0 and total > 0:
        latched[e] = 1
        counts[e, 0] = total
        for g in range(G):
            if active[e, k, g] != 0:
                p = particle[e, k, g]
                original_particle[e, g] = p
                original_sphere[e, g] = sphere[e, k, g]
                original_rest[e, g] = rest[e, k, g]
                inside = int(0)
                for j in range(sizes[t]):
                    if ids[t, j] == p:
                        inside = 1
                membership[e, g] = inside
                counts[e, 1] += inside
                if inside == 0:
                    counts[e, 5] = 1  # wrong patch is permanent
    valid_count = int(0)
    retained = int(0)
    invalid = int(0)
    maximum = float(0.0)
    sum_residual = float(0.0)
    for g in range(G):
        p = original_particle[e, g]
        if p < 0:
            continue
        s = original_sphere[e, g]
        if (
            active[e, k, g] == 0
            or particle[e, k, g] != p
            or sphere[e, k, g] != s
            or wp.length_sq(rest[e, k, g] - original_rest[e, g]) > 0.0
            or sa[e, s] == 0
            or released[e, k] != 0
        ):
            lost[e, g] = 1
        delta = q[e, p] - sq[e, s]
        r = wp.abs(wp.length(delta) - wp.length(original_rest[e, g]))
        displacement[e, g] = delta
        residual[e, g] = r
        if membership[e, g] == 0:
            continue
        maximum = wp.max(maximum, r)
        sum_residual += r
        if lost[e, g] == 0:
            valid_count += 1
            if r <= 0.002:
                retained += 1
        else:
            invalid += 1
    counts[e, 2] = valid_count
    counts[e, 3] = retained
    counts[e, 4] = invalid
    stats[e, 0] = maximum
    stats[e, 1] = sum_residual / float(wp.max(counts[e, 1], 1))


@wp.kernel
def jaw_measure(
    angle: wp.array2d(dtype=wp.float32),
    held: wp.array2d(dtype=wp.int32),
    surface_count: wp.array2d(dtype=wp.int32),
    volume_count: wp.array2d(dtype=wp.int32),
    surface_active: wp.array3d(dtype=wp.int32),
    volume_active: wp.array3d(dtype=wp.int32),
    sphere_radius: wp.array(dtype=wp.float32),
    nearest: wp.array(dtype=wp.int32),
    k: int,
    jaw_state: wp.array2d(dtype=wp.float32),
    mechanism_counts: wp.array2d(dtype=wp.int32),
    nearest_radius: wp.array(dtype=wp.float32),
):
    e = wp.tid()
    jaw_state[e, 0] = angle[e, k]
    jaw_state[e, 1] = float(held[e, k])
    mechanism_counts[e, 0] = surface_count[e, k]
    mechanism_counts[e, 1] = volume_count[e, k]
    ns = int(0)
    nv = int(0)
    for g in range(surface_active.shape[2]):
        ns += surface_active[e, k, g]
    for g in range(volume_active.shape[2]):
        nv += volume_active[e, k, g]
    mechanism_counts[e, 2] = ns
    mechanism_counts[e, 3] = nv
    nearest_radius[e] = 0.0
    if nearest[e] >= 0:
        nearest_radius[e] = sphere_radius[nearest[e]]


class GraspTargetProbe:
    """One named tissue/grasper. Reconstruct after binding invalidation or hard rebuild.

    geometry (E,4,3): target, normal, grasp centre, nearest sphere position.
    counts (E,6): acquisition size, patch size, valid, retained, invalid, wrong patch.
    stats (E,2): maximum/mean radial residual on original patch particles, metres.
    """

    fields = (
        "target",
        "armed",
        "latched",
        "original_particle",
        "original_sphere",
        "original_rest",
        "membership",
        "lost",
        "displacement",
        "residual",
        "geometry",
        "nearest",
        "gap",
        "counts",
        "stats",
        "jaw_state",
        "mechanism_counts",
        "nearest_radius",
    )

    def __init__(self, binding, tissue="tissue", grasper="grasper"):
        self.binding = binding
        scene = binding.scene
        self.body = b = scene.body(tissue)
        systems = binding.systems
        self.volume = next(s for s in systems if s.id == "volumetric_grasp")
        self.surface = next(s for s in systems if s.id == "grasp")
        self.ic = self.volume.instruments
        self.k = self.ic.names.index(grasper)
        if self.volume.body is not b:
            raise ValueError("probe tissue differs from volumetric grasp body")
        if b.tet.tris.shape[0] != 1:
            raise ValueError("v1 probe requires a single mesh variant")
        self.triangles = b.tet.tris[0]
        self.rest = scene.field("q0", b).numpy()[0]
        attached = b.pin.index[0] if b.pin is not None else np.empty(0, np.int32)
        self.edge_length, table, sizes, self.eligible = patches(
            self.rest, self.triangles, attached[attached >= 0]
        )
        self.patch_ids, self.patch_counts = table, sizes
        dev, E, G = scene.device, scene.envs, self.volume.capacity
        self.table = wp.array(table, dtype=wp.int32, device=dev)
        self.sizes = wp.array(sizes, dtype=wp.int32, device=dev)
        self.tris = wp.array(self.triangles, dtype=wp.vec3i, device=dev)
        start = self.ic.host_sphere_start[self.k]
        end = start + self.ic.host_sphere_counts[self.k]
        jaws = np.arange(start, end)[
            self.ic.host_sphere_chain[start:end] == CHAIN_FIXED
        ]
        self.jaws = wp.array(jaws, dtype=wp.int32, device=dev)
        for name in self.fields:
            shape, dtype = (E,), wp.int32
            if name in ("original_particle", "original_sphere", "membership", "lost"):
                shape = (E, G)
            elif name in ("original_rest", "displacement"):
                shape, dtype = (E, G), wp.vec3
            elif name == "residual":
                shape, dtype = (E, G), wp.float32
            elif name == "geometry":
                shape, dtype = (E, 4), wp.vec3
            elif name == "mechanism_counts":
                shape = (E, 4)
            elif name == "jaw_state":
                shape, dtype = (E, 2), wp.float32
            elif name in ("gap", "nearest_radius", "stats"):
                shape, dtype = ((E, 2) if name == "stats" else (E,)), wp.float32
            elif name == "counts":
                shape = (E, 6)
            setattr(self, name, wp.zeros(shape, dtype=dtype, device=dev))
        self.clear()

    def check(self):
        self.binding.check_valid()
        self.binding.solver.check_binding()

    def clear(self, env_ids=None):
        self.check()
        # Reset is outside capture, masked rows only; peer latch and physical state are untouched.
        selected = (
            np.arange(self.binding.scene.envs)
            if env_ids is None
            else np.asarray(env_ids, dtype=int)
        )
        for name in self.fields:
            array = getattr(self, name)
            host = array.numpy()
            host[selected] = (
                -1
                if name in ("target", "nearest", "original_particle", "original_sphere")
                else 0
            )
            array.assign(host)

    def select(self, target, env=0):
        self.check()
        if not 0 <= target < len(self.rest) or not self.eligible[target]:
            raise ValueError("ineligible target")
        if self.latched.numpy()[env]:
            raise RuntimeError("clear the attempt before changing an acquired target")
        ids = self.target.numpy()
        ids[env] = target
        self.target.assign(ids)

    def arm(self, env=0):
        self.check()
        values = self.armed.numpy()
        values[env] = 1
        self.armed.assign(values)

    def sample(self):
        self.check()
        b, v, ic = self.body, self.volume, self.ic
        scene = self.binding.scene
        wp.launch(
            measure,
            dim=scene.envs,
            inputs=[
                scene.field("q", b),
                self.tris,
                self.table,
                self.sizes,
                self.target,
                self.jaws,
                ic.sphere_q,
                ic.sphere_radius,
                ic.sphere_active,
                scene.field("radius", b),
                self.k,
                v.active,
                v.particle,
                v.sphere,
                v.rest,
                ic.released,
            ],
            outputs=[
                self.armed,
                self.latched,
                self.original_particle,
                self.original_sphere,
                self.original_rest,
                self.membership,
                self.lost,
                self.displacement,
                self.residual,
                self.geometry,
                self.nearest,
                self.gap,
                self.counts,
                self.stats,
            ],
            device=scene.device,
        )

        wp.launch(
            jaw_measure,
            dim=scene.envs,
            inputs=[
                ic.jaw_angle,
                ic.held,
                self.surface.count,
                v.count,
                self.surface.active,
                v.active,
                ic.sphere_radius,
                self.nearest,
                self.k,
            ],
            outputs=[self.jaw_state, self.mechanism_counts, self.nearest_radius],
            device=scene.device,
        )

    def read(self):
        self.check()
        return {name: getattr(self, name).numpy() for name in self.fields}

    def torch_views(self):
        """Wait current Torch stream on the producer Warp stream, then return zero-copy views."""
        self.check()
        if self.binding.scene.device.is_cuda:
            import torch

            wp.stream_to_torch(
                wp.get_stream(self.binding.scene.device)
            ).record_event().wait(torch.cuda.current_stream())
        return {name: wp.to_torch(getattr(self, name)) for name in self.fields}
