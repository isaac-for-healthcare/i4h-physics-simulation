# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scene: plain Warp arrays with (env, local) body ranges, plus host records on bodies.

The main particle block is world-major, Newton's layout: env ``e`` owns the row
``[e * W, (e + 1) * W)`` holding every body one after another (``W`` is the sum of the
body capacities), and a body at ``offset`` with ``count`` particles is the strided
``(E, count)`` view ``arr.reshape((E, W))[:, offset : offset + count]``. Kernels never see
this: they take ``scene.field(name, body)`` views and index them ``[env, local]``, and
code that must build flat index tables uses ``scene.layout(body)``. ``count`` is a
capacity: within a body, particles are laid out part-major with a per-part capacity, and
an env whose mesh variant is smaller marks its spare slots inactive (flags 0, inverse
mass 0).

Loaders attach host records to a ``Body``. Immutable mesh data (topology, rest geometry,
initial active masks, connectors, pins) is stored once per mesh variant, ``(V, ...)``,
and every env points at a variant through ``Body.variant_of_env``; independent per-env
geometry is simply ``V == E``. Mutable state (positions, active masks) and the particle
rest rows (``ParticleRest``) are per env, ``(E, ...)``. ``Scene.allocate()`` binds authored particle storage; the system that reads any
other record uses it according to its declared ownership contract.

``Scene.allocate()`` builds the Newton model behind the main block (one world per env,
``newton_model.build_model``) and its state. Native State q/qd and Model radius remain
aliases. q0 aliases native Model particle_q; exact inv_mass/flags0 are Model
custom particle attributes; q_prev and
mutable flags are State custom particle attributes. All are authored before
finalization, including history defaults. Binding never initializes history or
mutates native Model flags.

Particles live in two blocks with the same fields (``Block.FIELDS``): the main block
(``scene.main``; ``scene.q`` and friends are its arrays) holds tissue, shell, rod and
tree bodies, and the private pool block (``scene.pool``) the fluid pools (``add_body``
with ``pool=True``), which stays body-major: a pool's envs follow one another from
``Body.offset`` (the fluid system's span arithmetic relies on it). ``Body.offset`` is the
offset in the body's own block's layout.
``scene.field(name, body)`` views a body's live array in its block; ``scene.view(arr,
body)`` applies the body's layout to the array given (a live array or a snapshot copy)
and rejects an array of the other block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp
from i4h_physics.checks import check_int

ACTIVE_INT = 1  # same bit as newton.ParticleFlags.ACTIVE, so ported omnisurg kernels keep their test
ACTIVE = wp.constant(wp.int32(ACTIVE_INT))
INACTIVE = wp.constant(wp.int32(~ACTIVE_INT))  # ``flags & INACTIVE`` clears the bit

VEC3_BYTES = 12
F32_BYTES = 4
I32_BYTES = 4


@dataclass(eq=False)
class Part:
    """Body-local capacity ranges of one part. Shared by all envs."""

    index: int
    particle_start: int
    particle_count: int
    tri_start: int = 0
    tri_count: int = 0


@dataclass(eq=False, repr=False)
class ParticleRest:
    """Per-env particle rest state of one body, one row per env."""

    q0: np.ndarray  # (E, n, 3) float32
    inv_mass: np.ndarray  # (E, n) float32
    radius: np.ndarray  # (E, n) float32
    flags: np.ndarray  # (E, n) int32


@dataclass(eq=False, repr=False)
class TetMesh:
    """Per-variant tet topology with body-local particle indices; spare slots have active0 0.

    Surface triangles are part-major with capacity: ``[tri_start, tri_start + authored)`` holds the
    authored skin of the largest variant, ``[cut_start, cut_start + cut_capacity)`` is reserved for
    faces exposed by cutting (``deletable`` parts only); unused slots are degenerate ``(a, a, a)``.
    Face tables use the outward corner order ``((1,2,3), (0,3,2), (0,1,3), (0,2,1))`` and edge
    tables the corner pairs ``((0,1), (0,2), (0,3), (1,2), (1,3), (2,3))``
    (omnisurg/tet/bundle/centrelines.py:34-47)."""

    tets: np.ndarray  # (V, T, 4) int32
    active0: np.ndarray  # (V, T) int32 initial masks; the per-env live mask lives on the tet system
    rest_volume: np.ndarray  # (V, T) float32
    part: np.ndarray  # (T,) int32
    edges: np.ndarray  # (V, S, 2) int32
    edge_active0: np.ndarray  # (V, S) int32
    edge_rest: np.ndarray  # (V, S) float32
    edge_part: np.ndarray  # (S,) int32
    spring: np.ndarray  # (P,) float32 per-part stiffness
    spring_damping: np.ndarray  # (P,)
    volume: np.ndarray  # (P,)
    volume_damping: np.ndarray  # (P,)
    tris: np.ndarray  # (V, F, 3) int32 surface triangles, body-local, part-major with capacity
    tri_part: np.ndarray  # (F,) int32
    uvs: np.ndarray  # (V, count, 2) float32, zeros when a variant has none
    heat: np.ndarray  # (P,) int32, 1 where the part conducts heat and burns
    deletable: np.ndarray  # (P,) int32, 1 where tets may be removed (burn, cut, stretch)
    cut_start: np.ndarray  # (P,) int32 first triangle slot reserved for exposed faces
    cut_capacity: np.ndarray  # (P,) int32 number of those slots (0 unless deletable)
    tet_edges: np.ndarray  # (V, T, 6) int32 edge id of each corner pair, -1 when the edge list lacks it
    face_neighbor: np.ndarray  # (V, T, 4) int32 tet across each face, -1 on the boundary
    face_tri: np.ndarray  # (V, T, 4) int32 authored triangle slot matching the face, -1
    tri_tet: np.ndarray  # (V, F) int32 tet owning each authored triangle, -1 for padding and unmatched
    support0: np.ndarray  # (V, count) int32 live tets incident to each particle
    edge_support0: np.ndarray  # (V, S) int32 live tets incident to each edge


@dataclass(eq=False, repr=False)
class RodMesh:
    """Per-variant rest data of a Cosserat rod of ``P`` nodes laid out root first (omniendo builder.py:136-179).

    ``frames`` are the node material frames (local z along the centreline tangent, x/y parallel
    transported); the direct backend uses all ``P``, the PBD backend the first ``P - 1`` as segment
    frames. ``rest_darboux`` is ``Im(conj(q_i) q_{i+1})`` per edge. ``inv_mass_q`` is 0 at a locked root.
    ``start`` / ``direction`` define the insertion track (``catheter_control``). The rod systems accept a
    single variant (``V == 1``): a shorter padded variant would need live node counts in the sweep."""

    frames: np.ndarray  # (V, P, 4) float32 xyzw
    rest_len: np.ndarray  # (V, P-1) float32
    rest_darboux: np.ndarray  # (V, P-1, 3) float32
    inv_mass_q: np.ndarray  # (V, P) float32
    start: np.ndarray  # (V, 3) float32 root position at rest (the insertion track starts here)
    direction: np.ndarray  # (V, 3) float32 unit tangent at rest (the track direction)


@dataclass(eq=False, repr=False)
class RodGraph:
    """Per-variant rest data of an anatomical centreline graph (omniendo ``vessels[].centerline`` with dynamics):
    ``P`` nodes, ``S`` segments with one material frame each, ``K`` bend pairs from a BFS spanning tree
    (pbd_cosserat.py ``derive_bend_pairs``; a graph with cycles keeps its extra segments as stretch constraints),
    node radii for the lumen, the cardinal-spline topology of the render paths, and the root (the catheter's
    entry). Pinned nodes and segments have zero inverse mass in the ``ParticleRest`` and ``inv_mass_q``.
    Consumed by ``rod_pbd`` (dynamics), ``lumen_containment``, the skinning and the control layouts. One variant."""

    positions: np.ndarray  # (V, P, 3) float32 rest positions, scene coordinates
    edges: np.ndarray  # (V, S, 2) int32 node pairs
    bend_pairs: np.ndarray  # (V, K, 2) int32 segment pairs
    frames: np.ndarray  # (V, S, 4) float32 xyzw segment frames (local z along the segment)
    rest_darboux: np.ndarray  # (V, K, 4) float32 xyzw Im(conj(q_a) q_b), w >= 0
    rest_len: np.ndarray  # (V, S) float32
    inv_mass_q: np.ndarray  # (V, S) float32, 0 at pinned segments
    radii: np.ndarray  # (V, P) float32 lumen radii (collision radius scale applied)
    spline_controls: np.ndarray  # (V, S, 4) int32 (prev, start, end, next) nodes of each segment's render path
    spline_neighbors: np.ndarray  # (V, S, 2) int32 previous / next segment on the path, -1 at a path end
    root_arc: np.ndarray  # (V, P) float32 geodesic arc length from the root
    root: int  # node nearest the catheter's entry
    root_neighbor: int  # the root's single neighbour when the root is open (degree 1), else -1
    render_paths: list  # node index lists between non-degree-2 nodes
    material_radius: np.ndarray | None = None  # node radii before collision-radius scaling




@dataclass(eq=False, repr=False)
class ShellMesh:
    """Per-variant rest data of a deformable shell (cloth) body: a vessel wall (omniendo ``dynamics.mode:
    deformable``). Unique triangle edges carry distance constraints, interior edges dihedral bending
    constraints between their two triangles (Newton ``add_cloth_mesh`` topology). Pinned vertices have zero
    inverse mass in the body's ``ParticleRest``. One variant (``V == 1``)."""

    tris: np.ndarray  # (V, F, 3) int32 body-local, winding as authored
    edges: np.ndarray  # (V, S, 2) int32 unique undirected edges
    edge_rest: np.ndarray  # (V, S) float32
    dihedrals: np.ndarray  # (V, D, 4) int32 (opposite0, opposite1, edge_v0, edge_v1), interior edges only
    dihedral_rest: np.ndarray  # (V, D) float32 signed rest angle
    dihedral_edge_len: np.ndarray  # (V, D) float32 |x_l - x_k| at rest (scales the bending compliance)
    uvs: np.ndarray  # (V, n, 2) float32, zeros unless authored


@dataclass(eq=False, repr=False)
class FluidRest:
    """Per-body position-based-fluid parameters (omnisurg fluid.pbf ``particles`` and ``solver`` blocks), asset
    properties like tet materials, read by the ``fluid`` system at construction. Radii and ``iterations``
    are baked into launches; ``relaxation``, ``viscosity`` and ``cohesion`` seed its live ``(E, nf)``
    arrays. ``visual`` (``color``, ``render_radius``) is read by the renderer only."""

    interaction_radius: float  # smoothing length h, also the neighbour query radius
    rest_distance: float  # particle spacing at rest density
    iterations: int  # density solves per substep
    max_neighbors: int  # neighbour list capacity per particle
    density_epsilon: float
    relaxation: float
    viscosity: float
    cohesion: float


@dataclass(eq=False, repr=False)
class StaticMesh:
    """A textured or flat-coloured, non-simulated mesh (the cavity, a vessel). Read by the renderer and by
    the static contact and rod containment systems."""

    name: str
    points: np.ndarray  # (n, 3) float32, world space
    indices: np.ndarray  # (f, 3) int32, winding as authored
    uvs: np.ndarray | None  # (n, 2) float32
    normals: np.ndarray  # (n, 3) float32, smooth over welded positions
    tangents: np.ndarray | None  # (n, 4) float32, xyz + handedness








@dataclass(eq=False, repr=False)
class ConnectorSet:
    """Per-variant tri-point connectors; ``particle`` is -1 for padding slots."""

    particle: np.ndarray  # (V, C) int32
    tri: np.ndarray  # (V, C, 3) int32
    bar: np.ndarray  # (V, C, 3) float32
    rest: np.ndarray  # (V, C) float32
    group: np.ndarray  # (C,) int32
    stiffness: np.ndarray  # (G,) float32
    damping: np.ndarray  # (G,)
    weight_ratio: np.ndarray  # (G,)


@dataclass(eq=False, repr=False)
class PinSet:
    """Per-variant particles pulled toward a fixed rest position; ``index`` is -1 for padding."""

    index: np.ndarray  # (V, A) int32
    rest: np.ndarray  # (V, A, 3) float32




class Body:
    def __init__(self, name: str, offset: int, rest: ParticleRest, parts=None, variant_of_env=None, variants=1,
                 integrate=True, pool=False):
        self.name = name
        self.offset = offset  # in the body's block: the column in a main-block row, the first slot in the pool block
        self.pool = bool(pool)  # True: a fluid pool, stored in the private pool block
        self.count = rest.q0.shape[1]
        self.rest = rest
        self.integrate = bool(integrate)  # False: the body's own system predicts and finalizes it (fluid pools)
        self.parts: dict[str, Part] = dict(parts or {})
        envs = rest.q0.shape[0]
        self.variants = int(variants)  # number of mesh variants loaded, whether or not every env uses one
        self.variant_of_env = (np.zeros(envs, np.int32) if variant_of_env is None
                               else np.ascontiguousarray(variant_of_env, dtype=np.int32))
        if self.variant_of_env.shape != (envs,):
            raise ValueError(f"body {name!r}: variant_of_env must have shape ({envs},)")
        if envs and not (0 <= self.variant_of_env.min() and self.variant_of_env.max() < self.variants):
            raise ValueError(f"body {name!r}: variant_of_env must be in [0, {self.variants})")
        self.variant: wp.array | None = None  # (E,) int32 on the device after Scene.allocate(); shared by systems
        self.tet: TetMesh | None = None
        self.connectors: ConnectorSet | None = None
        self.pin: PinSet | None = None
        self.regions = None  # immutable material edge membership from the asset loader
        self.region_properties = None  # persistent tables allocated by the material system
        self.rod_material = None  # immutable inputs and derived regional products
        self.rod_controls = None  # transient layout/steering channels
        self.rod: RodMesh | None = None
        self.tree: RodGraph | None = None
        self.shell: ShellMesh | None = None
        self.fluid: FluidRest | None = None

    def __repr__(self) -> str:
        return (f"Body({self.name!r}, offset={self.offset}, count={self.count}, parts={list(self.parts)}, "
                f"variants={self.variants}{', pool' if self.pool else ''})")


class Block:
    """One set of per-particle arrays and the bodies stored in it (``Scene.main`` or ``Scene.pool``). A world-major
    block stores env rows of width ``width`` (every body side by side); otherwise each body's envs follow one another."""

    FIELDS = ("q", "qd", "q_prev", "q0", "inv_mass", "radius", "flags", "flags0")

    def __init__(self, name: str, world_major: bool):
        self.name = name
        self.world_major = bool(world_major)
        self.bodies: list[Body] = []
        self.placement: tuple[int, int, int] | None = None  # (base, stride, col) inside a host model; None: alone
        self.size: int | None = None  # length of the block's arrays (a host model's particle count, else count(E))
        for field in self.FIELDS:
            setattr(self, field, None)

    @property
    def width(self) -> int:
        return sum(b.count for b in self.bodies)

    def count(self, envs: int) -> int:
        return envs * self.width

    def next_offset(self, envs: int) -> int:
        return self.width if self.world_major else self.count(envs)

    def place(self, envs: int) -> tuple[int, int, int]:
        """``(base, stride, col)``: env ``e``'s rows start at ``base + e * stride + col`` of the block's arrays. Alone
        (standalone, or a host with no particles of its own) that is ``(0, width, 0)``."""
        return self.placement if self.placement is not None else (0, self.width, 0)

    def rows(self, arr: wp.array, envs: int) -> wp.array:
        """``(E, width)`` view of the block's rows in a full-length array (strided inside a host model). No copy."""
        base, stride, col = self.place(envs)
        return arr[base : base + envs * stride].reshape((envs, stride))[:, col : col + self.width]


class Scene:
    def __init__(self, device, envs: int, gravity=(0.0, 0.0, -9.81)):
        """``gravity=None``: the case leaves it to the model's owner (standalone: ``(0, 0, -9.81)``; a host's model keeps
        its own, see ``newton_model.finish_model``)."""
        self.device = wp.get_device(device)
        self.envs = check_int("sim.envs", envs, 1)
        self.gravity_declared = gravity is not None
        self._gravity = tuple(float(g) for g in (gravity if gravity is not None else (0.0, 0.0, -9.81)))
        self.main = Block("main", world_major=True)  # tissue, shell, rod and tree bodies; ``scene.q`` and friends
        self.pool = Block("pool", world_major=False)  # fluid pools: private, body-major
        self._fields: dict = {}  # scene.field views, built once (a launch argument is then just a cached object)
        self.static_meshes: list[StaticMesh] = []
        self.q = self.qd = self.q_prev = self.q0 = None
        self.inv_mass = self.radius = self.flags = self.flags0 = self.gravity = None
        self.model = self.state = None  # the Newton model and state behind the main block (``allocate``)
        # Inside a host model (Isaac Lab): where each world's rows were added and the world's (position, xyzw quaternion).
        # Our particles stay in the component frame (the case's); only the boundary with the host is transformed.
        self.world_starts: list | None = None
        self.env_xforms: list | None = None

    @property
    def bodies(self) -> list[Body]:
        """The bodies of the main block (fluid pools are ``pools``)."""
        return self.main.bodies

    @property
    def pools(self) -> list[Body]:
        """The fluid pools (the private pool block)."""
        return self.pool.bodies

    @property
    def particle_count(self) -> int:
        """Particles of the main block (the length of ``scene.q``); pools are ``pool_count``."""
        return self.main.count(self.envs)

    @property
    def pool_count(self) -> int:
        return self.pool.count(self.envs)

    def block(self, body: Body) -> Block:
        return self.pool if body.pool else self.main

    @property
    def env_rotated(self) -> bool:
        """Some world's orientation differs from the component frame's (Isaac Lab env rotations)."""
        return self.env_xforms is not None and any(
            not np.allclose(np.asarray(q, np.float64), (0.0, 0.0, 0.0, 1.0), atol=1.0e-7) for _, q in self.env_xforms)

    def refresh_gravity(self, model=None) -> None:
        """After the model's gravity changed (pass model for an explicit pre-bind refresh): a rotated component's own ``R_e^T g`` copy is rewritten in place (the
        captured graphs keep its buffer); otherwise ``gravity`` views the model's array and is already current."""
        model = model if model is not None else self.model
        if self.env_rotated and model is not None:
            from i4h_physics.newton_model import component_gravity

            model.i4h.scene_gravity[:self.envs].assign(component_gravity(model, self))

    def adopt_gravity(self, gravity) -> None:
        """Take the host's gravity as the case gravity (a case that declares none)."""
        self._gravity = tuple(float(g) for g in gravity)

    @property
    def gravity_vector(self) -> tuple:
        """The case gravity (the live per-env values are ``scene.gravity``, the model's per-world array)."""
        return self._gravity

    @property
    def down(self) -> np.ndarray:
        """Unit gravity direction (-z when gravity is zero); fluid pools are parked far along it."""
        g = np.asarray(self._gravity, np.float32)
        n = float(np.linalg.norm(g))
        return g / n if n > 0.0 else np.float32([0.0, 0.0, -1.0])

    # host-side construction -------------------------------------------------

    def add_body(self, name: str, rest: ParticleRest, parts=None, variant_of_env=None, variants=1, integrate=True,
                 pool=False) -> Body:
        """``pool=True`` stores the body in the private pool block (fluid pools); names are unique across both."""
        if self.q is not None:
            raise RuntimeError("add_body after allocate()")
        if any(b.name == name for b in self.bodies + self.pools):
            raise ValueError(f"duplicate body {name!r}")
        if not integrate and not pool:  # particles integrates the main block as a whole
            raise ValueError(f"body {name!r}: only a pool may skip the particles integration")
        E, n = rest.q0.shape[:2]
        expected = {"q0": (E, n, 3), "inv_mass": (E, n), "radius": (E, n), "flags": (E, n)}
        for key, shape in expected.items():
            arr = getattr(rest, key)
            if E != self.envs or arr.shape != shape:
                raise ValueError(f"body {name!r}: {key} has shape {arr.shape}, expected {(self.envs, n, *shape[2:])}")
        block = self.pool if pool else self.main
        body = Body(name, block.next_offset(self.envs), rest, parts, variant_of_env, variants, integrate, pool)
        block.bodies.append(body)
        return body

    def body(self, name: str) -> Body:
        """A body or a fluid pool by name."""
        for b in self.bodies + self.pools:
            if b.name == name:
                return b
        raise ValueError(f"no body {name!r}; bodies: {[b.name for b in self.bodies + self.pools]}")

    def estimate_bytes(self, construction: bool = False) -> dict[str, int]:
        """Device bytes ``allocate`` needs: the Newton model and state behind the main block, the declared particle
        defaults, history, flags and initialization templates, and the private pool block. The
        steady figure by default; ``construction`` adds the transient peak of building the Newton model, which is
        what ``allocate`` checks against free memory."""
        from i4h_physics.newton_model import estimate_bytes as newton_bytes

        sizes = {"q": VEC3_BYTES, "qd": VEC3_BYTES, "q_prev": VEC3_BYTES, "q0": VEC3_BYTES, "inv_mass": F32_BYTES, "radius": F32_BYTES, "flags": I32_BYTES,
                 "flags0": I32_BYTES}
        n, m = self.particle_count, self.pool_count
        out = newton_bytes(n, construction)
        out.update({f"pool_{field}": max(1, m) * size for field, size in sizes.items()})
        out["pool_initialization_templates"] = max(1, m) * 40
        out["pool_identity"] = max(1, m) * 8
        out["body_variants"] = len(self.bodies + self.pools) * self.envs * 4
        out["component_gravity"] = self.envs * 12 if self.env_rotated else 0
        out["static_geometry"] = sum(max(1, len(m.points)) * 20 + max(1, m.indices.size) * 12 for m in self.static_meshes)
        from i4h_physics.tet_storage import estimate_tet_bytes
        out["tet_storage"] = estimate_tet_bytes(self)
        from i4h_physics.surface_storage import estimate_surface_bytes
        out["surface_storage"] = estimate_surface_bytes(self)
        return out

    def check_memory(self, estimate: dict[str, int]) -> None:
        """Fail before allocating when ``estimate`` does not fit the device's free memory."""
        total = sum(estimate.values())
        if not self.device.is_cuda:  # Warp reports 0 free on CPU without psutil
            return
        free = self.device.free_memory
        if total > free:
            largest = sorted(estimate.items(), key=lambda kv: -kv[1])[:2]
            detail = ", ".join(f"{name} {b / 1e9:.3f} GB" for name, b in largest)
            raise MemoryError(
                f"need {total / 1e9:.3f} GB but device {self.device} has {free / 1e9:.3f} GB free; largest: {detail}"
            )

    def allocate(self, model=None, state=None, *, control=None) -> None:
        """Alias declared Scene storage in the selected Newton Model and State.

        Without a host model, author exact worlds and finalize a standalone model.
        With a model and state, validate layout and leave their arrays untouched.
        """
        if self.q is not None:
            raise RuntimeError("allocate() called twice; systems and captured graphs hold the existing arrays")
        from i4h_physics.newton_model import build_model, finish_model, place_main, rebuild_particle_bvh

        dev = self.device
        if model is None:
            if state is not None:
                raise ValueError("a state needs its model")
            self.check_memory(self.estimate_bytes(construction=True))  # the construction peak, before finalize allocates
            self.model = build_model(self)
            self.state = self.model.state()
        else:
            if state is None:
                raise ValueError("an owner's model comes with its state")
            if model.device != dev or model.world_count != self.envs:
                raise ValueError("host model device/world count does not match the authored scene")
            for name in ("particle_q", "particle_qd", "particle_f"):
                array = getattr(state, name)
                if model.particle_count and (array is None or array.device != dev or array.shape != (model.particle_count,) or array.dtype is not wp.vec3):
                    raise ValueError(f"host state {name} has incompatible device or capacity")
            place_main(model, self)
            finish_model(model, self, state, host=True)
            self.model, self.state = model, state
        if model is None:
            rebuild_particle_bvh(self.model, self.state)
        if not hasattr(self, "declarations"):
            raise RuntimeError("native particles require authored storage declarations")
        from i4h_physics.scene_storage import validate_scene
        validate_scene(self)
        self.control = control if control is not None else self.model.control()
        base, stride, col = self.main.place(self.envs)
        self.storage = self.declarations.bind(self.model, self.state, self.control,
            placements={self.particle_domain: (base + col, stride)} if self.main.width else {})
        self._alias_main()
        for name in Block.FIELDS:
            setattr(self.pool, name, self.attr(self.pool_domain, name))
        self.pool.size = self.pool_count
        for field in Block.FIELDS:  # scene.q and friends are the main block's arrays
            setattr(self, field, getattr(self.main, field))
        self.gravity = self.attr(self.gravity_domain, "gravity")
        self.storage.register_package(identities=((self.model, "gravity", self.model.gravity),))
        for b in self.bodies + self.pools:
            b.variant = self.attr(self.body_domains[b.name], "variant")

    def _alias_main(self) -> None:
        """Attach full native backing aliases without writing defaults or history."""
        m, s, main = self.model, self.state, self.main
        main.size = m.particle_count
        main.q0 = m.particle_q if m.particle_count else self.storage.attr(self.particle_domain, "q0").flatten()
        for name in ("inv_mass", "flags0", "q_prev", "flags"):
            setattr(main, name, getattr(m.i4h if name in ("inv_mass", "flags0") else s.i4h, f"particle_{name}"))
        # Empty native backing has one inaccessible element; expose empty aliases.
        if not m.particle_count:
            for name in ("inv_mass", "flags0", "q_prev", "flags"):
                setattr(main, name, getattr(main, name)[:0])
            main.q = main.q0
            main.qd = main.q0
            main.radius = main.inv_mass
        else:
            main.radius = m.particle_radius
            main.q, main.qd = s.particle_q, s.particle_qd

    # device-side helpers ----------------------------------------------------

    def view(self, arr: wp.array, body: Body) -> wp.array:
        """``(E, count)`` view of one body's slice of ``arr``, a per-particle array laid out like the body's block:
        a live array of that block or a copy of one (a snapshot). No copy, and never a substitute: the result
        always aliases ``arr``. An array of the wrong length, or a live array of the other block, is rejected."""
        own, other = (self.pool, self.main) if body.pool else (self.main, self.pool)
        size = own.size if own.size is not None else own.count(self.envs)
        if arr.shape[0] != size or (other.q is not None and any(
                arr.ptr == getattr(other, f).ptr and getattr(other, f).size for f in Block.FIELDS)):
            raise ValueError(f"body {body.name!r} lives in the {own.name} block ({size} particles); "
                             f"got an array of {arr.shape[0]}" + (" from the other block" if arr.shape[0] == size else ""))
        if own.world_major:
            return own.rows(arr, self.envs)[:, body.offset : body.offset + body.count]
        n = self.envs * body.count
        return arr[body.offset : body.offset + n].reshape((self.envs, body.count))

    def block_field(self, name: str) -> wp.array:
        """``(E, W)`` view of the main block's live array ``name``: every main-block body of each env in one row, for
        kernels that treat all of them alike (integration, bounds, reset); one launch instead of one per body."""
        key = (name, "main", None)
        cached = self._fields.get(key)
        if cached is None:
            if name not in Block.FIELDS:
                raise ValueError(f"unknown particle field {name!r}; known: {Block.FIELDS}")
            cached = self._fields[key] = self.main.rows(getattr(self.main, name), self.envs)
        return cached

    def static_wp_mesh(self, name: str) -> wp.Mesh:
        """The device ``wp.Mesh`` of static mesh ``name``, built once and shared by every system that queries it (chole's
        three ``static_contact`` systems and a ``rod_containment`` would otherwise each upload the cavity)."""
        meshes = {m.name: m for m in self.static_meshes}
        if name not in meshes:
            raise ValueError(f"{name!r}: unknown static mesh; known: {sorted(meshes)}")
        if getattr(self, "storage", None) is None:
            raise RuntimeError("scene has no completed storage declarations")
        self.storage.check()
        cached = self._fields.get(("static_mesh", name, None))
        if cached is None:
            points_domain, indices_domain = self.static_domains[name]
            points = self.attr(points_domain, "points")
            indices = self.attr(indices_domain, "indices")
            cached = self._fields[("static_mesh", name, None)] = wp.Mesh(points, indices)  # static: never refit
        return cached

    def layout(self, body: Body) -> tuple[int, int]:
        """``(offset, env_stride)``: particle ``(env, local)`` of ``body`` is ``offset + env * env_stride + local`` in its
        block's flat arrays. For code that must build flat index tables (render meshes); kernels take views."""
        block = self.block(body)
        if not block.world_major:
            return body.offset, body.count
        base, stride, col = block.place(self.envs)
        return base + col + body.offset, stride

    def part_view(self, arr: wp.array, body: Body, part: str) -> wp.array:
        """``(E, part capacity)`` strided view of one part. No copy."""
        p = body.parts[part]
        return self.view(arr, body)[:, p.particle_start : p.particle_start + p.particle_count]

    def attr(self, domain, key):
        """Return the cached zero-copy view of an explicitly declared field."""
        storage = getattr(self, "storage", None)
        if storage is None:
            raise RuntimeError("scene has no completed storage declarations")
        return storage.attr(domain, key)

    def field(self, name: str, body: Body, part: str | None = None) -> wp.array:
        """``(E, count)`` view of the live per-particle array ``name`` (``"q"``, ``"flags"``, ...) of the block the
        body lives in; one part's ``(E, capacity)`` view when ``part`` is given."""
        key = (name, id(body), part)
        cached = self._fields.get(key)
        if cached is not None:
            return cached
        if name not in Block.FIELDS:
            raise ValueError(f"unknown particle field {name!r}; known: {Block.FIELDS}")
        arr = getattr(self.block(body), name)
        if arr is None:
            raise RuntimeError("scene.field before allocate()")
        view = self.view(arr, body) if part is None else self.part_view(arr, body, part)
        self._fields[key] = view
        return view

    def initialize_particle_history(self, mask: wp.array) -> None:
        """After an explicit teleport, copy selected owned positions into history only.

        Binding and cache refresh never call this: restored history is independent
        of live positions. The caller orders its writes before this device operation.
        """
        self.storage.check()
        views = [(self.block_field, self.main.width)] if self.main.width else []
        views += [((lambda name, b=b: self.field(name, b)), b.count) for b in self.pools if b.count]
        for f, n in views:
            wp.launch(initialize_particle_history, dim=(self.envs, n), inputs=[mask, f("q")],
                      outputs=[f("q_prev")], device=self.device)

    def reset(self, mask: wp.array, positions: bool = True, velocities: bool = True) -> None:
        """Restore the envs where ``mask[env]`` is true. Always: the particle flags from ``flags0`` (topology and flags
        go back together) and ``q_prev = q``. ``positions`` restores the main block's ``q`` from ``q0`` first,
        ``velocities`` zeroes its ``qd``; the private pool block is always restored in full."""
        self.storage.check()
        views = [(self.block_field, self.main.width, positions, velocities)] if self.main.width else []
        views += [((lambda name, b=b: self.field(name, b)), b.count, True, True) for b in self.pools]  # pools: in full
        for f, n, pos, vel in views:  # one launch for the whole main block, one per pool
            wp.launch(
                reset_particles,
                dim=(self.envs, n),
                inputs=[mask, f("q0"), f("flags0"), int(pos), int(vel)],
                outputs=[f("q"), f("qd"), f("q_prev"), f("flags")],
                device=self.device,
            )


@wp.kernel
def initialize_particle_history(mask: wp.array(dtype=wp.bool), q: wp.array2d(dtype=wp.vec3),
                                previous: wp.array2d(dtype=wp.vec3)):
    env, i = wp.tid()
    if mask[env]:
        previous[env, i] = q[env, i]


@wp.kernel
def restore_variant_rows(
    mask: wp.array(dtype=wp.bool),
    variant: wp.array(dtype=wp.int32),
    src: wp.array2d(dtype=Any),
    dst: wp.array2d(dtype=Any),
):
    """``dst[env] = src[variant[env]]`` for selected envs: masked reset of a per-env state
    (active masks, live triangles, support counts) from its per-variant initial value."""
    env, k = wp.tid()
    if mask[env]:
        dst[env, k] = src[variant[env], k]


@wp.kernel
def fill_rows(mask: wp.array(dtype=wp.bool), value: Any, dst: wp.array2d(dtype=Any)):
    """``dst[env, :] = value`` for selected envs: masked reset of per-env scratch (heat, slots,
    counters). An ``(E,)`` array is passed as ``arr.reshape((E, 1))``, a body-local ``(E*count,)``
    one as ``arr.reshape((E, count))``; both are views."""
    env, k = wp.tid()
    if mask[env]:
        dst[env, k] = value


# Pattern from newton/_src/solvers/vbd/particle_vbd_kernels.py:1350-1375 (reset_particle_state),
# reimplemented over one body's (E, count) views.
@wp.kernel
def reset_particles(
    mask: wp.array(dtype=wp.bool),
    q0: wp.array2d(dtype=wp.vec3),
    flags0: wp.array2d(dtype=wp.int32),
    positions: wp.int32,
    velocities: wp.int32,
    q: wp.array2d(dtype=wp.vec3),
    qd: wp.array2d(dtype=wp.vec3),
    q_prev: wp.array2d(dtype=wp.vec3),
    flags: wp.array2d(dtype=wp.int32),
):
    env, i = wp.tid()
    if not mask[env]:
        return
    if positions != 0:
        q[env, i] = q0[env, i]
    q_prev[env, i] = q[env, i]
    if velocities != 0:
        qd[env, i] = wp.vec3(0.0, 0.0, 0.0)
    flags[env, i] = flags0[env, i]
