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
rest rows (``ParticleRest``) are per env, ``(E, ...)``. ``Scene.allocate()`` uploads the
particle rest state; the system that reads any other record uploads it and owns the
device arrays.

``Scene.allocate()`` builds the Newton model behind the main block (one world per env,
``newton_model.build_model``) and its state: the main block's ``q0``, ``inv_mass``,
``radius`` and ``flags`` are the model's particle arrays, ``q`` and ``qd`` the state's,
``gravity`` an ``(E,)`` view of the model's per-world gravity; ``q_prev`` and ``flags0``
stay scene-owned. Nothing is copied between the two.

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

_VEC3 = 12
_F32 = 4
_I32 = 4


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

    def refresh_gravity(self) -> None:
        """After the model's gravity changed: a rotated component's own ``R_e^T g`` copy is rewritten in place (the
        captured graphs keep its buffer); otherwise ``gravity`` views the model's array and is already current."""
        if self.env_rotated and self.gravity is not None and self.model is not None:
            from i4h_physics.newton_model import component_gravity

            self.gravity.assign(component_gravity(self.model, self))

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
        """Device bytes ``allocate`` needs: the Newton model and state behind the main block (its q, qd, q0, inv_mass,
        radius and flags are Newton's arrays), the scene-owned q_prev and flags0, and the private pool block. The
        steady figure by default; ``construction`` adds the transient peak of building the Newton model, which is
        what ``allocate`` checks against free memory."""
        from i4h_physics.newton_model import estimate_bytes as newton_bytes

        sizes = {"q": _VEC3, "qd": _VEC3, "q_prev": _VEC3, "q0": _VEC3, "inv_mass": _F32, "radius": _F32, "flags": _I32,
                 "flags0": _I32}
        n, m = self.particle_count, self.pool_count
        out = {"q_prev": n * _VEC3, "flags0": n * _I32}
        out.update(newton_bytes(n, construction))
        out.update({f"pool_{field}": m * size for field, size in sizes.items()})
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

    def allocate(self, model=None, state=None) -> None:
        """Build the Newton model and state behind the main block and alias them (standalone), or take an owner's
        finalized ``model`` and ``state`` (Isaac Lab's NewtonManager finalizes from ``newton_model.build_builder`` and
        owns ``state_0``): the per-env rows are written into them here, then aliased the same way."""
        if self.q is not None:
            raise RuntimeError("allocate() called twice; systems and captured graphs hold the existing arrays")
        from i4h_physics.newton_model import (
            build_model,
            component_gravity,
            finish_model,
            place_main,
            rebuild_particle_bvh,
        )

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
            place_main(model, self)
            finish_model(model, self, state, host=True)
            self.model, self.state = model, state
        rebuild_particle_bvh(self.model, self.state)
        self._alias_main()
        self._allocate_block(self.pool)
        for field in Block.FIELDS:  # scene.q and friends are the main block's arrays
            setattr(self, field, getattr(self.main, field))
        if self.env_rotated:  # the component frame differs per env: our own (E,) gravity, R_e^T g (refresh_gravity)
            self.gravity = wp.array(component_gravity(self.model, self), dtype=wp.vec3, device=dev)
        else:
            self.gravity = self.model.gravity[: self.envs]  # (E,) view of Newton's per-world gravity (the last slot is world -1)
        for b in self.bodies + self.pools:
            b.variant = wp.array(b.variant_of_env, dtype=wp.int32, device=dev)

    def _alias_main(self) -> None:
        """The main block's arrays are the Newton model's and state's: q0, inv_mass, radius and flags on the model
        (flags are mutated in place: cuts and burns are model edits, so Newton sees them), q and qd on the state; q_prev
        and flags0 stay scene-owned. An empty main block (a fluid-only scene) gets zero-length arrays: Newton allocates
        no particle arrays then."""
        m, s, main = self.model, self.state, self.main
        if m.particle_count == 0:
            self._allocate_block(main)
            return
        base, stride, col = main.place(self.envs)
        if base + (self.envs - 1) * stride + col + main.width > m.particle_count:
            raise RuntimeError(f"newton model has {m.particle_count} particles; the main block's placement {main.place(self.envs)} "
                               f"needs {base + (self.envs - 1) * stride + col + main.width}")
        main.size = m.particle_count  # host particles share the arrays; every main-block array has this length
        main.q0, main.inv_mass, main.radius, main.flags = m.particle_q, m.particle_inv_mass, m.particle_radius, m.particle_flags
        main.q, main.qd = s.particle_q, s.particle_qd
        main.q_prev = wp.clone(m.particle_q)
        main.flags0 = wp.clone(m.particle_flags)

    def _allocate_block(self, block: Block) -> None:
        dev, n = self.device, block.count(self.envs)

        # Only for the body-major pool block and an empty main block: a populated main block is the Newton model's
        # arrays (newton_model.main_rows lays out its world-major rows).
        assert not (block.world_major and block.bodies), "a populated main block is allocated by build_model"

        def flat(field, dtype):  # (E, n, *tail) rows of every body -> (N, *tail), each body's envs one after another
            rows = [getattr(b.rest, field) for b in block.bodies]
            if not rows:
                return np.zeros((0, 3) if field == "q0" else (0,), dtype)
            return np.concatenate([r.reshape((-1,) + r.shape[2:]) for r in rows])

        block.q0 = wp.array(flat("q0", np.float32), dtype=wp.vec3, device=dev)
        block.q = wp.clone(block.q0)
        block.q_prev = wp.clone(block.q0)
        block.qd = wp.zeros(n, dtype=wp.vec3, device=dev)
        block.inv_mass = wp.array(flat("inv_mass", np.float32), dtype=wp.float32, device=dev)
        block.radius = wp.array(flat("radius", np.float32), dtype=wp.float32, device=dev)
        block.flags0 = wp.array(flat("flags", np.int32), dtype=wp.int32, device=dev)
        block.flags = wp.clone(block.flags0)

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
        cached = self._fields.get(("static_mesh", name, None))
        if cached is None:
            m = meshes[name]
            self.check_memory({"mesh": m.points.nbytes + m.indices.nbytes})
            points = wp.array(m.points, dtype=wp.vec3, device=self.device)
            indices = wp.array(np.ascontiguousarray(m.indices.reshape(-1), dtype=np.int32), dtype=wp.int32, device=self.device)
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

    def reset(self, mask: wp.array, positions: bool = True, velocities: bool = True) -> None:
        """Restore the envs where ``mask[env]`` is true. Always: the particle flags from ``flags0`` (topology and flags
        go back together) and ``q_prev = q``. ``positions`` restores the main block's ``q`` from ``q0`` first,
        ``velocities`` zeroes its ``qd``; the private pool block is always restored in full."""
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
