# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M9a part 1: the shell vessel body (loader, XPBD edge and dihedral constraints) and the two-way dynamic
containment (single-triangle numbers from omniendo tests/test_xcath_two_way_collision.py), then the aorta_shell case."""

import os

import numpy as np
import pytest
import warp as wp
from conftest import env_slice
from helpers import VESSELS, find, run_rates, signed_phi
from i4h_endoluminal.loaders.vessel_mesh import build_shell
from i4h_endoluminal.systems.rods.catheter_control import CatheterControl
from i4h_isaaclab.composition import load_case
from i4h_physics import Scene, Stepper
from i4h_physics.scene import ACTIVE_INT, Part, ParticleRest
from i4h_physics.systems.contact.shell_containment import ShellContainment
from i4h_physics.systems.mechanics.particles import Particles
from i4h_physics.systems.mechanics.shell import Shell
from i4h_physics.systems.rods.rod_direct import RodDirect

needs_aorta = pytest.mark.skipif(not os.environ.get("I4H_DATA") and (not (VESSELS / 'DynamicAorta.usdc').is_file()), reason=f"DynamicAorta.usdc not under {VESSELS}")

# omniendo tests/test_solver_xpbd_rod_deformable.py:17-37: a unit quad of two triangles hanging from its top edge.
QUAD_POINTS = np.float32([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
QUAD_TRIS = np.int32([[0, 2, 1], [1, 2, 3]])
PIN_TOP = [{"type": "sphere", "center": [0.5, 0.0, 1.0], "radius": 0.6}]  # vertices 0 and 1


def shell_scene(points, tris, envs=1, pins=(), gravity=(0.0, 0.0, 0.0), density=1.0, extra_bodies=()):
    """A scene with one shell body (and optional extra ``(name, ParticleRest)`` bodies), allocated."""
    scene = Scene("cuda:0", envs, gravity=gravity)
    for name, rest in extra_bodies:
        scene.add_body(name, rest, {name: Part(0, 0, rest.q0.shape[1])})
    rest, mesh = build_shell("wall", points, tris, envs, density=density, pins=pins)
    body = scene.add_body("wall", rest, {"wall": Part(0, 0, len(points), tri_start=0, tri_count=len(tris))})
    body.shell = mesh
    scene.allocate()
    return scene, body


def quad_sim(pins, gravity=(0.0, 0.0, 0.0), dt=1.0 / 120.0, **shell):
    scene, body = shell_scene(QUAD_POINTS, QUAD_TRIS, pins=pins, gravity=gravity)
    sim = Stepper.create(scene, [Particles(scene, damping=1.0), Shell(scene, body="wall", **shell)], dt=dt, substeps=1, iterations=1)
    return sim, scene, body


def test_shell_loader_quad():
    rest, mesh = build_shell("quad", QUAD_POINTS, QUAD_TRIS, 2, pins=[{"type": "boundary"}])
    assert mesh.edges.shape == (1, 5, 2) and mesh.dihedrals.shape == (1, 1, 4)
    assert set(map(tuple, mesh.edges[0])) == {(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)}
    assert np.allclose(mesh.dihedral_rest, 0.0) and set(mesh.dihedrals[0, 0, 2:]) == {1, 2}
    assert np.allclose(mesh.dihedral_edge_len[0], np.sqrt(2.0))
    assert rest.q0.shape == (2, 4, 3) and (rest.inv_mass == 0.0).all()  # every vertex is on the boundary
    rest, _ = build_shell("quad", QUAD_POINTS, QUAD_TRIS, 1)
    assert np.allclose(1.0 / rest.inv_mass[0], [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])  # area/3 per incident triangle
    rest, _ = build_shell("quad", QUAD_POINTS, QUAD_TRIS, 1, pins=PIN_TOP)
    assert (rest.inv_mass[0, :2] == 0.0).all() and (rest.inv_mass[0, 2:] > 0.0).all()
    rest, _ = build_shell("quad", QUAD_POINTS, QUAD_TRIS, 1, pins=[{"type": "box", "minimum": [-1, -1, -0.5], "maximum": [2, 1, 0.5]}])
    assert (rest.inv_mass[0] == [6.0, 3.0, 0.0, 0.0]).all()
    with pytest.raises(ValueError, match="matched no vertices"):
        build_shell("quad", QUAD_POINTS, QUAD_TRIS, 1, pins=[{"type": "sphere", "center": [5, 5, 5], "radius": 0.1}])
    with pytest.raises(ValueError, match="boundary, sphere or box"):
        build_shell("quad", QUAD_POINTS, QUAD_TRIS, 1, pins=[{"type": "ring"}])


def test_shell_rest_holds():
    sim, scene, body = quad_sim(PIN_TOP)
    q0 = scene.q.numpy().copy()
    sim.prepare()
    for _ in range(20):
        sim.step()
    assert np.abs(scene.q.numpy() - q0).max() < 1.0e-6


def test_shell_gravity_edge_error():
    """omniendo test_gravity_constraint_error: 120 steps under gravity, 12 sweeps, edge error < 5% (Jacobi)."""
    sim, scene, body = quad_sim(PIN_TOP, gravity=(0.0, 0.0, -9.81), iterations=12)
    q0 = scene.q.numpy().copy()
    sim.prepare()
    for _ in range(120):
        sim.step()
    q = scene.q.numpy()
    assert np.isfinite(q).all()  # the quad hangs from taut vertical edges: it does not fall, it must not stretch
    assert np.abs(q[:2] - q0[:2]).max() < 1.0e-6  # the pinned row is exact
    edges = body.shell.edges[0]
    lengths = np.linalg.norm(q[edges[:, 1]] - q[edges[:, 0]], axis=1)
    assert np.abs(lengths / body.shell.edge_rest[0] - 1.0).max() < 0.05


def test_shell_dihedral_flattens():
    """Bending only: a vertex displaced out of the plane returns towards it (omniendo dihedral toggle test)."""
    pins = PIN_TOP + [{"type": "sphere", "center": [0.0, 0.0, 0.0], "radius": 0.1}]  # 0, 1, 2 pinned; 3 free
    sim, scene, body = quad_sim(pins, iterations=16, stretch=0.0, bend=1.0e5)
    q = scene.q.numpy()
    q[3, 1] = 0.3
    scene.q.assign(q)
    scene.q_prev.assign(q)
    sim.prepare()
    heights = []
    for _ in range(5):
        sim.step()
        heights.append(abs(float(scene.q.numpy()[3, 1])))
    # One stiff step snaps the vertex back, the carried velocity overshoots once, then it settles: the envelope
    # decays to under 1% of the displacement.
    assert all(np.isfinite(heights)) and max(heights) < 0.3 and heights[-1] < 0.01 * 0.3 and heights[-1] < heights[1], heights
    assert np.abs(scene.q.numpy()[:3] - q[:3]).max() < 1.0e-6


# omniendo tests/test_xcath_two_way_collision.py:22-105: one rod point above a unit right triangle.
TRI_POINTS = np.float32([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
TRI = np.int32([[0, 1, 2]])
ROD_POINT = np.float32([0.25, 0.25, 0.05])  # barycentric (0.5, 0.25, 0.25)


def contact_scene(pins=(), z=0.05, **kw):
    node = ParticleRest(q0=np.float32([[[0.25, 0.25, z]]]), inv_mass=np.ones((1, 1), np.float32),
                        radius=np.full((1, 1), 0.1, np.float32), flags=np.full((1, 1), ACTIVE_INT, np.int32))
    scene, wall = shell_scene(TRI_POINTS, TRI, pins=pins, extra_bodies=[("rod", node)])
    cont = ShellContainment(scene, body="rod", shell="wall", sign=1.0, max_distance=1.0, **kw)
    sim = Stepper.create(scene, [cont], dt=1.0 / 60.0, substeps=1, iterations=1, cuda_graph=False)
    sim.prepare()
    return sim, scene, cont


def surface_point(scene, wall):
    q = scene.view(scene.q, wall).numpy()[0]
    return 0.5 * q[0] + 0.25 * q[1] + 0.25 * q[2]


def test_two_way_half_split():
    sim, scene, cont = contact_scene()
    cont.substep_start(scene, sim.dt, 0, 1)
    cont.iteration(scene, sim.dt, 0, 1)
    rod_z = float(scene.view(scene.q, scene.body("rod")).numpy()[0, 0, 2])
    wall_z = float(surface_point(scene, scene.body("wall"))[2])
    assert abs(rod_z - (-0.025)) < 1.0e-6 and abs(wall_z - 0.075) < 1.0e-6 and abs(wall_z - rod_z - 0.1) < 1.0e-6
    assert cont.contact_count.numpy()[0] == 1 and abs(float(cont.contact_depth.numpy()[0]) - 0.15) < 1.0e-6
    assert (cont.wall_delta.numpy() == 0.0).all()  # applied and cleared
    before = scene.view(scene.q, scene.body("rod")).numpy().copy()
    cont.iteration(scene, sim.dt, 0, 1)  # the wall boxes are refit: the resolved contact leaves less than 1%
    after = scene.view(scene.q, scene.body("rod")).numpy()
    assert np.abs(after - before).max() < 0.01 * 0.075


def test_two_way_pinned_triangle_moves_only_the_rod():
    sim, scene, cont = contact_scene(pins=[{"type": "boundary"}])
    wall0 = scene.view(scene.q, scene.body("wall")).numpy().copy()
    cont.substep_start(scene, sim.dt, 0, 1)
    cont.iteration(scene, sim.dt, 0, 1)
    assert abs(float(scene.view(scene.q, scene.body("rod")).numpy()[0, 0, 2]) - (-0.1)) < 1.0e-6
    assert np.array_equal(scene.view(scene.q, scene.body("wall")).numpy(), wall0)


def test_two_way_partial_support():
    """Vertices 0 and 2 pinned: only vertex 1 moves, bounded by half the penetration, separation stays the radius."""
    pins = [{"type": "sphere", "center": [0.0, 0.0, 0.0], "radius": 0.1}, {"type": "sphere", "center": [0.0, 1.0, 0.0], "radius": 0.1}]
    sim, scene, cont = contact_scene(pins=pins)
    wall = scene.body("wall")
    wall0 = scene.view(scene.q, wall).numpy()[0].copy()
    cont.substep_start(scene, sim.dt, 0, 1)
    cont.iteration(scene, sim.dt, 0, 1)
    q = scene.view(scene.q, wall).numpy()[0]
    assert np.array_equal(q[[0, 2]], wall0[[0, 2]]) and 0.0 < abs(q[1, 2]) <= 0.5 * 0.15 + 1.0e-6
    rod_z = float(scene.view(scene.q, scene.body("rod")).numpy()[0, 0, 2])
    assert abs(float(surface_point(scene, wall)[2]) - rod_z - 0.1) < 1.0e-6


def test_zero_response_equals_one_way():
    a = contact_scene(target_response=0.0)
    b = contact_scene(two_way=False)
    for sim, scene, cont in (a, b):
        cont.substep_start(scene, sim.dt, 0, 1)
        cont.iteration(scene, sim.dt, 0, 1)
    qa, qb = (s.q.numpy() for s in (a[1], b[1]))
    assert np.abs(qa - qb).max() < 1.0e-7
    assert abs(float(a[1].view(a[1].q, a[1].body("rod")).numpy()[0, 0, 2]) - (-0.1)) < 1.0e-6


def test_two_sided_crossing_returns_to_its_side():
    """A node that crossed to z = -0.3 with a remembered outside side is projected back to +radius."""
    sim, scene, cont = contact_scene(pins=[{"type": "boundary"}], z=-0.3, two_sided=True)
    cont.sides.fill_(1.0)  # the side history says "outside"; no substep_start so it is not reclassified
    cont.iteration(scene, sim.dt, 0, 1)
    assert abs(float(scene.view(scene.q, scene.body("rod")).numpy()[0, 0, 2]) - 0.1) < 1.0e-6


# --- aorta_shell ---------------------------------------------------------------------------------------------

RATES = np.float32([[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]])


def phi_against_live_wall(sim, env):
    """Signed distance of env's catheter nodes to that env's current wall (a wp.Mesh built here, sign -1 inside)."""
    scene, wall, rod = sim.scene, sim.scene.body("aorta"), sim.scene.body("catheter")
    points = wp.array(env_slice(scene, scene.q, wall, env), dtype=wp.vec3, device=scene.device)
    indices = wp.array(wall.shell.tris[0].reshape(-1), dtype=wp.int32, device=scene.device)
    mesh = wp.Mesh(points, indices)
    q = wp.array(env_slice(scene, scene.q, rod, env), dtype=wp.vec3, device=scene.device)
    out = wp.zeros(rod.count, dtype=wp.float32, device=scene.device)
    wp.launch(signed_phi, dim=rod.count, inputs=[mesh.id, q], outputs=[out], device=scene.device)
    return out.numpy(), q.numpy()


@needs_aorta
def test_aorta_shell_nodes_stay_inside():
    sim = load_case("aorta_shell", envs=2)
    scene, wall, _rod = sim.scene, sim.scene.body("aorta"), sim.scene.body("catheter")
    wall0 = scene.view(scene.q, wall).numpy().copy()
    pinned = wall.rest.inv_mass[0] == 0.0
    assert pinned.sum() == 344  # omniendo: 344 vertices on the open rims
    run_rates(sim, 300, RATES)
    assert np.isfinite(scene.q.numpy()).all()
    cont = find(sim, ShellContainment)
    assert (cont.contact_count.numpy() > 0).all()
    for env in range(2):
        phi, q = phi_against_live_wall(sim, env)
        inside = q[:, 0] > 6.35 + 0.3  # nodes well past the track end are in the vessel
        assert inside.any() and phi[inside].max() <= -0.02 + 2.0e-3, (env, phi[inside].max())
    q_wall = scene.view(scene.q, wall).numpy()
    assert np.array_equal(q_wall[:, pinned], wall0[:, pinned])  # pins bitwise
    assert 0.0 < np.abs(q_wall - wall0).max() < 0.05  # the wall yielded (about a radius) but did not fly


@needs_aorta
def test_aorta_shell_env_isolation():
    a, b = load_case("aorta_shell", envs=2), load_case("aorta_shell", envs=2)
    run_rates(a, 200, RATES)
    run_rates(b, 200, np.float32([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]))
    for name in ("catheter", "aorta"):
        qa, qb = (env_slice(s.scene, s.scene.q, s.scene.body(name), 1) for s in (a, b))
        assert np.allclose(qa, qb, rtol=1e-4, atol=1e-3), name
    qa0, qb0 = (env_slice(s.scene, s.scene.q, s.scene.body("catheter"), 0) for s in (a, b))
    assert not np.allclose(qa0, qb0, atol=1e-2)  # env 0 did move differently


@needs_aorta
def test_aorta_shell_masked_reset():
    sim = load_case("aorta_shell", envs=2)
    scene = sim.scene
    run_rates(sim, 150, RATES)
    cont = find(sim, ShellContainment)
    keep = {name: env_slice(scene, scene.q, scene.body(name), 0) for name in ("catheter", "aorta")}
    sides0 = cont.sides.numpy().reshape(2, -1)[0].copy()
    sim.reset([1])
    for name, before in keep.items():
        body = scene.body(name)
        assert np.array_equal(env_slice(scene, scene.q, body, 0), before), name
        assert np.array_equal(env_slice(scene, scene.q, body, 1), env_slice(scene, scene.q0, body, 1)), name
    sides = cont.sides.numpy().reshape(2, -1)
    assert np.array_equal(sides[0], sides0) and (sides[1] == -1.0).all()
    assert cont.contact_count.numpy()[1] == 0 and cont.contact_depth.numpy()[1] == 0.0


@needs_aorta
def test_aorta_shell_captured_equals_uncaptured():
    captured, eager = load_case("aorta_shell", envs=2), load_case("aorta_shell", envs=2)
    eager.cuda_graph = False
    run_rates(captured, 120, RATES)
    run_rates(eager, 120, RATES)
    assert captured.graph is not None and eager.graph is None
    for name in ("catheter", "aorta"):
        qc, qe = (s.scene.view(s.scene.q, s.scene.body(name)).numpy() for s in (captured, eager))
        assert np.allclose(qc, qe, rtol=1e-4, atol=1e-3), name


@needs_aorta
def test_aorta_shell_system_order_and_batching():
    sim = load_case("aorta_shell", envs=3)
    kinds = [type(s) for s in sim.systems]
    assert kinds.index(Particles) < kinds.index(RodDirect) < kinds.index(CatheterControl) < kinds.index(Shell) < kinds.index(ShellContainment)
    dims = []
    real = wp.launch

    def record(kernel, dim, *args, **kwargs):
        dims.append(tuple(dim) if isinstance(dim, (tuple, list)) else (dim,))
        return real(kernel, dim, *args, **kwargs)

    wp.launch = record
    try:
        sim.cuda_graph = False
        sim.prepare()
        sim.step()
    finally:
        wp.launch = real
    assert dims and all(d[0] == 3 for d in dims if len(d) > 1)

    assert "contact_depth" in ShellContainment.state_products
