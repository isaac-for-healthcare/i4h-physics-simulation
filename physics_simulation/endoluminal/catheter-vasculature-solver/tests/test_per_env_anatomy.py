# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for batches whose envs hold different anatomy.

The property that matters is the same one the uniform batch is held to: an env
in a mixed batch must reproduce the result of simulating its anatomy alone.
Padding a smaller vessel up to the batch's window must change nothing, and
neither may sharing a launch with a different patient.

Runs on CPU so CI covers the batched kernels, and additionally on CUDA when a
device is present (those cases carry the ``gpu`` marker).
"""

from __future__ import annotations

import numpy as np
import pytest

wp = pytest.importorskip("warp")
wp.init()

from catheter_vasculature_solver import CathRodSolver, RodConfig  # noqa: E402
from catheter_vasculature_solver.vessel_deformation import (  # noqa: E402
    CenterlineData,
    CenterlineDynamicsParams,
    CenterlineVesselRuntime,
    build_centerline_tree,
)
from catheter_vasculature_solver.vessel_deformation.pbd_cosserat import CosseratRod  # noqa: E402
from catheter_vasculature_solver.vessel_deformation.pbd_cosserat import step as cosserat_step  # noqa: E402

ROD_HEIGHT = 0.5
SEGMENT_LENGTH = 0.02
NUM_SEGMENTS = 16
LUMEN_RADIUS = 0.02
CATHETER_RADIUS = 0.002
ENTRY = np.array([0.0, 0.0, ROD_HEIGHT], dtype=np.float32)


def _device_params():
    params = [pytest.param("cpu", id="cpu")]
    try:
        has_cuda = wp.get_cuda_device_count() > 0
    except Exception:  # pragma: no cover - no CUDA driver at all
        has_cuda = False
    if has_cuda:
        params.append(pytest.param("cuda:0", id="cuda", marks=pytest.mark.gpu))
    return params


DEVICES = _device_params()


# --------------------------------------------------------------------------- #
# Anatomy builders
# --------------------------------------------------------------------------- #
def _polyline_segments(points: np.ndarray, radius: float):
    starts, ends = points[:-1], points[1:]
    r = np.full(len(starts), radius, dtype=np.float32)
    return starts, ends, r


def _tree(pieces, radius=LUMEN_RADIUS, origin=ENTRY):
    """A centerline tree from polylines laid out from ``ENTRY``, moved to start at ``origin``."""
    shift = np.asarray(origin, np.float32) - ENTRY
    starts, ends, radii = zip(*(_polyline_segments(p + shift, radius) for p in pieces), strict=True)
    starts, ends, radii = np.concatenate(starts), np.concatenate(ends), np.concatenate(radii)
    data = CenterlineData(starts, ends, np.zeros(len(starts), dtype=np.int32), radii, radii, radii, radii)
    return build_centerline_tree(data, root_position=np.asarray(origin, np.float32))


def _line(a, b, segments):
    t = np.linspace(0.0, 1.0, segments + 1, dtype=np.float32)[:, None]
    return (np.asarray(a, np.float32)[None] * (1.0 - t) + np.asarray(b, np.float32)[None] * t).astype(np.float32)


def _curved_lumen(curvature: float, segments: int = 12, radius: float = LUMEN_RADIUS, reverse: bool = False,
                  origin=ENTRY):
    """A single vessel along +X that bends toward +Y distally.

    ``reverse`` authors it distal to proximal, so the entry is not node 0.
    """
    length = SEGMENT_LENGTH * NUM_SEGMENTS * 1.25
    x = np.linspace(0.0, length, segments + 1, dtype=np.float32)
    points = np.stack([x, curvature * (x / length) ** 2, np.full_like(x, ROD_HEIGHT)], axis=1)
    return _tree([points[::-1] if reverse else points], radius, origin)


def _fork_lumen(origin=ENTRY):
    """A trunk that splits in two, with a different node and edge count."""
    junction = (0.15, 0.0, ROD_HEIGHT)
    return _tree([
        _line(ENTRY, junction, 6),
        _line(junction, (0.42, 0.07, ROD_HEIGHT), 9),
        _line(junction, (0.38, -0.05, ROD_HEIGHT), 5),
    ], origin=origin)


VESSEL_KWARGS = dict(
    params=CenterlineDynamicsParams(iterations=2, root_locked=True),
    catheter_radius=CATHETER_RADIUS,
    max_distance=0.5,
    catheter_max_delta=0.01,
    vessel_max_delta=0.01,
)


def _rod_config(device: str) -> RodConfig:
    cfg = RodConfig()
    cfg.device = device
    cfg.geometry.num_segments = NUM_SEGMENTS
    cfg.geometry.segment_length = SEGMENT_LENGTH
    return cfg


def _solver(device, num_envs, *, runtime=None, track_start=ENTRY, track_dir=(1.0, 0.0, 0.0),
            track_path=None, track_enabled=False) -> CathRodSolver:
    return CathRodSolver(
        _rod_config(device),
        num_envs=num_envs,
        collision_mesh=None,
        track_start=np.asarray(track_start, dtype=np.float32),
        track_dir=np.asarray(track_dir, dtype=np.float32),
        track_path=track_path,
        track_length=0.6,
        tip_num_edges=6,
        particle_radius=CATHETER_RADIUS,
        segment_length=SEGMENT_LENGTH,
        collision_enabled=False,
        track_enabled=track_enabled,
        centerline_runtime=runtime,
    )


def _advance(solver, steps: int = 30):
    dt = _rod_config("cpu").solver.dt
    for _ in range(steps):
        solver.step(dt)


def _catheter(solver) -> np.ndarray:
    points = NUM_SEGMENTS + 1
    return solver.positions.detach().cpu().numpy().reshape((-1, points, 3))


# --------------------------------------------------------------------------- #
# CosseratRod with per-env topology
# --------------------------------------------------------------------------- #
CHAIN = np.stack([np.linspace(0.0, 0.3, 7), np.zeros(7), np.zeros(7)], axis=1).astype(np.float32)
TREE = np.array(
    [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0], [0.4, 0, 0],
     [0.5, 0.1, 0], [0.6, 0.2, 0], [0.5, -0.1, 0], [0.6, -0.2, 0]],
    dtype=np.float32,
)
TREE_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (4, 7), (7, 8)]
CHAIN_EDGES = [(i, i + 1) for i in range(len(CHAIN) - 1)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["pbd", "scr"])
def test_mixed_topology_rod_matches_each_rod_alone(device, mode):
    """Every env of a mixed batch must follow its own topology's trajectory exactly."""
    kwargs = dict(orientation_mode=mode)
    alone = {
        "chain": CosseratRod(CHAIN, device, edges=CHAIN_EDGES, fixed_nodes=[0], fixed_segments=[0], **kwargs),
        "tree": CosseratRod(TREE, device, edges=TREE_EDGES, fixed_nodes=[0], fixed_segments=[0], **kwargs),
    }
    layout = ["tree", "chain", "tree", "chain"]
    shapes = {"chain": (CHAIN, CHAIN_EDGES), "tree": (TREE, TREE_EDGES)}
    batch = CosseratRod.from_envs(
        [shapes[name][0] for name in layout], device,
        env_edges=[shapes[name][1] for name in layout],
        env_fixed_nodes=[[0]] * len(layout),
        env_fixed_segments=[[0]] * len(layout),
        **kwargs,
    )
    assert not batch.uniform_topology
    assert batch.n_nodes_per_env == len(TREE) and batch.n_seg_per_env == len(TREE_EDGES)

    for _ in range(20):
        for rod in (*alone.values(), batch):
            cosserat_step(rod, 0.005, gravity=(0.0, 0.0, -9.81), iterations=3)

    per_env = batch.positions_per_env()
    for env, name in enumerate(layout):
        reference = alone[name].positions()
        np.testing.assert_allclose(per_env[env, : len(reference)], reference, rtol=0, atol=1.0e-6)
        # Padding stays parked on the env's root, which never moves.
        np.testing.assert_array_equal(per_env[env, len(reference):], np.broadcast_to(
            shapes[name][0][0], per_env[env, len(reference):].shape))


def test_padding_is_inert():
    """Padded nodes and segments carry no mass and join no projection."""
    batch = CosseratRod.from_envs(
        [CHAIN, TREE], "cpu", env_edges=[CHAIN_EDGES, TREE_EDGES], env_fixed_nodes=[[0], [0]],
    )
    nodes, segments = batch.n_nodes_per_env, batch.n_seg_per_env
    real_nodes, real_segments = len(CHAIN), len(CHAIN_EDGES)
    np.testing.assert_array_equal(batch.env_node_counts, [len(CHAIN), len(TREE)])
    np.testing.assert_array_equal(batch.env_segment_counts, [len(CHAIN_EDGES), len(TREE_EDGES)])
    assert np.all(batch.inv_mass.numpy()[real_nodes:nodes] == 0.0)
    assert np.all(batch.inv_mass_q.numpy()[real_segments:segments] == 0.0)
    padded = set(range(real_segments, segments))
    for groups in (batch.stretch_groups, batch.seg_groups):
        members = set(np.concatenate([g.numpy() for g in groups]).tolist())
        assert not members & padded
    assert batch.local_edges is None  # a mixed batch has no single topology to report


def test_from_envs_rejects_mismatched_lists():
    with pytest.raises(ValueError, match="env_edges"):
        CosseratRod.from_envs([CHAIN, TREE], "cpu", env_edges=[CHAIN_EDGES])
    with pytest.raises(ValueError, match="outside"):
        CosseratRod.from_envs([CHAIN], "cpu", env_edges=[[(0, 99)]])


# --------------------------------------------------------------------------- #
# CenterlineVesselRuntime.from_trees: different patients, one batch
# --------------------------------------------------------------------------- #
def _alone(device, tree, response, **solver_kwargs):
    runtime = CenterlineVesselRuntime.from_tree(
        tree, device=device, num_envs=1, two_way=response > 0.0, vessel_response=response, **VESSEL_KWARGS
    )
    solver = _solver(device, 1, runtime=runtime, **solver_kwargs)
    _advance(solver)
    return _catheter(solver)[0], runtime.positions.copy(), float(solver.contact_depth_per_env[0])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("response", [0.0, 0.6])
def test_mixed_patients_match_each_patient_alone(device, response):
    """Catheter, wall and contact per env must equal that patient simulated alone.

    The vessels begin ahead of the catheter's root, so part of the shaft lies
    behind each patient's entry and is only left alone if that patient's own
    entry is the one tested.
    """
    entry = ENTRY + np.array([0.06, 0.0, 0.0], np.float32)
    trees = [_curved_lumen(0.05, origin=entry), _fork_lumen(entry),
             _curved_lumen(0.10, segments=8, reverse=True, origin=entry)]
    assert len({len(t.positions) for t in trees}) == len(trees), "topologies must differ"
    assert len({int(t.root) for t in trees}) > 1, "entries must sit at different node indices"
    env_trees = [1, 0, 2, 1]

    runtime = CenterlineVesselRuntime.from_trees(
        trees, device=device, env_trees=env_trees, two_way=response > 0.0, vessel_response=response,
        **VESSEL_KWARGS,
    )
    assert runtime.per_env_vessels and runtime.num_envs == len(env_trees)
    solver = _solver(device, len(env_trees), runtime=runtime)
    _advance(solver)
    catheter, wall, depth = _catheter(solver), runtime.positions_per_env, solver.contact_depth_per_env

    references = [_alone(device, tree, response) for tree in trees]
    for env, index in enumerate(env_trees):
        ref_catheter, ref_wall, ref_depth = references[index]
        np.testing.assert_allclose(catheter[env], ref_catheter, rtol=0, atol=1.0e-6)
        np.testing.assert_allclose(wall[env, : len(ref_wall)], ref_wall, rtol=0, atol=1.0e-6)
        assert depth[env] == pytest.approx(ref_depth, abs=1.0e-6)
    assert depth.max() > 0.0, "no contact to compare"


@pytest.mark.parametrize("device", DEVICES)
def test_mixed_patient_reset_is_per_env(device):
    runtime = CenterlineVesselRuntime.from_trees(
        [_curved_lumen(0.08), _fork_lumen()], device=device, two_way=True, vessel_response=0.6,
        **VESSEL_KWARGS,
    )
    solver = _solver(device, 2, runtime=runtime)
    rest = runtime.positions_per_env.copy()
    _advance(solver)
    deformed = runtime.positions_per_env.copy()
    assert np.abs(deformed - rest).max() > 1.0e-5

    solver.reset([1])
    after = runtime.positions_per_env
    np.testing.assert_allclose(after[1], rest[1], rtol=0, atol=1.0e-6)
    np.testing.assert_array_equal(after[0], deformed[0])


def test_mixed_patient_skinning_uses_only_real_edges():
    trees = [_curved_lumen(0.05), _fork_lumen()]
    runtime = CenterlineVesselRuntime.from_trees(trees, device="cpu", **VESSEL_KWARGS)
    smaller = int(np.argmin([len(t.edges) for t in trees]))
    nodes = np.asarray(trees[smaller].positions, dtype=np.float32)
    sides = 6
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ring = np.stack([np.zeros(sides), np.cos(angles), np.sin(angles)], axis=1) * LUMEN_RADIUS
    vertices = np.concatenate([n + ring for n in nodes]).astype(np.float32)
    triangles = [
        [i * sides + j, (i + 1) * sides + j, (i + 1) * sides + (j + 1) % sides]
        for i in range(len(nodes) - 1) for j in range(sides)
    ]
    runtime.attach_surface(vertices, np.asarray(triangles, dtype=np.int32), env=smaller)
    assert np.isfinite(runtime.surface_positions).all()
    assert runtime.edge_counts[smaller] == len(trees[smaller].edges)


def test_from_trees_validates_inputs():
    tree = _curved_lumen(0.05)
    with pytest.raises(ValueError, match="at least one"):
        CenterlineVesselRuntime.from_trees([], device="cpu")
    with pytest.raises(ValueError, match="env_trees"):
        CenterlineVesselRuntime.from_trees([tree], device="cpu", env_trees=[0, 1])
    with pytest.raises(TypeError, match="open_root"):
        CenterlineVesselRuntime.from_trees([tree], device="cpu", open_root=0)


@pytest.mark.parametrize("device", DEVICES)
def test_per_env_radii_match_each_width_alone(device):
    """Same vessel layout, different lumen widths per env (``from_tree(env_radii=...)``)."""
    narrow, wide = _curved_lumen(0.08, radius=0.012), _curved_lumen(0.08, radius=0.03)
    runtime = CenterlineVesselRuntime.from_tree(
        narrow, device=device, num_envs=2, env_radii=np.stack([narrow.radii, wide.radii]),
        two_way=True, vessel_response=0.6, **VESSEL_KWARGS,
    )
    solver = _solver(device, 2, runtime=runtime)
    _advance(solver)
    catheter = _catheter(solver)
    for env, tree in enumerate((narrow, wide)):
        ref_catheter, ref_wall, _ = _alone(device, tree, 0.6)
        np.testing.assert_allclose(catheter[env], ref_catheter, rtol=0, atol=1.0e-6)
        np.testing.assert_allclose(runtime.positions_per_env[env], ref_wall, rtol=0, atol=1.0e-6)
    with pytest.raises(ValueError, match="env_radii"):
        CenterlineVesselRuntime.from_tree(narrow, device="cpu", num_envs=2, env_radii=narrow.radii)


# --------------------------------------------------------------------------- #
# Per-env route guidance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("device", DEVICES)
def test_per_env_track_axes_match_each_axis_alone(device):
    starts = np.array([ENTRY, ENTRY + [0.0, 0.05, 0.0]], dtype=np.float32)
    dirs = np.array([[1.0, 0.0, 0.0], [1.0, 0.2, 0.0]], dtype=np.float32)
    batched = _solver(device, 2, track_start=starts, track_dir=dirs, track_enabled=True)
    _advance(batched, steps=20)
    for env in range(2):
        alone = _solver(device, 1, track_start=starts[env], track_dir=dirs[env], track_enabled=True)
        _advance(alone, steps=20)
        np.testing.assert_allclose(_catheter(batched)[env], _catheter(alone)[0], rtol=0, atol=1.0e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_per_env_track_paths_of_different_lengths(device):
    """Each env follows its own route, and routes need not have equal point counts."""
    # The bend sits inside the guided (non-tip) span, so the two routes really differ there.
    bend = ENTRY + [0.05, 0.0, 0.0]
    routes = [
        _line(ENTRY, ENTRY + [0.4, 0.0, 0.0], 10),
        np.concatenate([_line(ENTRY, bend, 2), _line(bend, ENTRY + [0.4, 0.08, 0.0], 9)[1:]]),
    ]
    assert len(routes[0]) != len(routes[1])
    batched = _solver(device, 2, track_path=routes, track_enabled=True)
    _advance(batched, steps=20)
    for env, route in enumerate(routes):
        alone = _solver(device, 1, track_path=route, track_enabled=True)
        _advance(alone, steps=20)
        np.testing.assert_allclose(_catheter(batched)[env], _catheter(alone)[0], rtol=0, atol=1.0e-6)
    assert np.abs(_catheter(batched)[0] - _catheter(batched)[1]).max() > 1.0e-3

    shared = _solver(device, 2, track_path=routes[1], track_enabled=True)
    _advance(shared, steps=20)
    np.testing.assert_allclose(_catheter(shared)[0], _catheter(shared)[1], rtol=0, atol=0)


def test_track_inputs_must_be_shared_or_per_env():
    three = np.repeat(ENTRY[None], 3, axis=0)
    with pytest.raises(ValueError, match="track_start"):
        _solver("cpu", 2, track_start=three)
    with pytest.raises(ValueError, match="track_path"):
        _solver("cpu", 2, track_path=[_line(ENTRY, ENTRY + [0.3, 0, 0], 3)] * 3)
    with pytest.raises(ValueError, match="non-zero"):
        _solver("cpu", 2, track_dir=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))


# --------------------------------------------------------------------------- #
# Newton builder: per-env starting shape
# --------------------------------------------------------------------------- #
class _Builder:
    def __init__(self):
        self.positions = []

    @property
    def particle_count(self):
        return len(self.positions)

    def add_particle(self, pos, vel, mass, radius=None):
        del vel, mass, radius
        self.positions.append(tuple(pos))
        return len(self.positions) - 1


def test_builder_places_each_env_on_its_own_start(cpu_rod_config):
    from catheter_vasculature_solver.isaaclab_integration import add_catheter_rod_to_builder

    points = int(cpu_rod_config.geometry.num_segments) + 1
    starts = np.array([[0.0, 0.0, 0.5], [0.0, 0.1, 0.5], [0.2, 0.0, 0.4]], dtype=np.float32)
    builder = _Builder()
    add_catheter_rod_to_builder(builder, cpu_rod_config, start=starts, num_envs=3)
    placed = np.asarray(builder.positions, dtype=np.float32).reshape(3, points, 3)
    np.testing.assert_allclose(placed[:, 0], starts, rtol=0, atol=1.0e-6)

    shapes = placed[::-1].copy()
    builder = _Builder()
    add_catheter_rod_to_builder(builder, cpu_rod_config, positions=shapes, num_envs=3)
    np.testing.assert_allclose(np.asarray(builder.positions).reshape(3, points, 3), shapes, rtol=0, atol=1.0e-6)

    with pytest.raises(ValueError, match="start"):
        add_catheter_rod_to_builder(_Builder(), cpu_rod_config, start=starts[:2], num_envs=3)
