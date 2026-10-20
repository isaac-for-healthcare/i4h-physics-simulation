# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The scene's two particle blocks: the main block (``scene.q`` and friends) and the private pool block of the
fluid pools. ``scene.field`` resolves a body's live arrays in its own block; ``scene.view`` applies a body's
layout to the array it is given (live or a snapshot copy) and rejects arrays of the wrong block."""

import numpy as np
import pytest
import warp as wp
from i4h_isaaclab.composition import load_case
from i4h_physics import Scene, Stepper
from i4h_physics.loaders import add_fluid
from i4h_physics.systems.fluids.fluid import Fluid
from i4h_physics.systems.mechanics.bounds import Bounds
from i4h_standalone.render.snapshots import Snapshots
from i4h_surgical.systems.fluids.emitter import Emitter


def test_pools_live_in_the_pool_block():
    sim = load_case("two_worlds")
    scene = sim.scene
    water, block = scene.body("water"), scene.body("block")
    assert water.pool and water in scene.pools and water not in scene.bodies
    assert not block.pool and block in scene.bodies
    assert scene.particle_count == scene.q.shape[0] == scene.envs * sum(b.count for b in scene.bodies)
    assert scene.pool_count == scene.pool.q.shape[0] == scene.envs * water.count
    assert scene.field("q", water).ptr == scene.pool.q.ptr + scene.layout(water)[0] * 12
    assert scene.field("q", block).ptr == scene.q.ptr + scene.layout(block)[0] * 12
    assert scene.field("q", water, "water").shape == (scene.envs, water.count)


def test_view_rejects_the_other_block():
    scene = load_case("two_worlds").scene
    water, block = scene.body("water"), scene.body("block")
    with pytest.raises(ValueError, match="pool block"):
        scene.view(scene.q, water)  # a main-block array for a pool
    with pytest.raises(ValueError, match="main block"):
        scene.view(scene.pool.q, block)  # a pool-block array for a main-block body
    with pytest.raises(ValueError, match="unknown particle field"):
        scene.field("gravity", block)


def test_duplicate_names_across_blocks_are_rejected():
    scene = Scene("cuda:0", 1)
    add_fluid(scene, "water", 8)
    with pytest.raises(ValueError, match="duplicate body"):
        add_fluid(scene, "water", 8)


def test_a_held_snapshot_keeps_its_own_data():
    """Views of a published snapshot read the snapshot, never the live state, after the simulation moves on."""
    sim = load_case("two_worlds")
    scene, water, block = sim.scene, sim.scene.body("water"), sim.scene.body("block")
    for _ in range(20):  # the emitter brings water to life
        sim.step()
    ring = Snapshots(sim)
    ring.publish()
    wp.synchronize()
    snap = ring.acquire()
    held_pool = scene.view(snap.pool_q, water).numpy().copy()
    held_block = scene.view(snap.q, block).numpy().copy()
    for _ in range(20):
        sim.step()
    wp.synchronize()
    assert np.array_equal(scene.view(snap.pool_q, water).numpy(), held_pool)
    assert np.array_equal(scene.view(snap.q, block).numpy(), held_block)
    assert not np.array_equal(scene.field("q", water).numpy(), held_pool), "live water moved on"


def fluid_only(cuda_graph):
    scene = Scene("cuda:0", 2, (0.0, 0.0, -9.81))
    add_fluid(scene, "water", 64, mass=0.001, radius=0.003, interaction_radius=0.012, rest_distance=0.006)
    scene.allocate()
    fluid = Fluid(scene)
    systems = [fluid, Emitter(scene, fluid=fluid, body="water", position=(0.0, 0.0, 0.1), rate=600.0, speed=0.0),
               Bounds(scene, lower=(-0.5, -0.5, 0.0), upper=(0.5, 0.5, 1.0))]
    return Stepper.create(scene, systems, dt=1.0 / 60.0, substeps=2, iterations=2, cuda_graph=cuda_graph)


@pytest.mark.parametrize("cuda_graph", [False, True], ids=["eager", "captured"])
def test_fluid_only_scene_steps_and_resets(cuda_graph):
    """An empty main block (no tissue, no rods): zero-length ``scene.q``, the pool block carries everything."""
    sim = fluid_only(cuda_graph)
    scene, water = sim.scene, sim.scene.body("water")
    assert scene.particle_count == 0 and scene.q.shape == (0,)
    for _ in range(30):
        sim.step()
    flags = scene.field("flags", water).numpy()
    assert (flags != 0).any(axis=1).all(), "water came alive in both envs"
    assert np.isfinite(scene.field("q", water).numpy()).all()
    sim.reset([1])
    flags = scene.field("flags", water).numpy()
    assert (flags[1] == 0).all() and (flags[0] != 0).any(), "the masked reset parks env 1 only"
    assert np.array_equal(scene.field("q", water).numpy()[1], scene.field("q0", water).numpy()[1])


def test_main_block_is_world_major():
    """Env e's row of the main block holds every body side by side (Newton's world-grouped order)."""
    from helpers import deterministic
    from i4h_isaaclab.composition import read_case

    sim = load_case(deterministic(read_case("two_worlds"), spacer_before="rod"), envs=3)
    scene = sim.scene
    W = scene.main.width
    assert W == sum(b.count for b in scene.bodies) and scene.q.shape == (3 * W,)
    rows = scene.q0.numpy().reshape(3, W, 3)
    for b in scene.bodies:
        offset, stride = scene.layout(b)
        assert stride == W and offset == b.offset
        np.testing.assert_array_equal(rows[:, offset : offset + b.count], b.rest.q0)
        view = scene.field("q0", b)
        assert view.shape == (3, b.count) and view.strides[0] == W * 12


def test_observation_views_stay_zero_copy_when_strided():
    """isaaclab.views._flat: a strided (E, count) body view is still a zero-copy (E, 3 * count) torch view."""
    torch = pytest.importorskip("torch")
    from helpers import deterministic
    from i4h_isaaclab.composition import read_case
    from i4h_isaaclab.views import observation_views

    sim = load_case(deterministic(read_case("two_worlds"), spacer_before="rod"), envs=3)
    scene = sim.scene
    views = observation_views(sim)
    for b, v in zip([b for b in scene.bodies if b.integrate], views):
        assert v.shape == (3, 3 * b.count) and v.data_ptr() == scene.field("q", b).ptr
        assert torch.equal(v.cpu(), torch.from_numpy(scene.field("q", b).numpy().reshape(3, -1)))
