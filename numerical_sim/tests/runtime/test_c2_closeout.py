# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""C2.5: nonzero-capacity face emission after physical State rebinding."""
from collections import defaultdict

import newton
import numpy as np
import pytest
import warp as wp
from helpers import find, kill
from i4h_physics import Stepper, author_case
from i4h_physics.surface_storage import STATE_FIELDS
from i4h_physics.systems.mechanics.surface import Surface


def physical_arrays(state):
    names = ["surface_" + key for key in STATE_FIELDS]
    names += ["tet_active", "tet_edge_active", "particle_flags", "particle_q_prev"]
    return {**{name: getattr(state.i4h, name).numpy() for name in names},
            "particle_q": state.particle_q.numpy(), "particle_qd": state.particle_qd.numpy()}


def byte_snapshot(arrays, worlds=None):
    return {name: (value if worlds is None else value.reshape(3, -1)[worlds]).tobytes()
            for name, value in arrays.items()}


def oriented(tri):
    """Canonicalize cyclic rotations only; reversed winding must fail."""
    a, b, c = map(int, tri)
    return min((a, b, c), (b, c, a), (c, a, b))


def topology(tets):
    incidence = defaultdict(list)
    for owner, (a, b, c, d) in enumerate(tets):
        for tri in ((b, c, d), (a, d, c), (a, b, d), (a, c, b)):
            incidence[tuple(sorted(tri))].append((owner, oriented(tri)))
    assert all(len(rows) in (1, 2) for rows in incidence.values())
    return incidence


def expected_faces(incidence, dead):
    boundary, cut = set(), set()
    for rows in incidence.values():
        live = [row for row in rows if row[0] not in dead]
        if len(live) == 1:
            (boundary if len(rows) == 1 else cut).add(live[0])
    return boundary, cut


def emitted_faces(surface, start=0, end=None):
    tris = surface.tris.numpy()[0, start:end]
    owners = surface.tri_tet.numpy()[0, start:end]
    rows = [(int(owner), oriented(tri)) for owner, tri in zip(owners, tris, strict=True) if owner >= 0]
    assert len(rows) == len(set(rows)), "duplicate emitted face"
    return set(rows)


@pytest.mark.parametrize("model_kind", ["pbd", "xpbd"])
@pytest.mark.parametrize("device,capture", [("cpu", False), ("cuda:0", False), ("cuda:0", True)],
                         ids=["cpu-eager", "cuda-eager", "cuda-captured"])
def test_post_rebind_emits_additional_faces(model_kind, device, capture):
    cfg = {"sim": {"device": device, "envs": 3, "gravity": [0, 0, 0]},
           "assets": {"body": {"type": "tet_grid", "dim": [3, 3, 3], "cell": .02,
                                "parts": {"body": {"spring": 1., "volume": 1.,
                                                   "deletable": True, "cut_faces": 128}}}},
           "systems": [{"type": "particles"}, {"type": "tet", "body": "body", "model": model_kind},
                       {"type": "surface", "tet": "tet"}]}
    authored = author_case(cfg)
    builder = newton.ModelBuilder()
    for world in range(3):
        builder.begin_world(gravity=(0, 0, 0))
        authored.add_world(builder, world)
        builder.end_world()
    authored.complete(builder)
    model = builder.finalize(device=device)
    model.particle_grid = None
    selected, reference, untouched = model.state(), model.state(), model.state()
    untouched_bytes = byte_snapshot(physical_arrays(untouched))
    model_bytes = {key: value.numpy().tobytes() for key, value in vars(model.i4h).items()
                   if isinstance(value, wp.array)}

    def bind(state):
        control = model.control()
        scene, systems = authored.bind(model, state, control)
        sim = Stepper.create(scene, systems, control=control, cuda_graph=capture, substeps=1, iterations=1)
        sim.prepare()
        assert (sim.graph is not None) == capture
        return sim, find(sim, Surface)

    sim, surface = bind(selected)
    ref_sim, ref = bind(reference)
    assert byte_snapshot(physical_arrays(selected)) == byte_snapshot(physical_arrays(reference))
    assert selected.i4h.surface_tris.ptr != reference.i4h.surface_tris.ptr != untouched.i4h.surface_tris.ptr
    unselected = byte_snapshot(physical_arrays(selected), [1, 2])
    incidence = topology(surface.tets.numpy()[0])
    # Pick interior, nonadjacent deletions using a geometric incidence oracle.
    candidates = [t for t in range(surface.active.shape[1]) if len(expected_faces(incidence, {t})[1]) == 4]
    first = candidates[0]
    second = next(t for t in candidates[1:] if len(expected_faces(incidence, {first, t})[1]) == 8)
    cut_start, capacity = int(surface.cut_start.numpy()[0]), int(surface.cut_capacity.numpy()[0])
    assert capacity >= 8

    def rebuild(sf, scene):
        if capture:
            with wp.ScopedCapture(device=device) as recording:
                sf.step_end(scene, sim.dt)
            wp.capture_launch(recording.graph)
        else:
            sf.step_end(scene, sim.dt)

    for runtime, sf in ((sim, surface), (ref_sim, ref)):
        kill(runtime, sf, 0, first)
        rebuild(sf, runtime.scene)
        assert sf.cut_count.numpy().tolist() == [[4], [0], [0]]
        assert emitted_faces(sf, cut_start, cut_start + capacity) == expected_faces(incidence, {first})[1]
        kill(runtime, sf, 0, second)
        assert sf.dirty.numpy().tolist() == [1, 0, 0]
        assert sf.deleted.numpy().tolist() == [2, 0, 0]
    pending = byte_snapshot(physical_arrays(selected))
    reference_pending = byte_snapshot(physical_arrays(reference))
    rebound, fresh = bind(selected)
    assert fresh.tris.ptr == surface.tris.ptr
    assert pending == byte_snapshot(physical_arrays(selected))
    assert reference_pending == byte_snapshot(physical_arrays(reference))
    assert fresh.dirty.numpy().tolist() == [1, 0, 0]

    rebuild(fresh, rebound.scene)
    rebuild(ref, ref_sim.scene)
    boundary, cut = expected_faces(incidence, {first, second})
    assert len(cut) == 8
    for runtime, sf in ((rebound, fresh), (ref_sim, ref)):
        assert emitted_faces(sf) == boundary | cut
        assert emitted_faces(sf, cut_start, cut_start + capacity) == cut
        np.testing.assert_array_equal(sf.cut_count.numpy(), [[8], [0], [0]])
        np.testing.assert_array_equal(sf.dirty.numpy(), 0)
        np.testing.assert_array_equal(sf.overflow.numpy(), 0)
        np.testing.assert_array_equal(sf.deleted.numpy(), [2, 0, 0])
        active = sf.active.numpy()[0].astype(bool)
        assert set(np.flatnonzero(~active)) == {first, second}
        tets = sf.tets.numpy()[0]
        support = np.bincount(tets[active].ravel(), minlength=sf.body.count)
        np.testing.assert_array_equal(sf.support.numpy().reshape(3, -1)[0], support)
        edge_ids = sf.tet_edges.numpy()[0, active].ravel()
        edge_support = np.bincount(edge_ids[edge_ids >= 0], minlength=sf.edge_active.shape[1])
        np.testing.assert_array_equal(sf.edge_support.numpy()[0], edge_support)
        np.testing.assert_array_equal(sf.edge_active.numpy()[0], edge_support > 0)
        flags = runtime.scene.field("flags", sf.body).numpy()[0]
        np.testing.assert_array_equal(flags & 1, support > 0)
        for owner, tri in emitted_faces(sf):
            assert active[owner] and np.all(support[list(tri)] > 0)
        assert byte_snapshot(physical_arrays(runtime.state), [1, 2]) == unselected
    for key in STATE_FIELDS:
        if key not in ("tris", "tri_tet"):  # atomic slot order is deliberately unspecified
            np.testing.assert_array_equal(getattr(fresh, key).numpy(), getattr(ref, key).numpy())
    # Execute the prepared solver graph/eager path for each mechanics model too.
    rebound.step()
    ref_sim.step()
    assert emitted_faces(fresh) == emitted_faces(ref) == boundary | cut
    assert untouched_bytes == byte_snapshot(physical_arrays(untouched))
    assert model_bytes == {key: value.numpy().tobytes() for key, value in vars(model.i4h).items()
                           if isinstance(value, wp.array)}
