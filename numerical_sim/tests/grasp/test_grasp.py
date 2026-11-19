# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pytest
from i4h_surgical.config import load_case, read_case
from i4h_surgical.grasp_geometry import approach_increment, normals, select_targets
from i4h_surgical.grasp_probe import GraspTargetProbe
from i4h_surgical.grasp_snapshot import HarnessSnapshot
from i4h_surgical.grasp_teacher import GraspTeacher


@pytest.fixture
def sim():
    data = os.environ.get("I4H_DATA")
    if not data:
        pytest.skip("requires explicit I4H_DATA production assets")
    cfg = read_case("liver_pinned_grasp_1x", data=data)
    sim = load_case(
        cfg, device="cuda:0", envs=2
    )  # explicit two-world isolation override
    sim.prepare()
    return sim


def capture(probe, env=0, wrong=False):
    """Inject a synthetic capture to exercise observer semantics independently of contact."""
    target = int(np.flatnonzero(probe.eligible)[0])
    probe.select(target, env)
    particle = target
    if wrong:
        particle = next(
            v
            for v in np.flatnonzero(probe.eligible)
            if v not in probe.patch_ids[target]
        )
    sphere = int(probe.jaws.numpy()[0])
    for name, value in [("particle", particle), ("sphere", sphere), ("active", 1)]:
        array = getattr(probe.volume, name)
        host = array.numpy()
        host[env, probe.k, 0] = value
        array.assign(host)
    r = probe.volume.rest.numpy()
    r[env, probe.k, 0] = (
        probe.binding.scene.field("q", probe.body).numpy()[env, particle]
        - probe.ic.sphere_q.numpy()[env, sphere]
    )
    probe.volume.rest.assign(r)
    probe.arm(env)
    probe.sample()
    return target, particle, sphere


def test_fixture_and_physical_layout(sim):
    p = GraspTargetProbe(sim.physics)
    assert [b.name for b in sim.scene.bodies] == ["tissue"]
    assert list(p.body.parts) == ["liver"]
    assert list(sim.scene.instruments) == ["grasper"]
    assert p.body.count == 1986
    assert p.body.pin.index.shape == (1, 391)
    assert np.all(
        sim.scene.field("inv_mass", p.body).numpy() > 0
    )  # soft attachments, never fixed
    assert p.ic.actions.reshape((2, 10)).shape == (2, 10)
    assert sim.substeps == sim.iterations == 8
    assert not any(
        s.id in ("bounds", "connectors", "static_contact") for s in sim.systems
    )


def test_patches_filters_and_target_change_without_recapture(sim):
    p = GraspTargetProbe(sim.physics)
    attached = p.body.pin.index[0]
    assert np.all(~p.eligible[attached])
    assert np.all(p.patch_counts[p.eligible] >= 5)
    for i, row in enumerate(p.patch_ids):
        n = p.patch_counts[i]
        assert np.all(row[n:] == -1)
        assert not set(row[:n]) & set(attached)
    graph = sim.graph
    targets = np.flatnonzero(p.eligible)[:2]
    for t in targets:
        p.select(int(t))
        p.sample()
        np.testing.assert_array_equal(
            p.geometry.numpy()[0, 0], sim.scene.field("q", p.body).numpy()[0, t]
        )
    assert sim.graph is graph
    normal, valid = normals(p.rest, p.triangles)
    ids, facing = select_targets(
        p.rest,
        p.rest,
        p.triangles,
        p.eligible,
        sim.scene.instruments["grasper"].trocar,
        p.ic.root.numpy()[0, 0],
    )
    assert len(ids) <= 8
    assert all(valid[t] and facing[t] >= 0.25 for t in ids)
    assert all(np.linalg.norm(p.rest[t] - p.rest[ids[0]]) <= 0.03 for t in ids)


@pytest.mark.parametrize("capture_step", [1, 2])
def test_first_capture_each_decimated_step_and_never_replace(sim, capture_step):
    p = GraspTargetProbe(sim.physics)
    t = int(np.flatnonzero(p.eligible)[0])
    p.select(t)
    p.arm()
    for _ in range(capture_step - 1):
        p.sample()
        assert p.latched.numpy()[0] == 0  # empty retry
    _, particle, _ = capture(p)
    assert p.latched.numpy()[0] == 1
    saved = p.original_particle.numpy().copy()
    current = p.volume.particle.numpy()
    current[0, 0, 0] = particle + 1
    p.volume.particle.assign(current)
    p.sample()
    np.testing.assert_array_equal(saved, p.original_particle.numpy())
    assert p.counts.numpy()[0, 4] == 1
    current[0, 0, 0] = particle
    p.volume.particle.assign(current)
    p.sample()
    assert p.counts.numpy()[0, 4] == 1  # a replacement cannot heal the original binding


def test_wrong_patch_and_release_latch(sim):
    p = GraspTargetProbe(sim.physics)
    capture(p, wrong=True)
    assert p.counts.numpy()[0, 5] == 1
    initial = p.original_particle.numpy().copy()
    p.volume.active.zero_()
    p.sample()
    np.testing.assert_array_equal(p.original_particle.numpy(), initial)
    with pytest.raises(RuntimeError):
        p.select(int(np.flatnonzero(p.eligible)[1]))


def test_radial_slip_vs_tangential_motion(sim):
    p = GraspTargetProbe(sim.physics)
    _, particle, sphere = capture(p)
    scene = sim.scene
    sq = p.ic.sphere_q.numpy()[0, sphere]
    original = p.original_rest.numpy()[0, 0]
    length = np.linalg.norm(original)
    tangent = np.cross(original, [1.0, 0.0, 0.0])
    tangent /= np.linalg.norm(tangent)
    q = scene.field("q", p.body).numpy()
    q[0, particle] = sq + tangent * length
    scene.field("q", p.body).assign(q)
    p.sample()
    assert p.counts.numpy()[0, 3] == 1
    assert p.stats.numpy()[0, 0] < 1.0e-6
    q[0, particle] = sq + tangent * (length + 0.003)
    scene.field("q", p.body).assign(q)
    p.sample()
    assert p.volume.active.numpy()[0, 0, 0] == 1
    assert p.counts.numpy()[0, 3] == 0
    assert p.stats.numpy()[0, 0] == pytest.approx(0.003, abs=1.0e-6)


def bytes_of(sim):
    from i4h_surgical.grasp_snapshot import continuation_arrays

    return {
        key: a.numpy().copy()
        for key, a in continuation_arrays(sim.physics, sim.state, sim.control).items()
    }


def test_snapshot_roundtrip_exact_repeatability_and_rejection(sim, tmp_path):
    p = GraspTargetProbe(sim.physics)
    ident = {"schema": 1, "test": "two_worlds"}
    counters = {"step": 3, "time_s": 0.05}
    snapshot = HarnessSnapshot(sim.physics, ident, counters)
    path = tmp_path / "bundle.npz"
    snapshot.write(path)
    sim.step()
    snapshot.load(path)
    snapshot.restore(counters, p)
    assert all(snapshot.exact().values())
    positions = []
    for _ in range(3):
        snapshot.restore(counters, p)
        sim.step()
        positions.append(sim.scene.field("q", p.body).numpy())
    assert (
        max(np.linalg.norm(q - positions[0], axis=-1).max() for q in positions)
        <= 1.0e-5
    )
    snapshot.identity = {"schema": 2}
    before = bytes_of(sim)
    with pytest.raises(ValueError, match="identity"):
        snapshot.load(path)
    for key, a in bytes_of(sim).items():
        np.testing.assert_array_equal(a, before[key])
    snapshot.attachments.rest.zero_()
    with pytest.raises(ValueError, match="attachment"):
        snapshot.restore(counters, p)


@pytest.mark.parametrize("operation", ["episode_reset", "snapshot_restore"])
def test_selected_reset_preserves_peer_state_control_and_latch(sim, operation):
    p = GraspTargetProbe(sim.physics)
    snapshot = HarnessSnapshot(sim.physics, {"test": "isolation"}, {"step": 0})
    capture(p, env=0)
    capture(p, env=1)
    before = bytes_of(sim)
    peer = {name: getattr(p, name).numpy()[1].copy() for name in p.fields}
    if operation == "episode_reset":
        sim.reset([0])
        p.clear([0])
    else:
        snapshot.restore({}, p, [0])
    after = bytes_of(sim)
    # Check row ownership from continuation maps, including layouts not shaped (E,...).
    for (owner, key, world, *_), (name, value) in zip(
        sim.solver.continuation.entries, after.items()
    ):
        mask = world.numpy() == 1
        np.testing.assert_array_equal(value[mask], before[name][mask])
    for name in p.fields:
        np.testing.assert_array_equal(getattr(p, name).numpy()[1], peer[name])
    assert p.latched.numpy()[0] == 0


def measurement(latched=0, counts=None, target=(0.0, 0.0, 0.0)):
    return dict(
        latched=latched,
        counts=np.array(counts if counts is not None else [0] * 6),
        geometry=np.array([target, [0, 1, 0], [0, 0, 0], [0, 0.0034, 0.0]]),
        gap=-0.0001,
    )


def test_closure_adaptive_steps_and_overshoot():
    target = np.zeros(3)
    normal = np.array([0, 1.0, 0])
    for gap in [0.004, 0.001, 0.00001, -0.0002]:
        sphere = normal * (0.0035 + gap)
        delta = approach_increment(target, normal, sphere, 0.0035, gap)
        assert np.linalg.norm(delta) <= (0.001 if gap >= 0.002 else 0.0001) + 1.0e-12
        if gap < 0.002:
            assert np.linalg.norm(delta) <= gap + 0.00025 + 1.0e-12
    teacher = GraspTeacher([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [[-1] * 3, [1] * 3])
    teacher.command(measurement(), 0.002, 0.0015, 0.3)
    assert teacher.phase == "capture" and teacher.action[7] == 1
    assert -0.0005 <= teacher.closure["gap"] <= 0
    other = GraspTeacher([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [[-1] * 3, [1] * 3])
    m = measurement()
    m["gap"] = -0.001
    other.command(m, 0.002, 0.0015, 0.3)
    assert other.phase == "reposition" and other.action[7] == 0


def test_empty_retry_wrong_patch_hold_streak_and_target_independence():
    teacher = GraspTeacher([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [[-1] * 3, [1] * 3])
    teacher.phase = "capture"
    for _ in range(59):
        teacher.observe(measurement(), np.zeros(3), 0, 0, 0, 0)
    assert not teacher.done
    teacher.observe(measurement(), np.zeros(3), 0, 0, 0, 0)
    assert teacher.failure == "empty_capture"
    teacher = GraspTeacher([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [[-1] * 3, [1] * 3])
    teacher.phase = "retract"
    teacher.destination = np.array([0, 0.01, 0])
    teacher.acquisition_step = 0
    m = measurement(1, [1, 1, 1, 1, 0, 0], target=(0, 0.01, 0))
    for _ in range(50):
        teacher.observe(m, np.zeros(3), 0, 1, 0, 1)
    off = measurement(1, [1, 1, 1, 1, 0, 0], target=(0, 0, 0))
    teacher.observe(off, np.zeros(3), 0, 1, 0, 1)
    assert teacher.hold_streak == 0
    for _ in range(59):
        teacher.observe(m, np.zeros(3), 0, 1, 0, 1)
    assert teacher.phase == "hold"
    teacher.observe(m, np.zeros(3), 0, 1, 0, 1)
    assert teacher.phase == "release"


def test_release_independent_counts_and_active_bindings():
    t = GraspTeacher([0, 0, 0, 0, 0, 0, 1, 1, 0, 0], [[-1] * 3, [1] * 3])
    t.phase = "release"
    t.release_step = 0
    t.observe(measurement(), np.zeros(3), 0, 1, 1, 1)
    assert t.surface_clear_step is None  # zero count alone is insufficient
    t.observe(measurement(), np.zeros(3), 0, 1, 0, 1)
    assert t.surface_clear_step == 2 and t.volume_clear_step is None
    t.observe(measurement(), np.zeros(3), 0, 0, 0, 0)
    assert t.volume_clear_step == 3 and t.phase == "relax"


def test_invalidation_and_fresh_binding(sim):
    p = GraspTargetProbe(sim.physics)
    views = p.torch_views()
    assert views["counts"].data_ptr() == p.counts.ptr
    sim.physics.invalidate()
    with pytest.raises(RuntimeError, match="invalidated"):
        p.sample()
    from i4h_physics.bindings import PhysicsBinding

    sim.physics = PhysicsBinding(sim)
    fresh = GraspTargetProbe(sim.physics)
    assert not fresh.latched.numpy().any()


def test_teacher_rejects_wrong_patch_and_half_retention():
    for counts, expected in (
        ([1, 0, 0, 0, 0, 1], "wrong_patch_capture"),
        ([3, 3, 3, 1, 0, 0], "retention_loss"),
    ):
        t = GraspTeacher([0, 0, 0, 0, 0, 0, 1, 1, 0, 0], [[-1] * 3, [1] * 3])
        t.phase = "capture"
        t.observe(measurement(1, counts), np.zeros(3), 0, 3, 0, 3)
        assert t.failure == expected
    t = GraspTeacher([0, 0, 0, 0, 0, 0, 1, 1, 0, 0], [[-1] * 3, [1] * 3])
    t.phase = "capture"
    t.observe(measurement(1, [3, 3, 3, 2, 0, 0]), np.zeros(3), 0, 3, 0, 3)
    assert t.phase == "qualify"


def test_snapshot_corrupt_payload_and_layout_rejection(sim, tmp_path):
    snap = HarnessSnapshot(sim.physics, {"schema": 1}, {"step": 0})
    path = tmp_path / "bundle.npz"
    snap.write(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    key = next(iter(snap.arrays))
    data[key] = data[key].copy()
    data[key].flat[0] = 123
    np.savez_compressed(tmp_path / "corrupt.npz", **data)
    with pytest.raises(ValueError, match="buffer"):
        snap.load(tmp_path / "corrupt.npz")
    data[key] = data[key].reshape(-1)[:1]
    np.savez_compressed(tmp_path / "layout.npz", **data)
    with pytest.raises(ValueError, match="buffer"):
        snap.load(tmp_path / "layout.npz")


def test_mesh_patch_neighbourhood_and_normal_rejection():
    from i4h_surgical.grasp_geometry import patches

    rest = np.array([[x * 0.01, y * 0.01, 0.0] for y in range(4) for x in range(4)])
    tris = []
    for y in range(3):
        for x in range(3):
            a = y * 4 + x
            tris.extend([[a, a + 1, a + 4], [a + 1, a + 5, a + 4]])
    tris = np.array(tris, np.int32)
    length, table, sizes, eligible = patches(rest, tris, np.array([0]))
    assert length == pytest.approx(0.01)
    assert (
        1 in table[0, : sizes[0]] and 4 in table[0, : sizes[0]]
    )  # one-ring survives centre exclusion
    assert 0 not in table and not eligible[0]
    assert np.all(sizes[eligible] >= 5)
    good, _ = select_targets(
        rest,
        rest,
        tris,
        eligible,
        np.array([0.015, 0.015, 1.0]),
        np.array([0.015, 0.015, 0.0]),
    )
    bad, _ = select_targets(
        rest, rest, tris[:, ::-1], eligible, np.array([0.015, 0.015, 1.0]), np.zeros(3)
    )
    assert good and not bad
    assert len(good) == len(set(good))


def test_thin_component_adapter_uses_current_binding(sim):
    from i4h_isaaclab.component import I4hComponent
    from i4h_isaaclab.grasp import bind_grasp_probe

    component = I4hComponent.__new__(I4hComponent)
    component.sim = sim
    p = bind_grasp_probe(component)
    assert p.binding is sim.physics
    capture(p)
    state = p.read()
    assert state["jaw_state"].shape == (2, 2)
    assert state["mechanism_counts"].shape == (2, 4)
    assert state["nearest_radius"][0] > 0


def test_teacher_nonfinite_and_workspace_failure():
    for root, finite, reason in (
        (np.zeros(3), False, "non_finite"),
        (np.array([2.0, 0, 0]), True, "workspace_exit"),
    ):
        t = GraspTeacher([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [[-1] * 3, [1] * 3])
        t.observe(measurement(), root, 0, 0, 0, 0, finite)
        assert t.failure == reason
